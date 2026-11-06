"""Finetuning LightningModule: pretrained encoder + classification head."""

import os
from datetime import datetime

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torch import nn, optim

from src.config import get_checkpoint_dir
from src.model.classifier import Classifier
from src.model.encoder import Encoder, build_encoder_from_checkpoint, build_encoder_from_config
from src.model.subject_adapter import SubjectAdapter


class FinetuneModel(L.LightningModule):
    """Load a pretrained encoder and attach a learnable query + classification head.

    ``encoder`` may be passed directly to rebuild the architecture without
    re-reading a pretrain checkpoint (used by :func:`build_finetune_model`).
    """

    def __init__(
        self,
        pretrain_path: str | None = None,
        n_cls: int = 2,
        load_weights: bool = True,
        use_bias: bool = False,
        max_channel_num: int | None = None,
        segment_t: int | None = None,
        version: str | None = None,
        task: str | None = None,
        dependency: str | None = None,
        split_id: int | None = None,
        run_name: str | None = None,
        run_dir: str | None = None,
        encoder: Encoder | None = None,
        **kwargs,
    ):
        super().__init__()

        self.pretrain_path = pretrain_path
        self.load_weights = load_weights
        if encoder is not None:
            self.enc_model = encoder
        else:
            self.enc_model = build_encoder_from_checkpoint(pretrain_path, load_weights=load_weights)

        self.version = version
        self.task = task
        self.dependency = dependency
        self.split_id = split_id
        self.run_name = run_name
        # Explicit run directory shared by every process; falls back to ``{run_name}-{timestamp}``.
        self.run_dir = run_dir
        self._run_dir: str | None = None  # resolved on first access

        self.n_dim = self.enc_model.n_dim
        self.n_cls = n_cls
        self.encode_n_tok = 8

        self.use_bias = use_bias
        self._max_channel_num = max_channel_num
        self._segment_t = segment_t
        self.query = nn.Parameter(torch.randn(1, self.encode_n_tok, self.n_dim))

        if self.use_bias:
            assert max_channel_num is not None and segment_t is not None, "missing channel_number or segment_t"
            self.projection = SubjectAdapter(max_channel_num=max_channel_num, segment_t=segment_t)

        self.classifier = Classifier(n_dim=self.n_dim, n_encoder_tok=self.encode_n_tok, n_cls=self.n_cls)

        self.test_results = {
            "batch_size": [],
            "y_true": [],
            "y_pred": [],
            "logits": [],
            "probs": [],
        }

    # ------------------------------------------------------------------ forward
    def _encode(self, x, channel, position):
        """Route through the query-only or the subject-adapter bias path."""
        query = self.query.expand(x.shape[0], -1, -1)
        if not self.use_bias:
            return self.enc_model.get_latent_with_query(x=x, query=query, valid_mask=channel, pos=position)
        return self.enc_model.get_latent_with_bias(x=x, query=query, valid_mask=channel, bias=self.projection, pos=position)

    def sanitize_batch(self, batch):
        if len(batch) == 4:
            x, channels, label, pos = batch
        else:
            x, channels, label = batch
            pos = None

        x = (x - torch.mean(x, axis=-1, keepdim=True)) / (torch.std(x, axis=-1, keepdim=True) + 1e-8)
        x = x.float()
        if pos is not None:
            pos = (pos - torch.mean(pos, axis=-1, keepdim=True)) / (torch.std(pos, axis=-1, keepdim=True) + 1e-8)
            pos = pos.float()

        return x, channels, label, pos

    def training_step(self, batch, batch_idx):
        x, channel, label, position = self.sanitize_batch(batch)
        latent = self._encode(x, channel, position)
        logits = self.classifier(latent)

        loss = F.cross_entropy(logits, label)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, channel, label, position = self.sanitize_batch(batch)
        latent = self._encode(x, channel, position)
        logits = self.classifier(latent)

        loss = F.cross_entropy(logits, label)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(logits, axis=1)
        self.log("test_loss", loss, prog_bar=True)

        self.test_results["batch_size"].append(x.shape[0])
        self.test_results["y_true"].append(label.detach().cpu().numpy())
        self.test_results["y_pred"].append(preds.detach().cpu().numpy())
        self.test_results["logits"].append(logits.detach().cpu().numpy())
        self.test_results["probs"].append(probs.detach().cpu().numpy())

    def on_test_epoch_start(self):
        self.test_results = {
            "batch_size": [],
            "y_true": [],
            "y_pred": [],
            "logits": [],
            "probs": [],
        }

    def on_test_epoch_end(self):
        """Gather per-rank test results so ``summarize`` covers the full test set under DDP.

        Each rank only evaluates its own data shard, so without this gather the
        metrics on rank 0 would be computed on a fraction of the samples. After
        gathering, every rank holds the identical (complete) result lists.
        """
        if not torch.distributed.is_initialized() or torch.distributed.get_world_size() <= 1:
            return

        world_size = torch.distributed.get_world_size()
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, self.test_results)

        merged = {key: [] for key in self.test_results}
        for rank_results in gathered:
            for key in merged:
                merged[key].extend(rank_results[key])
        self.test_results = merged

    def summarize(self):
        y_true = np.concatenate(self.test_results["y_true"])
        y_pred = np.concatenate(self.test_results["y_pred"])
        probs = np.concatenate(self.test_results["probs"])
        total_samples = y_true.shape[0]

        summary = {
            "total_samples": int(total_samples),
            "accuracy": float(accuracy_score(y_true=y_true, y_pred=y_pred)),
            "f1_score": float(f1_score(y_true=y_true, y_pred=y_pred, average="macro")),
            "cohens_kappa": float(cohen_kappa_score(y1=y_true, y2=y_pred)),
        }

        try:
            auc = roc_auc_score(
                y_true=F.one_hot(torch.tensor(y_true).long(), num_classes=self.n_cls).numpy(),
                y_score=probs,
                multi_class="ovr",
                average="macro",
            )
            summary["auc"] = float(auc)
        except Exception as e:
            summary["auc"] = -1
            print(f"AUC ERROR: {e}")
        return summary

    def configure_optimizers(self):
        params = [
            {"params": self.classifier.parameters(), "lr": 1e-4, "weight_decay": 0},
            {"params": self.enc_model.encoder.parameters(), "lr": 3e-5, "weight_decay": 0.05},
            {"params": self.enc_model.embedder.parameters(), "lr": 3e-5, "weight_decay": 0.05},
            {"params": self.enc_model.latent_token, "lr": 3e-4, "weight_decay": 0.01},
            {"params": self.query, "lr": 3e-4, "weight_decay": 0.01},
        ]
        if self.use_bias:
            params.append({"params": self.projection.parameters(), "lr": 3e-4, "weight_decay": 0.001})

        return optim.AdamW(params)

    # -------------------------------------------------------------- checkpoints
    def on_train_epoch_end(self):
        self.save_finetune()

    def save_finetune(self):
        """Save this epoch's finetuned weights into the run directory."""
        # Only the main process saves when running under DDP.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        run_dir = self._get_run_dir()
        torch.save(self._checkpoint_payload(), os.path.join(run_dir, f"epoch-{self.current_epoch}.ckpt"))

    def _checkpoint_payload(self) -> dict:
        return {
            "state_dict": self.state_dict(),
            "config": {
                "version": self.version,
                "task": self.task,
                "dependency": self.dependency,
                "split_id": self.split_id,
                "encoder": {
                    "n_dim": self.enc_model.n_dim,
                    "window_size": self.enc_model.window_size,
                    "max_sec": self.enc_model.max_sec,
                    "depth": self.enc_model.depth,
                    "embedder_params": self.enc_model.embedder_params,
                },
                "finetune": {
                    "n_cls": self.n_cls,
                    "use_bias": self.use_bias,
                    "max_channel_num": self._max_channel_num,
                    "segment_t": self._segment_t,
                },
            },
        }

    def _default_run_name(self) -> str:
        """``{version}-{task}-{dependency}-{subject-n|fold-n}``."""
        if self.dependency == "independent":
            split_tag = f"fold-{self.split_id}"
        else:
            split_tag = f"subject-{self.split_id}"
        return f"{self.version}-{self.task}-{self.dependency}-{split_tag}"

    def _get_run_dir(self) -> str:
        """Create (once) this run's checkpoint directory and save its config.

        If ``self.run_dir`` was provided it is used as-is (so every DDP process
        agrees on the path); otherwise it is ``{run_name}-{timestamp}`` under the
        configured ``checkpoint_dir``. ``config.yaml`` is written once.
        """
        if self._run_dir is None:
            if self.run_dir is not None:
                self._run_dir = self.run_dir
            else:
                if self.version is None or self.task is None:
                    raise ValueError("version and task must be set to save finetune checkpoints")
                base_name = self.run_name or self._default_run_name()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._run_dir = os.path.join(get_checkpoint_dir(), f"{base_name}-{timestamp}")
            os.makedirs(self._run_dir, exist_ok=True)
            self._save_config(self._run_dir)
        return self._run_dir

    def _save_config(self, run_dir: str) -> None:
        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(self._checkpoint_payload()["config"], f, sort_keys=False)


def build_finetune_model(path_to_ckpt: str, load_weights: bool = True) -> FinetuneModel:
    """Rebuild a finetuned model from a finetune checkpoint (state_dict + config).

    The checkpoint stores the encoder architecture and the finetune settings, so
    no pretrain checkpoint is required at evaluation time.
    """
    loaded_content = torch.load(path_to_ckpt, map_location="cpu", weights_only=False)
    config = loaded_content["config"]

    encoder = build_encoder_from_config(config["encoder"])
    finetune = config["finetune"]
    model = FinetuneModel(
        encoder=encoder,
        n_cls=finetune["n_cls"],
        use_bias=finetune["use_bias"],
        max_channel_num=finetune["max_channel_num"],
        segment_t=finetune["segment_t"],
        version=config.get("version"),
        task=config.get("task"),
        dependency=config.get("dependency"),
        split_id=config.get("split_id"),
    )
    if load_weights:
        model.load_state_dict(loaded_content["state_dict"])
    return model
