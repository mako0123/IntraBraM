"""Finetuning LightningModule: pretrained encoder + classification head."""

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, roc_auc_score
from torch import nn, optim

from src.model.classifier import Classifier
from src.model.encoder import build_encoder_from_checkpoint
from src.model.subject_adapter import SubjectAdapter


class FinetuneModel(L.LightningModule):
    """Load a pretrained encoder and attach a learnable query + classification head."""

    def __init__(
        self,
        pretrain_path: str,
        n_cls: int = 2,
        load_weights: bool = True,
        use_bias: bool = False,
        max_channel_num: int | None = None,
        segment_t: int | None = None,
        **kwargs,
    ):
        super().__init__()

        self.pretrain_path = pretrain_path
        self.load_weights = load_weights
        self.enc_model = build_encoder_from_checkpoint(self.pretrain_path, load_weights=load_weights)

        self.n_dim = self.enc_model.n_dim
        self.n_cls = n_cls
        self.encode_n_tok = 8

        self.use_bias = use_bias
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
