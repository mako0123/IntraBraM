"""Pretraining LightningModule for IntraBraM."""

import os
from datetime import datetime

import lightning as L
import torch
from torch import optim

from src.config import get_checkpoint_dir, get_embedder_params
from src.model.decoder import Decoder
from src.model.encoder import Encoder, build_encoder_from_checkpoint


class IntraBraM(L.LightningModule):
    """Contrastive + reconstruction pretraining wrapper around Encoder/Decoder."""

    def __init__(
        self,
        n_dim: int,
        window_size: int,
        max_sec: int,
        depth: int,
        embedder_params: str,
        _train_no_reconstruction: bool = False,
        _train_no_contrastive: bool = False,
        _train_mask_rate: float = 0.5,
        _train_temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Snapshot the constructor arguments before resolving the embedder config
        # (``embedder_params`` is stored as a variant name so checkpoints stay portable).
        local_vars = locals()
        self.config_dict = {k: v for k, v in local_vars.items() if k != "self" and not callable(v)}

        self.n_dim = n_dim
        self.window_size = window_size
        self.max_sec = max_sec
        self.depth = depth
        self.embedder_params = get_embedder_params(embedder_params)
        self.version_name = embedder_params

        self._train_no_reconstruction = _train_no_reconstruction
        self._train_no_contrastive = _train_no_contrastive
        self._train_mask_rate = _train_mask_rate
        self._train_temperature = _train_temperature

        self.encoder = Encoder(
            embedder_params=self.embedder_params,
            n_dim=self.n_dim,
            window_size=self.window_size,
            max_sec=self.max_sec,
            depth=self.depth,
            _train_temperature=self._train_temperature,
        )
        self.decoder = Decoder(
            n_dim=self.n_dim,
            window_size=self.window_size,
            max_sec=self.max_sec,
            depth=2,
            _train_mask_rate=self._train_mask_rate,
        )

    def sanitize_batch(self, batch):
        x = batch["eeg"]
        channels = batch["mask"]
        pos = batch["mni_pos"]

        x = (x - torch.mean(x, axis=-1, keepdim=True)) / (torch.std(x, axis=-1, keepdim=True) + 1e-8)
        if pos is not None:
            pos = pos / 50.0

        x = x.float()
        pos = pos.float()
        return x, channels, pos

    def training_step(self, batch, batch_idx):
        x, channels, pos = self.sanitize_batch(batch)

        latent, embedding, contrastive_loss = self.encoder(x=x, valid_mask=channels, pos=pos)
        _, reconstruction_loss = self.decoder(x=x, embedding=embedding, latent=latent, channels=channels, pos=pos)

        self.log("train/contrast", value=contrastive_loss, prog_bar=True)
        self.log("train/reconstruction", value=reconstruction_loss, prog_bar=True)
        self.log("train/total_loss", value=reconstruction_loss + contrastive_loss, prog_bar=True)

        if self._train_no_contrastive:
            return reconstruction_loss
        if self._train_no_reconstruction:
            return contrastive_loss
        return reconstruction_loss + contrastive_loss

    def validation_step(self, batch, batch_idx):
        x, channels, pos = self.sanitize_batch(batch)

        latent, embedding, contrastive_loss = self.encoder(x=x, valid_mask=channels, pos=pos)
        _, reconstruction_loss = self.decoder(x=x, embedding=embedding, latent=latent, channels=channels, pos=pos)

        self.log("valid/contrast", value=contrastive_loss, prog_bar=True)
        self.log("valid/reconstruction", value=reconstruction_loss, prog_bar=True)
        self.log("valid/total_loss", value=reconstruction_loss + contrastive_loss, prog_bar=True)

        if self._train_no_contrastive:
            return reconstruction_loss
        if self._train_no_reconstruction:
            return contrastive_loss
        return reconstruction_loss + contrastive_loss

    def configure_optimizers(self):
        encoder_params = self.encoder._pretrain_parameter_setting()
        decoder_params = {"params": self.decoder.parameters(), "lr": 1e-4, "weight_decay": 1e-3}
        return optim.AdamW(encoder_params + [decoder_params])

    def on_train_epoch_end(self):
        self.save_encoder()

    def save_encoder(self):
        # Only the main process saves when running under DDP.
        if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
            return
        checkpoint_dir = get_checkpoint_dir()
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(
            checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M%S_%f") + self.version_name + ".ckpt"
        )
        torch.save({"state_dict": self.encoder.state_dict(), "config": self.config_dict}, path)

    @classmethod
    def load_encoder(cls, path: str, load_weights: bool = True) -> Encoder:
        return build_encoder_from_checkpoint(path, load_weights=load_weights)
