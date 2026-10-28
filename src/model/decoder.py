"""Decoder that reconstructs masked input windows from encoder latents."""

import torch
import torch.nn.functional as F
from torch import nn

from src.model.attentions import StackedCrossAttentionLayers
from src.model.positional_encodings import MixedPositionalEncoding, SinusoidalPositionalEncoding


class Decoder(nn.Module):
    """Mask part of the input windows and reconstruct them from the encoder latents."""

    def __init__(
        self,
        n_dim: int = 256,
        window_size: int = 64,
        max_sec: int = 32,
        depth: int = 2,
        dropout: float = 0.1,
        _train_mask_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        # config
        self.n_dim = n_dim
        self.window_size = window_size
        self.max_sec = max_sec
        self.dropout = dropout
        self.depth = depth
        self._train_mask_rate = _train_mask_rate

        # parameters
        self.mask_token = nn.Parameter(torch.randn(size=(1, 1, 1, n_dim)) * 0.05)

        # modules
        self.mixed_positional_encoder = MixedPositionalEncoding(n_dim=self.n_dim)
        self.temporal_encoder = SinusoidalPositionalEncoding(n_dim=self.n_dim, max_len=self.max_sec)

        self.decoder = StackedCrossAttentionLayers(n_dim=self.n_dim, n_head=8, depth=self.depth, dropout=self.dropout)
        self.linear = nn.Linear(self.n_dim, self.window_size)

    def forward(self, x, embedding, latent, channels, pos=None):
        B, C, _ = x.shape
        T = embedding.shape[1] // C

        reconstruct_seq = embedding
        reconstruct_seq = reconstruct_seq.reshape(B * C, T, -1)

        # Randomly mask windows with the learned mask token.
        mask = torch.bernoulli(torch.ones(B * C, T, device=reconstruct_seq.device) * self._train_mask_rate)
        reconstruct_seq[mask == 1] = self.mask_token

        reconstruct_seq = self.temporal_encoder(reconstruct_seq)
        reconstruct_seq = reconstruct_seq.reshape(B, C, T, -1).transpose(2, 1)  # (B, T, C, d)
        reconstruct_seq = self.mixed_positional_encoder(reconstruct_seq, pos)
        reconstruct_seq = reconstruct_seq.transpose(2, 1).reshape(B, C * T, -1)
        valid_mask = torch.ones((latent.shape[0], latent.shape[1]), dtype=torch.bool, device=x.device)

        reconstruct_seq = self.decoder(x=latent, query=reconstruct_seq, valid_mask=valid_mask)
        reconstruct_seq = reconstruct_seq.reshape(B, C, T, -1)
        reconstruct_seq = self.linear(reconstruct_seq)
        reconstruct_seq = reconstruct_seq.reshape(B, C, -1)

        mask = mask.reshape(B, C, T, 1).expand(-1, -1, -1, self.window_size).reshape(B, C, -1)
        channels = channels.reshape(B, C, 1).expand(-1, -1, self.window_size * T)

        # Reconstruction loss is only evaluated on valid, masked channels.
        loss = F.mse_loss(
            x[torch.logical_and(mask == 1, channels == 1)],
            reconstruct_seq[torch.logical_and(mask == 1, channels == 1)],
        )
        return reconstruct_seq, loss
