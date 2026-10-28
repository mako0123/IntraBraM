"""Positional encodings for channels and time steps."""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, n_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, n_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, n_dim, 2).float() * (-math.log(10000.0) / n_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MixedPositionalEncoding(nn.Module):
    """Combine sinusoidal channel encoding with a learnable MNI-coordinate encoding.

    With ``pos=None`` only the sinusoidal channel encoding is applied. Otherwise
    the normalized (x, y, z) MNI coordinates are fused both sinusoidally and
    through a small learned MLP.
    """

    def __init__(self, n_dim: int = 256, max_channel_num: int = 320):
        super().__init__()
        self.n_dim = n_dim
        self.max_channel_num = max_channel_num

        self.positional_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        self.sinusodial_positional_encoding = SinusoidalPositionalEncoding(
            n_dim=self.n_dim, max_len=max_channel_num
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None):
        # x: (B, N, C, t); pos: (B, C, 3)
        if pos is None:
            B, T, C = x.shape[0:3]
            x = x.reshape(B * T, C, -1)
            x = x + self.sinusodial_positional_encoding(x)
            x = x.reshape(B, T, C, -1)
            return x

        T = x.shape[1]
        pos = (pos - torch.mean(pos, dim=1, keepdim=True)) / (torch.std(pos, dim=1, keepdim=True) + 1e-8)
        pos = pos.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, C, 3)

        div_term = torch.exp(
            torch.arange(0, self.n_dim, 4, device=x.device).float() * (-math.log(10000.0) / self.n_dim)
        )

        pe = torch.zeros_like(x)
        pe[:, :, :, 0::4] = torch.sin(pos[:, :, :, 0:1] * div_term)
        pe[:, :, :, 1::4] = torch.sin(pos[:, :, :, 1:2] * div_term)
        pe[:, :, :, 2::4] = torch.sin(pos[:, :, :, 2:3] * div_term)
        pe[:, :, :, 3::4] = self.positional_encoder(pos)
        return x + pe
