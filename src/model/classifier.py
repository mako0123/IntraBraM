"""Classification head applied on top of the encoded latent tokens."""

import torch
from einops.layers.torch import Rearrange
from torch import nn


class Classifier(nn.Module):
    """Flatten the latent tokens and project them to class logits."""

    def __init__(self, n_dim: int, n_encoder_tok: int, n_cls: int):
        super().__init__()
        self.n_dim = n_dim
        self.n_encoder_tok = n_encoder_tok
        self.n_cls = n_cls

        self.cls_head = nn.Sequential(
            Rearrange("b c d -> b (c d)"),
            nn.Linear(self.n_dim * self.n_encoder_tok, 512),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, self.n_cls),
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.cls_head(x)
