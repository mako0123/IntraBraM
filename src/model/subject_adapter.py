"""Subject-specific channel adapter added during finetuning."""

import torch
from einops import rearrange
from torch import nn


class SubjectAdapter(nn.Module):
    """Learn a per-subject channel bias added to the embedded windows.

    ``max_channel_num`` and ``segment_t`` are only used to size the projection;
    the module itself is shape-agnostic at runtime.
    """

    def __init__(self, max_channel_num: int, segment_t: int):
        super().__init__()
        self.max_channel_num = max_channel_num
        self.segment_t = segment_t

        self.channel_proj = nn.Sequential(
            nn.Linear(self.max_channel_num, self.max_channel_num * 8),
            nn.ReLU(),
            nn.Linear(self.max_channel_num * 8, self.max_channel_num),
            nn.LayerNorm(self.max_channel_num),
        )

    def forward(self, embedding: torch.Tensor):
        channel_bias = self.channel_proj(rearrange(embedding, "b c n t -> b n t c"))
        channel_bias = rearrange(channel_bias, "b n t c -> b c n t")

        embedding = embedding + channel_bias
        return embedding
