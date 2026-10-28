"""Cross- and self-attention building blocks used by the encoder and decoder."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class CrossAttn(nn.Module):
    """Multi-head cross attention where ``query`` attends over ``x``."""

    def __init__(self, n_dim: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.d_k = self.n_dim // self.n_head
        self.scale_factor = math.sqrt(self.d_k)

        self.W_q = nn.Linear(self.n_dim, self.n_dim)
        self.W_k = nn.Linear(self.n_dim, self.n_dim)
        self.W_v = nn.Linear(self.n_dim, self.n_dim)
        self.W_o = nn.Linear(self.n_dim, self.n_dim)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(self, Q, K, V, valid_mask=None):
        # Q, K, V: (B, n_head, N, d_k); valid_mask: (B, N)
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale_factor

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)
            attn_score.masked_fill_(valid_mask, value=-1e9)

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)

        output = torch.matmul(attn_score, V)
        return output, attn_score

    def forward(self, x, query, valid_mask):
        # x: (B, L, d); query: (B, n_query, d)
        B, L, _ = x.shape
        B, n_query, _ = query.shape

        Q = self.W_q(query.expand(B, -1, -1))
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(B, n_query, self.n_head, self.d_k).transpose(1, 2)
        K = K.reshape(B, L, self.n_head, self.d_k).transpose(1, 2)
        V = V.reshape(B, L, self.n_head, self.d_k).transpose(1, 2)

        valid_mask = ~valid_mask  # True where a channel is invalid
        output, attn_score = self.scaled_dot_product(Q, K, V, valid_mask=valid_mask)
        output = output.transpose(1, 2).flatten(start_dim=2, end_dim=3)
        output = self.W_o(output)

        return output, attn_score


class SelfAttn(nn.Module):
    """Multi-head self attention used inside the encoder layers."""

    def __init__(self, n_dim: int, n_head: int, dropout: float = 0.1, **kwargs):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.d_k = self.n_dim // self.n_head
        self.scale_factor = math.sqrt(self.d_k)

        self.W_q = nn.Linear(self.n_dim, self.n_dim)
        self.W_k = nn.Linear(self.n_dim, self.n_dim)
        self.W_v = nn.Linear(self.n_dim, self.n_dim)
        self.W_o = nn.Linear(self.n_dim, self.n_dim)

        self.pad_token = nn.Parameter(torch.randn(1, 1, self.n_dim), requires_grad=True)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product(self, Q, K, V, valid_mask=None):
        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale_factor

        if valid_mask is not None:
            valid_mask = valid_mask.unsqueeze(1).unsqueeze(2)
            attn_score.masked_fill_(valid_mask, value=-1e9)

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)

        output = torch.matmul(attn_score, V)
        return output, attn_score

    def forward(self, x, valid_mask=None):
        B, L, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(B, L, self.n_head, self.d_k).transpose(1, 2)
        K = K.reshape(B, L, self.n_head, self.d_k).transpose(1, 2)
        V = V.reshape(B, L, self.n_head, self.d_k).transpose(1, 2)

        if valid_mask is not None:
            valid_mask = ~valid_mask

        output, attn_score = self.scaled_dot_product(Q, K, V, valid_mask=valid_mask)
        output = output.transpose(1, 2).flatten(start_dim=2, end_dim=3)
        output = self.W_o(output)

        return output, attn_score


class CrossAttentionLayer(nn.Module):
    """Cross attention followed by a residual MLP (feed-forward) block."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.n_dim

        self.cross_attn = CrossAttn(n_dim=self.n_dim, n_head=self.n_head, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.n_dim),
            nn.Linear(self.n_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.n_dim),
        )

    def forward(self, x, query, valid_mask, output_attention: bool = False):
        output, attn_score = self.cross_attn(x=x, query=query, valid_mask=valid_mask)
        query = output + query
        query = self.ffn(query) + query

        if not output_attention:
            return query
        return query, attn_score


class SelfAttentionLayer(nn.Module):
    """Self attention followed by a residual MLP (feed-forward) block."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.n_dim

        self.self_attn = SelfAttn(n_dim=self.n_dim, n_head=self.n_head, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.n_dim),
            nn.Linear(self.n_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.ffn_dim, self.n_dim),
        )

    def forward(self, x):
        output, _ = self.self_attn(x=x)
        x = output + x
        x = self.ffn(x) + x
        return x


class CrossAttentionBasedEncoderLayer(nn.Module):
    """One encoder layer: cross attention into the EEG tokens, then self attention."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.sa_layer = SelfAttentionLayer(n_dim=self.n_dim, n_head=self.n_head, ffn_dim=self.ffn_dim, dropout=self.dropout)
        self.ca_layer = CrossAttentionLayer(n_dim=self.n_dim, n_head=self.n_head, ffn_dim=self.ffn_dim, dropout=self.dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape=self.n_dim)

    def forward(self, x, latent, valid_mask, output_attention: bool = False):
        if output_attention is False:
            latent = self.ca_layer(x=x, query=latent, valid_mask=valid_mask, output_attention=output_attention)
            latent = self.layer_norm(latent)
            latent = self.sa_layer(latent)
            return latent

        latent, attn_score = self.ca_layer(x=x, query=latent, valid_mask=valid_mask, output_attention=output_attention)
        latent = self.layer_norm(latent)
        latent = self.sa_layer(latent)
        return latent, attn_score


class StackedCrossAttentionLayers(nn.Module):
    """A stack of :class:`CrossAttentionLayer` (used by the decoder)."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        depth: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.depth = depth
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.attention_layers = nn.ModuleList(
            [
                CrossAttentionLayer(n_dim=self.n_dim, n_head=self.n_head, ffn_dim=self.ffn_dim, dropout=dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x, query, valid_mask):
        for layer in self.attention_layers:
            query = layer(x=x, query=query, valid_mask=valid_mask)
        return query


class StackedEncoderLayers(nn.Module):
    """A stack of :class:`CrossAttentionBasedEncoderLayer` (the IntraBraM encoder)."""

    def __init__(
        self,
        n_dim: int,
        n_head: int,
        depth: int,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_head = n_head
        self.depth = depth
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        self.layers = nn.ModuleList(
            [
                CrossAttentionBasedEncoderLayer(n_dim=self.n_dim, n_head=self.n_head, ffn_dim=self.ffn_dim, dropout=self.dropout)
                for _ in range(self.depth)
            ]
        )

    def forward(self, x, latent, valid_mask=None):
        for layer in self.layers:
            latent = layer(x=x, latent=latent, valid_mask=valid_mask)
        return latent

    def forward_with_attention(self, x, latent, valid_mask=None):
        attn_score_list = []
        for layer in self.layers:
            latent, attn_score = layer(x=x, latent=latent, valid_mask=valid_mask, output_attention=True)
            attn_score_list.append(attn_score.unsqueeze(1))
        return latent, torch.concatenate(attn_score_list, dim=1)
