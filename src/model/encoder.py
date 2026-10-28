"""IntraBraM encoder: CNN patch embedder + cross-attention transformer."""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from src.model.attentions import StackedEncoderLayers
from src.model.patch_embedder import CNNPatchEmbedder
from src.model.positional_encodings import MixedPositionalEncoding, SinusoidalPositionalEncoding


class Encoder(nn.Module):
    """Embed multi-channel EEG windows into latent tokens.

    Pretraining (``forward``) produces two noisy views of the input and pulls
    their latent tokens together with a contrastive loss. Finetuning instead
    queries the embedded windows with a learned token (``get_latent_with_query``)
    or with a subject-adapter bias (``get_latent_with_bias``).
    """

    def __init__(
        self,
        embedder_params: dict,
        n_dim: int = 256,
        window_size: int = 256,
        max_sec: int = 32,
        depth: int = 2,
        _train_temperature: float = 0.7,
        **kwargs,
    ):
        super().__init__()

        # config
        self.n_dim = n_dim
        self.window_size = window_size
        self.max_sec = max_sec
        self.depth = depth
        self.encode_n_tok = 16
        self._train_temperature = _train_temperature

        # Learned tokens. ``mask_token`` is unused by the encoder itself but is
        # kept for backward compatibility with released checkpoints.
        self.mask_token = nn.Parameter(torch.randn(size=(1, 1, self.n_dim)) * 0.05, requires_grad=True)
        self.latent_token = nn.Parameter(torch.randn(size=(1, self.encode_n_tok, self.n_dim)) * 0.05, requires_grad=True)

        # modules
        self.embedder = CNNPatchEmbedder(embedder_params, window_size=self.window_size, n_dim=self.n_dim)
        self.mixed_positional_encoding = MixedPositionalEncoding(n_dim=self.n_dim)
        # Unused, kept for backward compatibility with released checkpoints.
        self.sin_position_encoder = SinusoidalPositionalEncoding(n_dim=self.n_dim, max_len=320)
        self.temporal_encoder = SinusoidalPositionalEncoding(n_dim=self.n_dim, max_len=self.max_sec)

        self.encoder = StackedEncoderLayers(n_dim=self.n_dim, n_head=8, depth=self.depth)
        self.embedding_projection = nn.Linear(in_features=self.n_dim, out_features=8)

    def forward(self, x: torch.Tensor, valid_mask: torch.BoolTensor, pos: torch.Tensor | None = None):
        # Build two noisy views of the same windows for contrastive pretraining.
        embedding = self.embedder(x)
        B, C, N, t = embedding.shape
        valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, C * N)

        embedding = rearrange(embedding, "b c n t -> (b c) n t")

        embedding_1 = embedding * 0.95 + 0.05 * torch.randn_like(embedding)
        embedding_2 = embedding * 0.95 + 0.05 * torch.randn_like(embedding)

        if pos is not None:
            pos = pos + torch.randn_like(pos) * torch.std(pos) * 0.05

        embedding_1 = self.temporal_encoder(embedding_1)
        embedding_1 = embedding_1.reshape(B, C, N, t)
        embedding_1 = embedding_1.transpose(2, 1)
        embedding_1 = self.mixed_positional_encoding(embedding_1, pos)
        embedding_1 = embedding_1.reshape(B, C * N, t)

        embedding_2 = self.temporal_encoder(embedding_2)
        embedding_2 = embedding_2.reshape(B, C, N, t)
        embedding_2 = embedding_2.transpose(2, 1)
        embedding_2 = self.mixed_positional_encoding(embedding_2)
        embedding_2 = embedding_2.reshape(B, C * N, t)

        latent_1 = self.encoder(x=embedding_1, latent=self.latent_token.expand(B, -1, -1), valid_mask=valid_mask)
        latent_2 = self.encoder(x=embedding_2, latent=self.latent_token.expand(B, -1, -1), valid_mask=valid_mask)

        contrastive_loss = self.get_contrastive_loss(latent_1, latent_2)
        embedding = rearrange(embedding, "(b c) n t -> b (c n) t", b=B, c=C)
        return latent_1, embedding, contrastive_loss

    def get_latent_with_query(self, x, query, valid_mask, pos=None):
        """Encode the EEG windows and cross-attend them with ``query``."""
        embedding = self.embedder(x)
        B, C, N, t = embedding.shape
        valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, C * N)

        embedding = embedding.reshape(B * C, N, t)
        embedding = self.temporal_encoder(embedding)
        embedding = embedding.reshape(B, C, N, t)
        embedding = embedding.transpose(2, 1)
        embedding = self.mixed_positional_encoding(embedding, pos)
        embedding = embedding.reshape(B, C * N, t)

        latent = self.encoder(x=embedding, latent=query, valid_mask=valid_mask)
        return latent

    def get_latent_with_bias(self, x, query, valid_mask, bias, pos=None):
        """Like ``get_latent_with_query`` but add a subject-specific ``bias`` first."""
        embedding = self.embedder(x)
        B, C, N, t = embedding.shape
        valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, C * N)

        embedding = embedding.reshape(B * C, N, t)
        embedding = self.temporal_encoder(embedding)
        embedding = embedding.reshape(B, C, N, t)
        embedding = embedding.transpose(2, 1)
        embedding = self.mixed_positional_encoding(embedding, pos)
        embedding = embedding.reshape(B, C, N, t)

        embedding = bias(embedding)
        embedding = embedding.reshape(B, C * N, t)
        latent = self.encoder(x=embedding, latent=query, valid_mask=valid_mask)

        return latent

    def get_latent_with_bias_and_attention(self, x, query, valid_mask, bias, pos=None):
        """Like ``get_latent_with_bias`` but also return the cross-attention scores."""
        embedding = self.embedder(x)
        B, C, N, t = embedding.shape
        valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, N).reshape(B, C * N)

        embedding = embedding.reshape(B * C, N, t)
        embedding = self.temporal_encoder(embedding)
        embedding = embedding.reshape(B, C, N, t)
        embedding = embedding.transpose(2, 1)
        embedding = self.mixed_positional_encoding(embedding, pos)
        embedding = embedding.reshape(B, C, N, t)

        embedding = bias(embedding)
        embedding = embedding.reshape(B, C * N, t)
        latent, attn_score = self.encoder.forward_with_attention(x=embedding, latent=query, valid_mask=valid_mask)

        return latent, attn_score

    def get_contrastive_loss(self, latent_1, latent_2):
        """InfoNCE-style contrastive loss between two views of the same windows.

        latent_1, latent_2: (B, N, n_dim)
        """
        B, N, n_dim = latent_1.shape

        latent_1 = F.normalize(latent_1, p=2, dim=-1)
        latent_2 = F.normalize(latent_2, p=2, dim=-1)

        latent_1 = latent_1.reshape(B * N, -1)
        latent_2 = latent_2.reshape(B * N, -1)

        similarity_matrix = torch.matmul(latent_1, latent_2.transpose(-1, -2)) / self._train_temperature
        labels = torch.arange(B * N, device=similarity_matrix.device)

        loss_i = F.cross_entropy(similarity_matrix, labels)
        loss_j = F.cross_entropy(similarity_matrix.transpose(-1, -2), labels)

        return loss_i + loss_j

    def _pretrain_parameter_setting(self):
        """Parameter groups for pretraining (separate weight decay for the projection)."""
        projection_params = []
        other_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "embedding_projection" in name:
                projection_params.append(param)
            else:
                other_params.append(param)

        return [
            {"params": projection_params, "weight_decay": 0.1, "lr": 1e-4},
            {"params": other_params, "weight_decay": 1e-3, "lr": 1e-4},
        ]


def build_encoder_from_checkpoint(path_to_ckpt: str, load_weights: bool = True) -> Encoder:
    """Build an :class:`Encoder` from a pretrain checkpoint.

    The checkpoint stores ``{"state_dict": ..., "config": ...}`` where ``config``
    holds the constructor arguments (with ``embedder_params`` as a variant name).
    """
    from src.config import get_embedder_params

    loaded_content = torch.load(path_to_ckpt, map_location="cpu", weights_only=False)
    config = dict(loaded_content["config"])
    # Resolve the embedder variant name to its layer config.
    config["embedder_params"] = get_embedder_params(config["embedder_params"])

    encoder = Encoder(**config)
    if load_weights:
        encoder.load_state_dict(loaded_content["state_dict"])
    return encoder
