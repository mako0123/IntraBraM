"""CNN patch embedder that turns raw multi-channel EEG into per-window tokens."""

from einops import rearrange
from torch import nn, Tensor


class SingleChannelCNNBlock(nn.Module):
    """One convolutional block applied to a single channel's window.

    A stride-2 max pool follows the convolution by default, so three stacked
    blocks shrink the temporal dimension by a factor of 8.
    """

    def __init__(
        self,
        input_channel: int,
        output_channel: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        max_pool: bool = True,
        norm: bool = True,
        relu: bool = True,
    ):
        super().__init__()
        self.input_channels = input_channel
        self.output_channels = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.module = nn.Sequential()

        cnn_kwargs = dict(
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        # Only add the stride to the convolution when there is no max pool.
        if not max_pool:
            cnn_kwargs.update(stride=self.stride)

        self.cnn = nn.Conv1d(**cnn_kwargs)
        self.module.append(self.cnn)

        self.norm = nn.BatchNorm1d(num_features=self.output_channels) if norm else nn.Identity()
        self.module.append(self.norm)

        self.relu = nn.ReLU(inplace=True) if relu else nn.Identity()
        self.module.append(self.relu)

        self.max_pool = (
            nn.MaxPool1d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
            if max_pool
            else nn.Identity()
        )
        self.module.append(self.max_pool)

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)


class CNNPatchEmbedder(nn.Module):
    """Embed each non-overlapping temporal window into an ``n_dim`` token.

    Input: (B, C, T). Output: (B, C, N, n_dim) with N = T // window_size.
    """

    def __init__(self, embedder_parms: dict, n_dim: int, window_size: int = 256):
        super().__init__()
        self.window_size = window_size
        self.n_dim = n_dim

        self.cnn1 = SingleChannelCNNBlock(**embedder_parms["layer0"])
        self.cnn2 = SingleChannelCNNBlock(**embedder_parms["layer1"])
        self.cnn3 = SingleChannelCNNBlock(**embedder_parms["layer2"])

        # The three pooling blocks shrink each window by 8x.
        self.ffn = nn.Sequential(
            nn.LayerNorm(embedder_parms["layer2"]["output_channel"] * (window_size // 8)),
            nn.Linear(embedder_parms["layer2"]["output_channel"] * (window_size // 8), self.n_dim * 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.n_dim * 4, self.n_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, T = x.shape
        x = x.unsqueeze(dim=-1).reshape(B, C, -1, self.window_size)

        # (B, C, N, T) -> treat every channel-window pair as an independent sample.
        B, C, N, T = x.shape
        x = rearrange(x, "b c n t -> (b c n) 1 t")

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        x = x.reshape(B * C * N, -1)
        x = self.ffn(x)
        x = x.reshape(B, C, N, -1)
        return x
