"""Abstract base class for the IntraBraM pretraining dataset."""

from abc import ABC, abstractmethod
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Keys and shapes returned by ``__getitem__``.
#   eeg     : FloatTensor of shape (num_channels, segment_len)
#   mask    : BoolTensor of shape (num_channels,)
#   mni_pos : FloatTensor of shape (num_channels, 3)
SAMPLE_KEYS = ("eeg", "mask", "mni_pos")

ArrayLike = Union[np.ndarray, torch.Tensor]


class BasePretrainDataset(Dataset, ABC):
    """Abstract pretraining dataset enforcing a fixed ``__getitem__`` schema.

    Subclasses only implement the raw data loading in :meth:`load_sample` (and
    :meth:`__len__`); this base class converts the returned fields to the
    expected dtypes and validates their shapes, so downstream code can rely on:

        {"eeg": FloatTensor (C, T), "mask": BoolTensor (C,), "mni_pos": FloatTensor (C, 3)}

    where ``C`` is the number of channels and ``T`` the number of time samples
    (a multiple of ``window_size``).
    """

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def load_sample(self, idx: int) -> Dict[str, ArrayLike]:
        """Load the raw fields for sample ``idx``.

        Must return a dict with the keys ``eeg``, ``mask`` and ``mni_pos``; the
        values may be numpy arrays or torch tensors with the shapes above.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.load_sample(idx)
        return self._standardize(sample)

    def _standardize(self, sample: Dict[str, ArrayLike]) -> Dict[str, torch.Tensor]:
        if not isinstance(sample, dict):
            raise TypeError(f"load_sample must return a dict, got {type(sample).__name__}")

        missing = [key for key in SAMPLE_KEYS if key not in sample]
        if missing:
            raise KeyError(f"load_sample is missing required key(s): {missing}")

        eeg = self._as_tensor(sample["eeg"], dtype=torch.float32)
        mask = self._as_tensor(sample["mask"], dtype=torch.bool)
        mni_pos = self._as_tensor(sample["mni_pos"], dtype=torch.float32)

        if eeg.ndim != 2:
            raise ValueError(f"'eeg' must have shape (num_channels, segment_len), got {tuple(eeg.shape)}")
        if mask.ndim != 1:
            raise ValueError(f"'mask' must have shape (num_channels,), got {tuple(mask.shape)}")
        if mni_pos.ndim != 2 or mni_pos.shape[1] != 3:
            raise ValueError(f"'mni_pos' must have shape (num_channels, 3), got {tuple(mni_pos.shape)}")

        num_channels = eeg.shape[0]
        if mask.shape[0] != num_channels:
            raise ValueError(
                f"'mask' channel count {mask.shape[0]} != 'eeg' channel count {num_channels}"
            )
        if mni_pos.shape[0] != num_channels:
            raise ValueError(
                f"'mni_pos' channel count {mni_pos.shape[0]} != 'eeg' channel count {num_channels}"
            )

        return {"eeg": eeg, "mask": mask, "mni_pos": mni_pos}

    @staticmethod
    def _as_tensor(value: ArrayLike, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.from_numpy(np.asarray(value)).to(dtype=dtype)
