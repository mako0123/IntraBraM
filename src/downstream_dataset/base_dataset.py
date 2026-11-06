"""Abstract base class for downstream (finetuning / evaluation) datasets."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

# Required keys returned by ``load_sample``; ``mni_pos`` is optional.
REQUIRED_KEYS = ("eeg", "mask", "label")

ArrayLike = Union[np.ndarray, torch.Tensor]


class BaseDownstreamDataset(Dataset, ABC):
    """Abstract downstream dataset with a unified ``__getitem__`` contract.

    Subclasses implement :meth:`load_sample` (returning a dict with ``eeg``,
    ``mask``, ``label`` and, when ``use_pos`` is set, ``mni_pos``) and
    :meth:`__len__`, and must set ``self.subject_ids`` (one integer subject id
    per sample) so the split helpers can group samples by subject.
    ``__getitem__`` normalizes dtypes/shapes and returns ``(eeg, mask, label)``
    or ``(eeg, mask, label, mni_pos)``, which is what
    :class:`~src.model.finetune_model.FinetuneModel` consumes.

    Per-sample shapes:
        eeg     : (num_channels, segment_len) float32
        mask    : (num_channels,) bool
        label   : () long
        mni_pos : (num_channels, 3) float32
    """

    def __init__(self, path_to_dataset: str = "", use_pos: bool = False, **kwargs):
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.use_pos = use_pos
        self.subject_ids: List[int] = []  # one per sample; set by subclasses

    @abstractmethod
    def load_sample(self, idx: int) -> Dict[str, ArrayLike]:
        """Return the raw fields for sample ``idx`` as a dict."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int):
        sample = self.load_sample(idx)
        return self._standardize(sample)

    # ------------------------------------------------------------- subject info
    def get_subject_ids(self) -> List[int]:
        """Return the subject id of every sample."""
        if len(self.subject_ids) != len(self):
            raise ValueError(
                f"subject_ids must have one entry per sample: {len(self.subject_ids)} != {len(self)}"
            )
        return self.subject_ids

    def get_indices_by_subject(self) -> Dict[int, List[int]]:
        """Group sample indices by subject id."""
        grouped = defaultdict(list)
        for idx, subject_id in enumerate(self.get_subject_ids()):
            grouped[subject_id].append(idx)
        return dict(grouped)

    def get_num_subjects(self) -> int:
        return len(self.get_indices_by_subject())

    # ------------------------------------------------------------------- helpers
    def _standardize(self, sample: Dict[str, ArrayLike]):
        if not isinstance(sample, dict):
            raise TypeError(f"load_sample must return a dict, got {type(sample).__name__}")

        missing = [key for key in REQUIRED_KEYS if key not in sample]
        if missing:
            raise KeyError(f"load_sample is missing required key(s): {missing}")

        eeg = self._as_tensor(sample["eeg"], dtype=torch.float32)
        mask = self._as_tensor(sample["mask"], dtype=torch.bool)
        label = self._as_tensor(sample["label"], dtype=torch.long).reshape(())

        if eeg.ndim != 2:
            raise ValueError(f"'eeg' must have shape (num_channels, segment_len), got {tuple(eeg.shape)}")
        if mask.ndim != 1:
            raise ValueError(f"'mask' must have shape (num_channels,), got {tuple(mask.shape)}")
        if mask.shape[0] != eeg.shape[0]:
            raise ValueError(f"'mask' size {mask.shape[0]} != 'eeg' channels {eeg.shape[0]}")

        if not self.use_pos:
            return eeg, mask, label

        if "mni_pos" not in sample:
            raise KeyError("use_pos=True but 'mni_pos' is missing from load_sample")
        mni_pos = self._as_tensor(sample["mni_pos"], dtype=torch.float32)
        if mni_pos.ndim != 2 or mni_pos.shape[1] != 3:
            raise ValueError(f"'mni_pos' must have shape (num_channels, 3), got {tuple(mni_pos.shape)}")
        if mni_pos.shape[0] != eeg.shape[0]:
            raise ValueError(f"'mni_pos' channels {mni_pos.shape[0]} != 'eeg' channels {eeg.shape[0]}")

        return eeg, mask, label, mni_pos

    @staticmethod
    def _as_tensor(value: ArrayLike, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(dtype=dtype)
        return torch.from_numpy(np.asarray(value)).to(dtype=dtype)
