"""A dummy downstream dataset built with random arrays."""

import numpy as np

from src.downstream_dataset.base_dataset import BaseDownstreamDataset


class DummyDownstreamDataset(BaseDownstreamDataset):
    """Random-data dataset with ``num_subjects`` integer subject ids.

    Subject ids are assigned round-robin (``idx % num_subjects``) so every
    subject owns roughly the same number of samples.
    """

    def __init__(
        self,
        num_channels: int = 224,
        segment_len: int = 1280,
        num_classes: int = 2,
        num_samples: int = 1024,
        num_subjects: int = 8,
        use_pos: bool = False,
        seed: int = 0,
        path_to_dataset: str = "",
        **kwargs,
    ):
        super().__init__(path_to_dataset=path_to_dataset, use_pos=use_pos, **kwargs)
        self.num_channels = num_channels
        self.segment_len = segment_len
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.num_subjects = num_subjects
        self.seed = seed
        self.subject_ids = [idx % num_subjects for idx in range(num_samples)]

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int):
        # Seed by index so each sample is identical across workers and runs.
        rng = np.random.default_rng(self.seed + idx)

        sample = {
            "eeg": rng.standard_normal((self.num_channels, self.segment_len)).astype(np.float32),
            # All channels are valid here; a real dataset may set some to False.
            "mask": np.ones(self.num_channels, dtype=bool),
            "label": int(rng.integers(0, self.num_classes)),
        }
        if self.use_pos:
            sample["mni_pos"] = rng.uniform(-70.0, 70.0, size=(self.num_channels, 3)).astype(np.float32)
        return sample
