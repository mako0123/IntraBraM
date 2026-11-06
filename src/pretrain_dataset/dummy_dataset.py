"""A dummy pretraining dataset built with random arrays."""

import numpy as np

from src.pretrain_dataset.base_dataset import BasePretrainDataset


class DummyPretrainDataset(BasePretrainDataset):
    """Random-data dataset used as a template for :class:`BasePretrainDataset`.

    ``load_sample`` generates deterministic pseudo-random arrays so the dataset
    is reproducible (per index) and does not require any real data files.
    """

    def __init__(
        self,
        num_channels: int = 224,
        segment_len: int = 1280,
        num_samples: int = 1024,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_channels = num_channels
        self.segment_len = segment_len
        self.num_samples = num_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def load_sample(self, idx: int):
        # Seed by index so each sample is identical across workers and runs.
        rng = np.random.default_rng(self.seed + idx)

        eeg = rng.standard_normal((self.num_channels, self.segment_len)).astype(np.float32)
        # All channels are valid here; a real dataset may set some to False.
        mask = np.ones(self.num_channels, dtype=bool)
        mni_pos = rng.uniform(-70.0, 70.0, size=(self.num_channels, 3)).astype(np.float32)

        return {"eeg": eeg, "mask": mask, "mni_pos": mni_pos}
