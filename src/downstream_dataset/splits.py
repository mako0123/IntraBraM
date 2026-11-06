"""Train/test split helpers for downstream datasets.

Two protocols are supported:

* ``dependent``   : every subject is split independently into train/test
                   (seeded random split), one finetune run per subject.
* ``independent`` : subjects are divided evenly into ``n_fold`` folds; for
                   fold ``k`` the subjects of fold ``k`` are the test set and
                   all other subjects form the train set (n-fold CV).
"""

import random
from typing import List, Tuple

from src.downstream_dataset.base_dataset import BaseDownstreamDataset


def dependent_split(
    dataset: BaseDownstreamDataset,
    subject_id: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Randomly split one subject's samples into train/test indices."""
    grouped = dataset.get_indices_by_subject()
    if subject_id not in grouped:
        raise KeyError(f"subject_id {subject_id} not found; available: {sorted(grouped)}")

    indices = list(grouped[subject_id])
    rng = random.Random(seed + subject_id)
    rng.shuffle(indices)

    split = int(len(indices) * train_ratio)
    return indices[:split], indices[split:]


def independent_split(
    dataset: BaseDownstreamDataset,
    fold_idx: int,
    n_fold: int,
) -> Tuple[List[int], List[int]]:
    """Return train/test indices for fold ``fold_idx`` of an n-fold subject split."""
    grouped = dataset.get_indices_by_subject()
    subjects = sorted(grouped)
    n_subjects = len(subjects)

    if n_subjects % n_fold != 0:
        raise ValueError(
            f"number of subjects ({n_subjects}) must be divisible by n_fold ({n_fold})"
        )
    if not 0 <= fold_idx < n_fold:
        raise ValueError(f"fold_idx must be in [0, {n_fold}), got {fold_idx}")

    fold_size = n_subjects // n_fold
    test_subjects = set(subjects[fold_idx * fold_size:(fold_idx + 1) * fold_size])

    train_indices: List[int] = []
    test_indices: List[int] = []
    for subject_id, idxs in grouped.items():
        if subject_id in test_subjects:
            test_indices.extend(idxs)
        else:
            train_indices.extend(idxs)
    return train_indices, test_indices


def get_split_ids(dataset: BaseDownstreamDataset, dependency: str, n_fold: int | None = None) -> list:
    """Return the split ids to iterate over.

    For ``dependent`` this is the sorted subject ids (which may be non-contiguous);
    for ``independent`` it is ``range(n_fold)``.
    """
    if dependency == "dependent":
        return sorted(dataset.get_indices_by_subject().keys())
    if dependency == "independent":
        if n_fold is None:
            raise ValueError("n_fold is required for independent splitting")
        if dataset.get_num_subjects() % n_fold != 0:
            raise ValueError(
                f"number of subjects ({dataset.get_num_subjects()}) must be divisible by n_fold ({n_fold})"
            )
        return list(range(n_fold))
    raise ValueError(f"Unknown dependency '{dependency}'; expected 'dependent' or 'independent'")


def get_num_splits(dataset: BaseDownstreamDataset, dependency: str, n_fold: int | None = None) -> int:
    """Number of finetune runs: number of subjects (dependent) or folds (independent)."""
    return len(get_split_ids(dataset, dependency, n_fold))
