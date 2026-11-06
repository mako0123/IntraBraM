"""Downstream (finetuning / evaluation) datasets."""

from src.downstream_dataset.base_dataset import BaseDownstreamDataset
from src.downstream_dataset.dummy_dataset import DummyDownstreamDataset
from src.downstream_dataset.splits import (
    dependent_split,
    get_num_splits,
    get_split_ids,
    independent_split,
)

__all__ = [
    "BaseDownstreamDataset",
    "DummyDownstreamDataset",
    "build_downstream_dataset",
    "dependent_split",
    "independent_split",
    "get_num_splits",
    "get_split_ids",
]


def build_downstream_dataset(config: dict) -> BaseDownstreamDataset:
    """Build a downstream dataset from a YAML config dict.

    This turns a ``dummy_*.yaml`` config into a :class:`DummyDownstreamDataset`
    with ``num_subjects`` integer subject ids. Replace this factory with your
    real dataset class when finetuning on real data.
    """
    return DummyDownstreamDataset(
        path_to_dataset=config.get("path_to_dataset", ""),
        num_channels=config["max_channel_num"],
        segment_len=int(config["segment_t"] * config["frequency"]),
        num_classes=config["n_cls"],
        num_samples=config["num_samples"],
        num_subjects=config.get("num_subjects", 8),
        use_pos=config.get("use_pos", False),
        seed=config.get("seed", 0),
    )
