"""Configuration loading helpers.

All model/training hyperparameters live in ``config/`` as YAML files so the
code can run on a fresh machine without hard-coded paths or constants.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import yaml

# Repository root (the parent directory of ``src/``).
ROOT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR: str = os.path.join(ROOT_DIR, "config")

_MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model", "model.yaml")
_EMBEDDER_CONFIG_PATH = os.path.join(CONFIG_DIR, "model", "embedder.yaml")
_PATH_CONFIG_PATH = os.path.join(CONFIG_DIR, "path.yaml")
_PRETRAIN_CONFIG_PATH = os.path.join(CONFIG_DIR, "train", "pretrain.yaml")
_FINETUNE_CONFIG_PATH = os.path.join(CONFIG_DIR, "train", "finetune.yaml")

DOWNSTREAM_DATASET_DIR = os.path.join(CONFIG_DIR, "downstream_dataset")
DEFAULT_DOWNSTREAM_TASK = "dummy"


def get_downstream_dataset_config_path(task: str, dependency: str) -> str:
    """Return the YAML path for a task/protocol pair (e.g. ``dummy``/``dependent``)."""
    return os.path.join(DOWNSTREAM_DATASET_DIR, f"{task}_{dependency}.yaml")


def get_default_downstream_dataset_config_path(dependency: str) -> str:
    """Return the default downstream dataset YAML for a protocol."""
    return get_downstream_dataset_config_path(DEFAULT_DOWNSTREAM_TASK, dependency)


def load_yaml(path: str) -> Dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_variants() -> Dict[str, Any]:
    """Return the architecture config of every IntraBraM variant."""
    return load_yaml(_MODEL_CONFIG_PATH)


def get_model_config(version: str) -> Dict[str, Any]:
    """Return the architecture config of a single variant (e.g. ``intrabram-base``)."""
    cfg = get_model_variants()
    if version not in cfg:
        raise KeyError(f"Unknown model variant '{version}'. Available: {sorted(cfg)}")
    return cfg[version]


def get_embedder_params(name: str) -> Dict[str, Any]:
    """Return the CNN patch-embedder layer config for a variant name."""
    cfg = load_yaml(_EMBEDDER_CONFIG_PATH)
    if name not in cfg:
        raise KeyError(f"Unknown embedder config '{name}'. Available: {sorted(cfg)}")
    return cfg[name]


def get_path_config() -> Dict[str, Any]:
    """Return the contents of ``config/path.yaml``."""
    return load_yaml(_PATH_CONFIG_PATH)


def get_checkpoint_dir() -> str:
    """Return the directory where pretraining saves encoder checkpoints."""
    return get_path_config()["checkpoint_dir"]


def get_pretrain_config() -> Dict[str, Any]:
    """Return the default pretraining configuration."""
    return load_yaml(_PRETRAIN_CONFIG_PATH)


def get_finetune_config() -> Dict[str, Any]:
    """Return the default finetuning configuration."""
    return load_yaml(_FINETUNE_CONFIG_PATH)


def load_downstream_dataset_config(path: str) -> Dict[str, Any]:
    """Load a downstream dataset YAML config from an explicit path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Downstream dataset config not found: {path}")
    return load_yaml(path)
