"""Finetuning entry point (IntraBraM backbone + classification head).

Usage:
    python script/finetune.py --version intrabram-base --dependency dependent --subject 0
    python script/finetune.py --version intrabram-base --dependency independent --fold-idx 2

The dataset config is read from ``--dataset-config`` (default: the dummy YAML for
``--dependency``); the task name comes from the config's ``task`` field, and
training hyperparameters from ``config/train/finetune.yaml``. Finetuned weights
are saved per epoch under ``{version}-{task}-{dependency}-{subject-n|fold-n}-{timestamp}``
in the checkpoint directory from ``config/path.yaml``.
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.strategies import DDPStrategy

from src.config import (
    get_checkpoint_dir,
    get_default_downstream_dataset_config_path,
    get_finetune_config,
    load_downstream_dataset_config,
)
from src.downstream_dataset import build_downstream_dataset, dependent_split, independent_split
from src.model.finetune_model import FinetuneModel

VERSIONS = ["intrabram-large", "intrabram-mid", "intrabram-base", "intrabram-small"]


def latest_checkpoint(run_dir: str) -> str:
    """Return the highest-epoch ``epoch-*.ckpt`` inside ``run_dir``."""
    ckpts = glob.glob(os.path.join(run_dir, "epoch-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no epoch-*.ckpt found in {run_dir}")
    return max(ckpts, key=lambda p: int(Path(p).stem.split("-")[1]))


def get_train_indices(dataset, dependency: str, split_id: int, dataset_config: dict):
    """Return the training indices for one subject (dependent) or fold (independent)."""
    if dependency == "dependent":
        train_idx, _ = dependent_split(
            dataset,
            split_id,
            train_ratio=dataset_config.get("train_ratio", 0.8),
            seed=dataset_config.get("seed", 42),
        )
    elif dependency == "independent":
        train_idx, _ = independent_split(dataset, split_id, n_fold=dataset_config["n_fold"])
    else:
        raise ValueError(f"Unknown dependency '{dependency}'; expected 'dependent' or 'independent'")
    return train_idx


def default_run_dir(version: str, task: str, dependency: str, split_id: int) -> str:
    """``{checkpoint_dir}/{version}-{task}-{dependency}-{subject-n|fold-n}-{timestamp}``."""
    split_tag = f"fold-{split_id}" if dependency == "independent" else f"subject-{split_id}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(get_checkpoint_dir(), f"{version}-{task}-{dependency}-{split_tag}-{timestamp}")


def run_finetune(
    version: str,
    dependency: str,
    split_id: int,
    checkpoint: str,
    dataset_config: dict,
    run_dir: str,
    epochs: int = 3,
    batch_size: int = 8,
    num_workers: int = 8,
    seed: int = 42,
    use_bias: bool = False,
    load_weights: bool = True,
    devices=None,
) -> str:
    """Finetune one split and return the shared run directory holding the epoch checkpoints.

    The task name is read from ``dataset_config["task"]``. ``run_dir`` is passed
    to every process so all ranks agree on where the (rank-0) checkpoints live.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    task = dataset_config["task"]

    dataset = build_downstream_dataset(dataset_config)
    train_idx = get_train_indices(dataset, dependency, split_id, dataset_config)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    
    n_devices = len(devices) 
    use_ddp = n_devices > 1
    strategy = DDPStrategy(find_unused_parameters=True) if use_ddp else "auto"

    model = FinetuneModel(
        pretrain_path=checkpoint,
        n_cls=dataset_config["n_cls"],
        load_weights=load_weights,
        use_bias=use_bias,
        max_channel_num=dataset_config["max_channel_num"],
        segment_t=dataset_config["segment_t"],
        version=version,
        task=task,
        dependency=dependency,
        split_id=split_id,
        run_dir=run_dir,
    )

    # ``save_finetune`` handles epoch checkpoints; Lightning checkpointing is disabled.
    trainer = L.Trainer(
        max_epochs=epochs,
        devices=devices,
        log_every_n_steps=10,
        enable_checkpointing=False,
        logger=False,
        strategy=strategy,
    )
    trainer.fit(model, train_loader)
    return run_dir


def parse_args() -> argparse.Namespace:
    cfg = get_finetune_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--version", default="intrabram-base", choices=VERSIONS, help="IntraBraM variant.")
    parser.add_argument("--dependency", default="dependent", choices=["dependent", "independent"])
    parser.add_argument("--subject", type=int, default=None, help="Subject id (required for dependent).")
    parser.add_argument("--fold-idx", type=int, default=None, help="Fold index (required for independent).")
    parser.add_argument("--checkpoint", required=True, help="Pretrain encoder checkpoint to load.")
    parser.add_argument("--dataset-config", default=None, help="Dataset YAML; defaults to the dummy YAML for --dependency.")
    parser.add_argument("--run-dir", default=None, help="Explicit run directory (default: computed with a timestamp).")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"], help="Number of finetuning epochs.")
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--devices", default=None, help="Comma-separated device ids (default from config).")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--use-bias", action="store_true", default=None, help="Add the subject-adapter bias path.")

    args = parser.parse_args()

    if args.dependency == "dependent":
        if args.subject is None:
            parser.error("--subject is required when --dependency dependent")
        args.split_id = args.subject
    else:
        if args.fold_idx is None:
            parser.error("--fold-idx is required when --dependency independent")
        args.split_id = args.fold_idx

    if args.devices is not None:
        parsed = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
        args.devices = parsed
    else:
        args.devices = cfg["devices"]
    if args.use_bias is None:
        args.use_bias = bool(cfg.get("use_bias", False))

    return args


def main() -> None:
    args = parse_args()
    train_cfg = get_finetune_config()

    checkpoint = args.checkpoint
    dataset_config_path = args.dataset_config or get_default_downstream_dataset_config_path(args.dependency)
    dataset_config = load_downstream_dataset_config(dataset_config_path)
    run_dir = args.run_dir or default_run_dir(args.version, dataset_config["task"], args.dependency, args.split_id)

    run_finetune(
        version=args.version,
        dependency=args.dependency,
        split_id=args.split_id,
        checkpoint=checkpoint,
        dataset_config=dataset_config,
        run_dir=run_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_bias=args.use_bias,
        load_weights=train_cfg.get("load_weights", True),
        devices=args.devices,
    )
    # Only the main process reports the directory (child DDP processes also rerun main()).
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
