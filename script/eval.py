"""Evaluation entry point: load a finetuned checkpoint and test it.

Usage:
    python script/eval.py --checkpoint checkpoints/intrabram-base-dummy-dependent-subject-0-.../epoch-2.ckpt

The task, dependency and split id are read from the checkpoint config; the
dataset YAML defaults to ``config/downstream_dataset/{task}_{dependency}.yaml``.
Metrics are written to ``results/{version}-{task}-{dependency}-split-{n}/``.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.strategies import DDPStrategy

from src.config import (
    get_downstream_dataset_config_path,
    get_finetune_config,
    load_downstream_dataset_config,
)
from src.downstream_dataset import build_downstream_dataset, dependent_split, independent_split
from src.model.finetune_model import build_finetune_model

RESULTS_DIR = "results"


def get_test_indices(dataset, dependency: str, split_id: int, dataset_config: dict):
    """Return the test indices for one subject (dependent) or fold (independent)."""
    if dependency == "dependent":
        _, test_idx = dependent_split(
            dataset,
            split_id,
            train_ratio=dataset_config.get("train_ratio", 0.8),
            seed=dataset_config.get("seed", 42),
        )
    elif dependency == "independent":
        _, test_idx = independent_split(dataset, split_id, n_fold=dataset_config["n_fold"])
    else:
        raise ValueError(f"Unknown dependency '{dependency}'; expected 'dependent' or 'independent'")
    return test_idx


def run_eval(
    checkpoint: str,
    dataset_config: dict,
    dependency: str,
    split_id: int,
    batch_size: int = 8,
    num_workers: int = 8,
    devices=None,
) -> dict:
    """Load a finetuned checkpoint, run the held-out test split, and return metrics."""
    model = build_finetune_model(checkpoint)

    dataset = build_downstream_dataset(dataset_config)
    test_idx = get_test_indices(dataset, dependency, split_id, dataset_config)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, num_workers=num_workers)

    n_devices = len(devices) 
    use_ddp = n_devices > 1
    strategy = DDPStrategy(find_unused_parameters=True) if use_ddp else "auto"
    
    trainer = L.Trainer(
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        strategy=strategy,
    )
    trainer.test(model, test_loader)
    return model.summarize()


def parse_args() -> argparse.Namespace:
    cfg = get_finetune_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--checkpoint", required=True, help="Finetune checkpoint (epoch-n.ckpt).")
    parser.add_argument("--dataset-config", default=None, help="Dataset YAML; defaults to {task}_{dependency}.yaml.")
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--devices", default=None, help="Comma-separated device ids (default from config).")
    parser.add_argument("--results-path", default=None, help="Explicit results csv path (default: computed).")

    args = parser.parse_args()

    if args.devices is not None:
        parsed = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
        args.devices = parsed
    else:
        args.devices = cfg["devices"]
    return args


def main() -> None:
    args = parse_args()

    # Read split metadata from the checkpoint config.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_cfg = ckpt["config"]
    version, task = ckpt_cfg.get("version"), ckpt_cfg.get("task")
    dependency, split_id = ckpt_cfg.get("dependency"), ckpt_cfg.get("split_id")

    dataset_config_path = args.dataset_config or get_downstream_dataset_config_path(task, dependency)
    dataset_config = load_downstream_dataset_config(dataset_config_path)

    metrics = run_eval(
        checkpoint=args.checkpoint,
        dataset_config=dataset_config,
        dependency=dependency,
        split_id=split_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        devices=args.devices,
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    metrics["time_stamp"] = str(datetime.now())
    metrics["checkpoint"] = args.checkpoint

    if args.results_path is not None:
        out_path = args.results_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(RESULTS_DIR, f"{version}-{task}-{dependency}-split-{split_id}")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{task}-{timestamp}.csv")
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"RESULTS_CSV={out_path}")


if __name__ == "__main__":
    main()
