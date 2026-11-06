"""Batch finetune + eval over every split of a downstream dataset config.

Usage:
    python script/run_experiment.py --version intrabram-base --dependency dependent \
        --dataset-config config/downstream_dataset/dummy_dependent.yaml
    python script/run_experiment.py --version intrabram-base --dependency independent \
        --dataset-config config/downstream_dataset/dummy_independent.yaml --checkpoint /path/to/ckpt

Each split is run as a separate ``finetune.py`` / ``eval.py`` subprocess. This is
intentional: the multi-GPU DDP launcher re-runs the calling script in its child
processes, so wrapping only a single split keeps that safe. Checkpoints are saved
by rank 0 only, and per-rank test results are gathered before summarizing.
"""

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.config import (
    get_checkpoint_dir,
    get_default_downstream_dataset_config_path,
    get_finetune_config,
    load_downstream_dataset_config,
)
from src.downstream_dataset import build_downstream_dataset, get_split_ids

RESULTS_DIR = "results"
VERSIONS = ["intrabram-large", "intrabram-mid", "intrabram-base", "intrabram-small"]


def _latest_checkpoint(run_dir: str) -> str:
    ckpts = glob.glob(os.path.join(run_dir, "epoch-*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"no epoch-*.ckpt found in {run_dir}")
    return max(ckpts, key=lambda p: int(Path(p).stem.split("-")[1]))


def parse_args() -> argparse.Namespace:
    cfg = get_finetune_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--version", default="intrabram-base", choices=VERSIONS)
    parser.add_argument("--dependency", required=True, choices=["dependent", "independent"])
    parser.add_argument("--checkpoint", required=True, help="Pretrain encoder checkpoint to load.")
    parser.add_argument("--dataset-config", default=None, help="Dataset YAML; defaults to the dummy YAML for --dependency.")
    parser.add_argument("--epochs", type=int, default=cfg["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"])
    parser.add_argument("--devices", default=None, help="Comma-separated device ids (default from config).")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42))
    parser.add_argument("--use-bias", action="store_true", default=None)

    args = parser.parse_args()

    if args.devices is not None:
        parsed = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
        args.devices = parsed
    else:
        args.devices = cfg["devices"]
    if args.use_bias is None:
        args.use_bias = bool(cfg.get("use_bias", False))
    return args


def _run(cmd) -> None:
    print(f"[run_experiment] {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    args = parse_args()

    checkpoint = args.checkpoint
    dataset_config_path = args.dataset_config or get_default_downstream_dataset_config_path(args.dependency)
    dataset_config = load_downstream_dataset_config(dataset_config_path)
    task = dataset_config["task"]

    dataset = build_downstream_dataset(dataset_config)
    split_ids = get_split_ids(dataset, args.dependency, dataset_config.get("n_fold"))
    print(f"Running {len(split_ids)} split(s) for task={task} dependency={args.dependency}: {split_ids}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, f"{args.version}-{task}-{args.dependency}")
    device_arg = ",".join(str(d) for d in args.devices)
    rows = []

    for split_id in split_ids:
        split_tag = f"fold-{split_id}" if args.dependency == "independent" else f"subject-{split_id}"
        run_dir = os.path.join(get_checkpoint_dir(), f"{args.version}-{task}-{args.dependency}-{split_tag}-{timestamp}")
        results_path = os.path.join(results_dir, f"{split_tag}-{timestamp}.csv")
        split_args = ["--subject", str(split_id)] if args.dependency == "dependent" else ["--fold-idx", str(split_id)]

        finetune_cmd = [
            sys.executable, str(_SCRIPT_DIR / "finetune.py"),
            "--version", args.version,
            "--dependency", args.dependency,
            *split_args,
            "--checkpoint", checkpoint,
            "--dataset-config", dataset_config_path,
            "--run-dir", run_dir,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--seed", str(args.seed),
            "--devices", device_arg,
        ]
        if args.use_bias:
            finetune_cmd.append("--use-bias")
        _run(finetune_cmd)

        ckpt = _latest_checkpoint(run_dir)
        eval_cmd = [
            sys.executable, str(_SCRIPT_DIR / "eval.py"),
            "--checkpoint", ckpt,
            "--dataset-config", dataset_config_path,
            "--results-path", results_path,
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--devices", device_arg,
        ]
        _run(eval_cmd)

        row = pd.read_csv(results_path).iloc[0].to_dict()
        row["split_id"] = split_id
        rows.append(row)
        print(f"split {split_id}: accuracy={row.get('accuracy')}, f1={row.get('f1_score')}, auc={row.get('auc')}", flush=True)

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"{task}-{args.dependency}-{timestamp}.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Results saved to {out_path}", flush=True)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
