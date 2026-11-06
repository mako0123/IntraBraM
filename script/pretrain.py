"""Pretraining entry point for IntraBraM (uses the DummyPretrainDataset).

Usage:
    python script/pretrain.py --version intrabram-base
    python script/pretrain.py --version intrabram-small --epochs 1 --devices 1

The model architecture is initialized from ``config/model/model.yaml`` and the
training hyperparameters from ``config/train/pretrain.yaml``. Command-line flags
override individual training settings for quick experimentation.

Please replace the DummyPretrainDatset with real-world dataset when pretraining the model

"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Make the repository root importable when running `python script/pretrain.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from src.config import get_model_config, get_pretrain_config
from src.model.pretrain_model import IntraBraM
from src.pretrain_dataset import DummyPretrainDataset # Pleace replace the dummy dataset with the real pretrain dataset when pretraining the model

VERSIONS = ["intrabram-large", "intrabram-mid", "intrabram-base", "intrabram-small"]


def parse_args() -> argparse.Namespace:
    cfg = get_pretrain_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--version", default="intrabram-base", choices=VERSIONS,
                        help="IntraBraM variant to pretrain.")
    parser.add_argument("--epochs", type=int, default=cfg["max_epochs"],
                        help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=cfg["batch_size"],
                        help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=cfg["num_workers"],
                        help="DataLoader workers.")
    parser.add_argument("--devices", default=None,
                        help="Comma-separated device ids (e.g. '0,1'). Defaults to the config value.")
    parser.add_argument("--seed", type=int, default=cfg.get("seed", 42),
                        help="Random seed.")
    parser.add_argument("--no-reconstruction", action="store_true", default=None,
                        help="Disable the reconstruction loss.")
    parser.add_argument("--no-contrastive", action="store_true", default=None,
                        help="Disable the contrastive loss.")

    args = parser.parse_args()

    # Resolve the devices (comma-separated CLI string or the config list).
    if args.devices is not None:
        parsed = [int(d.strip()) for d in args.devices.split(",") if d.strip()]
        args.devices = parsed
    else:
        args.devices = cfg["devices"]

    # Store_true flags default to None; fall back to the config value.
    if args.no_reconstruction is None:
        args.no_reconstruction = bool(cfg.get("no_reconstruction", False))
    if args.no_contrastive is None:
        args.no_contrastive = bool(cfg.get("no_contrastive", False))

    return args


def main() -> None:
    args = parse_args()
    train_cfg = get_pretrain_config()
    model_cfg = get_model_config(args.version)

    # Build the model from config files.
    model = IntraBraM(
        n_dim=model_cfg["n_dim"],
        window_size=model_cfg["window_size"],
        max_sec=model_cfg["max_sec"],
        depth=model_cfg["depth"],
        embedder_params=model_cfg["embedder"],
        _train_no_reconstruction=args.no_reconstruction,
        _train_no_contrastive=args.no_contrastive,
        _train_mask_rate=train_cfg.get("mask_rate", 0.5),
        _train_temperature=train_cfg.get("temperature", 0.7),
    )

    # Dummy dataset: replace with a real BasePretrainDataset subclass for real data.
    ds_cfg = train_cfg.get("dataset", {})
    segment_len = ds_cfg.get("segment_len", 1280)
    assert segment_len % model_cfg["window_size"] == 0, (
        f"dataset.segment_len ({segment_len}) must be a multiple of window_size ({model_cfg['window_size']})"
    )
    train_dataset = DummyPretrainDataset(
        num_channels=ds_cfg.get("num_channels", 224),
        segment_len=segment_len,
        num_samples=ds_cfg.get("num_samples", 1024),
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=train_cfg["checkpoint"]["dirpath"],
        filename=f"{args.version}-" + "{epoch:02d}-{step}",
        every_n_train_steps=train_cfg["checkpoint"]["every_n_train_steps"],
    )
    logger = TensorBoardLogger("logs", name=args.version, version=datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    # DDP needs find_unused_parameters because some encoder params are not used in every forward.
    n_devices = len(args.devices) 
    use_ddp = n_devices > 1
    strategy = DDPStrategy(find_unused_parameters=True) if use_ddp else "auto"

    trainer = L.Trainer(
        max_epochs=args.epochs,
        strategy=strategy,
        log_every_n_steps=10,
        devices=args.devices,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
