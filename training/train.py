"""
training/train.py

Command-line training entry point.

Trains one model (one seed, one configuration) and writes a checkpoint
and a JSON results file to results/<run_name>/.

Usage examples
--------------
# In-domain, MoE, seed 1:
python training/train.py --model_type moe --seed 1

# Cross-country (France → Spain), GatedMoE, seed 3:
python training/train.py --model_type gated_moe --seed 3 --source FR --target ES

# Sensor ablation — S2 only:
python training/train.py --model_type universal --seed 1 --no_s1

# Partial-season (first 7 months):
python training/train.py --model_type moe --seed 1 --n_months 7

# Autoencoder (for reconstruction-based verification):
python training/train.py --model_type autoencoder --seed 1

Run sweeps
----------
Use experiments/classification.py (or the individual experiment scripts) to
launch all seeds/models for a given experiment rather than calling this
script in a shell loop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RESULTS_DIR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEEDS,
)
from training.dataset import (
    build_datasets,
    build_cross_country_datasets,
    build_drought_datasets,
)
from training.lightning_module import CropModelLightning
from training.lightning_module_autoencoder import MoEAutoencoderLightning


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a single model instance.")
    p.add_argument("--model_type", default="moe",
                   choices=["universal", "moe", "gated_moe", "autoencoder"],
                   help="Model architecture.")
    p.add_argument("--seed", type=int, default=1, choices=SEEDS,
                   help="Random seed (also controls train/val/test split).")

    # Sensor ablation
    p.add_argument("--no_s1", action="store_true",
                   help="Disable S1 input (S2-only ablation).")
    p.add_argument("--no_s2", action="store_true",
                   help="Disable S2 input (S1-only ablation).")

    # Cross-country / drought
    p.add_argument("--source", default=None, choices=["FR", "ES"],
                   help="Source country for cross-country experiment.")
    p.add_argument("--target", default=None, choices=["FR", "ES"],
                   help="Target country for cross-country experiment.")
    p.add_argument("--drought", action="store_true",
                   help="Use Spain 2022 → Spain 2023 drought experiment.")

    # Partial season
    p.add_argument("--n_months", type=int, default=None,
                   help="Keep only the first n_months timesteps.")

    # Training knobs
    p.add_argument("--batch_size",   type=int,   default=BATCH_SIZE)
    p.add_argument("--max_epochs",   type=int,   default=MAX_EPOCHS)
    p.add_argument("--patience",     type=int,   default=PATIENCE)
    p.add_argument("--lb_coeff",     type=float, default=0.01,
                   help="Load-balance loss weight (GatedMoE only).")

    # Output
    p.add_argument("--run_name", default=None,
                   help="Override the auto-generated run name.")
    p.add_argument("--gpus", type=int, default=1,
                   help="Number of GPUs (0 = CPU).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Run-name helper
# ---------------------------------------------------------------------------

def make_run_name(args) -> str:
    parts = [args.model_type, f"seed{args.seed}"]
    if args.no_s1:
        parts.append("no_s1")
    if args.no_s2:
        parts.append("no_s2")
    if args.source and args.target:
        parts.append(f"{args.source}to{args.target}")
    if args.drought:
        parts.append("drought")
    if args.n_months is not None:
        parts.append(f"{args.n_months}mo")
    return "_".join(parts)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def make_loaders(train_ds, val_ds, test_ds, batch_size: int):
    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True,
              persistent_workers=True)
    train_dl = DataLoader(train_ds, shuffle=True,  **kw)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kw)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kw)
    return train_dl, val_dl, test_dl


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args     = parse_args(argv)
    run_name = args.run_name or make_run_name(args)
    out_dir  = RESULTS_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)

    # ---- Choose dataset builder ----------------------------------------
    seed_idx = SEEDS.index(args.seed)   # position within the seeds list

    if args.drought:
        splits = build_drought_datasets(seeds=[args.seed],
                                        n_months=args.n_months)
        train_ds, val_ds, test_ds = splits[0]

    elif args.source and args.target:
        splits = build_cross_country_datasets(
            source=args.source, target=args.target,
            seeds=[args.seed], n_months=args.n_months
        )
        train_ds, val_ds, test_ds = splits[0]

    else:
        splits = build_datasets(seeds=[args.seed], n_months=args.n_months)
        train_ds, val_ds, test_ds = splits[0]

    train_dl, val_dl, test_dl = make_loaders(
        train_ds, val_ds, test_ds, args.batch_size
    )

    # ---- Instantiate model ---------------------------------------------
    if args.model_type == "autoencoder":
        module = MoEAutoencoderLightning()
        monitor = "val/recon_loss"
    else:
        module = CropModelLightning(
            model_type   = args.model_type,
            use_s2       = not args.no_s2,
            use_s1       = not args.no_s1,
            patience     = args.patience,
            lb_coeff     = args.lb_coeff,
        )
        monitor = "val/loss"

    # ---- Callbacks -----------------------------------------------------
    ckpt_cb = ModelCheckpoint(
        dirpath    = out_dir / "checkpoints",
        filename   = "best",
        monitor    = monitor,
        mode       = "min",
        save_top_k = 1,
    )
    stop_cb = EarlyStopping(
        monitor  = monitor,
        patience = args.patience,
        mode     = "min",
        verbose  = True,
    )

    # ---- Trainer -------------------------------------------------------
    accelerator = "gpu" if args.gpus > 0 else "cpu"
    devices     = args.gpus if args.gpus > 0 else 1

    trainer = pl.Trainer(
        accelerator  = accelerator,
        devices      = devices,
        max_epochs   = args.max_epochs,
        callbacks    = [ckpt_cb, stop_cb],
        log_every_n_steps = 10,
        enable_progress_bar = True,
    )

    trainer.fit(module, train_dl, val_dl)

    # ---- Test ----------------------------------------------------------
    test_results = trainer.test(module, test_dl, ckpt_path="best")

    # ---- Save results --------------------------------------------------
    meta = dict(
        run_name   = run_name,
        model_type = args.model_type,
        seed       = args.seed,
        no_s1      = args.no_s1,
        no_s2      = args.no_s2,
        source     = args.source,
        target     = args.target,
        drought    = args.drought,
        n_months   = args.n_months,
        test       = test_results[0] if test_results else {},
        best_ckpt  = str(ckpt_cb.best_model_path),
    )
    (out_dir / "results.json").write_text(json.dumps(meta, indent=2))
    print(f"\nResults saved to {out_dir}/results.json")
    return meta


if __name__ == "__main__":
    main()
