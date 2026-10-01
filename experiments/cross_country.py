"""
experiments/cross_country.py

§3.3 — Cross-country generalisation.

Two transfer directions:
    FR → ES : train on France, test on Spain
    ES → FR : train on Spain, test on France

For each direction and each architecture (universal, moe, gated_moe),
trains 5 seeds and reports macro-F1 on the held-out target country.

Also computes per-class F1 to check whether generalisation is
crop-type-specific.

Output
------
results/cross_country/
    per_seed_f1.csv         F1 for every direction × model × seed
    summary_table.csv       mean ± std macro-F1 and per-class F1
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEEDS, N_CLASSES, CROP_TO_LABEL
from training.dataset import build_cross_country_datasets
from training.lightning_module import CropModelLightning

EXPERIMENT_DIR  = RESULTS_DIR / "cross_country"
LABEL_TO_CROP   = {v: k for k, v in CROP_TO_LABEL.items()}
MODEL_TYPES     = ["universal", "moe", "gated_moe"]
DIRECTIONS      = [("FR", "ES"), ("ES", "FR")]


def train_and_eval(model_type, source, target, seed, train_ds, val_ds, test_ds):
    ckpt_dir = EXPERIMENT_DIR / "checkpoints" / model_type / f"{source}to{target}" / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    module   = CropModelLightning(model_type=model_type)
    ckpt_cb  = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val/loss", mode="min", save_top_k=1,
    )
    stop_cb  = EarlyStopping(monitor="val/loss", patience=PATIENCE, mode="min")
    trainer  = pl.Trainer(
        accelerator="auto", devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[ckpt_cb, stop_cb],
        enable_progress_bar=False,
        logger=False,
    )
    kw = dict(batch_size=BATCH_SIZE, num_workers=4, persistent_workers=True)
    trainer.fit(
        module,
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
    )
    best = CropModelLightning.load_from_checkpoint(ckpt_cb.best_model_path)
    best.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4):
            x_s2, x_s1, m2, m1, labels, cids, uids = batch
            logits, _ = best._forward(x_s2, x_s1, m2, m1)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    macro  = f1_score(labels, preds, average="macro", zero_division=0)
    per_cls = f1_score(labels, preds, average=None,
                       labels=list(range(N_CLASSES)), zero_division=0)
    return macro, per_cls


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for source, target in DIRECTIONS:
        print(f"\n{'='*60}\n  {source} → {target}\n{'='*60}")
        all_splits = build_cross_country_datasets(
            source=source, target=target, seeds=SEEDS
        )

        for model_type in MODEL_TYPES:
            print(f"\n  [{model_type}]")
            for si, seed in enumerate(SEEDS):
                pl.seed_everything(seed, workers=True)
                train_ds, val_ds, test_ds = all_splits[si]
                macro, per_cls = train_and_eval(
                    model_type, source, target, seed, train_ds, val_ds, test_ds
                )
                print(f"    seed {seed}: {macro:.4f}")
                row = dict(direction=f"{source}→{target}", model=model_type,
                           seed=seed, macro_f1=macro)
                for ci, f1 in enumerate(per_cls):
                    row[f"f1_{LABEL_TO_CROP[ci]}"] = f1
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "per_seed_f1.csv", index=False)

    f1_cols = ["macro_f1"] + [f"f1_{LABEL_TO_CROP[i]}" for i in range(N_CLASSES)]
    summary = df.groupby(["direction", "model"])[f1_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(EXPERIMENT_DIR / "summary_table.csv")
    print("\n=== Cross-country summary ===")
    print(summary.to_string())
    print(f"\nResults written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
