"""
experiments/classification.py

§3.1 — In-domain classification results.

Trains all 5 seeds for each of the three architectures (Universal GRU,
MoE, Gated-MoE) on the pooled FR + ES 2022 dataset and reports test-set
macro-F1 and per-class F1.

Output
------
results/classification/
    summary_table.csv        mean ± std F1 per model across seeds
    per_class_f1.csv         per-class F1 for each model × seed
    confusion_matrices.npz   one confusion matrix per model × seed
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, BATCH_SIZE, MAX_EPOCHS, PATIENCE, SEEDS, N_CLASSES, CROP_TO_LABEL
from training.dataset import build_datasets
from training.lightning_module import CropModelLightning
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

EXPERIMENT_DIR = RESULTS_DIR / "classification"
LABEL_TO_CROP  = {v: k for k, v in CROP_TO_LABEL.items()}
MODEL_TYPES    = ["universal", "moe", "gated_moe"]


def train_one(model_type: str, seed: int,
              train_ds, val_ds) -> CropModelLightning:
    """Train a single model, return the best checkpoint as a loaded module."""
    ckpt_dir = EXPERIMENT_DIR / "checkpoints" / model_type / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    module = CropModelLightning(model_type=model_type)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val/loss", mode="min", save_top_k=1,
    )
    stop_cb = EarlyStopping(
        monitor="val/loss", patience=PATIENCE, mode="min"
    )
    trainer = pl.Trainer(
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
    return best


@torch.no_grad()
def evaluate(module: CropModelLightning, test_ds) -> dict:
    """Return all-parcel predictions and labels from the test set."""
    module.eval()
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    all_preds, all_labels = [], []
    for batch in loader:
        x_s2, x_s1, m2, m1, labels, cids, uids = batch
        logits, _ = module._forward(x_s2, x_s1, m2, m1)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return {"preds": preds, "labels": labels}


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    all_splits = build_datasets(seeds=SEEDS)   # one per seed

    records    = []
    cm_store   = {}

    for model_type in MODEL_TYPES:
        print(f"\n{'='*60}\n  {model_type}\n{'='*60}")
        for seed_i, seed in enumerate(SEEDS):
            pl.seed_everything(seed, workers=True)
            train_ds, val_ds, test_ds = all_splits[seed_i]

            print(f"  seed {seed} — training…")
            module = train_one(model_type, seed, train_ds, val_ds)

            res    = evaluate(module, test_ds)
            preds, labels = res["preds"], res["labels"]

            macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
            per_cls  = f1_score(labels, preds, average=None, labels=list(range(N_CLASSES)),
                                zero_division=0)
            cm       = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))

            print(f"  seed {seed} — macro F1: {macro_f1:.4f}")

            row = dict(model=model_type, seed=seed, macro_f1=macro_f1)
            for ci, f1 in enumerate(per_cls):
                row[f"f1_{LABEL_TO_CROP[ci]}"] = f1
            records.append(row)
            cm_store[f"{model_type}_seed{seed}"] = cm

    # ---- Summary table -------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "per_seed_f1.csv", index=False)

    f1_cols = ["macro_f1"] + [f"f1_{LABEL_TO_CROP[i]}" for i in range(N_CLASSES)]
    summary = df.groupby("model")[f1_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(EXPERIMENT_DIR / "summary_table.csv")
    print("\n=== Summary ===")
    print(summary.to_string())

    # ---- Confusion matrices --------------------------------------------
    np.savez_compressed(
        EXPERIMENT_DIR / "confusion_matrices.npz",
        **{k: v for k, v in cm_store.items()},
    )
    print(f"\nResults written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
