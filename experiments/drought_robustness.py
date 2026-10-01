"""
experiments/drought_robustness.py

§3.8 — Drought-year robustness.

Trains models on Spain 2022 and evaluates on Spain 2023 (drought year).
Compares the distribution shift in macro-F1 relative to in-domain 2022
performance and checks whether performance degradation differs across
crop types (drought affects different crops differently).

Requires:
    data/sequences_2023/ES_sequences_2023.npz
    (produced by extraction/assemble_sequences_2023.py)

Output
------
results/drought_robustness/
    per_seed_f1.csv        F1 for every model × seed × eval_year
    summary_table.csv      mean ± std macro-F1 and per-class F1
    year_comparison.pdf    bar chart: 2022 vs 2023 performance per model
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RESULTS_DIR, SEEDS, N_CLASSES, CROP_TO_LABEL, BATCH_SIZE, MAX_EPOCHS, PATIENCE,
)
from training.dataset import build_drought_datasets, build_datasets
from training.lightning_module import CropModelLightning

EXPERIMENT_DIR = RESULTS_DIR / "drought_robustness"
LABEL_TO_CROP  = {v: k for k, v in CROP_TO_LABEL.items()}
MODEL_TYPES    = ["universal", "moe", "gated_moe"]


def train_on_es2022(model_type, seed, train_ds, val_ds):
    ckpt_dir = EXPERIMENT_DIR / "checkpoints" / model_type / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    module  = CropModelLightning(model_type=model_type)
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val/loss", mode="min", save_top_k=1,
    )
    stop_cb = EarlyStopping(monitor="val/loss", patience=PATIENCE, mode="min")
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
def evaluate(module, test_ds):
    module.eval()
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)
    all_p, all_l = [], []
    for batch in loader:
        x_s2, x_s1, m2, m1, labels, cids, uids = batch
        logits, _ = module._forward(x_s2, x_s1, m2, m1)
        all_p.append(logits.argmax(-1).cpu().numpy())
        all_l.append(labels.cpu().numpy())
    preds  = np.concatenate(all_p)
    labels = np.concatenate(all_l)
    macro  = f1_score(labels, preds, average="macro", zero_division=0)
    per_cls = f1_score(labels, preds, average=None,
                       labels=list(range(N_CLASSES)), zero_division=0)
    return macro, per_cls


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # Drought splits: train/val=ES2022, test=ES2023
    drought_splits = build_drought_datasets(seeds=SEEDS)

    # In-domain ES-only 2022 splits (for baseline comparison)
    indomain_splits = build_datasets(countries=["ES"], seeds=SEEDS)

    records = []

    for model_type in MODEL_TYPES:
        print(f"\n=== {model_type} ===")
        for si, seed in enumerate(SEEDS):
            pl.seed_everything(seed, workers=True)
            train_ds, val_ds, test_ds_2023 = drought_splits[si]
            _, _, test_ds_2022 = indomain_splits[si]

            print(f"  seed {seed}: training on ES 2022…")
            module = train_on_es2022(model_type, seed, train_ds, val_ds)

            # Eval on 2022 test (in-domain)
            f1_22, per_22 = evaluate(module, test_ds_2022)
            # Eval on 2023 test (drought year)
            f1_23, per_23 = evaluate(module, test_ds_2023)

            print(f"  seed {seed}: 2022={f1_22:.4f}, 2023={f1_23:.4f}")

            for year, macro, per_cls in ((2022, f1_22, per_22), (2023, f1_23, per_23)):
                row = dict(model=model_type, seed=seed, year=year, macro_f1=macro)
                for ci, f1 in enumerate(per_cls):
                    row[f"f1_{LABEL_TO_CROP[ci]}"] = f1
                records.append(row)

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "per_seed_f1.csv", index=False)

    f1_cols = ["macro_f1"] + [f"f1_{LABEL_TO_CROP[i]}" for i in range(N_CLASSES)]
    summary = df.groupby(["model", "year"])[f1_cols].agg(["mean", "std"]).round(4)
    summary.to_csv(EXPERIMENT_DIR / "summary_table.csv")
    print("\n=== Drought robustness summary ===")
    print(summary.to_string())

    # ---- Bar chart: 2022 vs 2023 ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    x  = np.arange(len(MODEL_TYPES))
    bw = 0.35
    for i, (year, offset) in enumerate(((2022, -bw/2), (2023, bw/2))):
        means = [df[(df.model == m) & (df.year == year)]["macro_f1"].mean()
                 for m in MODEL_TYPES]
        stds  = [df[(df.model == m) & (df.year == year)]["macro_f1"].std()
                 for m in MODEL_TYPES]
        ax.bar(x + offset, means, bw, label=str(year),
               yerr=stds, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels(MODEL_TYPES)
    ax.set_ylabel("Macro F1"); ax.set_ylim(0, 1)
    ax.set_title("Drought robustness: Spain 2022 vs 2023")
    ax.legend()
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(EXPERIMENT_DIR / f"year_comparison.{ext}", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
