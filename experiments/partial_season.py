"""
experiments/partial_season.py

§3.7 — Early-season classification.

Evaluates how classification accuracy degrades when only the first
n_months of the 13-month season are available at inference time.

Cut-offs tested: 5, 7, 9 months and the full 13-month sequence
(as defined in config.PARTIAL_CUTOFFS).

The SAME model weights are used for all cut-offs — no retraining.
The model was trained on full sequences; at inference, the sequence
is truncated to n_months timesteps.

Requires pre-trained checkpoints from classification.py.

Output
------
results/partial_season/
    per_seed_f1.csv       F1 for every model × n_months × seed
    summary_table.csv     mean ± std macro-F1 by model × n_months
    accuracy_curve.pdf    macro-F1 vs n_months line chart
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RESULTS_DIR, SEEDS, N_CLASSES, CROP_TO_LABEL, PARTIAL_CUTOFFS, BATCH_SIZE,
)
from training.dataset import build_datasets
from training.lightning_module import CropModelLightning

EXPERIMENT_DIR = RESULTS_DIR / "partial_season"
LABEL_TO_CROP  = {v: k for k, v in CROP_TO_LABEL.items()}
MODEL_TYPES    = ["universal", "moe", "gated_moe"]


@torch.no_grad()
def eval_truncated(
    module: CropModelLightning,
    test_ds_full,
    n_months: int | None,
) -> float:
    """
    Evaluate the model on a truncated test set.

    n_months=None uses the full sequence.
    The test_ds_full must have been built with n_months=None so that
    we can truncate on the fly here.
    """
    module.eval()
    loader = DataLoader(test_ds_full, batch_size=512, shuffle=False, num_workers=4)
    all_preds, all_labels = [], []
    for batch in loader:
        x_s2, x_s1, m2, m1, labels, cids, uids = batch
        if n_months is not None:
            x_s2 = x_s2[:, :n_months, :]
            x_s1 = x_s1[:, :n_months, :]
            m2   = m2[:,   :n_months]
            m1   = m1[:,   :n_months]
        logits, _ = module._forward(x_s2, x_s1, m2, m1)
        all_preds.append(logits.argmax(-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_base  = RESULTS_DIR / "classification" / "checkpoints"
    all_splits = build_datasets(seeds=SEEDS, n_months=None)   # full sequences

    records = []

    for model_type in MODEL_TYPES:
        print(f"\n=== {model_type} ===")
        for si, seed in enumerate(SEEDS):
            ckpt_path = ckpt_base / model_type / f"seed{seed}" / "best.ckpt"
            if not ckpt_path.exists():
                print(f"  seed {seed}: checkpoint not found, skipping.")
                continue

            module = CropModelLightning.load_from_checkpoint(str(ckpt_path))
            _, _, test_ds = all_splits[si]

            for n_months in PARTIAL_CUTOFFS:
                f1 = eval_truncated(module, test_ds, n_months)
                label = "full" if n_months is None else str(n_months)
                print(f"  seed {seed}, {label} months: {f1:.4f}")
                records.append(dict(
                    model=model_type, seed=seed,
                    n_months=n_months if n_months is not None else 13,
                    macro_f1=f1,
                ))

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "per_seed_f1.csv", index=False)

    summary = df.groupby(["model", "n_months"])["macro_f1"].agg(
        mean_f1="mean", std_f1="std"
    ).round(4)
    summary.to_csv(EXPERIMENT_DIR / "summary_table.csv")
    print("\n=== Partial-season summary ===")
    print(summary.to_string())

    # ---- Accuracy curve ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ci, model_type in enumerate(MODEL_TYPES):
        sub = df[df["model"] == model_type].groupby("n_months")["macro_f1"]
        mu, std = sub.mean(), sub.std()
        xs = list(mu.index)
        ax.plot(xs, mu.values, marker="o", label=model_type, color=colors[ci])
        ax.fill_between(xs, mu.values - std.values, mu.values + std.values,
                        alpha=0.2, color=colors[ci])
    ax.set_xlabel("Months of season available")
    ax.set_ylabel("Macro F1")
    ax.set_title("Early-season classification accuracy")
    ax.legend()
    ax.set_xticks(sorted(df["n_months"].unique()))
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(EXPERIMENT_DIR / f"accuracy_curve.{ext}", dpi=150)
    plt.close(fig)

    print(f"\nAll outputs written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
