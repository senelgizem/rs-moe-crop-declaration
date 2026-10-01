"""
experiments/sensor_ablation.py

§3.2 — Sensor ablation (Table 4).

Trains each architecture with three sensor configurations:
    - Both S1 and S2  (full model)
    - S2 only         (no_s1=True)
    - S1 only         (no_s2=True)

and reports macro-F1 drop relative to the full-sensor baseline.

Output
------
results/sensor_ablation/
    per_seed_f1.csv        F1 for every model × config × seed
    summary_table.csv      mean ± std, with delta vs full
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
from training.dataset import build_datasets
from training.lightning_module import CropModelLightning

EXPERIMENT_DIR = RESULTS_DIR / "sensor_ablation"
LABEL_TO_CROP  = {v: k for k, v in CROP_TO_LABEL.items()}
MODEL_TYPES    = ["universal", "moe", "gated_moe"]
SENSOR_CONFIGS = [
    dict(use_s1=True,  use_s2=True,  tag="both"),
    dict(use_s1=False, use_s2=True,  tag="s2_only"),
    dict(use_s1=True,  use_s2=False, tag="s1_only"),
]


def train_and_eval(model_type, sensor_cfg, seed, train_ds, val_ds, test_ds):
    tag     = sensor_cfg["tag"]
    ckpt_dir = EXPERIMENT_DIR / "checkpoints" / model_type / tag / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    module = CropModelLightning(
        model_type=model_type,
        use_s1=sensor_cfg["use_s1"],
        use_s2=sensor_cfg["use_s2"],
    )
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
    best.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4):
            x_s2, x_s1, m2, m1, labels, cids, uids = batch
            logits, _ = best._forward(x_s2, x_s1, m2, m1)
            all_preds.append(logits.argmax(-1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    import numpy as np
    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return f1_score(labels, preds, average="macro", zero_division=0)


def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    all_splits = build_datasets(seeds=SEEDS)
    records    = []

    for model_type in MODEL_TYPES:
        for cfg in SENSOR_CONFIGS:
            tag = cfg["tag"]
            print(f"\n{model_type} / {tag}")
            for si, seed in enumerate(SEEDS):
                pl.seed_everything(seed, workers=True)
                train_ds, val_ds, test_ds = all_splits[si]
                f1 = train_and_eval(model_type, cfg, seed, train_ds, val_ds, test_ds)
                print(f"  seed {seed}: {f1:.4f}")
                records.append(dict(model=model_type, config=tag, seed=seed, macro_f1=f1))

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "per_seed_f1.csv", index=False)

    summary = df.groupby(["model", "config"])["macro_f1"].agg(["mean", "std"]).round(4)
    # Add delta vs "both" baseline
    both = summary.xs("both", level="config")["mean"].rename("baseline")
    summary = summary.join(both, on="model")
    summary["delta"] = (summary["mean"] - summary["baseline"]).round(4)
    summary = summary.drop(columns="baseline")
    summary.to_csv(EXPERIMENT_DIR / "summary_table.csv")
    print("\n=== Sensor ablation summary ===")
    print(summary.to_string())
    print(f"\nResults written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
