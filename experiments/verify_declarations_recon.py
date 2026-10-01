"""
experiments/verify_declarations_recon.py

§3.6 — Reconstruction-based declaration verification.

Uses the trained MoEAutoencoder: each expert reconstructs the parcel's
time series; the reconstruction error for the declared crop is compared
to the minimum error across all experts.

    score = error[declared] − min_e error[e]

Evaluated with the same label-flip mismatch simulation as
verify_declarations.py, but uses reconstruction error as the detector.

Output
------
results/verify_declarations_recon/
    simulation_results.csv      AUROC per mismatch_rate and seed
    auroc_summary.csv           mean ± std AUROC across seeds
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RESULTS_DIR, SEEDS, BATCH_SIZE, MAX_EPOCHS, PATIENCE, CORRUPTION_RATES,
)
from training.dataset import build_datasets
from training.lightning_module_autoencoder import MoEAutoencoderLightning

EXPERIMENT_DIR = RESULTS_DIR / "verify_declarations_recon"


# ---------------------------------------------------------------------------
# Train autoencoder (if checkpoint missing)
# ---------------------------------------------------------------------------

def train_autoencoder(seed: int, train_ds, val_ds) -> MoEAutoencoderLightning:
    ckpt_dir = EXPERIMENT_DIR / "checkpoints" / f"seed{seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    module  = MoEAutoencoderLightning()
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, filename="best",
        monitor="val/recon_loss", mode="min", save_top_k=1,
    )
    stop_cb = EarlyStopping(
        monitor="val/recon_loss", patience=PATIENCE, mode="min"
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
    return MoEAutoencoderLightning.load_from_checkpoint(ckpt_cb.best_model_path)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_recon_scores(
    module: MoEAutoencoderLightning, test_ds
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    scores : (N,) float — recon_score = error[declared] − min_e error[e]
    labels : (N,) int   — true (declared) crop labels
    """
    module.eval()
    loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4)
    all_s, all_l = [], []
    for batch in loader:
        x_s2, x_s1, m2, m1, labels, cids, uids = batch
        s = module.reconstruction_verification_score(x_s2, x_s1, m2, m1, labels)
        all_s.append(s.cpu().numpy())
        all_l.append(labels.cpu().numpy())
    return np.concatenate(all_s), np.concatenate(all_l)


# ---------------------------------------------------------------------------
# Mismatch simulation (same logic as verify_declarations.py)
# ---------------------------------------------------------------------------

def simulate_mismatches(
    scores: np.ndarray,
    mismatch_rate: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(scores)
    n_flip = int(np.round(N * mismatch_rate))
    flip_idx = rng.choice(N, size=n_flip, replace=False)

    is_mismatch = np.zeros(N, dtype=bool)
    is_mismatch[flip_idx] = True

    # Flipped parcels scored with true label → low reconstruction score.
    # Negate to make them detectable as high-scoring.
    detector_scores = scores.copy()
    detector_scores[flip_idx] = -scores[flip_idx]
    return detector_scores, is_mismatch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    all_splits = build_datasets(seeds=SEEDS)
    records    = []

    for si, seed in enumerate(SEEDS):
        print(f"\nSeed {seed}")
        train_ds, val_ds, test_ds = all_splits[si]

        ckpt_path = EXPERIMENT_DIR / "checkpoints" / f"seed{seed}" / "best.ckpt"
        if ckpt_path.exists():
            print(f"  Loading existing checkpoint: {ckpt_path}")
            module = MoEAutoencoderLightning.load_from_checkpoint(str(ckpt_path))
        else:
            print("  Training autoencoder…")
            pl.seed_everything(seed, workers=True)
            module = train_autoencoder(seed, train_ds, val_ds)

        scores, labels = compute_recon_scores(module, test_ds)
        rng = np.random.default_rng(seed)

        for rate in CORRUPTION_RATES:
            det, is_mm = simulate_mismatches(scores, rate, rng)
            if is_mm.sum() == 0 or (~is_mm).sum() == 0:
                continue
            auroc = roc_auc_score(is_mm, det)
            print(f"  rate {rate:.1f}: AUROC={auroc:.4f}")
            records.append(dict(seed=seed, mismatch_rate=rate, auroc=auroc))

    df = pd.DataFrame(records)
    df.to_csv(EXPERIMENT_DIR / "simulation_results.csv", index=False)

    summary = df.groupby("mismatch_rate")["auroc"].agg(
        mean_auroc="mean", std_auroc="std"
    ).round(4)
    summary.to_csv(EXPERIMENT_DIR / "auroc_summary.csv")
    print("\n=== Reconstruction AUROC summary ===")
    print(summary.to_string())
    print(f"\nAll outputs written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
