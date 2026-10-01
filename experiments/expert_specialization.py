"""
experiments/expert_specialization.py

§3.4 — Expert specialisation analysis.

Loads trained MoE checkpoints and analyses which expert each parcel is
routed to, compared with its true crop label.  Also plots the mean ±
std time series of NDVI and VH for each crop class, averaged over the
test-set parcels that are correctly classified.

Figures produced
----------------
results/expert_specialization/
    routing_heatmap.pdf      expert × crop routing matrix (fraction)
    routing_heatmap.png
    ts_NDVI_wheat.pdf        mean time series per crop (one file per feature)
    ts_VH_wheat.pdf
    ...
    specialization_index.csv  per-expert dominant crop and purity score
    routing_confusion.csv     expert × crop count matrix
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
import pytorch_lightning as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    RESULTS_DIR, SEEDS, N_CLASSES, CROP_TO_LABEL, FEATURE_ORDER,
    S2_FEATURE_IDX, S1_FEATURE_IDX, BATCH_SIZE,
)
from training.dataset import build_datasets
from training.lightning_module import CropModelLightning

EXPERIMENT_DIR = RESULTS_DIR / "expert_specialization"
LABEL_TO_CROP  = {v: k for k, v in CROP_TO_LABEL.items()}

# TS plot: which feature indices and names to plot
PLOT_FEATURES = {
    "NDVI": FEATURE_ORDER.index("NDVI"),
    "VH":   FEATURE_ORDER.index("VH"),
}


# ---------------------------------------------------------------------------
# Routing extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_routing_and_series(module: CropModelLightning, test_ds):
    """
    Run the MoE model on the test set and return:
        expert_assigned : (N,) int — argmax of logits (predicted expert)
        true_labels     : (N,) int
        X_norm          : (N, T, F) float — normalised input sequences
        correct_mask    : (N,) bool — correctly classified parcels
    """
    module.eval()
    loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=4)

    all_exp, all_lbl, all_X = [], [], []
    for batch in loader:
        x_s2, x_s1, m2, m1, labels, cids, uids = batch
        logits, _ = module._forward(x_s2, x_s1, m2, m1)
        assigned = logits.argmax(-1).cpu().numpy()
        all_exp.append(assigned)
        all_lbl.append(labels.cpu().numpy())

        # Reconstruct full (T, F) tensor from split inputs
        T = x_s2.shape[1]
        F = len(FEATURE_ORDER)
        X_full = torch.zeros(x_s2.shape[0], T, F)
        X_full[:, :, list(S2_FEATURE_IDX)] = x_s2.cpu()
        X_full[:, :, list(S1_FEATURE_IDX)] = x_s1.cpu()
        all_X.append(X_full.numpy())

    expert_assigned = np.concatenate(all_exp)
    true_labels     = np.concatenate(all_lbl)
    X_norm          = np.concatenate(all_X, axis=0)
    correct_mask    = (expert_assigned == true_labels)
    return expert_assigned, true_labels, X_norm, correct_mask


# ---------------------------------------------------------------------------
# Routing heatmap
# ---------------------------------------------------------------------------

def plot_routing_heatmap(routing_matrix: np.ndarray, out_dir: Path):
    """
    routing_matrix shape: (N_CLASSES, N_CLASSES) — rows=true crop, cols=expert
    Normalised row-wise to show fraction routed to each expert.
    """
    frac = routing_matrix / routing_matrix.sum(axis=1, keepdims=True)
    crop_names = [LABEL_TO_CROP[i] for i in range(N_CLASSES)]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frac, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels([f"E{i}" for i in range(N_CLASSES)])
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(crop_names)
    ax.set_xlabel("Expert assigned"); ax.set_ylabel("True crop class")
    ax.set_title("MoE expert routing (fraction per crop)")
    for r in range(N_CLASSES):
        for c in range(N_CLASSES):
            ax.text(c, r, f"{frac[r, c]:.2f}", ha="center", va="center",
                    color="white" if frac[r, c] > 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Fraction")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"routing_heatmap.{ext}", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------

def plot_mean_timeseries(X_norm, true_labels, correct_mask, timesteps, out_dir):
    """
    For each (feature, crop) pair, plot mean ± std over correctly classified
    test parcels.  Timesteps are YYYYMMDD strings; labels are YYYY-MM.
    """
    tick_labels = [f"{t[:4]}-{t[4:6]}" for t in timesteps]

    for feat_name, feat_idx in PLOT_FEATURES.items():
        fig, ax = plt.subplots(figsize=(9, 4))
        for crop_lbl, crop_name in LABEL_TO_CROP.items():
            sel = (true_labels == crop_lbl) & correct_mask
            if sel.sum() < 5:
                continue
            series = X_norm[sel, :, feat_idx]           # (n_sel, T)
            mu  = np.nanmean(series, axis=0)
            std = np.nanstd(series,  axis=0)
            ax.plot(mu, label=crop_name)
            ax.fill_between(range(len(mu)), mu - std, mu + std, alpha=0.2)

        ax.set_xticks(range(len(tick_labels)))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{feat_name} (z-scored)")
        ax.set_title(f"Mean ± std {feat_name} — correctly classified test parcels")
        ax.legend(fontsize=8)
        plt.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"ts_{feat_name}.{ext}", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Specialisation index
# ---------------------------------------------------------------------------

def compute_specialization_index(routing_matrix: np.ndarray) -> pd.DataFrame:
    """
    For each expert, compute:
        dominant_crop : crop class with the most parcels routed here
        purity        : fraction of routed parcels that are the dominant crop
    routing_matrix rows=true_crop, cols=expert
    """
    rows = []
    for exp_i in range(N_CLASSES):
        col      = routing_matrix[:, exp_i]
        dom_lbl  = int(col.argmax())
        purity   = col[dom_lbl] / col.sum() if col.sum() > 0 else 0.0
        rows.append(dict(
            expert       = f"E{exp_i}",
            dominant_crop= LABEL_TO_CROP[dom_lbl],
            n_routed     = int(col.sum()),
            purity       = round(purity, 4),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load existing MoE checkpoints (trained by classification.py) ----
    ckpt_base = RESULTS_DIR / "classification" / "checkpoints" / "moe"
    if not ckpt_base.exists():
        print(
            "No MoE checkpoints found in results/classification/checkpoints/moe/.\n"
            "Run experiments/classification.py first."
        )
        return

    all_splits = build_datasets(seeds=SEEDS)

    # Accumulate routing matrices across seeds
    routing_agg  = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
    first_timesteps = None

    for si, seed in enumerate(SEEDS):
        ckpt_path = ckpt_base / f"seed{seed}" / "best.ckpt"
        if not ckpt_path.exists():
            print(f"  seed {seed}: checkpoint not found, skipping.")
            continue

        print(f"  seed {seed}: loading checkpoint…")
        module = CropModelLightning.load_from_checkpoint(str(ckpt_path))
        _, _, test_ds = all_splits[si]

        exp_assigned, true_labels, X_norm, correct_mask = (
            get_routing_and_series(module, test_ds)
        )

        # Build routing matrix (rows=true, cols=assigned)
        mat = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        for tl, ea in zip(true_labels, exp_assigned):
            mat[tl, ea] += 1
        routing_agg += mat

        # Store first seed's timestep labels for plotting
        if first_timesteps is None:
            import json
            seq_path = None
            for country in ("FR", "ES"):
                from config import SEQ_DIR
                candidate = SEQ_DIR / f"{country}_sequences_meta.json"
                if candidate.exists():
                    first_timesteps = json.loads(candidate.read_text())["timesteps"]
                    break

    # ---- Routing heatmap -----------------------------------------------
    routing_df = pd.DataFrame(
        routing_agg.astype(int),
        index   = [LABEL_TO_CROP[i] for i in range(N_CLASSES)],
        columns = [f"E{i}" for i in range(N_CLASSES)],
    )
    routing_df.to_csv(EXPERIMENT_DIR / "routing_confusion.csv")
    plot_routing_heatmap(routing_agg, EXPERIMENT_DIR)

    # ---- Specialisation index ------------------------------------------
    spec_df = compute_specialization_index(routing_agg)
    spec_df.to_csv(EXPERIMENT_DIR / "specialization_index.csv", index=False)
    print("\n=== Specialisation index ===")
    print(spec_df.to_string(index=False))

    # ---- Time-series plots (last seed's X_norm used for illustration) ---
    if first_timesteps is not None:
        # Re-run last seed for plot data
        last_seed = SEEDS[-1]
        ckpt_path = ckpt_base / f"seed{last_seed}" / "best.ckpt"
        if ckpt_path.exists():
            module = CropModelLightning.load_from_checkpoint(str(ckpt_path))
            _, _, test_ds = all_splits[-1]
            _, true_labels, X_norm, correct_mask = get_routing_and_series(module, test_ds)
            plot_mean_timeseries(X_norm, true_labels, correct_mask,
                                 first_timesteps, EXPERIMENT_DIR)
            print(f"Time-series plots saved to {EXPERIMENT_DIR}/")

    print(f"\nAll outputs written to {EXPERIMENT_DIR}/")


if __name__ == "__main__":
    main()
