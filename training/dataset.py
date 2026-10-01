"""
training/dataset.py

PyTorch Dataset and data-loading utilities for the parcel sequence files
produced by extraction/assemble_sequences.py.

Public API
----------
build_datasets(countries, seeds, n_months)
    → list of (train_ds, val_ds, test_ds) per seed

build_cross_country_datasets(source, target, seeds, n_months)
    → list of (train_ds, val_ds, test_ds) per seed, norm from source train

ParcelSequenceDataset
    __getitem__ returns (x_s2, x_s1, mask_s2, mask_s1, label, country_idx, unique_id)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    SEQ_DIR, SEQ_2023_DIR,
    CROP_TO_LABEL, N_CLASSES,
    FEATURE_ORDER, S2_FEATURE_IDX, S1_FEATURE_IDX,
    TRAIN_FRAC, VAL_FRAC, SEEDS,
)

COUNTRY_TO_IDX = {"FR": 0, "ES": 1}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sequences(country: str, year: int = 2022) -> dict:
    """
    Load the .npz for a given country and year.

    Returns a dict with keys: X, field_id, crop_class, area_ha, unique_id,
    timesteps, feature_names.
    """
    if year == 2022:
        path = SEQ_DIR / f"{country}_sequences.npz"
    elif year == 2023:
        path = SEQ_2023_DIR / f"{country}_sequences_2023.npz"
    else:
        raise ValueError(f"Unsupported year: {year}")

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run extraction/assemble_sequences.py first."
        )
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Normalisation (train-set statistics, applied to all splits)
# ---------------------------------------------------------------------------

def compute_train_normalization_stats(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-feature z-score statistics, pooled across ALL training parcels AND
    all timesteps.

    X_train shape: (N, T, F)
    Returns:
        mean  (F,)  float32
        std   (F,)  float32  — clamped to ≥ 1e-6 to avoid division by zero
    """
    mean = np.nanmean(X_train, axis=(0, 1)).astype(np.float32)   # (F,)
    std  = np.nanstd(X_train,  axis=(0, 1)).astype(np.float32)   # (F,)
    std  = np.maximum(std, 1e-6)
    return mean, std


def apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise X in place (NaN positions remain NaN)."""
    return (X - mean[None, None, :]) / std[None, None, :]


# ---------------------------------------------------------------------------
# Stratified splits
# ---------------------------------------------------------------------------

def make_splits(
    indices: np.ndarray,
    strata: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train / val / test split.

    Strata are formed by crop_class × country joint labels so that
    class balance and country balance are preserved in all three splits.

    Parameters
    ----------
    indices   : 1-D array of integer indices into the full dataset
    strata    : 1-D array of string labels, same length as indices
    train_frac, val_frac : fractions (test_frac = 1 - train - val)
    seed      : random seed for reproducibility

    Returns
    -------
    train_idx, val_idx, test_idx  (1-D int arrays)
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for stratum in np.unique(strata):
        mask = strata == stratum
        idx  = indices[mask]
        idx  = rng.permutation(idx)
        n    = len(idx)
        n_tr = max(1, int(np.floor(n * train_frac)))
        n_va = max(1, int(np.floor(n * val_frac)))
        # Make sure test is non-empty
        if n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        train_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_va])
        test_idx.append(idx[n_tr + n_va:])

    return (np.concatenate(train_idx),
            np.concatenate(val_idx),
            np.concatenate(test_idx))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ParcelSequenceDataset(Dataset):
    """
    Parcel-level time-series dataset.

    Each item is a tuple:
        x_s2       (T, 10)  float32  — S2 features, z-scored, NaN→0
        x_s1       (T,  4)  float32  — S1 features, z-scored, NaN→0
        mask_s2    (T,)     bool     — True where original value was valid
        mask_s1    (T,)     bool     — True where original value was valid
        label      ()       int64    — crop class index (CROP_TO_LABEL)
        country    ()       int64    — 0=FR, 1=ES
        unique_id  str              — parcel identifier

    Parameters
    ----------
    X          : (N, T, F) float32 — already z-scored; NaN where no obs
    labels     : (N,) int
    country_ids: (N,) int
    unique_ids : (N,) str
    n_months   : keep only the first n_months timesteps (None = all)
    """

    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        country_ids: np.ndarray,
        unique_ids: np.ndarray,
        n_months: Optional[int] = None,
    ):
        T = X.shape[1] if n_months is None else n_months
        self.X           = X[:, :T, :]
        self.labels      = labels
        self.country_ids = country_ids
        self.unique_ids  = unique_ids
        self.s2_idx      = list(S2_FEATURE_IDX)
        self.s1_idx      = list(S1_FEATURE_IDX)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x    = self.X[idx]                         # (T, F)
        x_s2 = x[:, self.s2_idx]                  # (T, 10)
        x_s1 = x[:, self.s1_idx]                  # (T,  4)

        # Validity mask: a timestep is valid if ANY S2 feature is non-NaN
        mask_s2 = ~np.isnan(x_s2).all(axis=-1)    # (T,)
        mask_s1 = ~np.isnan(x_s1).all(axis=-1)    # (T,)

        # Replace NaN with 0 AFTER computing mask (model ignores masked pos)
        x_s2 = np.where(np.isnan(x_s2), 0.0, x_s2).astype(np.float32)
        x_s1 = np.where(np.isnan(x_s1), 0.0, x_s1).astype(np.float32)

        return (
            torch.from_numpy(x_s2),
            torch.from_numpy(x_s1),
            torch.from_numpy(mask_s2),
            torch.from_numpy(mask_s1),
            torch.tensor(self.labels[idx],       dtype=torch.long),
            torch.tensor(self.country_ids[idx],  dtype=torch.long),
            self.unique_ids[idx],
        )


# ---------------------------------------------------------------------------
# Helpers: label / country arrays from raw .npz dicts
# ---------------------------------------------------------------------------

def _make_labels(crop_class: np.ndarray) -> np.ndarray:
    labels = np.array([CROP_TO_LABEL[c] for c in crop_class], dtype=np.int64)
    return labels


def _make_country_ids(country_col: np.ndarray | None, country_str: str,
                      n: int) -> np.ndarray:
    if country_col is not None and country_col.ndim > 0:
        return np.array([COUNTRY_TO_IDX.get(c, -1) for c in country_col],
                        dtype=np.int64)
    return np.full(n, COUNTRY_TO_IDX[country_str], dtype=np.int64)


# ---------------------------------------------------------------------------
# build_datasets: in-domain, both countries pooled
# ---------------------------------------------------------------------------

def build_combined_dataset(countries: Sequence[str] = ("FR", "ES")) -> dict:
    """
    Load and concatenate sequence files for the requested countries.

    Validates that feature_names and timesteps match across files.
    Returns a dict with keys: X, labels, country_ids, unique_ids, strata.
    """
    all_X, all_labels, all_cids, all_uids = [], [], [], []
    ref_features = ref_timesteps = None

    for country in countries:
        d  = load_sequences(country)
        fn = list(d["feature_names"])
        ts = list(d["timesteps"])

        if ref_features is None:
            ref_features, ref_timesteps = fn, ts
        else:
            if fn != ref_features:
                raise ValueError(
                    f"feature_names mismatch between {countries[0]} and {country}"
                )
            if ts != ref_timesteps:
                raise ValueError(
                    f"timesteps mismatch between {countries[0]} and {country}"
                )

        n = len(d["X"])
        labels     = _make_labels(d["crop_class"])
        country_col = d.get("country", None)
        cids       = _make_country_ids(country_col, country, n)

        all_X.append(d["X"])
        all_labels.append(labels)
        all_cids.append(cids)
        all_uids.append(d["unique_id"])

    X          = np.concatenate(all_X,    axis=0)
    labels     = np.concatenate(all_labels)
    country_ids = np.concatenate(all_cids)
    unique_ids  = np.concatenate(all_uids)

    # Stratification key: crop × country
    crop_names = [list(CROP_TO_LABEL.keys())[l] for l in labels]
    cname_arr  = np.array(list(COUNTRY_TO_IDX.keys()))[country_ids]
    strata     = np.array([f"{c}_{cn}" for c, cn in zip(crop_names, cname_arr)])

    return dict(X=X, labels=labels, country_ids=country_ids,
                unique_ids=unique_ids, strata=strata)


def build_datasets(
    countries: Sequence[str] = ("FR", "ES"),
    seeds: Sequence[int] = SEEDS,
    n_months: Optional[int] = None,
) -> list[tuple[ParcelSequenceDataset, ParcelSequenceDataset, ParcelSequenceDataset]]:
    """
    Build one (train, val, test) triple per seed.

    Normalization statistics are computed from the train split of each seed
    independently, then applied to val and test.

    Returns
    -------
    list of (train_ds, val_ds, test_ds) — one per seed
    """
    pool = build_combined_dataset(countries)
    X, labels, cids, uids, strata = (
        pool["X"], pool["labels"], pool["country_ids"],
        pool["unique_ids"], pool["strata"],
    )
    indices = np.arange(len(labels))
    results = []

    for seed in seeds:
        tr_idx, va_idx, te_idx = make_splits(
            indices, strata, TRAIN_FRAC, VAL_FRAC, seed
        )

        mean, std = compute_train_normalization_stats(X[tr_idx])
        X_norm    = apply_normalization(X, mean, std)

        make = lambda idx: ParcelSequenceDataset(
            X_norm[idx], labels[idx], cids[idx], uids[idx], n_months
        )
        results.append((make(tr_idx), make(va_idx), make(te_idx)))

    return results


# ---------------------------------------------------------------------------
# build_cross_country_datasets: source → target transfer
# ---------------------------------------------------------------------------

def build_cross_country_datasets(
    source: str,
    target: str,
    seeds: Sequence[int] = SEEDS,
    n_months: Optional[int] = None,
) -> list[tuple[ParcelSequenceDataset, ParcelSequenceDataset, ParcelSequenceDataset]]:
    """
    Cross-country evaluation.

    Train and val come from `source`; the entire `target` dataset is used
    as the test set.  Normalization statistics are derived from the source
    train split only.

    Parameters
    ----------
    source, target : "FR" or "ES"
    seeds          : list of random seeds for reproducible train/val splits
    n_months       : truncate sequences to first n_months timesteps

    Returns
    -------
    list of (train_ds, val_ds, test_ds) — one per seed
    """
    src = build_combined_dataset([source])
    tgt = build_combined_dataset([target])

    src_X, src_labels, src_cids, src_uids, src_strata = (
        src["X"], src["labels"], src["country_ids"],
        src["unique_ids"], src["strata"],
    )
    tgt_X, tgt_labels, tgt_cids, tgt_uids = (
        tgt["X"], tgt["labels"], tgt["country_ids"], tgt["unique_ids"],
    )

    src_indices = np.arange(len(src_labels))
    results = []

    for seed in seeds:
        tr_idx, va_idx, _ = make_splits(
            src_indices, src_strata, TRAIN_FRAC, VAL_FRAC, seed
        )

        mean, std  = compute_train_normalization_stats(src_X[tr_idx])
        src_X_norm = apply_normalization(src_X, mean, std)
        tgt_X_norm = apply_normalization(tgt_X, mean, std)

        train_ds = ParcelSequenceDataset(
            src_X_norm[tr_idx], src_labels[tr_idx], src_cids[tr_idx],
            src_uids[tr_idx], n_months
        )
        val_ds = ParcelSequenceDataset(
            src_X_norm[va_idx], src_labels[va_idx], src_cids[va_idx],
            src_uids[va_idx], n_months
        )
        # Full target as test
        test_ds = ParcelSequenceDataset(
            tgt_X_norm, tgt_labels, tgt_cids, tgt_uids, n_months
        )
        results.append((train_ds, val_ds, test_ds))

    return results


# ---------------------------------------------------------------------------
# build_drought_datasets: 2022 train, 2023 test (Spain only)
# ---------------------------------------------------------------------------

def build_drought_datasets(
    seeds: Sequence[int] = SEEDS,
    n_months: Optional[int] = None,
) -> list[tuple[ParcelSequenceDataset, ParcelSequenceDataset, ParcelSequenceDataset]]:
    """
    Drought robustness experiment (§3.8).

    Train + val: Spain 2022 (same splits as in-domain).
    Test: Spain 2023 (full dataset, different year / weather regime).
    Normalization statistics from Spain 2022 train split.
    """
    src = build_combined_dataset(["ES"])          # 2022
    tgt_data = load_sequences("ES", year=2023)

    tgt_X      = tgt_data["X"]
    tgt_labels = _make_labels(tgt_data["crop_class"])
    tgt_cids   = _make_country_ids(None, "ES", len(tgt_X))
    tgt_uids   = tgt_data["unique_id"]

    src_X, src_labels, src_cids, src_uids, src_strata = (
        src["X"], src["labels"], src["country_ids"],
        src["unique_ids"], src["strata"],
    )
    src_indices = np.arange(len(src_labels))
    results = []

    for seed in seeds:
        tr_idx, va_idx, _ = make_splits(
            src_indices, src_strata, TRAIN_FRAC, VAL_FRAC, seed
        )

        mean, std  = compute_train_normalization_stats(src_X[tr_idx])
        src_X_norm = apply_normalization(src_X, mean, std)
        tgt_X_norm = apply_normalization(tgt_X, mean, std)

        train_ds = ParcelSequenceDataset(
            src_X_norm[tr_idx], src_labels[tr_idx], src_cids[tr_idx],
            src_uids[tr_idx], n_months
        )
        val_ds = ParcelSequenceDataset(
            src_X_norm[va_idx], src_labels[va_idx], src_cids[va_idx],
            src_uids[va_idx], n_months
        )
        test_ds = ParcelSequenceDataset(
            tgt_X_norm, tgt_labels, tgt_cids, tgt_uids, n_months
        )
        results.append((train_ds, val_ds, test_ds))

    return results


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building in-domain datasets (5 seeds)…")
    splits = build_datasets()
    tr, va, te = splits[0]
    print(f"  Seed 1: train={len(tr)}, val={len(va)}, test={len(te)}")
    x_s2, x_s1, m2, m1, lbl, cid, uid = tr[0]
    print(f"  x_s2 shape: {x_s2.shape}, x_s1 shape: {x_s1.shape}")
    print(f"  label={lbl.item()}, country={cid.item()}, uid={uid}")
