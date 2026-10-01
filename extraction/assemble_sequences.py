"""
extraction/assemble_sequences.py

Merges the chunked GEE export CSVs (from extract_timeseries.js) into the
final per-country sequence files used for training and evaluation:
    data/sequences/FR_sequences.npz
    data/sequences/ES_sequences.npz

Each .npz contains:
    X             (N, 13, 14)  float32  -- NaN where no valid observation
    field_id      (N,)         str
    crop_class    (N,)         str      -- "wheat" | "barley" | "maize" | "sunflower"
    area_ha       (N,)         float32
    unique_id     (N,)         str      -- "{country}_{i:06d}"
    timesteps     (13,)        str      -- "YYYYMM01" labels
    feature_names (14,)        str      -- matches config.FEATURE_ORDER exactly

Usage:
    python extraction/assemble_sequences.py
    python extraction/assemble_sequences.py --countries FR
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GEE_EXPORTS_DIR, SEQ_DIR, FEATURE_ORDER

COUNTRIES     = ["FR", "ES"]
METADATA_COLS = ["field_id", "country", "crop_class", "area_ha"]


def load_and_merge(country: str) -> pd.DataFrame:
    pattern = f"{country}_chunk*.csv"
    files   = sorted(GEE_EXPORTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {GEE_EXPORTS_DIR}.\n"
            "Download the GEE Drive export folder there first."
        )
    print(f"[{country}] {len(files)} chunk files found")
    dfs = [pd.read_csv(f) for f in files]

    # Try key-based merge first (field_id unique within country)
    if not any(df["field_id"].duplicated().any() for df in dfs):
        print(f"[{country}] field_id unique within chunks — key merge")
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(
                df.drop(columns=[c for c in METADATA_COLS if c != "field_id"]),
                on="field_id", how="inner"
            )
    else:
        # Spain: field_id is only unique within province; verify row-positional alignment
        print(f"[{country}] field_id has duplicates — verifying positional alignment")
        base, ok = dfs[0][["area_ha"]].reset_index(drop=True), True
        for i, df in enumerate(dfs[1:], 1):
            other = df[["area_ha"]].reset_index(drop=True)
            if len(other) != len(base):
                print(f"  chunk {i}: DIFFERENT ROW COUNT — not safe to merge")
                ok = False
                continue
            mismatch = (~np.isclose(base["area_ha"].values,
                                     other["area_ha"].values, equal_nan=True)).mean()
            print(f"  chunk {i} vs chunk 0: area_ha mismatch fraction = {mismatch:.4%}")
            if mismatch > 0.001:
                ok = False
        if not ok:
            raise ValueError(f"[{country}] Positional alignment FAILED — cannot merge safely.")
        print(f"[{country}] alignment confirmed — row-position merge")
        merged = dfs[0].reset_index(drop=True)
        for df in dfs[1:]:
            merged = pd.concat(
                [merged, df.drop(columns=METADATA_COLS).reset_index(drop=True)], axis=1
            )

    return merged


def discover_timesteps(columns: list) -> list:
    probe   = FEATURE_ORDER[0]
    pattern = re.compile(rf"^{probe}_(\d{{8}})$")
    dates   = sorted({m.group(1) for col in columns if (m := pattern.match(col))})
    if not dates:
        raise ValueError(f"No columns matched '{probe}_<YYYYMMDD>' — check FEATURE_ORDER.")
    return dates


def build_array(df: pd.DataFrame, timesteps: list) -> np.ndarray:
    N, T, F = len(df), len(timesteps), len(FEATURE_ORDER)
    arr     = np.full((N, T, F), np.nan, dtype=np.float32)
    missing = []
    for ti, date in enumerate(timesteps):
        for fi, feat in enumerate(FEATURE_ORDER):
            col = f"{feat}_{date}"
            if col in df.columns:
                arr[:, ti, fi] = df[col].values
            else:
                missing.append(col)
    if missing:
        print(f"  WARNING: {len(missing)} expected columns missing, e.g. {missing[:3]}")
    return arr


def process(country: str) -> None:
    print(f"\n=== {country} ===")
    df        = load_and_merge(country)
    timesteps = discover_timesteps(list(df.columns))
    print(f"[{country}] {len(timesteps)} timesteps: {timesteps[0]} … {timesteps[-1]}")
    X         = build_array(df, timesteps)
    nan_frac  = float(np.isnan(X).mean())
    print(f"[{country}] X shape: {X.shape}  NaN fraction: {nan_frac:.1%}")

    out = SEQ_DIR / f"{country}_sequences.npz"
    np.savez_compressed(
        out,
        X             = X,
        field_id      = df["field_id"].values,
        crop_class    = df["crop_class"].values,
        area_ha       = df["area_ha"].values.astype(np.float32),
        unique_id     = np.array([f"{country}_{i:06d}" for i in range(len(df))]),
        timesteps     = np.array(timesteps),
        feature_names = np.array(FEATURE_ORDER),
    )
    print(f"[{country}] saved → {out}")

    meta = dict(
        country       = country,
        n_parcels     = int(len(df)),
        n_timesteps   = len(timesteps),
        n_features    = len(FEATURE_ORDER),
        timesteps     = timesteps,
        feature_names = FEATURE_ORDER,
        class_balance = df["crop_class"].value_counts().to_dict(),
        nan_fraction  = nan_frac,
    )
    (SEQ_DIR / f"{country}_sequences_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=COUNTRIES,
                    choices=COUNTRIES, help="Countries to process (default: both)")
    args = ap.parse_args()
    for c in args.countries:
        process(c)


if __name__ == "__main__":
    main()
