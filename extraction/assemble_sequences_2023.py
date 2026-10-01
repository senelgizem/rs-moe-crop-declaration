"""
extraction/assemble_sequences_2023.py

Merges the chunked GEE export CSVs for the Spain 2023 drought-robustness
experiment (from extract_timeseries_2023.js) into the sequence file used
for cross-year evaluation:
    data/sequences_2023/ES_sequences_2023.npz

The 2023 export uses the same parcel footprint as 2022 (same field_id list),
but a fresh GEE extraction over the 2023 season.  The .npz schema is
identical to the 2022 files produced by assemble_sequences.py so that
drought_robustness.py can load both without special-casing.

Usage:
    python extraction/assemble_sequences_2023.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import GEE_EXPORTS_2023_DIR, SEQ_2023_DIR, FEATURE_ORDER

COUNTRY = "ES"
METADATA_COLS = ["field_id", "country", "crop_class", "area_ha"]


def load_and_merge_2023() -> pd.DataFrame:
    """
    Merge chunked 2023 GEE exports for Spain.

    Spain's field_id is only unique within province, so chunks may contain
    duplicate field_id values across provinces.  Use row-positional merge
    after verifying alignment via area_ha (same logic as assemble_sequences.py).
    """
    pattern = "ES_chunk*.csv"
    files   = sorted(GEE_EXPORTS_2023_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' in {GEE_EXPORTS_2023_DIR}.\n"
            "Download the 2023 GEE Drive export folder there first."
        )
    print(f"[ES-2023] {len(files)} chunk files found")
    dfs = [pd.read_csv(f) for f in files]

    # Try key-based merge first; fall back to positional if field_id has dupes
    if not any(df["field_id"].duplicated().any() for df in dfs):
        print("[ES-2023] field_id unique within chunks — key merge")
        merged = dfs[0]
        for df in dfs[1:]:
            merged = merged.merge(
                df.drop(columns=[c for c in METADATA_COLS if c != "field_id"]),
                on="field_id", how="inner"
            )
    else:
        print("[ES-2023] field_id has duplicates — verifying positional alignment")
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
            raise ValueError("[ES-2023] Positional alignment FAILED — cannot merge safely.")
        print("[ES-2023] alignment confirmed — row-position merge")
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


def process() -> None:
    print("\n=== ES-2023 ===")
    df        = load_and_merge_2023()
    timesteps = discover_timesteps(list(df.columns))
    print(f"[ES-2023] {len(timesteps)} timesteps: {timesteps[0]} … {timesteps[-1]}")
    X        = build_array(df, timesteps)
    nan_frac = float(np.isnan(X).mean())
    print(f"[ES-2023] X shape: {X.shape}  NaN fraction: {nan_frac:.1%}")

    SEQ_2023_DIR.mkdir(parents=True, exist_ok=True)
    out = SEQ_2023_DIR / "ES_sequences_2023.npz"
    np.savez_compressed(
        out,
        X             = X,
        field_id      = df["field_id"].values,
        crop_class    = df["crop_class"].values,
        area_ha       = df["area_ha"].values.astype(np.float32),
        unique_id     = np.array([f"ES2023_{i:06d}" for i in range(len(df))]),
        timesteps     = np.array(timesteps),
        feature_names = np.array(FEATURE_ORDER),
    )
    print(f"[ES-2023] saved → {out}")

    meta = dict(
        country       = "ES",
        year          = 2023,
        n_parcels     = int(len(df)),
        n_timesteps   = len(timesteps),
        n_features    = len(FEATURE_ORDER),
        timesteps     = timesteps,
        feature_names = FEATURE_ORDER,
        class_balance = df["crop_class"].value_counts().to_dict(),
        nan_fraction  = nan_frac,
    )
    (SEQ_2023_DIR / "ES_sequences_2023_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true",
                    help="Print alignment check without writing output")
    args = ap.parse_args()
    if args.check_only:
        df = load_and_merge_2023()
        print(f"Check complete — {len(df)} parcels, no output written.")
    else:
        process()


if __name__ == "__main__":
    main()
