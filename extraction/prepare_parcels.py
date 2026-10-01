"""
extraction/prepare_parcels.py

Prepare LPIS parcel data for Sentinel-1/2 time-series extraction.

Input
-----
Harmonised IACS geoparquet files from Zenodo record 15692199:
    France : GSA-FR_FR-2022.geoparquet
    Spain  : GSA-ES_ZAR-2022.geoparquet
             GSA-ES_HEC-2022.geoparquet
             GSA-ES_NAV-2022.geoparquet
             GSA-ES_TER-2022.geoparquet

Place the downloaded files in a directory and pass it with --iacs_dir.

Output
------
Two GeoJSON files written to --out_dir (default: data/parcel_assets/):
    FR_parcels.geojson   ~50 k parcels, Nouvelle-Aquitaine + Occitanie
    ES_parcels.geojson   ~50 k parcels, ZAR + HEC + NAV + TER provinces

These files are then uploaded as Earth Engine table assets:
    earthengine upload table --asset_id users/<you>/FR_parcels FR_parcels.geojson
    earthengine upload table --asset_id users/<you>/ES_parcels ES_parcels.geojson

After upload, fill in the PARCEL_ASSETS dict inside
extraction/extract_timeseries.js and run that script in the GEE Code Editor.

Parcel selection criteria
-------------------------
* Crop class must be one of: wheat, maize, sunflower, barley
  (IACS crop codes resolved via a hard-coded mapping below)
* Parcel area ≥ 0.5 ha  (strips very small parcels that produce noisy zonal stats)
* Geometry is valid (no null / empty geometries)
* France: regions Nouvelle-Aquitaine (11) and Occitanie (76)
* Spain:  all four provinces provided (ZAR, HEC, NAV, TER)

The output retains three columns for downstream use:
    field_id    — original IACS field identifier (string)
    crop_class  — canonical crop name  (wheat / maize / sunflower / barley)
    area_ha     — parcel area in hectares

Usage
-----
    python extraction/prepare_parcels.py \\
        --iacs_dir /path/to/iacs_geoparquets \\
        --out_dir  data/parcel_assets
"""

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ---------------------------------------------------------------------------
# Crop-code → canonical class mapping
# ---------------------------------------------------------------------------
# IACS files use numeric or alpha codes that differ by country.
# The keys below cover the codes found in the 2022 harmonised inventory.
# Extend as needed for other years or countries.

FR_CROP_MAP: dict[str, str] = {
    # Wheat (blé tendre / blé dur)
    "BLE_TENDRE":  "wheat",
    "BLE_DUR":     "wheat",
    "BTH":         "wheat",
    "BDH":         "wheat",
    # Maize (maïs grain + maïs fourrage counted as maize)
    "MAIS_GRAIN":  "maize",
    "MAIS_FOURR":  "maize",
    "MIS":         "maize",
    "MIE":         "maize",
    # Sunflower (tournesol)
    "TOURN":       "sunflower",
    "TRN":         "sunflower",
    # Barley (orge d'hiver / orge de printemps)
    "ORGE_HIVER":  "barley",
    "ORGE_PRINT":  "barley",
    "ORP":         "barley",
    "ORH":         "barley",
}

ES_CROP_MAP: dict[str, str] = {
    # Wheat (trigo blando / trigo duro)
    "1111": "wheat",   # trigo blando
    "1112": "wheat",   # trigo duro
    "TRB":  "wheat",
    "TRD":  "wheat",
    # Maize
    "1221": "maize",
    "MAI":  "maize",
    # Sunflower
    "1462": "sunflower",
    "GIR":  "sunflower",
    # Barley (cebada)
    "1121": "barley",
    "CEB":  "barley",
}

# Columns that may hold the crop code (tried in order)
_FR_CODE_COLS  = ["crop_type", "CODE_CULTU", "code_cultu", "crop_code", "CODE_GROUPE"]
_ES_CODE_COLS  = ["crop_type", "tipo_cultivo", "cultivo_codigo", "crop_code", "PRODUCTO"]
_FIELD_ID_COLS = ["field_id", "FIELD_ID", "ID_PARCEL", "id_parcel", "fid", "FID"]
_AREA_COLS     = ["area_ha", "AREA_HA", "area", "AREA", "surf_ha", "SURF_HA"]

# France NUTS-2 codes for Nouvelle-Aquitaine and Occitanie
FR_REGIONS = {"FRK1", "FRK2", "FRL0"}   # FR61/FR62/FR63/FR72/FR73 harmonised
FR_REGION_CODES = {11, 76}               # INSEE département grouping (alternative)
_FR_REGION_COLS = ["reg", "REG", "nuts2", "NUTS2", "region_code", "REGION_CODE"]

MIN_AREA_HA = 0.5

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_col(df: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    """Return the first candidate column present in *df*, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_crop(series: pd.Series, code_map: dict[str, str]) -> pd.Series:
    """Map raw crop codes → canonical class name; NaN where unmapped."""
    return series.astype(str).str.strip().str.upper().map(
        {k.upper(): v for k, v in code_map.items()}
    )


def _read_geoparquet(path: Path) -> gpd.GeoDataFrame:
    print(f"  Reading {path.name} …", flush=True)
    gdf = gpd.read_file(path)
    print(f"    {len(gdf):,} rows, CRS={gdf.crs}", flush=True)
    return gdf


def _compute_area_ha(gdf: gpd.GeoDataFrame, area_col: str | None) -> pd.Series:
    """Return area in hectares: use existing column or reproject + compute."""
    if area_col and area_col in gdf.columns:
        return gdf[area_col].astype(float)
    # Reproject to an equal-area CRS for computation
    utm = gdf.to_crs(gdf.estimate_utm_crs())
    return utm.geometry.area / 10_000.0


def _select_cols(
    gdf: gpd.GeoDataFrame,
    field_id_col: str,
    crop_class: pd.Series,
    area_ha: pd.Series,
) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "field_id":   gdf[field_id_col].astype(str),
            "crop_class": crop_class,
            "area_ha":    area_ha.round(4),
        },
        geometry=gdf.geometry,
        crs=gdf.crs,
    )


# ---------------------------------------------------------------------------
# France
# ---------------------------------------------------------------------------

def prepare_france(iacs_dir: Path) -> gpd.GeoDataFrame:
    """Load, filter and harmonise French LPIS parcels."""
    candidate = iacs_dir / "GSA-FR_FR-2022.geoparquet"
    if not candidate.exists():
        # Allow alternate naming
        matches = sorted(iacs_dir.glob("GSA-FR*2022*.geoparquet"))
        if not matches:
            raise FileNotFoundError(
                f"No French IACS geoparquet found in {iacs_dir}.\n"
                "Expected: GSA-FR_FR-2022.geoparquet"
            )
        candidate = matches[0]

    gdf = _read_geoparquet(candidate)

    # ------------------------------------------------------------------
    # Region filter: Nouvelle-Aquitaine + Occitanie
    # ------------------------------------------------------------------
    reg_col = _first_col(gdf, _FR_REGION_COLS)
    if reg_col:
        raw = gdf[reg_col].astype(str).str.strip()
        # Try NUTS2 codes first, fall back to INSEE integer codes
        nuts_mask = raw.isin(FR_REGIONS) | raw.isin({str(c) for c in FR_REGION_CODES})
        if nuts_mask.any():
            gdf = gdf[nuts_mask].copy()
            print(f"    After region filter: {len(gdf):,} rows")
        else:
            print(f"    WARNING: region column '{reg_col}' values not recognised; "
                  "skipping region filter — all of France will be used.")
    else:
        print("    WARNING: no region column found; skipping France region filter.")

    # ------------------------------------------------------------------
    # Crop-code mapping
    # ------------------------------------------------------------------
    crop_col = _first_col(gdf, _FR_CODE_COLS)
    if crop_col is None:
        raise KeyError(
            f"No crop-code column found in France file. "
            f"Columns present: {list(gdf.columns)}"
        )
    crop_class = _resolve_crop(gdf[crop_col], FR_CROP_MAP)
    gdf = gdf[crop_class.notna()].copy()
    crop_class = crop_class[crop_class.notna()]
    print(f"    After crop filter: {len(gdf):,} rows")

    # ------------------------------------------------------------------
    # Area filter
    # ------------------------------------------------------------------
    area_ha = _compute_area_ha(gdf, _first_col(gdf, _AREA_COLS))
    mask = area_ha >= MIN_AREA_HA
    gdf = gdf[mask].copy()
    crop_class = crop_class[mask]
    area_ha = area_ha[mask]
    print(f"    After area≥{MIN_AREA_HA} ha filter: {len(gdf):,} rows")

    # ------------------------------------------------------------------
    # Geometry validity
    # ------------------------------------------------------------------
    valid = gdf.geometry.is_valid & ~gdf.geometry.is_empty
    gdf = gdf[valid].copy()
    crop_class = crop_class[valid]
    area_ha = area_ha[valid]
    print(f"    After geometry filter: {len(gdf):,} rows")

    field_col = _first_col(gdf, _FIELD_ID_COLS)
    if field_col is None:
        # Fall back to sequential integer index
        gdf["_field_id"] = [f"FR_{i}" for i in range(len(gdf))]
        field_col = "_field_id"

    out = _select_cols(gdf, field_col, crop_class.reset_index(drop=True),
                       area_ha.reset_index(drop=True))
    print(f"    Crop distribution:\n{out['crop_class'].value_counts().to_string()}")
    return out


# ---------------------------------------------------------------------------
# Spain
# ---------------------------------------------------------------------------

def prepare_spain(iacs_dir: Path) -> gpd.GeoDataFrame:
    """Load, filter and harmonise Spanish LPIS parcels (ZAR, HEC, NAV, TER)."""
    provinces = ["ZAR", "HEC", "NAV", "TER"]
    parts: list[gpd.GeoDataFrame] = []

    for prov in provinces:
        candidate = iacs_dir / f"GSA-ES_{prov}-2022.geoparquet"
        if not candidate.exists():
            matches = sorted(iacs_dir.glob(f"GSA-ES*{prov}*2022*.geoparquet"))
            if not matches:
                print(f"  WARNING: {candidate.name} not found — skipping {prov}.")
                continue
            candidate = matches[0]

        gdf = _read_geoparquet(candidate)
        gdf["_province"] = prov

        # Crop mapping
        crop_col = _first_col(gdf, _ES_CODE_COLS)
        if crop_col is None:
            raise KeyError(
                f"No crop-code column found in Spain/{prov} file. "
                f"Columns present: {list(gdf.columns)}"
            )
        crop_class = _resolve_crop(gdf[crop_col], ES_CROP_MAP)
        gdf = gdf[crop_class.notna()].copy()
        crop_class = crop_class[crop_class.notna()]

        # Area
        area_ha = _compute_area_ha(gdf, _first_col(gdf, _AREA_COLS))
        mask = area_ha >= MIN_AREA_HA
        gdf = gdf[mask].copy()
        crop_class = crop_class[mask]
        area_ha = area_ha[mask]

        # Geometry
        valid = gdf.geometry.is_valid & ~gdf.geometry.is_empty
        gdf = gdf[valid].copy()
        crop_class = crop_class[valid]
        area_ha = area_ha[valid]

        field_col = _first_col(gdf, _FIELD_ID_COLS)
        if field_col is None:
            gdf["_field_id"] = [f"ES_{prov}_{i}" for i in range(len(gdf))]
            field_col = "_field_id"

        part = _select_cols(gdf, field_col,
                            crop_class.reset_index(drop=True),
                            area_ha.reset_index(drop=True))
        part["_province"] = prov
        print(f"    {prov}: {len(part):,} parcels after filters")
        parts.append(part)

    if not parts:
        raise RuntimeError("No Spanish provinces could be loaded.")

    combined = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    print(f"    Spain total: {len(combined):,} rows")
    print(f"    Crop distribution:\n{combined['crop_class'].value_counts().to_string()}")

    # Drop helper column
    combined = combined.drop(columns=["_province"], errors="ignore")
    return combined


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare LPIS parcel data for GEE extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--iacs_dir",
        type=Path,
        required=True,
        help="Directory containing downloaded IACS geoparquet files.",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/parcel_assets"),
        help="Directory for output GeoJSON files.",
    )
    parser.add_argument(
        "--country",
        choices=["FR", "ES", "both"],
        default="both",
        help="Which country (or both) to process.",
    )
    parser.add_argument(
        "--crs",
        default="EPSG:4326",
        help="Output CRS (GEE requires WGS84 = EPSG:4326).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.country in ("FR", "both"):
        print("\n=== France ===")
        fr = prepare_france(args.iacs_dir)
        fr = fr.to_crs(args.crs)
        out_path = args.out_dir / "FR_parcels.geojson"
        fr.to_file(out_path, driver="GeoJSON")
        print(f"  → Saved {len(fr):,} parcels to {out_path}")
        print(
            "  Next step:\n"
            f"    earthengine upload table \\\n"
            f"      --asset_id users/<you>/FR_parcels \\\n"
            f"      {out_path}"
        )

    if args.country in ("ES", "both"):
        print("\n=== Spain ===")
        es = prepare_spain(args.iacs_dir)
        es = es.to_crs(args.crs)
        out_path = args.out_dir / "ES_parcels.geojson"
        es.to_file(out_path, driver="GeoJSON")
        print(f"  → Saved {len(es):,} parcels to {out_path}")
        print(
            "  Next step:\n"
            f"    earthengine upload table \\\n"
            f"      --asset_id users/<you>/ES_parcels \\\n"
            f"      {out_path}"
        )

    print("\nDone. Upload the GeoJSON files to Earth Engine, then run "
          "extraction/extract_timeseries.js in the GEE Code Editor.")


if __name__ == "__main__":
    main()
