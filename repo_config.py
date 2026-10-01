"""
config.py

Single source of truth for paths and shared constants. Set CROP_MOE_ROOT
to your project root (parent of data/, checkpoints/, results/). If unset,
defaults to the current working directory.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CROP_MOE_ROOT", "."))

DATA_DIR        = PROJECT_ROOT / "data"
SEQ_DIR         = DATA_DIR / "sequences"
EXTRACTED_DIR   = DATA_DIR / "extracted"
PREPARED_DIR    = DATA_DIR / "prepared"
GEE_EXPORTS_DIR = DATA_DIR / "gee_exports"
GEE_2023_DIR    = DATA_DIR / "gee_exports_2023"
BOUNDARY_PATH   = DATA_DIR / "boundary" / "gaul_2025_l1.shp"

CHECKPOINT_DIR  = PROJECT_ROOT / "checkpoints"
RESULTS_DIR     = PROJECT_ROOT / "results"
FIGURES_DIR     = PROJECT_ROOT / "figures"

# ── Crop labels ────────────────────────────────────────────────────────────
CROP_TO_LABEL = {"wheat": 0, "maize": 1, "sunflower": 2, "barley": 3}
LABEL_TO_CROP = {v: k for k, v in CROP_TO_LABEL.items()}
N_CLASSES     = len(CROP_TO_LABEL)

# ── Feature schema (14 per monthly composite × 13 months = 182 total) ─────
# TCARI was excluded after a division-by-near-zero instability was identified
# in its formula under low red-band reflectance; see
# extraction/remove_tcari_feature.py for the diagnostic.
FEATURE_ORDER = [
    # Sentinel-2 (10 features)
    "B5", "B6", "B7", "B11", "B12",
    "NDVI", "EVI", "NDWI", "NDMI",
    "s2_valid_obs",
    # Sentinel-1 (4 features)
    "VV", "VH", "VH_VV",
    "s1_valid_obs",
]

S2_FEATURE_IDX = list(range(0, 10))   # B5..NDMI + s2_valid_obs
S1_FEATURE_IDX = list(range(10, 14))  # VV, VH, VH_VV, s1_valid_obs

# ── Training hyperparameters (Section 2.5.8) ───────────────────────────────
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
MAX_EPOCHS   = 50
PATIENCE     = 10   # early stopping on val macro-F1 (or val best-fit acc for autoencoder)
SEEDS        = [1, 2, 3, 4, 5]

# ── Experiment constants ───────────────────────────────────────────────────
CORRUPTION_RATES   = [0.1, 0.2, 0.3]
PARTIAL_CUTOFFS    = [5, 7, 9, None]  # None = full 13-month season
SESOI_LOW          = 0.01   # TOST SESOI for in-domain comparisons
SESOI_HIGH         = 0.03   # TOST SESOI for distribution-shift comparisons

for _d in (SEQ_DIR, CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
