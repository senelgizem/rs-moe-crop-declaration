# rs-moe-crop-declaration
Code for the paper

> **Detecting Crop Declaration Mismatches with Sentinel-1/2: An Interpretable Mixture-of-Experts Approach**  
> Gizem Şenel

---
##Overview

PyTorch/PyTorch Lightning implementations of three crop-monitoring architectures
(Universal GRU, label-conditioned MoE, and Gated-MoE), a reconstruction-based
autoencoder variant for interpretable declaration verification, Google Earth Engine
extraction scripts, and fully reproducible experiment scripts covering every result
reported in the paper.

This repository reproduces all results in the paper.  The pipeline has three phases:

1. **Extraction** — prepare LPIS parcel data locally (`extraction/prepare_parcels.py`), upload to Google Earth Engine, run Sentinel-1/2 time-series extraction (`extraction/extract_timeseries.js`), then assemble the chunked CSV outputs into `.npz` sequence files (`extraction/assemble_sequences.py`).

2. **Training** — PyTorch / PyTorch Lightning model training (`training/train.py`).

3. **Experiments** — one script per paper section in `experiments/`; statistical tests in `analysis/`.

---

## Repository structure
```
rs-moe-crop-declaration/
├── config.py                        # all paths and hyper-parameters
├── extraction/
│   ├── prepare_parcels.py           # filter IACS parcel data locally
│   ├── extract_timeseries.js        # GEE script: S1/S2 extraction (2022)
│   ├── extract_timeseries_2023.js   # GEE script: Spain 2023 drought year
│   ├── assemble_sequences.py        # merge GEE CSV chunks → .npz
│   └── assemble_sequences_2023.py   # Spain 2023 variant
├── models/
│   ├── encoder.py                   # SensorFusionFrontend (shared)
│   ├── universal.py                 # Universal GRU baseline
│   ├── moe.py                       # MoE (label-conditioned)
│   ├── gated_moe.py                 # Gated-MoE (learned gate)
│   └── autoencoder.py               # MoE Autoencoder (reconstruction verifier)
├── training/
│   ├── dataset.py                   # ParcelSequenceDataset + builders
│   ├── lightning_module.py          # LightningModule for classifier models
│   ├── lightning_module_autoencoder.py
│   └── train.py                     # CLI training entry point
├── experiments/
│   ├── classification.py            # §3.1 — in-domain classification (Table 3)
│   ├── sensor_ablation.py           # §3.2 — sensor ablation (Table 4)
│   ├── cross_country.py             # §3.3 — cross-country transfer (Tables 5-6)
│   ├── expert_specialization.py     # §3.4 — MoE routing analysis (Figs 3-4, Table 7)
│   ├── verify_declarations.py       # §3.5 — classification-based verification (Tables 8-9)
│   ├── verify_declarations_recon.py # §3.6 — reconstruction-based verification (Table 10)
│   ├── partial_season.py            # §3.7 — early-season classification (Table 11)
│   └── drought_robustness.py        # §3.8 — drought-year robustness (Tables 12-13)
├── analysis/
│   └── statistical_analysis.py      # RM-ANOVA + TOST equivalence tests
└── data/                            # (not tracked by git — see below)
    ├── gee_exports/                 # raw GEE CSV chunks (2022)
    ├── gee_exports_2023/            # raw GEE CSV chunks (2023)
    ├── sequences/                   # assembled .npz files (2022)
    └── sequences_2023/              # assembled .npz files (2023)
```


## Setup

```bash
git clone https://github.com/senelgizem/rs-moe-crop-declaration.git
cd rs-moe-crop-declaration
pip install -r requirements.txt
```

If the repository root is not your working directory, set:

```bash
export CROP_MOE_ROOT=/path/to/rs-moe-crop-declaration
```

---

## Reproducing the results

Run experiments in order (each later script depends on checkpoints from earlier ones):

```bash
# 1. In-domain classification (trains all 3 × 5 = 15 models)
python experiments/classification.py

# 2. Sensor ablation
python experiments/sensor_ablation.py

# 3. Cross-country transfer
python experiments/cross_country.py

# 4. MoE expert specialisation (reads classification/ checkpoints)
python experiments/expert_specialization.py

# 5. Declaration verification (reads classification/ checkpoints)
python experiments/verify_declarations.py

# 6. Reconstruction-based verification (trains autoencoder)
python experiments/verify_declarations_recon.py

# 7. Early-season (reads classification/ checkpoints; no retraining)
python experiments/partial_season.py

# 8. Drought robustness (requires data/sequences_2023/)
python experiments/drought_robustness.py

# 9. Statistical tests
python analysis/statistical_analysis.py
```

Outputs are written to `results/<experiment_name>/`.

---

## Model overview

| Model | Architecture | Gate |
|---|---|---|
| **Universal GRU** | SensorFusion → shared GRU → Linear(4) | — |
| **MoE** | SensorFusion → 4 crop-specific GRUs → fit scores | None (label-conditioned) |
| **Gated-MoE** | SensorFusion → 4 expert GRUs blended by learned gate | Linear softmax |
| **MoE Autoencoder** | SensorFusion → 4 encoder-decoder experts | — (reconstruction error) |

All classifiers share the same `SensorFusionFrontend`:
- S2 GRU (hidden=32) + S1 GRU (hidden=32) → concat → Linear(64→48) + ReLU

Training: AdamW (lr=1e-3, weight_decay=1e-4), batch=64, max_epochs=50, patience=10, 5 seeds.

---

## Repeated-measures design

Every architecture comparison uses **5 random seeds**, with each seed
varying both the train/val/test data split and model weight initialization.
All results are reported as mean ± SD across seeds. Per-seed raw values are
saved as `results/*_raw.csv`; seed-aggregated summaries as `results/*_summary.csv`.
This design is described in §2.5.8 of the paper.

---


## Citation

to be added
