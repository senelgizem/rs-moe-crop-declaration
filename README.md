# rs-moe-crop-declaration
Code for the paper "Detecting Crop Declaration Mismatches with Sentinel-1/2: An Interpretable Mixture-of-Experts Approach"

---
## What this repository contains

PyTorch/PyTorch Lightning implementations of three crop-monitoring architectures
(Universal GRU, label-conditioned MoE, and Gated-MoE), a reconstruction-based
autoencoder variant for interpretable declaration verification, Google Earth Engine
extraction scripts, and fully reproducible experiment scripts covering every result
reported in the paper.

---

## Repository structure

```
extraction/          GEE JavaScript extraction scripts + Python assembly
models/              Architecture definitions (all four models)
training/            Training loop shared across architectures
experiments/         One script per paper experiment (§3.1–§3.8)
analysis/            Repeated-measures ANOVA + TOST equivalence tests
```

---
## Script → paper table/figure mapping

| Script | Paper section | Output |
|--------|--------------|--------|
| `experiments/classification.py` | §3.1 | Table 3 (macro-F1 per architecture) |
| `experiments/sensor_ablation.py` | §3.2 | Table 4 (sensor ablation) |
| `experiments/cross_country.py` | §3.3 | Table 5–6 (transfer gaps, per-crop) |
| `experiments/expert_specialization.py` | §3.4 | Figure 3 (timing), Figure 4 (heatmap), Table 7 (FP rates) |
| `experiments/verify_declarations.py` | §3.5 | Table 8–9 (softmax-gap AUC, confusion pairs) |
| `experiments/verify_declarations_recon.py` | §3.6 | Table 10 (reconstruction-based AUC) |
| `experiments/partial_season.py` | §3.7 | Table 11 (pre-harvest truncation) |
| `experiments/drought_robustness.py` | §3.8 | Table 12–13 (climate vs. geographic shift) |
| `analysis/statistical_analysis.py` | Appendix A | All ANOVA and TOST tables |
| `extraction/assemble_sequences.py` | §2.3 | FR_sequences.npz, ES_sequences.npz |
| `extraction/assemble_sequences_2023.py` | §2.5.7 | ES_2023_sequences.npz |

---

## Environment setup

Requires Python ≥ 3.10.

```bash
git clone https://github.com/gizemsenel/rs-moe-crop-declaration.git
cd rs-moe-crop-declaration
pip install -r requirements.txt
```

Set the project root (where `data/`, `checkpoints/`, `results/` will live):

```bash
export CROP_MOE_ROOT=/path/to/your/project/root
# Windows: set CROP_MOE_ROOT=D:\path\to\project
```

If `CROP_MOE_ROOT` is not set, scripts default to the current working directory.

---
## Reproducing results

After placing the sequence files and checkpoints in the correct locations:

```bash
# §3.1 Classification (Table 3)
python experiments/classification.py --seeds 1 2 3 4 5

# §3.2 Sensor ablation (Table 4)
python experiments/sensor_ablation.py --seeds 1 2 3 4 5

# §3.3 Cross-country transfer (Tables 5–6)
python training/train.py --model_type moe --source ES --target FR --seed 1
# ... (all transfer configurations)
python experiments/cross_country.py

# §3.4 Expert specialization / interpretability (Figures 3–4, Table 7)
python experiments/expert_specialization.py

# §3.5 Softmax-gap verification (Tables 8–9)
python experiments/verify_declarations.py

# §3.6 Reconstruction-based verification (Table 10)
python experiments/verify_declarations_recon.py

# §3.7 Partial-season (Table 11)
python experiments/partial_season.py --seeds 1 2 3 4 5

# §3.8 Drought robustness (Tables 12–13)
python experiments/drought_robustness.py

# Appendix A: ANOVA + TOST tables (requires all results CSVs)
python analysis/statistical_analysis.py
```

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
