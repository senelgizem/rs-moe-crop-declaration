"""
analysis/statistical_analysis.py

Statistical comparisons for the paper.

Tests run
---------
1. RM-ANOVA across the three architectures (within-subject factor = model,
   5 observations per level = 5 seeds, matched on dataset split).
   Null hypothesis: all three models have equal macro-F1.
   Implemented via pingouin's rm_anova with Greenhouse-Geisser correction.

2. Pairwise TOST equivalence tests (universal vs. moe, universal vs.
   gated_moe, moe vs. gated_moe).
   Two one-sided t-tests (paired, within-seed), equivalence bounds
   ±SESOI_LOW (in-domain) or ±SESOI_HIGH (cross-country / distribution shift).

   SESOI (smallest effect size of interest):
       ±0.01 macro-F1 for in-domain comparisons
       ±0.03 macro-F1 for cross-country / drought comparisons

   Equivalence declared when both one-sided p < 0.05.

All results written to results/statistical_analysis/.

Usage
-----
python analysis/statistical_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RESULTS_DIR, SESOI_LOW, SESOI_HIGH

try:
    import pingouin as pg
except ImportError:
    pg = None

from scipy import stats

ANALYSIS_DIR = RESULTS_DIR / "statistical_analysis"
MODEL_TYPES  = ["universal", "moe", "gated_moe"]


# ---------------------------------------------------------------------------
# TOST (Two One-Sided T-Tests)
# ---------------------------------------------------------------------------

def tost_paired(
    a: np.ndarray,
    b: np.ndarray,
    sesoi: float,
) -> dict:
    """
    Paired TOST for equivalence of two conditions a and b.

    Equivalence is declared when the effect (mean difference) falls within
    the interval (−sesoi, +sesoi), i.e. when both one-sided tests reject:
        H1: μ_d < −sesoi    (lower bound)
        H2: μ_d >  sesoi    (upper bound)

    Parameters
    ----------
    a, b   : paired observations (n,) — e.g. F1 per seed for model A and B
    sesoi  : equivalence bound (positive scalar)

    Returns
    -------
    dict with:
        mean_diff  : mean(a) − mean(b)
        ci_95_low, ci_95_high  : 95% CI on the difference
        t_low, p_low   : lower one-sided test (H1: diff < −sesoi)
        t_high, p_high : upper one-sided test (H2: diff >  sesoi)
        equivalent     : bool — True if both p < 0.05
    """
    d   = np.asarray(a) - np.asarray(b)
    n   = len(d)
    mu  = d.mean()
    se  = d.std(ddof=1) / np.sqrt(n)

    # Upper one-sided test: H0: mean_d >= sesoi  → reject → mean_d < sesoi
    t_high = (mu - sesoi)  / se
    p_high = stats.t.cdf(t_high, df=n - 1)   # left-tail p for t < threshold

    # Lower one-sided test: H0: mean_d <= −sesoi → reject → mean_d > −sesoi
    t_low  = (mu + sesoi)  / se
    p_low  = stats.t.sf(t_low,  df=n - 1)    # right-tail p for t > threshold

    ci_half = stats.t.ppf(0.975, df=n - 1) * se
    equivalent = (p_high < 0.05) and (p_low < 0.05)

    return dict(
        mean_diff   = float(mu),
        ci_95_low   = float(mu - ci_half),
        ci_95_high  = float(mu + ci_half),
        t_low       = float(t_low),
        p_low       = float(p_low),
        t_high      = float(t_high),
        p_high      = float(p_high),
        equivalent  = equivalent,
    )


# ---------------------------------------------------------------------------
# RM-ANOVA
# ---------------------------------------------------------------------------

def rm_anova_3models(df_wide: pd.DataFrame) -> dict:
    """
    Repeated-measures ANOVA across 3 models (within-subject = model,
    subjects = seeds).

    Parameters
    ----------
    df_wide : DataFrame with columns ['seed', 'universal', 'moe', 'gated_moe']

    Returns
    -------
    dict with F, p, eps (Greenhouse-Geisser), and a human-readable summary
    """
    if pg is None:
        # Fallback: Friedman test (non-parametric)
        stat, p = stats.friedmanchisquare(
            df_wide["universal"].values,
            df_wide["moe"].values,
            df_wide["gated_moe"].values,
        )
        return dict(test="Friedman", statistic=float(stat), p=float(p),
                    note="pingouin not installed; used Friedman test.")

    df_long = df_wide.melt(id_vars="seed", var_name="model", value_name="macro_f1")
    aov = pg.rm_anova(
        data=df_long, dv="macro_f1", within="model", subject="seed",
        correction=True,          # Greenhouse-Geisser
        detailed=False,
    )
    row = aov.iloc[0]
    return dict(
        test  = "RM-ANOVA (Greenhouse-Geisser corrected)",
        F     = float(row["F"]),
        p     = float(row["p-unc"]),
        eps   = float(row.get("eps", float("nan"))),
        p_GG  = float(row.get("p-GG-corr", row["p-unc"])),
    )


# ---------------------------------------------------------------------------
# Load results from experiment CSVs
# ---------------------------------------------------------------------------

def load_f1(csv_path: Path, model_col: str = "model",
            f1_col: str = "macro_f1") -> pd.DataFrame:
    """Load a per-seed F1 CSV and return a wide table (seeds × models)."""
    df = pd.read_csv(csv_path)
    if model_col not in df.columns or f1_col not in df.columns:
        raise ValueError(f"Expected columns '{model_col}', '{f1_col}' in {csv_path}")
    wide = (
        df[df[model_col].isin(MODEL_TYPES)]
        .groupby(["seed", model_col])[f1_col].mean().unstack(model_col)
        .reset_index()
    )
    return wide


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def run_experiment(
    csv_path: Path,
    label: str,
    sesoi: float,
) -> list[dict]:
    """
    Run RM-ANOVA + all pairwise TOST for one experiment CSV.

    Returns a list of result dicts.
    """
    if not csv_path.exists():
        print(f"  MISSING: {csv_path} — skipping {label}")
        return []

    try:
        wide = load_f1(csv_path)
    except Exception as e:
        print(f"  ERROR loading {csv_path}: {e}")
        return []

    if set(MODEL_TYPES) - set(wide.columns):
        print(f"  Some model types missing in {csv_path}, skipping {label}.")
        return []

    rows = []

    # RM-ANOVA
    anova_res = rm_anova_3models(wide)
    p_anova   = anova_res.get("p_GG", anova_res.get("p", float("nan")))
    print(f"  RM-ANOVA: F={anova_res.get('F', '?'):.3f}, "
          f"p={p_anova:.4f}  ({anova_res['test']})")
    rows.append(dict(experiment=label, comparison="omnibus", **anova_res,
                     sesoi=sesoi))

    # Pairwise TOST
    pairs = [
        ("universal", "moe"),
        ("universal", "gated_moe"),
        ("moe",       "gated_moe"),
    ]
    for m1, m2 in pairs:
        if m1 not in wide.columns or m2 not in wide.columns:
            continue
        res = tost_paired(wide[m1].values, wide[m2].values, sesoi)
        equiv_str = "EQUIV" if res["equivalent"] else "NOT equiv"
        print(f"  TOST {m1} vs {m2}: Δ={res['mean_diff']:+.4f}  "
              f"95%CI=[{res['ci_95_low']:+.4f},{res['ci_95_high']:+.4f}]  "
              f"p_low={res['p_low']:.4f}  p_high={res['p_high']:.4f}  → {equiv_str}")
        rows.append(dict(experiment=label, comparison=f"{m1}_vs_{m2}",
                         sesoi=sesoi, **res))

    return rows


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        # (CSV path,  label,  SESOI)
        (RESULTS_DIR / "classification"        / "per_seed_f1.csv",
         "in_domain", SESOI_LOW),
        (RESULTS_DIR / "cross_country"         / "per_seed_f1.csv",
         "cross_country_FR→ES", SESOI_HIGH),
        (RESULTS_DIR / "cross_country"         / "per_seed_f1.csv",
         "cross_country_ES→FR", SESOI_HIGH),
        (RESULTS_DIR / "drought_robustness"    / "per_seed_f1.csv",
         "drought_2023", SESOI_HIGH),
    ]

    all_rows = []
    for csv_path, label, sesoi in experiments:
        print(f"\n{'='*60}\n  {label}  (SESOI=±{sesoi})\n{'='*60}")

        # For cross-country, filter by direction
        if label.startswith("cross_country"):
            direction = label.split("_", 2)[-1].replace("→", "→")
            try:
                df_raw = pd.read_csv(csv_path)
                if "direction" in df_raw.columns:
                    df_raw = df_raw[df_raw["direction"] == direction]
                    tmp_path = ANALYSIS_DIR / f"tmp_{label}.csv"
                    df_raw.to_csv(tmp_path, index=False)
                    rows = run_experiment(tmp_path, label, sesoi)
                    tmp_path.unlink(missing_ok=True)
                else:
                    rows = run_experiment(csv_path, label, sesoi)
            except Exception as e:
                print(f"  Error filtering direction: {e}")
                rows = []
        elif label == "drought_2023":
            try:
                df_raw = pd.read_csv(csv_path)
                if "year" in df_raw.columns:
                    df_raw = df_raw[df_raw["year"] == 2023]
                    tmp_path = ANALYSIS_DIR / "tmp_drought.csv"
                    df_raw.to_csv(tmp_path, index=False)
                    rows = run_experiment(tmp_path, label, sesoi)
                    tmp_path.unlink(missing_ok=True)
                else:
                    rows = run_experiment(csv_path, label, sesoi)
            except Exception as e:
                print(f"  Error filtering drought year: {e}")
                rows = []
        else:
            rows = run_experiment(csv_path, label, sesoi)

        all_rows.extend(rows)

    if all_rows:
        out = pd.DataFrame(all_rows)
        out.to_csv(ANALYSIS_DIR / "all_tests.csv", index=False)
        print(f"\n\nAll results written to {ANALYSIS_DIR}/all_tests.csv")

        # Summary: equivalence declaration per comparison
        tost_rows = out[out["comparison"] != "omnibus"].copy()
        if "equivalent" in tost_rows.columns:
            equiv_summary = tost_rows.groupby(["experiment", "comparison"])[
                ["mean_diff", "ci_95_low", "ci_95_high", "equivalent"]
            ].first()
            print("\n=== Equivalence summary ===")
            print(equiv_summary.to_string())
    else:
        print("\nNo results produced — run experiment scripts first.")


if __name__ == "__main__":
    main()
