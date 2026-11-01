"""
Step 2b: Statistical analysis of the clean crop dataset.
Run from the project root:  python analyze_data.py
All results are saved as CSV/JSON files in outputs/.
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.oneway import anova_oneway

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "Crop_clean.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
ALPHA = 0.05
NORMALITY_SHARE_REQUIRED = 0.80  # practical rule, documented in the report

NAMES = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Relative humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}


# ---------------------------------------------------------------- helpers
def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def eta_squared(samples) -> float:
    """eta^2 = SS_between / SS_total."""
    allv = np.concatenate(samples)
    gm = allv.mean()
    ss_between = sum(len(s) * (s.mean() - gm) ** 2 for s in samples)
    ss_total = ((allv - gm) ** 2).sum()
    return float(ss_between / ss_total)


def eta_squared_h(h: float, k: int, n: int) -> float:
    """Kruskal-Wallis effect size: eta^2_H = (H - k + 1) / (n - k)."""
    return float(max((h - k + 1) / (n - k), 0.0))


def effect_label(e: float) -> str:
    if e < 0.01:
        return "negligible"
    if e < 0.06:
        return "small"
    if e < 0.14:
        return "medium"
    return "large"


def corr_strength(r: float) -> str:
    a = abs(r)
    if a < 0.1:
        return "negligible"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "strong"


def main() -> None:
    if not CLEAN_PATH.exists():
        print("ERROR: run clean_data.py first (data/Crop_clean.csv not found).")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)
    crops = sorted(df[TARGET].unique())
    samples = {f: [df.loc[df[TARGET] == c, f].to_numpy(dtype=float) for c in crops]
               for f in FEATURES}
    n_total, k = len(df), len(crops)
    print(f"Loaded {n_total} rows, {k} crops, {len(FEATURES)} features.\n")

    # ------------------------------------------- 1. Descriptive statistics
    desc = df[FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(columns={"25%": "q1", "50%": "median", "75%": "q3"})
    desc["iqr"] = desc["q3"] - desc["q1"]
    desc["skewness"] = df[FEATURES].skew()
    desc["excess_kurtosis"] = df[FEATURES].kurt()
    desc["cv_percent"] = desc["std"] / desc["mean"] * 100
    desc.index.name = "feature"
    desc = desc[["count", "mean", "median", "std", "min", "q1", "q3", "max",
                 "iqr", "skewness", "excess_kurtosis", "cv_percent"]]
    desc.round(4).to_csv(OUT_DIR / "statistical_summary.csv")
    print("1. DESCRIPTIVE STATISTICS (all crops pooled)")
    print(desc.round(2).to_string(), "\n")

    agg = df.groupby(TARGET)[FEATURES].agg(["mean", "median", "std", "min", "max"])
    agg.columns = [f"{f}_{s}" for f, s in agg.columns]
    agg.round(4).to_csv(OUT_DIR / "crop_wise_summary.csv")
    print("   Crop-wise means:")
    print(df.groupby(TARGET)[FEATURES].mean().round(2).to_string(), "\n")

    # ------------------------------------------------ 2. Correlation
    pearson = df[FEATURES].corr(method="pearson")
    spearman = df[FEATURES].corr(method="spearman")
    pearson.round(4).to_csv(OUT_DIR / "correlation_pearson.csv")
    spearman.round(4).to_csv(OUT_DIR / "correlation_spearman.csv")

    rows = []
    for a, b in combinations(FEATURES, 2):
        r, p_r = stats.pearsonr(df[a], df[b])
        rho, p_s = stats.spearmanr(df[a], df[b])
        rows.append({
            "feature_1": a, "feature_2": b,
            "pearson_r": r, "pearson_p": p_r,
            "spearman_rho": rho, "spearman_p": p_s,
            "strength": corr_strength(r),
            "direction": "positive" if r > 0 else "negative",
        })
    pairs = pd.DataFrame(rows)
    pairs = pairs.reindex(pairs["pearson_r"].abs().sort_values(ascending=False).index)
    pairs.round(6).to_csv(OUT_DIR / "correlation_pairs.csv", index=False)
    print("2. STRONGEST CORRELATIONS (pooled data, sorted by |Pearson r|)")
    print(pairs[["feature_1", "feature_2", "pearson_r", "spearman_rho", "strength"]]
          .head(8).round(3).to_string(index=False), "\n")

    # ---------------------------------------------------- 3. Outliers
    out_rows = []
    for f in FEATURES:
        q1, q3 = df[f].quantile(0.25), df[f].quantile(0.75)
        iqr = q3 - q1
        pooled = int(((df[f] < q1 - 1.5 * iqr) | (df[f] > q3 + 1.5 * iqr)).sum())
        g = df.groupby(TARGET)[f]
        gq1 = g.transform(lambda s: s.quantile(0.25))
        gq3 = g.transform(lambda s: s.quantile(0.75))
        giqr = gq3 - gq1
        within = int(((df[f] < gq1 - 1.5 * giqr) | (df[f] > gq3 + 1.5 * giqr)).sum())
        out_rows.append({"feature": f, "outliers_pooled_iqr": pooled,
                         "outliers_within_crop_iqr": within, "n": n_total})
    outliers = pd.DataFrame(out_rows)
    outliers.to_csv(OUT_DIR / "outlier_summary.csv", index=False)
    print("3. OUTLIERS (1.5 x IQR rule; kept in the data, only counted)")
    print(outliers.to_string(index=False), "\n")

    # ----------------------------------------- 4. ANOVA assumption checks
    assume_rows = []
    for f in FEATURES:
        sh_p = [stats.shapiro(s).pvalue for s in samples[f]]
        n_normal = int(sum(p >= ALPHA for p in sh_p))
        lev_stat, lev_p = stats.levene(*samples[f], center="median")
        variances = [s.var(ddof=1) for s in samples[f]]
        assume_rows.append({
            "feature": f,
            "groups_passing_shapiro": n_normal,
            "groups_total": k,
            "share_normal": n_normal / k,
            "normality_ok": (n_normal / k) >= NORMALITY_SHARE_REQUIRED,
            "levene_stat": lev_stat, "levene_p": lev_p,
            "equal_variance_ok": lev_p >= ALPHA,
            "max_to_min_variance_ratio": max(variances) / min(variances),
        })
    assume = pd.DataFrame(assume_rows)
    assume.round(6).to_csv(OUT_DIR / "anova_assumptions.csv", index=False)
    print("4. ASSUMPTION CHECKS (Shapiro-Wilk per crop, Levene across crops)")
    print(assume[["feature", "groups_passing_shapiro", "groups_total",
                  "levene_p", "max_to_min_variance_ratio"]]
          .round(4).to_string(index=False), "\n")

    # ---------------------------------------------------- 5. ANOVA family
    bonf_alpha = ALPHA / len(FEATURES)
    res_rows = []
    for f in FEATURES:
        s = samples[f]
        f_stat, p_f = stats.f_oneway(*s)
        w = anova_oneway(s, use_var="unequal", welch_correction=True)
        h_stat, p_h = stats.kruskal(*s)
        eta2 = eta_squared(s)
        eta2_h = eta_squared_h(h_stat, k, n_total)

        a = assume.loc[assume["feature"] == f].iloc[0]
        if a["normality_ok"] and a["equal_variance_ok"]:
            primary, p_primary = "One-way ANOVA", p_f
        elif a["normality_ok"]:
            primary, p_primary = "Welch's ANOVA", float(w.pvalue)
        else:
            primary, p_primary = "Kruskal-Wallis", p_h

        sig = bool(p_primary < ALPHA)
        sig_b = bool(p_primary < bonf_alpha)
        eta_primary = eta2_h if primary == "Kruskal-Wallis" else eta2
        effect = effect_label(eta_primary)
        verdict = "reject H0" if sig_b else "fail to reject H0 (Bonferroni)"
        interp = (
            f"{NAMES[f]}: primary test = {primary} (p = {fmt_p(p_primary)}), "
            f"eta^2 = {eta_primary:.3f} ({effect}); {verdict}."
        )
        res_rows.append({
            "feature": f,
            "H0": f"Mean {NAMES[f]} is equal across all {k} crops.",
            "H1": "At least one crop has a different mean.",
            "f_classical": f_stat, "p_classical": p_f,
            "f_welch": float(w.statistic), "p_welch": float(w.pvalue),
            "h_kruskal": h_stat, "p_kruskal": p_h,
            "eta_squared": eta2, "eta_squared_H": eta2_h,
            "eta_squared_primary": eta_primary,
            "effect_size": effect,
            "primary_test": primary, "primary_p": p_primary,
            "alpha": ALPHA, "bonferroni_alpha": bonf_alpha,
            "significant_at_0.05": sig, "significant_bonferroni": sig_b,
            "interpretation": interp,
        })
    anova = pd.DataFrame(res_rows)

    # ------------------------------------- 6. Tukey HSD post-hoc (pairwise)
    tukey_rows = []
    sig_pairs = {}
    for f in FEATURES:
        t = stats.tukey_hsd(*samples[f])
        count = 0
        for i, j in combinations(range(k), 2):
            p = float(t.pvalue[i, j])
            count += p < ALPHA
            tukey_rows.append({
                "feature": f, "crop_a": crops[i], "crop_b": crops[j],
                "mean_diff_a_minus_b": float(samples[f][i].mean() - samples[f][j].mean()),
                "p_adjusted": p, "significant": p < ALPHA,
            })
        sig_pairs[f] = int(count)
    n_pairs = k * (k - 1) // 2
    anova["tukey_significant_pairs"] = anova["feature"].map(sig_pairs)
    anova["tukey_total_pairs"] = n_pairs
    anova["tukey_note"] = np.where(
        anova["feature"].map(assume.set_index("feature")["equal_variance_ok"]),
        "equal variances: Tukey appropriate",
        "unequal variances: Tukey is exploratory (Games-Howell preferred)",
    )
    anova.round(8).to_csv(OUT_DIR / "anova_results.csv", index=False)
    pd.DataFrame(tukey_rows).round(8).to_csv(OUT_DIR / "tukey_pairwise.csv", index=False)

    print(f"5. ONE-WAY ANOVA FAMILY (alpha = {ALPHA}, Bonferroni alpha = {bonf_alpha:.4f})")
    show = anova[["feature", "f_classical", "p_classical", "f_welch", "p_welch",
                  "h_kruskal", "p_kruskal", "eta_squared", "effect_size",
                  "primary_test"]].copy()
    for c in ["p_classical", "p_welch", "p_kruskal"]:
        show[c] = show[c].map(fmt_p)
    print(show.round(3).to_string(index=False), "\n")
    for line in anova["interpretation"]:
        print("  -", line)
    print("\n6. TUKEY HSD: significant pairs out of", n_pairs)
    print(anova[["feature", "tukey_significant_pairs", "tukey_note"]]
          .to_string(index=False), "\n")

    # --------------------------------------------------------- 7. Summary
    summary = {
        "n_rows": n_total, "n_crops": k, "features": FEATURES, "alpha": ALPHA,
        "bonferroni_alpha": bonf_alpha,
        "normality_share_required": NORMALITY_SHARE_REQUIRED,
        "files_written": sorted(p.name for p in OUT_DIR.glob("*.csv")),
    }
    with open(OUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("DONE. Result files saved in:", OUT_DIR)


if __name__ == "__main__":
    main()