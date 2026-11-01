"""
Step 7 test: checks that every analytics endpoint the dashboard uses returns
the columns the charts and tables expect. Run:  python test_dashboard_data.py
"""
from app import app

client = app.test_client()
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


TABLES = {
    "descriptive": ["feature", "mean", "median", "std", "min", "q1", "q3", "max", "iqr", "skewness", "cv_percent"],
    "outliers": ["feature", "outliers_pooled_iqr", "outliers_within_crop_iqr", "n"],
    "correlation_pairs": ["feature_1", "feature_2", "pearson_r", "spearman_rho", "strength"],
    "anova": ["feature", "primary_test", "primary_p", "h_kruskal", "f_welch", "f_classical",
              "eta_squared", "eta_squared_H", "effect_size", "bonferroni_alpha",
              "significant_bonferroni", "tukey_significant_pairs", "tukey_total_pairs"],
    "anova_assumptions": ["feature", "groups_passing_shapiro", "groups_total", "levene_p",
                          "max_to_min_variance_ratio", "normality_ok", "equal_variance_ok"],
    "model_comparison": ["model", "cv_f1_macro_mean", "cv_f1_macro_std", "cv_log_loss_mean", "test_accuracy",
                         "test_f1_macro", "test_top3_accuracy", "test_log_loss", "selected"],
    "robustness": ["noise_level_x_feature_std", "model", "accuracy"],
    "feature_importance": ["feature", "perm_importance_mean", "perm_importance_std"],
    "confused_pairs": ["true_crop", "predicted_crop", "count"],
    "classification_report": ["name", "precision", "recall", "f1-score", "support"],
}
print("=" * 72 + "\nTABLE ENDPOINTS: required columns\n" + "=" * 72)
for name, cols in TABLES.items():
    r = client.get(f"/api/analytics/{name}")
    data = r.get_json()
    missing = [c for c in cols if c not in data.get("columns", [])]
    check(f"{name}", r.status_code == 200 and not missing, f"missing: {missing}" if missing else f"{len(data['rows'])} rows")

print("\n" + "=" * 72 + "\nMATRICES AND SPECIAL ENDPOINTS\n" + "=" * 72)
for name in ["correlation_pearson", "correlation_spearman", "confusion_matrix"]:
    d = client.get(f"/api/analytics/{name}").get_json()
    n_r, n_c = len(d["row_labels"]), len(d["col_labels"])
    ok = n_r == n_c and len(d["matrix"]) == n_r and all(len(row) == n_c for row in d["matrix"])
    check(f"{name} is a square matrix", ok, f"{n_r} x {n_c}")
d = client.get("/api/analytics/confusion_matrix").get_json()
check("confusion matrix has 22 crops", len(d["row_labels"]) == 22)
d = client.get("/api/analytics/crop_distribution").get_json()
check("crop distribution has 22 crops with counts", len(d["rows"]) == 22 and all("count" in r for r in d["rows"]))
d = client.get("/api/analytics/histograms").get_json()
check("histograms cover 7 features, 20 bins",
      len(d["features"]) == 7 and all(len(h["counts"]) == 20 and len(h["edges"]) == 21 for h in d["features"].values()))
d = client.get("/api/analytics/profiles").get_json()
need = [f"{f}_{s}" for f in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"] for s in ("mean", "p05", "p95")]
check("profiles have mean, p05, p95 for every feature", all(c in d["columns"] for c in need) and "crop" in d["columns"])
check("anova has an effect size column the charts can use",
      any(c in client.get("/api/analytics/anova").get_json()["columns"] for c in ("eta_squared_primary", "eta_squared_H")))

print("\n" + "=" * 72 + "\nRECOMMENDATION PAYLOAD FOR THE COMPATIBILITY CHARTS\n" + "=" * 72)
s = client.get("/api/sample?crop=rice").get_json()["inputs"]
r = client.post("/api/recommend", json={**s, "selected_crop": "rice"}).get_json()
check("top-3 rows carry feature_fits for all 7 inputs", all(len(c["feature_fits"]) == 7 for c in r["top3"]))
check("selected crop carries feature_fits", len(r["selected"]["feature_fits"]) == 7)
check("all 22 crops are returned with profile_fit and rank",
      len(r["all_crops"]) == 22 and all("profile_fit" in c and "rank" in c for c in r["all_crops"]))
check("feature_fits are between 0 and 1", all(0 <= v <= 1 for c in r["top3"] for v in c["feature_fits"].values()))

print("\n" + "=" * 72)
print(f"{sum(results)} of {len(results)} checks passed.")
if not all(results):
    print("Some checks FAILED. Fix them before continuing.")