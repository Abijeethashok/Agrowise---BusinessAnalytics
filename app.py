"""
AgroWise Flask backend (Step 5).

Run (development):   python app.py
Then open:           http://127.0.0.1:5000/api/health

The trained model is loaded ONCE at startup; requests never retrain anything.
"""
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from recommender import FEATURES, HARD_LIMITS, InputError, Recommender

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024      # 16 KB request bodies
app.json.sort_keys = False
app.json.ensure_ascii = False

# ---------------------------------------------------------------- model state
STATE = {"rec": None, "error": None}


def load_recommender() -> None:
    try:
        STATE["rec"] = Recommender(BASE_DIR)
        STATE["error"] = None
    except Exception as exc:        # reported via /api/health instead of crashing
        STATE["rec"] = None
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        app.logger.error("Model failed to load: %s", STATE["error"])


load_recommender()


class ModelUnavailable(Exception):
    pass


def require_model() -> Recommender:
    if STATE["rec"] is None:
        raise ModelUnavailable(STATE["error"] or "Model is not loaded.")
    return STATE["rec"]


# ------------------------------------------------------------------ helpers
FEATURE_INFO = {
    "N": {"unit": "dataset ratio", "step": 1,
          "help": "Nitrogen content of the soil, given as a unitless ratio in the "
                  "source dataset. It is not kg/ha or ppm."},
    "P": {"unit": "dataset ratio", "step": 1,
          "help": "Phosphorus content of the soil, given as a unitless ratio in the "
                  "source dataset. It is not kg/ha or ppm."},
    "K": {"unit": "dataset ratio", "step": 1,
          "help": "Potassium content of the soil, given as a unitless ratio in the "
                  "source dataset. It is not kg/ha or ppm."},
    "temperature": {"unit": "°C", "step": 0.1,
                    "help": "Temperature in degrees Celsius. The source does not "
                            "state the averaging period; enter values on a "
                            "comparable basis."},
    "humidity": {"unit": "%", "step": 0.1,
                 "help": "Relative humidity in percent (0 to 100)."},
    "ph": {"unit": "pH", "step": 0.01,
           "help": "Soil pH on the 0 to 14 scale (below 7 acidic, above 7 alkaline)."},
    "rainfall": {"unit": "mm", "step": 0.1,
                 "help": "Rainfall in millimetres. The source does not state whether "
                         "this is annual, seasonal or monthly; enter values on a "
                         "comparable basis."},
}
SAMPLE_DECIMALS = {"N": 0, "P": 0, "K": 0, "temperature": 1, "humidity": 1,
                   "ph": 2, "rainfall": 1}
PROFILE_STATS = ["mean", "sd", "min", "p05", "q1", "median", "q3", "p95", "max"]

# name -> (file in outputs/, kind)
TABLE_SPECS = {
    "descriptive": ("statistical_summary.csv", "table"),
    "crop_summary": ("crop_wise_summary.csv", "table"),
    "profiles": ("crop_profiles.csv", "table"),
    "correlation_pairs": ("correlation_pairs.csv", "table"),
    "outliers": ("outlier_summary.csv", "table"),
    "anova": ("anova_results.csv", "table"),
    "anova_assumptions": ("anova_assumptions.csv", "table"),
    "model_comparison": ("model_comparison.csv", "table"),
    "cv_folds": ("cv_fold_metrics.csv", "table"),
    "classification_report": ("classification_report.csv", "indexed"),
    "confused_pairs": ("confused_pairs.csv", "table"),
    "feature_importance": ("feature_importance.csv", "table"),
    "robustness": ("robustness_check.csv", "table"),
    "correlation_pearson": ("correlation_pearson.csv", "matrix"),
    "correlation_spearman": ("correlation_spearman.csv", "matrix"),
    "confusion_matrix": ("confusion_matrix.csv", "matrix"),
}
SPECIAL_NAMES = ["overview", "crop_distribution", "histograms"]
ANALYTICS_NAMES = sorted(list(TABLE_SPECS) + SPECIAL_NAMES)

REQUIRED_OUTPUTS = sorted({f for f, _ in TABLE_SPECS.values()} | {
    "cleaning_log.json", "data_inspection.json", "training_summary.json"})


def api_error(status: int, code: str, message: str, **extra):
    return jsonify({"error": code, "message": message, **extra}), status


def load_json_file(name: str):
    path = OUT_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_table(path: Path, kind: str) -> dict:
    if kind == "matrix":
        df = pd.read_csv(path, index_col=0)
        return {"row_labels": [str(i) for i in df.index],
                "col_labels": [str(c) for c in df.columns],
                "matrix": json.loads(df.to_json(orient="values"))}
    if kind == "indexed":
        df = pd.read_csv(path, index_col=0).reset_index(names="name")
    else:
        df = pd.read_csv(path)
    return {"columns": list(df.columns),
            "rows": json.loads(df.to_json(orient="records"))}


# ------------------------------------------------------------- error handling
@app.errorhandler(ModelUnavailable)
def handle_model_unavailable(exc):
    return api_error(503, "model_unavailable",
                     "The prediction model is not available. Run train_model.py "
                     "and build_profiles.py, then restart the app.",
                     detail=str(exc))


@app.errorhandler(HTTPException)
def handle_http_error(exc):
    if request.path.startswith("/api/"):
        return api_error(exc.code, exc.name.lower().replace(" ", "_"), exc.description)
    return exc


@app.errorhandler(Exception)
def handle_unexpected(exc):
    app.logger.exception("Unhandled error")
    if request.path.startswith("/api/"):
        return api_error(500, "server_error",
                         "Unexpected server error. See the server log for details.")
    return "Internal server error", 500


# --------------------------------------------------------------------- pages
@app.get("/")
def page_home():
    return render_template("index.html")


@app.get("/recommend")
def page_recommend():
    return render_template("recommend.html")


@app.get("/analytics")
def page_analytics():
    return render_template("analytics.html")


@app.get("/about")
def page_about():
    return render_template("about.html")


# ----------------------------------------------------------------------- API
@app.get("/api/health")
def health():
    rec = STATE["rec"]
    present = {name: (OUT_DIR / name).exists() for name in REQUIRED_OUTPUTS}
    return jsonify({
        "status": "ok" if rec is not None else "degraded",
        "model_loaded": rec is not None,
        "error": STATE["error"],
        "model_name": rec.model_name if rec else None,
        "n_crops": len(rec.classes) if rec else None,
        "features": rec.features if rec else FEATURES,
        "notes": rec.notes if rec else [],
        "sklearn_version": sklearn.__version__,
        "missing_outputs": sorted(n for n, ok in present.items() if not ok),
    })


@app.get("/api/meta")
def meta():
    rec = require_model()
    feats = []
    for f in rec.features:
        info = FEATURE_INFO.get(f, {"unit": "", "step": 1, "help": ""})
        lo, hi = HARD_LIMITS[f]
        feats.append({
            "key": f, "label": rec.labels[f], "unit": info["unit"],
            "help": info["help"], "step": info["step"],
            "hard_min": lo, "hard_max": hi,
            "dataset_min": rec.ranges[f]["min"], "dataset_max": rec.ranges[f]["max"],
        })
    crops = sorted(({"key": c, "display": rec.display[c]} for c in rec.classes),
                   key=lambda x: x["display"])
    return jsonify({
        "model_name": rec.model_name,
        "features": feats,
        "crops": crops,
        "notes": {
            "model_inputs": "All seven soil and weather values above are inputs to the model.",
            "selected_crop": "Your selected crop is not a model input. It is only "
                             "used to assess how well your conditions match that "
                             "crop's records in the dataset.",
        },
        "disclaimer": ("Scores rank similarity to crop records in the dataset. They "
                       "are not probabilities of successful growth or yield."),
    })


@app.get("/api/sample")
def sample():
    rec = require_model()
    arg = request.args.get("crop")
    try:
        key = rec.resolve_crop(arg) if arg else None
    except InputError:
        return api_error(404, "unknown_crop", f"'{arg}' is not a crop in this dataset.")
    if key is None:
        key = "rice" if "rice" in rec.classes else rec.classes[0]
    inputs = {f: round(float(rec.profiles.loc[key, f"{f}_median"]),
                       SAMPLE_DECIMALS.get(f, 2)) for f in rec.features}
    return jsonify({
        "crop": key, "display": rec.display[key], "inputs": inputs,
        "note": f"Median conditions of {rec.display[key]} records in the dataset.",
    })


@app.get("/api/crop/<crop>")
def crop_profile(crop):
    rec = require_model()
    try:
        key = rec.resolve_crop(crop)
    except InputError:
        key = None
    if key is None:
        return api_error(404, "unknown_crop", f"'{crop}' is not a crop in this dataset.")
    row = rec.profiles.loc[key]
    feats = [{"key": f, "label": rec.labels[f],
              **{s: float(row[f"{f}_{s}"]) for s in PROFILE_STATS}}
             for f in rec.features]
    return jsonify({
        "crop": key, "display": rec.display[key],
        "n_records": int(row["n_records"]), "features": feats,
        "note": ("Typical range = 5th to 95th percentile of this crop's dataset "
                 "records. A data-derived reference, not an agronomic optimum."),
    })


@app.post("/api/recommend")
def api_recommend():
    rec = require_model()
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return api_error(400, "bad_request",
                         "Send a JSON object with Content-Type: application/json.")
    inputs = payload.get("inputs", payload)
    if not isinstance(inputs, dict):
        return api_error(400, "bad_request", "'inputs' must be a JSON object.")
    try:
        result = rec.recommend(inputs, payload.get("selected_crop"))
    except InputError as exc:
        return api_error(400, "validation_error", "Some inputs are invalid.",
                         errors=exc.errors)
    return jsonify(result)


# ----------------------------------------------------------------- analytics
@lru_cache(maxsize=1)
def compute_histograms() -> dict:
    path = DATA_DIR / "Crop_clean.csv"
    if not path.exists():
        raise FileNotFoundError("data/Crop_clean.csv")
    df = pd.read_csv(path)
    out = {}
    for f in FEATURES:
        counts, edges = np.histogram(df[f].to_numpy(float), bins=20)
        out[f] = {"edges": [round(float(e), 4) for e in edges],
                  "counts": [int(c) for c in counts]}
    return out


def analytics_overview():
    insp = load_json_file("data_inspection.json")
    clean = load_json_file("cleaning_log.json")
    train = load_json_file("training_summary.json")
    missing = [n for n, v in [("data_inspection.json", insp),
                              ("cleaning_log.json", clean),
                              ("training_summary.json", train)] if v is None]
    if missing:
        return api_error(404, "output_not_found",
                         f"Missing: {', '.join(missing)}. Run the analysis and "
                         f"training scripts first.")
    m = train.get("selected_test_metrics", {})
    return jsonify({
        "name": "overview",
        "rows_raw": clean.get("rows_raw"), "rows_clean": clean.get("rows_clean"),
        "n_crops": clean.get("n_classes"), "n_features": len(FEATURES),
        "missing_cells": sum(insp.get("missing_values", {}).values()),
        "duplicate_rows": insp.get("duplicate_rows"),
        "selected_model": train.get("selected_model"),
        "test_accuracy": m.get("accuracy"), "test_f1_macro": m.get("f1_macro"),
        "test_top3_accuracy": m.get("top3_accuracy"),
        "random_guess_baseline": train.get("random_guess_baseline_accuracy"),
        "n_train": train.get("n_train"), "n_test": train.get("n_test"),
        "cv_folds": train.get("cv_folds"),
        "test_misclassified": train.get("test_misclassified"),
    })


def analytics_crop_distribution():
    clean = load_json_file("cleaning_log.json")
    if clean is None:
        return api_error(404, "output_not_found",
                         "cleaning_log.json not found. Run clean_data.py first.")
    rec = STATE["rec"]
    rows = [{"crop": c, "display": rec.display[c] if rec else c.title(), "count": n}
            for c, n in clean.get("class_counts", {}).items()]
    rows.sort(key=lambda r: r["display"])
    return jsonify({"name": "crop_distribution", "rows": rows})


def analytics_histograms():
    try:
        hist = compute_histograms()
    except FileNotFoundError:
        return api_error(404, "output_not_found",
                         "data/Crop_clean.csv not found. Run clean_data.py first.")
    rec = STATE["rec"]
    labels = rec.labels if rec else {}
    return jsonify({"name": "histograms", "bins": 20,
                    "features": {f: {"label": labels.get(f, f), **h}
                                 for f, h in hist.items()}})


SPECIAL_HANDLERS = {"overview": analytics_overview,
                    "crop_distribution": analytics_crop_distribution,
                    "histograms": analytics_histograms}


@app.get("/api/analytics/<name>")
def analytics(name):
    if name in SPECIAL_HANDLERS:
        return SPECIAL_HANDLERS[name]()
    spec = TABLE_SPECS.get(name)
    if spec is None:
        return api_error(404, "unknown_dataset",
                         f"Unknown analytics dataset '{name}'.",
                         available=ANALYTICS_NAMES)
    filename, kind = spec
    path = OUT_DIR / filename
    if not path.exists():
        return api_error(404, "output_not_found",
                         f"{filename} not found. Run the analysis, training and "
                         f"profile scripts first.")
    payload = read_table(path, kind)
    return jsonify({"name": name, "source_file": filename, **payload})


if __name__ == "__main__":
    debug = os.environ.get("AGROWISE_DEBUG") == "1"
    print("AgroWise running at http://127.0.0.1:5000  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=debug)