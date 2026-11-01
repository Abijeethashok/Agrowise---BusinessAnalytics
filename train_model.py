"""
Step 3: Train, compare, evaluate and save the crop recommendation models.
Run from the project root:  python train_model.py

Outputs
  models/trained_model.joblib
  models/preprocessing_artifacts.joblib
  models/model_metadata.json
  outputs/model_comparison.csv, cv_fold_metrics.csv, classification_report.csv,
          confusion_matrix.csv, confused_pairs.csv, feature_importance.csv,
          robustness_check.csv, training_summary.json
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score,
                             top_k_accuracy_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------ config
BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "Crop_clean.csv"
MODEL_DIR = BASE_DIR / "models"
OUT_DIR = BASE_DIR / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
TIE_MARGIN = 0.005          # practical-equivalence margin on CV macro-F1
NOISE_LEVELS = [0.0, 0.10, 0.25, 0.50]   # fraction of each feature's std

METRICS = ["accuracy", "precision_macro", "recall_macro",
           "f1_macro", "f1_weighted", "top3_accuracy", "log_loss"]

FEATURE_LABELS = {
    "N": "Nitrogen (N, dataset ratio)",
    "P": "Phosphorus (P, dataset ratio)",
    "K": "Potassium (K, dataset ratio)",
    "temperature": "Temperature (°C)",
    "humidity": "Relative humidity (%)",
    "ph": "Soil pH",
    "rainfall": "Rainfall (mm)",
}
# Hard physical limits, used only to keep noisy test inputs physically valid
PHYSICAL_BOUNDS = {
    "N": (0, None), "P": (0, None), "K": (0, None), "rainfall": (0, None),
    "humidity": (0, 100), "ph": (0, 14), "temperature": (None, None),
}
SPECIAL_NAMES = {
    "kidneybeans": "Kidney Beans", "mothbeans": "Moth Beans",
    "mungbean": "Mung Bean", "blackgram": "Black Gram",
    "pigeonpeas": "Pigeon Peas", "chickpea": "Chickpea",
}


def display_name(crop: str) -> str:
    return SPECIAL_NAMES.get(crop, crop.title())


def build_models() -> dict:
    # No feature scaling: decision trees, random forests and Gaussian NB are
    # unaffected by monotonic rescaling of individual features.
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(model, X, y_true, classes) -> dict:
    proba = model.predict_proba(X)
    if list(model.classes_) != list(classes):
        raise RuntimeError("Class order mismatch between model and label set.")
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision_macro": precision_score(y_true, pred, average="macro",
                                           zero_division=0),
        "recall_macro": recall_score(y_true, pred, average="macro",
                                     zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted",
                                zero_division=0),
        "top3_accuracy": top_k_accuracy_score(y_true, proba, k=3,
                                              labels=list(classes)),
        "log_loss": log_loss(y_true, proba, labels=list(classes)),
    }


def section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def main() -> None:
    if not CLEAN_PATH.exists():
        print("ERROR: data/Crop_clean.csv not found. Run clean_data.py first.")
        sys.exit(1)

    df = pd.read_csv(CLEAN_PATH)
    X = df[FEATURES].astype(float)
    y = df[TARGET].astype(str).to_numpy()
    classes = sorted(set(y))
    print(f"Loaded {len(df)} rows, {len(classes)} classes, "
          f"{len(FEATURES)} features. scikit-learn {sklearn.__version__}")

    # ------------------------------------------------ 1. stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    section("1. STRATIFIED TRAIN/TEST SPLIT")
    print(f"Train rows: {len(X_train)}   Test rows: {len(X_test)}")
    print(f"Test rows per class (min/max): "
          f"{pd.Series(y_test).value_counts().min()}/"
          f"{pd.Series(y_test).value_counts().max()}")
    baseline = 1 / len(classes)
    print(f"Random-guess accuracy baseline (1/{len(classes)}): {baseline:.4f}")

    # ------------------------------ 2. cross-validation on TRAINING data only
    section(f"2. {N_SPLITS}-FOLD STRATIFIED CROSS-VALIDATION (training set only)")
    models = build_models()
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)
    fold_rows = []
    for name, proto in models.items():
        for fold, (tr, va) in enumerate(skf.split(X_train, y_train), start=1):
            m = clone(proto).fit(X_train.iloc[tr], y_train[tr])
            scores = evaluate(m, X_train.iloc[va], y_train[va], classes)
            fold_rows.append({"model": name, "fold": fold, **scores})
        print(f"  finished CV for {name}")
    cv_df = pd.DataFrame(fold_rows)
    cv_df.round(6).to_csv(OUT_DIR / "cv_fold_metrics.csv", index=False)
    cv_mean = cv_df.groupby("model")[METRICS].mean()
    cv_std = cv_df.groupby("model")[METRICS].std(ddof=1)

    # ----------------------------------------- 3. model selection (CV only)
    best_f1 = cv_mean["f1_macro"].max()
    tied = [m for m in cv_mean.index
            if best_f1 - cv_mean.loc[m, "f1_macro"] <= TIE_MARGIN]
    selected = min(tied, key=lambda m: cv_mean.loc[m, "log_loss"])
    rule = (f"Highest mean CV macro-F1; models within {TIE_MARGIN} of the best "
            f"are treated as tied, and the tie is broken by lowest mean CV "
            f"log-loss (better probability estimates for ranking crops). "
            f"The test set is not used for selection.")
    section("3. MODEL SELECTION (based on cross-validation only)")
    print(rule)
    print(f"Models considered tied on macro-F1: {tied}")
    print(f"SELECTED MODEL: {selected}")

    # ------------------------------------ 4. refit on training set, test set
    section("4. HELD-OUT TEST SET EVALUATION (all models, trained on train split)")
    fitted, comp_rows = {}, []
    for name, proto in models.items():
        t0 = time.time()
        m = clone(proto).fit(X_train, y_train)
        fit_sec = time.time() - t0
        fitted[name] = m
        test_scores = evaluate(m, X_test, y_test, classes)
        row = {"model": name}
        for k in METRICS:
            row[f"cv_{k}_mean"] = cv_mean.loc[name, k]
            row[f"cv_{k}_std"] = cv_std.loc[name, k]
            row[f"test_{k}"] = test_scores[k]
        row["fit_seconds"] = fit_sec
        row["selected"] = name == selected
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows)
    comp.round(6).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    ci = comp.set_index("model")
    show = pd.DataFrame({
        "cv_f1_macro": [f"{cv_mean.loc[r, 'f1_macro']:.4f} ± {cv_std.loc[r, 'f1_macro']:.4f}"
                        for r in ci.index],
        "cv_logloss": cv_mean.loc[ci.index, "log_loss"].round(4),
        "test_acc": ci["test_accuracy"].round(4),
        "test_f1_macro": ci["test_f1_macro"].round(4),
        "test_f1_wtd": ci["test_f1_weighted"].round(4),
        "test_top3": ci["test_top3_accuracy"].round(4),
        "test_logloss": ci["test_log_loss"].round(4),
    }, index=ci.index)
    print(show.to_string())

    # ------------------------------------ 5. selected model: error analysis
    model = fitted[selected]
    pred = model.predict(X_test)
    section(f"5. ERROR ANALYSIS FOR SELECTED MODEL: {selected}")
    rep = pd.DataFrame(classification_report(
        y_test, pred, labels=classes, output_dict=True, zero_division=0)).T
    rep.round(4).to_csv(OUT_DIR / "classification_report.csv")
    cm = confusion_matrix(y_test, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        OUT_DIR / "confusion_matrix.csv")
    pairs = [{"true_crop": classes[i], "predicted_crop": classes[j],
              "count": int(cm[i, j])}
             for i in range(len(classes)) for j in range(len(classes))
             if i != j and cm[i, j] > 0]
    pairs_df = pd.DataFrame(pairs, columns=["true_crop", "predicted_crop", "count"]
                            ).sort_values("count", ascending=False)
    pairs_df.to_csv(OUT_DIR / "confused_pairs.csv", index=False)
    n_err = int(len(y_test) - np.trace(cm))
    print(f"Misclassified test rows: {n_err} of {len(y_test)}")
    if n_err:
        print(pairs_df.head(10).to_string(index=False))
    else:
        print("No misclassifications on the test set.")

    # ------------------------------------ 6. permutation feature importance
    section("6. PERMUTATION FEATURE IMPORTANCE (test set, drop in accuracy)")
    perm = permutation_importance(model, X_test, y_test, n_repeats=20,
                                  random_state=RANDOM_STATE, scoring="accuracy")
    imp = pd.DataFrame({
        "feature": FEATURES,
        "perm_importance_mean": perm.importances_mean,
        "perm_importance_std": perm.importances_std,
    })
    if hasattr(model, "feature_importances_"):
        imp["impurity_importance"] = model.feature_importances_
    imp = imp.sort_values("perm_importance_mean", ascending=False)
    imp.round(6).to_csv(OUT_DIR / "feature_importance.csv", index=False)
    print(imp.round(4).to_string(index=False))

    # -------------------------------------------- 7. noise robustness check
    section("7. ROBUSTNESS: TEST ACCURACY WHEN NOISE IS ADDED TO INPUTS")
    feat_std = X_train.std().to_numpy()
    rob_rows = []
    for level in NOISE_LEVELS:
        rng = np.random.default_rng(RANDOM_STATE)
        noise = rng.normal(0.0, 1.0, size=X_test.shape) * feat_std * level
        Xn = X_test.copy() + noise
        for f in FEATURES:
            lo, hi = PHYSICAL_BOUNDS[f]
            Xn[f] = Xn[f].clip(lower=lo, upper=hi)
        for name, m in fitted.items():
            p = m.predict(Xn)
            rob_rows.append({
                "noise_level_x_feature_std": level, "model": name,
                "accuracy": accuracy_score(y_test, p),
                "f1_macro": f1_score(y_test, p, average="macro",
                                     zero_division=0)})
    rob = pd.DataFrame(rob_rows)
    rob.round(6).to_csv(OUT_DIR / "robustness_check.csv", index=False)
    print(rob.pivot(index="noise_level_x_feature_std", columns="model",
                    values="accuracy").round(4).to_string())

    # ------------------------------------------------------- 8. save model
    section("8. SAVING ARTIFACTS")
    joblib.dump(model, MODEL_DIR / "trained_model.joblib")
    artifacts = {
        "features": FEATURES,
        "feature_labels": FEATURE_LABELS,
        "classes": list(model.classes_),
        "crop_display_names": {c: display_name(c) for c in classes},
        "feature_ranges": {f: {"min": float(df[f].min()),
                               "max": float(df[f].max())} for f in FEATURES},
        "selected_model": selected,
        "selection_rule": rule,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_train_rows": int(len(X_train)),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    joblib.dump(artifacts, MODEL_DIR / "preprocessing_artifacts.joblib")
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(artifacts, fh, indent=2)

    summary = {
        "selected_model": selected, "selection_rule": rule,
        "models_tied_on_f1": tied,
        "random_guess_baseline_accuracy": baseline,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "cv_folds": N_SPLITS, "tie_margin": TIE_MARGIN,
        "test_misclassified": n_err,
        "selected_test_metrics": {k: float(comp.loc[comp['selected'], f'test_{k}'].iloc[0])
                                  for k in METRICS},
    }
    with open(OUT_DIR / "training_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved model and metadata to: {MODEL_DIR}")

    # ------------------------- 9. reload check (artifacts load, consistent)
    section("9. RELOAD CHECK")
    loaded = joblib.load(MODEL_DIR / "trained_model.joblib")
    art = joblib.load(MODEL_DIR / "preprocessing_artifacts.joblib")
    same = np.array_equal(loaded.predict(X_test), pred)
    print(f"Reloaded model reproduces test predictions exactly: {same}")
    sample = pd.DataFrame([X_test.iloc[0].to_dict()], columns=art["features"])
    p1 = loaded.predict_proba(sample)[0]
    p2 = loaded.predict_proba(sample)[0]
    print(f"Repeated prediction identical: {np.array_equal(p1, p2)}")
    top = np.argsort(p1)[::-1][:3]
    print("Smoke test on first test row "
          f"(true crop: {y_test[0]}): top-3 = "
          + ", ".join(f"{art['classes'][i]} ({p1[i]:.3f})" for i in top))
    if not same:
        print("WARNING: reload check failed; do not continue until fixed.")
    print("\nDONE.")


if __name__ == "__main__":
    main()