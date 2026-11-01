"""
Step 2a: Clean the raw crop dataset.
Run from the project root:  python clean_data.py
Output: data/Crop_clean.csv and outputs/cleaning_log.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
CLEAN_PATH = BASE_DIR / "data" / "Crop_clean.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"


def main() -> None:
    if not RAW_PATH.exists():
        print(f"ERROR: raw dataset not found at {RAW_PATH}")
        sys.exit(1)

    df = pd.read_csv(RAW_PATH)
    log = {"rows_raw": int(len(df))}

    # 1. Standardise column names and labels
    df.columns = [c.strip() for c in df.columns]
    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        print(f"ERROR: missing required columns: {missing_cols}")
        sys.exit(1)
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower()

    # 2. Force numeric features (non-numeric becomes NaN and is then counted)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Missing values: drop and log (nothing is imputed)
    rows_with_missing = int(df.isna().any(axis=1).sum())
    df = df.dropna()
    log["rows_dropped_missing"] = rows_with_missing

    # 4. Exact duplicates
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    log["rows_dropped_duplicate"] = n_dup

    # 5. Physically impossible values
    valid = (
        df["ph"].between(0, 14)
        & df["humidity"].between(0, 100)
        & df["temperature"].between(-10, 60)
        & (df[["N", "P", "K", "rainfall"]] >= 0).all(axis=1)
    )
    log["rows_dropped_invalid_range"] = int((~valid).sum())
    df = df[valid].reset_index(drop=True)

    # 6. Save
    log["rows_clean"] = int(len(df))
    log["n_classes"] = int(df[TARGET].nunique())
    log["class_counts"] = {k: int(v) for k, v in df[TARGET].value_counts().items()}
    df.to_csv(CLEAN_PATH, index=False)
    with open(OUT_DIR / "cleaning_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print("CLEANING SUMMARY")
    for k in ["rows_raw", "rows_dropped_missing", "rows_dropped_duplicate",
              "rows_dropped_invalid_range", "rows_clean", "n_classes"]:
        print(f"  {k:28} {log[k]}")
    print(f"\nClean data saved to: {CLEAN_PATH}")
    print(f"Cleaning log saved to: {OUT_DIR / 'cleaning_log.json'}")


if __name__ == "__main__":
    main()