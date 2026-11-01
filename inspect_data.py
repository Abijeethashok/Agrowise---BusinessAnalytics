"""
Step 1: Inspect the raw crop recommendation dataset.
Run from the project root:  python inspect_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

EXPECTED_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    # ---- 1. Load -----------------------------------------------------------
    if not DATA_PATH.exists():
        print(f"ERROR: dataset not found at:\n  {DATA_PATH}")
        print("Download it manually from the Figshare page and save it there.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    report = {"file": str(DATA_PATH.name)}

    # ---- 2. Shape and columns ---------------------------------------------
    section("1. SHAPE AND COLUMNS")
    print(f"Rows: {df.shape[0]}   Columns: {df.shape[1]}")
    print("Column names (repr shows hidden spaces):")
    for c in df.columns:
        print(f"  {c!r:20} dtype={df[c].dtype}")
    report["rows"] = int(df.shape[0])
    report["columns"] = list(df.columns)

    missing_expected = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_expected:
        print(f"\nWARNING: expected feature columns not found: {missing_expected}")
    report["missing_expected_columns"] = missing_expected

    # ---- 3. Preview --------------------------------------------------------
    section("2. FIRST 5 ROWS")
    print(df.head().to_string())

    # ---- 4. Missing values -------------------------------------------------
    section("3. MISSING VALUES")
    missing = df.isna().sum()
    print(missing.to_string())
    print(f"Total missing cells: {int(missing.sum())}")
    report["missing_values"] = {k: int(v) for k, v in missing.items()}

    # ---- 5. Duplicates -----------------------------------------------------
    section("4. DUPLICATE ROWS")
    n_dup = int(df.duplicated().sum())
    print(f"Fully duplicated rows: {n_dup}")
    report["duplicate_rows"] = n_dup

    # ---- 6. Target label ---------------------------------------------------
    section("5. TARGET (CROP LABEL)")
    candidates = [
        c for c in df.columns
        if c not in EXPECTED_FEATURES and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(candidates) != 1:
        print(f"Could not identify a single label column. Candidates: {candidates}")
        report["label_column"] = None
    else:
        label = candidates[0]
        report["label_column"] = label
        print(f"Label column detected: {label!r}")

        raw_unique = df[label].dropna().unique()
        cleaned_unique = {str(v).strip().lower() for v in raw_unique}
        print(f"Distinct labels as stored: {len(raw_unique)}")
        print(f"Distinct labels after strip/lowercase: {len(cleaned_unique)}")
        if len(raw_unique) != len(cleaned_unique):
            print("WARNING: labels differ only by case/spaces and need cleaning.")

        counts = df[label].value_counts()
        print("\nRecords per crop:")
        print(counts.to_string())
        print(f"\nSmallest class: {int(counts.min())}   Largest class: {int(counts.max())}")
        report["n_classes"] = int(counts.shape[0])
        report["class_counts"] = {str(k): int(v) for k, v in counts.items()}

    # ---- 7. Numeric summary ------------------------------------------------
    section("6. NUMERIC SUMMARY")
    present = [c for c in EXPECTED_FEATURES if c in df.columns]
    print(df[present].describe().round(2).to_string())

    # ---- 8. Plausibility checks -------------------------------------------
    section("7. SIMPLE PLAUSIBILITY CHECKS")
    checks = {}
    if "ph" in df.columns:
        checks["ph outside 0-14"] = int(((df["ph"] < 0) | (df["ph"] > 14)).sum())
    if "humidity" in df.columns:
        checks["humidity outside 0-100"] = int(
            ((df["humidity"] < 0) | (df["humidity"] > 100)).sum()
        )
    for col in ["N", "P", "K", "rainfall"]:
        if col in df.columns:
            checks[f"{col} negative"] = int((df[col] < 0).sum())
    for name, n in checks.items():
        print(f"{name:28} -> {n} rows")
    report["plausibility_violations"] = checks

    # ---- 9. Save report ----------------------------------------------------
    out_path = OUT_DIR / "data_inspection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    section("DONE")
    print(f"Inspection summary saved to: {out_path}")


if __name__ == "__main__":
    main()