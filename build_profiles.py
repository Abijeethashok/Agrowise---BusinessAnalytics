"""
Step 4a: Build per-crop reference profiles from the clean dataset.
Run from the project root:  python build_profiles.py
Output: outputs/crop_profiles.csv
"""
import sys
from pathlib import Path

import pandas as pd

from recommender import FEATURES, TARGET, compute_profiles

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "Crop_clean.csv"
OUT_PATH = BASE_DIR / "outputs" / "crop_profiles.csv"


def main() -> None:
    if not CLEAN_PATH.exists():
        print("ERROR: data/Crop_clean.csv not found. Run clean_data.py first.")
        sys.exit(1)
    df = pd.read_csv(CLEAN_PATH)
    prof = compute_profiles(df)
    OUT_PATH.parent.mkdir(exist_ok=True)
    prof.round(4).to_csv(OUT_PATH)

    print(f"Profiles built for {len(prof)} crops x {len(FEATURES)} features.")
    print(f"Columns per crop: {prof.shape[1]} "
          f"(9 statistics x {len(FEATURES)} features + n_records)")
    print(f"Records per crop (min/max): {int(prof['n_records'].min())}/"
          f"{int(prof['n_records'].max())}")
    print("\nTypical range (5th-95th percentile) for three example crops:")
    for crop in [c for c in ["rice", "chickpea", "apple"] if c in prof.index]:
        print(f"\n  {crop}")
        for f in FEATURES:
            print(f"    {f:12} mean {prof.loc[crop, f'{f}_mean']:8.2f}   "
                  f"typical {prof.loc[crop, f'{f}_p05']:8.2f} to "
                  f"{prof.loc[crop, f'{f}_p95']:8.2f}")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()