"""
Step 4b: Check the recommendation logic and write example outputs.
Run from the project root:  python test_recommender.py
Writes: outputs/example_recommendations.json, outputs/assessment_validation.csv
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from recommender import (FEATURES, TARGET, InputError, Recommender,
                         compute_profiles)

BASE_DIR = Path(__file__).resolve().parent
CLEAN_PATH = BASE_DIR / "data" / "Crop_clean.csv"
OUT_DIR = BASE_DIR / "outputs"
RANDOM_STATE, TEST_SIZE = 42, 0.20     # must match train_model.py

results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def section(t: str) -> None:
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def median_input(rec: Recommender, crop: str) -> dict:
    return {f: float(rec.profiles.loc[crop, f"{f}_median"]) for f in FEATURES}


def main() -> None:
    try:
        rec = Recommender()
    except FileNotFoundError as e:
        print("ERROR:", e)
        sys.exit(1)
    df = pd.read_csv(CLEAN_PATH)
    print(f"Loaded model: {rec.model_name}; {len(rec.classes)} crops.")
    for n in rec.notes:
        print("NOTE:", n)

    # ---------------------------------------------------------- 1. validation
    section("1. INPUT VALIDATION")
    good = median_input(rec, rec.classes[0])
    check("valid input accepted", bool(rec.validate(good)[0]))
    bad_cases = {
        "pH = 15": {**good, "ph": 15},
        "negative nitrogen": {**good, "N": -5},
        "humidity = 130": {**good, "humidity": 130},
        "text in temperature": {**good, "temperature": "hot"},
        "missing rainfall": {k: v for k, v in good.items() if k != "rainfall"},
        "NaN value": {**good, "K": float("nan")},
        "empty string": {**good, "P": "  "},
    }
    for name, case in bad_cases.items():
        try:
            rec.recommend(case)
            check(f"rejects {name}", False)
        except InputError as e:
            check(f"rejects {name}", True, "; ".join(e.errors.values())[:60])
    try:
        rec.recommend(good, selected_crop="dragonfruit")
        check("rejects unknown crop", False)
    except InputError:
        check("rejects unknown crop", True)
    _, warn = rec.validate({**good, "N": 250})
    check("warns when outside dataset range", len(warn) == 1, warn[0][:70] if warn else "")
    check("accepts display name for crop",
          rec.resolve_crop("Kidney Beans") == "kidneybeans")

    # ------------------------------------------------- 2. structure + consistency
    section("2. STRUCTURE AND CONSISTENCY")
    r1 = rec.recommend(good, selected_crop=rec.classes[0])
    r2 = rec.recommend(good, selected_crop=rec.classes[0])
    check("same input gives identical output",
          json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True))
    check("top-3 has three distinct crops, ranks 1-3",
          len({c["crop"] for c in r1["top3"]}) == 3
          and [c["rank"] for c in r1["top3"]] == [1, 2, 3])
    check("model scores sum to 1",
          abs(sum(c["model_score"] for c in r1["all_crops"]) - 1.0) < 1e-6)
    check("all 22 crops ranked 1..22",
          sorted(c["rank"] for c in r1["all_crops"]) == list(range(1, len(rec.classes) + 1)))
    check("profile fit within 0..1",
          all(0 <= c["profile_fit"] <= 1 for c in r1["all_crops"]))
    check("result is JSON-serialisable", bool(json.dumps(r1)))

    # -------------------------------------- 3. inputs change the recommendations
    section("3. DIFFERENT INPUTS GIVE DIFFERENT RECOMMENDATIONS")
    tops = {}
    for crop in rec.classes:
        r = rec.recommend(median_input(rec, crop))
        tops[crop] = r["top3"][0]["crop"]
    n_self = sum(tops[c] == c for c in rec.classes)
    check("recommendations vary across inputs", len(set(tops.values())) > 1,
          f"{len(set(tops.values()))} distinct top-1 crops")
    print(f"  INFO: median-profile input returns its own crop at rank 1 for "
          f"{n_self}/{len(rec.classes)} crops")
    miss = {c: t for c, t in tops.items() if c != t}
    if miss:
        print("  INFO: exceptions (input crop -> model top-1):",
              ", ".join(f"{a}->{b}" for a, b in miss.items()))

    # ------------- 4. held-out evidence with profiles built from TRAIN rows only
    section("4. HELD-OUT CHECK (profiles from training rows only)")
    X, y = df[FEATURES], df[TARGET].astype(str).to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    tr_df = X_tr.copy()
    tr_df[TARGET] = y_tr
    rec_tr = Recommender(profiles=compute_profiles(tr_df))
    idx = {c: i for i, c in enumerate(rec_tr.classes)}

    top1 = top3 = fit1 = fit3 = 0
    label_true, label_wrong = Counter(), Counter()
    rows = []
    errors_found = []
    for n in range(len(X_te)):
        clean = {f: float(X_te.iloc[n][f]) for f in FEATURES}
        ctx = rec_tr._score(clean)
        true_i = idx[y_te[n]]
        order = list(ctx["order"])
        top1 += order[0] == true_i
        top3 += true_i in order[:3]
        fit_order = list(np.lexsort((ctx["proba"], ctx["fit"]))[::-1])
        fit1 += fit_order[0] == true_i
        fit3 += true_i in fit_order[:3]
        label_true[rec_tr._components(ctx, true_i)[2]] += 1
        for k in range(len(rec_tr.classes)):
            if k != true_i:
                label_wrong[rec_tr._components(ctx, k)[2]] += 1
        if order[0] != true_i:
            errors_found.append((n, true_i, order[0], ctx))
    N = len(X_te)
    print(f"  Model ranking      : top-1 {top1 / N:.4f}   top-3 {top3 / N:.4f}")
    print(f"  Profile fit alone  : top-1 {fit1 / N:.4f}   top-3 {fit3 / N:.4f}")

    n_wrong = sum(label_wrong.values())
    val_rows = []
    print("\n  Assessment label when the user selects...")
    print(f"  {'':28}{'Strong':>9}{'Moderate':>10}{'Low':>8}")
    for name, cnt, tot in [("the TRUE crop", label_true, N),
                           ("a WRONG crop", label_wrong, n_wrong)]:
        s, m, l = (cnt[k] / tot for k in ("strong", "moderate", "low"))
        print(f"  {name:28}{s:9.3f}{m:10.3f}{l:8.3f}")
        val_rows.append({"selected_crop_is": name, "n_evaluations": tot,
                         "share_strong": s, "share_moderate": m, "share_low": l})
    pd.DataFrame(val_rows).round(6).to_csv(OUT_DIR / "assessment_validation.csv", index=False)
    s_true = label_true["strong"] / N
    s_wrong = label_wrong["strong"] / n_wrong
    check("'Strong' is much more common for true crop than wrong crops",
          s_true > 5 * max(s_wrong, 1e-9) or (s_wrong == 0 and s_true > 0),
          f"{s_true:.3f} vs {s_wrong:.3f}")

    print(f"\n  Misclassified held-out rows: {len(errors_found)}")
    for n, ti, pi, ctx in errors_found[:5]:
        print(f"   row {n}: true={rec_tr.classes[ti]}  model top-1={rec_tr.classes[pi]}  "
              f"profile fit true={ctx['fit'][ti]:.2f} vs predicted={ctx['fit'][pi]:.2f}")
        for c_t, c_p in zip(rec_tr._feature_comparison(ctx, ti),
                            rec_tr._feature_comparison(ctx, pi)):
            print(f"     {c_t['feature']:12} value {c_t['input']:8.2f} | "
                  f"true crop: {c_t['status']:21} | predicted crop: {c_p['status']}")

    # ------------------------------------------------- 5. example outputs
    section("5. EXAMPLE RECOMMENDATIONS (saved for your reports)")
    a, b = rec.classes[0], rec.classes[-1]
    examples = {
        f"typical_{a}_select_{a}": rec.recommend(median_input(rec, a), a),
        f"typical_{a}_select_{b}": rec.recommend(median_input(rec, a), b),
    }
    mixed = {f: 0.5 * (median_input(rec, a)[f] + median_input(rec, b)[f]) for f in FEATURES}
    examples[f"blend_of_{a}_and_{b}_select_{a}"] = rec.recommend(mixed, a)
    for name, r in examples.items():
        sel = r["selected"]
        print(f"\n  {name}")
        print("    top-3:", ", ".join(f"{c['display']} ({c['model_score_display']}, "
                                      f"fit {c['profile_fit_display']})" for c in r["top3"]))
        print(f"    selected: {sel['display']} -> {sel['label']}  "
              f"[model {sel['model_component']}, profile {sel['profile_component']}, "
              f"rank {sel['rank']}, fit {sel['profile_fit_display']}]")
    blend_key = f"blend_of_{a}_and_{b}_select_{a}"
    typical_key = f"typical_{a}_select_{a}"
    check("typical input gives no caution", not examples[typical_key]["cautions"])
    check("blended (unusual) input gives a caution", bool(examples[blend_key]["cautions"]))
    print("    caution text:", examples[blend_key]["cautions"][0][:110], "...")
    with open(OUT_DIR / "example_recommendations.json", "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)

    section("SUMMARY")
    print(f"  {sum(results)} of {len(results)} automatic checks passed.")
    print("  Saved: outputs/example_recommendations.json, outputs/assessment_validation.csv")
    if not all(results):
        print("  Some checks FAILED. Fix them before continuing to Step 5.")


if __name__ == "__main__":
    main()