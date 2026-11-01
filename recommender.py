"""
AgroWise recommendation logic (Step 4).

Used by build_profiles.py, test_recommender.py and (in Step 5) app.py.

What this module does
  * validates farmer inputs
  * loads the saved model + artifacts (never retrains)
  * ranks crops by model score (ties broken by log-probability, then profile fit)
  * computes a transparent PROFILE-FIT score from data-derived crop ranges
  * assesses a selected crop (Strong / Moderate / Low model support)

Important: scores describe similarity to crop records in the dataset.
They are NOT probabilities of successful growth or yield.
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"

# Hard limits: values outside these are rejected as impossible/implausible.
HARD_LIMITS = {
    "N": (0, 300), "P": (0, 300), "K": (0, 300),
    "temperature": (-10, 55), "humidity": (0, 100),
    "ph": (0, 14), "rainfall": (0, 2000),
}

# ---- documented assessment thresholds (design choices, not agronomic standards)
HIGH_PROB = 0.60              # model "high": rank 1 and score >= this
MEDIUM_PROB = 0.10            # model "medium": rank <= 3 or score >= this
TOP_N_MEDIUM = 3
CLOSE_FIT = 0.80              # profile "close": fit >= this ...
MAX_OUTSIDE_FOR_CLOSE = 1     # ... and at most this many features outside range
PARTIAL_FIT = 0.50            # profile "partial": fit >= this
LOG_FLOOR = -1e12             # replaces -inf log-probabilities

LABEL_TABLE = {
    ("high", "close"): "strong", ("high", "partial"): "moderate",
    ("high", "poor"): "moderate",
    ("medium", "close"): "moderate", ("medium", "partial"): "moderate",
    ("medium", "poor"): "low",
    ("low", "close"): "moderate", ("low", "partial"): "low",
    ("low", "poor"): "low",
}
LABEL_TEXT = {
    "strong": "Strong model support",
    "moderate": "Moderate model support; further comparison recommended",
    "low": "Low model support; consider alternative crops",
}

DISCLAIMER = ("Scores rank similarity to crop records in the dataset. They are "
              "not probabilities of successful growth or yield, and reference "
              "ranges are data-derived, not agronomic optima.")


class InputError(ValueError):
    """Raised for invalid user input; .errors maps field name -> message."""

    def __init__(self, errors: dict):
        self.errors = errors
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))


# ------------------------------------------------------------ profile builder
def compute_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-crop reference statistics for every feature (one row per crop)."""
    g = df.groupby(TARGET)
    parts = {}
    for f in FEATURES:
        s = g[f]
        parts[f"{f}_mean"] = s.mean()
        parts[f"{f}_sd"] = s.std(ddof=1)
        parts[f"{f}_min"] = s.min()
        parts[f"{f}_p05"] = s.quantile(0.05)
        parts[f"{f}_q1"] = s.quantile(0.25)
        parts[f"{f}_median"] = s.median()
        parts[f"{f}_q3"] = s.quantile(0.75)
        parts[f"{f}_p95"] = s.quantile(0.95)
        parts[f"{f}_max"] = s.max()
    prof = pd.DataFrame(parts)
    prof["n_records"] = g.size()
    prof.index.name = "crop"
    return prof


def _fmt_score(p: float) -> str:
    if p >= 0.999:
        return ">=99.9%"
    if p < 0.001:
        return "<0.1%"
    return f"{p * 100:.1f}%"


def _fmt_val(feature: str, v: float) -> str:
    return f"{v:.2f}" if feature == "ph" else f"{v:.1f}"


class Recommender:
    def __init__(self, base_dir=None, profiles: pd.DataFrame = None):
        base = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        model_path = base / "models" / "trained_model.joblib"
        art_path = base / "models" / "preprocessing_artifacts.joblib"
        for p in (model_path, art_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p.name}. Run 'python train_model.py' first.")
        self.model = joblib.load(model_path)
        self.art = joblib.load(art_path)
        self.features = list(self.art["features"])
        self.classes = list(self.art["classes"])
        self.labels = self.art["feature_labels"]
        self.display = self.art["crop_display_names"]
        self.ranges = self.art["feature_ranges"]
        self.model_name = self.art["selected_model"]
        if list(self.model.classes_) != self.classes:
            raise RuntimeError("Model classes do not match saved artifacts.")

        self.notes = []
        if self.art.get("sklearn_version") != sklearn.__version__:
            self.notes.append(
                f"Model trained with scikit-learn {self.art.get('sklearn_version')} "
                f"but running {sklearn.__version__}; results may differ.")

        if profiles is None:
            prof_path = base / "outputs" / "crop_profiles.csv"
            if not prof_path.exists():
                raise FileNotFoundError(
                    "Missing outputs/crop_profiles.csv. Run 'python build_profiles.py'.")
            profiles = pd.read_csv(prof_path, index_col="crop")
        self.profiles = profiles.reindex(self.classes)
        if self.profiles.isna().any().any():
            raise RuntimeError("Crop profiles are incomplete for the model's classes.")

        def arr(stat):
            return np.column_stack(
                [self.profiles[f"{f}_{stat}"].to_numpy(float) for f in self.features])

        self.p_mean, self.p_sd = arr("mean"), arr("sd")
        self.p_min, self.p_max = arr("min"), arr("max")
        self.p_lo, self.p_hi = arr("p05"), arr("p95")
        span = np.array([self.ranges[f]["max"] - self.ranges[f]["min"]
                         for f in self.features], dtype=float)
        self.width_floor = 0.02 * span
        self.sd_floor = 0.01 * span
        self.n_feat = len(self.features)

    # ------------------------------------------------------------ validation
    def validate(self, raw: dict):
        errors, clean, warnings = {}, {}, []
        for f in self.features:
            label = self.labels[f]
            val = raw.get(f)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors[f] = f"{label} is required."
                continue
            try:
                x = float(val.strip()) if isinstance(val, str) else float(val)
            except (TypeError, ValueError):
                errors[f] = f"{label} must be a number."
                continue
            if not math.isfinite(x):
                errors[f] = f"{label} must be a finite number."
                continue
            lo, hi = HARD_LIMITS[f]
            if not lo <= x <= hi:
                errors[f] = f"{label} must be between {lo} and {hi}."
                continue
            clean[f] = x
            r = self.ranges[f]
            if x < r["min"]:
                warnings.append(
                    f"{label} = {_fmt_val(f, x)} is below the lowest value in the "
                    f"dataset ({_fmt_val(f, r['min'])}); the model is extrapolating.")
            elif x > r["max"]:
                warnings.append(
                    f"{label} = {_fmt_val(f, x)} is above the highest value in the "
                    f"dataset ({_fmt_val(f, r['max'])}); the model is extrapolating.")
        if errors:
            raise InputError(errors)
        return clean, warnings

    def resolve_crop(self, selected):
        if selected is None or (isinstance(selected, str) and not selected.strip()):
            return None
        s = str(selected).strip().lower()
        if s in self.classes:
            return s
        for key, disp in self.display.items():
            if s == disp.lower():
                return key
        raise InputError({"selected_crop": f"'{selected}' is not a crop in this dataset."})

    # --------------------------------------------------------------- scoring
    def _score(self, clean: dict) -> dict:
        frame = pd.DataFrame([clean], columns=self.features)
        proba = self.model.predict_proba(frame)[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            try:
                logp = self.model.predict_log_proba(frame)[0]
            except Exception:
                logp = np.log(proba)
        logp = np.where(np.isfinite(logp), logp, LOG_FLOOR)

        x = np.array([clean[f] for f in self.features], dtype=float)
        width = np.maximum(self.p_hi - self.p_lo, self.width_floor)
        below = np.clip(self.p_lo - x, 0, None)
        above = np.clip(x - self.p_hi, 0, None)
        dist = below + above
        fit_mat = np.clip(1.0 - dist / width, 0.0, 1.0)
        fit = fit_mat.mean(axis=1)
        inside = dist == 0
        z = (x - self.p_mean) / np.maximum(self.p_sd, self.sd_floor)

        order = np.lexsort((fit, logp))[::-1]      # primary: logp, secondary: fit
        rank = np.empty(len(order), dtype=int)
        rank[order] = np.arange(1, len(order) + 1)
        return {"x": x, "clean": clean, "proba": proba, "fit": fit,
                "fit_mat": fit_mat, "inside": inside, "z": z,
                "order": order, "rank": rank}

    def _components(self, ctx: dict, i: int):
        p, r = float(ctx["proba"][i]), int(ctx["rank"][i])
        if r == 1 and p >= HIGH_PROB:
            mc = "high"
        elif r <= TOP_N_MEDIUM or p >= MEDIUM_PROB:
            mc = "medium"
        else:
            mc = "low"
        f = float(ctx["fit"][i])
        n_out = self.n_feat - int(ctx["inside"][i].sum())
        if f >= CLOSE_FIT and n_out <= MAX_OUTSIDE_FOR_CLOSE:
            pc = "close"
        elif f >= PARTIAL_FIT:
            pc = "partial"
        else:
            pc = "poor"
        return mc, pc, LABEL_TABLE[(mc, pc)]

    # -------------------------------------------------------------- builders
    def _crop_row(self, ctx: dict, i: int) -> dict:
        crop = self.classes[i]
        p = float(ctx["proba"][i])
        return {
            "crop": crop, "display": self.display[crop],
            "rank": int(ctx["rank"][i]),
            "model_score": p, "model_score_display": _fmt_score(p),
            "profile_fit": float(ctx["fit"][i]),
            "profile_fit_display": f"{ctx['fit'][i] * 100:.0f}%",
            "features_inside": int(ctx["inside"][i].sum()),
            "features_total": self.n_feat,
            "feature_fits": {f: float(ctx["fit_mat"][i, j])
                             for j, f in enumerate(self.features)},
        }

    def _feature_comparison(self, ctx: dict, i: int) -> list:
        rows = []
        for j, f in enumerate(self.features):
            x = float(ctx["x"][j])
            lo, hi = float(self.p_lo[i, j]), float(self.p_hi[i, j])
            if x < lo:
                status = "below typical range"
            elif x > hi:
                status = "above typical range"
            else:
                status = "within typical range"
            rows.append({
                "feature": f, "label": self.labels[f], "input": x,
                "mean": float(self.p_mean[i, j]), "sd": float(self.p_sd[i, j]),
                "p05": lo, "p95": hi,
                "min": float(self.p_min[i, j]), "max": float(self.p_max[i, j]),
                "status": status, "z_score": float(ctx["z"][i, j]),
                "feature_fit": float(ctx["fit_mat"][i, j]),
                "outside_observed_min_max": bool(x < self.p_min[i, j] or x > self.p_max[i, j]),
            })
        return rows

    def _distinguishing(self, i: int, j: int, top: int = 3) -> list:
        """Features that most separate crop i from crop j (standardised mean difference)."""
        sd_i = np.maximum(self.p_sd[i], self.sd_floor)
        sd_j = np.maximum(self.p_sd[j], self.sd_floor)
        d = np.abs(self.p_mean[i] - self.p_mean[j]) / np.sqrt((sd_i ** 2 + sd_j ** 2) / 2)
        out = []
        for k in np.argsort(d)[::-1][:top]:
            f = self.features[k]
            out.append({
                "feature": f, "label": self.labels[f],
                "standardised_difference": float(d[k]),
                "crop_a": self.classes[i], "mean_a": float(self.p_mean[i, k]),
                "range_a": [float(self.p_lo[i, k]), float(self.p_hi[i, k])],
                "crop_b": self.classes[j], "mean_b": float(self.p_mean[j, k]),
                "range_b": [float(self.p_lo[j, k]), float(self.p_hi[j, k])],
            })
        return out

    def _assessment(self, ctx: dict, i: int) -> dict:
        mc, pc, label_key = self._components(ctx, i)
        row = self._crop_row(ctx, i)
        comp = self._feature_comparison(ctx, i)
        outside = [c for c in comp if c["status"] != "within typical range"]
        gaps = sorted(outside, key=lambda c: (c["feature_fit"], -abs(c["z_score"])))[:3]

        sel_rank, sel_fit = int(ctx["rank"][i]), float(ctx["fit"][i])
        alts = []
        for k, crop in enumerate(self.classes):
            if k == i:
                continue
            beats_model = int(ctx["rank"][k]) < sel_rank
            beats_profile = float(ctx["fit"][k]) > sel_fit
            if beats_model or beats_profile:
                r = self._crop_row(ctx, k)
                r["better_on_model_rank"] = beats_model
                r["better_on_profile_fit"] = beats_profile
                alts.append(r)
        alts.sort(key=lambda r: (-(r["better_on_model_rank"] + r["better_on_profile_fit"]),
                                 r["rank"]))
        return {
            **row,
            "model_component": mc, "profile_component": pc,
            "label_key": label_key, "label": LABEL_TEXT[label_key],
            "feature_comparison": comp,
            "largest_gaps": gaps,
            "alternatives": alts[:3],
            "reference_note": ("Typical range = central 90% (5th to 95th percentile) "
                               "of this crop's dataset records. It is a data-derived "
                               "reference, not a scientifically guaranteed optimum."),
        }

    def _explanation(self, ctx: dict) -> str:
        i = int(ctx["order"][0])
        row = self._crop_row(ctx, i)
        comp = self._feature_comparison(ctx, i)
        out = [c for c in comp if c["status"] != "within typical range"]
        text = (f"{row['display']} is ranked first by the {self.model_name} model "
                f"(model support score {row['model_score_display']}). "
                f"{row['features_inside']} of {row['features_total']} entered values "
                f"fall inside the typical range of {row['display']} records in the dataset")
        if out:
            names = ", ".join(f"{c['label']} ({c['status'].split()[0]})" for c in out)
            text += f"; outside the range: {names}."
        else:
            text += "."
        if len(ctx["order"]) > 1:
            j = int(ctx["order"][1])
            d = self._distinguishing(i, j, top=2)
            parts = [
                f"{x['label']}, average {_fmt_val(x['feature'], x['mean_a'])} for "
                f"{self.display[x['crop_a']]} vs {_fmt_val(x['feature'], x['mean_b'])} for "
                f"{self.display[x['crop_b']]}" for x in d]
            text += (f" In the dataset, the conditions that most separate "
                     f"{self.display[self.classes[i]]} from {self.display[self.classes[j]]} "
                     f"are: " + "; ".join(parts) + ".")
        return text

    # ------------------------------------------------------------ public API
    def recommend(self, raw_inputs: dict, selected_crop=None) -> dict:
        errors, clean, warnings = {}, None, []
        try:
            clean, warnings = self.validate(raw_inputs)
        except InputError as e:
            errors.update(e.errors)
        sel_key = None
        try:
            sel_key = self.resolve_crop(selected_crop)
        except InputError as e:
            errors.update(e.errors)
        if errors:
            raise InputError(errors)

        ctx = self._score(clean)
        order = ctx["order"]
        top3 = [self._crop_row(ctx, int(i)) for i in order[:3]]
        by_fit = sorted(range(len(self.classes)),
                        key=lambda k: (-ctx["fit"][k], ctx["rank"][k]))[:3]
        top1, top2 = int(order[0]), int(order[1])
        cautions = []
        best_fit = float(ctx["fit"].max())
        if best_fit < CLOSE_FIT:
            cautions.append(
                "The entered conditions do not closely match any single crop's "
                f"typical ranges in the dataset (best profile fit: {best_fit * 100:.0f}%). "
                "Model scores are relative rankings and can be very high even for "
                "unusual inputs, so read this recommendation with caution.")
        return {
            "inputs": {f: clean[f] for f in self.features},
            "warnings": warnings,
            "cautions": cautions,
            "notes": list(self.notes),
            "model_name": self.model_name,
            "top3": top3,
            "closest_by_profile": [self._crop_row(ctx, k) for k in by_fit],
            "distinguishing": self._distinguishing(top1, top2),
            "explanation": self._explanation(ctx),
            "selected": (self._assessment(ctx, self.classes.index(sel_key))
                         if sel_key else None),
            "all_crops": [self._crop_row(ctx, int(i)) for i in order],
            "disclaimer": DISCLAIMER,
        }