"""Step 12: fit the predetermined calibrator set on the LATER reserved calib
period, after the blend exponents are already fixed by 03_fit_blend.py.

Candidates (frozen in blend_manifest.json): none (uncalibrated control),
sigmoid, isotonic, odds_band. Every calibrated probability is renormalised
within its complete race before it is scored, so each candidate is compared as a
probability vector summing to 1.

PRE-COMMITTED PROMOTION RULE, written here before the eval panel is scored by
any script in this stage: each calibrator is fit on the chronologically FIRST
70% of calib races and scored by race-level log loss on the LAST 30%; the lowest
wins (``none`` can win), and the winner is then refit on the whole calib period.
05_evaluate.py reports EVERY candidate on eval regardless of this rule, so the
rule selects what would be promoted, it does not filter what is reported.

The eval panel and the final holdout are never read here.

Writes data/audit/12/calibrators/ and reports/improvement/12/calibration_policy.json.
Run: .venv/Scripts/python.exe reports/improvement/12/04_fit_calibrators.py
"""
import hashlib
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from models.blend import PowerBlend, race_level_log_loss
from models.calibration import (
    IsotonicCalibrator,
    OddsBandCalibrator,
    SigmoidCalibrator,
    normalize_within_race,
)

MANIFEST = "reports/improvement/12/blend_manifest.json"
PANEL = "data/audit/12/walkforward_oos.parquet"
BLEND_DIR = "data/audit/12/blend"
CAL_DIR = "data/audit/12/calibrators"
OUT = "reports/improvement/12/calibration_policy.json"

VAL_FRAC = 0.30           # chronological tail of calib used to pick the winner
METHODS = ("none", "sigmoid", "isotonic", "odds_band")

# line name -> (model probability column, blend artifact or explicit exponents)
LINES = {
    "market_only": ("condlogit_indep_prob", (0.0, 1.0)),
    "indep_only": ("condlogit_indep_prob", (1.0, 0.0)),
    "combined_indep": ("condlogit_indep_prob", "combined_indep.json"),
    "combined_indep_forced": ("condlogit_indep_prob", "combined_indep_forced.json"),
    "combined_market": ("condlogit_market_prob", "combined_market.json"),
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fit_one(method: str, p, odds, y):
    if method == "none":
        return None
    if method == "sigmoid":
        return SigmoidCalibrator.fit(p, y)
    if method == "isotonic":
        return IsotonicCalibrator.fit(p, y)
    if method == "odds_band":
        return OddsBandCalibrator.fit(p, odds, y, method="isotonic")
    raise ValueError(method)


def apply_one(cal, p, odds) -> np.ndarray:
    if cal is None:
        return np.asarray(p, dtype=float)
    if isinstance(cal, OddsBandCalibrator):
        return np.asarray(cal.predict(p, odds), dtype=float)
    return np.asarray(cal.predict(p), dtype=float)


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    panel = pd.read_parquet(PANEL)
    cal = panel[panel["period"] == "calib"].reset_index(drop=True)
    complete = cal["market_ref_prob"].notna().groupby(cal["race_uid"]).transform("all")
    cal = cal[complete.to_numpy()].sort_values("race_date").reset_index(drop=True)

    races = cal["race_uid"].drop_duplicates().tolist()   # already chronological
    n_fit = int(round(len(races) * (1.0 - VAL_FRAC)))
    fit_races = set(races[:n_fit])
    is_fit = cal["race_uid"].isin(fit_races).to_numpy()
    print(f"calib: {len(cal)} rows / {len(races)} races; "
          f"fit {int(is_fit.sum())} rows / {n_fit} races, "
          f"val {int((~is_fit).sum())} rows / {len(races) - n_fit} races")

    y = cal["won"].to_numpy(float)
    rid = cal["race_uid"].to_numpy()
    odds = cal["bet_price"].to_numpy(float)
    ref = cal["market_ref_prob"].to_numpy(float)

    os.makedirs(CAL_DIR, exist_ok=True)
    policy = {
        "manifest": MANIFEST,
        "manifest_sha256": sha256_file(MANIFEST),
        "panel": PANEL,
        "panel_sha256": sha256_file(PANEL),
        "window": manifest["windows"]["calib"],
        "n_rows": int(len(cal)),
        "n_races": int(len(races)),
        "promotion_rule": (
            f"fit on the first {100 * (1 - VAL_FRAC):.0f}% of calib races, score race-level "
            f"log loss (after within-race normalisation) on the last {100 * VAL_FRAC:.0f}%, "
            "lowest wins including the uncalibrated control; winner refit on all of calib. "
            "Committed before any eval-panel scoring in this stage."
        ),
        "odds_column_used_by_odds_band": (
            "bet_price — the pre-off REFERENCE price. Never an executable best-of-N quote "
            "(see reports/improvement/12/fl_recalibration_audit.json)."
        ),
        "lines": {},
    }

    for line, (prob_col, spec) in LINES.items():
        if isinstance(spec, str):
            blend = PowerBlend.load(os.path.join(BLEND_DIR, spec))
            blend_ref = {"artifact": os.path.join(BLEND_DIR, spec),
                         "sha256": sha256_file(os.path.join(BLEND_DIR, spec))}
        else:
            blend = PowerBlend(*spec, metadata={"fixed_control": True})
            blend_ref = {"artifact": None, "fixed_control": True}

        p_blend = blend.transform(cal[prob_col].to_numpy(float), ref, rid)
        usable = np.isfinite(p_blend)
        entry = {
            "model_prob_col": prob_col,
            "alpha": blend.alpha, "beta": blend.beta, "blend": blend_ref,
            "n_scoreable_rows": int(usable.sum()),
            "candidates": {},
        }

        val_scores = {}
        for method in METHODS:
            m_fit = is_fit & usable
            fitted_val = fit_one(method, p_blend[m_fit], odds[m_fit], y[m_fit])
            m_val = (~is_fit) & usable
            p_val = apply_one(fitted_val, p_blend[m_val], odds[m_val])
            p_val = normalize_within_race(p_val, rid[m_val])
            val_scores[method] = race_level_log_loss(y[m_val], p_val, rid[m_val])

            # Production artifact: refit on the WHOLE calib period.
            full = fit_one(method, p_blend[usable], odds[usable], y[usable])
            art = None
            if full is not None:
                art = os.path.join(CAL_DIR, f"{line}__{method}.pkl")
                with open(art, "wb") as fh:
                    pickle.dump(full, fh)
                blob = open(art, "rb").read()
                assert b"numpy" not in blob, f"{art} leaked a numpy object"
            entry["candidates"][method] = {
                "artifact": art,
                "calib_val_tail_race_log_loss": val_scores[method],
                "n_bands": (len(full.centers) if isinstance(full, OddsBandCalibrator) else None),
            }

        best = min(val_scores, key=lambda k: (np.inf if np.isnan(val_scores[k])
                                              else val_scores[k]))
        entry["promoted_by_rule"] = best
        policy["lines"][line] = entry
        pretty = " ".join(f"{m}={val_scores[m]:.5f}" for m in METHODS)
        print(f"{line} (a={blend.alpha}, b={blend.beta}): {pretty} -> promoted '{best}'")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(policy, fh, indent=2)
    print(f"wrote {OUT}")
    print("CALIBRATORS_12_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
