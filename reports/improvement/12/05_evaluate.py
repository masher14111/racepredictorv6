"""Step 12: score every candidate on the RESERVED evaluation panel.

The panel is step 10's frozen dev-OOS window (2026-08-08..2026-08-27). Nothing
in this script fits, tunes or selects anything — the blend exponents come from
03_fit_blend.py (blend_dev only) and the calibrators from 04_fit_calibrators.py
(calib only). Every candidate and the market line are scored over the IDENTICAL
complete-reference-book race set via models.head_to_head.head_to_head.

The final holdout window is never loaded.

Writes reports/improvement/12/scorecard_12.json and
reports/improvement/12/breakdowns/*.csv.
Run: .venv/Scripts/python.exe reports/improvement/12/05_evaluate.py
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from features.lgbm_adapter import FINAL_FEATURE_COLS, INDEPENDENT_FEATURE_COLS, build_lgbm_matrix
from models.blend import PowerBlend, race_level_log_loss
from models.calibration import OddsBandCalibrator, normalize_within_race
from models.devig import devig
from models.head_to_head import ODDS_BANDS, head_to_head

MANIFEST = "reports/improvement/12/blend_manifest.json"
POLICY = "reports/improvement/12/calibration_policy.json"
BLEND_DIR = "data/audit/12/blend"
CAL_DIR = "data/audit/12/calibrators"
STEP10_PRED = "data/audit/10/dev_oos_predictions.parquet"
OUT = "reports/improvement/12/scorecard_12.json"
BREAKDOWN_DIR = "reports/improvement/12/breakdowns"
EVAL_PANEL_OUT = "data/audit/12/eval_panel_scored.parquet"

CARRY_COLS = ("region", "going_band", "going_is_all_weather", "distance_furlongs",
              "field_size")
METHODS = ("none", "sigmoid", "isotonic", "odds_band")
MIN_EV = 0.05                      # config.yaml value.min_expected_value
N_BOOT = 2000
BOOT_SEED = 12

HEADLINE = ("market_only__none", "indep_only__none", "indep_only__odds_band",
            "combined_indep_forced__none", "combined_market__none",
            "combined_market__sigmoid")


# ── panel construction ───────────────────────────────────────────────────────


def build_eval_panel(manifest) -> pd.DataFrame:
    w = manifest["windows"]["eval"]
    start, end = pd.Timestamp(w["start"]), pd.Timestamp(w["end_exclusive"])
    df = pd.read_parquet(manifest["data"]["rebuilt_matrix_path"])
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    day = pd.to_datetime(win["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    ev = win[(day >= start) & (day < end)].reset_index(drop=True)
    del df, win
    panel = load_panel(df=ev, config=PanelConfig(feature_cols=CARRY_COLS))

    ref = pd.read_parquet(STEP10_PRED)
    assert len(panel) == len(ref), f"panel {len(panel)} != step10 {len(ref)}"
    assert panel["race_uid"].nunique() == ref["race_uid"].nunique()
    probs = ref[["race_uid", "horse_id", "condlogit_indep_prob", "condlogit_market_prob",
                 "catboost_indep_prob", "catboost_market_prob"]]
    panel = panel.merge(probs, on=["race_uid", "horse_id"], how="left")
    assert panel["condlogit_indep_prob"].notna().all(), "step-10 merge left gaps"

    # LightGBM: step 10 left its dev-OOS probabilities only in a per-branch ledger
    # covering 3,347/5,561 rows (its own recorded gap). Re-score the SAVED boosters
    # here so the whole panel is covered; nothing is refitted.
    from models.lgbm_softmax import LGBMSoftmaxModel
    for branch, cols in (("indep", INDEPENDENT_FEATURE_COLS),
                         ("market", FINAL_FEATURE_COLS)):
        path = f"data/audit/10/models_lgbm_{branch}/lgbm_won.txt"
        model = LGBMSoftmaxModel.load(path)
        X, _, rid = build_lgbm_matrix(ev, inference=True, feature_cols=cols)
        scored = pd.DataFrame({
            "race_uid": ev["race_uid"].to_numpy(), "horse_id": ev["horse_id"].to_numpy(),
            f"lgbm_{branch}_prob": model.predict_proba(X, rid),
        }).drop_duplicates(subset=["race_uid", "horse_id"])
        panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")

    panel["market_ref_prob"] = devig(panel["bet_price"], panel["race_uid"],
                                     method="proportional")
    complete = panel["market_ref_prob"].notna().groupby(panel["race_uid"]).transform("all")
    panel = panel[complete.to_numpy()].reset_index(drop=True)
    return panel


# ── strata ───────────────────────────────────────────────────────────────────


def field_band(n: float) -> str:
    for lo, hi, label in ((0, 6, "2-5"), (6, 9, "6-8"), (9, 12, "9-11"),
                          (12, 16, "12-15")):
        if lo <= n < hi:
            return label
    return "16+"


def distance_cat(f: float) -> str:
    if not np.isfinite(f):
        return "unknown"
    if f < 7:
        return "sprint(<7f)"
    if f < 9.5:
        return "mile(7-9.5f)"
    if f < 12:
        return "middle(9.5-12f)"
    return "staying(>=12f)"


def add_strata(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["_field_band"] = panel["field_size"].astype(float).map(field_band)
    surface = np.where(panel["going_is_all_weather"].astype(bool), "all_weather", "turf")
    panel["_regime"] = pd.Series(surface) + " / " + \
        panel["distance_furlongs"].astype(float).map(distance_cat)
    panel["_region"] = panel["region"].astype(str)
    band = pd.Series("other", index=panel.index, dtype=object)
    o = panel["bet_price"].astype(float)
    for label, lo, hi in ODDS_BANDS:
        band[(o >= lo) & (o < hi)] = label
    panel["_odds_band"] = band
    return panel


# ── candidate construction ───────────────────────────────────────────────────


def load_cal(line: str, method: str):
    if method == "none":
        return None
    path = os.path.join(CAL_DIR, f"{line}__{method}.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def apply_cal(cal, p, odds) -> np.ndarray:
    if cal is None:
        return np.asarray(p, dtype=float)
    if isinstance(cal, OddsBandCalibrator):
        return np.asarray(cal.predict(p, odds), dtype=float)
    return np.asarray(cal.predict(p), dtype=float)


def build_candidates(panel: pd.DataFrame, policy) -> tuple[pd.DataFrame, dict]:
    rid = panel["race_uid"].to_numpy()
    odds = panel["bet_price"].to_numpy(float)
    ref = panel["market_ref_prob"].to_numpy(float)
    meta = {}

    for line, entry in policy["lines"].items():
        blend = PowerBlend(entry["alpha"], entry["beta"])
        p_blend = blend.transform(panel[entry["model_prob_col"]].to_numpy(float), ref, rid)
        for method in METHODS:
            col = f"{line}__{method}"
            p = normalize_within_race(apply_cal(load_cal(line, method), p_blend, odds), rid)
            panel[col] = p
            meta[col] = {"line": line, "calibrator": method,
                         "alpha": entry["alpha"], "beta": entry["beta"],
                         "model_prob_col": entry["model_prob_col"],
                         "promoted_by_rule": method == entry["promoted_by_rule"],
                         "transferred_blend": False}

    # Transfer arms: the SAME exponents, applied to the other two model families'
    # independent probabilities. Their exponents were selected on the conditional
    # logit's walk-forward panel, not on their own — labelled, never promoted.
    fenty = policy["lines"]["combined_indep_forced"]
    fblend = PowerBlend(fenty["alpha"], fenty["beta"])
    for fam in ("catboost", "lgbm"):
        src = f"{fam}_indep_prob"
        panel[f"{fam}_indep_only__none"] = normalize_within_race(
            panel[src].to_numpy(float), rid)
        meta[f"{fam}_indep_only__none"] = {
            "line": f"{fam}_indep_only", "calibrator": "none", "alpha": 1.0, "beta": 0.0,
            "model_prob_col": src, "promoted_by_rule": False, "transferred_blend": False}
        col = f"{fam}_combined_forced__none"
        panel[col] = fblend.transform(panel[src].to_numpy(float), ref, rid)
        meta[col] = {"line": f"{fam}_combined_forced", "calibrator": "none",
                     "alpha": fblend.alpha, "beta": fblend.beta, "model_prob_col": src,
                     "promoted_by_rule": False, "transferred_blend": True}
    return panel, meta


# ── metrics ──────────────────────────────────────────────────────────────────


def stratum_rows(panel, col, dim, key_col) -> list[dict]:
    y = panel["won"].to_numpy(float)
    p = panel[col].to_numpy(float)
    rid = panel["race_uid"].to_numpy()
    race_level = dim in ("field_size", "regime", "region")
    rows = []
    for key, idx in panel.groupby(key_col, dropna=False).groups.items():
        m = panel.index.isin(idx)
        n = int(m.sum())
        if n == 0:
            continue
        mean_p = float(np.nanmean(p[m]))
        actual = float(y[m].mean())
        rows.append({
            "candidate": col, "dimension": dim, "stratum": str(key),
            "n_runners": n, "n_races": int(pd.unique(rid[m]).size),
            "n_wins": int(y[m].sum()), "actual_rate": actual,
            "mean_prob": mean_p, "calibration_gap": actual - mean_p,
            "ae": (actual / mean_p if mean_p > 0 else np.nan),
            "brier": float(np.nanmean((p[m] - y[m]) ** 2)),
            "race_log_loss": (race_level_log_loss(y[m], p[m], rid[m])
                              if race_level else np.nan),
        })
    return rows


def selected_bet_rows(panel, col) -> list[dict]:
    """EV subset under the candidate's OWN probability at the decision price."""
    ev = panel[col].to_numpy(float) * panel["bet_price"].to_numpy(float) - 1.0
    sel = pd.Series(np.where(ev >= MIN_EV, f"EV>={MIN_EV}", f"EV<{MIN_EV}"),
                    index=panel.index)
    tmp = panel.assign(_sel=sel)
    return stratum_rows(tmp, col, "selected_bet", "_sel")


def bootstrap_gap(panel, col, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Race-clustered bootstrap of (market log loss - candidate log loss)."""
    rng = np.random.default_rng(seed)
    races = panel["race_uid"].drop_duplicates().to_numpy()
    idx_by_race = {r: g.to_numpy() for r, g in panel.groupby("race_uid").groups.items()}
    y = panel["won"].to_numpy(float)
    pm = panel[col].to_numpy(float)
    pk = panel["market_ref_prob"].to_numpy(float)
    pos = {r: i for i, r in enumerate(panel.index)}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        rows = np.concatenate([idx_by_race[r] for r in pick])
        rows = np.array([pos[r] for r in rows])
        # Re-label so resampled copies of one race stay separate groups.
        reps = np.concatenate([[i] * len(idx_by_race[r]) for i, r in enumerate(pick)])
        diffs.append(race_level_log_loss(y[rows], pk[rows], reps)
                     - race_level_log_loss(y[rows], pm[rows], reps))
    d = np.asarray(diffs, dtype=float)
    return {"mean_gap": float(np.mean(d)),
            "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_beats_market": float(np.mean(d > 0)), "n_boot": int(n_boot)}


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    policy = json.load(open(POLICY, "r", encoding="utf-8"))

    panel = add_strata(build_eval_panel(manifest))
    panel, meta = build_candidates(panel, policy)
    print(f"eval panel: {len(panel)} rows / {panel['race_uid'].nunique()} races "
          f"(complete reference books only)")

    scorecard = {
        "manifest": MANIFEST, "policy": POLICY,
        "eval_window": manifest["windows"]["eval"],
        "n_rows": int(len(panel)), "n_races": int(panel["race_uid"].nunique()),
        "base_rate": float(panel["won"].mean()),
        "market_reference": None, "candidates": {},
    }

    for col, m in meta.items():
        h = head_to_head(panel, prob_col=col, odds_col="bet_price",
                         race_id_col="race_uid", label_col="won")
        h.pop("odds_band_table")
        scorecard["market_reference"] = h["market"]
        entry = dict(m)
        entry.update({
            "n_races_kept": h["n_races_kept"], "n_runners": h["n_runners"],
            "log_loss": h["model"]["log_loss"],
            "brier_runner_level": h["model"]["brier_runner_level"],
            "ece": h["model"]["ece"],
            "log_loss_gap_vs_market": h["gaps"]["log_loss"],
            "beats_market_logloss": h["model_beats_market_logloss"],
        })
        if col in HEADLINE:
            entry["bootstrap_race_clustered"] = bootstrap_gap(panel, col)
        scorecard["candidates"][col] = entry
        print(f"{col:38s} LL={entry['log_loss']:.5f} "
              f"gap={entry['log_loss_gap_vs_market']:+.5f} "
              f"brier={entry['brier_runner_level']:.5f} ece={entry['ece']:.5f} "
              f"beats={entry['beats_market_logloss']}")

    # ── breakdowns ───────────────────────────────────────────────────────────
    os.makedirs(BREAKDOWN_DIR, exist_ok=True)
    rows = []
    for col in HEADLINE:
        for dim, key in (("odds_band", "_odds_band"), ("field_size", "_field_band"),
                         ("regime", "_regime"), ("region", "_region")):
            rows += stratum_rows(panel, col, dim, key)
        rows += selected_bet_rows(panel, col)
        rows += stratum_rows(panel.assign(_all="ALL"), col, "overall", "_all")
    bd = pd.DataFrame(rows)
    bd.to_csv(os.path.join(BREAKDOWN_DIR, "calibration_by_stratum.csv"), index=False)
    print(f"wrote {BREAKDOWN_DIR}/calibration_by_stratum.csv ({len(bd)} rows)")

    scorecard["breakdown_csv"] = os.path.join(BREAKDOWN_DIR, "calibration_by_stratum.csv")
    scorecard["breakdown_note"] = (
        "race_log_loss is only defined for race-level strata (field_size, regime, region, "
        "overall); it is NaN for odds_band and selected_bet, which slice RUNNERS inside a "
        "race — those rows report runner-level Brier, A/E and the calibration gap instead."
    )
    scorecard["selected_bet_rule"] = (
        f"EV = candidate_prob * bet_price - 1 >= {MIN_EV} (config.yaml value.min_expected_value). "
        "The subset therefore differs per candidate by construction; n_runners is reported."
    )
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    print(f"wrote {OUT}")

    keep = [c for c in panel.columns if not c.startswith("_")]
    panel[keep].to_parquet(EVAL_PANEL_OUT, engine="pyarrow", index=False)
    print(f"wrote {EVAL_PANEL_OUT}")
    print("EVALUATE_12_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
