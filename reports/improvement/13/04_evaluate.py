"""Step 13: select ensemble-blend weights on blend_dev OOF rows, fit a
calibrator for the blended probability on calib-window OOF rows (both
GENUINELY out-of-fold — no row was ever in its own scoring model's training
set), then score every individual learner AND both ensembles (CatBoost+
LightGBM vs CatBoost+LightGBM+XGBoost, indep and market branches) on the
untouched eval panel (= step 10's dev_oos, 618 races). Nothing here is fit or
selected on the eval panel; the final holdout is never loaded.

Run: .venv/Scripts/python.exe reports/improvement/13/04_evaluate.py
"""
import itertools
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from models.blend import race_level_log_loss
from models.calibration import fit_calibrator, normalize_within_race
from models.head_to_head import head_to_head

# Columns that already sum to 1.0 within race (blend outputs, market_ref_prob)
# and must NOT be renormalised again before scoring — everything else (a raw
# per-runner model probability, e.g. catboost_indep_prob) needs the same
# normalize_within_race step models.head_to_head.head_to_head applies
# internally, or race_level_log_loss silently scores an un-normalised vector
# and disagrees with the head_to_head numbers printed just above it.
_PRE_NORMALISED_PREFIXES = ("blend_", "market_ref_prob")

MANIFEST = "reports/improvement/13/manifest.json"
OOF_PARQUET = "data/audit/13/oof_dev_core.parquet"
EVAL_PANEL = "data/audit/13/eval_panel_scored_with_xgb.parquet"
OUT_SCORECARD = "reports/improvement/13/scorecard_13.json"
OUT_WEIGHTS = "data/audit/13/blend_weights.json"
CALIB_DIR = "data/audit/13/calibrators"

WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
N_BOOT = 2000
BOOT_SEED = 13

EVAL_COL = {  # OOF column name -> eval-panel column name
    "cb_indep": "catboost_indep_prob", "cb_market": "catboost_market_prob",
    "lgbm_indep": "lgbm_indep_prob", "lgbm_market": "lgbm_market_prob",
    "xgb_indep": "xgboost_indep_prob", "xgb_market": "xgboost_market_prob",
}

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


# ── N-model weighted geometric blend (power_blend generalised to N inputs) ──


def _race_codes(race_ids) -> np.ndarray:
    keys = pd.Index(np.asarray(race_ids, dtype=object).astype(str))
    return pd.factorize(keys)[0]


def weighted_blend(prob_cols: dict, weights: dict, race_ids) -> np.ndarray:
    """p ∝ prod_i p_i**w_i, renormalised within race. weight 0 excludes a
    member entirely (never a forced equal share)."""
    n = len(race_ids)
    codes = _race_codes(race_ids)
    active = {k: v for k, v in prob_cols.items() if weights.get(k, 0.0) > 0}
    if not active:
        return np.full(n, np.nan)
    log_p = np.zeros(n)
    usable = np.ones(n, dtype=bool)
    for k, p in active.items():
        p = np.asarray(p, dtype=float)
        usable &= np.isfinite(p) & (p > 0)
        log_p = log_p + weights[k] * np.log(np.clip(p, 1e-12, None))
    complete = pd.Series(usable).groupby(codes).transform("all").to_numpy()
    out = np.full(n, np.nan)
    if not complete.any():
        return out
    lp = log_p[complete]
    grp = pd.Series(lp).groupby(codes[complete])
    z = np.exp(lp - grp.transform("max").to_numpy())
    tot = pd.Series(z).groupby(codes[complete]).transform("sum").to_numpy()
    out[complete] = np.where(tot > 0, z / tot, np.nan)
    return out


def select_weights(prob_cols: dict, names: list, y, race_ids) -> dict:
    """Exhaustive grid search over WEIGHT_GRID**len(names), zero allowed,
    never all-zero. Returns {weights, race_log_loss, n_grid_points}."""
    best = None
    n_points = 0
    for combo in itertools.product(WEIGHT_GRID, repeat=len(names)):
        if not any(combo):
            continue
        n_points += 1
        weights = dict(zip(names, combo))
        p = weighted_blend(prob_cols, weights, race_ids)
        ll = race_level_log_loss(y, p, race_ids)
        if not np.isfinite(ll):
            continue
        if best is None or ll < best["race_log_loss"]:
            best = {"weights": weights, "race_log_loss": float(ll)}
    best["n_grid_points"] = n_points
    return best


def _normalized(panel, col, race_ids) -> np.ndarray:
    """Within-race-normalised probability for `col`, unless it already sums
    to 1.0 by construction (a blend output or market_ref_prob)."""
    p = panel[col].to_numpy(float)
    if col.startswith(_PRE_NORMALISED_PREFIXES):
        return p
    return normalize_within_race(p, race_ids)


def bootstrap_delta(panel, col_a, col_b, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Race-clustered bootstrap of (race_level_log_loss(b) - race_level_log_loss(a)).

    Positive => a has the LOWER (better) loss than b. Mirrors
    reports/improvement/12/05_evaluate.py::bootstrap_gap, generalised to any
    two probability columns (not just market vs one candidate). Both columns
    are within-race-normalised first (matching models.head_to_head.head_to_head's
    own convention) unless already normalised by construction.
    """
    rng = np.random.default_rng(seed)
    races = panel["race_uid"].drop_duplicates().to_numpy()
    idx_by_race = {r: g.to_numpy() for r, g in panel.groupby("race_uid").groups.items()}
    y = panel["won"].to_numpy(float)
    rid = panel["race_uid"].to_numpy()
    pa = _normalized(panel, col_a, rid)
    pb = _normalized(panel, col_b, rid)
    pos = {r: i for i, r in enumerate(panel.index)}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(races, size=len(races), replace=True)
        rows = np.concatenate([idx_by_race[r] for r in pick])
        rows = np.array([pos[r] for r in rows])
        reps = np.concatenate([[i] * len(idx_by_race[r]) for i, r in enumerate(pick)])
        diffs.append(race_level_log_loss(y[rows], pb[rows], reps)
                     - race_level_log_loss(y[rows], pa[rows], reps))
    d = np.asarray(diffs, dtype=float)
    return {"mean_gap": float(np.mean(d)), "ci95_low": float(np.percentile(d, 2.5)),
            "ci95_high": float(np.percentile(d, 97.5)),
            "p_a_beats_b": float(np.mean(d > 0)), "n_boot": int(n_boot)}


def diversity(oof: pd.DataFrame, cols: list) -> dict:
    sub = oof.loc[oof["covered"], cols].dropna()
    corr = sub.corr(method="pearson")
    resid = sub.sub(oof.loc[sub.index, "won"], axis=0)
    resid_corr = resid.corr(method="pearson")
    return {
        "n_rows": int(len(sub)),
        "prob_correlation": corr.round(4).to_dict(),
        "residual_correlation": resid_corr.round(4).to_dict(),
    }


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    oof = pd.read_parquet(OOF_PARQUET)
    panel = pd.read_parquet(EVAL_PANEL)
    log(f"OOF rows: {len(oof)} (covered={int(oof['covered'].sum())}); eval panel: "
        f"{len(panel)} rows / {panel['race_uid'].nunique()} races")

    oof["_day"] = pd.to_datetime(oof["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    bd = manifest["windows"]["blend_dev"]
    cw = manifest["windows"]["calib"]
    blend_dev = oof[oof["covered"] & (oof["_day"] >= pd.Timestamp(bd["start"])) &
                    (oof["_day"] < pd.Timestamp(bd["end_exclusive"]))].reset_index(drop=True)
    calib_win = oof[oof["covered"] & (oof["_day"] >= pd.Timestamp(cw["start"])) &
                    (oof["_day"] < pd.Timestamp(cw["end_exclusive"]))].reset_index(drop=True)
    log(f"blend_dev OOF rows: {len(blend_dev)} / {blend_dev['race_uid'].nunique()} races "
        f"({bd['start']}..{bd['end_exclusive']})")
    log(f"calib OOF rows: {len(calib_win)} / {calib_win['race_uid'].nunique()} races "
        f"({cw['start']}..{cw['end_exclusive']})")

    # ── diversity (all OOF-covered rows) ──────────────────────────────────────
    diversity_report = {
        "indep": diversity(oof, ["cb_indep", "lgbm_indep", "xgb_indep"]),
        "market": diversity(oof, ["cb_market", "lgbm_market", "xgb_market"]),
    }

    # ── select blend weights on blend_dev (OOF only) ──────────────────────────
    weights_report = {}
    os.makedirs(CALIB_DIR, exist_ok=True)
    for branch in ("indep", "market"):
        cb, lg, xg = f"cb_{branch}", f"lgbm_{branch}", f"xgb_{branch}"
        prob_cols_bd = {cb: blend_dev[cb].to_numpy(), lg: blend_dev[lg].to_numpy(), xg: blend_dev[xg].to_numpy()}
        y_bd = blend_dev["won"].to_numpy()
        rid_bd = blend_dev["race_uid"].to_numpy()

        w2 = select_weights(prob_cols_bd, [cb, lg], y_bd, rid_bd)
        w3 = select_weights(prob_cols_bd, [cb, lg, xg], y_bd, rid_bd)
        weights_report[branch] = {"cb_lgbm": w2, "cb_lgbm_xgb": w3}
        log(f"[{branch}] cb_lgbm weights={w2['weights']} dev_race_ll={w2['race_log_loss']:.5f} "
            f"(n_grid={w2['n_grid_points']})")
        log(f"[{branch}] cb_lgbm_xgb weights={w3['weights']} dev_race_ll={w3['race_log_loss']:.5f} "
            f"(n_grid={w3['n_grid_points']}, xgb_weight={w3['weights'][xg]})")

        # ── fit a calibrator for each blend on calib-window OOF rows ──────────
        prob_cols_cw = {cb: calib_win[cb].to_numpy(), lg: calib_win[lg].to_numpy(), xg: calib_win[xg].to_numpy()}
        y_cw = calib_win["won"].to_numpy()
        rid_cw = calib_win["race_uid"].to_numpy()
        for label, w in (("cb_lgbm", w2["weights"]), ("cb_lgbm_xgb", w3["weights"])):
            p_cw = weighted_blend(prob_cols_cw, w, rid_cw)
            ok = np.isfinite(p_cw) & np.isfinite(y_cw)
            cal, method = fit_calibrator("auto", p_cw[ok], y_cw[ok].astype(int))
            with open(os.path.join(CALIB_DIR, f"{branch}__{label}.pkl"), "wb") as fh:
                pickle.dump(cal, fh)
            weights_report[branch][label]["calibration_method"] = method
            weights_report[branch][label]["n_calib_rows"] = int(ok.sum())
            log(f"[{branch}] {label}: calibrator={method} (n={int(ok.sum())})")

    with open(OUT_WEIGHTS, "w", encoding="utf-8") as fh:
        json.dump(weights_report, fh, indent=2, default=str)
    log(f"wrote {OUT_WEIGHTS}")

    # ── apply to the eval panel (report only — nothing fit here) ──────────────
    rid_ev = panel["race_uid"].to_numpy()
    for branch in ("indep", "market"):
        cb, lg, xg = f"cb_{branch}", f"lgbm_{branch}", f"xgb_{branch}"
        ev_cols = {cb: panel[EVAL_COL[cb]].to_numpy(), lg: panel[EVAL_COL[lg]].to_numpy(),
                  xg: panel[EVAL_COL[xg]].to_numpy()}
        for label in ("cb_lgbm", "cb_lgbm_xgb"):
            entry = weights_report[branch][label]
            p_raw = weighted_blend(ev_cols, entry["weights"], rid_ev)
            with open(os.path.join(CALIB_DIR, f"{branch}__{label}.pkl"), "rb") as fh:
                cal = pickle.load(fh)
            finite = np.isfinite(p_raw)
            p_cal = p_raw.copy()
            p_cal[finite] = np.asarray(cal.predict(p_raw[finite]), dtype=float)
            panel[f"blend_{branch}_{label}__raw"] = normalize_within_race(p_raw, rid_ev)
            panel[f"blend_{branch}_{label}__calib"] = normalize_within_race(p_cal, rid_ev)

    # ── score every individual learner + both ensembles ────────────────────────
    scorecard = {
        "manifest": MANIFEST, "eval_panel": EVAL_PANEL,
        "n_rows": int(len(panel)), "n_races": int(panel["race_uid"].nunique()),
        "diversity": diversity_report, "blend_weights": weights_report,
        "candidates": {},
    }
    individual = {
        "condlogit_indep": "condlogit_indep_prob", "condlogit_market": "condlogit_market_prob",
        "catboost_indep": "catboost_indep_prob", "catboost_market": "catboost_market_prob",
        "lgbm_indep": "lgbm_indep_prob", "lgbm_market": "lgbm_market_prob",
        "xgboost_indep": "xgboost_indep_prob", "xgboost_market": "xgboost_market_prob",
    }
    ensemble = {
        "cb_lgbm_indep__raw": "blend_indep_cb_lgbm__raw", "cb_lgbm_indep__calib": "blend_indep_cb_lgbm__calib",
        "cb_lgbm_xgb_indep__raw": "blend_indep_cb_lgbm_xgb__raw", "cb_lgbm_xgb_indep__calib": "blend_indep_cb_lgbm_xgb__calib",
        "cb_lgbm_market__raw": "blend_market_cb_lgbm__raw", "cb_lgbm_market__calib": "blend_market_cb_lgbm__calib",
        "cb_lgbm_xgb_market__raw": "blend_market_cb_lgbm_xgb__raw", "cb_lgbm_xgb_market__calib": "blend_market_cb_lgbm_xgb__calib",
    }
    all_candidates = {**individual, **ensemble}
    for name, col in all_candidates.items():
        h2h = head_to_head(panel.dropna(subset=[col]), prob_col=col, odds_col="bet_price",
                           race_id_col="race_uid", label_col="won")
        h2h.pop("odds_band_table")
        scorecard["candidates"][name] = h2h
        log(f"[{name}] model_ll={h2h['model']['log_loss']:.5f} market_ll={h2h['market']['log_loss']:.5f} "
            f"beats_market={h2h['model_beats_market_logloss']} n_races={h2h['n_races_kept']}")

    # ── race-clustered bootstrap: each candidate vs market, and 3-model vs 2-model ──
    boot = {}
    for name, col in all_candidates.items():
        sub = panel.dropna(subset=[col, "market_ref_prob"]).reset_index(drop=True)
        boot[f"{name}_vs_market"] = bootstrap_delta(sub, col, "market_ref_prob")
    for branch in ("indep", "market"):
        for suffix in ("raw", "calib"):
            a = f"blend_{branch}_cb_lgbm_xgb__{suffix}"   # 3-model
            b = f"blend_{branch}_cb_lgbm__{suffix}"        # 2-model
            sub = panel.dropna(subset=[a, b]).reset_index(drop=True)
            boot[f"3model_minus_2model_{branch}_{suffix}"] = bootstrap_delta(sub, a, b)
            log(f"[3-model vs 2-model, {branch}/{suffix}] "
                f"{boot[f'3model_minus_2model_{branch}_{suffix}']}")
    scorecard["bootstrap_race_clustered"] = boot

    with open(OUT_SCORECARD, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    log(f"wrote {OUT_SCORECARD}")
    print("EVALUATE_13_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
