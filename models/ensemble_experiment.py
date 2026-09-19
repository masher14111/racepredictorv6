"""Ensemble experiment: does blending beat the single CatBoost on win prob?

Question (memory/model-09-baseline-audit.md, prompt 13): test whether an
**ensemble** of diverse learners produces a better *calibrated* win probability
than the single CatBoost the pipeline ships today.

Design — everything below mirrors ``models/train.py`` so the numbers are
comparable to the production model, and every learner sees the *same* rows:

* **Feature set** — the price-free whitelist (``PRICE_FREE_FEATURE_COLS``). The
  priced ``v3`` model builds its market features from ``odds_finish`` (the
  finishing SP), which is outcome-adjacent leakage (audit C2), so improving it
  is meaningless. ``v3nf`` is the trustworthy, calibrated model and is the only
  honest baseline to beat. The market is folded back in *separately* (see the
  ``+market`` rows) so its contribution is isolated, not smuggled in.
* **Split** — sort by ``race_date``; last 20 % is the held-out test tail
  (identical to ``train._time_split``). The most-recent 15 % of train is carved
  off as the time-ordered **calibration slice**; base models never see it.
* **Weights** — odds-inverse sample weights, cap 20 (``train._sample_weights``),
  applied to every learner so fitting parity holds.
* **Calibration** — ``fit_calibrator("auto", ...)`` on the calibration slice,
  exactly as production. Every probability reported is *calibrated*, because the
  served probability is what matters; raw AUC is unchanged by monotone calib.
* **Base learners** — CatBoost (tuned ``v3nf`` params), XGBoost (diverse GBDT),
  and a regularised LogisticRegression on a median-imputed, standardised subset.
* **Blends** — (a) simple average of the three calibrated probs; (b) **stacking**
  with a logistic meta-learner fit on out-of-fold base predictions (TimeSeries
  OOF, no leakage); (c) each blend ``+market`` (implied_prob as an extra stack
  input / averaged component).

Run:  ``python -m models.ensemble_experiment``  (writes a JSON report under
``docs/calibration/`` and prints a metrics table).  Pure experiment — imports
nothing from this module into the serving path.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from models.calibration import brier_score, fit_calibrator
from models.features import PRICE_FREE_FEATURE_COLS
from models.split_utils import group_time_series_split
from models.targets import add_targets
from models.train import _group_carve, _sample_weights, _time_split
from models.tuner import _hardware_params
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_PARQUET = os.path.join(_BASE, "data", "features", "training.parquet")
_REPORT_DIR = os.path.join(_BASE, "docs", "calibration")

TARGET = "won"
CALIB_FRAC = 0.15          # most-recent fraction of train held out to calibrate
OOF_SPLITS = 4             # TimeSeriesSplit folds for the stacker's OOF preds
CB_ITERS = 600             # fixed iters (no early stopping) → clean OOF/refit parity
XGB_ROUNDS = 600
RNG = 42

# Columns that are 100 % null in the matrix (paywalled / not yet wired). Trees
# ignore them; the linear learner must drop them (a constant column is useless
# and the scaler would divide by zero).
_DEAD_COLS = {"class_change", "recent_form_avg", "pace_bias",
              "timeform_rating", "rating_rank"}


# ── metrics ──────────────────────────────────────────────────────────────────

def ece(y, p, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins) — matches the audit's ECE."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(y)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(total)


def metrics(y, p) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "brier": brier_score(y, p),
        "ece": ece(y, p),
    }


# ── base learners ────────────────────────────────────────────────────────────

def _cb_params() -> dict:
    """Tuned v3nf 'won' params + production hardware + Balanced weighting."""
    cfg = {"task_type": "GPU", "devices": "0", "thread_count": -1}
    # mirror train.py's GPU→CPU guard
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() <= 0:
            cfg["task_type"] = "CPU"
    except Exception:  # noqa: BLE001
        cfg["task_type"] = "CPU"
    return {
        "iterations": CB_ITERS,
        "learning_rate": 0.0128,
        "depth": 4,
        "l2_leaf_reg": 9.70,
        "bagging_temperature": 0.442,
        "random_strength": 0.0342,
        "border_count": 46,
        "loss_function": "Logloss",
        "auto_class_weights": "Balanced",
        "verbose": False,
        "allow_writing_files": False,
        "random_seed": RNG,
        **_hardware_params(cfg),
    }


def fit_catboost(X, y, w):
    m = CatBoostClassifier(**_cb_params())
    m.fit(X, y, sample_weight=w, verbose=False)
    return lambda Z: m.predict_proba(Z)[:, 1], m


def fit_xgboost(X, y, w):
    from xgboost import XGBClassifier
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    spw = (neg / pos) if pos > 0 else 1.0
    m = XGBClassifier(
        n_estimators=XGB_ROUNDS,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=spw,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RNG,
    )
    m.fit(X, y, sample_weight=w, verbose=False)
    return lambda Z: m.predict_proba(Z)[:, 1], m


def fit_logreg(X, y, w, keep_idx):
    """Median-impute + standardise a curated (non-dead) subset, then L2 logistic.

    keep_idx selects the columns that aren't 100 % null; the pipeline carries the
    column slice so the returned predictor accepts the *full* feature matrix.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, class_weight="balanced",
                                  max_iter=2000, random_state=RNG)),
    ])
    pipe.fit(X[:, keep_idx], y, lr__sample_weight=w)
    return lambda Z: pipe.predict_proba(Z[:, keep_idx])[:, 1], pipe


# ── experiment ───────────────────────────────────────────────────────────────

@dataclass
class Slices:
    X_core: np.ndarray
    y_core: np.ndarray
    w_core: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    ip_cal: np.ndarray
    ip_te: np.ndarray
    ip_core: np.ndarray
    feat_cols: list = field(default_factory=list)
    keep_idx: np.ndarray = None
    race_uid_core: np.ndarray = None
    race_date_core: np.ndarray = None


def _prep() -> Slices:
    logger.info("ensemble: loading %s", _TRAIN_PARQUET)
    df = pd.read_parquet(_TRAIN_PARQUET)
    df = add_targets(df, 3)
    df["_sample_weight"] = _sample_weights(df, 20.0)

    # Whole-race chronological splits throughout (models.split_utils, step 02)
    # — never a row-count cut, so no race_uid is ever shared across train/test
    # or the calibration carve below.
    train_df, test_df = _time_split(df, 0.2)
    core_df, cal_df = _group_carve(train_df, CALIB_FRAC)

    feat_cols = [c for c in PRICE_FREE_FEATURE_COLS if c in df.columns]
    keep_idx = np.array([i for i, c in enumerate(feat_cols) if c not in _DEAD_COLS])

    def _xy(frame):
        mask = frame[TARGET].notna().values
        X = frame.loc[mask, feat_cols].astype(float).values
        y = frame.loc[mask, TARGET].astype(int).values
        w = frame.loc[mask, "_sample_weight"].to_numpy(dtype=float)
        ip = pd.to_numeric(frame.loc[mask, "implied_prob"],
                           errors="coerce").to_numpy(dtype=float)
        race_uid = frame.loc[mask, "race_uid"].to_numpy()
        race_date = frame.loc[mask, "race_date"].to_numpy()
        return X, y, w, ip, race_uid, race_date

    X_core, y_core, w_core, ip_core, race_uid_core, race_date_core = _xy(core_df)
    X_cal, y_cal, _w_cal, ip_cal, _ruc, _rdc = _xy(cal_df)
    X_te, y_te, _w_te, ip_te, _rut, _rdt = _xy(test_df)

    logger.info("ensemble: core=%d cal=%d test=%d feats=%d (linear uses %d)",
                len(X_core), len(X_cal), len(X_te), len(feat_cols), len(keep_idx))
    return Slices(X_core, y_core, w_core, X_cal, y_cal, X_te, y_te,
                  ip_cal, ip_te, ip_core, feat_cols, keep_idx,
                  race_uid_core, race_date_core)


def _calibrated(raw_cal, y_cal, raw_other):
    """Fit an auto calibrator on (raw_cal, y_cal); return calibrated raw_other."""
    cal, method = fit_calibrator("auto", raw_cal, y_cal)
    return np.asarray(cal.predict(raw_other), dtype=float), method


def _oof_base_preds(s: Slices) -> dict:
    """Out-of-fold base predictions over the core (for the stacker, no leakage).

    Whole-race-group-safe walk-forward split (models.split_utils, step 02):
    fold k trains on whole races strictly earlier than its validation races, and
    no race's rows are ever split across a fold boundary — a plain row-indexed
    TimeSeriesSplit cannot guarantee that. The first fold's training block has
    no OOF coverage, so we return the boolean mask of rows that actually got a
    prediction alongside the prediction columns.
    """
    n = len(s.X_core)
    oof = {k: np.full(n, np.nan) for k in ("cb", "xgb", "lr")}
    covered = np.zeros(n, dtype=bool)
    folds = group_time_series_split(s.race_uid_core, s.race_date_core, OOF_SPLITS)
    for fold, (tr, va) in enumerate(folds):
        logger.info("ensemble: OOF fold %d/%d (train=%d val=%d)",
                    fold + 1, OOF_SPLITS, len(tr), len(va))
        Xtr, ytr, wtr = s.X_core[tr], s.y_core[tr], s.w_core[tr]
        if len(np.unique(ytr)) < 2:
            continue
        cb, _ = fit_catboost(Xtr, ytr, wtr)
        xgb, _ = fit_xgboost(Xtr, ytr, wtr)
        lr, _ = fit_logreg(Xtr, ytr, wtr, s.keep_idx)
        oof["cb"][va] = cb(s.X_core[va])
        oof["xgb"][va] = xgb(s.X_core[va])
        oof["lr"][va] = lr(s.X_core[va])
        covered[va] = True
    oof["_covered"] = covered
    return oof


def run() -> dict:
    t0 = time.time()
    s = _prep()

    # ── 1. fit base learners on the core; calibrate each on the cal slice ──
    logger.info("ensemble: fitting base learners on core")
    cb_pred, _ = fit_catboost(s.X_core, s.y_core, s.w_core)
    xgb_pred, _ = fit_xgboost(s.X_core, s.y_core, s.w_core)
    lr_pred, _ = fit_logreg(s.X_core, s.y_core, s.w_core, s.keep_idx)

    raw = {
        "cb":  (cb_pred(s.X_cal),  cb_pred(s.X_te)),
        "xgb": (xgb_pred(s.X_cal), xgb_pred(s.X_te)),
        "lr":  (lr_pred(s.X_cal),  lr_pred(s.X_te)),
    }

    results: dict = {}
    cal_methods: dict = {}
    calib_te: dict = {}   # calibrated test prob per base learner
    calib_cal: dict = {}  # calibrated cal-slice prob per base learner
    for name, (rc, rt) in raw.items():
        ct, method = _calibrated(rc, s.y_cal, rt)
        cc, _ = _calibrated(rc, s.y_cal, rc)
        calib_te[name] = ct
        calib_cal[name] = cc
        cal_methods[name] = method
        results[name] = metrics(s.y_te, ct)

    # ── 2. simple average of the three calibrated probs (then re-calibrate) ──
    avg_cal = np.mean([calib_cal[n] for n in ("cb", "xgb", "lr")], axis=0)
    avg_te = np.mean([calib_te[n] for n in ("cb", "xgb", "lr")], axis=0)
    results["avg_raw"] = metrics(s.y_te, avg_te)
    avg_te_c, _ = _calibrated(avg_cal, s.y_cal, avg_te)
    results["avg_calib"] = metrics(s.y_te, avg_te_c)

    # ── 3. stacking: logistic meta-learner on OOF base predictions ──
    oof = _oof_base_preds(s)
    cov = oof["_covered"]
    meta_X = np.column_stack([oof["cb"][cov], oof["xgb"][cov], oof["lr"][cov]])
    meta_y = s.y_core[cov]
    meta = LogisticRegression(C=1.0, max_iter=2000, random_state=RNG)
    meta.fit(meta_X, meta_y)

    # base learners are already fit on the full core → stack their cal/test preds
    stk_cal_X = np.column_stack([raw["cb"][0], raw["xgb"][0], raw["lr"][0]])
    stk_te_X = np.column_stack([raw["cb"][1], raw["xgb"][1], raw["lr"][1]])
    stk_cal = meta.predict_proba(stk_cal_X)[:, 1]
    stk_te = meta.predict_proba(stk_te_X)[:, 1]
    results["stack_raw"] = metrics(s.y_te, stk_te)
    stk_te_c, stk_method = _calibrated(stk_cal, s.y_cal, stk_te)
    results["stack_calib"] = metrics(s.y_te, stk_te_c)
    stack_coef = {"cb": float(meta.coef_[0, 0]),
                  "xgb": float(meta.coef_[0, 1]),
                  "lr": float(meta.coef_[0, 2]),
                  "intercept": float(meta.intercept_[0])}

    # ── 4. +market variants (isolate the market's contribution) ──
    # Market implied prob folded in as an extra stack input. NOTE: implied_prob
    # in this matrix derives from odds_finish (finishing SP) — audit C2 leakage —
    # so these rows are an UPPER BOUND on the live market gain, not a live number.
    mkt_cal = np.nan_to_num(s.ip_cal, nan=np.nanmedian(s.ip_cal))
    mkt_te = np.nan_to_num(s.ip_te, nan=np.nanmedian(s.ip_te))
    results["market_only"] = metrics(s.y_te, mkt_te)

    meta_Xm = np.column_stack([meta_X, mkt_cal_for_oof(oof, cov, s)])
    meta_m = LogisticRegression(C=1.0, max_iter=2000, random_state=RNG)
    meta_m.fit(meta_Xm, meta_y)
    stk_cal_Xm = np.column_stack([stk_cal_X, mkt_cal])
    stk_te_Xm = np.column_stack([stk_te_X, mkt_te])
    stk_te_m = meta_m.predict_proba(stk_te_Xm)[:, 1]
    stk_cal_m = meta_m.predict_proba(stk_cal_Xm)[:, 1]
    stk_te_mc, _ = _calibrated(stk_cal_m, s.y_cal, stk_te_m)
    results["stack+market_calib"] = metrics(s.y_te, stk_te_mc)
    stack_market_coef = {"cb": float(meta_m.coef_[0, 0]),
                         "xgb": float(meta_m.coef_[0, 1]),
                         "lr": float(meta_m.coef_[0, 2]),
                         "market": float(meta_m.coef_[0, 3]),
                         "intercept": float(meta_m.intercept_[0])}

    # ── 5. inference latency (per 1000 runners, calibrated) ──
    latency = _latency(s, cb_pred, xgb_pred, lr_pred, meta)

    elapsed = time.time() - t0
    report = {
        "target": TARGET,
        "feature_cols": s.feat_cols,
        "rows": {"core": len(s.X_core), "cal": len(s.X_cal), "test": len(s.X_te)},
        "base_rate_test": float(s.y_te.mean()),
        "calibration_methods": cal_methods,
        "stack_method": stk_method,
        "stack_coef": stack_coef,
        "stack_market_coef": stack_market_coef,
        "metrics": results,
        "latency_ms_per_1k": latency,
        "wall_seconds": round(elapsed, 1),
    }
    _write_report(report)
    _print_table(report)
    return report


def mkt_cal_for_oof(oof, cov, s: Slices):
    """Market implied prob aligned to the OOF-covered core rows (median-filled)."""
    filled = np.nan_to_num(s.ip_core, nan=np.nanmedian(s.ip_core))
    return filled[cov]


def _latency(s, cb_pred, xgb_pred, lr_pred, meta) -> dict:
    Z = s.X_te[: min(1000, len(s.X_te))]
    scale = 1000.0 / len(Z)

    def _t(fn):
        t = time.time()
        fn()
        return round((time.time() - t) * 1000.0 * scale, 2)

    cb_ms = _t(lambda: cb_pred(Z))
    xgb_ms = _t(lambda: xgb_pred(Z))
    lr_ms = _t(lambda: lr_pred(Z))

    def _stack():
        cols = np.column_stack([cb_pred(Z), xgb_pred(Z), lr_pred(Z)])
        meta.predict_proba(cols)
    stack_ms = _t(_stack)
    return {"catboost": cb_ms, "xgboost": xgb_ms, "logreg": lr_ms,
            "stack_total": stack_ms}


def _write_report(report: dict) -> None:
    os.makedirs(_REPORT_DIR, exist_ok=True)
    path = os.path.join(_REPORT_DIR, "ensemble_experiment.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info("ensemble: wrote %s", path)


def _print_table(report: dict) -> None:
    print("\n=== Ensemble experiment — target=%s  (test n=%d, base rate %.3f) ===" %
          (report["target"], report["rows"]["test"], report["base_rate_test"]))
    print("calibration:", report["calibration_methods"], "| stack:", report["stack_method"])
    print(f"{'model':<22}{'AUC':>9}{'logloss':>10}{'Brier':>9}{'ECE':>9}")
    order = ["cb", "xgb", "lr", "market_only", "avg_raw", "avg_calib",
             "stack_raw", "stack_calib", "stack+market_calib"]
    for k in order:
        m = report["metrics"].get(k)
        if not m:
            continue
        print(f"{k:<22}{m['auc']:>9.4f}{m['logloss']:>10.4f}"
              f"{m['brier']:>9.4f}{m['ece']:>9.4f}")
    print("\nstack coef:", report["stack_coef"])
    print("stack+market coef:", report["stack_market_coef"])
    print("latency ms/1k:", report["latency_ms_per_1k"])
    print("wall seconds:", report["wall_seconds"])


if __name__ == "__main__":
    run()
