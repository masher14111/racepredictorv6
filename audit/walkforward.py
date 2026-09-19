"""Expanding-window walk-forward audit driver (requirements 3, 5, 6).

Per fold, strictly inside the fold's TRAIN slice (test rows are never touched):

* CatBoost fit on the train core (production v3nf recipe: tuned params from the
  shipped meta, odds-inverse sample weights, 10% early-stopping tail);
* the base per-prob calibrator on the train calibration tail (``auto`` —
  isotonic vs sigmoid chosen within the tail, exactly ``models.train``);
* the favourite-longshot OddsBandCalibrator on the same tail (both the
  production ``isotonic`` band method and a ``sigmoid``-band arm, so the
  probability-extreme fix can be selected on validation folds, never the final
  test);
* a race-level temperature ``T`` for grouped softmax scaling (requirement 5's
  coherent race-level alternative), fit by minimising race-level NLL on the
  tail.

The scored output carries, per runner row: ``p_raw`` (uncalibrated),
``p_ind`` (base-calibrated, price-free — the INDEPENDENT line), ``p_adj``
(F-L market-adjusted, isotonic bands), ``p_adj_sig`` (F-L, sigmoid bands), and
the race-coherent variants ``norm_ind`` / ``norm_adj`` / ``norm_temp`` /
``norm_raw``. Cross-fitting for the F-L recalibrator is inherent: every row's
``p_adj`` comes from a recalibrator fit on data strictly before its fold's
origin.

Nothing here writes model artifacts; per-fold models are transient. Fold
metadata (dates, sizes, chosen calibration method, temperature, F-L band
support, isotonic endpoint values) is returned for the report.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from backtest.splitter import WalkForwardConfig, walk_forward_folds
from models.calibration import (
    OddsBandCalibrator,
    fit_calibrator,
    normalize_within_race,
)
from models.features import PRICE_FREE_FEATURE_COLS
from utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


@dataclass
class AuditWFConfig:
    """Walk-forward audit knobs. Defaults mirror the production v3nf recipe."""

    feature_cols: list = dc_field(default_factory=lambda: list(PRICE_FREE_FEATURE_COLS))
    min_train_days: int = 365
    test_window_days: int = 30
    final_test_start: Optional[str] = None   # folds never test on/after this date
    iterations: int = 1000
    early_stopping_rounds: int = 50
    best_params: dict = dc_field(default_factory=dict)
    calibration_method: str = "auto"
    calibration_frac: float = 0.15
    min_calib_rows: int = 200
    fl_min_rows: int = 400
    use_sample_weights: bool = True
    max_sample_weight: float = 20.0
    task_type: str = "GPU"
    devices: str = "0"
    thread_count: int = -1
    random_seed: int = 42


def _sample_weights(df: pd.DataFrame, max_weight: float) -> np.ndarray:
    """models.train._sample_weights, verbatim semantics."""
    ip = pd.to_numeric(df.get("implied_prob"), errors="coerce")
    ip = ip.fillna(1.0).clip(lower=1.0 / max_weight)
    return (1.0 / ip).clip(upper=max_weight).to_numpy(dtype=float)


def _matrix(df: pd.DataFrame, cols: list) -> np.ndarray:
    X = pd.DataFrame(index=df.index)
    for c in cols:
        X[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return X.astype(float).to_numpy()


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(q / (1.0 - q))


def grouped_softmax(scores: np.ndarray, race_ids: np.ndarray,
                    temperature: float = 1.0) -> np.ndarray:
    """Race-coherent softmax over per-runner scores at a given temperature."""
    s = np.asarray(scores, dtype=float) / max(float(temperature), 1e-6)
    out = np.full(len(s), np.nan)
    ser = pd.Series(s)
    codes = pd.factorize(pd.Index(race_ids))[0]
    for code in np.unique(codes):
        m = codes == code
        z = s[m]
        z = z - np.nanmax(z)
        e = np.exp(z)
        tot = np.nansum(e)
        out[m] = e / tot if tot > 0 else np.nan
    _ = ser  # keep pandas import honest
    return out


def fit_temperature(p_ind: np.ndarray, won: np.ndarray,
                    race_ids: np.ndarray) -> float:
    """Fit the grouped-softmax temperature by race-level NLL on (p_ind, won).

    Scores are the independent probs' logits; T=1 reproduces a plain
    logit-softmax. Races without exactly >=1 recorded winner are skipped.
    """
    s = _logit(p_ind)
    won = np.asarray(won, dtype=float)
    rid = np.asarray(race_ids)

    def nll(T: float) -> float:
        q = grouped_softmax(s, rid, temperature=T)
        per_race = []
        for r in pd.unique(rid):
            m = rid == r
            w = won[m].astype(bool)
            if w.any() and np.isfinite(q[m][w]).all():
                per_race.append(-np.log(np.clip(q[m][w], _EPS, 1.0)).sum())
        return float(np.mean(per_race)) if per_race else np.inf

    res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-3})
    return float(res.x) if res.success else 1.0


def _iso_endpoints(fl: OddsBandCalibrator) -> list[dict]:
    """Support + endpoint values per F-L band (for the extremes report)."""
    out = []
    for centre, cal in zip(fl.centers, fl.calibrators):
        row = {"log_odds_centre": round(float(centre), 4),
               "odds_centre": round(float(np.exp(centre)), 2),
               "type": type(cal).__name__}
        if hasattr(cal, "x"):
            row.update(x_min=cal.x[0], x_max=cal.x[-1],
                       y_min=cal.y[0], y_max=cal.y[-1], n_thresholds=len(cal.x))
        out.append(row)
    return out


def run_walkforward_audit(panel: pd.DataFrame, cfg: AuditWFConfig
                          ) -> tuple[pd.DataFrame, list[dict]]:
    """Score every fold's test window with all probability lines/arms.

    ``panel`` must carry: race_date, race_uid, horse_id, won, bet_price,
    close_price, and the feature columns. Returns (scored frame, fold meta).
    """
    from catboost import CatBoostClassifier

    panel = panel.sort_values("race_date").reset_index(drop=True)
    rd = pd.to_datetime(panel["race_date"], utc=True, errors="coerce")

    final_start = (pd.to_datetime(cfg.final_test_start, utc=True)
                   if cfg.final_test_start else None)

    folds = walk_forward_folds(panel["race_date"], WalkForwardConfig(
        min_train_days=cfg.min_train_days,
        test_window_days=cfg.test_window_days))
    if final_start is not None:
        folds = [f for f in folds if f.test_end < final_start]
    if not folds:
        raise ValueError("no usable walk-forward folds before the final test window")

    scored_parts, meta = [], []
    for fold in folds:
        t0 = time.time()
        tr = fold.train_mask(rd).to_numpy()
        te = fold.test_mask(rd).to_numpy()
        train_df = panel.loc[tr]
        test_df = panel.loc[te]
        info = {"fold": fold.index, "train_end": str(fold.train_end.date()),
                "test_start": str(fold.test_start.date()),
                "test_end": str(fold.test_end.date()),
                "n_train": int(tr.sum()), "n_test": int(te.sum())}
        if info["n_test"] == 0 or train_df["won"].nunique() < 2:
            info["scored"] = False
            meta.append(info)
            continue

        # ── inside-fold fitting (train slice only) ────────────────────────────
        train_df = train_df.sort_values("race_date").reset_index(drop=True)
        X = _matrix(train_df, cfg.feature_cols)
        y = pd.to_numeric(train_df["won"], errors="coerce").fillna(0).astype(int).to_numpy()
        w = (_sample_weights(train_df, cfg.max_sample_weight)
             if cfg.use_sample_weights else np.ones(len(train_df)))

        n_cal = int(len(X) * cfg.calibration_frac)
        if n_cal < cfg.min_calib_rows or (len(X) - n_cal) < cfg.min_calib_rows:
            n_cal = 0
        if n_cal:
            X_core, y_core, w_core = X[:-n_cal], y[:-n_cal], w[:-n_cal]
            cal_df = train_df.iloc[-n_cal:]
            X_cal, y_cal = X[-n_cal:], y[-n_cal:]
        else:
            X_core, y_core, w_core = X, y, w
            cal_df = X_cal = y_cal = None

        n_val = max(1, int(len(X_core) * 0.10))
        params = {
            "iterations": cfg.iterations,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "early_stopping_rounds": cfg.early_stopping_rounds,
            "random_seed": cfg.random_seed,
            "verbose": False,
            "allow_writing_files": False,
            **({"task_type": "GPU", "devices": cfg.devices}
               if str(cfg.task_type).upper() == "GPU"
               else {"task_type": "CPU", "thread_count": cfg.thread_count}),
            **{k: v for k, v in (cfg.best_params or {}).items() if v is not None},
        }
        model = CatBoostClassifier(**params)
        model.fit(X_core[:-n_val], y_core[:-n_val], sample_weight=w_core[:-n_val],
                  eval_set=(X_core[-n_val:], y_core[-n_val:]), verbose=False)

        calibrator = fl_iso = fl_sig = None
        method_used = None
        temperature = 1.0
        if X_cal is not None and len(np.unique(y_cal)) > 1:
            raw_cal = model.predict_proba(X_cal)[:, 1]
            calibrator, method_used = fit_calibrator(cfg.calibration_method,
                                                     raw_cal, y_cal)
            ind_cal = np.asarray(calibrator.predict(raw_cal), dtype=float)
            odds_cal = pd.to_numeric(cal_df["bet_price"], errors="coerce").to_numpy()
            try:
                fl_iso = OddsBandCalibrator.fit(ind_cal, odds_cal, y_cal,
                                                method="isotonic",
                                                min_rows=cfg.fl_min_rows)
            except ValueError as exc:
                logger.warning("audit.wf fold %d: F-L isotonic skipped (%s)",
                               fold.index, exc)
            try:
                fl_sig = OddsBandCalibrator.fit(ind_cal, odds_cal, y_cal,
                                                method="sigmoid",
                                                min_rows=cfg.fl_min_rows)
            except ValueError as exc:
                logger.warning("audit.wf fold %d: F-L sigmoid skipped (%s)",
                               fold.index, exc)
            temperature = fit_temperature(
                ind_cal, y_cal, cal_df["race_uid"].to_numpy())

        # ── score the fold's test window ──────────────────────────────────────
        Xt = _matrix(test_df, cfg.feature_cols)
        p_raw = np.asarray(model.predict_proba(Xt)[:, 1], dtype=float)
        p_ind = (np.asarray(calibrator.predict(p_raw), dtype=float)
                 if calibrator is not None else p_raw.copy())
        odds = pd.to_numeric(test_df["bet_price"], errors="coerce").to_numpy(float)

        def _fl(fl_cal, base):
            if fl_cal is None:
                return base.copy()
            out = base.copy()
            ok = np.isfinite(out) & np.isfinite(odds) & (odds > 1.0)
            if ok.any():
                recal = np.asarray(fl_cal.predict(out[ok], odds[ok]), dtype=float)
                good = np.isfinite(recal)
                out[np.flatnonzero(ok)[good]] = recal[good]
            return out

        p_adj = _fl(fl_iso, p_ind)
        p_adj_sig = _fl(fl_sig, p_ind)
        rid = test_df["race_uid"].to_numpy()

        part = test_df[["race_date", "race_uid", "horse_id", "won",
                        "bet_price", "close_price"]].copy()
        for extra in ("venue", "field_size_panel", "book_complete", "region",
                      "race_class", "data_completeness"):
            if extra in test_df.columns:
                part[extra] = test_df[extra].to_numpy()
        part["fold"] = fold.index
        part["p_raw"] = p_raw
        part["p_ind"] = p_ind
        part["p_adj"] = p_adj
        part["p_adj_sig"] = p_adj_sig
        part["norm_raw"] = normalize_within_race(p_raw, rid)
        part["norm_ind"] = normalize_within_race(p_ind, rid)
        part["norm_adj"] = normalize_within_race(p_adj, rid)
        part["norm_temp"] = grouped_softmax(_logit(p_ind), rid,
                                            temperature=temperature)
        scored_parts.append(part)

        info.update(scored=True, calib_method=method_used,
                    temperature=round(temperature, 4),
                    best_iteration=int(model.get_best_iteration() or 0),
                    fit_seconds=round(time.time() - t0, 1),
                    fl_iso_bands=_iso_endpoints(fl_iso) if fl_iso else None)
        meta.append(info)
        logger.info("audit.wf fold %d: %d train / %d test, calib=%s, T=%.3f, %.1fs",
                    fold.index, info["n_train"], info["n_test"],
                    method_used, temperature, info["fit_seconds"])

    scored = pd.concat(scored_parts, ignore_index=True) if scored_parts \
        else pd.DataFrame()
    return scored, meta
