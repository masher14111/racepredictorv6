"""Pluggable per-fold model factory for the walk-forward engine.

The engine asks a factory to ``fit`` on a fold's *train* slice and then
``predict`` win probabilities for that fold's *test* slice. The factory owns its
own feature list, fitting and calibration, so swapping models (or plugging an
external scorer) is a one-line change and the engine never sees feature details.

Protocols
---------
``ModelFactory``  : ``feature_cols: list[str]`` + ``fit(train_df) -> FittedModel``
``FittedModel``   : ``predict(df) -> np.ndarray`` of P(win) per row

The default :class:`CatBoostFactory` mirrors ``models.train``'s honest-probability
recipe — **price-free** features (no odds-leakage), a time-ordered calibration
tail carved off the *train* slice only, and an isotonic/auto calibrator — so a
fold's probabilities are calibrated exactly the way the live value layer serves
them, without ever touching the test rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from models.calibration import OddsBandCalibrator, fit_calibrator
from models.features import PRICE_FREE_FEATURE_COLS
from utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class FittedModel(Protocol):
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...


@runtime_checkable
class ModelFactory(Protocol):
    feature_cols: Sequence[str]

    def fit(self, train_df: pd.DataFrame) -> FittedModel: ...


# ── default: price-free CatBoost with a leak-safe calibration tail ───────────

@dataclass
class CatBoostFactory:
    """Train a CatBoost win-probability model per fold.

    Parameters
    ----------
    feature_cols       : model inputs. Defaults to the price-free whitelist so the
                         probability never depends on the market price (the value
                         layer's core invariant).
    iterations         : boosting rounds (early stopping usually halts earlier).
    learning_rate/depth/l2_leaf_reg : core CatBoost knobs.
    calibration_method : "isotonic" | "sigmoid" | "auto" | "none".
    calibration_frac   : fraction of the (date-sorted) train tail held out to fit
                         the calibrator — the model never sees it.
    task_type/devices/thread_count : hardware (CPU is the safe portable default).
    params             : extra CatBoost kwargs merged last (e.g. reused best_params).
    """

    feature_cols: Sequence[str] = field(default_factory=lambda: list(PRICE_FREE_FEATURE_COLS))
    iterations: int = 500
    learning_rate: Optional[float] = None
    depth: int = 6
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 50
    calibration_method: str = "auto"
    calibration_frac: float = 0.15
    min_calib_rows: int = 200
    # Favourite-longshot recalibration (models.calibration.OddsBandCalibrator),
    # mirroring the live value layer. When True, after the base per-fold calibrator
    # an odds-band recalibrator is fit on the SAME train calibration tail — using
    # the leak-safe execution price (`bet_price`) — and applied in predict(). Off by
    # default so the existing engine/tests are unchanged.
    fl_recalibrate: bool = True
    fl_min_rows: int = 300
    auto_class_weights: Optional[str] = "Balanced"
    task_type: str = "CPU"
    devices: str = "0"
    thread_count: int = -1
    random_seed: int = 42
    params: dict = field(default_factory=dict)

    def _hardware(self) -> dict:
        tt = str(self.task_type).upper()
        if tt == "GPU":
            return {"task_type": "GPU", "devices": str(self.devices)}
        return {"task_type": "CPU", "thread_count": int(self.thread_count)}

    def fit(self, train_df: pd.DataFrame) -> "_CatBoostModel":
        from catboost import CatBoostClassifier

        df = train_df.sort_values("race_date").reset_index(drop=True)
        cols = [c for c in self.feature_cols if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
        y = pd.to_numeric(df["won"], errors="coerce").fillna(0).astype(int).to_numpy()

        # Carve a time-ordered calibration tail off TRAIN only (never test).
        method = str(self.calibration_method).strip().lower()
        calibrate = method in ("isotonic", "sigmoid", "auto")
        n_cal = int(len(X) * self.calibration_frac) if calibrate else 0
        if n_cal < self.min_calib_rows or (len(X) - n_cal) < self.min_calib_rows:
            n_cal = 0
        if n_cal > 0:
            X_core, y_core = X[:-n_cal], y[:-n_cal]
            X_cal, y_cal = X[-n_cal:], y[-n_cal:]
            cal_df = df.iloc[-n_cal:]          # keeps bet_price for F-L recalibration
        else:
            X_core, y_core, X_cal, y_cal, cal_df = X, y, None, None, None

        # Hold out the last 10% of core for early stopping.
        n_val = max(1, int(len(X_core) * 0.10))
        params = {
            "iterations": self.iterations,
            "depth": self.depth,
            "l2_leaf_reg": self.l2_leaf_reg,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_seed": self.random_seed,
            "verbose": False,
            "allow_writing_files": False,
            **self._hardware(),
        }
        if self.learning_rate is not None:
            params["learning_rate"] = self.learning_rate
        params.update(self.params)
        if "auto_class_weights" not in params and "scale_pos_weight" not in params \
                and self.auto_class_weights:
            params["auto_class_weights"] = self.auto_class_weights

        model = CatBoostClassifier(**params)
        if len(X_core) > n_val and len(np.unique(y_core[:-n_val])) > 1:
            model.fit(X_core[:-n_val], y_core[:-n_val],
                      eval_set=(X_core[-n_val:], y_core[-n_val:]), verbose=False)
        else:  # tiny fold — fit without an eval set
            model.fit(X_core, y_core, verbose=False)

        calibrator = None
        fl_calibrator = None
        if X_cal is not None and len(np.unique(y_cal)) > 1:
            raw = model.predict_proba(X_cal)[:, 1]
            calibrator, used = fit_calibrator(method, raw, y_cal)
            logger.debug("backtest.model: fold calibrated via %s (n_cal=%d)", used, n_cal)

            # Favourite-longshot recalibration: fit OddsBandCalibrator on the
            # base-calibrated cal probs vs the cal slice's execution price. It
            # corrects the by-odds-band A/E bias (favourites under-rated, longshots
            # over-rated) exactly as the live predictor's F-L recalibrator does.
            if self.fl_recalibrate and cal_df is not None and "bet_price" in cal_df.columns:
                base = calibrator.predict(raw) if calibrator is not None else raw
                odds = pd.to_numeric(cal_df["bet_price"], errors="coerce").to_numpy()
                try:
                    fl_calibrator = OddsBandCalibrator.fit(
                        base, odds, y_cal, method="isotonic", min_rows=self.fl_min_rows)
                except ValueError as exc:
                    logger.warning("backtest.model: F-L recalibrator skipped (%s)", exc)

        return _CatBoostModel(model, calibrator, cols, fl_calibrator)


@dataclass
class _CatBoostModel:
    model: object
    calibrator: Optional[object]
    cols: Sequence[str]
    fl_calibrator: Optional[object] = None

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.cols if c in df.columns]
        X = df[cols].apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()
        raw = self.model.predict_proba(X)[:, 1]
        p = np.asarray(self.calibrator.predict(raw), dtype=float) \
            if self.calibrator is not None else raw
        # F-L recalibration needs the per-runner execution price; apply only where it
        # is finite and fall back to the base-calibrated prob elsewhere.
        if self.fl_calibrator is not None and "bet_price" in df.columns:
            odds = pd.to_numeric(df["bet_price"], errors="coerce").to_numpy()
            ok = np.isfinite(odds)
            if ok.any():
                recal = np.asarray(self.fl_calibrator.predict(p[ok], odds[ok]), dtype=float)
                p = p.copy()
                p[ok] = recal
        return p


# ── escape hatch: wrap any scoring function as a factory ─────────────────────

@dataclass
class CallableModelFactory:
    """Adapt an arbitrary ``fit``/``predict`` pair to the factory protocol.

    ``fit_fn(train_df) -> state`` is called per fold; ``predict_fn(state, df) ->
    probs`` scores that fold's test rows. Handy for plugging a scikit-learn model
    or a deterministic stub (the test-suite uses this to verify no-leakage without
    booting CatBoost).
    """

    fit_fn: Callable[[pd.DataFrame], object]
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray]
    feature_cols: Sequence[str] = field(default_factory=lambda: list(PRICE_FREE_FEATURE_COLS))

    def fit(self, train_df: pd.DataFrame) -> "_CallableModel":
        return _CallableModel(self.predict_fn, self.fit_fn(train_df))


@dataclass
class _CallableModel:
    predict_fn: Callable[[object, pd.DataFrame], np.ndarray]
    state: object

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.predict_fn(self.state, df), dtype=float)
