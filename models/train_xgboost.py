"""Train/score XGBoost as a third boosted-tree candidate (step 13).

Mirrors ``models/train.py``'s CatBoost interface (Optuna-tuned, whole-race
chronological CV via ``models.split_utils``, a time-ordered calibration carve,
``fit_calibrator`` from ``models.calibration``) and ``backtest/holdout.py``'s
frozen-model load/score pair, so XGBoost slots into the same experiment
protocol as CatBoost and grouped-softmax LightGBM rather than a parallel
pipeline. Candidate-only: not imported by any serving path, no `FEATURE_COLS`
or bundle change (D09/D27/D43 pattern).

CPU-only by construction (``tree_method="hist"``): CatBoost already owns the
GPU in this experiment (``models/tuner.py::_hardware_params``) and the scope
note for this stage explicitly allows CPU-or-sequential-GPU — CPU avoids any
GPU-driver interaction with the CatBoost/LightGBM runs sharing the same
session, and XGBoost's histogram method is fast enough on this machine's CPU
for the row/column counts here (measured in reports/improvement/13).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from models.calibration import brier_score, fit_calibrator
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.split_utils import group_time_series_split
from models.targets import add_targets
from models.train import _group_carve, _load_model_cfg, _sample_weights, _time_split

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(logging.WARNING)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_MIN_CALIB_ROWS = 200
_MIN_CORE_ROWS = 200

_MODEL_RE = re.compile(r"^xgboost_(?P<target>.+)_(?P<tag>[^_]+)\.json$")

# Search space: XGBoost analogue of models/tuner.py's CatBoost space. Class
# imbalance is handled by a fixed scale_pos_weight (neg/pos ratio of the
# fitting fold), not tuned — the odds-inverse sample weight already carries
# most of the imbalance correction, matching the existing ensemble_experiment.py
# precedent rather than adding a second tuned imbalance knob.
_SPACE = {
    "learning_rate": (0.005, 0.3),
    "max_depth": (3, 10),
    "min_child_weight": (1.0, 20.0),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
    "reg_lambda": (0.1, 30.0),
    "reg_alpha": (1e-3, 10.0),
}


def _hardware_params(cfg: dict) -> dict:
    n_jobs = int(cfg.get("thread_count", -1))
    return {"tree_method": "hist", "n_jobs": n_jobs if n_jobs != 0 else -1}


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return (neg / pos) if pos > 0 else 1.0


def _suggest_params(trial: "optuna.Trial") -> dict:
    lr_lo, lr_hi = _SPACE["learning_rate"]
    d_lo, d_hi = _SPACE["max_depth"]
    mcw_lo, mcw_hi = _SPACE["min_child_weight"]
    ss_lo, ss_hi = _SPACE["subsample"]
    cs_lo, cs_hi = _SPACE["colsample_bytree"]
    l2_lo, l2_hi = _SPACE["reg_lambda"]
    l1_lo, l1_hi = _SPACE["reg_alpha"]
    return {
        "learning_rate": trial.suggest_float("learning_rate", lr_lo, lr_hi, log=True),
        "max_depth": trial.suggest_int("max_depth", d_lo, d_hi),
        "min_child_weight": trial.suggest_float("min_child_weight", mcw_lo, mcw_hi, log=True),
        "subsample": trial.suggest_float("subsample", ss_lo, ss_hi),
        "colsample_bytree": trial.suggest_float("colsample_bytree", cs_lo, cs_hi),
        "reg_lambda": trial.suggest_float("reg_lambda", l2_lo, l2_hi, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", l1_lo, l1_hi, log=True),
    }


def _cv_splits(X, cfg, race_ids, order_keys):
    n_splits = cfg["cv_folds"]
    if race_ids is not None and order_keys is not None:
        try:
            return list(group_time_series_split(race_ids, order_keys, n_splits))
        except ValueError as exc:
            logger.warning("train_xgboost: falling back to row-indexed TimeSeriesSplit (%s)", exc)
    return list(TimeSeriesSplit(n_splits=n_splits).split(X))


def _objective(trial, X, y, w, cfg, splits) -> float:
    from xgboost import XGBClassifier

    params = _suggest_params(trial)
    scores = []
    for train_idx, val_idx in splits:
        y_tr = y[train_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        model = XGBClassifier(
            n_estimators=cfg["iterations"],
            eval_metric="logloss",
            early_stopping_rounds=cfg["early_stopping_rounds"],
            scale_pos_weight=_scale_pos_weight(y_tr),
            random_state=int(cfg.get("random_seed", 42)),
            verbosity=0,
            **_hardware_params(cfg),
            **params,
        )
        model.fit(
            X[train_idx], y_tr, sample_weight=w[train_idx],
            eval_set=[(X[val_idx], y[val_idx])], verbose=False,
        )
        p = model.predict_proba(X[val_idx])[:, 1]
        scores.append(float(log_loss(y[val_idx], np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])))
        trial.report(float(np.mean(scores)), step=len(scores))
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores)) if scores else float("inf")


def run_study(X_train, y_train, sample_weight, cfg, race_ids=None, order_keys=None) -> dict:
    """Bounded Optuna search; returns XGBoost-ready tuned hyperparams.

    Same shape as ``models.tuner.run_study`` (whole-race chronological CV via
    ``models.split_utils.group_time_series_split``, log-loss objective on the
    unweighted validation fold, matched ``n_trials``/``cv_folds``).
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    w = np.asarray(sample_weight, dtype=float)
    splits = _cv_splits(X, cfg, race_ids, order_keys)
    seed = int(cfg.get("random_seed", 42))
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: _objective(trial, X, y, w, cfg, splits),
        n_trials=cfg["optuna_trials"], show_progress_bar=False,
    )
    logger.info("train_xgboost: best CV logloss=%.5f params=%s",
                study.best_value, study.best_params)
    return dict(study.best_params)


# ── train ────────────────────────────────────────────────────────────────────


def train(df=None, targets=None, trials=None, model_dir=None,
          feature_cols=None, version_tag=None, no_tune=False):
    """Train XGBoost classifier(s) and save artefacts; mirrors ``models.train.train``.

    Scope-limited to whatever ``targets`` the caller passes (step 13 scores
    ``won`` only, matching step 10's CatBoost scorecard scope, D24).
    """
    from xgboost import XGBClassifier

    cfg = _load_model_cfg()
    if targets is not None:
        cfg["targets"] = targets
    if trials is not None:
        cfg["optuna_trials"] = trials
    if model_dir is not None:
        cfg["model_dir"] = model_dir
    if version_tag is not None:
        cfg["version_tag"] = version_tag

    calib_method = cfg["calibration_method"].strip().lower()
    calibrate = calib_method in ("isotonic", "sigmoid", "auto")
    calib_frac = cfg["calibration_size"]
    feat_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)

    if df is None:
        raise ValueError("train_xgboost.train: df must be supplied (candidate-only module)")
    if len(df) == 0:
        logger.error("train_xgboost: empty training matrix")
        return None

    df = add_targets(df, cfg["show_positions"])
    if "market_type" in df.columns:
        df = df[df["market_type"].astype("string").str.upper() == "WIN"].copy()
    df["_sample_weight"] = _sample_weights(df, cfg["max_sample_weight"])

    train_df, test_df = _time_split(df, cfg["test_size"])
    n_train = len(train_df)

    out_dir = cfg["model_dir"] if os.path.isabs(cfg["model_dir"]) else os.path.join(_BASE, cfg["model_dir"])
    os.makedirs(out_dir, exist_ok=True)

    avail_cols = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(avail_cols)
    if missing:
        logger.warning("train_xgboost: %d feature_cols not in df, skipping: %s",
                       len(missing), sorted(missing))

    meta = {"targets": {}, "feature_cols": avail_cols, "train_rows": n_train,
            "test_rows": len(test_df), "model_type": "xgboost"}

    for target in cfg["targets"]:
        if target not in train_df.columns:
            logger.warning("train_xgboost: target '%s' not in dataframe, skipping", target)
            continue
        logger.info("train_xgboost: ── target=%s ──", target)

        tr_sub = train_df.loc[train_df[target].notna()].reset_index(drop=True)
        te_sub = test_df.loc[test_df[target].notna()].reset_index(drop=True)

        def _xy(frame):
            X = frame[avail_cols].astype(float).values
            y = frame[target].astype(int).values
            w = frame["_sample_weight"].to_numpy(dtype=float)
            return X, y, w

        X_te, y_te, _w_te = _xy(te_sub)

        if calibrate and len(tr_sub) > 0:
            core_df, cal_df = _group_carve(tr_sub, calib_frac)
        else:
            core_df, cal_df = tr_sub, tr_sub.iloc[0:0]
        if len(cal_df) < _MIN_CALIB_ROWS or len(core_df) < _MIN_CORE_ROWS:
            core_df, cal_df = tr_sub, tr_sub.iloc[0:0]
        n_cal = len(cal_df)

        X_core, y_core, w_core = _xy(core_df)
        X_cal, y_cal = (_xy(cal_df)[0], _xy(cal_df)[1]) if n_cal > 0 else (None, None)

        if no_tune:
            best_params = {}
        else:
            logger.info("train_xgboost: running Optuna (%d trials)", cfg["optuna_trials"])
            best_params = run_study(
                X_core, y_core, w_core, cfg,
                race_ids=core_df["race_uid"].to_numpy(),
                order_keys=core_df["race_date"].to_numpy(),
            )

        fit_df, es_df = _group_carve(core_df, 0.10)
        X_fit, y_fit, w_fit = _xy(fit_df)
        if len(es_df) > 0:
            X_es, y_es, _w_es = _xy(es_df)
        else:
            X_es = y_es = None

        final_params = {
            "n_estimators": cfg["iterations"],
            "eval_metric": "logloss",
            "early_stopping_rounds": cfg["early_stopping_rounds"] if X_es is not None else None,
            "scale_pos_weight": _scale_pos_weight(y_fit),
            "random_state": int(cfg.get("random_seed", 42)),
            "verbosity": 0,
            **_hardware_params(cfg),
            **best_params,
        }
        model = XGBClassifier(**final_params)
        if X_es is not None:
            model.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=[(X_es, y_es)], verbose=False)
        else:
            model.fit(X_fit, y_fit, sample_weight=w_fit, verbose=False)

        vtag = cfg["version_tag"]
        calibrator = None
        method_used = None
        if X_cal is not None and len(np.unique(y_cal)) > 1:
            raw_cal = model.predict_proba(X_cal)[:, 1]
            calibrator, method_used = fit_calibrator(calib_method, raw_cal, y_cal)
            cal_path = os.path.join(out_dir, f"xgboost_{target}_{vtag}_calib.pkl")
            with open(cal_path, "wb") as fh:
                pickle.dump(calibrator, fh)
            logger.info("train_xgboost: saved %s calibrator %s (n_cal=%d)", method_used, cal_path, n_cal)
        elif calibrate:
            logger.warning("train_xgboost: calibration skipped for %s (insufficient slice)", target)

        raw_te = np.asarray(model.predict_proba(X_te)[:, 1], dtype=float)
        p_te = np.asarray(calibrator.predict(raw_te), dtype=float) if calibrator is not None else raw_te
        auc = float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) > 1 else float("nan")
        brier = brier_score(y_te, p_te)

        model_path = os.path.join(out_dir, f"xgboost_{target}_{vtag}.json")
        model.save_model(model_path)
        logger.info("train_xgboost: saved %s", model_path)

        meta["targets"][target] = {
            "best_params": best_params,
            "test_auc": auc,
            "test_brier": brier,
            "calibrated": calibrator is not None,
            "calibration_method": method_used,
            "n_cal": n_cal,
            "n_core": len(core_df),
            "n_fit": len(fit_df),
            "n_early_stop": len(es_df),
        }

    meta_path = os.path.join(out_dir, f"xgboost_{cfg['version_tag']}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("train_xgboost: saved meta to %s", meta_path)
    return meta


# ── frozen model load / score (backtest.holdout analogue, XGBoost-specific) ──


class FrozenXGBoost:
    def __init__(self, model, calibrator, feature_cols, target, tag, meta):
        self.model = model
        self.calibrator = calibrator
        self.feature_cols = feature_cols
        self.target = target
        self.tag = tag
        self.meta = meta


def _load_pickle(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_frozen_model(model_path: str) -> FrozenXGBoost:
    from xgboost import XGBClassifier

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"model not found: {path}")
    m = _MODEL_RE.match(path.name)
    if not m:
        raise ValueError(f"model filename {path.name!r} is not xgboost_<target>_<tag>.json")
    target, tag = m.group("target"), m.group("tag")
    model_dir = path.parent

    model = XGBClassifier()
    model.load_model(str(path))

    meta_path = model_dir / f"xgboost_{tag}_meta.json"
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    feature_cols = list(meta.get("feature_cols") or [])
    if not feature_cols:
        raise ValueError(f"{meta_path.name} has no feature_cols")

    calibrator = _load_pickle(model_dir / f"xgboost_{target}_{tag}_calib.pkl")
    logger.info("train_xgboost: loaded frozen model %s (target=%s tag=%s, %d features, calibrator=%s)",
                path.name, target, tag, len(feature_cols),
                type(calibrator).__name__ if calibrator else None)
    return FrozenXGBoost(model, calibrator, feature_cols, target, tag, meta)


def _feature_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    frame = pd.DataFrame(index=df.index)
    for c in cols:
        frame[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    return frame.astype(float).to_numpy()


def score(frozen: FrozenXGBoost, df: pd.DataFrame) -> np.ndarray:
    """Per-runner calibrated win probability (NOT within-race normalised)."""
    X = _feature_matrix(df, frozen.feature_cols)
    raw = np.asarray(frozen.model.predict_proba(X)[:, 1], dtype=float)
    if frozen.calibrator is not None:
        return np.asarray(frozen.calibrator.predict(raw), dtype=float)
    return raw
