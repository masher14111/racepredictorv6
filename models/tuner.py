"""Optuna hyperparameter study for CatBoost binary classifiers.

run_study returns CatBoost-ready hyperparams (the tuned search-space values plus
the resolved class-weighting key — `auto_class_weights` or `scale_pos_weight`).
It deliberately omits fixed params (loss_function, iterations, hardware): the
caller merges those before fitting the final model.

Design notes
------------
* CV is **strictly chronological and whole-race-group-safe**
  (``models.split_utils.group_time_series_split``) when the caller passes
  ``race_ids``/``order_keys``: each fold validates on whole races that are
  chronologically later than every race in its training fold, and no race's
  rows are ever split between a fold's train and validation sides (step 02 —
  plain row-indexed ``TimeSeriesSplit`` does not guarantee that, since a race
  whose rows straddle a fold boundary would leak). ``race_ids``/``order_keys``
  are optional for callers that have no race identity (e.g. ad-hoc arrays in
  tests); omitting them falls back to a plain row-indexed ``TimeSeriesSplit``
  with a logged warning, and to a per-row fallback whenever there are too few
  unique groups to form the requested number of folds.
* The objective optimises a **calibration-aware** metric (log-loss by default,
  also Brier or AUC) on the *unweighted* validation fold, so a model whose raw
  probabilities are well-ranked but inflated (the A/E≈0.2 problem) is penalised.
* Class-imbalance handling is part of the search space, letting the study back
  off the double-correction (auto_class_weights + odds-inverse sample_weight)
  that the baseline audit (C5) flagged as the root of inflated raw probs.
* A pruner (median by default) plus CatBoost's own early stopping keep the study
  from burning trials on hopeless configurations.
"""
import logging

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from models.split_utils import group_time_series_split
from utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(logging.WARNING)

# Default search space — overridden per key by cfg["tuning"] when present.
_DEFAULT_SPACE = {
    "metric": "logloss",
    "pruner": "median",
    "pruner_warmup_trials": 5,
    "learning_rate": [5e-4, 0.3],
    "depth": [4, 10],
    "l2_leaf_reg": [1.0, 30.0],
    "bagging_temperature": [0.0, 2.0],
    "random_strength": [1e-9, 10.0],
    "border_count": [32, 254],
    "class_weight_modes": ["Balanced", "SqrtBalanced", "none", "scale_pos_weight"],
    "scale_pos_weight": [1.0, 25.0],
}


def _hardware_params(cfg):
    """CatBoost device params from cfg. task_type/devices/thread_count are
    optional (default CPU) so callers passing a minimal cfg stay portable.
    border_count is capped at 254 because GPU rejects 255."""
    task_type = str(cfg.get("task_type", "CPU")).upper()
    params = {"task_type": task_type}
    if task_type == "GPU":
        params["devices"] = str(cfg.get("devices", "0"))
    else:
        params["thread_count"] = int(cfg.get("thread_count", -1))
    return params


def _space(cfg):
    """Merge cfg["tuning"] over the defaults so any subset can be configured."""
    space = dict(_DEFAULT_SPACE)
    space.update(cfg.get("tuning", {}) or {})
    return space


def _metric_meta(metric):
    """(direction, eval_metric, lower_is_better) for the chosen objective metric."""
    metric = str(metric).lower()
    if metric == "auc":
        return "maximize", "AUC", False
    # logloss and brier are both minimised; CatBoost early-stops on Logloss
    return "minimize", "Logloss", True


def _class_weight_params(trial, space):
    """Sample the imbalance-handling scheme. Returns CatBoost-ready keys, so the
    final-fit merge never sees the synthetic 'class_weight_mode' search param."""
    modes = list(space["class_weight_modes"])
    mode = trial.suggest_categorical("class_weight_mode", modes)
    if mode == "scale_pos_weight":
        lo, hi = space["scale_pos_weight"]
        return {"scale_pos_weight": trial.suggest_float(
            "scale_pos_weight", float(lo), float(hi), log=True)}
    if mode.lower() == "none":
        # Explicit None (CatBoost's no-weighting default) so the choice is
        # distinguishable from "untuned" downstream and survives the final-fit
        # merge — otherwise train.py would re-inject its Balanced fallback.
        return {"auto_class_weights": None}
    return {"auto_class_weights": mode}


def _suggest_params(trial, space):
    """Tuned CatBoost hyperparams for a trial (search-space values only)."""
    lr_lo, lr_hi = space["learning_rate"]
    d_lo, d_hi = space["depth"]
    l2_lo, l2_hi = space["l2_leaf_reg"]
    bt_lo, bt_hi = space["bagging_temperature"]
    rs_lo, rs_hi = space["random_strength"]
    bc_lo, bc_hi = space["border_count"]
    params = {
        "learning_rate": trial.suggest_float("learning_rate", float(lr_lo), float(lr_hi), log=True),
        "depth": trial.suggest_int("depth", int(d_lo), int(d_hi)),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", float(l2_lo), float(l2_hi), log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", float(bt_lo), float(bt_hi)),
        "random_strength": trial.suggest_float("random_strength", float(rs_lo), float(rs_hi), log=True),
        "border_count": trial.suggest_int("border_count", int(bc_lo), int(bc_hi)),
    }
    params.update(_class_weight_params(trial, space))
    return params


def _score(y_true, y_prob, metric):
    if metric == "auc":
        return float(roc_auc_score(y_true, y_prob))
    if metric == "brier":
        return float(brier_score_loss(y_true, y_prob))
    return float(log_loss(y_true, y_prob, labels=[0, 1]))


def _cv_splits(X, cfg, race_ids, order_keys):
    """Chronological CV folds: whole-race-group-safe when race identity is
    available and there are enough unique races; otherwise a row-indexed
    ``TimeSeriesSplit`` fallback (logged, since it cannot guarantee a race's
    rows all land on one side of a fold boundary)."""
    n_splits = cfg["cv_folds"]
    if race_ids is not None and order_keys is not None:
        try:
            return list(group_time_series_split(race_ids, order_keys, n_splits))
        except ValueError as exc:
            logger.warning("tuner: falling back to row-indexed TimeSeriesSplit (%s)", exc)
    else:
        logger.warning(
            "tuner: no race_ids/order_keys supplied — using row-indexed "
            "TimeSeriesSplit, which cannot guarantee whole-race fold boundaries")
    return list(TimeSeriesSplit(n_splits=n_splits).split(X))


def _objective(trial, X, y, sample_weight, cfg, space, metric, splits):
    tuned = _suggest_params(trial, space)
    params = {
        **tuned,
        "iterations": cfg["iterations"],
        "loss_function": "Logloss",
        "eval_metric": _metric_meta(metric)[1],
        "verbose": False,
        "allow_writing_files": False,
        **_hardware_params(cfg),
    }
    scores = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        y_tr = y[train_idx]
        # A TimeSeriesSplit fold can be single-class on rare targets; skip it.
        if len(np.unique(y_tr)) < 2:
            continue
        model = CatBoostClassifier(**params)
        model.fit(
            X[train_idx], y_tr,
            sample_weight=sample_weight[train_idx],
            eval_set=(X[val_idx], y[val_idx]),
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose=False,
        )
        y_prob = model.predict_proba(X[val_idx])[:, 1]
        # Unweighted: the metric should reflect true-frequency calibration, not
        # the odds-inverse fitting weights.
        scores.append(_score(y[val_idx], y_prob, metric))
        trial.report(float(np.mean(scores)), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    if not scores:
        # No usable fold (degenerate target) — report a neutral-bad score.
        return float("inf") if _metric_meta(metric)[2] else 0.0
    return float(np.mean(scores))


def _make_pruner(space):
    name = str(space.get("pruner", "median")).lower()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(space.get("pruner_warmup_trials", 5)))


def run_study(X_train, y_train, sample_weight, cfg, race_ids=None, order_keys=None):
    """Run an Optuna study; return the best CatBoost-ready hyperparams dict.

    Parameters
    ----------
    X_train       : np.ndarray (n, n_features)
    y_train       : np.ndarray (n,) int 0/1
    sample_weight : np.ndarray (n,) float, odds-inverse weights
    cfg           : dict with keys cv_folds, optuna_trials, iterations,
                    early_stopping_rounds, and optional "tuning" sub-dict
                    (search space + metric + pruner). See config.yaml model.tuning.
    race_ids      : optional per-row race identity (e.g. race_uid). When given
                    together with order_keys, CV folds are whole-race-group-safe
                    (models.split_utils.group_time_series_split) — no race's
                    rows split across a fold boundary. Omit only for callers
                    with no race identity (falls back to row-indexed
                    TimeSeriesSplit, logged as a warning: step 02).
    order_keys    : optional per-row chronological key (e.g. race_date),
                    required alongside race_ids.

    Returns
    -------
    dict of tuned hyperparams plus the resolved class-weight key
    (auto_class_weights or scale_pos_weight). The synthetic 'class_weight_mode'
    selector is NOT returned, so the dict is safe to spread into the final fit.
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    w = np.asarray(sample_weight, dtype=float)

    space = _space(cfg)
    metric = str(space.get("metric", "logloss")).lower()
    direction, _, lower_better = _metric_meta(metric)
    seed = int(cfg.get("random_seed", 42))

    # Folds are fixed once per study (not per trial): the chronological/
    # group-safety guarantee is a property of the data split, not of any
    # trial's hyperparameters.
    splits = _cv_splits(X, cfg, race_ids, order_keys)

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=_make_pruner(space),
    )
    study.optimize(
        lambda trial: _objective(trial, X, y, w, cfg, space, metric, splits),
        n_trials=cfg["optuna_trials"],
        show_progress_bar=False,
    )

    # Re-derive the CatBoost-ready params from the winning trial's raw values so
    # the synthetic 'class_weight_mode' selector never leaks into the final fit.
    best = study.best_trial
    tuned = {k: v for k, v in best.params.items() if k != "class_weight_mode"}
    mode = best.params.get("class_weight_mode")
    if mode is not None:
        tuned.pop("scale_pos_weight", None)  # re-added below only if mode selects it
        if mode == "scale_pos_weight":
            tuned["scale_pos_weight"] = best.params["scale_pos_weight"]
        elif mode.lower() == "none":
            tuned["auto_class_weights"] = None  # explicit: no weighting (see _class_weight_params)
        else:
            tuned["auto_class_weights"] = mode

    logger.info("tuner: best CV %s=%.5f (%s)  params=%s",
                metric, study.best_value,
                "lower=better" if lower_better else "higher=better", tuned)
    return tuned
