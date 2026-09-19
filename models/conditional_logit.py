"""Simple race conditional-logit baseline (McFadden multinomial logit).

Linear analogue of ``models.lgbm_softmax``'s per-race grouped-softmax objective:
one scalar utility ``w . x`` per runner, softmax within race, minimising the
mean per-race negative log-likelihood plus L2 (ridge) regularisation on ``w``.
Inputs are standardised on TRAIN-fold statistics only (mean/std frozen at fit
time, applied unchanged at scoring time), matching the point-in-time discipline
used everywhere else in this codebase — a race's probability must never depend
on which other rows happen to share its scoring batch.

This is step 10's "simple race conditional-logit baseline": the same grouped-
softmax scoring rule as the LightGBM line (``softmax_by_race``), but a linear
score function with zero interaction terms or trees, so any measured gap over
this line is attributable to model capacity, not to a labelling or evaluation
difference. A single race with one runner (walkover) contributes probability
1.0 and no gradient, matching ``models.lgbm_softmax.grouped_softmax``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from models.lgbm_softmax import softmax_by_race
from models.split_utils import group_time_series_split

logger = logging.getLogger(__name__)


class ConditionalLogitModel:
    """L2-regularised linear conditional logit, scored via per-race softmax."""

    def __init__(self, l2: float = 1.0) -> None:
        self.l2 = float(l2)
        self.feature_name: Optional[list] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.metadata: dict = {}

    # ── standardisation (train-fitted, frozen) ──────────────────────────────

    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        mean = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        std = np.where((std > 1e-12) & np.isfinite(std), std, 1.0)
        mean = np.where(np.isfinite(mean), mean, 0.0)
        self.mean_, self.std_ = mean, std
        return self._standardize_apply(X)

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mean_) / self.std_
        return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    # ── objective (grouped softmax cross-entropy + ridge) ───────────────────

    @staticmethod
    def _neg_log_lik_grad(w, Xs, y, sw, group_starts, group_sizes, l2):
        scores = Xs @ w
        probs = np.empty_like(scores)
        nll = 0.0
        for start, n in zip(group_starts, group_sizes):
            end = start + n
            s = scores[start:end]
            s = s - s.max()
            e = np.exp(s)
            p = e / e.sum()
            probs[start:end] = p
            yi = y[start:end]
            winner = yi.astype(bool)
            if winner.any():
                nll -= (sw[start:end][winner]
                        * np.log(np.clip(p[winner], 1e-12, 1.0))).sum()
        n_races = max(len(group_sizes), 1)
        resid = sw * (probs - y)
        grad = (Xs.T @ resid) / n_races + l2 * w
        nll = nll / n_races + 0.5 * l2 * float(w @ w)
        return nll, grad

    def fit(self, X, y, race_ids, sample_weight=None) -> "ConditionalLogitModel":
        """Fit on (X, y, race_ids). Rows need not be pre-sorted by race.

        ``sample_weight`` (optional, per-row) scales each row's contribution to
        the grouped cross-entropy loss/gradient — e.g. the existing odds-inverse
        weight, a race-normalised weight, or ``None`` for the unweighted
        objective (every row weight 1.0). Weighting only the winner-row NLL term
        would be equivalent to rescaling that race's contribution; weighting the
        (probs - y) residual too keeps the gradient consistent with the weighted
        loss actually being minimised.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_name = list(X.columns)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_name = [f"x{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y, dtype=float)
        race_arr = np.asarray(race_ids)
        w_arr = (np.ones(len(y_arr)) if sample_weight is None
                else np.asarray(sample_weight, dtype=float))
        if not (len(X_arr) == len(y_arr) == len(race_arr) == len(w_arr)):
            raise ValueError("X, y, race_ids and sample_weight must have equal length.")

        order = np.argsort(race_arr, kind="stable")
        Xo, yo, wo, rid_o = X_arr[order], y_arr[order], w_arr[order], race_arr[order]
        _, group_sizes = np.unique(rid_o, return_counts=True)
        group_starts = np.concatenate([[0], np.cumsum(group_sizes)[:-1]])

        Xs = self._standardize_fit(Xo)
        w0 = np.zeros(Xs.shape[1])
        res = minimize(
            self._neg_log_lik_grad, w0,
            args=(Xs, yo, wo, group_starts, group_sizes, self.l2),
            jac=True, method="L-BFGS-B",
            options={"maxiter": 500},
        )
        self.coef_ = res.x
        self.metadata = {
            "model_type": "conditional_logit",
            "l2": self.l2,
            "n_features": len(self.feature_name),
            "n_rows": int(len(Xo)),
            "n_races": int(len(group_sizes)),
            "weighted": sample_weight is not None,
            "converged": bool(res.success),
            "final_objective": float(res.fun),
        }
        return self

    # ── prediction ───────────────────────────────────────────────────────────

    def predict_raw(self, X) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not trained / loaded.")
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_name if c not in X.columns]
            if missing:
                raise ValueError(f"Input missing trained feature(s): {missing}")
            X_arr = X.loc[:, self.feature_name].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        Xs = self._standardize_apply(X_arr)
        return Xs @ self.coef_

    def predict_proba(self, X, race_ids) -> np.ndarray:
        """Win probability normalised to sum 1.0 within each race."""
        raw = self.predict_raw(X)
        return softmax_by_race(raw, race_ids)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path) -> None:
        if self.coef_ is None:
            raise RuntimeError("Model not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "feature_name": self.feature_name,
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "coef": self.coef_.tolist(),
            "l2": self.l2,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ConditionalLogitModel":
        path = Path(path)
        state = json.loads(path.read_text(encoding="utf-8"))
        model = cls(l2=state["l2"])
        model.feature_name = state["feature_name"]
        model.mean_ = np.array(state["mean"], dtype=float)
        model.std_ = np.array(state["std"], dtype=float)
        model.coef_ = np.array(state["coef"], dtype=float)
        model.metadata = state.get("metadata", {})
        return model


# ── bounded tuning (matched budget with the tree models' Optuna studies) ──────


def cv_race_log_loss(X, y, race_ids, order_keys, l2: float, n_splits: int = 4) -> float:
    """Mean out-of-fold race-level log loss for one L2 value.

    Uses the same whole-race walk-forward CV primitive as ``models.tuner``
    (``models.split_utils.group_time_series_split``) so no race's rows are ever
    split across a fold boundary.
    """
    Xf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    y_arr = np.asarray(y, dtype=float)
    race_arr = np.asarray(race_ids)
    order_arr = np.asarray(order_keys)

    scores = []
    for train_idx, val_idx in group_time_series_split(race_arr, order_arr, n_splits):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        y_tr = y_arr[train_idx]
        if len(np.unique(y_tr)) < 2:
            continue
        model = ConditionalLogitModel(l2=l2).fit(
            Xf.iloc[train_idx], y_tr, race_arr[train_idx])
        p = model.predict_proba(Xf.iloc[val_idx], race_arr[val_idx])
        yv = y_arr[val_idx]
        rv = race_arr[val_idx]
        per_race = []
        for rid in pd.unique(rv):
            m = rv == rid
            winner_p = p[m][yv[m].astype(bool)]
            if winner_p.size:
                per_race.append(-float(np.log(np.clip(winner_p, 1e-12, 1.0)).sum()))
        if per_race:
            scores.append(float(np.mean(per_race)))
    return float(np.mean(scores)) if scores else float("nan")


def tune_l2(X, y, race_ids, order_keys, *, n_trials: int = 12, n_splits: int = 4,
           seed: int = 42) -> dict:
    """Bounded Optuna search over the single L2 hyperparameter.

    Matches the tuning-budget shape used for CatBoost/LightGBM (Optuna,
    whole-race chronological CV, race-level log loss objective) with one
    dimension instead of many, since a linear conditional logit has one
    regularisation knob.
    """
    import optuna

    optuna.logging.set_verbosity(logging.WARNING)

    def objective(trial):
        l2 = trial.suggest_float("l2", 1e-4, 10.0, log=True)
        return cv_race_log_loss(X, y, race_ids, order_keys, l2, n_splits=n_splits)

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return {"l2": float(study.best_params["l2"]), "cv_log_loss": float(study.best_value)}
