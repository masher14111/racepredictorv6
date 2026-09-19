"""LightGBM grouped-softmax win-probability model line (v4 second model).

Ported from racing_ingestion v1
-------------------------------
Sources merged into this one self-contained module:
  * racing_ingestion/ml/model/objective.py  -> grouped_softmax / cross_entropy_obj / grouped_logloss_eval
  * racing_ingestion/ml/model/softmax.py     -> softmax_by_race / SoftmaxWrapper
  * racing_ingestion/ml/model/lgbm_model.py  -> RaceWinPredictor (here: LGBMSoftmaxModel)

This is a SECOND model line that lives ALONGSIDE the CatBoost models in
``models/`` — it is not a replacement. Where CatBoost trains one independent
binary classifier per target (``won`` / ``placed_2`` / ``showed``), this is a
single LightGBM booster trained with a per-race grouped softmax cross-entropy
objective (McFadden's conditional logit): every runner gets one raw utility
score, and ``softmax_by_race`` turns the scores into win probabilities that
**sum to 1.0 within each race**.

Architecture
------------
Training objective : grouped softmax cross-entropy (McFadden's logit)
Normalisation      : per-race softmax applied to raw booster scores
Group constraint   : all runners of a race must occupy consecutive rows
                     (this module sorts by ``race_id`` for you in ``fit``)

Environment (see project prompt "STEP 0")
-----------------------------------------
LightGBM 4.6.0 ships a pure-Python wheel (``lightgbm-4.6.0-py3-none-win_amd64``)
that imports cleanly on the main CPython 3.14 interpreter, so we take the
**3.14-native** path: the main app imports ``lightgbm`` directly, no ``.venv-lgbm``
and no subprocess hand-off. The chosen environment is recorded in the model
metadata json (``env`` field) and printed on save. To keep this option open,
the module deliberately depends only on ``pandas`` / ``numpy`` / ``lightgbm`` —
no v4-3.14-only imports — so a bare interpreter could still run it if a future
LightGBM/Python combination ever forced the dedicated-venv fallback.

Leakage constraint
------------------
Market features fed to this model MUST come from a PRE-OFF price
(``morningwap`` / ``ppwap``), never the finishing SP (``odds_finish``). ``fit``
defensively rejects a feature matrix that carries a known post-result column
(see ``_LEAKAGE_FEATURES``).

Feature-count robustness
------------------------
racing_ingestion v1 broke when its pipeline grew from 43 -> 45 features because
prediction passed columns positionally. This port stores the trained
``feature_name`` list in the bundle and, at predict time, selects exactly those
columns **by name, in training order** from the input DataFrame — extra columns
are ignored and column reordering is harmless. A missing trained feature raises
a clear error rather than silently mis-aligning.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Default hyperparameters (canonical LightGBM names; override via constructor) ──
# Tuned for typical UK/IRE racing volumes (~50-300 races/day x ~10 runners).
# Regularisation is biased conservative to avoid memorising individual horses.
_DEFAULT_PARAMS: dict = {
    "boosting_type":     "gbdt",
    "num_leaves":         63,
    "max_depth":          -1,
    "learning_rate":      0.03,
    "n_estimators":       1500,   # consumed as num_boost_round, not passed to params
    "bagging_fraction":   0.8,    # sklearn alias: subsample
    "bagging_freq":       5,      # sklearn alias: subsample_freq
    "feature_fraction":   0.7,    # sklearn alias: colsample_bytree
    "min_data_in_leaf":   20,     # sklearn alias: min_child_samples
    "lambda_l1":          0.1,
    "lambda_l2":          0.1,
    "min_gain_to_split":  0.01,   # sklearn alias: min_split_gain
    "verbosity":          -1,
    "num_threads":        -1,     # all cores (12 on this machine)
    "seed":               42,
}

# Feature names that would leak the result. Market signal must be pre-off
# (morningwap / ppwap); the finishing SP is forbidden.
_LEAKAGE_FEATURES: frozenset[str] = frozenset(
    {"odds_finish", "sp_finish", "finish_position", "finishing_position", "result_position"}
)


# ── Core grouped softmax ─────────────────────────────────────────────────────


def grouped_softmax(preds: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax applied independently within each race group.

    Args:
        preds:       Flat array of raw model scores, ordered by group.
        group_sizes: Integer runner counts per race, same order as ``preds``
                     (``sum(group_sizes) == len(preds)``).

    Returns:
        Softmax probabilities, same shape as ``preds``. Single-runner races
        (void / walkover) trivially return probability 1.0.
    """
    probs = np.empty_like(preds, dtype=np.float64)
    offset = 0
    for n in group_sizes.astype(np.int64):
        end = offset + n
        if n == 1:
            probs[offset] = 1.0
        else:
            raw = preds[offset:end]
            shifted = raw - raw.max()          # log-sum-exp stability
            exp_s = np.exp(shifted)
            probs[offset:end] = exp_s / exp_s.sum()
        offset = end
    return probs


def softmax_by_race(
    raw_scores: np.ndarray,
    race_ids: np.ndarray | pd.Series,
) -> np.ndarray:
    """Convert raw scores into per-race probability simplices, preserving row order.

    Unlike :func:`grouped_softmax`, this does NOT require runners of a race to be
    consecutive — it groups by ``race_id`` value and writes results back in the
    original row order, so it is safe for arbitrary prediction batches.

    Args:
        raw_scores: 1-D array of booster outputs, one per runner.
        race_ids:   Matching race identifiers (any hashable type); runners of the
                    same race share a value.

    Returns:
        Probability array in the same order as ``raw_scores``; probabilities
        within each race sum to 1.0.
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    race_ids_arr = np.asarray(race_ids)
    if raw_scores.shape[0] != race_ids_arr.shape[0]:
        raise ValueError(
            f"raw_scores ({raw_scores.shape[0]}) and race_ids "
            f"({race_ids_arr.shape[0]}) length mismatch."
        )
    probs = np.zeros_like(raw_scores, dtype=np.float64)

    _, inverse = np.unique(race_ids_arr, return_inverse=True)
    for race_idx in range(inverse.max() + 1 if inverse.size else 0):
        mask = inverse == race_idx
        raw = raw_scores[mask]
        if raw.size == 1:
            probs[mask] = 1.0
        else:
            shifted = raw - raw.max()
            exp_s = np.exp(shifted)
            probs[mask] = exp_s / exp_s.sum()
    return probs


# ── LightGBM custom objective + eval metric ──────────────────────────────────


def cross_entropy_obj(
    preds: np.ndarray,
    train_data: "lgb.Dataset",
) -> tuple[np.ndarray, np.ndarray]:
    """LightGBM custom objective: per-race grouped softmax cross-entropy.

    Gradient  : ``p_i - y_i``      (zero when the winner is perfectly identified)
    Hessian   : ``p_i (1 - p_i)``  (clipped at 1e-6 to stay positive/convex)

    ``train_data`` must have been built with ``group=<runner counts per race>``.
    """
    labels = train_data.get_label()
    group_sizes = train_data.get_group()
    if group_sizes is None:
        raise RuntimeError(
            "Dataset has no group information. "
            "Pass group=race_group_sizes when building lgb.Dataset."
        )
    probs = grouped_softmax(preds, group_sizes)
    grad = probs - labels
    hess = np.maximum(probs * (1.0 - probs), 1e-6)
    return grad, hess


def grouped_logloss_eval(
    preds: np.ndarray,
    train_data: "lgb.Dataset",
) -> tuple[str, float, bool]:
    """LightGBM custom eval: mean per-race negative log-likelihood (lower better)."""
    labels = train_data.get_label()
    group_sizes = train_data.get_group()
    probs = grouped_softmax(preds, group_sizes)

    per_race_nll: list[float] = []
    offset = 0
    for n in group_sizes.astype(np.int64):
        end = offset + n
        winner_mask = labels[offset:end].astype(bool)
        winner_probs = probs[offset:end][winner_mask]
        if winner_probs.size > 0:
            per_race_nll.append(-float(np.log(np.clip(winner_probs, 1e-7, 1.0)).sum()))
        offset = end

    score = float(np.mean(per_race_nll)) if per_race_nll else 0.0
    return "grouped_logloss", score, False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sort_by_race(
    X: pd.DataFrame,
    y: np.ndarray,
    race_ids: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Stable-sort rows so every race occupies a consecutive block.

    Returns ``(X_sorted, y_sorted, group_sizes)`` where ``group_sizes`` are the
    per-race runner counts in block order (what LightGBM's ``group=`` expects).
    """
    order = np.argsort(race_ids, kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
    y_sorted = y[order]
    # race_ids[order] is now sorted, so np.unique's counts line up with the blocks.
    _, group_sizes = np.unique(race_ids[order], return_counts=True)
    return X_sorted, y_sorted, group_sizes.astype(np.int64)


def _detect_env() -> str:
    """Human-readable tag of the interpreter running LightGBM (STEP 0 record)."""
    return f"py{sys.version_info.major}.{sys.version_info.minor}-native"


# ── Model wrapper ────────────────────────────────────────────────────────────


class LGBMSoftmaxModel:
    """End-to-end LightGBM grouped-softmax win-probability model.

    Example
    -------
    ::

        model = LGBMSoftmaxModel()
        model.fit(X_train, y_train, race_ids_train,
                  X_val=X_val, y_val=y_val, race_ids_val=race_ids_val)
        proba = model.predict_proba(X_test, race_ids_test)   # sums to 1.0 per race
        model.save("models/lgbm_softmax_v1.txt")
        reloaded = LGBMSoftmaxModel.load("models/lgbm_softmax_v1.txt")
    """

    def __init__(self, **params) -> None:
        self._params: dict = {**_DEFAULT_PARAMS, **params}
        self._booster: lgb.Booster | None = None
        self.feature_name: list[str] | None = None
        self.metadata: dict = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        race_ids: pd.Series | np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
        race_ids_val: pd.Series | np.ndarray | None = None,
        race_dates: pd.Series | np.ndarray | None = None,
        early_stopping_rounds: int = 80,
        verbose_eval: int = 0,
    ) -> "LGBMSoftmaxModel":
        """Train the booster with the grouped cross-entropy objective.

        Args:
            X:                     Feature DataFrame. Column names are stored and
                                   become the model's ``feature_name``; rows need
                                   NOT be pre-sorted (we sort by ``race_id``).
            y:                     Binary win labels (one winner per race).
            race_ids:              Race identifier per row, aligned with ``X``.
            X_val / y_val / race_ids_val:
                                   Optional validation set enabling early stopping.
            race_dates:            Optional per-row dates; only used to record the
                                   train window in metadata.
            early_stopping_rounds: Patience when a validation set is supplied.
            verbose_eval:          Log every N rounds (0 = silent).

        Returns:
            self (for chaining).
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame so feature names can be stored.")

        leaking = _LEAKAGE_FEATURES.intersection(X.columns)
        if leaking:
            raise ValueError(
                f"Refusing to train on post-result feature(s) {sorted(leaking)}; "
                "market features must come from a pre-off price (morningwap/ppwap)."
            )

        self.feature_name = list(X.columns)
        y_arr = np.asarray(y, dtype=np.float32)
        race_ids_arr = np.asarray(race_ids)
        if not (len(X) == len(y_arr) == len(race_ids_arr)):
            raise ValueError("X, y and race_ids must have equal length.")

        X_sorted, y_sorted, group_sizes = _sort_by_race(X, y_arr, race_ids_arr)
        train_ds = lgb.Dataset(
            X_sorted,
            label=y_sorted,
            group=group_sizes,
            feature_name=self.feature_name,
            free_raw_data=False,
        )

        valid_sets = [train_ds]
        valid_names = ["train"]
        callbacks: list = []
        if verbose_eval and verbose_eval > 0:
            callbacks.append(lgb.log_evaluation(period=verbose_eval))

        if X_val is not None and y_val is not None and race_ids_val is not None:
            yv = np.asarray(y_val, dtype=np.float32)
            rv = np.asarray(race_ids_val)
            Xv_sorted, yv_sorted, group_sizes_val = _sort_by_race(X_val, yv, rv)
            val_ds = lgb.Dataset(
                Xv_sorted,
                label=yv_sorted,
                group=group_sizes_val,
                reference=train_ds,
                feature_name=self.feature_name,
                free_raw_data=False,
            )
            valid_sets.append(val_ds)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=bool(verbose_eval))
            )

        lgb_params = {k: v for k, v in self._params.items() if k != "n_estimators"}
        n_rounds = int(self._params.get("n_estimators", 1500))
        # LightGBM >= 4.0: custom objective is passed as a callable in params.
        lgb_params["objective"] = cross_entropy_obj
        lgb_params.setdefault("metric", "None")  # rely on feval, suppress built-ins

        logger.info(
            "LGBMSoftmax: training %d rounds | %d rows | %d races | %d features",
            n_rounds, len(X_sorted), len(group_sizes), len(self.feature_name),
        )
        self._booster = lgb.train(
            lgb_params,
            train_ds,
            num_boost_round=n_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=grouped_logloss_eval,
            callbacks=callbacks,
        )

        best_it = self._booster.best_iteration if self._booster.best_iteration > 0 else n_rounds
        self.metadata = {
            "model_type": "lgbm_softmax",
            "lgbm_version": lgb.__version__,
            "env": _detect_env(),
            "python_executable": sys.executable,
            "feature_name": list(self.feature_name),
            "n_features": len(self.feature_name),
            "n_rows": int(len(X_sorted)),
            "n_races": int(len(group_sizes)),
            "train_window": self._train_window(race_dates),
            "best_iteration": int(best_it),
            "params": {k: v for k, v in self._params.items()},
        }
        logger.info("LGBMSoftmax: training complete (best_iteration=%d).", best_it)
        return self

    @staticmethod
    def _train_window(race_dates) -> dict | None:
        if race_dates is None:
            return None
        s = pd.to_datetime(pd.Series(np.asarray(race_dates)), errors="coerce").dropna()
        if s.empty:
            return None
        return {"start": str(s.min().date()), "end": str(s.max().date())}

    # ── Prediction ───────────────────────────────────────────────────────────

    def _select_features(self, X: pd.DataFrame | np.ndarray):
        """Select exactly the trained features, by name and in training order.

        DataFrame input: extra columns ignored, reordered to match training,
        missing trained features raise. ndarray input: assumed already aligned.
        """
        if self.feature_name is None:
            raise RuntimeError("Model not trained / loaded. Call fit() or load() first.")
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_name if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Input is missing {len(missing)} trained feature(s): {missing[:10]}"
                    + (" ..." if len(missing) > 10 else "")
                )
            return X.loc[:, self.feature_name]
        arr = np.asarray(X)
        if arr.shape[1] != len(self.feature_name):
            raise ValueError(
                f"ndarray has {arr.shape[1]} columns but model expects "
                f"{len(self.feature_name)}; pass a DataFrame for name-based selection."
            )
        return arr

    def predict_raw(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return raw utility scores (before softmax)."""
        if self._booster is None:
            raise RuntimeError("Model not trained / loaded.")
        return np.asarray(self._booster.predict(self._select_features(X)))

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
        race_ids: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Predict win probabilities normalised to sum 1.0 within each race."""
        raw = self.predict_raw(X)
        return softmax_by_race(raw, race_ids)

    def predict_dataframe(
        self,
        X: pd.DataFrame,
        race_id_col: str = "race_id",
    ) -> pd.DataFrame:
        """Return ``X`` with added ``raw_score`` and ``win_probability`` columns."""
        if race_id_col not in X.columns:
            raise ValueError(f"race_id_col {race_id_col!r} not found in input columns.")
        raw = self.predict_raw(X)
        proba = softmax_by_race(raw, X[race_id_col].to_numpy())
        return X.assign(raw_score=raw, win_probability=proba)

    # ── Feature importance ─────────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Feature importances sorted descending (columns: feature, importance, importance_pct)."""
        if self._booster is None:
            raise RuntimeError("Model not trained / loaded.")
        names = self._booster.feature_name()
        scores = self._booster.feature_importance(importance_type=importance_type)
        df = pd.DataFrame({"feature": names, "importance": scores})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        total = df["importance"].sum()
        df["importance_pct"] = (df["importance"] / total * 100).round(2) if total > 0 else 0.0
        return df

    # ── Persistence ────────────────────────────────────────────────────────────

    @staticmethod
    def _meta_path(model_path: str | Path) -> Path:
        return Path(model_path).with_suffix(".meta.json")

    def save(self, model_path: str | Path) -> None:
        """Save the booster (LightGBM text model) plus a sidecar ``.meta.json``.

        Mirrors the v4 CatBoost ``<tag>_meta.json`` convention: feature list,
        row/race counts, train window, params and the recorded environment.
        """
        if self._booster is None or self.feature_name is None:
            raise RuntimeError("Model not trained.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path))
        meta_path = self._meta_path(path)
        meta_path.write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")
        logger.info(
            "LGBMSoftmax saved: %s (env=%s, %d features, %d races) + %s",
            path, self.metadata.get("env"), self.metadata.get("n_features"),
            self.metadata.get("n_races"), meta_path.name,
        )

    @classmethod
    def load(cls, model_path: str | Path) -> "LGBMSoftmaxModel":
        """Load a saved booster and its metadata sidecar."""
        path = Path(model_path)
        booster = lgb.Booster(model_file=str(path))
        meta_path = cls._meta_path(path)
        metadata: dict = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        model = cls(**(metadata.get("params") or {}))
        model._booster = booster
        model.metadata = metadata
        # Prefer the stored training-order feature list; fall back to the booster's.
        model.feature_name = metadata.get("feature_name") or list(booster.feature_name())
        logger.info("LGBMSoftmax loaded: %s (%d features)", path, len(model.feature_name))
        return model
