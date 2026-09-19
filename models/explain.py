"""Per-prediction SHAP explanations — "why this horse".

Tree SHAP is exact and built into CatBoost (``get_feature_importance
(type="ShapValues")``), so no extra dependency is needed. For one race
(~10 runners) it is sub-millisecond, fast enough to call live.

The cost that matters live is *loading* the model from disk, so
:func:`get_explainer` memoises the loaded model + its feature whitelist
(read from the same ``*_meta.json`` the predictor uses). Reuse one explainer
across polling cycles::

    from models.explain import get_explainer
    ex = get_explainer(target="won", version_tag="v3nf")   # cached
    why = ex.explain(runner_row)        # runner_row: dict / pd.Series
    why["top_positive"]   # [{feature, label, value, contribution}, ...]

SHAP contributions are in the model's margin (log-odds) space: they sum to
``base_margin + Σ contributions = predicted_margin``, and ``sigmoid`` of that is
the raw model probability. Sign is what the UI cares about — a positive
contribution pushed this horse's win chance up, negative pushed it down.
Calibration is monotonic, so it never changes a driver's sign or ordering.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from models.features import FEATURE_COLS
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parent.parent

# Human-readable labels for the UI. Any feature absent here falls back to a
# title-cased version of its column name, so new features still render sanely.
FEATURE_LABELS = {
    "field_size": "Field size",
    "horse_speed": "Speed figure",
    "horse_speed_rank": "Speed rank in race",
    "speed_trend": "Speed trend (improving/declining)",
    "race_complexity": "Race competitiveness",
    "distance_furlongs": "Race distance",
    "historical_win_rate": "Recent win rate",
    "historical_place_rate": "Recent place rate",
    "jockey_win_rate": "Jockey win rate",
    "trainer_win_rate": "Trainer win rate",
    "jt_combo_win_rate": "Jockey+trainer win rate",
    "jt_combo_runs": "Jockey+trainer runs together",
    "going_pref_win_rate": "Win rate on this going",
    "going_pref_place_rate": "Place rate on this going",
    "course_win_rate": "Win rate at this course",
    "course_place_rate": "Place rate at this course",
    "course_runs": "Runs at this course",
    "distance_win_rate": "Win rate at this trip",
    "distance_place_rate": "Place rate at this trip",
    "distance_runs": "Runs at this trip",
    "days_since_last_run": "Days since last run",
    "horse_career_runs": "Career runs",
    "going_speed": "Going (numeric)",
    # market features (present only in the priced model)
    "implied_prob": "Market-implied probability",
    "overround_norm_prob": "Overround-adjusted market prob",
    "log_odds": "Log market odds",
    "market_rank": "Market rank in race",
    "ew_value_index": "Each-way value index",
    "odds_drift": "Odds drift",
    "odds_value_delta": "Odds value delta",
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, x))))


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


@lru_cache(maxsize=16)
def get_explainer(
    target: str = "won",
    version_tag: str = "v3nf",
    model_dir: Optional[str] = None,
) -> "Explainer":
    """Load (once, then cached) the model for ``target`` and wrap it in an
    :class:`Explainer`. Feature columns come from ``catboost_<tag>_meta.json``
    when present, else fall back to :data:`models.features.FEATURE_COLS`.

    Raises ``FileNotFoundError`` if the model ``.bin`` is missing.
    """
    base = Path(model_dir) if model_dir else _BASE / "models"
    path = base / f"catboost_{target}_{version_tag}.bin"
    if not path.exists():
        raise FileNotFoundError(f"explain: model not found: {path}")

    model = CatBoostClassifier()
    model.load_model(str(path))

    feature_cols = list(FEATURE_COLS)
    meta_path = base / f"catboost_{version_tag}_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                fc = json.load(fh).get("feature_cols")
            if fc:
                feature_cols = fc
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("explain: meta read failed (%s) — using FEATURE_COLS", exc)

    logger.info("explain: loaded %s explainer (%d features)", path.name, len(feature_cols))
    return Explainer(model, feature_cols, target=target)


class Explainer:
    """Wraps a trained CatBoost model to produce per-runner SHAP explanations.

    Stateless after construction and thread-safe for reads, so a single instance
    (via :func:`get_explainer`) can serve every race in a polling loop.
    """

    def __init__(
        self,
        model: CatBoostClassifier,
        feature_cols: list[str],
        target: str = "won",
    ) -> None:
        self.model = model
        self.feature_cols = list(feature_cols)
        self.target = target

    # ── internals ────────────────────────────────────────────────────────────
    def _matrix(self, rows: pd.DataFrame) -> np.ndarray:
        """Select feature_cols (filling any missing column with NaN) as float."""
        out = pd.DataFrame(index=rows.index)
        for c in self.feature_cols:
            out[c] = pd.to_numeric(rows[c], errors="coerce") if c in rows.columns else np.nan
        return out.astype(float).values

    def _shap(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        """Exact tree SHAP. Returns (contribs[n, n_feat], base_margin)."""
        sv = self.model.get_feature_importance(type="ShapValues", data=Pool(X))
        return sv[:, :-1], float(sv[0, -1])

    def _drivers(self, contrib: np.ndarray, values: np.ndarray, top_k: int) -> dict:
        order = np.argsort(contrib)[::-1]  # most positive → most negative
        pos, neg = [], []
        for i in order:
            c = float(contrib[i])
            if abs(c) < 1e-9:
                continue
            item = {
                "feature": self.feature_cols[i],
                "label": label_for(self.feature_cols[i]),
                "value": (None if (values[i] is None or (isinstance(values[i], float)
                          and np.isnan(values[i]))) else float(values[i])),
                "contribution": round(c, 4),
            }
            (pos if c > 0 else neg).append(item)
        return {"top_positive": pos[:top_k], "top_negative": neg[-top_k:][::-1]}

    # ── public API ───────────────────────────────────────────────────────────
    def explain(self, row: Union[pd.Series, dict], top_k: int = 6) -> dict:
        """Explain a single runner. ``row`` is a dict / Series of feature values.

        Returns a dict with the raw model probability, the SHAP base value, and
        the top positive / negative drivers (each: feature, label, value,
        contribution in log-odds). ``top_negative`` is ordered most-negative
        first.
        """
        rows = pd.DataFrame([dict(row)])
        X = self._matrix(rows)
        contribs, base = self._shap(X)
        margin = base + float(contribs[0].sum())
        out = {
            "target": self.target,
            "base_value": round(base, 4),
            "predicted_margin": round(margin, 4),
            "predicted_prob": round(_sigmoid(margin), 4),
        }
        out.update(self._drivers(contribs[0], X[0], top_k))
        return out

    def explain_frame(self, rows: pd.DataFrame, top_k: int = 6) -> list[dict]:
        """Explain every runner in ``rows`` in one SHAP pass (one race's field).

        Order of the returned list matches ``rows``.
        """
        if rows.empty:
            return []
        X = self._matrix(rows)
        contribs, base = self._shap(X)
        out = []
        for i in range(len(rows)):
            margin = base + float(contribs[i].sum())
            d = {
                "target": self.target,
                "base_value": round(base, 4),
                "predicted_margin": round(margin, 4),
                "predicted_prob": round(_sigmoid(margin), 4),
            }
            d.update(self._drivers(contribs[i], X[i], top_k))
            out.append(d)
        return out
