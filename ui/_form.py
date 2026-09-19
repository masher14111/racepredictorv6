"""Historical form + feature-row resolution for the detail pages.

The race / horse detail pages need two things the prediction cache
(``predictions.json``) does not carry:

1. **A horse's recent form** — its last runs (finishing position, field size,
   going, class, distance, speed figure, starting price) — for the Horse page's
   form table and the Race page's at-a-glance context.
2. **A feature row per runner** so :mod:`models.explain` can produce the SHAP
   "why" drivers. This is the data path that was silently empty before: the old
   ``data/features.parquet`` was a 5-row synthetic fixture whose ``horse_id``\\s
   (``h1``/``h2``) never matched the hashed live ids, so every breakdown fell
   back to "unavailable".

Both are served from the **labelled training matrix**
(``data/features/training_full.parquet``, ~248k rows, one row per past runner)
which is keyed on the same hashed ``horse_id`` the predictor emits — so a live
runner with prior runs resolves to its real history and a real feature profile.

Resolution order for the SHAP feature row is, in priority:

1. ``data/inference_features.parquet`` — exact per-runner rows the predictor
   persisted for *today's* races (the live, race-accurate source; written by
   ``models.predictor`` when it has live data). Source tag ``"live"``.
2. the most recent row for the horse in the training matrix — its latest known
   form profile. Source tag ``"history"``.

The core functions are Streamlit-free and take an injected loader so they can be
unit-tested headless; the module-level cached wrappers bind them to the real
parquet files for the pages.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ui._components import headline_win_prob

_BASE = Path(__file__).resolve().parent.parent
_MATRIX_PATH = _BASE / "data" / "features" / "training_full.parquet"
_INFERENCE_PATH = _BASE / "data" / "inference_features.parquet"

# Columns the form table reads from the matrix (others are ignored). Kept narrow
# so the cached read stays small even though the matrix is wide.
_FORM_COLS = [
    "race_date", "venue", "position", "field_size", "going", "going_band",
    "race_class", "distance", "distance_furlongs", "horse_speed",
    "horse_speed_rank", "odds_decimal", "sp", "won", "placed",
    "jockey_name", "trainer_name",
]
# Connection / aggregate stats live on the latest row (already computed by the
# feature engine over each horse's trailing window).
_STAT_COLS = [
    "jockey_name", "trainer_name", "jockey_win_rate", "trainer_win_rate",
    "jt_combo_win_rate", "jt_combo_runs", "historical_win_rate",
    "historical_place_rate", "horse_career_runs", "course_win_rate",
    "course_place_rate", "distance_win_rate", "distance_place_rate",
    "going_pref_win_rate", "going_pref_place_rate", "days_since_last_run",
]


# ── pure helpers (headless-testable) ──────────────────────────────────────────

def _coerce_num(val) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _coerce_str(val) -> Optional[str]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip()
    return s or None


def _run_record(row: pd.Series) -> dict:
    """One past run → a UI-ready dict (None for any missing field)."""
    rd = row.get("race_date")
    date_str = None
    if rd is not None and not (isinstance(rd, float) and pd.isna(rd)):
        date_str = rd.strftime("%Y-%m-%d") if hasattr(rd, "strftime") else str(rd)[:10]
    pos = _coerce_num(row.get("position"))
    return {
        "date": date_str,
        "venue": _coerce_str(row.get("venue")),
        "position": int(pos) if pos is not None else None,
        "field_size": (lambda f: int(f) if f is not None else None)(
            _coerce_num(row.get("field_size"))),
        "going": _coerce_str(row.get("going")),
        "race_class": _coerce_str(row.get("race_class")),
        "distance": _coerce_str(row.get("distance"))
                    or (lambda f: f"{f:g}f" if f is not None else None)(
                        _coerce_num(row.get("distance_furlongs"))),
        "speed": _coerce_num(row.get("horse_speed")),
        "speed_rank": (lambda f: int(f) if f is not None else None)(
            _coerce_num(row.get("horse_speed_rank"))),
        "sp": _coerce_num(row.get("sp")) or _coerce_num(row.get("odds_decimal")),
        "won": bool(_coerce_num(row.get("won"))) if _coerce_num(row.get("won")) is not None else None,
        "placed": bool(_coerce_num(row.get("placed"))) if _coerce_num(row.get("placed")) is not None else None,
    }


def horse_history_from(matrix: pd.DataFrame, horse_id: str, limit: int = 8) -> list[dict]:
    """Most-recent-first list of a horse's past runs from the training matrix.

    Returns ``[]`` when the horse has no rows. ``matrix`` is any DataFrame with
    a ``horse_id`` column (the injected real one, or a test fixture).
    """
    if matrix is None or matrix.empty or "horse_id" not in matrix.columns or not horse_id:
        return []
    sub = matrix[matrix["horse_id"].astype(str) == str(horse_id)]
    if sub.empty:
        return []
    if "race_date" in sub.columns:
        sub = sub.sort_values("race_date", ascending=False)
    else:
        sub = sub.iloc[::-1]
    return [_run_record(r) for _, r in sub.head(limit).iterrows()]


def connection_stats_from(matrix: pd.DataFrame, horse_id: str) -> dict:
    """Aggregate form/connection rates for a horse, read off its latest matrix row.

    These columns are the feature engine's trailing-window aggregates, so the
    most recent row carries the up-to-date figures. ``{}`` when absent.
    """
    if matrix is None or matrix.empty or "horse_id" not in matrix.columns or not horse_id:
        return {}
    sub = matrix[matrix["horse_id"].astype(str) == str(horse_id)]
    if sub.empty:
        return {}
    if "race_date" in sub.columns:
        sub = sub.sort_values("race_date", ascending=False)
    row = sub.iloc[0]
    out: dict = {}
    for c in _STAT_COLS:
        if c not in sub.columns:
            continue
        if c in ("jockey_name", "trainer_name"):
            out[c] = _coerce_str(row.get(c))
        else:
            out[c] = _coerce_num(row.get(c))
    out["runs"] = len(sub)
    return out


def feature_row_from(
    matrix: pd.DataFrame,
    inference: Optional[pd.DataFrame],
    horse_id: str,
) -> Optional[tuple[dict, str]]:
    """Resolve one runner's feature row for SHAP, with its source tag.

    Prefers the live inference store (exact for today's race), then the horse's
    most recent training-matrix row. Returns ``(row_dict, source)`` where
    ``source`` is ``"live"`` or ``"history"``, or ``None`` when neither has it.
    """
    if not horse_id:
        return None
    if inference is not None and not inference.empty and "horse_id" in inference.columns:
        hit = inference[inference["horse_id"].astype(str) == str(horse_id)]
        if not hit.empty:
            return hit.iloc[0].to_dict(), "live"
    if matrix is not None and not matrix.empty and "horse_id" in matrix.columns:
        hit = matrix[matrix["horse_id"].astype(str) == str(horse_id)]
        if not hit.empty:
            if "race_date" in hit.columns:
                hit = hit.sort_values("race_date", ascending=False)
            return hit.iloc[0].to_dict(), "history"
    return None


# ── derivations for the UI (pure, headless-testable) ──────────────────────────

def trend_label(speed_trend: Optional[float]) -> Optional[str]:
    """Form trend from the speed-figure trend feature.

    ``speed_trend`` is the slope of a horse's recent speed figures, so its sign
    is an honest "improving / declining" read. We deliberately do NOT call this
    early-pace "run-style" — sectional/early-position data isn't in this dataset,
    so claiming front-runner/hold-up would be fabricated. ``None`` when absent.
    """
    v = _coerce_num(speed_trend)
    if v is None:
        return None
    if v > 0.5:
        return "Improving"
    if v < -0.5:
        return "Declining"
    return "Steady"


def race_shape(selections: list[dict]) -> dict:
    """Plain-language summary of a race for the "model's view" panel.

    Reads only the prediction-cache fields (the headline win prob —
    ``won_prob_normalized`` with a ``won_prob`` fallback — plus ``value_bet``,
    ``horse_name``, ``rank``); how clear-cut the race is comes from the gap
    between the top two headline win probabilities.
    """
    ranked = sorted(
        (s for s in selections if headline_win_prob(s) is not None),
        key=lambda s: (s.get("rank") or 999),
    )
    n_value = sum(1 for s in selections if s.get("value_bet"))
    if not ranked:
        return {"top_name": None, "top_win_pct": None, "shape": "no scored runners",
                "n_value": n_value, "note": ""}
    top = ranked[0]
    p1 = float(headline_win_prob(top) or 0.0)
    p2 = float(headline_win_prob(ranked[1]) or 0.0) if len(ranked) > 1 else 0.0
    gap = p1 - p2
    if p1 >= 0.45 and gap >= 0.18:
        shape = "a strong, clear favourite"
    elif gap >= 0.10:
        shape = "a likely favourite over a chasing pack"
    elif gap >= 0.04:
        shape = "competitive at the head of the market"
    else:
        shape = "wide-open — little between the front pair"
    note = ""
    if n_value:
        note = ("The model rates the flagged runner(s) above their market price — "
                "see the value badge and EV on each.")
    elif p1 < 0.2:
        note = ("No standout: the model's top win probability is modest, so treat "
                "the forecast as a lean rather than a confident call.")
    return {
        "top_name": top.get("horse_name"),
        "top_win_pct": f"{p1 * 100:.0f}% win",
        "shape": shape,
        "n_value": n_value,
        "note": note,
    }


# ── cached loaders (bind the pure core to the real parquet files) ─────────────

def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file, or an empty frame when it is absent/unreadable.

    Best-effort by design: the detail pages degrade to an honest "no data" state
    rather than crash if the form store is missing or schema-drifted.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_matrix() -> pd.DataFrame:
    """The labelled training matrix (full frame).

    Returned whole — the form helpers read only the columns they need, and the
    SHAP explainer pulls its own ``feature_cols`` — so a narrowed copy would just
    risk dropping a column the explainer wants. Cached by Streamlit at the call
    site; kept Streamlit-free here so it stays importable and testable.
    """
    return _read_parquet(_MATRIX_PATH)


def load_inference() -> pd.DataFrame:
    """Per-runner feature rows the predictor persisted for today's races (or empty)."""
    return _read_parquet(_INFERENCE_PATH)
