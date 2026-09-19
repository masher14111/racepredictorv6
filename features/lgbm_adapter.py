"""Adapter: map v4's derived feature matrix into a clean, leak-free LightGBM matrix.

The LightGBM grouped-softmax line (``models/lgbm_softmax.py``) is a WIN model: unlike
the price-free CatBoost "v3nf" value variant, it SHOULD see the market price — the
market is the single strongest win signal. The one hard rule is chronological
integrity: every feature must be point-in-time, knowable strictly BEFORE the off.

v4's README admits the *priced* CatBoost model can leak the finishing SP. This
adapter is the firewall between v4's features and the LightGBM line:

  * Market features are REBUILT here from a genuine PRE-OFF price
    (``morningwap`` → ``ppwap`` fallback). We deliberately do NOT trust v4's
    in-place market columns, whose price selector (``derive._best_odds``:
    ``odds_decimal`` → ``sp`` → ``morningwap``) *could* admit a starting/board
    price if the upstream null-out of ``sp`` ever regressed. Re-deriving from
    ``morningwap``/``ppwap`` makes the price provenance self-evident and testable
    (``tools/audit_lgbm_features.py`` reconstructs them to prove it).

  * Any v4 feature derived from a POST-OFF / settling price is EXCLUDED and the
    reason is recorded in ``EXCLUDED_FEATURES``:
      - ``odds_drift``     = (morningwap − sp)/morningwap — uses ``sp`` (Betfair SP).
      - ``ew_value_index`` — its ``_ew_best_odds`` chain (odds_decimal → sp) can
                             pick up ``sp`` (Betfair SP).
    Raw post-off columns (``odds_finish``, ``sp``, ``position``, ``won``, …) are
    never selected — see ``POST_OFF_COLS``.

  * Every other selected v4 feature is a strictly-prior trailing rate, a
    within-race structural rank, or a pre-off racecard fact — all point-in-time
    (verified by reading features/derive.py, engine.py and the _*.py builders:
    each trailing primitive uses ``strict=True`` and the speed proxy averages only
    PRIOR runs' finish percentiles).

The 8 empirically-dead v4 columns (100%-null Timeform/class sources + low-signal
proxies, see ``models.features.EMPIRICALLY_DEAD_COLS``) are dropped by basing the
selection on ``LEAN_FEATURE_COLS`` — this keeps the audit's null-rate report clean.

Public API
----------
``build_lgbm_matrix(df, *, inference) -> (X, y, race_ids)``
    X        : float DataFrame of leak-free features (``FINAL_FEATURE_COLS`` order),
               ready for ``models.lgbm_softmax.LGBMSoftmaxModel``. NaN where a source
               is missing (LightGBM handles NaN natively).
    y        : np.ndarray of win labels (``won``) when ``inference=False``; ``None``
               when ``inference=True``.
    race_ids : np.ndarray race identifier per row (``race_uid``) for per-race softmax.

This module depends only on numpy/pandas + the v4 feature whitelist, mirroring the
dependency-light contract of ``models/lgbm_softmax.py``.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from models.features import LEAN_FEATURE_COLS, LEAN_PRICE_FREE_FEATURE_COLS

logger = logging.getLogger(__name__)

# ── Pre-off price sources ────────────────────────────────────────────────────
# Genuine pre-off Betfair prices in priority order. morningwap = morning
# weighted-average price; ppwap = pre-play (i.e. pre-off) weighted-average price.
# Deliberately EXCLUDES ``sp`` (Betfair SP / live board SP) and ``odds_finish``
# (finishing SP) — both are settling/closing prices that postdate the bet decision.
PREOFF_PRICE_COLS: tuple[str, ...] = ("morningwap", "ppwap")

# ── Market block re-derived from the pre-off price (NOT copied from df) ───────
_REBUILT_MARKET: tuple[str, ...] = (
    "implied_prob",
    "overround_norm_prob",
    "log_odds",
    "market_rank",
    "odds_value_delta",
    "form_market_disagreement",
)

# ── v4 features excluded from the LightGBM line, with the leakage reason ──────
EXCLUDED_FEATURES: dict[str, str] = {
    "odds_drift": (
        "built as (morningwap - sp)/morningwap; reads `sp` (Betfair SP), "
        "a settling/closing price -> post-off"
    ),
    "ew_value_index": (
        "its price selector `_ew_best_odds` chains odds_decimal -> sp; "
        "can admit `sp` (Betfair SP) -> post-off"
    ),
}

# ── Columns that must NEVER appear in the matrix (post-off / outcome) ─────────
# The audit asserts FINAL_FEATURE_COLS is disjoint from this set.
POST_OFF_COLS: frozenset[str] = frozenset(
    {
        "odds_finish", "sp", "odds_drift", "ew_value_index",
        "position", "won", "placed", "placed_2", "showed",
        "sp_finish", "finish_position", "finishing_position", "result_position",
    }
)

# Final feature order = v4's lean whitelist minus the excluded columns. Rebuilt
# market columns keep their original positions (recomputed from the pre-off price);
# every other column is selected from df as-is.
FINAL_FEATURE_COLS: list[str] = [c for c in LEAN_FEATURE_COLS if c not in EXCLUDED_FEATURES]
SELECTED_V4_COLS: list[str] = [c for c in FINAL_FEATURE_COLS if c not in _REBUILT_MARKET]

# Step 10: the price-free ("independent") branch of the same LightGBM line —
# every rebuilt/raw market column stripped, for the honest independent-vs-
# market-assisted comparison DESIGN.md requires. Same safety net as
# FINAL_FEATURE_COLS (LEAN_PRICE_FREE_FEATURE_COLS already excludes every
# PRICE_FEATURE_COLS/MARKET_DERIVED_FEATURE_COLS member — models/features.py).
INDEPENDENT_FEATURE_COLS: list[str] = [
    c for c in LEAN_PRICE_FREE_FEATURE_COLS if c not in EXCLUDED_FEATURES
]
_BLOCKED_FEATURE_COLS: frozenset[str] = frozenset(EXCLUDED_FEATURES) | POST_OFF_COLS

# freshness_band is the one categorical v4 feature. The current builder emits the
# ordinal int directly, but older parquets stored the string label — encode either
# form to this ordinal so the matrix is fully numeric for LightGBM. Order mirrors
# features/_freshness.py::_BAND_ORDER.
_FRESHNESS_BAND_ORDER: dict[str, int] = {
    "unknown": 0, "first_time": 1, "long_absence": 2,
    "layoff": 3, "normal": 4, "fresh": 5,
}


def _preoff_price(df: pd.DataFrame) -> pd.Series:
    """First valid pre-off price per row: morningwap, then ppwap. NaN if neither
    is a real decimal price (> 1.0). Never reads ``sp`` or ``odds_finish``."""
    price = pd.Series(np.nan, index=df.index, dtype=float)
    for col in PREOFF_PRICE_COLS:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            v = v.where(v > 1.0)
            price = price.where(price.notna(), v)
    return price


def _race_ids(df: pd.DataFrame) -> np.ndarray:
    """Per-race identifier for grouped softmax. Prefers the recovered ``race_uid``
    (venue + off-time); otherwise builds it via the v4 race key."""
    if "race_uid" in df.columns:
        return df["race_uid"].to_numpy()
    from features.derive import add_race_key  # local import: keep base deps light

    return add_race_key(df)["race_uid"].to_numpy()


def _encode_freshness_band(s: pd.Series) -> pd.Series:
    """Coerce freshness_band to its ordinal int, accepting either the int (current
    builder) or the string label (older parquets)."""
    num = pd.to_numeric(s, errors="coerce")
    mapped = s.map(
        lambda v: _FRESHNESS_BAND_ORDER.get(v, np.nan) if isinstance(v, str) else np.nan
    )
    return num.where(num.notna(), mapped)


def _rebuild_market(df: pd.DataFrame, price: pd.Series,
                    race_ids: np.ndarray) -> dict[str, pd.Series]:
    """Recompute the market block from the clean pre-off ``price`` only.

    implied_prob, log_odds            : per-runner functions of the pre-off price.
    overround_norm_prob, market_rank  : per-race (group on race_uid) normalisation.
    odds_value_delta                  : implied_prob - overround_norm_prob.
    form_market_disagreement          : |historical_win_rate - implied_prob|.
    """
    rid = pd.Series(np.asarray(race_ids), index=df.index)
    implied = 1.0 / price
    with np.errstate(divide="ignore", invalid="ignore"):
        log_odds = pd.Series(np.log(price.to_numpy(dtype=float)), index=df.index)
    totals = implied.groupby(rid).transform("sum")
    overround_norm = implied / totals.where(totals > 0)
    market_rank = implied.groupby(rid).rank(ascending=False, method="min")
    odds_value_delta = implied - overround_norm

    if "historical_win_rate" in df.columns:
        hist_wr = pd.to_numeric(df["historical_win_rate"], errors="coerce")
    else:
        hist_wr = pd.Series(np.nan, index=df.index)
    form_market_disagreement = (hist_wr - implied).abs()

    return {
        "implied_prob": implied,
        "overround_norm_prob": overround_norm,
        "log_odds": log_odds,
        "market_rank": market_rank,
        "odds_value_delta": odds_value_delta,
        "form_market_disagreement": form_market_disagreement,
    }


def build_lgbm_matrix(df: pd.DataFrame, *, inference: bool,
                      feature_cols: Optional[Sequence[str]] = None):
    """Build the leak-free (X, y, race_ids) triple for the LightGBM softmax line.

    Parameters
    ----------
    df : pd.DataFrame
        A v4 derived feature matrix (e.g. ``data/features/training.parquet`` or the
        live ``inference_features.parquet``) carrying the raw pre-off price columns
        plus the v4 derived columns and ``race_uid``.
    inference : bool
        True  -> serving: return all rows, ``y`` is None, no label required.
        False -> training: ``y`` is the ``won`` label; rows with a null label are
                 dropped and X/y/race_ids stay aligned.
    feature_cols : optional column whitelist, e.g. :data:`INDEPENDENT_FEATURE_COLS`
        for the price-free branch (step 10's independent-vs-market-assisted
        comparison). Defaults to :data:`FINAL_FEATURE_COLS` (market-assisted,
        unchanged default behaviour). Any excluded/post-off column is rejected
        even if explicitly requested — the leakage guard is never bypassable.

    Returns
    -------
    (X, y, race_ids)
    """
    final_cols = list(feature_cols) if feature_cols is not None else FINAL_FEATURE_COLS
    blocked = [c for c in final_cols if c in _BLOCKED_FEATURE_COLS]
    if blocked:
        raise ValueError(f"feature_cols includes excluded/post-off column(s): {blocked}")

    if df is None or len(df) == 0:
        empty = pd.DataFrame(columns=final_cols, dtype=float)
        y = None if inference else np.array([], dtype=float)
        return empty, y, np.array([], dtype=object)

    race_ids = np.asarray(_race_ids(df))
    price = _preoff_price(df)
    market = _rebuild_market(df, price, race_ids)

    X = pd.DataFrame(index=df.index)
    missing: list[str] = []
    for col in final_cols:
        if col in market:
            X[col] = market[col]
        elif col not in df.columns:
            X[col] = np.nan
            missing.append(col)
        elif col == "freshness_band":
            X[col] = _encode_freshness_band(df[col])
        else:
            X[col] = pd.to_numeric(df[col], errors="coerce")
    X = X.astype(float)

    if missing:
        logger.warning(
            "build_lgbm_matrix: %d v4 feature(s) absent from input, filled NaN: %s",
            len(missing), missing,
        )

    if inference:
        return X.reset_index(drop=True), None, race_ids

    if "won" not in df.columns:
        raise KeyError(
            "build_lgbm_matrix(inference=False) requires a `won` label column; "
            "got columns without it."
        )
    y_series = pd.to_numeric(df["won"], errors="coerce")
    keep = y_series.notna().to_numpy()
    X = X.loc[keep].reset_index(drop=True)
    y = y_series[keep].to_numpy(dtype=float)
    race_ids = race_ids[keep]
    return X, y, race_ids


def feature_provenance() -> dict:
    """Machine-readable provenance of the LightGBM feature set (used by the audit
    and tests). Documents how every column is sourced and what was excluded/why."""
    return {
        "preoff_price_priority": list(PREOFF_PRICE_COLS),
        "rebuilt_from_preoff_price": list(_REBUILT_MARKET),
        "selected_from_v4": list(SELECTED_V4_COLS),
        "excluded": dict(EXCLUDED_FEATURES),
        "post_off_denylist": sorted(POST_OFF_COLS),
        "final_feature_cols": list(FINAL_FEATURE_COLS),
    }
