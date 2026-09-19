"""Price movement signals from genuine pre-off prices (NEVER odds_finish).

Adds per-runner columns:
  - price_steam_pct:  percentage shortened from earliest pre-off to current price
  - relative_market_share: runner's implied-prob share of the per-race book
  - market_book_pct:  the book overround as a percentage (field sum of 1/d − 1)

All inputs are pre-off (morningwap, odds_decimal, ppwap) — odds_finish is NEVER
read, so these features are safe for live inference and do not leak the outcome
(audit C2). When any source price is missing the column degrades to NaN, which
CatBoost handles natively.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.derive import _race_cols


def _best_preoff_price(row) -> float | None:
    """First valid pre-off price in priority order: odds_decimal > ppwap > morningwap.

    odds_finish is deliberately excluded — it is the returned/finishing SP and
    would leak the outcome into training features.
    """
    for col in ("odds_decimal", "ppwap", "morningwap"):
        v = row.get(col)
        if v is not None and not pd.isna(v) and float(v) > 1.0:
            return float(v)
    return None


def _valid_odds(series) -> np.ndarray:
    """Decimal odds as float; anything <= 1 or non-numeric → NaN."""
    d = pd.to_numeric(pd.Series(series), errors="coerce").to_numpy(dtype=float).copy()
    d[~(d > 1.0)] = np.nan
    return d


def add_market_movement(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-movement features (pre-off only, leak-free by design)."""
    out = df.copy()

    # ── price_steam_pct: morning-to-current move ──────────────────────────
    # Positive = backed in (price shortened), negative = drifted out.
    # Uses morningwap as the baseline and odds_decimal/ppwap as current.
    mw = _valid_odds(out["morningwap"]) if "morningwap" in out.columns \
        else np.full(len(out), np.nan)
    cur = np.array([_best_preoff_price(r) for _, r in out.iterrows()], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["price_steam_pct"] = (mw - cur) / mw
    out["price_steam_pct"] = pd.to_numeric(out["price_steam_pct"], errors="coerce")

    # ── relative_market_share + market_book_pct ──────────────────────────
    # Per-race: implied prob of each runner / total book sum → share of
    # the total market. market_book_pct = (total - 1) * 100 (the overround).
    rkey = _race_cols(out)
    # implied = 1 / current price for each runner; NaN where price unknown
    implied = pd.Series(
        np.where(np.isfinite(cur), 1.0 / cur, np.nan), index=out.index
    )
    totals = implied.groupby([out[k] for k in rkey]).transform("sum")
    out["relative_market_share"] = implied / totals.where(totals > 0)
    out["relative_market_share"] = pd.to_numeric(
        out["relative_market_share"], errors="coerce"
    )
    out["market_book_pct"] = (totals - 1.0) * 100.0
    out["market_book_pct"] = pd.to_numeric(out["market_book_pct"], errors="coerce")

    return out