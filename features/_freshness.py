"""Binned freshness signals and market confidence indicators.

Adds per-runner columns:
  - freshness_band: categorical bucket for days_since_last_run
    (fresh 1-7, normal 8-21, layoff 22-90, long_absence 91+, first_time, unknown)
  - is_steaming:   binary — price shortened >10% from morning to current
  - is_drifting:   binary — price lengthened >10% from morning to current
  - market_confidence: [0,1] composite: (1 - abs(steam_pct)) * (1 - abs(drift_pct))
     clipped to zero — measure of market agreement on the price

All are leak-safe (no future data accessed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _freshness_band(days: float | None) -> str:
    """Map days-since-last-run to a human-readable band."""
    if days is None or pd.isna(days) or days < 0:
        return "unknown"
    if days == 0:
        return "first_time"
    if days <= 7:
        return "fresh"
    if days <= 21:
        return "normal"
    if days <= 90:
        return "layoff"
    return "long_absence"


def add_freshness_and_market(df: pd.DataFrame) -> pd.DataFrame:
    """Add freshness_band, is_steaming, is_drifting, market_confidence."""
    out = df.copy()

    # ── freshness_band (categorical string, CatBoost handles strings natively) ─
    dsr = (
        pd.to_numeric(out["days_since_last_run"], errors="coerce")
        if "days_since_last_run" in out.columns
        else pd.Series(pd.NA, index=out.index)
    )
    # Encode as ordinal integers so CatBoost's .astype(float) pipeline handles it.
    # Order: unknown=0, first_time=1, long_absence=2, layoff=3, normal=4, fresh=5
    _BAND_ORDER = {"unknown": 0, "first_time": 1, "long_absence": 2, "layoff": 3, "normal": 4, "fresh": 5}
    bands = [_freshness_band(v) for v in dsr]
    out["freshness_band"] = [_BAND_ORDER.get(b, 0) for b in bands]

    # ── market confidence signals ──────────────────────────────────────────
    steam = (
        pd.to_numeric(out["price_steam_pct"], errors="coerce")
        if "price_steam_pct" in out.columns
        else pd.Series(pd.NA, index=out.index)
    )
    # steam > 0 = price shortened (backed in). > 10% = significant
    out["is_steaming"] = steam > 0.10
    # steam < 0 = price lengthened (drifting out). < -10% = significant drift
    out["is_drifting"] = steam < -0.10

    # market_confidence: how settled is the price? Low when steam is large
    # (either direction = uncertainty). 1.0 = no movement, 0.5 = 50% move.
    abs_steam = steam.abs()
    out["market_confidence"] = (1.0 - abs_steam.clip(upper=1.0)).clip(lower=0.0)

    return out