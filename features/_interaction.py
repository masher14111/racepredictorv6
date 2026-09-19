"""Engineered interaction features that combine two existing signals.

CatBoost learns interactions natively, but feeding them explicitly reduces the
depth/trees needed to discover them — a benefit when the feature space grows.

Adds per-runner columns:
  - speed_distance_profile: horse_speed × distance_furlongs interaction
  - form_market_disagreement: abs(historical_win_rate - implied_prob) — how much
    the horse's form disagrees with the market
  - hot_connection: trainer_hot_strike_rate × jockey_win_rate — strong
    connections in a single feature

All columns degrade to NaN when any input is missing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Numeric column or all-NaN if absent."""
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered interaction features."""
    out = df.copy()

    # ── speed × distance — stamina/pace profile ─────────────────
    # A fast horse at a sprint may be lethal; same speed at 12f may be useless.
    # Multiplicative: high speed at long distance → large positive product,
    # high speed at sprint → moderate product.
    speed = _col(out, "horse_speed")
    dist_f = _col(out, "distance_furlongs")
    out["speed_distance_profile"] = speed * dist_f

    # ── form-vs-market disagreement ─────────────────────────────
    # A horse with a strong win rate that the market rates poorly (low implied_prob)
    # has high disagreement — this is where value lies. Absolute difference so
    # both under-rated and over-rated horses get a signal.
    hist_wr = _col(out, "historical_win_rate")
    imp = _col(out, "implied_prob")
    out["form_market_disagreement"] = (hist_wr - imp).abs()

    # ── hot connection: trainer form × jockey quality ───────────
    # A trainer in top form with a top jockey is a multiplier effect.
    tr_hot = _col(out, "trainer_hot_strike_rate")
    jk_win = _col(out, "jockey_win_rate")
    # Also use trainer_form_zscore as an alternative, fallback-friendly version
    tr_z = _col(out, "trainer_form_zscore")
    jk_hot = _col(out, "jockey_hot_strike_rate") if "jockey_hot_strike_rate" in out.columns \
        else pd.Series(np.nan, index=out.index)
    # Primary: trainer_hot × jockey_win (both are rates 0-1, product captures synergy)
    out["hot_connection"] = tr_hot * jk_win
    # Secondary: trainer_z × jockey_z (z-scores compound, more robust for small counts)
    jk_z = _col(out, "jockey_form_zscore") if "jockey_form_zscore" in out.columns \
        else pd.Series(np.nan, index=out.index)
    out["hot_connection_z"] = tr_z * jk_z

    return out