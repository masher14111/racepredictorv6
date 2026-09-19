"""Jockey form-cycle features: detect in-form riders.

Adds per-runner columns:
  - jockey_hot_strike_rate: win rate over a short recent window (default 14 days)
  - jockey_form_zscore:   z-score of (short_rate - long_rate) measuring whether
                            the jockey is currently outperforming their baseline

Both are leak-safe: only strictly-prior runs count toward a row's rate. Columns
degrade to NaN when jockey_id is absent or the jockey has fewer than the
minimum runners in the window.

Step 04 / DECISIONS D25-D26: a runner's WIN and PLACE row of the SAME race
(features/fuse.py) must count as ONE prior result for another jockey/trainer
window, never two, and must never let a horse leak its own market-sibling row
into its own jockey's rate. Delegated to features._trailing_fast, which
performs that market-duplicate collapse once, shared with every other trailing
aggregate in this codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features._trailing_fast import windowed_win_rate_days

_DEFAULT_SHORT_WINDOW_DAYS = 14
_DEFAULT_LONG_WINDOW_DAYS = 90
_DEFAULT_MIN_RUNNERS = 5


def _trailing_jockey_win_rate(
    df: pd.DataFrame,
    window_days: int,
    min_runners: int,
) -> pd.Series:
    """Leak-safe trailing win rate for each row's jockey over strictly-prior
    runs within ``window_days``, grouped by jockey_id."""
    n = len(df)
    if n == 0 or "jockey_id" not in df.columns or "race_date" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    rate, _ = windowed_win_rate_days(
        df.reset_index(drop=True), "jockey_id",
        window_days=window_days, min_runners=min_runners)
    return pd.Series(rate, index=df.index, dtype=float)


def _trailing_jockey_n(df: pd.DataFrame, window_days: int) -> pd.Series:
    """Count of strictly-prior known-position runs per jockey within window."""
    n = len(df)
    if n == 0 or "jockey_id" not in df.columns or "race_date" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    _, cnt = windowed_win_rate_days(
        df.reset_index(drop=True), "jockey_id",
        window_days=window_days, min_runners=0)
    return pd.Series(cnt, index=df.index, dtype=int)


def add_jockey_form(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add jockey_hot_strike_rate and jockey_form_zscore.

    Parameters
    ----------
    df : pd.DataFrame
        Must carry ``jockey_id``, ``race_date``, ``position`` columns.
    cfg : dict or None
        Optional override: ``short_window_days``, ``long_window_days``,
        ``min_runners``.
    """
    cfg = cfg or {}
    short_days = int(cfg.get("short_window_days", _DEFAULT_SHORT_WINDOW_DAYS))
    long_days = int(cfg.get("long_window_days", _DEFAULT_LONG_WINDOW_DAYS))
    min_r = int(cfg.get("min_runners_in_window", _DEFAULT_MIN_RUNNERS))

    out = df.copy()

    if "jockey_id" not in out.columns or "race_date" not in out.columns:
        out["jockey_hot_strike_rate"] = pd.NA
        out["jockey_form_zscore"] = pd.NA
        return out

    out["jockey_hot_strike_rate"] = _trailing_jockey_win_rate(out, short_days, min_r)
    long_rate = _trailing_jockey_win_rate(out, long_days, min_r)

    short_n = _trailing_jockey_n(out, short_days)
    long_rate_s = pd.to_numeric(long_rate, errors="coerce")
    short_rate_s = pd.to_numeric(out["jockey_hot_strike_rate"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(
            long_rate_s.to_numpy(dtype=float)
            * (1.0 - long_rate_s.to_numpy(dtype=float))
            / np.where(
                short_n.to_numpy(dtype=float) > 0,
                short_n.to_numpy(dtype=float),
                np.nan,
            )
        )
        z = (
            short_rate_s.to_numpy(dtype=float)
            - long_rate_s.to_numpy(dtype=float)
        ) / se

    out["jockey_form_zscore"] = pd.Series(
        np.where(np.isfinite(z), z, np.nan), index=out.index
    )
    return out
