"""Trainer form-cycle features: detect "in-form" yards.

Adds per-runner columns:
  - trainer_hot_strike_rate: win rate over a short recent window (default 14 days)
  - trainer_form_zscore:    z-score of (short_rate - long_rate) measuring whether
                             the trainer is currently outperforming their baseline

Both are leak-safe: only strictly-prior runs count toward a row's rate. Columns
degrade to NaN when trainer_id is absent or the trainer has fewer than the
minimum runners in the window.

Step 04 / DECISIONS D25-D26: a runner's WIN and PLACE row of the SAME race
(features/fuse.py) must count as ONE prior result for another trainer/jockey
window, never two, and must never let a horse leak its own market-sibling row
into its own trainer's rate. Delegated to features._trailing_fast, which
performs that market-duplicate collapse once, shared with every other trailing
aggregate in this codebase.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features._trailing_fast import windowed_win_rate_days

# Default config (overridable via cfg parameter)
_DEFAULT_SHORT_WINDOW_DAYS = 14
_DEFAULT_LONG_WINDOW_DAYS = 90
_DEFAULT_MIN_RUNNERS = 5


def _trailing_trainer_win_rate(
    df: pd.DataFrame,
    window_days: int,
    min_runners: int,
) -> pd.Series:
    """Leak-safe trailing win rate for each row's trainer over strictly-prior
    runs within ``window_days``, grouped by trainer_id.

    Returns a float Series of rates (NaN where insufficient data). Trainer rows
    with a missing trainer_id produce NaN (the groupby filters them).
    """
    n = len(df)
    if n == 0 or "trainer_id" not in df.columns or "race_date" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    rate, _ = windowed_win_rate_days(
        df.reset_index(drop=True), "trainer_id",
        window_days=window_days, min_runners=min_runners)
    return pd.Series(rate, index=df.index, dtype=float)


def _trailing_trainer_n(df: pd.DataFrame, window_days: int) -> pd.Series:
    """Count of strictly-prior known-position runs per trainer within window.
    Companion to _trailing_trainer_win_rate; returns the denominator so the
    z-score can be properly standardised.
    """
    n = len(df)
    if n == 0 or "trainer_id" not in df.columns or "race_date" not in df.columns:
        return pd.Series(0, index=df.index, dtype=int)
    # min_runners=0: this helper reports the raw count regardless of the
    # win-rate gate; the rate function applies min_runners separately.
    _, cnt = windowed_win_rate_days(
        df.reset_index(drop=True), "trainer_id",
        window_days=window_days, min_runners=0)
    return pd.Series(cnt, index=df.index, dtype=int)


def add_trainer_form(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add trainer_hot_strike_rate and trainer_form_zscore.

    Parameters
    ----------
    df : pd.DataFrame
        Must carry ``trainer_id``, ``race_date``, ``position`` columns.
    cfg : dict or None
        Optional override: ``short_window_days``, ``long_window_days``,
        ``min_runners``.
    """
    cfg = cfg or {}
    short_days = int(cfg.get("short_window_days", _DEFAULT_SHORT_WINDOW_DAYS))
    long_days = int(cfg.get("long_window_days", _DEFAULT_LONG_WINDOW_DAYS))
    min_r = int(cfg.get("min_runners_in_window", _DEFAULT_MIN_RUNNERS))

    out = df.copy()

    if "trainer_id" not in out.columns or "race_date" not in out.columns:
        out["trainer_hot_strike_rate"] = pd.NA
        out["trainer_form_zscore"] = pd.NA
        return out

    out["trainer_hot_strike_rate"] = _trailing_trainer_win_rate(
        out, short_days, min_r
    )
    long_rate = _trailing_trainer_win_rate(out, long_days, min_r)

    # z-score: (short - long) / std_est. Use a pooled binomial SE as the
    # standardiser — more stable than a per-trainer sample std when counts
    # are small.
    # pooled_p = long_rate (baseline), n = runs in short window.
    # SE = sqrt(pooled_p * (1 - pooled_p) / n_short)
    # z = (short - pooled_p) / SE
    short_n = _trailing_trainer_n(out, short_days)
    long_rate_s = pd.to_numeric(long_rate, errors="coerce")
    short_rate_s = pd.to_numeric(out["trainer_hot_strike_rate"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(
            long_rate_s.to_numpy(dtype=float)
            * (1.0 - long_rate_s.to_numpy(dtype=float))
            / np.where(short_n.to_numpy(dtype=float) > 0, short_n.to_numpy(dtype=float), np.nan)
        )
        z = (short_rate_s.to_numpy(dtype=float) - long_rate_s.to_numpy(dtype=float)) / se

    out["trainer_form_zscore"] = pd.Series(
        np.where(np.isfinite(z), z, np.nan), index=out.index
    )
    return out
