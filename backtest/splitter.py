"""Walk-forward (rolling-origin) cross-validation folds.

The single source of truth for the no-look-ahead guarantee. Each fold is a pair
of date cutoffs::

    train: race_date <= train_end (== origin T)
    test : train_end <  race_date <= test_end  (T, T+window]

so by construction no test race is ever <= a train race. Folds tile the timeline
forward by ``step_days`` (defaulting to a non-overlapping ``test_window_days``),
with either an **expanding** train window (default — train on everything up to T)
or a **rolling** one (only the last ``train_window_days`` before T).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class Fold:
    """One walk-forward step. All bounds are timezone-aware Timestamps.

    ``train_start`` is None for an expanding window (use all history up to
    ``train_end``); a Timestamp for a rolling window.
    """

    index: int
    train_start: Optional[pd.Timestamp]
    train_end: pd.Timestamp          # the origin T (inclusive upper bound for train)
    test_start: pd.Timestamp         # == train_end; test is strictly after it
    test_end: pd.Timestamp           # inclusive upper bound for test (T + window)

    def train_mask(self, race_date: pd.Series) -> pd.Series:
        rd = pd.to_datetime(race_date, utc=True, errors="coerce")
        m = rd <= self.train_end
        if self.train_start is not None:
            m &= rd >= self.train_start
        return m.fillna(False)

    def test_mask(self, race_date: pd.Series) -> pd.Series:
        rd = pd.to_datetime(race_date, utc=True, errors="coerce")
        m = (rd > self.train_end) & (rd <= self.test_end)
        return m.fillna(False)


@dataclass
class WalkForwardConfig:
    """Knobs for :func:`walk_forward_folds`.

    min_train_days     : history required before the first prediction is made.
    test_window_days   : length of each out-of-sample prediction window.
    step_days          : how far the origin rolls between folds. Defaults to
                         ``test_window_days`` (back-to-back, non-overlapping test
                         windows that tile the timeline exactly once).
    rolling            : False = expanding train window (recommended; more data).
                         True  = rolling window of ``train_window_days``.
    train_window_days  : rolling train span; defaults to ``min_train_days``.
    """

    min_train_days: int = 365
    test_window_days: int = 30
    step_days: Optional[int] = None
    rolling: bool = False
    train_window_days: Optional[int] = None


def walk_forward_folds(race_dates, config: WalkForwardConfig) -> list[Fold]:
    """Build the ordered list of :class:`Fold` for the span of ``race_dates``.

    The first origin is ``min(date) + min_train_days``; the last fold is the one
    whose test window still contains at least one calendar day of data. An empty
    list is returned when the data is shorter than ``min_train_days`` + one day.
    """
    rd = pd.to_datetime(pd.Series(race_dates), utc=True, errors="coerce").dropna()
    if rd.empty:
        return []

    start = rd.min().normalize()
    last = rd.max().normalize()
    step = int(config.step_days or config.test_window_days)
    if step <= 0:
        raise ValueError("step_days/test_window_days must be positive")

    window = pd.Timedelta(days=config.test_window_days)
    origin = start + pd.Timedelta(days=config.min_train_days)

    folds: list[Fold] = []
    i = 0
    while origin < last:
        test_end = origin + window
        train_start = None
        if config.rolling:
            span = config.train_window_days or config.min_train_days
            train_start = origin - pd.Timedelta(days=span)
        folds.append(Fold(
            index=i,
            train_start=train_start,
            train_end=origin,
            test_start=origin,
            test_end=test_end,
        ))
        origin = origin + pd.Timedelta(days=step)
        i += 1
    return folds
