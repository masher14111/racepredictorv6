"""Strategy (which runners to back) + stake schemes (how much to stake).

Both are deliberately small and pluggable so a backtest can sweep many
"bet when X" rules and staking plans without touching the engine:

    strategies = {
        "ev5_flat":   (Strategy("ev5", min_ev=0.05, min_odds=2, max_odds=26), FlatStake(10)),
        "edge_kelly": (Strategy("edge10", min_edge_pct=0.10), KellyStake(0.25)),
    }

A :class:`Strategy` is a pure *selection* filter over a scored frame; a stake
scheme decides the € amount for the runners it selects. The engine combines them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from backtest import metrics


@dataclass
class Strategy:
    """A pluggable bet-selection rule.

    Every gate is optional and ANDed together — leave one ``None`` to disable it.
    Built-in gates cover the requirements (EV threshold, probability-edge "bet
    when model prob beats implied by X%", longshot gating via an odds band, and a
    minimum model probability). For anything bespoke, pass ``selector`` — a
    ``DataFrame -> bool Series`` callable that fully overrides the gates.

    Parameters
    ----------
    name          : label used in reports / saved runs.
    min_ev        : back iff ``expected_value (p*d-1) >= min_ev``.
    min_edge_pct  : back iff ``p >= implied * (1 + min_edge_pct)`` — i.e. the
                    model prob exceeds the price-implied prob by this fraction.
    min_abs_edge  : back iff ``p - implied >= min_abs_edge`` (absolute pp edge).
    min_odds/max_odds : odds-band / longshot gate on the *execution* price.
    min_prob/max_prob : gate on the model win probability itself.
    selector      : optional custom predicate; when given the gates are ignored.
    """

    name: str
    min_ev: Optional[float] = None
    min_edge_pct: Optional[float] = None
    min_abs_edge: Optional[float] = None
    min_odds: Optional[float] = None
    max_odds: Optional[float] = None
    min_prob: Optional[float] = None
    max_prob: Optional[float] = None
    selector: Optional[Callable[[pd.DataFrame], pd.Series]] = None

    def select(self, scored: pd.DataFrame) -> pd.Series:
        """Boolean Series (index-aligned to ``scored``) of runners to back.

        ``scored`` must carry ``prob`` (model win prob) and ``bet_price``
        (execution decimal odds). ``ev``/``edge`` are derived if absent.
        """
        if self.selector is not None:
            mask = self.selector(scored).astype(bool)
            return pd.Series(mask, index=scored.index)

        p = pd.to_numeric(scored["prob"], errors="coerce").to_numpy(dtype=float)
        d = pd.to_numeric(scored["bet_price"], errors="coerce").to_numpy(dtype=float)
        implied = metrics.implied_prob(d)
        ev = metrics.expected_value(p, d)

        # Start with "has a backable price"; NaN-safe comparisons stay False.
        keep = np.isfinite(d) & (d > 1.0) & np.isfinite(p)
        if self.min_ev is not None:
            keep &= ev >= self.min_ev
        if self.min_edge_pct is not None:
            keep &= p >= implied * (1.0 + self.min_edge_pct)
        if self.min_abs_edge is not None:
            keep &= (p - implied) >= self.min_abs_edge
        if self.min_odds is not None:
            keep &= d >= self.min_odds
        if self.max_odds is not None:
            keep &= d <= self.max_odds
        if self.min_prob is not None:
            keep &= p >= self.min_prob
        if self.max_prob is not None:
            keep &= p <= self.max_prob
        return pd.Series(np.where(np.isfinite(keep), keep, False).astype(bool),
                         index=scored.index)


@dataclass
class FlatStake:
    """Level staking: the same ``unit`` € on every selected bet."""

    unit: float = 10.0
    name: str = "flat"

    def stake(self, bankroll: float, prob: float, decimal_odds: float) -> float:
        # Never stake more than the bankroll on hand (matches the bankruptcy guard
        # in the engine); flat staking is otherwise bankroll-independent.
        return float(min(self.unit, max(bankroll, 0.0)))


@dataclass
class KellyStake:
    """Fractional-Kelly staking: ``fraction * full_kelly * bankroll``.

    ``full_kelly = (p*d - 1)/(d - 1)`` is bankroll-optimal but wildly volatile, so
    real bankrolls bet a fraction of it (quarter-Kelly = ``fraction=0.25`` is the
    common default). ``cap`` bounds any single stake to a fraction of the bankroll
    so a mis-estimated huge edge can't bet the farm.
    """

    fraction: float = 0.25
    cap: float = 0.05
    name: str = "kelly"

    def stake(self, bankroll: float, prob: float, decimal_odds: float) -> float:
        f_full = metrics.kelly_fraction(prob, decimal_odds)
        f_full = float(f_full) if np.isfinite(f_full) else 0.0
        f = min(self.fraction * f_full, self.cap)
        return float(max(f, 0.0) * max(bankroll, 0.0))


# A stake scheme is anything with ``.stake(bankroll, prob, decimal_odds) -> €``
# and a ``.name``. FlatStake / KellyStake are the built-ins; custom schemes just
# implement the same surface.
StakeScheme = object
