"""Recommended PAPER stake for a runner, sized by fractional Kelly.

This answers "how much would you put on it?" with the same arithmetic the rest
of the project already uses (``utils.bet_tracker.kelly_stake``), clamped into a
configurable band — by default EUR 0.50 to EUR 800.

Three rules it will not bend, because bending them is how staking models lose
money rather than make it:

  1. **No edge, no bet.** Kelly is negative whenever the price is shorter than
     the model's probability justifies. That returns a flat zero here, not a
     token minimum stake. Most runners on most cards get zero, and that is the
     correct answer, not a gap to fill.
  2. **Calibrated probability only.** Callers must pass the normalized win
     probability (``won_prob_normalized``, AUC 0.78 / ECE 0.03). The raw
     ``won_prob`` saturates out-of-sample — mean 0.33 against a true 0.11 win
     rate — and staking on it would inflate every bet by roughly 3x. See
     ``ui._components.headline_win_prob``.
  3. **The ceiling is a clamp, never a target.** Reaching the top of the band
     requires full Kelly to independently ask for it. Nothing scales a stake up
     to look confident.

On the default settings (bankroll 1000, quarter-Kelly) the arithmetic maximum is
250 — full Kelly cannot exceed the whole bankroll, and a quarter of that is 250.
The 800 ceiling therefore only binds for a larger bankroll or a bolder
``kelly_fraction``. That is deliberate: the band is a guard rail, not a dial that
manufactures confidence.

PAPER ONLY. ``execution.paper_only`` is true while the model is rated NO-GO, and
nothing here is a real-money recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.config_loader import get_config

# Band thresholds on the *fraction of bankroll* full Kelly asks for. These label
# the recommendation; they never change the number.
#
# The ladder starts at "Token", not "No bet": every path that reaches it has
# already established a positive edge, so the smallest band still describes a
# bet. "No bet" is set explicitly on the zero-stake returns — labelling a live
# 1.00 stake "No bet" would contradict the number printed beside it.
_BANDS: tuple[tuple[float, str], ...] = (
    (0.00, "Token"),
    (0.03, "Small"),
    (0.08, "Moderate"),
    (0.15, "Strong"),
    (0.25, "Maximum"),
)


@dataclass(frozen=True)
class StakeAdvice:
    """A recommended paper stake and the reasoning behind it."""

    stake: float            # EUR, already clamped; 0.0 means "no bet"
    band: str               # human label, e.g. "Moderate"
    edge: float             # EV per unit staked; negative means the price is too short
    kelly_fraction: float   # fraction of bankroll full Kelly asks for
    capped: bool            # True when the band ceiling bound the stake
    reason: str             # one-line explanation for the UI tooltip

    @property
    def is_bet(self) -> bool:
        return self.stake > 0.0


def _cfg() -> dict:
    return get_config().get("staking", {}) or {}


def _band_for(kelly_f: float) -> str:
    label = _BANDS[0][1]
    for threshold, name in _BANDS:
        if kelly_f >= threshold:
            label = name
    return label


def recommend_stake(
    win_prob: Optional[float],
    odds_decimal: Optional[float],
    *,
    market_ev: Optional[float] = None,
    bankroll: Optional[float] = None,
    kelly_fraction: Optional[float] = None,
    min_stake: Optional[float] = None,
    max_stake: Optional[float] = None,
) -> StakeAdvice:
    """Size a paper stake for one runner.

    ``win_prob`` must be a calibrated probability — the value layer's
    ``value_win_prob`` where available, else the normalized headline. Never the
    raw ``won_prob``. ``odds_decimal`` is the best available price.

    ``market_ev`` is the project's own expected value for the selection
    (``expected_value``). When supplied it acts as a veto: a selection the value
    layer has already judged -EV gets no stake, whatever Kelly would say. That
    keeps the recommendation consistent with the EV shown beside it — a card
    reading "EV -11%" next to a stake would be the UI arguing with itself.

    Returns a zero-stake ``StakeAdvice`` whenever there is no edge or the inputs
    are unusable.
    """
    cfg = _cfg()
    bank = float(bankroll if bankroll is not None else cfg.get("bankroll", 1000.0))
    k_frac = float(
        kelly_fraction if kelly_fraction is not None else cfg.get("kelly_fraction", 0.25)
    )
    lo = float(min_stake if min_stake is not None else cfg.get("min_stake", 0.50))
    hi = float(max_stake if max_stake is not None else cfg.get("max_stake", 800.0))

    none_ = StakeAdvice(0.0, "No bet", 0.0, 0.0, False, "No price or probability available")
    if win_prob is None or odds_decimal is None:
        return none_
    try:
        p = float(win_prob)
        dec = float(odds_decimal)
    except (TypeError, ValueError):
        return none_
    if p != p or dec != dec:  # NaN
        return none_
    if not (0.0 < p < 1.0) or dec <= 1.0 or bank <= 0.0 or k_frac <= 0.0:
        return none_

    # The value layer's verdict vetoes the stake — see the docstring.
    if market_ev is not None:
        try:
            mev = float(market_ev)
        except (TypeError, ValueError):
            mev = None
        if mev is not None and mev == mev and mev <= 0.0:
            return StakeAdvice(
                0.0, "No bet", mev, 0.0, False,
                f"Value layer rates this {mev * 100:+.1f}% EV against the market",
            )

    b = dec - 1.0
    edge = b * p - (1.0 - p)          # EV per unit staked
    if edge <= 0.0:
        return StakeAdvice(
            0.0, "No bet", edge, 0.0, False,
            f"No edge: {dec:.2f} is shorter than the model's {p * 100:.1f}% justifies",
        )

    full_f = edge / b                  # full Kelly, as a fraction of bankroll
    raw = full_f * k_frac * bank
    stake = round(min(max(raw, lo), hi), 2)
    capped = raw > hi

    if raw < lo:
        reason = (
            f"Edge {edge * 100:.1f}% is real but tiny — rounded up to the "
            f"{lo:.2f} minimum"
        )
    elif capped:
        reason = f"Kelly asked for {raw:,.2f}; capped at the {hi:,.2f} ceiling"
    else:
        reason = (
            f"{k_frac:.2f} Kelly on a {edge * 100:.1f}% edge "
            f"({p * 100:.1f}% at {dec:.2f})"
        )

    return StakeAdvice(stake, _band_for(full_f), edge, full_f, capped, reason)
