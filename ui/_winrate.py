"""Realized win-rate formatting (Stage-4 audit, requirement 9).

A *realized* win rate is settled wins over settled bets — an observed frequency
with a denominator, a date window, a selection rule and sampling error. A
*predicted probability* is none of those things. Every UI surface showing a
realized rate must say the denominator and (where space allows) the 95%
interval, so a 100% rate over 3 bets can never masquerade as skill.

Streamlit-free so it unit-tests headless.
"""
from __future__ import annotations

import math
from typing import Optional


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion; (nan, nan) when n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = wins / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def realized_rate_value(wins: int, n: int) -> str:
    """Headline: ``"37% (74/200)"`` — the rate never travels without its
    denominator. ``"— (0/0)"`` when nothing has settled."""
    if n <= 0:
        return "— (0/0)"
    return f"{wins / n * 100:.0f}% ({wins}/{n})"


def realized_rate_sub(wins: int, n: int,
                      window_label: Optional[str] = None) -> str:
    """Support line: 95% CI plus the date window / selection rule when given."""
    if n <= 0:
        return "no settled bets yet"
    lo, hi = wilson_interval(wins, n)
    txt = f"95% CI {lo * 100:.0f}–{hi * 100:.0f}%"
    if window_label:
        txt += f" · {window_label}"
    return txt
