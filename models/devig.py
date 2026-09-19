"""
Market de-vigging — turn decimal bookmaker odds into a margin-free
probability vector that sums to 1.0 within each race.

This is the foundation of the value programme: racing_ingestion's whole edge is
that it scores the model against the **de-vigged market line**, not against race
outcomes. The de-vig is where that fair line is defined, so it lives in exactly
one place and everything downstream (Phase A, the UI) consumes it.

Three margin-removal methods, from naive to principled:

proportional
    p_i = (1/o_i) / Σ_j (1/o_j). The bookmaker margin is removed in proportion
    to each runner's implied probability. Cheap, parameter-free, the standard
    first baseline. It strips the *same fraction* of margin from every runner,
    leaving the favourite–longshot bias intact.

power
    p_i ∝ (1/o_i)^k, with the exponent k solved (Brent) so Σ_i p_i = 1. Raising
    sub-unit implied probabilities to a power k>1 shrinks longshots more than
    favourites, so this removes *more* margin from longshots — a one-parameter
    correction of the favourite–longshot bias.

shin  (Shin 1992/1993, insider-trading model)
    Assumes a fraction z of bettors hold inside information; the bookmaker shades
    prices to protect against them, taking a larger effective margin on longshots.
    Inverting the model recovers

        p_i = [ √(z² + 4(1−z)·π_i²/Π) − z ] / (2(1−z)),   π_i = 1/o_i, Π = Σ π_i

    with z solved by the fixed-point iteration

        z ← [ Σ_i √(z² + 4(1−z)·π_i²/Π) − 2 ] / (n − 2).

    Like power it lengthens longshots relative to proportional, but z is an
    interpretable quantity (the implied insider fraction) and the shape is
    derived rather than imposed. Falls back to ``power`` when the fixed point
    fails to converge or is ill-defined (n ≤ 2).

Pure functions, no I/O, no look-ahead: every function reads decimal odds only,
so the fair line can never peek at a finishing position. Decimal odds in,
probabilities out.

Partial books are never de-vigged. A race missing any runner's price is not a
complete book — its booksum is wrong, so margin removal is meaningless. The
public :func:`devig` flags every runner of such a race with NaN rather than
producing a misleading number from an incomplete set of odds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import brentq

logger = logging.getLogger(__name__)

# Decimal odds below this are treated as data errors and clipped. An honest
# decimal price is ≥ 1.0; 1.001 keeps 1/o strictly below 1 for the transforms.
_MIN_ODDS = 1.001
# Implied probs are clipped strictly below 1 so the power transform π**k is
# well-defined and Brent always brackets a sign change.
_MAX_IMPLIED = 1.0 - 1e-12

_SHIN_MAX_ITER = 1000
_SHIN_TOL = 1e-12
# z this close to 1 makes the 2(1−z) denominator blow up — treat as divergent.
_Z_CEILING = 1.0 - 1e-9


# ── Per-race input handling ───────────────────────────────────────────────────


def _as_odds(decimal_odds) -> np.ndarray:
    """Validate and clip a 1-D vector of decimal odds for ONE race."""
    odds = np.asarray(decimal_odds, dtype=np.float64).ravel()
    if odds.size == 0:
        raise ValueError("decimal_odds is empty")
    if not np.all(np.isfinite(odds)):
        raise ValueError("decimal_odds contains non-finite values")
    return np.clip(odds, _MIN_ODDS, None)


def implied_probs(decimal_odds) -> np.ndarray:
    """
    Raw implied (inverse) probabilities π_i = 1/o_i for one race.

    These are NOT normalised — they sum to the booksum (overround), which is
    > 1 for a real margin book. Use a ``de_vig_*`` function to obtain a
    margin-free probability vector that sums to 1.
    """
    return 1.0 / _as_odds(decimal_odds)


# ── Proportional ──────────────────────────────────────────────────────────────


def de_vig_proportional(decimal_odds) -> np.ndarray:
    """Strip the margin proportionally: p_i = (1/o_i) / Σ_j (1/o_j)."""
    pi = implied_probs(decimal_odds)
    total = pi.sum()
    if total <= 0:  # degenerate book — fall back to uniform
        return np.full(pi.size, 1.0 / pi.size)
    return pi / total


# ── Power ─────────────────────────────────────────────────────────────────────


def de_vig_power(decimal_odds, *, return_exponent: bool = False):
    """
    Power de-vig: p_i ∝ (1/o_i)^k with k solved so Σ_i p_i = 1.

    f(k) = Σ_i π_i^k − 1 is continuous and strictly decreasing in k (every
    π_i ∈ (0,1) after clipping), so a single Brent root on a wide bracket is
    robust and parameter-free.

    Args:
        return_exponent: also return the fitted exponent k.

    Returns:
        Probability vector (sums to 1), or ``(probs, k)`` if ``return_exponent``.
    """
    pi = implied_probs(decimal_odds)
    n = pi.size
    if n == 1:
        return (np.array([1.0]), 1.0) if return_exponent else np.array([1.0])

    pi = np.minimum(pi, _MAX_IMPLIED)  # guarantee π_i < 1 for the power transform

    def f(k: float) -> float:
        return float((pi**k).sum() - 1.0)

    # f(1e-6) ≈ n−1 > 0 (π^0→1); f(1e3) ≈ −1 < 0 (π<1 → π^k→0). Always brackets.
    try:
        k = float(brentq(f, 1e-6, 1e3, xtol=1e-14, maxiter=200))
    except (ValueError, RuntimeError):
        # Should not happen given the guaranteed bracket; degrade to proportional.
        logger.warning("de_vig_power: Brent failed; falling back to proportional")
        p = pi / pi.sum()
        return (p, 1.0) if return_exponent else p

    p = pi**k
    p = p / p.sum()
    return (p, k) if return_exponent else p


def power_exponent(decimal_odds) -> float:
    """Just the fitted power-de-vig exponent k for one race."""
    return de_vig_power(decimal_odds, return_exponent=True)[1]


# ── Shin (insider-trading fixed point) ────────────────────────────────────────


def de_vig_shin(decimal_odds, *, return_z: bool = False, max_iter: int = _SHIN_MAX_ITER):
    """
    Shin (1992/93) de-vig via the insider-fraction fixed point.

    Solves z from

        z ← [ Σ_i √(z² + 4(1−z)·π_i²/Π) − 2 ] / (n − 2)

    then backs out the margin-free probabilities. Includes a convergence guard:
    if the iteration does not converge, drives z to a degenerate value, or
    yields invalid probabilities — and whenever n ≤ 2 (z is not identified) —
    it falls back to :func:`de_vig_power`.

    Args:
        return_z: also return the fitted insider fraction z.

    Returns:
        Probability vector (sums to 1), or ``(probs, z)`` if ``return_z``.
    """
    pi = implied_probs(decimal_odds)
    n = pi.size

    if n == 1:
        return (np.array([1.0]), 0.0) if return_z else np.array([1.0])

    # n == 2: the (n−2) denominator vanishes and z is unidentified.
    if n == 2:
        return _shin_fallback(decimal_odds, return_z)

    Pi = pi.sum()
    if Pi <= 0:
        p = np.full(n, 1.0 / n)
        return (p, 0.0) if return_z else p

    c = pi**2 / Pi  # π_i² / Π, precomputed (z-independent)

    z = 0.0
    converged = False
    for _ in range(max_iter):
        s = np.sqrt(z * z + 4.0 * (1.0 - z) * c)
        z_new = (s.sum() - 2.0) / (n - 2.0)
        if not np.isfinite(z_new):
            break
        z_new = min(max(z_new, 0.0), _Z_CEILING)
        if abs(z_new - z) < _SHIN_TOL:
            z = z_new
            converged = True
            break
        z = z_new

    if not converged or z >= _Z_CEILING:
        logger.debug("de_vig_shin: fixed point did not converge (z=%.6g); using power", z)
        return _shin_fallback(decimal_odds, return_z)

    s = np.sqrt(z * z + 4.0 * (1.0 - z) * c)
    p = (s - z) / (2.0 * (1.0 - z))

    total = p.sum()
    if not np.all(np.isfinite(p)) or np.any(p < 0) or total <= 0:
        return _shin_fallback(decimal_odds, return_z)

    p = p / total
    return (p, float(z)) if return_z else p


def _shin_fallback(decimal_odds, return_z: bool):
    """Shin's documented fallback path: the power de-vig (z reported as NaN)."""
    p = de_vig_power(decimal_odds)
    return (p, float("nan")) if return_z else p


def shin_insider_fraction(decimal_odds) -> float:
    """The fitted Shin insider fraction z for one race (NaN if it fell back)."""
    return de_vig_shin(decimal_odds, return_z=True)[1]


# ── Dispatcher / registry ─────────────────────────────────────────────────────

# name → de-vig function taking ONE race's decimal odds, returning probs summing
# to 1. The public :func:`devig` below dispatches through this registry so the
# benchmark fair line is defined in exactly one place.
MARGIN_METHODS = {
    "proportional": de_vig_proportional,
    "power": de_vig_power,
    "shin": de_vig_shin,
}


# ── Public DataFrame-level API ────────────────────────────────────────────────


def devig(
    odds: pd.Series,
    race_ids: pd.Series,
    method: str = "proportional",
) -> np.ndarray:
    """
    De-vig a column of decimal odds, race by race, into win probabilities.

    Each race's probabilities sum to 1.0; the output is aligned row-for-row to
    the input ``odds``. This is the multi-race orchestrator over the per-race
    ``de_vig_*`` functions — it groups by ``race_ids``, de-vigs each group, and
    scatters the result back into input order.

    Partial books are never de-vigged. If any runner in a race has a missing
    (NaN) price, the booksum is incomplete and margin removal is meaningless, so
    **every** runner of that race is flagged with NaN in the output rather than
    given a misleading number. Complete races are unaffected.

    Args:
        odds:     Per-runner pre-off decimal odds. May contain NaN.
        race_ids: Race identifier per runner; same length/index as ``odds``.
        method:   ``"proportional"`` (default), ``"power"``, or ``"shin"``.

    Returns:
        ``np.ndarray`` of win probabilities aligned to ``odds``' rows. Rows
        belonging to an incomplete (NaN-containing) book are NaN.
    """
    if method not in MARGIN_METHODS:
        raise ValueError(
            f"unknown method '{method}'. Available: {sorted(MARGIN_METHODS)}"
        )
    if len(odds) != len(race_ids):
        raise ValueError(
            f"odds and race_ids length mismatch: {len(odds)} vs {len(race_ids)}"
        )

    devig_fn = MARGIN_METHODS[method]

    odds_arr = np.asarray(odds, dtype=np.float64).ravel()
    race_arr = np.asarray(race_ids).ravel()
    out = np.full(odds_arr.shape, np.nan, dtype=np.float64)

    if odds_arr.size == 0:
        return out

    # Positional grouping keeps NaN race ids together and is order-preserving
    # for the scatter-back. np.unique on object/str dtypes is fine here.
    for rid in pd.unique(race_arr):
        mask = race_arr == rid
        race_odds = odds_arr[mask]
        # Incomplete book → flag the whole race with NaN, never de-vig partial.
        if not np.all(np.isfinite(race_odds)):
            logger.debug(
                "devig: race %r has missing/non-finite odds; flagging as NaN", rid
            )
            continue
        out[mask] = devig_fn(race_odds)

    return out
