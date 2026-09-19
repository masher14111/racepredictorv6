"""Betting maths + backtest summary metrics.

Pure, NaN-safe functions used by the staking layer and the engine. Nothing here
reads config or touches disk, so every formula is unit-testable in isolation.

Conventions
-----------
* ``decimal_odds`` (``d``) is the European/Betfair price: a €1 winning back-bet
  returns €``d`` gross (stake + €``d``-1 profit). Valid prices are ``> 1``.
* ``prob`` (``p``) is the model's *win* probability for the runner.
* All money figures are in stake units (the engine works in € but the maths is
  unit-agnostic).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# A decimal price must clear this to be a real, backable price. 1.0 = "evens on
# the certainty" i.e. no profit; the historical feeds emit exactly 1.0 for an
# absent/invalid price, so we treat <= 1 as missing everywhere.
_MIN_VALID_ODDS = 1.0 + 1e-9


def _as_float(x) -> np.ndarray:
    """Coerce a scalar/array/Series to a writable 1-D float ndarray (errors -> NaN).

    ``np.atleast_1d`` so scalar inputs (the per-bet settle path) coerce cleanly;
    ``np.array(..., float)`` always yields a writable copy.
    """
    arr = np.atleast_1d(np.asarray(x, dtype="object"))
    return np.array(pd.to_numeric(pd.Series(arr), errors="coerce"), dtype=float)


def _is_scalar(x) -> bool:
    """True for a Python/NumPy scalar (so scalar inputs get scalar outputs)."""
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return False
    return np.ndim(x) == 0


def _squeeze(result: np.ndarray, *inputs):
    """Return a float when every input was scalar, else the array unchanged."""
    if all(_is_scalar(x) for x in inputs):
        return float(result.reshape(-1)[0])
    return result


def _valid_odds(decimal_odds) -> np.ndarray:
    """Decimal odds as float, with anything <= 1 (or non-numeric) set to NaN."""
    d = _as_float(decimal_odds)
    d[~(d > _MIN_VALID_ODDS)] = np.nan
    return d


def implied_prob(decimal_odds):
    """Market-implied win probability ``1 / d``. NaN for an invalid price.

    This is the *raw* (over-round-inflated) implied probability of a single
    runner. For a coherent field the runner-implied probabilities sum to > 1; see
    :func:`devig` to strip the book's margin when a real race grouping exists.
    """
    d = _valid_odds(decimal_odds)
    return _squeeze(1.0 / d, decimal_odds)


def expected_value(prob, decimal_odds):
    """EV per unit staked on a back bet: ``p * d - 1``.

    +0.05 means the bet returns 5% above stake *in expectation* at this price.
    NaN when the price is invalid.
    """
    p = _as_float(prob)
    d = _valid_odds(decimal_odds)
    return _squeeze(p * d - 1.0, prob, decimal_odds)


def edge(prob, decimal_odds):
    """Probability edge over the market: ``p - 1/d``.

    The model thinks the runner is this many percentage points more likely to
    win than the price implies. NaN when the price is invalid.
    """
    p = _as_float(prob)
    d = _valid_odds(decimal_odds)
    return _squeeze(p - 1.0 / d, prob, decimal_odds)


def kelly_fraction(prob, decimal_odds):
    """Full-Kelly stake as a fraction of bankroll: ``(p*d - 1) / (d - 1)``.

    Equivalent to ``EV / (d - 1)`` (the numerator *is* ``expected_value``).
    Floored at 0 — a non-positive Kelly means "no bet". NaN when the price is
    invalid; callers treat NaN as 0.
    """
    p = _as_float(prob)
    d = _valid_odds(decimal_odds)
    f = (p * d - 1.0) / (d - 1.0)
    f = np.where(np.isfinite(f), np.clip(f, 0.0, None), np.nan)
    return _squeeze(f, prob, decimal_odds)


def devig(implied_probs):
    """Strip the book/market over-round from a single race's runner-implied probs.

    Proportional (a.k.a. "basic"/Shin-free) de-vig: ``fair_i = q_i / sum(q)``.
    Returns ``(fair_probs ndarray, overround float)`` where ``overround =
    sum(q) - 1`` (0 for a perfectly fair market, positive for a real one).

    Only meaningful over the runners of ONE race; with a venue-day-collapsed key
    (see the audit's C3) the sum spans several races and the result is not a fair
    probability — the engine therefore leaves de-vig OFF unless a real race key is
    present and reports raw implied probabilities, which is the conservative
    choice for a backer (it never *under*-states the market's confidence).
    """
    q = _as_float(implied_probs)
    total = float(np.nansum(q))
    if total <= 0:
        return q, float("nan")
    return q / total, total - 1.0


def settle(stake, decimal_odds, won, commission: float = 0.0):
    """Net profit of a back bet. Vectorised.

    Win  -> ``stake * (d - 1) * (1 - commission)``  (commission on net winnings,
            as Betfair charges).
    Lose -> ``-stake``.

    ``won`` is 1/0 (or bool). NaN price or NaN stake -> 0 profit (no bet placed).
    """
    s = _as_float(stake)
    d = _valid_odds(decimal_odds)
    w = _as_float(won)
    gross_win = s * (d - 1.0) * (1.0 - float(commission))
    profit = np.where(w >= 0.5, gross_win, -s)
    # No valid price or no stake => no bet => no P&L.
    profit = np.where(np.isfinite(d) & np.isfinite(s) & (s > 0), profit, 0.0)
    return _squeeze(profit, stake, decimal_odds, won)


def clv_pct(bet_price, close_price):
    """Closing-line value as a fraction: ``d_bet / d_close - 1``.

    Positive means you took a bigger price than the market closed at (you beat
    the close) — the single best long-run indicator that a bet was +EV. NaN when
    either price is invalid.
    """
    db = _valid_odds(bet_price)
    dc = _valid_odds(close_price)
    return _squeeze(db / dc - 1.0, bet_price, close_price)


def max_drawdown(equity) -> float:
    """Largest peak-to-trough fractional decline of an equity/bankroll curve.

    Returns a non-negative fraction (0.20 == a 20% drawdown). Empty/degenerate
    curves return 0.0.
    """
    e = _as_float(equity)
    e = e[np.isfinite(e)]
    if e.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(e)
    # Guard against a non-positive peak (bankroll wiped out): clamp denominator.
    safe_peak = np.where(running_peak > 0, running_peak, np.nan)
    dd = (running_peak - e) / safe_peak
    dd = dd[np.isfinite(dd)]
    return float(dd.max()) if dd.size else 0.0


def _sharpe(returns) -> float | None:
    """Mean/standard-deviation ratio of a return series (sample std, ddof=1)."""
    r = _as_float(returns)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return None
    sd = float(r.std(ddof=1))
    # Treat a vanishing spread as "no signal". An exact ``== 0`` test is unsafe:
    # identical returns can yield a ~1e-17 std from float rounding, which would
    # otherwise explode the ratio. Scale the floor to the data's magnitude.
    scale = float(np.max(np.abs(r))) if r.size else 0.0
    if sd <= 1e-12 * max(scale, 1.0):
        return None
    return float(r.mean() / sd)


# Default predicted-probability buckets for A/E (win base-rate ~12%, so the
# action is in the low-prob region — fine-grained there, coarse in the tail).
_PROB_BINS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
# Default decimal-odds bands (favourites -> longshots).
_ODDS_BANDS = [
    ("odds-on (<2.0)", 1.0, 2.0),
    ("2.0-4.0", 2.0, 4.0),
    ("4.0-8.0", 4.0, 8.0),
    ("8.0-16.0", 8.0, 16.0),
    ("16.0-34.0", 16.0, 34.0),
    ("longshot (>34)", 34.0, float("inf")),
]


def ae_table(prob, won, *, group_values=None, edges=None, bands=None):
    """Actual/Expected calibration table.

    For each bucket: ``expected = sum(prob)``, ``actual = sum(won)``,
    ``ae = actual / expected``. A/E ≈ 1 means the win probabilities are honest in
    that bucket; A/E < 1 = over-confident, > 1 = under-confident.

    Bucketing is by ``prob`` deciles (``edges`` given) OR by ``group_values``
    falling into ``bands`` (list of ``(label, lo, hi)`` with ``lo <= v < hi``).
    Returns a list of per-bucket dicts (empty buckets dropped).
    """
    p = _as_float(prob)
    y = _as_float(won)
    rows = []

    if bands is not None:
        v = _as_float(group_values)
        for label, lo, hi in bands:
            mask = np.isfinite(v) & (v >= lo) & (v < hi) & np.isfinite(p) & np.isfinite(y)
            rows.append(_ae_row(label, p[mask], y[mask]))
    else:
        e = list(edges) if edges is not None else _PROB_BINS
        for lo, hi in zip(e[:-1], e[1:]):
            # include the right edge only in the final bucket so 1.0 lands
            upper_ok = (p <= hi) if hi == e[-1] else (p < hi)
            mask = np.isfinite(p) & np.isfinite(y) & (p >= lo) & upper_ok
            rows.append(_ae_row(f"{lo:.2f}-{hi:.2f}", p[mask], y[mask]))

    return [r for r in rows if r["n"] > 0]


def _ae_row(label: str, p: np.ndarray, y: np.ndarray) -> dict:
    n = int(p.size)
    expected = float(p.sum())
    actual = float(y.sum())
    ae = actual / expected if expected > 0 else None
    return {
        "bucket": label,
        "n": n,
        "expected": round(expected, 3),
        "actual": round(actual, 1),
        "mean_pred": round(float(p.mean()), 4) if n else None,
        "ae": round(ae, 3) if ae is not None else None,
    }


def summarize_bets(ledger: pd.DataFrame, initial_bankroll: float) -> dict:
    """Headline performance metrics for one strategy's settled-bet ledger.

    ``ledger`` must have columns: ``stake``, ``profit``, ``won``, ``bet_price``,
    ``close_price``, ``race_date`` and the running ``bankroll_after`` (post-bet
    bankroll, chronological). Returns a flat dict of scalars (JSON-friendly).
    """
    n = len(ledger)
    if n == 0:
        return {
            "n_bets": 0, "staked": 0.0, "returned": 0.0, "profit": 0.0,
            "yield_pct": None, "roi_pct": None, "hit_rate": None,
            "final_bankroll": round(float(initial_bankroll), 2),
            "bankroll_growth_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe_per_bet": None, "sharpe_daily": None,
            "clv_pct_mean": None, "beat_close_rate": None,
            "avg_odds": None, "avg_win_prob": None,
        }

    stake = _as_float(ledger["stake"])
    profit = _as_float(ledger["profit"])
    won = _as_float(ledger["won"])
    staked = float(np.nansum(stake))
    total_profit = float(np.nansum(profit))
    returned = staked + total_profit

    # Per-bet return on stake (for the per-bet Sharpe + a clean yield definition).
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bet_ret = np.where(stake > 0, profit / stake, np.nan)

    # Daily P&L on the *starting* bankroll → comparable across strategies.
    daily = (pd.DataFrame({"d": pd.to_datetime(ledger["race_date"]).values,
                           "p": profit})
             .groupby("d")["p"].sum())
    daily_ret = daily.to_numpy() / float(initial_bankroll) if initial_bankroll else daily.to_numpy()

    final_bankroll = float(_as_float(ledger["bankroll_after"])[-1]) if "bankroll_after" in ledger \
        else float(initial_bankroll) + total_profit
    equity = (np.concatenate([[initial_bankroll], _as_float(ledger["bankroll_after"])])
              if "bankroll_after" in ledger else
              float(initial_bankroll) + np.nancumsum(profit))

    cl = clv_pct(ledger["bet_price"], ledger["close_price"])
    beat = (_valid_odds(ledger["bet_price"]) > _valid_odds(ledger["close_price"]))
    yield_pct = (total_profit / staked) if staked > 0 else None

    return {
        "n_bets": int(n),
        "staked": round(staked, 2),
        "returned": round(returned, 2),
        "profit": round(total_profit, 2),
        # Yield == ROI on turnover (profit / amount staked); the two are
        # synonyms in betting. bankroll_growth captures compounding for Kelly.
        "yield_pct": round(100.0 * yield_pct, 3) if yield_pct is not None else None,
        "roi_pct": round(100.0 * yield_pct, 3) if yield_pct is not None else None,
        "hit_rate": round(float(np.nanmean(won)), 4),
        "final_bankroll": round(final_bankroll, 2),
        "bankroll_growth_pct": round(100.0 * (final_bankroll - initial_bankroll) / initial_bankroll, 3)
        if initial_bankroll else None,
        "max_drawdown_pct": round(100.0 * max_drawdown(equity), 3),
        "sharpe_per_bet": _round_opt(_sharpe(per_bet_ret)),
        "sharpe_daily": _round_opt(_sharpe(daily_ret)),
        "clv_pct_mean": _round_opt(float(np.nanmean(cl)) * 100.0 if np.isfinite(cl).any() else None),
        "beat_close_rate": round(float(np.nanmean(beat.astype(float))), 4),
        "avg_odds": round(float(np.nanmean(_valid_odds(ledger["bet_price"]))), 3),
        "avg_win_prob": round(float(np.nanmean(_as_float(ledger["prob"]))), 4)
        if "prob" in ledger else None,
    }


def _round_opt(v, nd: int = 4):
    return round(float(v), nd) if v is not None and np.isfinite(v) else None
