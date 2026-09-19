"""Honest strategy metrics, every interval race-clustered.

WHY this module exists
----------------------
Stage 4 (``reports/calibration_audit_20260727.md``) issued **MODEL NO-GO**, and
the specific way the earlier June "GO" stamps lied was *point estimates without
intervals*, computed over rows rather than races. Runners in the same race are
not independent draws — exactly one of them wins, they share the going, the
ground, the pace and the same market — so a row-level bootstrap treats ~8
correlated observations as 8 independent ones and shrinks every interval by
roughly ``sqrt(field_size)``. That is how a noise-sized edge acquires a
significance star.

So this module has one rule: **no interval is ever computed over rows.**
:func:`race_bootstrap_ci` resamples *races* with replacement and carries every
runner of a drawn race along with it. It is seeded (default
:data:`DEFAULT_SEED`) so a reported interval is reproducible, and every
strategy-level statistic that a report might publish goes through it.

Reading the numbers
-------------------
* ``n_bets`` / ``n_races`` / ``n_days`` come FIRST in :func:`compare_strategies`
  and in :meth:`StrategyMetrics.to_dict`. An ROI without its sample size is not
  a result.
* ``roi_ci``/``clv_ci``/``ae_ci``/``hit_rate_ci`` are ``None`` — never a
  fabricated ``0`` — when the ledger cannot support them (no ``closing_odds``
  column, no cluster column, an empty bet set). A missing interval means
  "unknown", which for this project means "not evidence".
* Nothing here decides whether to bet. It measures what a ledger did.

Void handling
-------------
A voided bet (non-runner, market void) had a stake returned and no exposure.
It is **counted in ``n_bets``** — it consumed a ticket slot and a decision —
but excluded from ``turnover``, ``total_return``, ``hit_rate``, the A/E ratio,
the calibration error, CLV and the losing-streak count, because there is no
settled outcome to attribute to the model. Counting a void as a loss would
understate the model; counting it as a win or folding its stake into turnover
would flatter ROI. Excluding it from every settled statistic while keeping it
visible in the sample count is the only reading that does neither.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from backtest import metrics as bt_metrics
from utils.logger import get_logger

logger = get_logger(__name__)

# The Stage-5 selection seed (config: execution.selection.seed). Hard-coded as a
# default so an interval computed ad hoc in a notebook matches the one a report
# publishes.
DEFAULT_SEED = 20260727
DEFAULT_N_BOOT = 1000
DEFAULT_LEVEL = 0.95

# Column the ledger contract requires for race clustering. Without it no honest
# interval can be produced, so every CI degrades to None rather than to a
# row-level (too narrow) one.
DEFAULT_CLUSTER_COL = "race_uid"

# Fixed, documented column order for compare_strategies(). Sample size first.
COMPARISON_COLUMNS: tuple[str, ...] = (
    "n_bets",
    "n_races",
    "n_days",
    "turnover",
    "total_return",
    "profit",
    "roi",
    "yield_pct",
    "roi_ci_lower",
    "roi_ci_upper",
    "roi_ci_excludes_zero",
    "hit_rate",
    "hit_rate_ci_lower",
    "hit_rate_ci_upper",
    "mean_clv_log",
    "clv_ci_lower",
    "clv_ci_upper",
    "clv_ci_excludes_zero",
    "mean_clv_pct",
    "beat_close_rate",
    "ae_ratio",
    "ae_ci_lower",
    "ae_ci_upper",
    "expected_wins",
    "actual_wins",
    "max_drawdown",
    "max_drawdown_pct",
    "longest_losing_streak",
    "calibration_ece",
)


# ── intervals ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Interval:
    """A confidence interval. ``lower``/``upper`` may be NaN when undeterminable."""

    lower: float
    upper: float
    level: float = DEFAULT_LEVEL

    def excludes_zero(self) -> bool:
        """True only when the whole interval sits strictly on one side of zero.

        A NaN bound is "unknown", and unknown never counts as significant.
        """
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
            return False
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def width(self) -> float:
        """``upper - lower`` (NaN when either bound is unknown)."""
        return float(self.upper - self.lower)

    def to_dict(self) -> dict:
        return {
            "lower": _jsonable(self.lower),
            "upper": _jsonable(self.upper),
            "level": float(self.level),
            "excludes_zero": self.excludes_zero(),
        }


def _jsonable(value: Any) -> Any:
    """NaN/inf → None so a report's JSON stays valid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return f if np.isfinite(f) else None


def _known(interval: Optional[Interval]) -> Optional[Interval]:
    """None for an interval with unknown bounds, so callers never publish NaN."""
    if interval is None:
        return None
    if not (np.isfinite(interval.lower) and np.isfinite(interval.upper)):
        return None
    return interval


# ── the clustered bootstrap ──────────────────────────────────────────────────


def _cluster_groups(clusters) -> list[np.ndarray]:
    """Row indices grouped by cluster label, one ndarray per distinct cluster.

    Rows whose cluster label is missing collapse into a single shared cluster
    rather than becoming singletons: that is the *wider*-interval choice, and an
    unlabelled row is exactly the case where we cannot prove independence.
    """
    series = pd.Series(np.asarray(clusters, dtype=object))
    codes = np.asarray(pd.factorize(series, use_na_sentinel=False)[0])
    if codes.size == 0:
        return []
    k = int(codes.max()) + 1
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.searchsorted(sorted_codes, np.arange(k), side="left")
    ends = np.searchsorted(sorted_codes, np.arange(k), side="right")
    return [order[s:e] for s, e in zip(starts, ends)]


def _bootstrap_statistics(
    groups: Sequence[np.ndarray],
    fn: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """``n_boot`` draws of ``fn`` over cluster-resampled row indices.

    Each draw picks ``len(groups)`` clusters with replacement and hands ``fn``
    every row of every drawn cluster — so a race either contributes all of its
    runners or none of them.
    """
    k = len(groups)
    if k == 0 or n_boot <= 0:
        return np.empty(0, dtype=float)
    rng = np.random.default_rng(int(seed))
    out = np.full(int(n_boot), np.nan, dtype=float)
    for b in range(int(n_boot)):
        picked = rng.integers(0, k, k)
        rows = np.concatenate([groups[i] for i in picked])
        try:
            out[b] = float(fn(rows))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            out[b] = np.nan
    return out


def _percentile_interval(stats: np.ndarray, level: float) -> Interval:
    finite = stats[np.isfinite(stats)] if stats.size else stats
    if finite.size == 0:
        return Interval(float("nan"), float("nan"), float(level))
    alpha = 1.0 - float(level)
    lo, hi = np.percentile(finite, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return Interval(float(lo), float(hi), float(level))


def race_bootstrap_ci(
    values,
    clusters,
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """Race-clustered percentile bootstrap interval for ``statistic(values)``.

    ``clusters`` must be parallel to ``values`` and carry the race identity
    (``race_uid``). Resampling is over *clusters*, never rows — see the module
    docstring for why that is the whole point. Passing ``np.arange(len(values))``
    as ``clusters`` reproduces the naive row bootstrap and is only ever useful
    for demonstrating how much narrower (i.e. how much more flattering) it is.

    Deterministic for a given ``seed``. Non-finite values are dropped first.
    Returns an :class:`Interval` whose bounds are NaN when nothing can be
    computed; callers publishing a number should treat that as "unknown".
    """
    v = np.asarray(pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce"),
                   dtype=float)
    c = np.asarray(clusters, dtype=object).ravel()
    if v.size != c.size:
        raise ValueError(f"values ({v.size}) and clusters ({c.size}) differ in length")

    keep = np.isfinite(v)
    if not keep.any():
        return Interval(float("nan"), float("nan"), float(level))
    v, c = v[keep], c[keep]

    groups = _cluster_groups(c)
    stats = _bootstrap_statistics(
        groups, lambda rows: statistic(v[rows]), n_boot=n_boot, seed=seed
    )
    return _percentile_interval(stats, level)


def _ratio_ci(
    numerator,
    denominator,
    clusters,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> Interval:
    """Clustered bootstrap of ``sum(numerator) / sum(denominator)``.

    ROI (profit / turnover) and A/E (wins / expected wins) are *ratios of sums*,
    not means of per-row ratios; bootstrapping the per-row ratio would weight a
    €0.50 bet the same as a €50 one.
    """
    num = np.asarray(pd.to_numeric(pd.Series(numerator), errors="coerce"), dtype=float)
    den = np.asarray(pd.to_numeric(pd.Series(denominator), errors="coerce"), dtype=float)
    c = np.asarray(clusters, dtype=object).ravel()
    if not (num.size == den.size == c.size):
        raise ValueError("numerator, denominator and clusters must be parallel")

    keep = np.isfinite(num) & np.isfinite(den)
    if not keep.any():
        return Interval(float("nan"), float("nan"), float(level))
    num, den, c = num[keep], den[keep], c[keep]

    def _stat(rows: np.ndarray) -> float:
        total = float(den[rows].sum())
        return float(num[rows].sum() / total) if total > 0 else float("nan")

    groups = _cluster_groups(c)
    return _percentile_interval(
        _bootstrap_statistics(groups, _stat, n_boot=n_boot, seed=seed), level
    )


# ── calibration ──────────────────────────────────────────────────────────────


def expected_calibration_error(prob, won, *, bins: int = 10) -> float:
    """Equal-width-bin ECE: ``sum_b (n_b/N) * |mean_pred_b - mean_actual_b|``.

    0 is perfect calibration. Rows with a non-finite probability or outcome are
    dropped; an empty input returns NaN (unknown), never 0 (perfect).
    """
    p = np.asarray(pd.to_numeric(pd.Series(prob), errors="coerce"), dtype=float)
    y = np.asarray(pd.to_numeric(pd.Series(won), errors="coerce"), dtype=float)
    if p.size != y.size:
        raise ValueError("prob and won must be the same length")
    keep = np.isfinite(p) & np.isfinite(y)
    p, y = p[keep], y[keep]
    n = p.size
    if n == 0 or bins < 1:
        return float("nan")

    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    # np.digitize with right=False puts p == 1.0 in an overflow bin; clamp it back.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, int(bins) - 1)
    ece = 0.0
    for b in range(int(bins)):
        mask = idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        ece += (n_b / n) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(ece)


# ── strategy metrics ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StrategyMetrics:
    """Everything a report may say about one strategy's ledger.

    Sample size leads deliberately. ``*_ci`` fields are ``None`` when the ledger
    cannot support an honest interval.
    """

    name: str
    n_bets: int
    n_races: int
    n_days: int
    turnover: float
    total_return: float
    profit: float
    roi: float
    yield_pct: float
    roi_ci: Optional[Interval]
    hit_rate: float
    hit_rate_ci: Optional[Interval]
    mean_clv_log: Optional[float]
    clv_ci: Optional[Interval]
    mean_clv_pct: Optional[float]
    beat_close_rate: Optional[float]
    ae_ratio: Optional[float]
    ae_ci: Optional[Interval]
    expected_wins: float
    actual_wins: float
    max_drawdown: float
    max_drawdown_pct: float
    longest_losing_streak: int
    calibration_ece: Optional[float]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            # ── sample size first, always ──
            "n_bets": int(self.n_bets),
            "n_races": int(self.n_races),
            "n_days": int(self.n_days),
            "turnover": _jsonable(self.turnover),
            "total_return": _jsonable(self.total_return),
            "profit": _jsonable(self.profit),
            "roi": _jsonable(self.roi),
            "yield_pct": _jsonable(self.yield_pct),
            "roi_ci": self.roi_ci.to_dict() if self.roi_ci else None,
            "hit_rate": _jsonable(self.hit_rate),
            "hit_rate_ci": self.hit_rate_ci.to_dict() if self.hit_rate_ci else None,
            "mean_clv_log": _jsonable(self.mean_clv_log),
            "clv_ci": self.clv_ci.to_dict() if self.clv_ci else None,
            "mean_clv_pct": _jsonable(self.mean_clv_pct),
            "beat_close_rate": _jsonable(self.beat_close_rate),
            "ae_ratio": _jsonable(self.ae_ratio),
            "ae_ci": self.ae_ci.to_dict() if self.ae_ci else None,
            "expected_wins": _jsonable(self.expected_wins),
            "actual_wins": _jsonable(self.actual_wins),
            "max_drawdown": _jsonable(self.max_drawdown),
            "max_drawdown_pct": _jsonable(self.max_drawdown_pct),
            "longest_losing_streak": int(self.longest_losing_streak),
            "calibration_ece": _jsonable(self.calibration_ece),
        }


def _empty_metrics(name: str) -> StrategyMetrics:
    """The zero-bet result. Every rate/interval is None ("unknown"), not 0."""
    return StrategyMetrics(
        name=name, n_bets=0, n_races=0, n_days=0,
        turnover=0.0, total_return=0.0, profit=0.0, roi=0.0, yield_pct=0.0,
        roi_ci=None, hit_rate=0.0, hit_rate_ci=None,
        mean_clv_log=None, clv_ci=None, mean_clv_pct=None, beat_close_rate=None,
        ae_ratio=None, ae_ci=None, expected_wins=0.0, actual_wins=0.0,
        max_drawdown=0.0, max_drawdown_pct=0.0, longest_losing_streak=0,
        calibration_ece=None,
    )


def _num(frame: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if col not in frame.columns:
        return None
    return np.asarray(pd.to_numeric(frame[col], errors="coerce"), dtype=float)


def _longest_losing_streak(won: np.ndarray) -> int:
    """Longest run of consecutive settled losers, in the order supplied."""
    longest = current = 0
    for w in won:
        if np.isfinite(w) and w >= 0.5:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return int(longest)


def evaluate_ledger(
    ledger: pd.DataFrame,
    *,
    name: str = "strategy",
    initial_bankroll: float = 1000.0,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    cluster_col: str = DEFAULT_CLUSTER_COL,
) -> StrategyMetrics:
    """Score one strategy's settled-bet ledger.

    Consumed columns
    ----------------
    Required: ``stake``, ``decimal_odds`` (the price *actually taken*, never the
    close), ``model_prob``, ``won`` (0/1).
    Clustering: ``race_uid`` (``cluster_col``). **Absent ⇒ every interval is
    ``None``** — a row-level interval would be narrower than the truth, and a
    too-narrow interval is worse than no interval.
    Optional: ``race_date`` (n_days, chronological ordering), ``horse_key``,
    ``returns``, ``profit``, ``closing_odds`` (CLV; absent ⇒ CLV metrics are
    ``None``, never 0), ``voided`` (bool).

    ``profit`` is used as given; otherwise it is derived from ``returns - stake``
    and, failing that, from :func:`backtest.metrics.settle` at zero commission —
    commission is a friction and belongs to ``execution/frictions.py``, applying
    it here would double-count it.

    Ordering: rows are sorted by ``race_date`` (stable) before the equity curve
    and the losing streak are computed, so a ledger assembled per-strategy rather
    than chronologically still yields the drawdown a bettor would have lived
    through. See the module docstring for void handling.
    """
    if ledger is None or len(ledger) == 0:
        return _empty_metrics(name)

    df = ledger.copy()
    if "race_date" in df.columns:
        df = df.assign(_order=pd.to_datetime(df["race_date"], errors="coerce"))
        df = df.sort_values("_order", kind="stable").drop(columns="_order")
    df = df.reset_index(drop=True)

    stake = _num(df, "stake")
    if stake is None:
        raise ValueError("ledger must carry a 'stake' column")
    odds = _num(df, "decimal_odds")
    if odds is None:
        raise ValueError("ledger must carry a 'decimal_odds' column")
    prob = _num(df, "model_prob")
    won = _num(df, "won")
    if won is None:
        raise ValueError("ledger must carry a 'won' column")
    won = np.where(np.isfinite(won), won, 0.0)

    if "voided" in df.columns:
        voided = pd.Series(df["voided"]).astype("boolean").fillna(False).to_numpy(dtype=bool)
    else:
        voided = np.zeros(len(df), dtype=bool)
    settled = ~voided

    # ── money ────────────────────────────────────────────────────────────────
    profit = _num(df, "profit")
    returns = _num(df, "returns")
    if profit is None:
        if returns is not None:
            profit = returns - stake
        else:
            profit = np.atleast_1d(bt_metrics.settle(stake, odds, won)).astype(float)
    profit = np.where(np.isfinite(profit), profit, 0.0)
    # A void returns the stake: zero exposure, zero P&L, whatever the ledger says.
    profit = np.where(voided, 0.0, profit)
    if returns is None:
        returns = np.where(voided, stake, stake + profit)
    returns = np.where(np.isfinite(returns), returns, 0.0)

    turnover = float(np.nansum(np.where(settled, stake, 0.0)))
    total_return = float(np.nansum(np.where(settled, returns, 0.0)))
    total_profit = float(np.nansum(profit))
    roi = float(total_profit / turnover) if turnover > 0 else 0.0

    # ── sample size ──────────────────────────────────────────────────────────
    n_bets = int(len(df))
    n_races = int(pd.Series(df[cluster_col]).nunique()) if cluster_col in df.columns else 0
    if "race_date" in df.columns:
        days = pd.to_datetime(df["race_date"], errors="coerce")
        n_days = int(days.dt.normalize().nunique())
    else:
        n_days = 0

    # ── clustering: no cluster column ⇒ no intervals at all ──────────────────
    if cluster_col in df.columns:
        clusters = np.asarray(df[cluster_col].astype(object))
    else:
        clusters = None
        logger.warning(
            "evaluate_ledger(%s): no %r column — every confidence interval is "
            "suppressed rather than computed row-level (which would be too narrow)",
            name, cluster_col,
        )

    def _ci_mean(values: np.ndarray, mask: np.ndarray) -> Optional[Interval]:
        if clusters is None or not mask.any():
            return None
        return _known(
            race_bootstrap_ci(values[mask], clusters[mask],
                              statistic=np.mean, n_boot=n_boot, seed=seed)
        )

    def _ci_ratio(num: np.ndarray, den: np.ndarray, mask: np.ndarray) -> Optional[Interval]:
        if clusters is None or not mask.any():
            return None
        return _known(
            _ratio_ci(num[mask], den[mask], clusters[mask], n_boot=n_boot, seed=seed)
        )

    # ── settled-only statistics ──────────────────────────────────────────────
    n_settled = int(settled.sum())
    hit_rate = float(np.nanmean(won[settled])) if n_settled else 0.0
    hit_rate_ci = _ci_mean(won, settled)
    roi_ci = _ci_ratio(profit, stake, settled)

    if prob is not None and n_settled:
        expected_wins = float(np.nansum(prob[settled]))
        actual_wins = float(np.nansum(won[settled]))
        ae_ratio = float(actual_wins / expected_wins) if expected_wins > 0 else None
        ae_ci = _ci_ratio(won, prob, settled) if expected_wins > 0 else None
        calibration_ece = expected_calibration_error(prob[settled], won[settled])
        if not np.isfinite(calibration_ece):
            calibration_ece = None
    else:
        expected_wins = 0.0
        actual_wins = float(np.nansum(won[settled])) if n_settled else 0.0
        ae_ratio = None
        ae_ci = None
        calibration_ece = None

    # ── closing-line value (optional column ⇒ None, never 0) ─────────────────
    close = _num(df, "closing_odds")
    mean_clv_log = clv_ci = mean_clv_pct = beat_close_rate = None
    if close is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            clv_log = np.log(np.where((odds > 1.0) & (close > 1.0), odds / close, np.nan))
        clv_log = np.where(np.isfinite(clv_log), clv_log, np.nan)
        clv_mask = settled & np.isfinite(clv_log)
        if clv_mask.any():
            mean_clv_log = float(np.nanmean(clv_log[clv_mask]))
            clv_ci = _ci_mean(clv_log, clv_mask)
            pct = np.atleast_1d(bt_metrics.clv_pct(odds, close)).astype(float)
            mean_clv_pct = float(np.nanmean(pct[clv_mask]))
            beat_close_rate = float(np.mean(odds[clv_mask] > close[clv_mask]))

    # ── drawdown & streak (chronological, settled bets only) ─────────────────
    settled_profit = profit[settled]
    if settled_profit.size:
        equity = float(initial_bankroll) + np.cumsum(settled_profit)
        curve = np.concatenate([[float(initial_bankroll)], equity])
        peak = np.maximum.accumulate(curve)
        max_dd = float(np.max(peak - curve))
        max_dd_pct = float(bt_metrics.max_drawdown(curve))
    else:
        max_dd = 0.0
        max_dd_pct = 0.0
    longest_losing_streak = _longest_losing_streak(won[settled])

    return StrategyMetrics(
        name=name,
        n_bets=n_bets,
        n_races=n_races,
        n_days=n_days,
        turnover=turnover,
        total_return=total_return,
        profit=total_profit,
        roi=roi,
        yield_pct=roi * 100.0,
        roi_ci=roi_ci,
        hit_rate=hit_rate,
        hit_rate_ci=hit_rate_ci,
        mean_clv_log=mean_clv_log,
        clv_ci=clv_ci,
        mean_clv_pct=mean_clv_pct,
        beat_close_rate=beat_close_rate,
        ae_ratio=ae_ratio,
        ae_ci=ae_ci,
        expected_wins=expected_wins,
        actual_wins=actual_wins,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        longest_losing_streak=longest_losing_streak,
        calibration_ece=calibration_ece,
    )


def compare_strategies(metrics: Mapping[str, StrategyMetrics]) -> pd.DataFrame:
    """Tidy one-row-per-strategy scorecard in :data:`COMPARISON_COLUMNS` order.

    The strategy name is the (named ``"strategy"``) index and the FIRST column is
    ``n_bets``: a reader scanning left to right meets the sample size before the
    ROI, which is the only ordering that makes a 12-bet 40% ROI read correctly.

    Interval fields are flattened to ``*_ci_lower`` / ``*_ci_upper`` (plus
    ``*_ci_excludes_zero`` for ROI and CLV, the two that carry a GO/NO-GO
    reading), with ``None`` where the interval could not be computed.
    """
    rows: list[dict] = []
    index: list[str] = []
    for key, m in (metrics or {}).items():
        index.append(str(key))
        roi_ci, hit_ci, clv_ci, ae_ci = m.roi_ci, m.hit_rate_ci, m.clv_ci, m.ae_ci
        rows.append({
            "n_bets": int(m.n_bets),
            "n_races": int(m.n_races),
            "n_days": int(m.n_days),
            "turnover": _jsonable(m.turnover),
            "total_return": _jsonable(m.total_return),
            "profit": _jsonable(m.profit),
            "roi": _jsonable(m.roi),
            "yield_pct": _jsonable(m.yield_pct),
            "roi_ci_lower": _jsonable(roi_ci.lower) if roi_ci else None,
            "roi_ci_upper": _jsonable(roi_ci.upper) if roi_ci else None,
            "roi_ci_excludes_zero": roi_ci.excludes_zero() if roi_ci else None,
            "hit_rate": _jsonable(m.hit_rate),
            "hit_rate_ci_lower": _jsonable(hit_ci.lower) if hit_ci else None,
            "hit_rate_ci_upper": _jsonable(hit_ci.upper) if hit_ci else None,
            "mean_clv_log": _jsonable(m.mean_clv_log),
            "clv_ci_lower": _jsonable(clv_ci.lower) if clv_ci else None,
            "clv_ci_upper": _jsonable(clv_ci.upper) if clv_ci else None,
            "clv_ci_excludes_zero": clv_ci.excludes_zero() if clv_ci else None,
            "mean_clv_pct": _jsonable(m.mean_clv_pct),
            "beat_close_rate": _jsonable(m.beat_close_rate),
            "ae_ratio": _jsonable(m.ae_ratio),
            "ae_ci_lower": _jsonable(ae_ci.lower) if ae_ci else None,
            "ae_ci_upper": _jsonable(ae_ci.upper) if ae_ci else None,
            "expected_wins": _jsonable(m.expected_wins),
            "actual_wins": _jsonable(m.actual_wins),
            "max_drawdown": _jsonable(m.max_drawdown),
            "max_drawdown_pct": _jsonable(m.max_drawdown_pct),
            "longest_losing_streak": int(m.longest_losing_streak),
            "calibration_ece": _jsonable(m.calibration_ece),
        })

    frame = pd.DataFrame(rows, columns=list(COMPARISON_COLUMNS))
    frame.index = pd.Index(index, name="strategy")
    return frame
