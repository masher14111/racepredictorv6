"""The forward-release gate (Stage-5 requirement 7).

Why this module exists
----------------------
Stage 4's MODEL NO-GO was not a close call: on 904 untouched races the price-free
line lost to the de-vigged pre-off market by 0.186 race log-loss (95% CI
[-0.225, -0.146]), CLV ran at -13.4%, and the PSI drift gate failed. Every
"positive" result that preceded it came from re-scored history. So the one thing
this gate must refuse to accept is more re-scored history.

Hence the first rule: **``evidence_kind`` must be ``'forward'``**. A backtest,
however long or however good it looks, fails immediately with a single
``forward_evidence`` criterion. Backtests are re-scored history; they cannot show
that the deployment tracked live markets in real time, took the price it claimed to
take, or survived a market that moved against it.

The remaining criteria are conjunctive — all must pass — and every one of them fails
closed. A metric that cannot be computed (missing column, empty ledger, degenerate
bootstrap) is *not* a pass; it is a fail with the reason recorded. An absent ledger
therefore fails every criterion rather than sailing through on vacuous truth.

Nothing here is a profitability claim. Clearing the gate records that the stated
thresholds held on forward evidence over the stated window, and that is all
``summary`` is allowed to say.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from backtest.metrics import clv_pct, max_drawdown
from utils.logger import get_logger

logger = get_logger(__name__)

# Column-name candidates, most authoritative first.
_PROB_COLS = ("prob", "model_prob", "win_prob", "calibrated_prob", "norm_prob")
_RACE_COLS = ("race_uid", "race_id")
_CLV_COLS = ("clv_pct", "clv", "clv_log")
_WON_COLS = ("won", "win", "settled_won")
_DATE_COLS = ("race_date", "date", "race_time")

_ECE_BINS = 10
_SECONDS_PER_WEEK = 7.0 * 24.0 * 3600.0

# The canonical metric names this module reasons about. ``metrics=`` may supply any
# subset; anything missing is computed from the ledger.
CANONICAL_METRICS = (
    "n_qualified_bets",
    "n_qualified_races",
    "weeks_elapsed",
    "mean_clv",
    "clv_ci95",
    "ae",
    "ece",
    "max_drawdown_pct",
)


# ── value objects ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CriterionResult:
    """One gate condition, with the observed value and what was required of it."""

    name: str
    passed: bool
    observed: float | str | None
    required: str
    detail: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "required": self.required,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class ForwardGateResult:
    """The conjunctive verdict over every forward-release criterion."""

    passed: bool
    criteria: tuple[CriterionResult, ...]
    failed: tuple[str, ...]
    weeks_elapsed: float
    n_qualified_bets: int
    n_qualified_races: int
    evidence_kind: str
    evaluated_at: str
    summary: str

    @property
    def state_label(self) -> str:
        total = len(self.criteria)
        if self.passed:
            return f"FORWARD GATE MET ({total} of {total} criteria passed)"
        return f"FORWARD GATE NOT MET ({len(self.failed)} of {total} criteria failed)"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "state_label": self.state_label,
            "criteria": [c.to_dict() for c in self.criteria],
            "failed": list(self.failed),
            "weeks_elapsed": self.weeks_elapsed,
            "n_qualified_bets": self.n_qualified_bets,
            "n_qualified_races": self.n_qualified_races,
            "evidence_kind": self.evidence_kind,
            "evaluated_at": self.evaluated_at,
            "summary": self.summary,
        }


# ── small helpers ─────────────────────────────────────────────────────────────
def _iso(now: Optional[datetime] = None) -> str:
    dt = now or datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat(timespec="seconds")


def _num(raw: Any) -> Optional[float]:
    try:
        if raw is None:
            return None
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _first_col(frame: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _floats(frame: pd.DataFrame, col: Optional[str]) -> np.ndarray:
    if col is None:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)


def _ci(raw: Any) -> Optional[tuple[float, float]]:
    if isinstance(raw, dict):
        lo, hi = _num(raw.get("lo", raw.get("lower"))), _num(raw.get("hi", raw.get("upper")))
        return (lo, hi) if lo is not None and hi is not None else None
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 2:
        lo, hi = _num(raw[0]), _num(raw[1])
        return (lo, hi) if lo is not None and hi is not None else None
    return None


# ── interop with execution.evaluation ─────────────────────────────────────────
# ``execution.evaluation`` owns the shared ledger maths (``evaluate_ledger`` /
# ``race_bootstrap_ci`` / ``expected_calibration_error``), and the report must not
# quote two different numbers for the same quantity. It is imported lazily because
# it is a sibling in this package and the report layer pulls in both.
#
# Its ledger schema is ``stake`` / ``decimal_odds`` / ``model_prob`` / ``won`` with
# ``race_uid`` clustering and optional ``closing_odds``; the forward paper ledger
# names some of those differently, so :data:`_SHARED_ALIASES` adapts them. Every
# call is guarded, and the local computations below take over if the shared module
# is unavailable — a gate that cannot be evaluated must still return a verdict, and
# that verdict is "not met".
_SHARED_ALIASES: dict[str, tuple[str, ...]] = {
    "stake": ("stake",),
    "decimal_odds": ("decimal_odds", "bet_price", "taken_price", "price"),
    "closing_odds": ("closing_odds", "close_price", "closing_price"),
    "model_prob": _PROB_COLS,
    "won": _WON_COLS,
    "race_uid": _RACE_COLS,
}


def _lazy_evaluation():
    try:
        from execution import evaluation as _evaluation

        return _evaluation
    except Exception as exc:  # pragma: no cover - only when the sibling is broken
        logger.warning("execution.evaluation unavailable (%s); computing locally", exc)
        return None


def _shared_frame(ledger: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``ledger`` with the shared module's column names filled in."""
    out = ledger.copy()
    for target, sources in _SHARED_ALIASES.items():
        if target in out.columns:
            continue
        for src in sources:
            if src in out.columns:
                out[target] = out[src]
                break
    return out


def _initial_bankroll(frame: pd.DataFrame) -> Optional[float]:
    """The bankroll the ledger actually started from, or ``None`` if unrecorded."""
    for col in ("bankroll_before", "bankroll_after"):
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").dropna()
            if len(values):
                return float(values.iloc[0])
    return None


def _shared_evaluate(ledger: pd.DataFrame, cfg) -> dict:
    """Score the forward ledger with the shared evaluator, mapped onto the canon.

    The ``qualified`` filter is applied *before* handing the frame over. The shared
    evaluator knows nothing about the column, so skipping this would score rejected
    and passed rows as though they were bets — inflating every count the gate reads
    and letting non-bets carry the CLV interval.
    """
    if ledger is None or len(ledger) == 0:
        return {}
    module = _lazy_evaluation()
    if module is None:
        return {}
    frame = _shared_frame(_qualified(ledger))
    if len(frame) == 0:
        return {}
    bankroll = _initial_bankroll(frame)
    try:
        metrics = module.evaluate_ledger(
            frame,
            name="forward",
            initial_bankroll=(bankroll if bankroll is not None else 1000.0),
            n_boot=int(cfg.selection.bootstrap_resamples),
            seed=int(cfg.selection.seed),
        )
    except Exception as exc:
        logger.warning(
            "evaluation.evaluate_ledger could not score the forward ledger (%s); "
            "computing locally",
            exc,
        )
        return {}

    n_bets = int(getattr(metrics, "n_bets", 0) or 0)
    if n_bets == 0:
        # An empty ledger scores 0.0 drawdown and 0 counts. Zero drawdown over zero
        # bets is vacuous, and a vacuous number must never satisfy a criterion.
        return {}

    ci = getattr(metrics, "clv_ci", None)
    clv_ci = None
    if ci is not None:
        lo, hi = _num(getattr(ci, "lower", None)), _num(getattr(ci, "upper", None))
        clv_ci = (lo, hi) if lo is not None and hi is not None else None

    out = {
        "n_qualified_bets": n_bets,
        # 0 means "no race_uid column", i.e. unknown — leave it to the local pass,
        # which reports 0 races rather than inventing one cluster per bet.
        "n_qualified_races": (int(metrics.n_races) or None),
        # mean_clv_pct is the fraction (bet/close - 1); clv_ci is on the log scale.
        # The gate only compares CLV against zero, where the two scales agree in
        # sign, so pairing them cannot flip a verdict.
        "mean_clv": _num(getattr(metrics, "mean_clv_pct", None)),
        "clv_ci95": clv_ci,
        "ae": _num(getattr(metrics, "ae_ratio", None)),
        "ece": _num(getattr(metrics, "calibration_ece", None)),
        # Only trust the drawdown when the ledger recorded the bankroll it ran on.
        # The shared evaluator falls back to an assumed €1000 start, and a large
        # assumed bankroll makes any drawdown look small — an unstated assumption
        # must not be what clears a ceiling.
        "max_drawdown_pct": (
            _num(getattr(metrics, "max_drawdown_pct", None))
            if bankroll is not None
            else None
        ),
    }
    return {k: v for k, v in out.items() if v is not None}


def _normalise_supplied(raw: Any) -> dict:
    """Map a caller-supplied ``metrics`` mapping onto :data:`CANONICAL_METRICS`."""
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    if not isinstance(raw, dict):
        return {}

    def pick(*names: str) -> Any:
        for name in names:
            if raw.get(name) is not None:
                return raw[name]
        return None

    drawdown = _num(pick("max_drawdown_pct", "max_drawdown"))
    if drawdown is not None and drawdown > 1.0:
        # A drawdown *fraction* cannot exceed 1.0, so >1 is unambiguously a
        # percentage. Anything <= 1.0 is read as a fraction, which is the stricter
        # of the two readings — ambiguity resolves against the gate, never for it.
        drawdown = drawdown / 100.0

    out = {
        "n_qualified_bets": pick("n_qualified_bets", "n_bets"),
        "n_qualified_races": pick("n_qualified_races", "n_races"),
        "weeks_elapsed": _num(pick("weeks_elapsed", "weeks")),
        "mean_clv": _num(pick("mean_clv", "mean_clv_pct", "clv_mean")),
        "clv_ci95": _ci(pick("clv_ci95", "clv_ci", "mean_clv_ci95")),
        "ae": _num(pick("ae", "ae_ratio", "actual_over_expected")),
        "ece": _num(pick("ece", "calibration_ece")),
        "max_drawdown_pct": drawdown,
    }
    return {k: v for k, v in out.items() if v is not None}


def _bootstrap_ci(
    values: np.ndarray, race_ids: Optional[np.ndarray], *, resamples: int, seed: int
) -> Optional[tuple[float, float]]:
    """Race-clustered 95% CI for the mean of ``values``.

    No race key means no honest interval: a row-level bootstrap would be narrower
    than the truth, and a too-narrow interval is worse than no interval.
    """
    if race_ids is None:
        return None
    module = _lazy_evaluation()
    if module is not None:
        try:
            interval = module.race_bootstrap_ci(
                values, race_ids, n_boot=int(resamples), seed=int(seed)
            )
        except Exception as exc:
            logger.debug("evaluation.race_bootstrap_ci raised (%s)", exc)
        else:
            lo = _num(getattr(interval, "lower", None))
            hi = _num(getattr(interval, "upper", None))
            if lo is not None and hi is not None:
                return (lo, hi)
            return None
    return _local_bootstrap_ci(values, race_ids, resamples=resamples, seed=seed)


def _local_bootstrap_ci(
    values: np.ndarray, race_ids: np.ndarray, *, resamples: int, seed: int
) -> Optional[tuple[float, float]]:
    """95% CI for the mean, resampling **races** rather than bets.

    Bets within a race share the same market state, so an i.i.d. bootstrap over bets
    understates the interval — the mistake that made the June CLV reads look tighter
    than they were.
    """
    mask = np.isfinite(values)
    if not mask.any():
        return None
    vals = values[mask]
    keys = np.asarray(race_ids, dtype=object)[mask]

    uniq, inverse = np.unique(keys, return_inverse=True)
    n_races = uniq.size
    if n_races < 2:
        return None

    sums = np.bincount(inverse, weights=vals, minlength=n_races)
    counts = np.bincount(inverse, minlength=n_races).astype(float)

    draws = int(max(1, resamples))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n_races, size=(draws, n_races))
    num = sums[idx].sum(axis=1)
    den = counts[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(den > 0, num / den, np.nan)
    means = means[np.isfinite(means)]
    if means.size == 0:
        return None
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


# ── local ledger metrics ──────────────────────────────────────────────────────
def _qualified(ledger: pd.DataFrame) -> pd.DataFrame:
    """Rows that count toward the gate.

    A ledger row *is* a qualified bet unless an explicit ``qualified`` flag says
    otherwise; a ledger that records rejected/passed candidates alongside placed
    ones must carry that column or its rows all count.
    """
    if "qualified" not in ledger.columns:
        return ledger
    mask = ledger["qualified"].map(lambda v: bool(v) and v == v).astype(bool)
    return ledger[mask]


def _race_keys(frame: pd.DataFrame) -> Optional[np.ndarray]:
    """Race cluster keys, or ``None`` when the ledger carries no race identifier.

    ``None`` is deliberate: treating each bet as its own race would both inflate the
    qualified-race count and shrink the clustered CLV interval — it would loosen two
    criteria at once.
    """
    col = _first_col(frame, _RACE_COLS)
    if col is None:
        return None
    return frame[col].astype(str).to_numpy(dtype=object)


def _clv_values(frame: pd.DataFrame) -> np.ndarray:
    """Per-bet CLV, preferring the two prices over any precomputed column.

    ``bet_price``/``close_price`` give an unambiguous fraction. A precomputed column
    may be a fraction or a log ratio; both share the sign of the mean, and the gate
    only compares against zero, so either is usable — but the computed fraction is
    preferred when the prices are present.
    """
    if "bet_price" in frame.columns and "close_price" in frame.columns:
        return np.atleast_1d(
            np.asarray(clv_pct(frame["bet_price"], frame["close_price"]), dtype=float)
        )
    col = _first_col(frame, _CLV_COLS)
    return _floats(frame, col)


def _ece(prob: np.ndarray, won: np.ndarray, bins: int = _ECE_BINS) -> Optional[float]:
    """Expected calibration error, deferring to the shared implementation."""
    mask = np.isfinite(prob) & np.isfinite(won)
    if not mask.any():
        return None
    module = _lazy_evaluation()
    if module is not None:
        try:
            return _num(module.expected_calibration_error(prob[mask], won[mask], bins=bins))
        except Exception as exc:
            logger.debug("evaluation.expected_calibration_error raised (%s)", exc)
    p, y = prob[mask], won[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = float(p.size)
    error = 0.0
    for b in range(bins):
        sel = which == b
        n = int(sel.sum())
        if n == 0:
            continue
        error += (n / total) * abs(float(p[sel].mean()) - float(y[sel].mean()))
    return float(error)


def _drawdown(frame: pd.DataFrame) -> Optional[float]:
    """Peak-to-trough bankroll drawdown as a fraction, or ``None`` if uncomputable.

    ``None`` fails the criterion. A ledger with no bankroll column cannot show it
    stayed inside the drawdown ceiling, and "cannot show" is not "did".
    """
    after = _floats(frame, "bankroll_after" if "bankroll_after" in frame.columns else None)
    before = _floats(frame, "bankroll_before" if "bankroll_before" in frame.columns else None)
    if after.size:
        equity = np.concatenate([before[:1], after]) if before.size else after
    elif before.size:
        equity = before
    else:
        return None
    equity = equity[np.isfinite(equity)]
    if equity.size == 0:
        return None
    return float(max_drawdown(equity))


def _local_metrics(ledger: pd.DataFrame, *, cfg, skip: frozenset = frozenset()) -> dict:
    """Compute the canonical metrics from the ledger.

    ``skip`` names metrics the caller already has, so the race-clustered bootstrap
    (the only expensive step) is not run twice.
    """
    frame = _qualified(ledger)
    n_bets = int(len(frame))
    out: dict[str, Any] = {
        "n_qualified_bets": n_bets,
        "n_qualified_races": 0,
        "weeks_elapsed": 0.0,
        "mean_clv": None,
        "clv_ci95": None,
        "ae": None,
        "ece": None,
        "max_drawdown_pct": None,
    }
    if n_bets == 0:
        return out

    race_keys = _race_keys(frame)
    out["n_qualified_races"] = 0 if race_keys is None else int(np.unique(race_keys).size)

    date_col = _first_col(frame, _DATE_COLS)
    if date_col is not None:
        dates = pd.to_datetime(frame[date_col], errors="coerce")
        if getattr(dates.dt, "tz", None) is not None:
            dates = dates.dt.tz_localize(None)
        dates = dates.dropna()
        if len(dates) >= 1:
            span = (dates.max() - dates.min()).total_seconds()
            out["weeks_elapsed"] = float(max(0.0, span) / _SECONDS_PER_WEEK)

    clv = _clv_values(frame)
    if clv.size and np.isfinite(clv).any():
        out["mean_clv"] = float(np.nanmean(clv))
        if "clv_ci95" not in skip:
            out["clv_ci95"] = _bootstrap_ci(
                clv,
                race_keys,
                resamples=int(cfg.selection.bootstrap_resamples),
                seed=int(cfg.selection.seed),
            )

    prob = _floats(frame, _first_col(frame, _PROB_COLS))
    won = _floats(frame, _first_col(frame, _WON_COLS))
    if prob.size and won.size:
        mask = np.isfinite(prob) & np.isfinite(won)
        expected = float(prob[mask].sum())
        if expected > 0:
            out["ae"] = float(won[mask].sum()) / expected
        out["ece"] = _ece(prob, won)

    out["max_drawdown_pct"] = _drawdown(frame)
    return out


def _gather_metrics(ledger: pd.DataFrame, *, cfg, metrics: Optional[dict]) -> dict:
    """Precomputed metrics where supplied, computed from the ledger otherwise.

    A caller-supplied ``metrics`` dict short-circuits the shared evaluation entirely
    (that is the point of the parameter); ``_local_metrics`` still fills whatever it
    did not cover, because a silently absent metric would otherwise read as ``None``
    and fail a criterion the evidence may in fact satisfy.
    """
    supplied = (
        _normalise_supplied(metrics)
        if metrics is not None
        else _shared_evaluate(ledger, cfg)
    )
    resolved = _local_metrics(ledger, cfg=cfg, skip=frozenset(supplied))
    resolved.update(supplied)
    if resolved.get("clv_ci95") is not None:
        resolved["clv_ci95"] = _ci(resolved["clv_ci95"])
    return resolved


# ── the gate ──────────────────────────────────────────────────────────────────
_BACKTEST_MESSAGE = (
    "A backtest, however long or however profitable it looks, can never satisfy the "
    "forward-release gate. A backtest is re-scored history: it cannot show that the "
    "deployment tracked live markets in real time, that the quoted price was actually "
    "available when the bet was issued, or that the result survived a market moving "
    "against it. Only forward (shadow/paper) evidence counts."
)


def evaluate_forward_gate(
    ledger,
    *,
    cfg,
    model_verdict,
    metrics: Optional[dict] = None,
    evidence_kind: str = "forward",
    now: Optional[datetime] = None,
) -> ForwardGateResult:
    """Evaluate every forward-release criterion. All must pass; none passes vacuously.

    ``ledger`` is the forward paper-betting ledger (a DataFrame, or ``None``/empty).
    ``metrics`` may carry precomputed :data:`CANONICAL_METRICS` to avoid re-deriving
    them; anything absent is computed here.
    """
    fg = cfg.forward_gate
    evaluated_at = _iso(now)
    kind = str(evidence_kind or "")

    # Rule zero: only forward evidence is admissible.
    if kind.strip().lower() != "forward":
        criterion = CriterionResult(
            name="forward_evidence",
            passed=False,
            observed=kind or None,
            required="evidence_kind == 'forward'",
            detail={"message": _BACKTEST_MESSAGE, "evidence_kind": kind},
        )
        return ForwardGateResult(
            passed=False,
            criteria=(criterion,),
            failed=("forward_evidence",),
            weeks_elapsed=0.0,
            n_qualified_bets=0,
            n_qualified_races=0,
            evidence_kind=kind,
            evaluated_at=evaluated_at,
            summary=(
                f"Forward-release gate NOT met: evidence_kind={kind!r} is not forward "
                f"evidence. {_BACKTEST_MESSAGE}"
            ),
        )

    if ledger is None or not isinstance(ledger, pd.DataFrame) or len(ledger) == 0:
        if ledger is not None and not isinstance(ledger, pd.DataFrame):
            logger.warning(
                "evaluate_forward_gate: ledger is %s, not a DataFrame; treating as "
                "missing (fail-closed)",
                type(ledger).__name__,
            )
        ledger = pd.DataFrame()

    m = _gather_metrics(ledger, cfg=cfg, metrics=metrics)
    weeks = float(m.get("weeks_elapsed") or 0.0)
    n_bets = int(m.get("n_qualified_bets") or 0)
    n_races = int(m.get("n_qualified_races") or 0)
    mean_clv = _num(m.get("mean_clv"))
    clv_ci = _ci(m.get("clv_ci95"))
    ae = _num(m.get("ae"))
    ece = _num(m.get("ece"))
    drawdown = _num(m.get("max_drawdown_pct"))

    criteria: list[CriterionResult] = []

    if fg.require_model_go:
        go = bool(getattr(model_verdict, "go", False))
        criteria.append(
            CriterionResult(
                name="model_go",
                passed=go,
                observed=str(getattr(model_verdict, "verdict_label", "NO-GO")),
                required="Stage-4 model verdict == GO",
                detail={
                    "reasons": list(getattr(model_verdict, "reasons", ()) or ()),
                    "source": str(getattr(model_verdict, "source", "")),
                    "available": bool(getattr(model_verdict, "available", False)),
                },
            )
        )

    criteria.append(
        CriterionResult(
            name="min_weeks",
            passed=weeks >= float(fg.min_weeks),
            observed=round(weeks, 3),
            required=f">= {float(fg.min_weeks):g} weeks of forward tracking",
            detail={"weeks_elapsed": weeks, "min_weeks": float(fg.min_weeks)},
        )
    )
    criteria.append(
        CriterionResult(
            name="min_qualified_bets",
            passed=n_bets >= int(fg.min_qualified_bets),
            observed=n_bets,
            required=f">= {int(fg.min_qualified_bets)} qualified bets",
            detail={"n_qualified_bets": n_bets},
        )
    )
    criteria.append(
        CriterionResult(
            name="min_qualified_races",
            passed=n_races >= int(fg.min_qualified_races),
            observed=n_races,
            required=f">= {int(fg.min_qualified_races)} qualified races",
            detail={"n_qualified_races": n_races},
        )
    )

    if fg.require_positive_mean_clv:
        criteria.append(
            CriterionResult(
                name="positive_mean_clv",
                passed=mean_clv is not None and mean_clv > 0.0,
                observed=(round(mean_clv, 6) if mean_clv is not None else None),
                required="mean CLV > 0",
                detail={
                    "mean_clv": mean_clv,
                    "note": (
                        "CLV is the single best forward indicator that a bet was +EV; "
                        "Stage 4 measured -13.4%."
                    ),
                },
            )
        )

    criteria.append(
        CriterionResult(
            name="clv_ci_lower",
            passed=clv_ci is not None and clv_ci[0] > float(fg.clv_ci_lower_above),
            observed=(round(clv_ci[0], 6) if clv_ci else None),
            required=f"95% race-clustered CI lower bound > {float(fg.clv_ci_lower_above):g}",
            detail={
                "clv_ci95": list(clv_ci) if clv_ci else None,
                "clustered_by": "race",
                "resamples": int(cfg.selection.bootstrap_resamples),
            },
        )
    )
    criteria.append(
        CriterionResult(
            name="ae_stable",
            passed=(ae is not None and float(fg.ae_min) <= ae <= float(fg.ae_max)),
            observed=(round(ae, 4) if ae is not None else None),
            required=f"{float(fg.ae_min):g} <= A/E <= {float(fg.ae_max):g}",
            detail={"ae": ae, "ae_min": float(fg.ae_min), "ae_max": float(fg.ae_max)},
        )
    )
    criteria.append(
        CriterionResult(
            name="calibration",
            passed=ece is not None and ece <= float(fg.max_calibration_ece),
            observed=(round(ece, 5) if ece is not None else None),
            required=f"ECE <= {float(fg.max_calibration_ece):g}",
            detail={"ece": ece, "bins": _ECE_BINS},
        )
    )
    criteria.append(
        CriterionResult(
            name="drawdown",
            passed=drawdown is not None and drawdown <= float(fg.max_drawdown_pct),
            observed=(round(drawdown, 5) if drawdown is not None else None),
            required=f"max drawdown <= {float(fg.max_drawdown_pct):g} of bankroll",
            detail={"max_drawdown_pct": drawdown},
        )
    )

    failed = tuple(c.name for c in criteria if not c.passed)
    passed = not failed
    total = len(criteria)

    if passed:
        summary = (
            f"Forward-release gate met: all {total} criteria satisfied over "
            f"{weeks:.1f} weeks of forward evidence, {n_bets} qualified bets across "
            f"{n_races} qualified races. This records that the stated thresholds held "
            f"on forward evidence over that window; it is not a claim about future "
            f"results, and staking remains subject to the paper-only override."
        )
    else:
        summary = (
            f"Forward-release gate NOT met: {len(failed)} of {total} criteria failed "
            f"({', '.join(failed)}). Observed {weeks:.1f} weeks of forward evidence, "
            f"{n_bets} qualified bets across {n_races} qualified races. "
            f"{total - len(failed)} criteria were met. The deployment stays paper-only."
        )

    result = ForwardGateResult(
        passed=passed,
        criteria=tuple(criteria),
        failed=failed,
        weeks_elapsed=weeks,
        n_qualified_bets=n_bets,
        n_qualified_races=n_races,
        evidence_kind=kind,
        evaluated_at=evaluated_at,
        summary=summary,
    )
    logger.info("forward gate: %s", result.state_label)
    return result
