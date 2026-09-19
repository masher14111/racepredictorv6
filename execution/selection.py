"""Anti-threshold-mining selection protocol (Stage-5 requirement 4).

Why this module exists
----------------------
Stage 4 issued MODEL NO-GO: on 904 untouched races the price-free line *loses* to
the de-vigged pre-off market by 0.186 race log-loss. The cheapest way to make that
finding disappear is to try many threshold combinations — min edge x min EV x odds
band — and report whichever one happened to land positive. With 64 candidates and
a 5% alpha you expect roughly three "significant" winners from pure noise, which is
exactly how the June GO stamps recorded in ``memory/stage4-audit-nogo-20260727.md``
came to exist.

The protocol this module enforces is therefore mechanical:

1. **Tune on TRAIN.** Every candidate in the grid is scored on the training window.
2. **Select ONCE on VALIDATION.** Only a shortlist reaches validation, and exactly
   one winner comes out of it.
3. **Score TEST exactly once.** The test window is scored for the winner and then
   locked. A window scored twice is no longer untouched, so a second run with a
   *different* winner raises :class:`TestWindowAlreadyScored` rather than quietly
   handing back a fresh number.
4. **Report the search size.** ``n_strategies_tried`` and the multiple-testing
   adjusted alpha travel with the result, and ``selection_bias_note`` says in plain
   English that the test score is a single draw.

The grid is hard-truncated to ``cfg.selection.max_strategies``. The cap is what the
correction is computed against, so silently evaluating more points than the cap
would understate the correction — the truncation is loud, and it drops from the
*deterministic enumeration order*, never by score (dropping by score is itself
mining).

Threshold candidates are also floored at the configured gate: ``build_grid`` never
proposes a ``min_edge``/``min_expected_value`` looser than ``cfg.gates``. Search may
tighten a gate, never relax one.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Fraction of the (truncated) grid that is carried from TRAIN to VALIDATION. The
# shortlist exists so validation is not simply a second full sweep; the winner is
# still chosen on validation alone.
SHORTLIST_FRACTION = 0.25

# Candidate threshold values. These are *absolute* thresholds, filtered against the
# configured gate floors in :func:`build_grid` — the grid may tighten a gate, never
# loosen it. 4 x 4 x 4 = 64, which is exactly ``max_strategies`` by default.
EDGE_GRID: tuple[float, ...] = (0.02, 0.03, 0.05, 0.08)
EV_GRID: tuple[float, ...] = (0.05, 0.08, 0.12, 0.20)
# ``None`` upper bound means "unbounded" — kept as None rather than ``inf`` so the
# lock file stays strictly valid JSON.
ODDS_BANDS: tuple[tuple[str, float, Optional[float]], ...] = (
    ("all", 1.0, None),
    ("short_lt_4", 1.0, 4.0),
    ("mid_4_12", 4.0, 12.0),
    ("long_ge_12", 12.0, None),
)

_TOL = 1e-12


class TestWindowAlreadyScored(RuntimeError):
    """Raised when a locked TEST window would be scored a second time.

    The test window's whole value is that it was seen once. Scoring it again for a
    different strategy converts it into another validation fold, so the caller must
    either accept the locked selection or explicitly pass ``force=True`` and wear
    the consequence in the report.
    """


# ── value objects ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StrategyCandidate:
    """One point in the threshold grid."""

    name: str
    params: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of one full train -> validate -> score-test-once pass."""

    selected: Optional[StrategyCandidate]
    n_strategies_tried: int
    correction: str
    alpha: float
    adjusted_alpha: float
    train_score: Optional[float]
    validation_score: Optional[float]
    test_score: Optional[float]
    test_scored_at: Optional[str]
    selection_bias_note: str
    reused_lock: bool
    lock_path: str
    candidates: tuple[dict, ...]
    windows: dict

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict() if self.selected else None,
            "n_strategies_tried": self.n_strategies_tried,
            "correction": self.correction,
            "alpha": self.alpha,
            "adjusted_alpha": self.adjusted_alpha,
            "train_score": self.train_score,
            "validation_score": self.validation_score,
            "test_score": self.test_score,
            "test_scored_at": self.test_scored_at,
            "selection_bias_note": self.selection_bias_note,
            "reused_lock": self.reused_lock,
            "lock_path": self.lock_path,
            "candidates": [dict(c) for c in self.candidates],
            "windows": dict(self.windows),
        }


# ── multiple-testing corrections ──────────────────────────────────────────────
def _m(m: int) -> int:
    """Number of comparisons, floored at 1 (``m <= 0`` is treated as one test)."""
    try:
        m_int = int(m)
    except (TypeError, ValueError):
        return 1
    return m_int if m_int >= 1 else 1


def bonferroni_alpha(alpha: float, m: int) -> float:
    """``alpha / m`` — the conservative family-wise correction."""
    return float(alpha) / float(_m(m))


def sidak_alpha(alpha: float, m: int) -> float:
    """``1 - (1 - alpha) ** (1 / m)`` — exact under independent tests."""
    return 1.0 - (1.0 - float(alpha)) ** (1.0 / float(_m(m)))


def adjusted_alpha(correction: str, alpha: float, m: int) -> float:
    """Dispatch to the named correction. Unknown names fall back to Bonferroni.

    Falling back to the *most conservative* correction is deliberate: a typo in the
    config must never widen the acceptance region.
    """
    name = str(correction or "").strip().lower()
    if name == "bonferroni":
        return bonferroni_alpha(alpha, m)
    if name == "sidak":
        return sidak_alpha(alpha, m)
    if name == "none":
        return float(alpha)
    logger.warning(
        "adjusted_alpha: unknown correction %r; using bonferroni (most conservative)",
        correction,
    )
    return bonferroni_alpha(alpha, m)


# ── grid ──────────────────────────────────────────────────────────────────────
def _candidate_name(edge: float, ev: float, band: str) -> str:
    return f"edge{edge:.3f}_ev{ev:.3f}_{band}"


def build_grid(cfg) -> list[StrategyCandidate]:
    """Enumerate the threshold grid, hard-truncated to ``max_strategies``.

    Candidates looser than the configured gate floors are dropped before the cap is
    applied; if the floors exclude every grid point the configured floor itself is
    used as the single candidate, so the grid is never empty.
    """
    gates = cfg.gates
    sel = cfg.selection

    edges = [e for e in EDGE_GRID if e >= float(gates.min_edge) - _TOL]
    if not edges:
        edges = [float(gates.min_edge)]
    evs = [v for v in EV_GRID if v >= float(gates.min_expected_value) - _TOL]
    if not evs:
        evs = [float(gates.min_expected_value)]

    candidates: list[StrategyCandidate] = []
    for edge in edges:
        for ev in evs:
            for band, lo, hi in ODDS_BANDS:
                candidates.append(
                    StrategyCandidate(
                        name=_candidate_name(edge, ev, band),
                        params={
                            "min_edge": float(edge),
                            "min_expected_value": float(ev),
                            "odds_band": band,
                            "odds_min": float(lo),
                            "odds_max": (float(hi) if hi is not None else None),
                        },
                    )
                )

    cap = max(1, int(sel.max_strategies))
    if len(candidates) > cap:
        dropped = [c.name for c in candidates[cap:]]
        shown = ", ".join(dropped[:5]) + (" ..." if len(dropped) > 5 else "")
        logger.warning(
            "build_grid: grid of %d exceeds max_strategies=%d — truncated; "
            "dropped %d candidate(s): %s",
            len(candidates),
            cap,
            len(dropped),
            shown,
        )
        candidates = candidates[:cap]
    return candidates


# ── chronological split ───────────────────────────────────────────────────────
def _dates(frame: pd.DataFrame, date_col: str) -> pd.Series:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def chronological_split(
    frame: pd.DataFrame, *, cfg, date_col: str = "race_date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``frame`` into (train, validation, test) **by date**, never randomly.

    A random split leaks: the same race card lands on both sides of the boundary and
    the model is scored on days it was tuned on. Splitting on the sorted set of
    distinct dates makes each date belong to exactly one window.
    """
    if frame is None or len(frame) == 0:
        raise ValueError("chronological_split: frame is empty")
    if date_col not in frame.columns:
        raise ValueError(f"chronological_split: missing date column {date_col!r}")

    work = frame.copy()
    work["__split_date"] = _dates(work, date_col)
    work = work[work["__split_date"].notna()]
    if len(work) == 0:
        raise ValueError(f"chronological_split: no parseable dates in {date_col!r}")

    distinct = sorted(pd.unique(work["__split_date"]))
    n = len(distinct)
    if n < 3:
        raise ValueError(
            f"chronological_split: need >= 3 distinct dates to hold out a test "
            f"window, got {n}"
        )

    train_fraction = float(cfg.selection.train_fraction)
    validation_fraction = float(cfg.selection.validation_fraction)
    n_train = int(math.floor(n * train_fraction))
    n_val = int(math.floor(n * validation_fraction))
    # Every window keeps at least one date; the test window is what remains.
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)

    train_dates = set(distinct[:n_train])
    val_dates = set(distinct[n_train : n_train + n_val])
    test_dates = set(distinct[n_train + n_val :])

    def _take(keep: set) -> pd.DataFrame:
        out = work[work["__split_date"].isin(keep)].drop(columns=["__split_date"])
        return out.reset_index(drop=True)

    logger.info(
        "chronological_split: %d dates -> train=%d validation=%d test=%d",
        n,
        len(train_dates),
        len(val_dates),
        len(test_dates),
    )
    return _take(train_dates), _take(val_dates), _take(test_dates)


def _window(frame: pd.DataFrame, date_col: str) -> list:
    """``[start, end, n_rows]`` for a split, dates as ``YYYY-MM-DD`` strings."""
    if frame is None or len(frame) == 0 or date_col not in frame.columns:
        return [None, None, 0]
    dates = _dates(frame, date_col).dropna()
    if dates.empty:
        return [None, None, int(len(frame))]
    return [
        dates.min().strftime("%Y-%m-%d"),
        dates.max().strftime("%Y-%m-%d"),
        int(len(frame)),
    ]


# ── lock file ─────────────────────────────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_lock(path) -> Optional[dict]:
    """Read the selection lock, or ``None`` when it is absent/unreadable."""
    p = str(path)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("load_lock: %s is unreadable (%s); treating as absent", p, exc)
        return None
    return payload if isinstance(payload, dict) else None


def write_lock(path, result: SelectionResult) -> None:
    """Persist the selection so the test window can never be re-scored by accident."""
    p = str(path)
    parent = os.path.dirname(os.path.abspath(p))
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = result.to_dict()
    payload["written_at"] = _now_iso()
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("write_lock: selection locked at %s", p)


def _normalise_params(params: Any) -> dict:
    """JSON round-trip so a lock read back from disk compares equal to a fresh one."""
    if not isinstance(params, dict):
        return {}
    return json.loads(json.dumps(params, default=str))


def _same_candidate(locked: Any, candidate: Optional[StrategyCandidate]) -> bool:
    if not isinstance(locked, dict) or candidate is None:
        return False
    if str(locked.get("name")) != candidate.name:
        return False
    return _normalise_params(locked.get("params")) == _normalise_params(candidate.params)


# ── scoring ───────────────────────────────────────────────────────────────────
def _score(
    evaluate_fn: Callable[[pd.DataFrame, StrategyCandidate], float],
    frame: pd.DataFrame,
    candidate: StrategyCandidate,
) -> float:
    """Score one candidate, mapping "no answer" to ``-inf`` rather than to a pass.

    A candidate that errors or returns NaN must lose the comparison, not win it by
    sorting ahead of a genuine low score.
    """
    try:
        raw = evaluate_fn(frame, candidate)
    except Exception as exc:  # a broken candidate loses; it does not abort the sweep
        logger.warning("evaluate_fn failed for %s: %s", candidate.name, exc)
        return float("-inf")
    if raw is None:
        return float("-inf")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return float("-inf")
    return value if math.isfinite(value) else float("-inf")


def _jsonable(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def _bias_note(*, m: int, k: int, correction: str, alpha: float, adj: float) -> str:
    return (
        f"{m} strategy variant(s) were scored on the training window; {k} were "
        f"carried to validation and the single winner was scored on the test window "
        f"exactly once. A family-wise alpha of {alpha:.4g} becomes {adj:.4g} under "
        f"the {correction} correction for {m} comparison(s). The reported test score "
        f"is a single draw from one window and does not by itself establish an edge."
    )


def run_selection(
    frame: pd.DataFrame,
    *,
    cfg,
    evaluate_fn: Callable[[pd.DataFrame, StrategyCandidate], float],
    lock_path: Optional[str] = None,
    force: bool = False,
) -> SelectionResult:
    """Run the full protocol: tune on train, select on validation, score test once.

    ``evaluate_fn(frame, candidate) -> float`` is injected (higher is better) so the
    protocol is independent of what is being scored.

    Train and validation may be re-scored freely — they are re-scored here on every
    call, which is also how a *different* winner is detected against an existing
    lock. Only the test window is one-shot.
    """
    lock = str(lock_path or cfg.selection.lock_file)
    grid = build_grid(cfg)
    train, validation, test = chronological_split(frame, cfg=cfg)

    m = len(grid)
    correction = str(cfg.selection.correction)
    alpha = float(cfg.selection.alpha)
    adj = adjusted_alpha(correction, alpha, m)

    # 1. TRAIN — every candidate.
    train_scores = {c.name: _score(evaluate_fn, train, c) for c in grid}

    # 2. VALIDATION — a shortlist, then exactly one winner.
    k = max(1, int(math.ceil(m * SHORTLIST_FRACTION)))
    shortlist = sorted(grid, key=lambda c: (-train_scores[c.name], c.name))[:k]
    validation_scores = {c.name: _score(evaluate_fn, validation, c) for c in shortlist}
    winner = sorted(shortlist, key=lambda c: (-validation_scores[c.name], c.name))[0]

    candidates = tuple(
        {
            **c.to_dict(),
            "train_score": _jsonable(train_scores[c.name]),
            "validation_score": (
                _jsonable(validation_scores[c.name])
                if c.name in validation_scores
                else None
            ),
            "selected": c.name == winner.name,
        }
        for c in grid
    )
    windows = {
        "train": _window(train, "race_date"),
        "validation": _window(validation, "race_date"),
        "test": _window(test, "race_date"),
    }
    note = _bias_note(m=m, k=k, correction=correction, alpha=alpha, adj=adj)

    def _result(
        *, test_score: Optional[float], test_scored_at: Optional[str], reused: bool
    ) -> SelectionResult:
        return SelectionResult(
            selected=winner,
            n_strategies_tried=m,
            correction=correction,
            alpha=alpha,
            adjusted_alpha=adj,
            train_score=_jsonable(train_scores[winner.name]),
            validation_score=_jsonable(validation_scores[winner.name]),
            test_score=test_score,
            test_scored_at=test_scored_at,
            selection_bias_note=note,
            reused_lock=reused,
            lock_path=lock,
            candidates=candidates,
            windows=windows,
        )

    # 3. TEST — once, and only once.
    existing = load_lock(lock)
    already_scored = bool(existing and existing.get("test_scored_at"))
    if already_scored and not force:
        if _same_candidate(existing.get("selected"), winner):
            logger.info(
                "run_selection: reusing locked selection %s (test window already "
                "scored at %s); test NOT re-scored",
                winner.name,
                existing.get("test_scored_at"),
            )
            raw_test = existing.get("test_score")
            return _result(
                test_score=(float(raw_test) if raw_test is not None else None),
                test_scored_at=str(existing.get("test_scored_at")),
                reused=True,
            )
        locked_name = (existing.get("selected") or {}).get("name")
        raise TestWindowAlreadyScored(
            f"test window at {lock} was already scored on {existing.get('test_scored_at')} "
            f"for strategy {locked_name!r}; scoring it again for {winner.name!r} would "
            f"make it a second validation fold. Re-run with force=True only if you "
            f"accept that the test result is no longer an untouched single draw."
        )

    if already_scored and force:
        logger.warning(
            "run_selection: force=True — re-scoring an already-scored TEST window "
            "(%s). The reported test result is no longer a single untouched draw.",
            lock,
        )

    test_score = _score(evaluate_fn, test, winner)
    result = _result(
        test_score=_jsonable(test_score), test_scored_at=_now_iso(), reused=False
    )
    write_lock(lock, result)
    return result
