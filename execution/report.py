"""Stage-5 reporting: forward-validation and today's-candidates, in Markdown.

Requirement 10 asks for a report that **separates historical backtest, shadow /
paper bets, and any future real execution**. The separation here is structural,
not a matter of wording:

* :func:`forward_validation_report` renders three lanes in a fixed order and
  always renders all three. The real-execution lane has no data path into it —
  :data:`REAL_EXECUTION_LANE` is a constant string, and there is no parameter
  that could fill it. A lane that cannot be populated cannot be mistaken for one
  that happens to be empty today.
* Every number carries its ``evidence_kind`` (``backtest`` / ``paper``), because
  the same ROI means different things in each lane, and only ``paper`` (forward)
  evidence can move the release gate.
* :func:`candidates_report` prints the model verdict and the forward-gate state
  **above** any runner, and when either fails it lists the failed criteria and
  prints no stake — requirement 4 of the acceptance list.

The honesty rule this module must never break (``CLAUDE.md``): beating the market
on log-loss is the gate, but a positive verdict with negative CLV is paper-only.
So the header renders the model verdict and the CLV read together, and there is
no code path that prints "recommended" or a real-money instruction.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_REPORT_DIR = "reports"

# The third lane. A constant, deliberately: see the module docstring.
REAL_EXECUTION_LANE = (
    "**No real-money execution exists.** This system is paper-only by "
    "configuration (`execution.paper_only`), enforced in storage by a "
    "`CHECK (paper_only = 1)` constraint on the ticket table and in code by "
    "`TicketStore` raising `RealMoneyTicketRefused`. This lane is not empty "
    "pending data — there is no code path that can fill it."
)

_LANES = ("backtest", "paper", "real")


# ── small formatting helpers ─────────────────────────────────────────────────
def _num(value: Any, digits: int = 4, *, pct: bool = False) -> str:
    """Format a number, or say plainly that there isn't one."""
    if value is None:
        return "n/a"
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if out != out:  # NaN
        return "n/a"
    return f"{out * 100:.2f}%" if pct else f"{out:.{digits}f}"


def _bounds(iv: Any) -> tuple[Any, Any]:
    """``(lower, upper)`` from an :class:`~execution.evaluation.Interval` or its dict.

    ``lower``/``upper`` are the canonical names the evaluation layer emits, in
    both the dataclass and its ``to_dict``. ``low``/``high`` are accepted only so
    a hand-built mapping cannot silently render as "n/a" — which is what this
    function did for every interval in the 2026-07-27 run, turning "ROI 31% with
    a 95% CI that includes zero" into an unqualified "ROI 31%".
    """
    if iv is None:
        return None, None
    if isinstance(iv, Mapping):
        get = iv.get
    else:
        def get(key):
            return getattr(iv, key, None)
    lo = get("lower")
    hi = get("upper")
    return (get("low") if lo is None else lo), (get("high") if hi is None else hi)


def _interval(iv: Any, digits: int = 4, *, pct: bool = False) -> str:
    """Render a confidence interval, including which side of zero it sits."""
    lo, hi = _bounds(iv)
    if lo is None or hi is None:
        return "n/a"
    return f"[{_num(lo, digits, pct=pct)}, {_num(hi, digits, pct=pct)}]"


def _tick(flag: Any) -> str:
    return "PASS" if flag else "FAIL"


def _as_dict(value: Any) -> dict:
    """Accept a dataclass with ``to_dict`` or a plain mapping."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    return dict(to_dict()) if callable(to_dict) else {}


def _now_iso(now: Optional[datetime] = None) -> str:
    return (now or datetime.now(tz=timezone.utc)).isoformat(timespec="seconds")


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    body = [list(map(str, r)) for r in rows]
    if not body:
        return ["_(no rows)_", ""]
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in body]
    out.append("")
    return out


# ── the strategy comparison block ────────────────────────────────────────────
def strategy_table(metrics: Mapping[str, Any]) -> list[str]:
    """Model vs de-vigged market vs favourite, on the same eligible races.

    Sample size leads every row. A reader who sees `n_bets` first is much less
    likely to read a 4-bet ROI as a finding.
    """
    rows = []
    for name, m in metrics.items():
        d = _as_dict(m)
        rows.append([
            name,
            d.get("n_bets", 0),
            d.get("n_races", 0),
            _num(d.get("roi"), pct=True),
            _interval(d.get("roi_ci"), pct=True),
            _num(d.get("hit_rate"), pct=True),
            _num(d.get("mean_clv_pct"), pct=True),
            _interval(d.get("clv_ci")),
            _num(d.get("ae_ratio"), 3),
            _interval(d.get("ae_ci"), 3),
            _num(d.get("max_drawdown_pct"), pct=True),
            d.get("longest_losing_streak", 0),
            _num(d.get("turnover"), 2),
        ])
    return _table(
        ["strategy", "bets", "races", "ROI", "ROI 95% CI", "hit rate",
         "mean CLV", "CLV 95% CI (log)", "A/E", "A/E 95% CI", "max DD",
         "worst streak", "turnover"],
        rows,
    )


def _clv_reading(metrics: Mapping[str, Any]) -> list[str]:
    """State what the CLV interval licenses — the one number that predicts edge."""
    model = _as_dict(metrics.get("model_only"))
    if not model:
        return []
    ci = model.get("clv_ci")
    lo, _hi = _bounds(ci)
    mean = model.get("mean_clv_log")
    if mean is None or lo is None:
        return [
            "**CLV read:** not measurable on this ledger (too few settled bets "
            "carrying a closing price). Absence of a CLV read is not a pass.",
            "",
        ]
    if lo > 0:
        verdict = (
            "the interval is entirely above zero, which is the one result that "
            "would support a genuine edge — it still needs the forward window "
            "below before it means anything about live betting"
        )
    elif mean > 0:
        verdict = (
            "positive on average but the interval spans zero, so it is not "
            "distinguishable from noise"
        )
    else:
        verdict = (
            "negative: bets are being struck at prices the market subsequently "
            "beat. Per `CLAUDE.md`, a positive model verdict with negative CLV "
            "is paper-only and never a green light to bet"
        )
    return [f"**CLV read:** mean {_num(mean)} log units "
            f"({_num(model.get('mean_clv_pct'), pct=True)}), 95% CI "
            f"{_interval(ci)} — {verdict}.", ""]


def _roi_reading(metrics: Mapping[str, Any]) -> list[str]:
    """State what the ROI interval licenses, next to the ROI itself.

    A headline ROI is the number a reader takes away, and on a 63-bet ledger it
    is almost always inside the noise. Printing the interval in a table cell is
    not enough — the 2026-07-27 run showed model_only at +31.10% ROI, which reads
    as a finding until you notice the 95% CI runs from -9.07% to +70.89%. This
    says it in words, in the same place.
    """
    model = _as_dict(metrics.get("model_only"))
    if not model or model.get("roi") is None:
        return []
    roi, ci = model.get("roi"), model.get("roi_ci")
    lo, hi = _bounds(ci)
    n = model.get("n_bets") or 0
    if lo is None or hi is None:
        verdict = "no interval could be computed, so the point estimate stands alone"
    elif lo > 0:
        verdict = (
            "the interval is entirely above zero on this window — a necessary "
            "condition for an edge, nowhere near a sufficient one, and it says "
            "nothing about the forward window"
        )
    elif hi < 0:
        verdict = "the interval is entirely below zero: this lost money, not noise"
    else:
        verdict = (
            f"the interval spans zero, so this result is **not distinguishable "
            f"from chance** on {int(n)} bet(s). It is not evidence of "
            f"profitability and must not be reported as a return"
        )
    return [f"**ROI read:** {_num(roi, pct=True)} over {int(n)} bet(s), 95% CI "
            f"{_interval(ci, pct=True)} — {verdict}.", ""]


# ── the selection-protocol block ─────────────────────────────────────────────
def selection_block(selection: Optional[Mapping]) -> list[str]:
    """How many strategies were tried, and what that does to the test number."""
    d = _as_dict(selection)
    if not d:
        return ["_Selection protocol did not run._", ""]
    def _cell(window: Any, i: int) -> Any:
        """One field of a split window, whatever shape it arrived in.

        ``selection`` emits ``[start, end, n_rows]``; a lock file round-tripped
        through JSON may hand back a mapping instead. Reading both means a
        reused lock cannot break the report the way it did before.
        """
        keys = ("start", "end", "n_rows")
        if isinstance(window, Mapping):
            value = window.get(keys[i])
        elif isinstance(window, (list, tuple)) and len(window) > i:
            value = window[i]
        else:
            return "?"
        return "?" if value is None else value

    sel = d.get("selected") or {}
    lines = [
        "### Selection protocol (anti-threshold-mining)",
        "",
        f"- Strategies tried: **{d.get('n_strategies_tried')}**",
        f"- Multiple-testing correction: **{d.get('correction')}**, "
        f"alpha {_num(d.get('alpha'), 3)} -> adjusted "
        f"**{_num(d.get('adjusted_alpha'), 5)}**",
        f"- Selected once: **{sel.get('name', 'none')}** "
        f"`{json.dumps(sel.get('params', {}), sort_keys=True)}`",
        f"- Train score {_num(d.get('train_score'))} · "
        f"validation {_num(d.get('validation_score'))} · "
        f"**test {_num(d.get('test_score'))}**",
        f"- Test window scored at: {d.get('test_scored_at') or 'not scored'}"
        f"{' (reused existing lock — NOT re-scored)' if d.get('reused_lock') else ''}",
        "",
    ]
    windows = d.get("windows") or {}
    if windows:
        # ``selection._window`` returns a positional ``[start, end, n_rows]``.
        lines += _table(
            ["window", "start", "end", "rows"],
            [[k] + [_cell(windows.get(k), i) for i in range(3)]
             for k in ("train", "validation", "test") if k in windows],
        )
    note = d.get("selection_bias_note")
    if note:
        lines += [f"> {note}", ""]
    lines += superseded_locks_block(d.get("superseded_locks"))
    return lines


def superseded_locks_block(locks: Any) -> list[str]:
    """Every earlier lock the test window was already spent on.

    The bias note claims the winner "was scored on the test window exactly
    once". That is true of the *current* lock and false of the programme as soon
    as a lock is reset — each superseded lock is another draw from the same
    window, and the effective number of comparisons is larger than the ``sidak``
    correction accounts for. Naming them, with the winner and score each one
    produced, is the only way the correction's understatement stays visible
    rather than being quietly repaired by a deleted file.
    """
    rows = [_as_dict(l) for l in (locks or [])]
    rows = [r for r in rows if r]
    if not rows:
        return []
    out = [
        f"**Selection lock was reset {len(rows)} time(s).** The test window has "
        f"therefore been scored more than once across this programme, so the "
        f"Šidák correction above **understates** the true multiplicity. Treat the "
        f"current test score as the more optimistic of several draws.",
        "",
    ]
    out += _table(
        ["superseded lock", "winner", "test score", "scored at", "why it was reset"],
        [[
            f"`{r.get('path', '?')}`",
            (_as_dict(r.get("selected")) or {}).get("name", "?"),
            _num(r.get("test_score"), 5),
            r.get("test_scored_at") or "?",
            r.get("reason") or "not recorded",
        ] for r in rows],
    )
    return out


def walk_config_block(walk: Optional[Mapping]) -> list[str]:
    """Which strategy the friction walk actually ran.

    Without this the table below is a set of numbers with no owner. A reader
    seeing ``model_only`` naturally assumes it is the strategy the selection
    protocol chose; when the walk falls back to config defaults — because the
    test window was already spent and the protocol refused to re-score it — that
    assumption is wrong, and the report has to say so rather than let the
    heading imply it.
    """
    d = _as_dict(walk)
    if not d:
        return []
    band = d.get("odds_band")
    band_text = (
        f"{_num(band[0], 2)}–{_num(band[1], 2)}" if band and len(band) == 2 else "whole book"
    )
    return [
        f"_Walk ran: **{d.get('strategy') or 'config defaults'}** "
        f"(source: {d.get('source')}) — min edge {_num(d.get('min_edge'), 3)}, "
        f"min EV {_num(d.get('min_expected_value'), 3)}, odds band {band_text}. "
        f"{d.get('band_applies_to', '')}_",
        "",
    ]


# ── the forward-release gate block ───────────────────────────────────────────
def forward_gate_block(gate: Optional[Mapping]) -> list[str]:
    """Every criterion, its observed value, and what was required of it."""
    d = _as_dict(gate)
    if not d:
        return ["_Forward gate did not run._", ""]
    lines = [
        "### Forward-release gate",
        "",
        f"**{d.get('state_label', 'UNKNOWN')}** — evidence kind: "
        f"`{d.get('evidence_kind')}`",
        "",
        f"- Weeks elapsed: {_num(d.get('weeks_elapsed'), 2)}",
        f"- Qualified bets: {d.get('n_qualified_bets')} over "
        f"{d.get('n_qualified_races')} races",
        "",
    ]
    lines += _table(
        ["criterion", "result", "observed", "required"],
        [[c.get("name"), _tick(c.get("passed")),
          _num(c.get("observed")) if isinstance(c.get("observed"), (int, float))
          else (c.get("observed") if c.get("observed") is not None else "n/a"),
          c.get("required")]
         for c in (d.get("criteria") or ())],
    )
    if d.get("failed"):
        lines += [
            "**Failed criteria:** " + ", ".join(f"`{f}`" for f in d["failed"]),
            "",
        ]
    if d.get("summary"):
        lines += [f"> {d['summary']}", ""]
    return lines


# ── the model verdict block ──────────────────────────────────────────────────
def model_verdict_block(verdict: Optional[Mapping]) -> list[str]:
    d = _as_dict(verdict)
    if not d:
        return ["_Model verdict unavailable — treated as NO-GO._", ""]
    lines = [
        f"### Model gate: **{d.get('verdict', 'NO-GO')}**",
        "",
        f"- Source: `{d.get('source')}`"
        f"{'' if d.get('available') else ' (artifact missing)'}",
        f"- Window: {(d.get('window') or ['?', '?'])[0]} .. "
        f"{(d.get('window') or ['?', '?'])[-1]} "
        f"({d.get('n_races')} races, {d.get('n_runners')} runners)",
        f"- Headline line: `{d.get('headline_line')}`",
        "",
    ]
    lines_map = d.get("lines") or {}
    if lines_map:
        lines += _table(
            ["line", "races", "model LL", "market LL", "delta", "delta 95% CI",
             "beats market", "beats best de-vig", "significant"],
            [[name, lv.get("n_races"), _num(lv.get("model_log_loss")),
              _num(lv.get("market_log_loss")), _num(lv.get("logloss_delta")),
              _interval({"low": (lv.get("logloss_delta_ci95") or [None, None])[0],
                         "high": (lv.get("logloss_delta_ci95") or [None, None])[-1]}),
              _tick(lv.get("beats_market_logloss")),
              _tick(lv.get("beats_best_devig")),
              _tick(lv.get("significant"))]
             for name, lv in lines_map.items()],
        )
    for reason in d.get("reasons") or ():
        lines.append(f"- {reason}")
    lines.append("")
    return lines


# ── the forward window / gap-ledger block (Stage 6, requirement 8) ──────────
REQUIRED_WINDOW_WEEKS = 8
REQUIRED_WINDOW_DAYS = REQUIRED_WINDOW_WEEKS * 7


def window_state_block(
    window_state: Optional[Mapping],
    *,
    weeks_elapsed_qualifying: Optional[float] = None,
    now: Optional[datetime] = None,
) -> list[str]:
    """Window status, days elapsed vs the 8-week requirement, and reset history.

    Rendered plainly and without hedging: a ``reset`` state names exactly what
    changed (from :func:`execution.window.verify_window`'s own diff), because a
    silent reset would let a reader assume evidence had been accumulating when
    the clock was actually zeroed by a retrain or a config edit.
    """
    d = _as_dict(window_state)
    lines = ["### Forward-validation window", ""]
    status = str(d.get("status") or "not_started")
    if status == "not_started":
        lines += [
            "**NOT STARTED.** Live capture may be running independently of this "
            "clock — starting the formal 8-week validation window is a separate, "
            "deliberate action (`execution.window.start_window`), not implied by "
            "capture running.",
            "",
        ]
        return lines

    window_start = d.get("window_start")
    stamp = now or datetime.now(tz=timezone.utc)
    days_elapsed = 0
    if window_start:
        from datetime import date as _date
        days_elapsed = max(0, (stamp.date() - _date.fromisoformat(window_start)).days)
    label = "RUNNING" if status == "running" else "RESET"
    lines += [
        f"**{label}** since {window_start or '?'} — day {days_elapsed} of "
        f"{REQUIRED_WINDOW_DAYS} calendar days "
        f"({_num(weeks_elapsed_qualifying, 2) if weeks_elapsed_qualifying is not None else 'n/a'} "
        f"gap-adjusted weeks of qualifying capture).",
        "",
    ]
    history = d.get("history") or ()
    resets = [h for h in history if isinstance(h, Mapping) and h.get("event") == "reset"]
    if resets:
        lines += [
            f"**Reset {len(resets)} time(s).** Each reset zeroed accumulated "
            "forward evidence because a frozen input changed:",
            "",
        ]
        lines += _table(
            ["at", "reason"],
            [[r.get("at", "?"), r.get("reason") or "?"] for r in resets],
        )
    return lines


def gap_ledger_block(gap_days: Optional[Mapping] = None, gap_entries: Optional[Sequence] = None) -> list[str]:
    """Capture history: which days counted as qualifying evidence and which were gaps.

    A day with no capture is a gap, never a day with zero qualifying bets — the
    two look identical in a bet ledger. Rendering every gap with its cause is
    what keeps a machine that was off for a month from reading, months later,
    as though it had produced a month of clean no-bet days.
    """
    days = dict(gap_days or {})
    if not days:
        return [
            "### Capture history",
            "",
            "_No capture days recorded yet._",
            "",
        ]
    n_captured = sum(1 for v in days.values() if v.get("status") == "captured")
    n_gap = len(days) - n_captured
    lines = [
        "### Capture history",
        "",
        f"- Days recorded: {len(days)} ({n_captured} captured, {n_gap} gap)",
        "",
    ]
    entries = list(gap_entries) if gap_entries is not None else sorted(
        (k, v.get("cause")) for k, v in days.items() if v.get("status") != "captured"
    )
    if entries:
        lines += _table(
            ["date", "cause"],
            [[day, cause or "?"] for day, cause in entries],
        )
    else:
        lines += ["_No gaps recorded — every capture day qualified._", ""]
    return lines


# ── the whole forward-validation report ──────────────────────────────────────
def forward_validation_report(
    *,
    model_verdict: Optional[Mapping] = None,
    forward_gate: Optional[Mapping] = None,
    backtest_metrics: Optional[Mapping[str, Any]] = None,
    backtest_summaries: Optional[Mapping[str, Mapping]] = None,
    paper_metrics: Optional[Mapping[str, Any]] = None,
    paper_summary: Optional[Mapping] = None,
    shadow_observation: Optional[Mapping] = None,
    selection: Optional[Mapping] = None,
    deployment_state: str = "PAPER-ONLY",
    walk_config: Optional[Mapping] = None,
    window_state: Optional[Mapping] = None,
    gap_days: Optional[Mapping] = None,
    weeks_elapsed_qualifying: Optional[float] = None,
    now: Optional[datetime] = None,
) -> str:
    """Render the dated forward-validation report.

    Three lanes, always all three, always in this order: **backtest**, then
    **paper / shadow**, then **real execution** (which is a constant refusal).
    Keeping the empty lanes visible is the point — a report that omitted the
    paper lane while it had no data would read, months later, as though the
    backtest lane were the whole story.
    """
    stamp = _now_iso(now)
    day = stamp[:10]
    out: list[str] = [
        f"# Forward-validation report — {day}",
        "",
        f"_Generated {stamp}. Deployment state: **{deployment_state}**._",
        "",
        "This report keeps three kinds of evidence apart. They are not "
        "interchangeable: a backtest says what the mechanics would have done, "
        "paper trading says what the system actually decided in real time, and "
        "only the second can satisfy the forward-release gate.",
        "",
        "---",
        "",
    ]

    out += model_verdict_block(model_verdict)
    out += ["---", ""]
    out += forward_gate_block(forward_gate)
    out += ["---", ""]

    # ── lane 1: historical backtest ──────────────────────────────────────────
    out += [
        "## Lane 1 — historical backtest",
        "",
        "_Evidence kind: `backtest`. Simulated frictions over a fixed "
        "historical window. This measures execution mechanics. It is **not** "
        "evidence that a strategy is profitable and it cannot satisfy the "
        "forward gate._",
        "",
    ]
    out += walk_config_block(walk_config)
    if backtest_metrics:
        out += strategy_table(backtest_metrics)
        out += _roi_reading(backtest_metrics)
        out += _clv_reading(backtest_metrics)
    else:
        out += ["_No backtest ledger._", ""]

    for name, summary in (backtest_summaries or {}).items():
        d = _as_dict(summary)
        if not d:
            continue
        blocked = d.get("blocked_by") or {}
        out += [
            f"**{name}** — {d.get('n_struck', 0)} struck of "
            f"{d.get('n_attempts', 0)} attempts "
            f"(archive-settled {_num(d.get('archive_coverage'), pct=True)}); "
            f"blocked by: "
            + (", ".join(f"`{k}` x{v}" for k, v in blocked.items()) or "nothing"),
            "",
        ]
    # State the limits of the lane next to its numbers, not in a footnote.
    limits = next(
        (_as_dict(s).get("not_exercised") for s in (backtest_summaries or {}).values()
         if _as_dict(s).get("not_exercised")),
        None,
    )
    if limits:
        out += ["**What this lane does not exercise:**", ""]
        out += [f"- {item}" for item in limits]
        out.append("")

    out += selection_block(selection)
    out += ["---", ""]

    # ── lane 2: paper / shadow ───────────────────────────────────────────────
    out += [
        "## Lane 2 — paper / shadow bets",
        "",
        "_Evidence kind: `paper`. Tickets issued in real time by the live gate "
        "against prices available at decision time, settled against results. "
        "**This is the only lane the forward-release gate reads.**_",
        "",
    ]
    if paper_metrics:
        out += strategy_table(paper_metrics)
        out += _roi_reading(paper_metrics)
        out += _clv_reading(paper_metrics)
    else:
        out += [
            "_No paper tickets recorded yet. The forward window has not started, "
            "so the forward gate cannot pass — this is the expected state until "
            "the live loop has run for the configured minimum._",
            "",
        ]
    if paper_summary:
        d = _as_dict(paper_summary)
        out += [
            f"- Decisions logged: {d.get('n_tickets', 0)} "
            f"({d.get('n_pass', 0)} PASS — no stake, not a wager; "
            f"{d.get('n_candidates', 0)} candidate)",
            f"- Candidate tickets (paper wagers): {d.get('n_candidates', 0)} "
            f"({d.get('n_open', 0)} open, {d.get('n_settled', 0)} settled)",
            f"- First ticket: {d.get('first_ticket') or 'n/a'} · "
            f"latest: {d.get('last_ticket') or 'n/a'}",
            f"- PASS/shadow priced-and-observable: {d.get('n_pass_priced', 0)} "
            f"({d.get('n_pass_observed', 0)} observed) -- probability/CLV "
            "disclosure evidence only, never a wager and never counted toward "
            "the candidate figures above",
            "",
        ]

    if shadow_observation:
        so = _as_dict(shadow_observation)
        out += [
            "**This cycle's PASS/shadow reconciliation** (separate from candidate "
            "settlement above; always zero-stake):",
            f"- Observed: {so.get('observed', 0)} "
            f"({so.get('already_observed', 0)} already observed, "
            f"{so.get('still_open', 0)} still pending a result, "
            f"{so.get('unmeasurable_clv', 0)} settled with no closing price)",
            "",
        ]

    out += window_state_block(
        window_state, weeks_elapsed_qualifying=weeks_elapsed_qualifying, now=now
    )
    out += gap_ledger_block(gap_days)

    out += ["---", ""]

    # ── lane 3: real execution ───────────────────────────────────────────────
    out += ["## Lane 3 — real-money execution", "", REAL_EXECUTION_LANE, ""]

    out += [
        "---",
        "",
        "## How to read this",
        "",
        "- A short backtest cannot show that a strategy is safe or profitable. "
        "Sample size, the ROI interval and the CLV interval are printed for "
        "exactly that reason.",
        "- Staking limits in this system are **risk ceilings, not profitability "
        "claims**. They must not be raised without forward evidence.",
        "- \"No bet\" is a valid and expected output.",
        "",
    ]
    return "\n".join(out)


# ── today's candidates ───────────────────────────────────────────────────────
_TICKET_COLUMNS: tuple[tuple[str, str], ...] = (
    ("horse_name", "horse"),
    ("race_uid", "race"),
    ("bookmaker", "book"),
    ("offered_odds", "offered odds"),
    ("quote_fetched_at", "quoted at"),
    ("quote_age_seconds", "age (s)"),
    ("model_prob", "model prob"),
    ("market_adjusted_prob", "mkt-adj prob"),
    ("fair_odds", "fair odds"),
    ("market_prob", "market prob"),
    ("edge", "edge"),
    ("expected_value", "EV"),
    ("max_stake", "max stake"),
    ("data_quality", "data quality"),
    ("validation_state", "validation"),
)


def _candidate_rows(candidates: Iterable[Mapping]) -> list[list[str]]:
    rows = []
    for c in candidates:
        d = _as_dict(c)
        rows.append([
            str(d.get(key, "") if d.get(key) is not None else "n/a")
            for key, _ in _TICKET_COLUMNS
        ])
    return rows


def _pass_section(passes: Sequence[Mapping], *, detail_path: Optional[str]) -> list[str]:
    """Why every declined runner was declined, without 300 identical paragraphs.

    Three views of the same set, because one view alone misleads:

    * **by condition** — how many runners each condition stopped. This is the
      honest headline: on a NO-GO day every runner fails ``model_validation``,
      and a reader must see that the gate is doing one thing, not 300 things.
    * **distinct reasons** — every reason string exactly once. Repeating the same
      four model-verdict lines under 356 runners buries the handful of reasons
      that actually vary between them.
    * **per runner** — the condition names each runner failed, compactly. The full
      reason text per runner goes to the CSV named underneath, so nothing is lost.
    """
    rows = [_as_dict(p) for p in passes]
    by_condition: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        conditions = set()
        for reason in row.get("pass_reasons") or ():
            text = str(reason)
            by_reason[text] = by_reason.get(text, 0) + 1
            conditions.add(text.split(":", 1)[0])
        for cond in conditions:
            by_condition[cond] = by_condition.get(cond, 0) + 1

    out = [
        "## PASS — why each runner was declined",
        "",
        f"{len(rows)} runners considered, {len(rows)} declined. A missing input "
        "is a PASS reason, never a waiver.",
        "",
        "### By condition",
        "",
    ]
    out += _table(
        ["condition", "runners stopped", "share"],
        [[cond, n, f"{n / max(len(rows), 1) * 100:.1f}%"]
         for cond, n in sorted(by_condition.items(), key=lambda kv: -kv[1])],
    )
    out += ["### Distinct reasons", ""]
    out += _table(
        ["reason", "runners"],
        [[reason, n] for reason, n in sorted(by_reason.items(), key=lambda kv: -kv[1])],
    )
    out += [
        "### Per runner",
        "",
        "Conditions failed, one row per runner. Full reason text for every runner "
        + (f"is in `{detail_path}`." if detail_path else "is in the run artifact."),
        "",
    ]
    out += _table(
        ["horse", "race", "conditions failed"],
        [[row.get("horse_name", "?"), row.get("race_uid", "?"),
          ", ".join(sorted({str(r).split(":", 1)[0]
                            for r in (row.get("pass_reasons") or ())})) or "?"]
         for row in rows],
    )
    return out


def candidates_report(
    *,
    candidates: Sequence[Mapping] = (),
    passes: Sequence[Mapping] = (),
    model_verdict: Optional[Mapping] = None,
    forward_gate: Optional[Mapping] = None,
    race_date: Optional[str] = None,
    paper_only: bool = True,
    pass_detail_path: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    """Render today's candidates — or, far more often, today's honest PASS.

    The verdict header is printed **before** any runner and the body short-
    circuits when the model is NO-GO or the forward gate is unmet: in that state
    the report names the failed criteria and shows no stake at all. There is no
    argument that turns this into a real-money recommendation.
    """
    stamp = _now_iso(now)
    day = race_date or stamp[:10]
    mv = _as_dict(model_verdict)
    fg = _as_dict(forward_gate)
    model_go = bool(mv.get("go"))
    gate_passed = bool(fg.get("passed"))
    releasable = model_go and gate_passed

    out = [
        f"# Today's candidates — {day}",
        "",
        f"_Generated {stamp}._",
        "",
        f"| gate | state |",
        f"|---|---|",
        f"| Model | **{mv.get('verdict', 'NO-GO')}** |",
        f"| Forward release | **{fg.get('state_label', 'FORWARD GATE NOT MET')}** |",
        f"| Deployment | **{'PAPER-ONLY' if paper_only or not releasable else 'CANDIDATE-ELIGIBLE'}** |",
        "",
    ]

    if not releasable:
        failed = list(fg.get("failed") or ())
        out += [
            "## No real-money recommendations",
            "",
            "This report contains **no real-money recommendations**, because at "
            "least one release gate is not met. The criteria that failed:",
            "",
        ]
        if not model_go:
            out += [
                "- **Model gate: NO-GO.** The model has not been shown to beat "
                "the de-vigged market on the headline line.",
            ]
            for reason in mv.get("reasons") or ():
                out.append(f"  - {reason}")
        if not gate_passed:
            out.append("- **Forward-release gate not met.**")
            for c in fg.get("criteria") or ():
                if not c.get("passed"):
                    out.append(
                        f"  - `{c.get('name')}`: observed "
                        f"{c.get('observed') if c.get('observed') is not None else 'n/a'}, "
                        f"required {c.get('required')}"
                    )
            if not (fg.get("criteria") or ()):
                out += [f"  - `{f}`" for f in failed]
        out += [
            "",
            "Runners are listed below **for paper tracking only**. A row in this "
            "table is a shadow ticket: it records what the system would have "
            "done so that forward evidence can accumulate. It is not advice and "
            "no money is staked.",
            "",
        ]

    if candidates:
        out += [
            "## Paper candidates" if not releasable else "## Candidates",
            "",
        ]
        out += _table([label for _, label in _TICKET_COLUMNS],
                      _candidate_rows(candidates))
        out += ["**Why each passed the gate:**", ""]
        for c in candidates:
            d = _as_dict(c)
            reasons = d.get("pass_reasons") or d.get("reasons") or ()
            out.append(
                f"- **{d.get('horse_name', '?')}** ({d.get('race_uid', '?')}): "
                + ("; ".join(str(r) for r in reasons) or "no reasons recorded")
            )
        out.append("")
    else:
        out += [
            "## No candidates",
            "",
            "Nothing cleared the candidate gate today. **\"No bet\" is a valid "
            "and expected output** — the gate passes by default and only "
            "produces a candidate when every condition is affirmatively met.",
            "",
        ]

    if passes:
        out += _pass_section(passes, detail_path=pass_detail_path)

    out += [
        "---",
        "",
        "_This system is paper-only. No real money is ever staked. Staking "
        "figures shown are risk ceilings under a conservative fractional-Kelly "
        "policy, not profitability claims._",
        "",
    ]
    return "\n".join(out)


# ── writing ──────────────────────────────────────────────────────────────────
def write_report(
    text: str,
    *,
    name: str,
    report_dir: str = DEFAULT_REPORT_DIR,
    now: Optional[datetime] = None,
) -> str:
    """Write ``text`` to ``<report_dir>/<name>_<YYYYMMDD>.md`` and return the path."""
    day = _now_iso(now)[:10].replace("-", "")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"{name}_{day}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("report: wrote %s (%d chars)", path, len(text))
    return path


def write_json(payload: Mapping, *, path: str) -> str:
    """Write the machine-readable twin of a report."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str, sort_keys=False)
    logger.info("report: wrote %s", path)
    return path
