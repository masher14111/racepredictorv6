"""ui/forward_validation.py — three lanes of evidence, kept apart on screen.

The page answers one question honestly: *what has this system actually shown,
and what has it merely simulated?* It is the on-screen twin of
``execution.report.forward_validation_report`` and obeys the same rule — three
lanes, always all three, in a fixed order:

1. **Historical backtest** — re-scored history with simulated frictions. Says
   what the mechanics would have done. Cannot satisfy the release gate.
2. **Paper / shadow** — tickets issued in real time against prices that were
   actually on the book. The only lane the forward gate reads.
3. **Real execution** — structurally empty. Not "no data yet": there is no code
   path that can fill it, because ``paper_only`` is enforced by a CHECK
   constraint on the ticket table.

Honesty rules baked in, matching ``ui/model_honesty.py``:

* The model verdict and the forward-gate state are rendered **above** any
  number, because every number below is conditional on them.
* A ROI is never shown without its sample size and its interval beside it. A
  20-bet ROI rendered as a big green number is the single most misleading thing
  this page could do.
* Nothing here places a bet, and no surface implies real money.

The data/formatter layer is Streamlit-free so it stays unit-testable headless
(``tests/ui/test_forward_validation.py``); :func:`render` is the only
Streamlit-bound entry point, called by ``ui/pages/16_Forward_Validation.py``.
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
_RUN_ARTIFACT = _ROOT / "data" / "execution" / "stage5_run.json"
_DAILY_LOOP_ARTIFACT = _ROOT / "data" / "execution" / "daily_loop_run.json"
_REPORT_DIR = _ROOT / "reports"
REQUIRED_WINDOW_DAYS = 56

LANES = ("backtest", "paper", "real")

LANE_BLURB = {
    "backtest": (
        "Re-scored history with simulated frictions — latency, rejections, "
        "Rule 4, commission. Measures the mechanics. Cannot satisfy the release "
        "gate, however good it looks."
    ),
    "paper": (
        "Tickets issued in real time against prices that were on the book at "
        "decision time, then settled against results. The only evidence the "
        "forward-release gate reads."
    ),
    "real": (
        "No real-money execution exists. Paper-only is enforced in storage by a "
        "CHECK constraint and in code by TicketStore refusing any other ticket. "
        "This lane is not awaiting data — nothing can fill it."
    ),
}


# ── loading ──────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_run(
    path: Path = _RUN_ARTIFACT, *, daily_loop_path: Path = _DAILY_LOOP_ARTIFACT
) -> Optional[dict]:
    """The freshest run artifact.

    ``scripts.daily_paper_loop`` (Stage 6) is the live, real-evidence run and
    takes priority; it carries no backtest lane of its own (Stage 6 does not run
    backtests), so Lane 1 falls back on whatever ``scripts.paper_betting``
    (Stage 5) last recorded for the backtest/selection keys only. ``None`` only
    when neither artifact has ever been written.
    """
    daily = _load_json(daily_loop_path)
    legacy = _load_json(path)
    if daily is None:
        return legacy
    if legacy is not None:
        for key in ("backtest", "backtest_summaries", "selection"):
            daily.setdefault(key, legacy.get(key))
    return daily


def latest_reports(report_dir: Path = _REPORT_DIR) -> dict[str, Optional[str]]:
    """Newest dated forward-validation and candidates reports, by name."""
    out: dict[str, Optional[str]] = {"forward_validation": None, "candidates": None}
    for key, stem in (("forward_validation", "forward_validation_"),
                      ("candidates", "todays_candidates_")):
        try:
            matches = sorted(report_dir.glob(f"{stem}*.md"))
        except OSError:
            matches = []
        if matches:
            out[key] = str(matches[-1])
    return out


# ── formatting ───────────────────────────────────────────────────────────────
def fmt_num(value: Any, digits: int = 4, *, pct: bool = False, dash: str = "—") -> str:
    if value is None:
        return dash
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    if out != out:
        return dash
    return f"{out * 100:.2f}%" if pct else f"{out:.{digits}f}"


def interval_bounds(iv: Any) -> tuple[Any, Any]:
    """``(lower, upper)`` — the names ``execution.evaluation.Interval`` emits.

    ``low``/``high`` are read as a fallback only. Getting this wrong is silent:
    the page renders an em-dash where an interval should be, and a reader takes
    a point estimate as unqualified. That is precisely what the report layer did
    to the 2026-07-27 ROI figures before this was fixed.
    """
    if not isinstance(iv, Mapping):
        return None, None
    lo = iv.get("lower")
    hi = iv.get("upper")
    return (iv.get("low") if lo is None else lo), (iv.get("high") if hi is None else hi)


def fmt_interval(iv: Any, digits: int = 4, *, pct: bool = False) -> str:
    lo, hi = interval_bounds(iv)
    if lo is None or hi is None:
        return "—"
    return f"{fmt_num(lo, digits, pct=pct)} … {fmt_num(hi, digits, pct=pct)}"


def sample_caveat(n_bets: Any) -> Optional[str]:
    """The warning that belongs next to a small-sample number.

    Thresholds are deliberately blunt. The point is not precision about where
    "enough" begins — it is that a reader never sees a ratio without being told
    how many bets produced it.
    """
    try:
        n = int(n_bets or 0)
    except (TypeError, ValueError):
        return "Sample size unknown — treat every ratio here as unmeasured."
    if n == 0:
        return "No settled bets. Every ratio below is undefined, not zero."
    if n < 50:
        return (f"Only {n} bets. At this size a ROI is dominated by whether one "
                f"or two long-priced runners happened to win.")
    if n < 200:
        return (f"{n} bets. Enough to see gross breakage, not enough to "
                f"distinguish a small edge from noise.")
    return None


def verdict_tone(go: Any) -> str:
    return "ok" if bool(go) else "bad"


def gate_rows(gate: Optional[Mapping]) -> list[dict]:
    """Every forward-gate criterion as a display row, failures first."""
    if not isinstance(gate, Mapping):
        return []
    rows = []
    for c in gate.get("criteria") or ():
        if not isinstance(c, Mapping):
            continue
        rows.append({
            "name": str(c.get("name", "?")),
            "passed": bool(c.get("passed")),
            "observed": c.get("observed"),
            "required": c.get("required"),
            "detail": c.get("detail"),
        })
    rows.sort(key=lambda r: r["passed"])
    return rows


def strategy_rows(metrics: Optional[Mapping]) -> list[dict]:
    """One row per strategy, sample size first — it governs how to read the rest."""
    if not isinstance(metrics, Mapping):
        return []
    rows = []
    for name, m in metrics.items():
        if not isinstance(m, Mapping):
            continue
        rows.append({
            "strategy": name,
            "bets": m.get("n_bets", 0),
            "races": m.get("n_races", 0),
            "roi": m.get("roi"),
            "roi_ci": m.get("roi_ci"),
            "hit_rate": m.get("hit_rate"),
            "clv_pct": m.get("mean_clv_pct"),
            "clv_ci": m.get("clv_ci"),
            "ae": m.get("ae_ratio"),
            "max_dd": m.get("max_drawdown_pct"),
            "streak": m.get("longest_losing_streak"),
            "caveat": sample_caveat(m.get("n_bets")),
        })
    rows.sort(key=lambda r: r["strategy"] != "model_only")
    return rows


def clv_verdict(metrics: Optional[Mapping], key: str = "model_only") -> Optional[dict]:
    """The CLV read, stated as what it licenses rather than as a number alone."""
    if not isinstance(metrics, Mapping):
        return None
    m = metrics.get(key)
    if not isinstance(m, Mapping):
        return None
    ci = m.get("clv_ci") if isinstance(m.get("clv_ci"), Mapping) else None
    mean = m.get("mean_clv_log")
    if mean is None:
        return {
            "tone": "warn",
            "text": "CLV is not measurable on this ledger. Absence of a CLV read "
                    "is not a pass.",
        }
    lo, _hi = interval_bounds(ci)
    if lo is not None and lo > 0:
        tone, text = "ok", (
            "The 95% interval is entirely above zero — the one result that would "
            "support a genuine edge. It still needs the forward window before it "
            "says anything about live betting."
        )
    elif float(mean) > 0:
        tone, text = "warn", (
            "Positive on average, but the interval spans zero: not "
            "distinguishable from noise."
        )
    else:
        tone, text = "bad", (
            "Negative. Bets are struck at prices the market then beat. A positive "
            "model verdict with negative CLV is paper-only and never a green "
            "light to bet."
        )
    return {"tone": tone, "mean": mean, "ci": ci,
            "pct": m.get("mean_clv_pct"), "text": text}


def deployment_line(run: Optional[Mapping]) -> dict:
    """The one-line answer: what is this system allowed to do right now?"""
    payload = run or {}
    verdict = payload.get("model_verdict") or {}
    gate = payload.get("forward_gate") or {}
    releasable = bool(verdict.get("go")) and bool(gate.get("passed"))
    return {
        "model": str(verdict.get("verdict", "NO-GO")),
        "gate": str(gate.get("state_label", "FORWARD GATE NOT MET")),
        "deployment": str(payload.get("deployment", "PAPER-ONLY")),
        "releasable": releasable,
        "paper_only": bool(payload.get("paper_only", True)),
        "generated_at": payload.get("generated_at"),
    }


def selection_rows(run: Optional[Mapping]) -> Optional[dict]:
    """The anti-mining protocol, flattened for display."""
    sel = (run or {}).get("selection")
    if not isinstance(sel, Mapping):
        return None
    selected = sel.get("selected") or {}
    return {
        "tried": sel.get("n_strategies_tried"),
        "correction": sel.get("correction"),
        "alpha": sel.get("alpha"),
        "adjusted_alpha": sel.get("adjusted_alpha"),
        "selected": selected.get("name"),
        "params": selected.get("params") or {},
        "train": sel.get("train_score"),
        "validation": sel.get("validation_score"),
        "test": sel.get("test_score"),
        "scored_at": sel.get("test_scored_at"),
        "reused_lock": bool(sel.get("reused_lock")),
        "note": sel.get("selection_bias_note"),
    }


def window_row(run: Optional[Mapping]) -> dict:
    """Forward-window status, days elapsed vs the 8-week requirement, resets.

    A reset must read as loudly here as it does in the Markdown report — a
    silent reset would let a viewer assume evidence had been accumulating
    across a retrain or config edit that actually zeroed the clock.
    """
    payload = run or {}
    win = payload.get("window") or {}
    status = str(win.get("status") or "not_started")
    window_start = win.get("window_start")
    days_elapsed = 0
    if window_start:
        from datetime import date as _date, datetime as _datetime, timezone as _tz
        gen = payload.get("generated_at")
        try:
            now = _datetime.fromisoformat(gen) if gen else _datetime.now(tz=_tz.utc)
        except ValueError:
            now = _datetime.now(tz=_tz.utc)
        days_elapsed = max(0, (now.date() - _date.fromisoformat(window_start)).days)
    resets = [h for h in (win.get("history") or ())
              if isinstance(h, Mapping) and h.get("event") == "reset"]
    return {
        "status": status,
        "running": status in ("running", "reset"),
        "window_start": window_start,
        "days_elapsed": days_elapsed,
        "required_days": REQUIRED_WINDOW_DAYS,
        "weeks_elapsed_qualifying": payload.get("weeks_elapsed_qualifying"),
        "resets": [{"at": r.get("at"), "reason": r.get("reason")} for r in resets],
    }


def gap_rows(run: Optional[Mapping]) -> list[dict]:
    """Every recorded gap day with its cause — a day with no capture is a gap,
    never a day with zero qualifying bets, and must be visible as such."""
    entries = (run or {}).get("gap_entries") or ()
    rows = []
    for entry in entries:
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            rows.append({"date": entry[0], "cause": entry[1] or "?"})
    return rows


def not_exercised(run: Optional[Mapping]) -> list[str]:
    """What the backtest lane cannot speak to — printed beside its numbers."""
    summaries = (run or {}).get("backtest_summaries") or {}
    for summary in summaries.values():
        if isinstance(summary, Mapping) and summary.get("not_exercised"):
            return [str(x) for x in summary["not_exercised"]]
    return []


# ── Streamlit render (the only Streamlit-bound entry point) ──────────────────
def render(*, run_path: Path = _RUN_ARTIFACT, report_dir: Path = _REPORT_DIR) -> None:
    import pandas as pd
    import streamlit as st

    from ui import _components as C

    run = load_run(run_path)
    state = deployment_line(run)

    st.markdown(
        C.app_bar(
            "Forward validation",
            "Backtest, paper and real execution — kept apart on purpose.",
        ),
        unsafe_allow_html=True,
    )

    if run is None:
        st.markdown(
            C.empty_state(
                "No Stage-5 run recorded",
                "Run <code>python -m scripts.paper_betting</code> to produce the "
                "forward-validation artifact and the dated reports.",
                hero=True,
            ),
            unsafe_allow_html=True,
        )
        return

    # The verdicts sit above every number, because every number is conditional
    # on them.
    st.markdown(
        C.kpi_rail([
            C.kpi("Model gate", state["model"],
                  tone=verdict_tone(state["model"] == "GO")),
            C.kpi("Forward release", "MET" if state["releasable"] else "NOT MET",
                  sub=state["gate"], tone=verdict_tone(state["releasable"])),
            C.kpi("Deployment", state["deployment"],
                  sub="no real money is ever staked", tone="warn"),
        ]),
        unsafe_allow_html=True,
    )
    if not state["releasable"]:
        st.warning(
            "At least one release gate is unmet, so this system issues **no "
            "real-money recommendations**. Paper tickets continue to accumulate "
            "so that forward evidence can exist to judge later. "
            "**\"No bet\" is a valid and expected output.**"
        )

    st.markdown(C.section("Forward-release gate"), unsafe_allow_html=True)
    rows = gate_rows(run.get("forward_gate"))
    if rows:
        st.dataframe(
            pd.DataFrame([{
                "criterion": r["name"],
                "": "PASS" if r["passed"] else "FAIL",
                # str-formatted: raw values mix str/float/None, which Arrow
                # cannot serialize (Streamlit logs a traceback and coerces).
                "observed": fmt_num(r["observed"]),
                "required": r["required"],
            } for r in rows]),
            hide_index=True, width="stretch",
        )

    # ── lane 1 ───────────────────────────────────────────────────────────────
    st.markdown(C.section("Lane 1 · historical backtest",
                          LANE_BLURB["backtest"]), unsafe_allow_html=True)
    bt = strategy_rows(run.get("backtest"))
    if bt:
        st.dataframe(
            pd.DataFrame([{
                "strategy": r["strategy"],
                "bets": r["bets"],
                "races": r["races"],
                "ROI": fmt_num(r["roi"], pct=True),
                "ROI 95% CI": fmt_interval(r["roi_ci"], pct=True),
                "hit rate": fmt_num(r["hit_rate"], pct=True),
                "mean CLV": fmt_num(r["clv_pct"], pct=True),
                "CLV 95% CI": fmt_interval(r["clv_ci"]),
                "A/E": fmt_num(r["ae"], 3),
                "max DD": fmt_num(r["max_dd"], pct=True),
                "worst streak": r["streak"],
            } for r in bt]),
            hide_index=True, width="stretch",
        )
        for row in bt:
            if row["caveat"]:
                st.caption(f"**{row['strategy']}** — {row['caveat']}")
        read = clv_verdict(run.get("backtest"))
        if read:
            st.markdown(
                f"**CLV read** — mean {fmt_num(read['mean'])} log "
                f"({fmt_num(read['pct'], pct=True)}), 95% CI "
                f"{fmt_interval(read['ci'])}. {read['text']}"
            )
    else:
        st.info("No backtest ledger in the last run.")

    limits = not_exercised(run)
    if limits:
        st.markdown("**What this lane does not exercise**")
        for item in limits:
            st.markdown(f"- {escape(item)}")

    sel = selection_rows(run)
    if sel:
        st.markdown(C.section("Selection protocol",
                              "tuned once, scored once"), unsafe_allow_html=True)
        st.markdown(
            f"- Strategies tried: **{sel['tried']}**\n"
            f"- Correction: **{sel['correction']}**, alpha {fmt_num(sel['alpha'], 3)} "
            f"→ adjusted **{fmt_num(sel['adjusted_alpha'], 5)}**\n"
            f"- Selected: **{sel['selected']}** `{json.dumps(sel['params'], sort_keys=True)}`\n"
            f"- Train {fmt_num(sel['train'])} · validation {fmt_num(sel['validation'])} "
            f"· **test {fmt_num(sel['test'])}**\n"
            f"- Test window scored at: {sel['scored_at'] or 'not scored'}"
            + (" (existing lock reused — not re-scored)" if sel["reused_lock"] else "")
        )
        if sel["note"]:
            st.caption(sel["note"])

    # ── lane 2 ───────────────────────────────────────────────────────────────
    st.markdown(C.section("Lane 2 · paper / shadow bets",
                          LANE_BLURB["paper"]), unsafe_allow_html=True)
    paper = run.get("paper") or {}
    st.markdown(
        f"- Decisions logged: **{paper.get('n_tickets', 0)}** "
        f"({paper.get('n_pass', 0)} PASS, {paper.get('n_candidates', 0)} candidate)\n"
        f"- Candidate tickets (paper wagers): **{paper.get('n_candidates', 0)}** "
        f"({paper.get('n_open', 0)} open, {paper.get('n_settled', 0)} settled)\n"
        f"- First: {paper.get('first_ticket') or '—'} · "
        f"latest: {paper.get('last_ticket') or '—'}"
    )

    st.markdown("**PASS / shadow observation** (disclosure evidence only — never a "
                "wager, never counted in the candidate figures above)")
    observe = run.get("observe") or {}
    st.markdown(
        f"- Priced-and-observable PASS tickets: **{paper.get('n_pass_priced', 0)}** "
        f"(**{paper.get('n_pass_observed', 0)}** observed to date)\n"
        f"- This cycle: {observe.get('observed', 0)} newly observed "
        f"({observe.get('already_observed', 0)} already observed, "
        f"{observe.get('still_open', 0)} pending a result, "
        f"{observe.get('unmeasurable_clv', 0)} settled with no closing price)"
    )
    if not paper.get("n_tickets"):
        st.info(
            "No paper tickets yet. The forward window has not started, so the "
            "forward gate cannot pass — this is the expected state until the "
            "live loop has run for the configured minimum."
        )

    win = window_row(run)
    st.markdown("**Forward-validation window**")
    if not win["running"]:
        st.info(
            "NOT STARTED. Live capture may be running independently of this "
            "clock — starting the formal 8-week window is a separate, "
            "deliberate action, not implied by capture running."
        )
    else:
        label = "RESET" if win["status"] == "reset" else "RUNNING"
        st.markdown(
            f"**{label}** since {win['window_start'] or '?'} — day "
            f"{win['days_elapsed']} of {win['required_days']} calendar days "
            f"({fmt_num(win['weeks_elapsed_qualifying'], 2)} gap-adjusted weeks "
            f"of qualifying capture)."
        )
        if win["resets"]:
            st.warning(
                f"Reset {len(win['resets'])} time(s) — each reset zeroed "
                "accumulated forward evidence because a frozen input changed:"
            )
            st.dataframe(pd.DataFrame(win["resets"]), hide_index=True,
                         width="stretch")

    gaps = gap_rows(run)
    st.markdown("**Capture history**")
    if gaps:
        st.warning(f"{len(gaps)} gap day(s) recorded — a day with no capture is "
                    "a gap, not a day with zero qualifying bets, and does not "
                    "count toward the forward gate.")
        st.dataframe(pd.DataFrame(gaps), hide_index=True, width="stretch")
    else:
        st.caption("No gap days recorded.")

    # ── lane 3 ───────────────────────────────────────────────────────────────
    st.markdown(C.section("Lane 3 · real-money execution",
                          LANE_BLURB["real"]), unsafe_allow_html=True)
    st.markdown(
        C.empty_state(
            "Structurally empty",
            "There is no code path that can fill this lane. "
            "<code>paper_only</code> is enforced by a CHECK constraint on the "
            "ticket table and by <code>TicketStore</code> refusing any other "
            "ticket.",
        ),
        unsafe_allow_html=True,
    )

    reports = latest_reports(report_dir)
    st.caption(
        "Dated reports: "
        + " · ".join(f"`{v}`" for v in reports.values() if v)
        + (f" · run generated {state['generated_at']}" if state["generated_at"] else "")
    )
