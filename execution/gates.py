"""The candidate gate: **PASS ("no bet") unless every condition is met**.

Stage 4 (``reports/calibration_audit_20260727.md``) issued MODEL NO-GO — on 904
untouched races the independent price-free line *loses* to the de-vigged pre-off
market by 0.186 race log-loss (95% CI [-0.225, -0.146]) with CLV at -13.4%. A
model in that state has no demonstrated edge, so the only defensible default
answer this module can give is **PASS**.

Three design rules follow directly, and none of them may be relaxed:

* **Affirmative evidence only.** A condition is met when the data proves it is
  met. Missing, unknown, unparseable or absent data is a PASS reason — never a
  waiver, never a default-true. Every branch below fails closed.
* **Consume decisions, never re-derive them.** ``models.predictor._race_ev_gate``
  already made the one race-level EV eligibility decision on the exact field the
  cache carries, and persisted it as ``ev_eligible`` / ``ev_gate``. A race stamped
  ineligible PASSes here with the race gate's own reasons prefixed ``race_gate:``.
  Re-deriving a friendlier answer from the same runners is precisely the failure
  mode Stage 4 was written to stop.
* **The price-free probability is the only probability that can qualify.** Stage 4
  measured the market-adjusted ``value_win_prob`` at 0.925 correlation with
  ``1/price`` — it is mostly a price echo, so it can never satisfy
  ``calibrated_probability`` on its own. Only ``value_win_prob_independent`` /
  ``model_win_prob_independent`` count, and the key actually used is recorded.

With the current NO-GO verdict ``model_validation`` always fails, so
:func:`evaluate_candidate` always returns PASS today. That is the correct
behaviour, not a bug, and there is deliberately no override, bypass or "advisory"
mode that issues a candidate anyway.

Public API
----------
    CONDITIONS, GateResult, evaluate_candidate, evaluate_race,
    race_uid, race_key, horse_key
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from utils.logger import get_logger

logger = get_logger(__name__)

CANDIDATE = "CANDIDATE"
PASS = "PASS"

#: Every condition the gate evaluates, in report order. A candidate is issued
#: only when every *enabled* condition is affirmatively met.
CONDITIONS: tuple[str, ...] = (
    "complete_card",
    "runner_history",
    "calibrated_probability",
    "reference_market",
    "executable_price",
    "source_health",
    "model_validation",
    "operating_cutoff",
    "field_size",
    "overround",
    "min_edge",
    "min_expected_value",
)

# condition -> the ``execution.gates`` config flag that enables it. ``None`` means
# the condition is a numeric threshold that is ALWAYS evaluated: a threshold has
# no "require" switch because turning a risk ceiling off is not a configuration
# choice this deployment offers.
_REQUIRE_FLAG: dict[str, Optional[str]] = {
    "complete_card": "require_complete_card",
    "runner_history": "require_runner_history",
    "calibrated_probability": "require_calibrated_probability",
    "reference_market": "require_reference_market",
    "executable_price": "require_executable_price",
    "source_health": "require_source_health",
    "model_validation": "require_model_validation",
    "field_size": None,
    "overround": None,
    "min_edge": None,
    "min_expected_value": None,
}

# The INDEPENDENT (price-free) win probability, in preference order. These are the
# only keys that can satisfy ``calibrated_probability``.
INDEPENDENT_PROB_KEYS: tuple[str, ...] = (
    "value_win_prob_independent",
    "model_win_prob_independent",
)
# Market-adjusted probabilities. Recorded for disclosure; never qualifying.
MARKET_ADJUSTED_PROB_KEYS: tuple[str, ...] = (
    "value_win_prob",
    "model_win_prob",
    "model_prob",
)

# The executable (takeable) price, in preference order — best board price first.
EXECUTABLE_ODDS_KEYS: tuple[str, ...] = (
    "executable_odds",
    "best_odds",
    "decimal_odds",
    "bet_price",
)
# Reference/consensus price used only to de-vig into a fair probability.
REFERENCE_ODDS_KEYS: tuple[str, ...] = ("reference_odds", "decimal_odds")

_BOOKMAKER_KEYS: tuple[str, ...] = (
    "bookmaker",
    "best_book",
    "executable_source",
    "book",
    "source",
)
_QUOTE_ODDS_KEYS: tuple[str, ...] = (
    "odds",
    "decimal_odds",
    "odds_decimal",
    "price",
    "offered_odds",
)
_FETCHED_AT_KEYS: tuple[str, ...] = ("fetched_at", "as_of", "quote_fetched_at")
_AGE_KEYS: tuple[str, ...] = ("age_seconds", "quote_age_seconds", "price_age_seconds")

# Sources that are not a bookmaker you could walk up to and take a price from.
# ``fused_consensus`` is the predictor's synthetic consensus line; betsp/timeform
# are historical SP / ratings feeds.
NON_EXECUTABLE_SOURCES: frozenset[str] = frozenset(
    {"fused_consensus", "consensus", "sp", "betsp", "timeform", "unknown", ""}
)


# ── value objects ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GateResult:
    """One runner's gate decision, with the full reasoning kept alongside it."""

    decision: str
    passed: tuple[str, ...] = ()
    pass_reasons: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    @property
    def is_candidate(self) -> bool:
        """True only for a clean CANDIDATE — a decision with no PASS reason."""
        return self.decision == CANDIDATE and not self.pass_reasons

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "is_candidate": self.is_candidate,
            "passed": list(self.passed),
            "pass_reasons": list(self.pass_reasons),
            "messages": list(self.messages),
            "detail": dict(self.detail),
        }


class _Ledger:
    """Accumulates per-condition outcomes so the result explains itself."""

    def __init__(self, detail: dict) -> None:
        self.detail = detail
        self.passed: list[str] = []
        self.reasons: list[str] = []
        self.messages: list[str] = []
        self.skipped: list[str] = []
        self.status: dict[str, str] = {}

    def met(self, condition: str) -> None:
        self.passed.append(condition)
        self.status[condition] = "met"

    def unmet(self, condition: str, reason: str, message: str) -> None:
        self.reasons.append(f"{condition}:{reason}")
        self.messages.append(message)
        self.status[condition] = "unmet"

    def unmet_many(
        self, condition: str, reasons: Sequence[str], message: str
    ) -> None:
        for reason in reasons or ("unspecified",):
            self.reasons.append(f"{condition}:{reason}")
        self.messages.append(message)
        self.status[condition] = "unmet"

    def skip(self, condition: str) -> None:
        self.skipped.append(condition)
        self.status[condition] = "skipped"

    def finish(self) -> GateResult:
        self.detail["conditions"] = dict(self.status)
        self.detail["skipped"] = list(self.skipped)
        if self.reasons:
            return GateResult(
                decision=PASS,
                passed=tuple(self.passed),
                pass_reasons=tuple(self.reasons),
                messages=tuple(self.messages),
                detail=self.detail,
            )
        message = (
            f"Candidate: every enabled condition met ({len(self.passed)} checked"
            + (f", {len(self.skipped)} skipped by config" if self.skipped else "")
            + ")."
        )
        return GateResult(
            decision=CANDIDATE,
            passed=tuple(self.passed),
            pass_reasons=(),
            messages=(message,),
            detail=self.detail,
        )


# ── small, fail-closed coercions ─────────────────────────────────────────────

def _f(value: Any) -> Optional[float]:
    """A finite float, or ``None``. NaN, NA, ``None`` and junk all collapse."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _s(value: Any) -> Optional[str]:
    """A non-empty stripped string, or ``None``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in ("nan", "nat", "none", "<na>"):
        return None
    return text


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` from a Mapping or an attribute off an object."""
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _first(obj: Any, keys: Sequence[str]) -> tuple[Optional[Any], Optional[str]]:
    """First non-``None`` value among ``keys`` plus the key it came from."""
    for key in keys:
        value = _get(obj, key)
        if value is not None:
            return value, key
    return None, None


def _parse_ts(value: Any) -> Optional[datetime]:
    """Parse a timestamp to an aware UTC datetime, or ``None``."""
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = _s(value)
        if text is None:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _now(now: Any = None) -> datetime:
    if now is not None:
        parsed = _parse_ts(now)
        if parsed is not None:
            return parsed
    from utils.timezone import now as _tz_now

    return _tz_now().astimezone(timezone.utc)


# ── identity helpers (shared with execution.tickets) ─────────────────────────
# Both delegate to execution.snapshots, which owns the cross-bookmaker key
# definitions (race_time over race_id, normalised horse name over horse_id).
# A gate and a ticket that keyed races differently from the snapshot store would
# not be able to join a decision back to the price it was made on — which is the
# whole audit trail.

def race_uid(race: Mapping) -> str:
    """This race's cross-bookmaker identity key."""
    from execution.snapshots import race_uid as _uid

    return _uid(race.get("race_id"), race.get("race_time"), race.get("venue"))


def race_key(race: Mapping) -> str:
    """This race's venue-qualified, minute-precision physical identity (Stage 20).

    ``race_uid`` above drops venue once an off-time parses, so two different
    venues at the same minute collide (B5). ``race_key`` never does — see
    ``execution.snapshots.race_key`` — and is what new duplicate/correlation
    checks and quote lookups should key on going forward.
    """
    from execution.snapshots import race_key as _key

    return _key(race.get("venue"), race.get("race_time"))


def horse_key(runner: Mapping) -> str:
    """This runner's cross-bookmaker identity key."""
    from execution.snapshots import horse_key as _key

    return _key(
        runner.get("horse_name") or runner.get("name"), runner.get("horse_id")
    )


# ── inputs the conditions read ───────────────────────────────────────────────

def _prob(value: Any) -> Optional[float]:
    """A probability strictly inside (0,1), or ``None``.

    Text is refused outright rather than coerced. Everywhere else in this module
    ``_f`` parses odds and thresholds that legitimately arrive as strings from
    JSON and CSV, but the model's own probability is the single number the whole
    bet rests on: if it reaches us as ``"0.3"`` then the layer that produced it is
    not the calibrated layer we validated, and the honest reading of an
    unrecognised producer is *no probability*, not a probability we hope is right.
    Numeric types are left to ``_f`` so numpy scalars still parse.
    """
    if isinstance(value, (str, bytes, bytearray)):
        return None
    out = _f(value)
    return out if out is not None and 0.0 < out < 1.0 else None


def _independent_prob(runner: Mapping) -> tuple[Optional[float], Optional[str]]:
    """The price-free probability and the key it came from, if valid in (0,1)."""
    for key in INDEPENDENT_PROB_KEYS:
        value = _prob(runner.get(key))
        if value is not None:
            return value, key
    return None, None


def _market_adjusted_prob(runner: Mapping) -> tuple[Optional[float], Optional[str]]:
    for key in MARKET_ADJUSTED_PROB_KEYS:
        value = _prob(runner.get(key))
        if value is not None:
            return value, key
    return None, None


def _runners(race: Mapping) -> list[Mapping]:
    """The COMPLETE declared field (audit req 2) — never the display top-N."""
    runners = race.get("runners")
    if isinstance(runners, Sequence) and not isinstance(runners, (str, bytes)):
        out = [r for r in runners if isinstance(r, Mapping)]
        if out:
            return out
    out = []
    for key in ("selections", "excluded_low_odds"):
        node = race.get(key)
        if isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            out.extend(r for r in node if isinstance(r, Mapping))
    return out


def _reference_book(race: Mapping) -> dict:
    """Reference-book state: completeness, book sum and source.

    Prefers the persisted ``ev_gate`` (the predictor's own decision). Only when
    the race carries no such claim is completeness derived from the runners
    themselves — and then it is an affirmative demonstration (every runner priced
    by the reference line), never an assumption.

    ``ev_gate['reference_overround']`` is the **excess** (``sum(1/d) - 1``), the
    repo-wide convention shared with ``backtest.metrics.devig`` and
    ``models.value.devig_field``. ``execution.gates.max_overround`` is a **book
    sum** (1.25 == a 25% book). The conversion is always ``book_sum = 1 + excess``
    — never the other way round, because reading a large excess as a book sum
    would turn a pathological book into a passing one.
    """
    gate = race.get("ev_gate")
    gate = gate if isinstance(gate, Mapping) else {}

    complete = gate.get("reference_book_complete")
    complete = complete if isinstance(complete, bool) else None
    excess = _f(gate.get("reference_overround"))
    source = _s(gate.get("reference_source"))

    if complete is None or excess is None:
        runners = _runners(race)
        if runners:
            refs = [_f(_first(r, REFERENCE_ODDS_KEYS)[0]) for r in runners]
            priced = [d for d in refs if d is not None and d > 1.0]
            derived_complete = len(priced) == len(refs) and bool(refs)
            if complete is None:
                complete = derived_complete
            if excess is None and derived_complete:
                excess = sum(1.0 / d for d in priced) - 1.0
        if source is None:
            for r in _runners(race):
                source = _s(r.get("reference_source"))
                if source:
                    break

    return {
        "complete": complete,
        "overround_excess": excess,
        "book_sum": (1.0 + excess) if excess is not None else None,
        "source": source,
    }


def _quote_view(quote: Any, runner: Mapping, ts: datetime) -> dict:
    """Normalise a snapshot Quote (or the runner's own recorded price).

    A supplied quote always wins: it is the point-in-time record of what was
    actually offered. Falling back to the runner's cached price is only for
    surfaces that have no snapshot store wired in, and it still has to carry a
    ``fetched_at`` — an age we cannot compute is an age we must not trust.
    """
    origin = "quote" if quote is not None else "runner"
    src = quote if quote is not None else runner

    odds = _f(_first(src, _QUOTE_ODDS_KEYS)[0])
    if odds is None and quote is None:
        odds = _f(_first(runner, EXECUTABLE_ODDS_KEYS)[0])

    bookmaker = _s(_first(src, _BOOKMAKER_KEYS)[0])
    if bookmaker is None and quote is not None:
        bookmaker = _s(_first(runner, _BOOKMAKER_KEYS)[0])

    fetched_raw, _ = _first(src, _FETCHED_AT_KEYS)
    fetched = _parse_ts(fetched_raw)

    age = _f(_first(src, _AGE_KEYS)[0])
    if age is None and fetched is not None:
        age = (ts - fetched).total_seconds()

    # The snapshot store stamps its own staleness verdict on every Quote. We
    # honour it even when our own arithmetic would call the price fresh: the
    # store knows the as_of it was read for, and this gate does not.
    flagged = _get(src, "is_stale")
    if not isinstance(flagged, bool):
        flagged = _get(src, "stale")
    flagged = flagged if isinstance(flagged, bool) else None

    return {
        "origin": origin if odds is not None else None,
        "odds": odds,
        "bookmaker": bookmaker,
        "market_type": _s(_get(src, "market_type")),
        "fetched_at": fetched.isoformat() if fetched else _s(fetched_raw),
        "age_seconds": round(age, 3) if age is not None else None,
        "is_stale": flagged,
        "ew_places": _f(_get(src, "ew_places")),
        "ew_reduction": _f(_get(src, "ew_reduction")),
    }


def _contributing_sources(race: Mapping, runner: Mapping, quote_view: Mapping) -> list[str]:
    """Every named source whose data this decision leans on."""
    names: list[str] = []
    gate = race.get("ev_gate")
    if isinstance(gate, Mapping):
        listed = gate.get("price_sources")
        if isinstance(listed, Sequence) and not isinstance(listed, (str, bytes)):
            names.extend(str(s) for s in listed)
        names.append(str(gate.get("reference_source") or ""))
    names.append(str(runner.get("reference_source") or ""))
    names.append(str(quote_view.get("bookmaker") or ""))

    out: list[str] = []
    for name in names:
        cleaned = name.strip().lower()
        if cleaned and cleaned not in NON_EXECUTABLE_SOURCES and cleaned not in out:
            out.append(cleaned)
    return out


def _health_records(source_health: Any) -> Optional[Mapping]:
    """The health map to consult, loading from disk when not supplied."""
    if source_health is not None:
        return source_health if isinstance(source_health, Mapping) else None
    try:
        from utils.source_health import get_health

        records = get_health()
    except Exception as exc:  # noqa: BLE001 — unreadable telemetry => unknown
        logger.warning("gates: source health unavailable (%s) — treating as unknown", exc)
        return None
    return records if isinstance(records, Mapping) else None


def _health_ok(record: Any, max_age: Optional[float]) -> tuple[bool, str]:
    """Is one source healthy AND fresh? Returns ``(ok, reason_fragment)``."""
    if not isinstance(record, Mapping):
        return False, "no_record"
    ok = record.get("ok")
    if not isinstance(ok, bool):
        ok = _s(record.get("status")) == "ok"
    if not ok:
        status = _s(record.get("status")) or "unknown"
        return False, f"status={status}"
    age = _f(record.get("age_seconds"))
    if age is None:
        return False, "age_unknown"
    if max_age is not None and age > max_age:
        return False, f"stale={int(age)}s>{int(max_age)}s"
    return True, ""


# ── the gate ─────────────────────────────────────────────────────────────────

def evaluate_candidate(
    runner: Mapping,
    race: Mapping,
    *,
    cfg,
    model_verdict,
    quote: Any = None,
    source_health: Any = None,
    now: Any = None,
) -> GateResult:
    """Decide one runner. Returns ``PASS`` unless every enabled condition is met.

    ``runner`` is a predictor runner dict, ``race`` its race dict (carrying
    ``ev_eligible`` / ``ev_gate``). ``quote`` is an optional point-in-time
    snapshot quote; ``source_health`` an optional ``{source: record}`` map (read
    from :mod:`utils.source_health` when omitted).
    """
    gates = cfg.gates
    ts = _now(now)

    book = _reference_book(race)
    qv = _quote_view(quote, runner, ts)
    verdict_label = _s(getattr(model_verdict, "verdict_label", None))

    detail: dict[str, Any] = {
        "evaluated_at": ts.isoformat(),
        "race_uid": race_uid(race),
        "race_key": race_key(race),
        "horse_key": horse_key(runner),
        "venue": _s(race.get("venue")),
        "race_time": _s(race.get("race_time")),
        "paper_only": bool(getattr(cfg, "paper_only", True)),
        "model_verdict": verdict_label,
        "ev_eligible": race.get("ev_eligible"),
        "reference_book_complete": book["complete"],
        "reference_overround_excess": book["overround_excess"],
        "reference_book_sum": book["book_sum"],
        "reference_source": book["source"],
        "quote_origin": qv["origin"],
        "executable_odds": qv["odds"],
        "bookmaker": qv["bookmaker"],
        "market_type": qv["market_type"],
        "quote_fetched_at": qv["fetched_at"],
        "quote_age_seconds": qv["age_seconds"],
        "quote_is_stale": qv["is_stale"],
        "ew_places": qv["ew_places"],
        "ew_reduction": qv["ew_reduction"],
        "max_quote_age_seconds": _f(getattr(cfg.snapshots, "max_age_seconds", None)),
        "thresholds": {
            "min_edge": _f(gates.min_edge),
            "min_expected_value": _f(gates.min_expected_value),
            "min_field_size": gates.min_field_size,
            "max_overround": _f(gates.max_overround),
        },
    }

    # ── 0. the race-level EV gate is authoritative ───────────────────────────
    # A race the predictor stamped ineligible has already PASSed on the exact
    # field the cache carries. We do not second-guess it, and we do not let a
    # disabled require_* flag waive it: this is a *decision*, not a condition.
    if race.get("ev_eligible") is False:
        gate = race.get("ev_gate")
        gate = gate if isinstance(gate, Mapping) else {}
        raw = [str(r) for r in (gate.get("reasons") or [])] or ["unspecified"]
        reasons = tuple(f"race_gate:{r}" for r in raw)
        detail["conditions"] = {c: "not_evaluated" for c in CONDITIONS}
        detail["skipped"] = []
        detail["race_gate_reasons"] = raw
        return GateResult(
            decision=PASS,
            passed=(),
            pass_reasons=reasons,
            messages=(
                "No bet: the race-level EV gate ruled this card ineligible ("
                + "; ".join(raw)
                + ").",
            ),
            detail=detail,
        )

    led = _Ledger(detail)

    def enabled(condition: str) -> bool:
        flag = _REQUIRE_FLAG[condition]
        if flag is None:
            return True
        if bool(getattr(gates, flag, True)):
            return True
        led.skip(condition)
        return False

    # ── 1. complete_card ─────────────────────────────────────────────────────
    if enabled("complete_card"):
        if race.get("ev_eligible") is not True:
            led.unmet(
                "complete_card",
                "ev_eligible_unknown",
                "No bet: the race carries no EV-eligibility stamp, so the card "
                "cannot be shown to be complete.",
            )
        elif book["complete"] is not True:
            led.unmet(
                "complete_card",
                "reference_book_incomplete",
                "No bet: the race-level gate did not report a complete reference "
                "book for this card.",
            )
        else:
            led.met("complete_card")

    # ── 2. runner_history ────────────────────────────────────────────────────
    if enabled("runner_history"):
        supported = runner.get("value_supported")
        if isinstance(supported, bool):
            if supported:
                led.met("runner_history")
            else:
                led.unmet(
                    "runner_history",
                    "no_prior_form",
                    "No bet: the runner has no prior in-window form, so the model "
                    "is scoring it at base rate.",
                )
        elif "historical_place_rate" in runner:
            if _f(runner.get("historical_place_rate")) is not None:
                led.met("runner_history")
            else:
                led.unmet(
                    "runner_history",
                    "no_prior_form",
                    "No bet: the runner's trailing place rate is null — no prior "
                    "in-window form.",
                )
        else:
            # Unknown support is not support. models.value assumes True for
            # legacy caches; a *bet* gate may not.
            led.unmet(
                "runner_history",
                "unknown",
                "No bet: the runner's form support is unknown.",
            )

    # ── 3. calibrated_probability (INDEPENDENT, price-free only) ─────────────
    prob, prob_key = _independent_prob(runner)
    adj_prob, adj_key = _market_adjusted_prob(runner)
    detail["model_prob"] = prob
    detail["probability_key"] = prob_key
    detail["market_adjusted_prob"] = adj_prob
    detail["market_adjusted_probability_key"] = adj_key
    if enabled("calibrated_probability"):
        if prob is not None:
            led.met("calibrated_probability")
        elif adj_prob is not None:
            led.unmet(
                "calibrated_probability",
                "market_adjusted_only",
                "No bet: only a market-adjusted probability is available. Stage 4 "
                "measured it at 0.925 correlation with 1/price — a price echo "
                "cannot be the basis of a bet against that same price.",
            )
        else:
            led.unmet(
                "calibrated_probability",
                "missing",
                "No bet: no independent price-free probability in (0,1) is "
                "available for this runner.",
            )

    # ── 4. reference_market ──────────────────────────────────────────────────
    if enabled("reference_market"):
        if book["complete"] is not True:
            led.unmet(
                "reference_market",
                "incomplete_reference_book",
                "No bet: the reference book is incomplete, so there is nothing "
                "coherent to de-vig against.",
            )
        elif book["book_sum"] is None:
            led.unmet(
                "reference_market",
                "overround_unavailable",
                "No bet: the reference book's overround is unavailable, so the "
                "fair line is undefined.",
            )
        else:
            led.met("reference_market")

    # ── 5. executable_price ──────────────────────────────────────────────────
    max_age = _f(getattr(cfg.snapshots, "max_age_seconds", None))
    if enabled("executable_price"):
        if qv["odds"] is None or qv["odds"] <= 1.0:
            led.unmet(
                "executable_price",
                "missing",
                "No bet: no executable price is available for this runner.",
            )
        elif qv["bookmaker"] is None or qv["bookmaker"].lower() in NON_EXECUTABLE_SOURCES:
            led.unmet(
                "executable_price",
                "no_named_bookmaker",
                "No bet: the price is not attributable to a named bookmaker — a "
                "consensus line is not a price you can take.",
            )
        elif qv["is_stale"] is True:
            led.unmet(
                "executable_price",
                "flagged_stale",
                "No bet: the snapshot store flagged this quote as stale for the "
                "decision instant it was read for.",
            )
        elif qv["age_seconds"] is None:
            led.unmet(
                "executable_price",
                "age_unknown",
                "No bet: the price carries no fetch timestamp, so its age cannot "
                "be established.",
            )
        elif qv["age_seconds"] < 0:
            led.unmet(
                "executable_price",
                "timestamp_in_future",
                "No bet: the price is timestamped in the future — a later price "
                "may never stand in for the price available now.",
            )
        elif max_age is not None and qv["age_seconds"] > max_age:
            led.unmet(
                "executable_price",
                f"stale={int(qv['age_seconds'])}s>{int(max_age)}s",
                f"No bet: the price is {int(qv['age_seconds'])}s old, beyond the "
                f"{int(max_age)}s executable window.",
            )
        else:
            led.met("executable_price")

    # ── 6. source_health ─────────────────────────────────────────────────────
    sources = _contributing_sources(race, runner, qv)
    detail["contributing_sources"] = sources
    if enabled("source_health"):
        records = _health_records(source_health)
        max_source_age = _f(getattr(cfg.safeguards, "max_source_age_seconds", None))
        if records is None:
            led.unmet(
                "source_health",
                "unavailable",
                "No bet: source-health telemetry is unavailable, so freshness "
                "cannot be established.",
            )
        elif not sources:
            led.unmet(
                "source_health",
                "no_contributing_sources",
                "No bet: no named contributing source could be identified for "
                "this runner.",
            )
        else:
            problems: list[str] = []
            states: dict[str, str] = {}
            for name in sources:
                ok, why = _health_ok(records.get(name), max_source_age)
                states[name] = "ok" if ok else (why or "unhealthy")
                if not ok:
                    problems.append(f"{name}:{why or 'unhealthy'}")
            detail["source_health"] = states
            if problems:
                led.unmet_many(
                    "source_health",
                    problems,
                    "No bet: contributing sources are unhealthy or stale ("
                    + "; ".join(problems)
                    + ").",
                )
            else:
                led.met("source_health")

    # ── 7. model_validation — the Stage-4 verdict, obeyed ────────────────────
    if enabled("model_validation"):
        if model_verdict is None:
            led.unmet(
                "model_validation",
                "verdict_unavailable",
                "No bet: no Stage-4 model verdict could be read.",
            )
        elif bool(getattr(model_verdict, "go", False)):
            led.met("model_validation")
        else:
            verdict_reasons = [
                str(r) for r in (getattr(model_verdict, "reasons", ()) or ())
            ]
            led.unmet_many(
                "model_validation",
                [verdict_label or "NO-GO", *verdict_reasons],
                "No bet: the model carries a "
                f"{verdict_label or 'NO-GO'} verdict"
                + (f" ({'; '.join(verdict_reasons)})" if verdict_reasons else "")
                + ". A model that does not beat the de-vigged market may not be "
                "bet, in paper or otherwise.",
            )

    # ── 7b. operating_cutoff — real wall-clock decision cutoff (Stage 20 / B6) ─
    # ``evaluate_candidate``'s own ``ts`` (evaluated_at, real wall-clock) is the
    # decision instant, never a day-truncated or file-mtime stand-in. A race
    # already at, past, or inside the configured pre-off buffer must PASS here,
    # not be silently ticketed as though the market were still open —
    # previously ``execution.safeguards.block_started_races`` existed but was
    # never called by the live loop (GAP-A), so an already-off race was ticketed
    # in full. An unreadable/missing off-time fails closed exactly like every
    # other condition in this module: an unknown off-time cannot be shown to be
    # in the future.
    buffer_s = _f(getattr(cfg.safeguards, "started_race_buffer_seconds", None)) or 0.0
    race_time_dt = _parse_ts(race.get("race_time"))
    seconds_to_off = (
        (race_time_dt - ts).total_seconds() if race_time_dt is not None else None
    )
    detail["seconds_to_off"] = round(seconds_to_off, 1) if seconds_to_off is not None else None
    detail["started_race_buffer_seconds"] = buffer_s
    if bool(getattr(cfg.safeguards, "block_started_races", True)):
        if seconds_to_off is None:
            led.unmet(
                "operating_cutoff",
                "race_time_unreadable",
                "No bet: the race's off-time is missing or unreadable, so it "
                "cannot be shown to be before the decision cutoff.",
            )
        elif seconds_to_off <= buffer_s:
            led.unmet(
                "operating_cutoff",
                f"started_or_within_buffer:{seconds_to_off:.0f}s<={buffer_s:.0f}s",
                f"No bet: the race is {seconds_to_off:.0f}s from off, at or "
                f"inside the {buffer_s:.0f}s pre-off buffer — it counts as "
                "started.",
            )
        else:
            led.met("operating_cutoff")
    else:
        led.skip("operating_cutoff")

    # ── 8. field_size ────────────────────────────────────────────────────────
    field_size = _f(race.get("field_size"))
    if field_size is None:
        runners = _runners(race)
        field_size = float(len(runners)) if runners else None
    detail["field_size"] = int(field_size) if field_size is not None else None
    if enabled("field_size"):
        if field_size is None:
            led.unmet(
                "field_size",
                "unknown",
                "No bet: the declared field size is unknown.",
            )
        elif field_size < float(gates.min_field_size):
            led.unmet(
                "field_size",
                f"{int(field_size)}<{int(gates.min_field_size)}",
                f"No bet: {int(field_size)} runners is below the "
                f"{int(gates.min_field_size)}-runner minimum, where de-vig and "
                "Rule 4 dominate the edge.",
            )
        else:
            led.met("field_size")

    # ── 9. overround ─────────────────────────────────────────────────────────
    book_sum = book["book_sum"]
    max_overround = _f(gates.max_overround)
    if enabled("overround"):
        if book_sum is None:
            led.unmet(
                "overround",
                "unknown",
                "No bet: the reference book's overround is unknown.",
            )
        elif max_overround is not None and book_sum > max_overround:
            led.unmet(
                "overround",
                f"{book_sum:.4f}>{max_overround:.4f}",
                f"No bet: the reference book sums to {book_sum:.3f}, looser than "
                f"the {max_overround:.3f} ceiling.",
            )
        else:
            led.met("overround")

    # ── 10/11. the numeric edge and EV thresholds ────────────────────────────
    # Both are computed from the INDEPENDENT probability only. The fair line is
    # the reference book de-vigged proportionally (fair_i = (1/ref_i)/book_sum),
    # matching models.value.devig_field; EV uses the EXECUTABLE price, which is
    # what actually pays.
    ref_odds = _f(_first(runner, REFERENCE_ODDS_KEYS)[0])
    market_prob = (1.0 / ref_odds) if ref_odds and ref_odds > 1.0 else None
    fair_prob = (
        market_prob / book_sum
        if market_prob is not None and book_sum and book_sum > 0
        else None
    )
    edge = (prob - fair_prob) if (prob is not None and fair_prob is not None) else None
    exec_odds = qv["odds"]
    expected_value = (
        prob * exec_odds - 1.0
        if (prob is not None and exec_odds is not None and exec_odds > 1.0)
        else None
    )
    detail.update(
        {
            "reference_odds": ref_odds,
            "market_prob": market_prob,
            "fair_prob": fair_prob,
            "fair_odds": (1.0 / fair_prob) if fair_prob else None,
            "edge": edge,
            "expected_value": expected_value,
        }
    )

    min_edge = _f(gates.min_edge)
    if enabled("min_edge"):
        if edge is None:
            led.unmet(
                "min_edge",
                "unavailable",
                "No bet: the edge over the de-vigged market cannot be computed.",
            )
        elif min_edge is not None and edge < min_edge:
            led.unmet(
                "min_edge",
                f"{edge:.4f}<{min_edge:.4f}",
                f"No bet: the edge over the fair line is {edge:+.4f}, below the "
                f"{min_edge:.4f} minimum.",
            )
        else:
            led.met("min_edge")

    min_ev = _f(gates.min_expected_value)
    if enabled("min_expected_value"):
        if expected_value is None:
            led.unmet(
                "min_expected_value",
                "unavailable",
                "No bet: expected value cannot be computed without an "
                "independent probability and an executable price.",
            )
        elif min_ev is not None and expected_value < min_ev:
            led.unmet(
                "min_expected_value",
                f"{expected_value:.4f}<{min_ev:.4f}",
                f"No bet: expected value is {expected_value:+.4f} per unit, below "
                f"the {min_ev:.4f} minimum.",
            )
        else:
            led.met("min_expected_value")

    result = led.finish()
    if result.decision == PASS:
        logger.debug(
            "gates: PASS %s / %s — %s",
            detail["race_uid"],
            detail["horse_key"],
            "; ".join(result.pass_reasons),
        )
    return result


def evaluate_race(
    race: Mapping,
    *,
    cfg,
    model_verdict,
    quotes: Any = None,
    source_health: Any = None,
    now: Any = None,
) -> list[GateResult]:
    """Evaluate every runner in a race's COMPLETE declared field.

    ``quotes`` may be a ``{horse_key: quote}`` mapping or a callable taking the
    runner dict — both optional. ``source_health`` is read once and reused so a
    card of runners produces one consistent view of the world.
    """
    records = _health_records(source_health)
    ts = _now(now)
    results: list[GateResult] = []
    for runner in _runners(race):
        if callable(quotes):
            quote = quotes(runner)
        elif isinstance(quotes, Mapping):
            quote = quotes.get(horse_key(runner))
        else:
            quote = None
        results.append(
            evaluate_candidate(
                runner,
                race,
                cfg=cfg,
                model_verdict=model_verdict,
                quote=quote,
                source_health=records,
                now=ts,
            )
        )
    return results
