"""The disclosure ticket: a full record of **every** candidate *and* every PASS.

Stage 4 issued MODEL NO-GO, so today every ticket this module writes is a PASS.
That is exactly why the PASS is recorded at all: a system that logs only the bets
it wanted to make cannot be audited, and a "no bet" with its reasons attached is
the most informative row this deployment currently produces. Every ticket carries
the same disclosure set regardless of decision — the price, its age, which
probability was used and whether it was price-free, the fair line, the edge, the
EV, the ceiling on stake, the data quality, and the validation state.

Two independent guarantees keep a real-money ticket out of the store:

1. :meth:`TicketStore.issue` refuses a ticket whose ``paper_only`` is False, and
2. the ``paper_tickets`` table carries ``CHECK (paper_only = 1)``, so a row
   written around the Python API by hand is rejected by SQLite itself.

Neither is redundant: the first explains itself, the second cannot be bypassed.
``Ticket.paper_only`` is decided once, by
:func:`execution.config.real_money_enabled` — the single function permitted to
answer "may this be real money?" — and with the Stage-4 NO-GO in place it can
only answer no.

Public API
----------
    Ticket, RealMoneyTicketRefused, make_ticket_id, build_ticket, TicketStore
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

from backtest import metrics
from execution.config import ExecutionConfig, real_money_enabled
# The coercion helpers are imported rather than re-declared on purpose: a ticket
# must render exactly the values the gate decided on, and two independent "is this
# a finite number?" rules is how a report starts disagreeing with its own gate.
from execution.gates import (
    GateResult,
    _f,
    _first,
    _now,
    _parse_ts,
    _s,
    horse_key,
    race_key,
    race_uid,
)
from utils.logger import get_logger
from utils.storage import DEFAULT_DB_PATH
from utils.storage.migrations import apply_migrations
from utils.storage.pool import ConnectionPool

logger = get_logger(__name__)

CANDIDATE = "CANDIDATE"
PASS = "PASS"

#: Columns the ticket occupies in ``paper_tickets`` — exactly ``to_row()``'s keys.
TICKET_COLUMNS: tuple[str, ...] = (
    "ticket_id",
    "issued_at",
    "as_of",
    "race_uid",
    "race_key",
    "race_id",
    "race_time",
    "venue",
    "horse_key",
    "horse_id",
    "horse_name",
    "bookmaker",
    "market_type",
    "bet_type",
    "offered_odds",
    "quote_fetched_at",
    "quote_age_seconds",
    "model_prob",
    "market_adjusted_prob",
    "fair_odds",
    "market_prob",
    "edge",
    "expected_value",
    "max_stake",
    "stake",
    "ew_places",
    "ew_reduction",
    "data_quality",
    "validation_state",
    "decision",
    "pass_reasons",
    "reasons_passed",
    "paper_only",
    "model_verdict",
    "forward_gate_state",
    "model_content_hash",
    "feature_schema_version",
    "config_hash",
    "prediction_cycle_id",
    "provenance_complete",
)

#: Written only by :meth:`TicketStore.settle`.
SETTLEMENT_COLUMNS: tuple[str, ...] = (
    "settled_at",
    "outcome",
    "returns",
    "profit",
    "closing_odds",
    "clv_pct",
    "settlement_detail",
)

# Only used when a hand-built GateResult carries no executable price in its detail.
_QUOTE_ODDS_KEYS: tuple[str, ...] = (
    "odds",
    "decimal_odds",
    "odds_decimal",
    "price",
    "offered_odds",
    "best_odds",
)

_OUTCOMES = {
    "won": "won",
    "win": "won",
    "winner": "won",
    "lost": "lost",
    "lose": "lost",
    "loss": "lost",
    "loser": "lost",
    "void": "void",
    "voided": "void",
    "non_runner": "void",
    "nr": "void",
    # An each-way ticket whose place leg won but whose win leg did not
    # (execution.settlement.STATUS_PLACE). Distinct from "won": the win
    # probability being scored (execution.evaluation, _forward_ledger) was
    # wrong even though the ticket returned money.
    "place": "placed",
    "placed": "placed",
}


class RealMoneyTicketRefused(RuntimeError):
    """Raised when a ticket that is not paper-only reaches the store.

    Stage 4's MODEL NO-GO makes this unreachable through
    :func:`build_ticket`; it exists so a hand-built ticket cannot slip past.
    """


# ── the ticket ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Ticket:
    """One fully disclosed decision — a candidate or, far more often, a PASS."""

    ticket_id: str
    issued_at: str
    as_of: Optional[str]
    race_uid: str
    race_id: Optional[str]
    race_time: Optional[str]
    venue: Optional[str]
    race_key: Optional[str]
    horse_key: str
    horse_id: Optional[str]
    horse_name: Optional[str]
    bookmaker: Optional[str]
    market_type: str
    bet_type: str
    offered_odds: Optional[float]
    quote_fetched_at: Optional[str]
    quote_age_seconds: Optional[float]
    # The INDEPENDENT price-free probability — the only one a decision may use.
    model_prob: Optional[float]
    # Recorded only when a market-adjusted probability was actually involved, so a
    # reader can always tell a price-free number from a price echo.
    market_adjusted_prob: Optional[float]
    fair_odds: Optional[float]
    market_prob: Optional[float]
    edge: Optional[float]
    expected_value: Optional[float]
    max_stake: Optional[float]
    stake: Optional[float]
    ew_places: Optional[int]
    ew_reduction: Optional[float]
    data_quality: str
    validation_state: str
    decision: str
    pass_reasons: tuple[str, ...]
    reasons_passed: tuple[str, ...]
    paper_only: bool = True
    model_verdict: Optional[str] = None
    forward_gate_state: Optional[str] = None
    # Stage 20 (B7): what produced this decision, so a fresh ticket can be
    # traced back to an exact served bundle, feature schema, config and
    # prediction cycle. Missing/None on any of the four is the honest reading
    # for a ticket built before this contract existed (from_row leaves them
    # None rather than fabricating a value) — never coerced to a fake match.
    model_content_hash: Optional[str] = None
    feature_schema_version: Optional[str] = None
    config_hash: Optional[str] = None
    prediction_cycle_id: Optional[str] = None
    # True only when all four of the above were affirmatively supplied. A
    # forward-gate eligibility check reads this, never re-derives it, so the
    # "missing provenance excludes eligibility" rule lives in exactly one place.
    provenance_complete: bool = False

    # ── views ────────────────────────────────────────────────────────────────
    @property
    def is_candidate(self) -> bool:
        return self.decision == CANDIDATE

    def to_dict(self) -> dict:
        out = asdict(self)
        out["pass_reasons"] = list(self.pass_reasons)
        out["reasons_passed"] = list(self.reasons_passed)
        out["is_candidate"] = self.is_candidate
        return out

    def to_row(self) -> dict:
        """Row form — one key per ``paper_tickets`` column, SQLite-native types."""
        row = {name: getattr(self, name) for name in TICKET_COLUMNS}
        row["pass_reasons"] = json.dumps(list(self.pass_reasons))
        row["reasons_passed"] = json.dumps(list(self.reasons_passed))
        row["paper_only"] = 1 if self.paper_only else 0
        row["provenance_complete"] = 1 if self.provenance_complete else 0
        row["ew_places"] = int(self.ew_places) if self.ew_places is not None else None
        for key in (
            "offered_odds",
            "quote_age_seconds",
            "model_prob",
            "market_adjusted_prob",
            "fair_odds",
            "market_prob",
            "edge",
            "expected_value",
            "max_stake",
            "stake",
            "ew_reduction",
        ):
            row[key] = _f(row[key])
        return row

    @classmethod
    def from_row(cls, row: Mapping) -> "Ticket":
        get = row.get if isinstance(row, Mapping) else (lambda k, d=None: row[k])
        return cls(
            ticket_id=str(get("ticket_id") or ""),
            issued_at=str(get("issued_at") or ""),
            as_of=_s(get("as_of")),
            race_uid=str(get("race_uid") or ""),
            race_id=_s(get("race_id")),
            race_time=_s(get("race_time")),
            venue=_s(get("venue")),
            race_key=_s(get("race_key")),
            horse_key=str(get("horse_key") or ""),
            horse_id=_s(get("horse_id")),
            horse_name=_s(get("horse_name")),
            bookmaker=_s(get("bookmaker")),
            market_type=str(get("market_type") or "win"),
            bet_type=str(get("bet_type") or "win"),
            offered_odds=_f(get("offered_odds")),
            quote_fetched_at=_s(get("quote_fetched_at")),
            quote_age_seconds=_f(get("quote_age_seconds")),
            model_prob=_f(get("model_prob")),
            market_adjusted_prob=_f(get("market_adjusted_prob")),
            fair_odds=_f(get("fair_odds")),
            market_prob=_f(get("market_prob")),
            edge=_f(get("edge")),
            expected_value=_f(get("expected_value")),
            max_stake=_f(get("max_stake")),
            stake=_f(get("stake")),
            ew_places=(int(get("ew_places")) if _f(get("ew_places")) is not None else None),
            ew_reduction=_f(get("ew_reduction")),
            data_quality=str(get("data_quality") or "unknown"),
            validation_state=str(get("validation_state") or ""),
            decision=str(get("decision") or PASS),
            pass_reasons=_decode_reasons(get("pass_reasons")),
            reasons_passed=_decode_reasons(get("reasons_passed")),
            paper_only=bool(get("paper_only")),
            model_verdict=_s(get("model_verdict")),
            forward_gate_state=_s(get("forward_gate_state")),
            model_content_hash=_s(get("model_content_hash")),
            feature_schema_version=_s(get("feature_schema_version")),
            config_hash=_s(get("config_hash")),
            prediction_cycle_id=_s(get("prediction_cycle_id")),
            provenance_complete=bool(get("provenance_complete")),
        )

    # ── human display ────────────────────────────────────────────────────────
    def render_lines(self, currency: str = "€") -> list[str]:
        """Every disclosure field, one line each, for a report or the UI.

        Nothing is elided for brevity: a reader must be able to see the price's
        age, which probability drove the number, and what the model verdict was,
        without following a link. While the ticket is paper-only every line
        carries the ``[PAPER]`` marker, so no single line can be screenshotted
        and read as a real-money recommendation.
        """
        mark = "[PAPER] " if self.paper_only else ""
        race_bits = " ".join(
            b for b in (self.venue, self.race_time) if b
        ) or self.race_uid or "unknown race"
        lines: list[str] = [
            f"Decision: {self.decision}",
            f"Horse: {self.horse_name or self.horse_key or 'unknown'}",
            f"Race: {race_bits}",
        ]

        if self.decision == CANDIDATE:
            lines.extend(
                [
                    f"Bookmaker: {self.bookmaker or 'unknown'}",
                    f"Offered odds: {_fmt(self.offered_odds, 2)}",
                    "Quote taken at: "
                    f"{self.quote_fetched_at or 'unknown'} "
                    f"(age {_fmt(self.quote_age_seconds, 1)}s)",
                    "Model probability (independent, price-free): "
                    f"{_fmt_pct(self.model_prob)}",
                ]
            )
            if self.market_adjusted_prob is not None:
                lines.append(
                    "Market-adjusted probability (price echo, not the basis of "
                    f"this decision): {_fmt_pct(self.market_adjusted_prob)}"
                )
            lines.extend(
                [
                    f"Fair odds (de-vigged reference): {_fmt(self.fair_odds, 3)}",
                    f"Market probability: {_fmt_pct(self.market_prob)}",
                    f"Edge over fair line: {_fmt(self.edge, 4)}",
                    f"Expected value: {_fmt(self.expected_value, 4)} per unit staked",
                    f"Maximum stake: {currency}{_fmt(self.max_stake, 2)}",
                ]
            )
            if self.stake is not None:
                lines.append(f"Stake: {currency}{_fmt(self.stake, 2)}")
            if self.ew_places is not None or self.ew_reduction is not None:
                lines.append(
                    f"Each-way terms: {self.ew_places or 'n/a'} places, "
                    f"reduction {_fmt(self.ew_reduction, 3)}"
                )
            lines.append(f"Data quality: {self.data_quality}")
            lines.append(f"Validation state: {self.validation_state}")
            lines.append(
                "Conditions passed: "
                + (", ".join(self.reasons_passed) if self.reasons_passed else "none")
            )
        else:
            lines.append("PASS reasons:")
            lines.extend(
                f"  - {reason}" for reason in (self.pass_reasons or ("unspecified",))
            )
            lines.append(f"Data quality: {self.data_quality}")
            lines.append(f"Validation state: {self.validation_state}")

        if self.paper_only:
            lines.append("Paper only: no real money is staked.")
        return [f"{mark}{line}" for line in lines]


# ── construction ─────────────────────────────────────────────────────────────

def make_ticket_id(race_uid: str, horse_key: str, bet_type: str, dedupe_key: str) -> str:
    """Deterministic id for one decision.

    Same race + runner + bet type + ``dedupe_key`` always yields the same id, so
    a re-run of the same card cannot silently duplicate the audit trail, and a
    ticket can be located from a report without a lookup table. ``dedupe_key``
    is deliberately a caller-chosen stability anchor (typically the calendar
    day), never the ticket's own real ``issued_at`` (Stage 20 / B6) — those two
    used to be the same value, which forced every ticket's disclosed issue time
    to be a fake day-truncated stand-in just to keep re-runs idempotent.
    """
    payload = "|".join(str(part or "") for part in (race_uid, horse_key, bet_type, dedupe_key))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _forward_gate_label(state: Any) -> str:
    text = _s(state)
    return text.upper() if text else "NOT MET"


def _forward_gate_passed(state: Any) -> bool:
    return _forward_gate_label(state) in ("PASSED", "PASS", "MET", "OK")


def _data_quality(detail: Mapping, cfg: ExecutionConfig) -> str:
    """A short label for how much the inputs to this decision can be trusted.

    Derived from the three things that actually degrade: the price's age, the
    contributing sources' health, and whether the reference book was complete.
    ``unknown`` is a real answer — it is what we say when none of the three could
    be established, and it is never rounded up to ``degraded``.
    """
    age = _f(detail.get("quote_age_seconds"))
    max_age = _f(getattr(cfg.snapshots, "max_age_seconds", None)) or 0.0
    complete = detail.get("reference_book_complete")
    health = detail.get("source_health")
    health_states = list(health.values()) if isinstance(health, Mapping) else []

    if age is None and complete is None and not health_states:
        return "unknown"

    unhealthy = any(state != "ok" for state in health_states)
    if age is None or (max_age > 0 and age > max_age) or complete is False or unhealthy:
        return "poor"
    if (max_age > 0 and age > 0.5 * max_age) or complete is None or not health_states:
        return "degraded"
    return "good"


def build_ticket(
    runner: Mapping,
    race: Mapping,
    gate_result: GateResult,
    *,
    cfg: ExecutionConfig,
    model_verdict,
    quote: Any = None,
    stake_decision: Any = None,
    forward_gate_state: Any = None,
    now: Any = None,
    dedupe_key: Any = None,
    provenance: Optional[Mapping] = None,
) -> Ticket:
    """Turn one gate decision into a fully disclosed ticket.

    Everything numeric is taken from ``gate_result.detail`` — the gate already
    computed the price, the fair line, the edge and the EV under its own
    fail-closed rules, and recomputing any of them here would be a second, quietly
    divergent opinion.

    ``now`` is the REAL issuance instant (``issued_at``) — Stage 20 (B6/B7)
    stopped a caller from passing a day-truncated stand-in here (it used to be
    the only way to get an idempotent ``ticket_id`` across a same-day re-run,
    which silently made every ticket's disclosed issue time a fake midnight).
    ``dedupe_key`` is that stable per-day anchor now, used ONLY for
    :func:`make_ticket_id` and defaulting to ``issued_at`` for any caller that
    does not supply one (unchanged behaviour for existing callers/tests).

    ``provenance`` (Stage 20 / B7), when supplied, carries
    ``model_content_hash`` / ``feature_schema_version`` / ``config_hash`` /
    ``prediction_cycle_id`` from the prediction cycle that produced ``race`` —
    a ticket records them as-is and never fabricates a missing one.
    """
    detail = gate_result.detail or {}
    ts = _now(now)
    issued_at = ts.isoformat()
    dedupe = _s(dedupe_key) if dedupe_key is not None else None
    if dedupe is None:
        dedupe = issued_at

    prov = provenance if isinstance(provenance, Mapping) else {}
    model_content_hash = _s(prov.get("model_content_hash"))
    feature_schema_version = _s(prov.get("feature_schema_version"))
    config_hash = _s(prov.get("config_hash"))
    prediction_cycle_id = _s(prov.get("prediction_cycle_id"))
    provenance_complete = all(
        v is not None
        for v in (model_content_hash, feature_schema_version, config_hash, prediction_cycle_id)
    )

    uid = _s(detail.get("race_uid")) or race_uid(race)
    rkey = _s(detail.get("race_key")) or race_key(race)
    hkey = _s(detail.get("horse_key")) or horse_key(runner)
    bet_type = _s(runner.get("bet_type")) or _s(race.get("bet_type")) or "win"
    # Upper-cased to match the snapshot store's market key, so a ticket joins back
    # to the quote it was priced from.
    market_type = (
        _s(detail.get("market_type"))
        or _s(runner.get("market_type"))
        or _s(race.get("market_type"))
        or "WIN"
    ).upper()

    # The permanent paper-only override, answered in exactly one place.
    paper_only = not real_money_enabled(
        cfg,
        model_go=bool(getattr(model_verdict, "go", False)),
        forward_gate_passed=_forward_gate_passed(forward_gate_state),
    )

    stake = _f(_get_any(stake_decision, ("stake", "recommended_stake")))
    max_stake = _f(_get_any(stake_decision, ("max_stake", "stake_cap", "cap")))
    if max_stake is None:
        # No staking decision supplied: disclose the per-bet liability ceiling,
        # which is knowable without a bankroll. Never invent a larger one.
        max_stake = _f(getattr(cfg.frictions, "max_stake_per_bet", None))
    if gate_result.decision != CANDIDATE:
        stake = None  # a PASS is a no-bet; it can never carry a stake

    # Each-way terms are only ever *recorded*, never assumed — the same rule that
    # forbids assuming best-odds-guaranteed. The quote's terms win: they are the
    # book's own terms at the instant the price was observed.
    ew_places = _f(
        detail.get("ew_places")
        if detail.get("ew_places") is not None
        else (_get_any(runner, ("ew_places",)) or _get_any(race, ("ew_places",)))
    )
    ew_reduction = _f(
        detail.get("ew_reduction")
        if detail.get("ew_reduction") is not None
        else (
            _get_any(runner, ("ew_reduction", "ew_fraction"))
            or _get_any(race, ("ew_reduction", "ew_fraction"))
        )
    )

    verdict_label = _s(getattr(model_verdict, "verdict_label", None)) or "NO-GO"
    validation_state = (
        f"MODEL {verdict_label} / FORWARD GATE {_forward_gate_label(forward_gate_state)}"
    )

    offered = _f(detail.get("executable_odds"))
    if offered is None and quote is not None:
        offered = _f(_first(quote, _QUOTE_ODDS_KEYS)[0])

    return Ticket(
        ticket_id=make_ticket_id(uid, hkey, bet_type, dedupe),
        issued_at=issued_at,
        as_of=_s(detail.get("evaluated_at")) or issued_at,
        race_uid=uid,
        race_id=_s(race.get("race_id")),
        race_time=_s(race.get("race_time")),
        venue=_s(race.get("venue")),
        race_key=rkey or None,
        horse_key=hkey,
        horse_id=_s(runner.get("horse_id")),
        horse_name=_s(runner.get("horse_name")) or _s(runner.get("name")),
        bookmaker=_s(detail.get("bookmaker")),
        market_type=market_type,
        bet_type=bet_type,
        offered_odds=offered,
        quote_fetched_at=_s(detail.get("quote_fetched_at")),
        quote_age_seconds=_f(detail.get("quote_age_seconds")),
        model_prob=_f(detail.get("model_prob")),
        market_adjusted_prob=_f(detail.get("market_adjusted_prob")),
        fair_odds=_f(detail.get("fair_odds")),
        market_prob=_f(detail.get("market_prob")),
        edge=_f(detail.get("edge")),
        expected_value=_f(detail.get("expected_value")),
        max_stake=max_stake,
        stake=stake,
        ew_places=int(ew_places) if ew_places is not None else None,
        ew_reduction=ew_reduction,
        data_quality=_data_quality(detail, cfg),
        validation_state=validation_state,
        decision=gate_result.decision,
        pass_reasons=tuple(gate_result.pass_reasons),
        reasons_passed=tuple(gate_result.passed),
        paper_only=paper_only,
        model_verdict=verdict_label,
        forward_gate_state=_forward_gate_label(forward_gate_state),
        model_content_hash=model_content_hash,
        feature_schema_version=feature_schema_version,
        config_hash=config_hash,
        prediction_cycle_id=prediction_cycle_id,
        provenance_complete=provenance_complete,
    )


# ── persistence ──────────────────────────────────────────────────────────────

class TicketStore:
    """Append-and-settle store for paper tickets, backed by ``paper_tickets``."""

    def __init__(self, db_path: Optional[str] = None, cfg: Optional[ExecutionConfig] = None) -> None:
        self._cfg = cfg or ExecutionConfig.from_config()
        path = db_path or getattr(self._cfg.snapshots, "db_path", None) or DEFAULT_DB_PATH
        self._pool = ConnectionPool(str(path))
        apply_migrations(self._pool)
        self._ensure_table()
        self._columns = self._table_columns()
        missing = [c for c in TICKET_COLUMNS if c not in self._columns]
        if missing:
            logger.warning(
                "tickets: paper_tickets is missing columns %s — those fields will "
                "not be persisted", ", ".join(missing),
            )

    @property
    def cfg(self) -> ExecutionConfig:
        """The config this store was opened with (frictions, staking, ...).

        Callers that settle tickets after the fact (``scripts.daily_paper_loop``)
        need the same commission/friction rules the store itself was built
        against, rather than constructing a second, potentially divergent
        ``ExecutionConfig.from_config()``.
        """
        return self._cfg

    # ── schema ───────────────────────────────────────────────────────────────
    def _conn(self) -> sqlite3.Connection:
        return self._pool.connection()

    def _ensure_table(self) -> None:
        """Assert migration 5 gave us ``paper_tickets``.

        Deliberately *not* a fallback ``CREATE TABLE`` — a second copy of the
        schema here is a second copy of ``CHECK (paper_only = 1)`` that could
        silently drift out of step with the migration. If the table is absent the
        database is not migrated, and that is a loud failure, not a repair job.
        """
        if not self._table_columns():
            raise RuntimeError(
                "paper_tickets is missing — run utils.storage.migrations "
                "(migration 5) against this database before issuing tickets"
            )

    def _table_columns(self) -> set[str]:
        rows = self._conn().execute("PRAGMA table_info(paper_tickets)").fetchall()
        return {str(r[1]) for r in rows}

    # ── writes ───────────────────────────────────────────────────────────────
    def issue(self, ticket: Ticket) -> str:
        """Persist a ticket. Refuses anything that is not paper-only.

        The refusal is the first of the two guarantees; the table's
        ``CHECK (paper_only = 1)`` is the second and cannot be reasoned around.
        """
        if not ticket.paper_only:
            raise RealMoneyTicketRefused(
                f"ticket {ticket.ticket_id} is not paper-only; this deployment "
                "never issues real-money tickets (Stage-4 MODEL NO-GO)"
            )
        row = ticket.to_row()
        cols = [c for c in TICKET_COLUMNS if c in self._columns]
        sql = (
            f"INSERT INTO paper_tickets ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})"
        )
        try:
            with self._pool.write_lock():
                conn = self._conn()
                conn.execute(sql, [row[c] for c in cols])
                conn.commit()
        except sqlite3.IntegrityError as exc:
            if "paper_only" in str(exc):
                raise RealMoneyTicketRefused(str(exc)) from exc
            raise ValueError(f"ticket {ticket.ticket_id} already issued: {exc}") from exc
        logger.info(
            "tickets: %s %s %s (%s)",
            ticket.decision,
            ticket.horse_name or ticket.horse_key,
            ticket.race_uid,
            "; ".join(ticket.pass_reasons) if ticket.pass_reasons else "all conditions met",
        )
        return ticket.ticket_id

    def settle(
        self,
        ticket_id: str,
        settlement: Any,
        *,
        closing_odds: Optional[float] = None,
        settled_at: Any = None,
    ) -> None:
        """Record the outcome, the closing price and the resulting CLV.

        ``settlement`` is an outcome label (``won`` / ``lost`` / ``void``) or a
        mapping carrying ``outcome`` plus any of ``returns`` / ``profit`` /
        ``detail``. A PASS may be settled too: it carries no stake, so returns and
        profit are zero, but its CLV against the close is exactly the evidence
        that tells us whether passing was right.
        """
        existing = self._raw(ticket_id)
        if existing is None:
            raise ValueError(f"unknown ticket: {ticket_id}")
        if _s(existing.get("settled_at")):
            raise ValueError(f"ticket already settled: {ticket_id}")

        if isinstance(settlement, Mapping):
            outcome_raw = settlement.get("outcome")
            returns = _f(settlement.get("returns"))
            profit = _f(settlement.get("profit"))
            extra = {k: v for k, v in settlement.items() if k not in ("outcome", "returns", "profit")}
        else:
            outcome_raw = settlement
            returns = profit = None
            extra = {}

        key = (_s(outcome_raw) or "").lower()
        outcome = _OUTCOMES.get(key)
        if outcome is None:
            raise ValueError(
                f"unknown settlement outcome {outcome_raw!r}; expected one of "
                f"{sorted(set(_OUTCOMES.values()))}"
            )

        ticket = Ticket.from_row(existing)
        stake = ticket.stake or 0.0
        odds = ticket.offered_odds
        commission = self._cfg.frictions.commission_for(ticket.bookmaker)
        if profit is None:
            if outcome == "void":
                profit = 0.0
            else:
                profit = float(
                    metrics.settle(stake, odds, 1.0 if outcome == "won" else 0.0, commission)
                )
        if returns is None:
            if outcome == "void":
                returns = stake
            elif outcome == "won":
                returns = stake + profit
            else:
                returns = 0.0

        close = _f(closing_odds)
        clv = None
        if close is not None and odds is not None:
            value = _f(metrics.clv_pct(odds, close))
            clv = value

        when = _parse_ts(settled_at) or _now(None)
        payload = {
            "settled_at": when.isoformat(),
            "outcome": outcome,
            "returns": _f(returns),
            "profit": _f(profit),
            "closing_odds": close,
            "clv_pct": clv,
            "settlement_detail": json.dumps(extra, default=str) if extra else None,
        }
        cols = [c for c in SETTLEMENT_COLUMNS if c in self._columns]
        sql = (
            f"UPDATE paper_tickets SET {', '.join(f'{c} = ?' for c in cols)} "
            "WHERE ticket_id = ?"
        )
        with self._pool.write_lock():
            conn = self._conn()
            conn.execute(sql, [payload[c] for c in cols] + [ticket_id])
            conn.commit()
        logger.info(
            "tickets: settled %s as %s (profit %.2f, clv %s)",
            ticket_id,
            outcome,
            payload["profit"] or 0.0,
            f"{clv:+.4f}" if clv is not None else "n/a",
        )

    # ── reads ────────────────────────────────────────────────────────────────
    def _raw(self, ticket_id: str) -> Optional[dict]:
        row = self._conn().execute(
            "SELECT * FROM paper_tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row is not None else None

    def get(self, ticket_id: str) -> Optional[Ticket]:
        row = self._raw(ticket_id)
        return Ticket.from_row(row) if row is not None else None

    def open_tickets(self) -> list[Ticket]:
        """Unsettled CANDIDATE tickets — the live exposure, PASSes excluded."""
        rows = self._conn().execute(
            "SELECT * FROM paper_tickets WHERE decision = ? AND settled_at IS NULL "
            "ORDER BY issued_at",
            (CANDIDATE,),
        ).fetchall()
        return [Ticket.from_row(dict(r)) for r in rows]

    def unobserved_pass_tickets(self) -> list[Ticket]:
        """Unsettled PASS tickets that still carry an executable price.

        A PASS is never a bet — it stays zero-stake even once "settled" — but
        its result and closing price are real evidence of whether passing was
        right (probability calibration, CLV coverage on the disclosure lane).
        Only rows with a recorded ``offered_odds`` are candidates: a PASS
        issued with no price at all (most PASS reasons fire before a price is
        even read) has nothing to compute a CLV against, so it is left for a
        later cycle rather than counted as an observation gap.
        """
        rows = self._conn().execute(
            "SELECT * FROM paper_tickets WHERE decision = ? AND settled_at IS NULL "
            "AND offered_odds IS NOT NULL ORDER BY issued_at",
            (PASS,),
        ).fetchall()
        return [Ticket.from_row(dict(r)) for r in rows]

    def all_tickets(
        self,
        *,
        since: Any = None,
        until: Any = None,
        decision: Optional[str] = None,
    ) -> list[Ticket]:
        sql = "SELECT * FROM paper_tickets WHERE 1=1"
        params: list[Any] = []
        start = _parse_ts(since)
        if start is not None:
            sql += " AND issued_at >= ?"
            params.append(start.isoformat())
        end = _parse_ts(until)
        if end is not None:
            sql += " AND issued_at <= ?"
            params.append(end.isoformat())
        if decision:
            sql += " AND decision = ?"
            params.append(str(decision))
        sql += " ORDER BY issued_at, ticket_id"
        rows = self._conn().execute(sql, params).fetchall()
        return [Ticket.from_row(dict(r)) for r in rows]

    def to_frame(self, **kw: Any) -> pd.DataFrame:
        """Ticket rows as a DataFrame, including the settlement columns."""
        tickets = self.all_tickets(**kw)
        if not tickets:
            return pd.DataFrame(columns=list(TICKET_COLUMNS) + list(SETTLEMENT_COLUMNS))
        settled = {t.ticket_id: (self._raw(t.ticket_id) or {}) for t in tickets}
        rows = []
        for ticket in tickets:
            row = ticket.to_dict()
            raw = settled.get(ticket.ticket_id, {})
            for col in SETTLEMENT_COLUMNS:
                row[col] = raw.get(col)
            rows.append(row)
        return pd.DataFrame(rows)

    def close(self) -> None:
        self._pool.close_all()


# ── small helpers ────────────────────────────────────────────────────────────

def _decode_reasons(raw: Any) -> tuple[str, ...]:
    """Reasons back from their stored form, tolerating a legacy delimited list."""
    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(str(r) for r in raw)
    text = _s(raw)
    if not text:
        return ()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return tuple(str(r) for r in parsed)
    return tuple(part.strip() for part in text.split("|") if part.strip())


def _get_any(obj: Any, keys: Sequence[str]) -> Any:
    if obj is None:
        return None
    value, _ = _first(obj, keys)
    return value


def _fmt(value: Optional[float], nd: int) -> str:
    return "unknown" if value is None else f"{value:.{nd}f}"


def _fmt_pct(value: Optional[float]) -> str:
    return "unknown" if value is None else f"{value:.4f} ({value * 100:.2f}%)"
