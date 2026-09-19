"""The disclosure ticket, and the two guarantees that keep real money out.

The tests fall into three groups:

* **disclosure** — every field requirement 8 lists is present on the ticket,
  whether it was a candidate or a PASS, and the PASS carries its reasons;
* **refusal** — a non-paper ticket is refused by the Python API *and* by the
  table constraint, independently. Both are checked, because the point of having
  two is that neither relies on the other;
* **store semantics** — duplicates, settlement, double settlement, CLV.
"""
from __future__ import annotations

import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from execution.config import ExecutionConfig, real_money_enabled
from execution.gates import GateResult, evaluate_candidate
from execution.model_gate import load_model_verdict
from execution.tickets import (
    CANDIDATE,
    PASS,
    RealMoneyTicketRefused,
    Ticket,
    TicketStore,
    build_ticket,
    make_ticket_id,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def cfg() -> ExecutionConfig:
    return ExecutionConfig.from_config()


@pytest.fixture
def db_path(tmp_path) -> str:
    return str(tmp_path / "tickets.db")


@pytest.fixture
def store(db_path, cfg) -> TicketStore:
    st = TicketStore(db_path, cfg=cfg)
    yield st
    st.close()


@pytest.fixture
def race() -> dict:
    return {
        "venue": "Naas",
        "race_time": "2026-07-27T15:40:00+00:00",
        "race_id": "SBTE_2_1028491168",
        "field_size": 6,
        "ev_eligible": True,
        "ev_gate": {
            "eligible": True,
            "reference_source": "livescorebet",
            "reference_book_complete": True,
            "reference_overround": 0.18,
            "price_sources": ["livescorebet"],
        },
        "runners": [
            {
                "horse_id": f"H{i}",
                "horse_name": f"Runner {i}",
                "reference_odds": 7.0,
                "best_odds": 8.0,
                "best_book": "livescorebet",
                "reference_source": "livescorebet",
                "value_supported": True,
                "value_win_prob_independent": 0.20,
                "value_win_prob": 0.16,
            }
            for i in range(6)
        ],
    }


@pytest.fixture
def quote() -> dict:
    return {
        "race_uid": "naas|2026-07-27T15:40",
        "horse_key": "runner0",
        "bookmaker": "livescorebet",
        "market_type": "WIN",
        "odds_decimal": 8.0,
        "fetched_at": NOW - timedelta(seconds=40),
        "as_of": NOW,
        "age_seconds": 40.0,
        "is_stale": False,
    }


def _ticket(race, cfg, quote, *, verdict=None, decision=None) -> Ticket:
    verdict = verdict if verdict is not None else load_model_verdict()
    runner = race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=verdict,
        quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
        now=NOW,
    )
    if decision is not None:
        result = replace(result, decision=decision)
    return build_ticket(
        runner, race, result, cfg=cfg, model_verdict=verdict,
        quote=quote, forward_gate_state="FORWARD GATE NOT MET", now=NOW,
    )


# ── disclosure (requirement 8) ───────────────────────────────────────────────
REQUIRED_DISCLOSURE = (
    "horse_name", "race_uid", "bookmaker", "offered_odds", "quote_fetched_at",
    "quote_age_seconds", "model_prob", "market_adjusted_prob", "fair_odds",
    "market_prob", "edge", "expected_value", "max_stake", "data_quality",
    "validation_state",
)


def test_every_disclosure_field_is_present_on_a_pass(race, cfg, quote):
    """A PASS discloses exactly what a candidate would. That is the point."""
    ticket = _ticket(race, cfg, quote)
    assert ticket.decision == PASS  # Stage 4 says NO-GO
    row = ticket.to_dict()
    for field in REQUIRED_DISCLOSURE:
        assert field in row, field
    # The numbers that make the decision auditable are actually populated, not
    # merely present as null columns.
    assert row["offered_odds"] == pytest.approx(8.0)
    assert row["quote_age_seconds"] == pytest.approx(40.0)
    assert row["model_prob"] == pytest.approx(0.20)
    assert row["fair_odds"] is not None
    assert row["market_prob"] is not None
    assert row["edge"] is not None
    assert row["expected_value"] is not None


def test_a_pass_ticket_carries_its_reasons_and_the_verdict_that_caused_them(
    race, cfg, quote
):
    ticket = _ticket(race, cfg, quote)
    assert ticket.pass_reasons, "a PASS with no reason is not a disclosure"
    assert any(r.startswith("model_validation:") for r in ticket.pass_reasons)
    assert ticket.model_verdict == "NO-GO"
    assert ticket.forward_gate_state == "FORWARD GATE NOT MET"


def test_the_independent_probability_is_the_one_recorded(race, cfg, quote):
    """Not the market-adjusted one: Stage 4 measured that at 0.925 vs 1/price."""
    ticket = _ticket(race, cfg, quote)
    assert ticket.model_prob == pytest.approx(0.20)
    assert ticket.market_adjusted_prob == pytest.approx(0.16)


def test_fair_odds_are_the_devigged_reference_not_the_offered_price(race, cfg, quote):
    """The ticket shows both readings of the price, and they are not the same number.

    ``market_prob`` is the *raw* implied probability of the reference price —
    what the book is quoting, vig included. ``fair_odds`` is that price with the
    declared overround removed. Requirement 8 asks for both, and the pair is only
    informative because they differ: the gap between them is the vig the bet has
    to overcome before any model edge counts.
    """
    ticket = _ticket(race, cfg, quote)
    assert ticket.market_prob == pytest.approx(1.0 / 7.0), "raw reference implied prob"
    # 1.18 is the race's declared reference_overround.
    assert ticket.fair_odds == pytest.approx(7.0 * 1.18, rel=1e-6)
    assert ticket.fair_odds != pytest.approx(ticket.offered_odds)


def test_max_stake_is_a_ceiling_and_never_a_recommendation(race, cfg, quote):
    """Under NO-GO the ticket still discloses the ceiling, but stakes nothing."""
    ticket = _ticket(race, cfg, quote)
    assert ticket.decision == PASS
    assert ticket.stake in (None, 0.0)
    assert ticket.max_stake is None or ticket.max_stake >= 0.0


def test_render_lines_never_say_bet_under_a_no_go(race, cfg, quote):
    text = " ".join(_ticket(race, cfg, quote).render_lines()).lower()
    assert "no bet" in text or "pass" in text
    assert "recommend" not in text


# ── the two independent refusals ─────────────────────────────────────────────
def test_the_api_refuses_a_real_money_ticket(race, cfg, quote, store):
    ticket = replace(_ticket(race, cfg, quote), paper_only=False)
    with pytest.raises(RealMoneyTicketRefused):
        store.issue(ticket)


def test_the_table_refuses_a_real_money_row_written_around_the_api(
    store, db_path, race, cfg, quote
):
    """The second guarantee: SQLite rejects it even with the Python API bypassed."""
    ticket = _ticket(race, cfg, quote)
    row = ticket.to_row()
    row["paper_only"] = 0
    cols = [c for c in row if c != "id"]
    conn = sqlite3.connect(db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                f"INSERT INTO paper_tickets ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})",
                [row[c] for c in cols],
            )
            conn.commit()
    finally:
        conn.close()


def test_real_money_is_off_under_the_real_verdict(cfg):
    verdict = load_model_verdict()
    assert verdict.go is False
    assert real_money_enabled(cfg, model_go=verdict.go, forward_gate_passed=True) is False
    assert real_money_enabled(cfg, model_go=True, forward_gate_passed=True) is False, (
        "cfg.paper_only is the permanent override; a GO must not unlock it"
    )


def test_the_paper_only_override_survives_a_go_and_a_passed_gate(cfg):
    """Requirement 9's permanent override: nothing but config can lift it."""
    assert cfg.paper_only is True
    unlocked = replace(cfg, paper_only=False)
    # Only with the override *explicitly* lifted AND both gates green does the
    # single authorised function change its answer — and even then nothing in
    # this repo calls it with paper_only False.
    assert real_money_enabled(unlocked, model_go=True, forward_gate_passed=True) is True
    assert real_money_enabled(unlocked, model_go=False, forward_gate_passed=True) is False
    assert real_money_enabled(unlocked, model_go=True, forward_gate_passed=False) is False


# ── store semantics ──────────────────────────────────────────────────────────
def test_ticket_ids_are_stable_and_distinguish_bet_types():
    a = make_ticket_id("naas|2026-07-27T15:40", "runner0", "win", "2026-07-27T12:00:00")
    b = make_ticket_id("naas|2026-07-27T15:40", "runner0", "win", "2026-07-27T12:00:00")
    c = make_ticket_id("naas|2026-07-27T15:40", "runner0", "each_way", "2026-07-27T12:00:00")
    assert a == b
    assert a != c


# ── Stage 20: identity, dedupe/issuance separation, provenance (B5/B6/B7) ────

def test_ticket_carries_a_venue_qualified_race_key(race, cfg, quote):
    ticket = _ticket(race, cfg, quote)
    assert ticket.race_key == "naas|2026-07-27T15:40"


def test_issued_at_is_the_real_instant_not_the_dedupe_key(race, cfg, quote):
    """Stage 20 / B6: a caller may pass a stable per-day ``dedupe_key`` for
    idempotent ticket_ids without that value leaking into the disclosed
    ``issued_at`` — the two used to be forced identical, so a re-run's
    ``issued_at`` was always a fake day-truncated midnight."""
    real_now = NOW + timedelta(hours=3, minutes=17, seconds=5)
    runner = race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
        now=real_now,
    )
    ticket = build_ticket(
        runner, race, result, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, forward_gate_state="FORWARD GATE NOT MET",
        now=real_now, dedupe_key="2026-07-27",
    )
    assert ticket.issued_at == real_now.isoformat()
    assert ticket.issued_at != "2026-07-27"


def test_same_day_rerun_with_a_stable_dedupe_key_is_idempotent(race, cfg, quote):
    """Two builds at genuinely different wall-clock instants on the same
    calendar day must still mint the identical ticket_id when given the same
    dedupe_key — this is what makes a same-day re-run a no-op duplicate."""
    runner = race["runners"][0]

    def _build(instant):
        result = evaluate_candidate(
            runner, race, cfg=cfg, model_verdict=load_model_verdict(),
            quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
            now=instant,
        )
        return build_ticket(
            runner, race, result, cfg=cfg, model_verdict=load_model_verdict(),
            quote=quote, forward_gate_state="FORWARD GATE NOT MET",
            now=instant, dedupe_key="2026-07-27",
        )

    first = _build(NOW)
    second = _build(NOW + timedelta(hours=6))
    assert first.ticket_id == second.ticket_id
    assert first.issued_at != second.issued_at


def test_build_ticket_without_dedupe_key_falls_back_to_issued_at(race, cfg, quote):
    """Unchanged behaviour for any existing caller that never passes dedupe_key."""
    ticket = _ticket(race, cfg, quote)
    assert ticket.ticket_id == make_ticket_id(
        ticket.race_uid, ticket.horse_key, ticket.bet_type, ticket.issued_at
    )


def test_provenance_is_recorded_and_marked_complete_when_all_four_present(
    race, cfg, quote
):
    runner = race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
        now=NOW,
    )
    ticket = build_ticket(
        runner, race, result, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, forward_gate_state="FORWARD GATE NOT MET", now=NOW,
        provenance={
            "model_content_hash": "sha256:abc",
            "feature_schema_version": "v3nf",
            "config_hash": "sha256:def",
            "prediction_cycle_id": "2026-07-27T12:00:00+00:00-abc123",
        },
    )
    assert ticket.model_content_hash == "sha256:abc"
    assert ticket.feature_schema_version == "v3nf"
    assert ticket.config_hash == "sha256:def"
    assert ticket.prediction_cycle_id == "2026-07-27T12:00:00+00:00-abc123"
    assert ticket.provenance_complete is True


def test_provenance_is_incomplete_when_any_field_is_missing(race, cfg, quote):
    ticket = _ticket(race, cfg, quote)  # no provenance supplied at all
    assert ticket.model_content_hash is None
    assert ticket.provenance_complete is False

    runner = race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
        now=NOW,
    )
    partial = build_ticket(
        runner, race, result, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, forward_gate_state="FORWARD GATE NOT MET", now=NOW,
        provenance={"model_content_hash": "sha256:abc"},  # 3 of 4 missing
    )
    assert partial.provenance_complete is False


def test_provenance_and_race_key_round_trip_through_the_store(store, race, cfg, quote):
    runner = race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, source_health={"livescorebet": {"status": "ok", "age_seconds": 30.0}},
        now=NOW,
    )
    ticket = build_ticket(
        runner, race, result, cfg=cfg, model_verdict=load_model_verdict(),
        quote=quote, forward_gate_state="FORWARD GATE NOT MET", now=NOW,
        provenance={
            "model_content_hash": "sha256:abc",
            "feature_schema_version": "v3nf",
            "config_hash": "sha256:def",
            "prediction_cycle_id": "cycle-1",
        },
    )
    store.issue(ticket)
    back = store.get(ticket.ticket_id)
    assert back.race_key == "naas|2026-07-27T15:40"
    assert back.model_content_hash == "sha256:abc"
    assert back.provenance_complete is True


def test_a_ticket_row_predating_the_provenance_contract_reads_as_incomplete(store):
    """A row written before Stage 20 has NULL provenance columns; ``from_row``
    must read that as an honest 'incomplete', never a fabricated match."""
    row = {
        "ticket_id": "legacy1", "issued_at": NOW.isoformat(), "race_uid": "x",
        "horse_key": "h", "market_type": "WIN", "bet_type": "win",
        "data_quality": "unknown", "validation_state": "MODEL NO-GO",
        "decision": PASS, "paper_only": 1,
    }
    ticket = Ticket.from_row(row)
    assert ticket.race_key is None
    assert ticket.model_content_hash is None
    assert ticket.provenance_complete is False


def test_duplicate_issue_is_refused(store, race, cfg, quote):
    """Requirement 9's duplicate prevention, at the storage layer."""
    ticket = _ticket(race, cfg, quote)
    store.issue(ticket)
    with pytest.raises(ValueError, match="already issued"):
        store.issue(ticket)


def test_a_settled_ticket_records_the_close_and_its_clv(store, race, cfg, quote):
    ticket = _ticket(race, cfg, quote)
    store.issue(ticket)
    store.settle(ticket.ticket_id, "lost", closing_odds=6.0, settled_at=NOW)

    settled = store.get(ticket.ticket_id)
    assert settled.decision == PASS
    frame = store.to_frame()
    row = frame[frame["ticket_id"] == ticket.ticket_id].iloc[0]
    assert row["closing_odds"] == pytest.approx(6.0)
    # Struck at 8.0, closed at 6.0 — positive CLV even though the bet lost. That
    # separation is the whole reason CLV is the leading indicator.
    assert row["clv_pct"] > 0


def test_double_settlement_is_refused(store, race, cfg, quote):
    ticket = _ticket(race, cfg, quote)
    store.issue(ticket)
    store.settle(ticket.ticket_id, "lost", closing_odds=6.0, settled_at=NOW)
    with pytest.raises(ValueError, match="already settled"):
        store.settle(ticket.ticket_id, "won", closing_odds=6.0, settled_at=NOW)


def test_settling_an_unknown_ticket_is_an_error_not_a_silent_insert(store):
    with pytest.raises(ValueError, match="unknown ticket"):
        store.settle("nope", "won")


def test_an_unknown_outcome_is_refused(store, race, cfg, quote):
    ticket = _ticket(race, cfg, quote)
    store.issue(ticket)
    with pytest.raises(ValueError, match="unknown settlement outcome"):
        store.settle(ticket.ticket_id, "probably won")


def test_a_pass_is_never_open_exposure(store, race, cfg, quote):
    """``open_tickets`` is the live risk, and a PASS carries none of it.

    The distinction matters to the exposure caps: if declined runners counted as
    open tickets, a NO-GO day with 356 PASSes would read as 356 live bets and the
    daily-exposure limit would block a system that had staked nothing.
    """
    store.issue(_ticket(race, cfg, quote))
    assert store.open_tickets() == []
    assert len(store.all_tickets()) == 1


def test_open_tickets_excludes_settled_ones(store, race, cfg, quote):
    # Synthetic: under Stage 4's NO-GO the gate cannot produce a CANDIDATE, so the
    # decision is forced here to exercise the store's own bookkeeping.
    ticket = _ticket(race, cfg, quote, decision=CANDIDATE)
    store.issue(ticket)
    assert [t.ticket_id for t in store.open_tickets()] == [ticket.ticket_id]
    store.settle(ticket.ticket_id, "lost", closing_odds=6.0, settled_at=NOW)
    assert store.open_tickets() == []


def test_to_frame_filters_by_decision(store, race, cfg, quote):
    ticket = _ticket(race, cfg, quote)
    store.issue(ticket)
    assert len(store.to_frame(decision=PASS)) == 1
    assert len(store.to_frame(decision=CANDIDATE)) == 0


def test_every_stored_ticket_is_paper_only(store, race, cfg, quote):
    store.issue(_ticket(race, cfg, quote))
    frame = store.to_frame()
    assert len(frame) == 1
    assert bool(frame["paper_only"].all())
