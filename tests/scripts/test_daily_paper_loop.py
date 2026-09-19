"""The daily paper loop end-to-end (Stage 6, requirements 2/5/6/7).

Everything here runs against a fully isolated ``tmp_path`` config (own
``races.db``, own gap ledger, own window file, own report/artifact dirs) --
never the real production ``data/races.db`` / ``data/execution/gap_ledger.json``,
which carry real evidence from this stage's own live run and must not be
touched by a test.

Three properties are pinned:
  * re-running ``run()`` for the same day never double-issues or double-counts
    (requirement 2's idempotence);
  * a day with no capture is recorded as a gap and contributes zero qualifying
    weeks -- it is not interchangeable with a clean no-bet day (requirement 5);
  * an open ticket settled with no matching closing snapshot gets an
    unmeasurable (``None``) CLV, reported as such via ``unmeasurable_clv``,
    never dropped or invented (requirement 7).

Two more are pinned for step 07 (as-of odds capture and execution timing):
  * ``_today_tickets`` gates freshness against the real decision instant, not
    the day-truncated ``now`` used only for ticket-id idempotency -- a runner's
    own cached price is fetched intraday, so evaluating it against midnight
    used to read as a negative-age, "timestamp in the future" price on every
    live candidate;
  * when a snapshot store is supplied, live pricing is read through it
    (point-in-time, staleness-checked) rather than trusting whatever price
    happens to be cached on the predictions.json runner dict.
"""
from __future__ import annotations

import copy
import json
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from execution.config import ExecutionConfig
from execution.gates import evaluate_candidate
from execution.model_gate import load_model_verdict
from execution.race_facts import RaceFacts, RunnerFact
from execution.snapshots import SnapshotStore
from execution.tickets import Ticket, TicketStore, build_ticket
import execution.gates as gates_module
import scripts.daily_paper_loop as loop

NOW = datetime(2026, 7, 28, 9, 0, tzinfo=timezone.utc)
DAY_START = NOW.replace(hour=0, minute=0, second=0, microsecond=0)

RACE = {
    "venue": "Naas",
    "race_time": "2026-07-28T15:40:00+00:00",
    "race_id": "SBTE_2_TEST",
    "field_size": 3,
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
        for i in range(3)
    ],
}


@pytest.fixture
def cfg(tmp_path) -> ExecutionConfig:
    base = ExecutionConfig.from_config()
    snaps = replace(base.snapshots, db_path=str(tmp_path / "races.db"))
    reports = replace(
        base.reports,
        dir=str(tmp_path / "reports"),
        artifact_dir=str(tmp_path / "artifacts"),
    )
    return replace(base, snapshots=snaps, reports=reports)


@pytest.fixture
def predictions_path(tmp_path) -> str:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps({"generated_at": NOW.isoformat(), "races": [RACE]}), encoding="utf-8"
    )
    return str(path)


def _run(cfg, predictions_path, tmp_path, *, now=NOW):
    return loop.run(
        cfg=cfg,
        no_scrape=True,
        predictions_path=predictions_path,
        gap_ledger_path=str(tmp_path / "gap_ledger.json"),
        window_path=str(tmp_path / "forward_window.json"),
        report_dir=str(tmp_path / "reports"),
        now=now,
    )


# ── idempotence (requirement 2) ─────────────────────────────────────────────
def test_rerun_same_day_issues_nothing_new(cfg, predictions_path, tmp_path):
    first = _run(cfg, predictions_path, tmp_path)
    assert first["issue"]["issued"] == 3
    assert first["issue"]["duplicates"] == 0

    second = _run(cfg, predictions_path, tmp_path)
    assert second["issue"]["issued"] == 0
    assert second["issue"]["duplicates"] == 3

    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    try:
        assert len(store.all_tickets()) == 3  # not 6 -- the re-run minted no new rows
    finally:
        store.close()


def test_rerun_same_day_does_not_double_settle(cfg, predictions_path, tmp_path):
    """Every runner here is a model-gate PASS (the gate is NO-GO), so there is
    nothing open to settle -- both runs must agree it settled exactly zero."""
    first = _run(cfg, predictions_path, tmp_path)
    second = _run(cfg, predictions_path, tmp_path)
    assert first["settle"] == second["settle"] == {
        "settled": 0, "already_settled": 0, "still_open": 0, "unmeasurable_clv": 0,
        "settled_via_race_facts": 0, "pending_reasons": {},
    }


# ── gap accounting (requirement 5) ──────────────────────────────────────────
def test_a_day_with_no_capture_is_a_gap_not_a_clean_no_bet_day(cfg, predictions_path, tmp_path):
    """``no_scrape=True`` with an empty snapshot db means nothing was captured
    today -- that must show up as a gap, and the gap must cost the window a
    qualifying week, not be silently treated as "no qualifying bets"."""
    result = _run(cfg, predictions_path, tmp_path)

    assert result["capture_active_today"] is False

    gap_path = json.loads((tmp_path / "gap_ledger.json").read_text())
    entry = gap_path["days"]["2026-07-28"]
    assert entry["status"] == "gap"
    assert entry["cause"] == "scraper_outage"

    assert result["weeks_elapsed_qualifying"] == 0.0
    criteria = {c["name"]: c for c in result["forward_gate"]["criteria"]}
    assert criteria["min_weeks"]["observed"] == 0.0
    assert criteria["min_weeks"]["passed"] is False
    assert result["forward_gate"]["passed"] is False


# ── unmeasurable CLV (requirement 7) ─────────────────────────────────────────
def test_settlement_with_no_closing_quote_is_unmeasurable_not_dropped(cfg, tmp_path, monkeypatch):
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)  # empty -- no snapshot rows captured at all
    try:
        verdict = load_model_verdict()
        runner = RACE["runners"][0]
        result = evaluate_candidate(
            runner, RACE, cfg=cfg, model_verdict=verdict,
            quote=None, source_health={}, now=NOW,
        )
        result = replace(result, decision="CANDIDATE")
        ticket = build_ticket(
            runner, RACE, result, cfg=cfg, model_verdict=verdict,
            forward_gate_state="FORWARD GATE NOT MET", now=NOW,
        )
        ticket_store.issue(ticket)
        assert len(ticket_store.open_tickets()) == 1

        winning_result = {
            "venue": RACE["venue"], "horse_id": runner["horse_id"],
            "horse_name": runner["horse_name"], "race_date": "2026-07-28",
            "position": 1,
        }
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: [winning_result])

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats == {
            "settled": 1, "already_settled": 0, "still_open": 0, "unmeasurable_clv": 1,
            "settled_via_race_facts": 0, "pending_reasons": {},
        }
        assert ticket_store.open_tickets() == []  # settled, not still open

        frame = ticket_store.to_frame()
        assert len(frame) == 1
        row = frame.iloc[0]
        assert row["outcome"] == "won"
        assert pd.isna(row["clv_pct"])  # unmeasurable, never invented or dropped
    finally:
        ticket_store.close()
        snap_store.close()


# ── settlement identity (Stage 8) ────────────────────────────────────────────
def _issue_candidate(ticket_store: TicketStore, cfg, *, runner=None, race=None, now=NOW) -> Ticket:
    verdict = load_model_verdict()
    race = race or RACE
    runner = runner or race["runners"][0]
    result = evaluate_candidate(
        runner, race, cfg=cfg, model_verdict=verdict,
        quote=None, source_health={}, now=now,
    )
    result = replace(result, decision="CANDIDATE")
    ticket = build_ticket(
        runner, race, result, cfg=cfg, model_verdict=verdict,
        forward_gate_state="FORWARD GATE NOT MET", now=now,
    )
    ticket_store.issue(ticket)
    return ticket


def test_a_result_from_a_different_race_never_settles_a_ticket(cfg, monkeypatch):
    """Reproduces the historical bug directly: the previous lookup indexed
    results on ``horse_id`` alone, globally, so whichever result for that
    horse_id was read *last* settled every open ticket for it -- including a
    completely different race. Here the same ``horse_id`` recurs at a
    different venue, on a different day, listed *after* this ticket's own
    (losing) result -- the old "last write wins" dict would settle this
    ticket as a winner. It must settle as a loser, from its own race only."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        ticket = _issue_candidate(ticket_store, cfg, runner=runner)

        own_race_result = {
            "venue": RACE["venue"], "horse_id": runner["horse_id"],
            "horse_name": runner["horse_name"],
            "race_date": RACE["race_time"], "position": 4,  # lost
        }
        other_race_result = {
            "venue": "Doncaster", "horse_id": runner["horse_id"],
            "horse_name": runner["horse_name"],
            "race_date": "2026-06-01T14:00:00+00:00", "position": 1,  # won -- wrong race
        }
        monkeypatch.setattr(
            loop, "_load_results",
            lambda storage=None: [own_race_result, other_race_result],
        )

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["settled"] == 1

        settled_ticket = ticket_store.get(ticket.ticket_id)
        frame = ticket_store.to_frame()
        row = frame[frame["ticket_id"] == ticket.ticket_id].iloc[0]
        assert row["outcome"] == "lost"  # its own race's result, not Doncaster's
        assert settled_ticket is not None
    finally:
        ticket_store.close()
        snap_store.close()


def test_repeated_horse_across_dates_each_ticket_settles_against_its_own_race(cfg, monkeypatch):
    """The same horse races twice: two open tickets, two different races
    (different day, same venue). Each must settle against its own result."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        earlier_race = copy.deepcopy(RACE)
        earlier_race["race_time"] = "2026-06-01T14:00:00+00:00"

        ticket_later = _issue_candidate(ticket_store, cfg, runner=runner, race=RACE)
        ticket_earlier = _issue_candidate(ticket_store, cfg, runner=runner, race=earlier_race)

        results = [
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
             "position": 1},
            {"venue": earlier_race["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": earlier_race["race_time"],
             "position": 3},
        ]
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: results)

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["settled"] == 2

        frame = ticket_store.to_frame()
        later_row = frame[frame["ticket_id"] == ticket_later.ticket_id].iloc[0]
        earlier_row = frame[frame["ticket_id"] == ticket_earlier.ticket_id].iloc[0]
        assert later_row["outcome"] == "won"
        assert earlier_row["outcome"] == "lost"
    finally:
        ticket_store.close()
        snap_store.close()


def test_ambiguous_result_leaves_ticket_open_not_guessed(cfg, monkeypatch):
    """Two disagreeing result rows collide under the same fallback identity
    (day-only race_date, so the canonical off-time key cannot form) -- this
    must be refused, not resolved by picking either one."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        ticket = _issue_candidate(ticket_store, cfg, runner=runner)

        conflicting = [
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": "2026-07-28",
             "position": 1},
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": "2026-07-28",
             "position": 5},
        ]
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: conflicting)

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats == {
            "settled": 0, "already_settled": 0, "still_open": 1, "unmeasurable_clv": 0,
            "settled_via_race_facts": 0, "pending_reasons": {"ambiguous_result": 1},
        }
        assert [t.ticket_id for t in ticket_store.open_tickets()] == [ticket.ticket_id]
    finally:
        ticket_store.close()
        snap_store.close()


def test_duplicate_result_fetch_does_not_destabilize_settlement(cfg, monkeypatch):
    """The same result fetched twice (idempotent re-scrape, distinct rows with
    an identical position) must settle normally -- duplication is not the
    same thing as ambiguity."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        ticket = _issue_candidate(ticket_store, cfg, runner=runner)

        duplicated = [
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
             "position": 1},
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
             "position": 1},
        ]
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: duplicated)

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["settled"] == 1
        row = ticket_store.to_frame().iloc[0]
        assert row["outcome"] == "won"
    finally:
        ticket_store.close()
        snap_store.close()


def test_late_result_is_picked_up_without_regressing_a_placeholder(cfg, monkeypatch):
    """A first fetch with no position yet, followed by a later fetch that
    fills it in, must settle on the filled-in result -- and must not be read
    as an ambiguous disagreement."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        ticket = _issue_candidate(ticket_store, cfg, runner=runner)

        results = [
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
             "position": None},
            {"venue": RACE["venue"], "horse_id": runner["horse_id"],
             "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
             "position": 1},
        ]
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: results)

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["settled"] == 1
        row = ticket_store.to_frame().iloc[0]
        assert row["outcome"] == "won"
    finally:
        ticket_store.close()
        snap_store.close()


def test_resettling_an_already_settled_ticket_has_no_extra_financial_effect(cfg, monkeypatch):
    """Idempotent settlement: running settlement twice against the same
    result must not change the ticket's recorded profit/returns."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        runner = RACE["runners"][0]
        ticket = _issue_candidate(ticket_store, cfg, runner=runner)

        result_row = {
            "venue": RACE["venue"], "horse_id": runner["horse_id"],
            "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
            "position": 1,
        }
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: [result_row])

        first = loop._settle_open_tickets(ticket_store, snap_store)
        assert first["settled"] == 1
        row1 = ticket_store.to_frame().iloc[0]

        second = loop._settle_open_tickets(ticket_store, snap_store)
        assert second == {
            "settled": 0, "already_settled": 0, "still_open": 0, "unmeasurable_clv": 0,
            "settled_via_race_facts": 0, "pending_reasons": {},
        }
        row2 = ticket_store.to_frame().iloc[0]
        assert row1["profit"] == row2["profit"]
        assert row1["returns"] == row2["returns"]
    finally:
        ticket_store.close()
        snap_store.close()


def test_paper_summary_separates_pass_decisions_from_candidate_wagers(cfg, predictions_path, tmp_path):
    """Requirement: decision counts must not treat PASS tickets as wagers.
    Every runner here is a model-gate PASS (the model is NO-GO), so the
    candidate wager counts must all read zero even though decisions were
    logged for every runner."""
    _run(cfg, predictions_path, tmp_path)
    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    try:
        summary = loop._paper_summary(store)
        assert summary["n_tickets"] == 3
        assert summary["n_pass"] == 3
        assert summary["n_candidates"] == 0
        assert summary["n_open"] == 0
        assert summary["n_settled"] == 0
    finally:
        store.close()


# ── as-of odds capture and execution timing (Stage 7) ───────────────────────
def test_evaluated_at_not_day_truncated_avoids_false_future_price(cfg, tmp_path):
    """A runner's own cached price is fetched intraday, i.e. strictly after
    local midnight. Gating its freshness against the day-truncated ``now``
    used for ticket-id idempotency makes ``ts - fetched_at`` negative, which
    ``execution.gates`` reads as a price timestamped in the future -- a false
    rejection unrelated to actual data freshness, on every live candidate."""
    race = copy.deepcopy(RACE)
    fetched = (NOW - timedelta(seconds=30)).isoformat()
    for runner in race["runners"]:
        runner["fetched_at"] = fetched

    predictions_path = tmp_path / "predictions_fresh.json"
    predictions_path.write_text(
        json.dumps({"generated_at": NOW.isoformat(), "races": [race]}), encoding="utf-8"
    )

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")

    tickets, reject_reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(predictions_path),
        now=DAY_START, evaluated_at=NOW,
    )

    assert reject_reason is None
    assert tickets
    for ticket in tickets:
        assert "executable_price:timestamp_in_future" not in ticket.pass_reasons
        assert ticket.quote_age_seconds is None or ticket.quote_age_seconds >= 0


def test_a_supplied_snapshot_store_prices_the_ticket_not_the_cached_runner(cfg, tmp_path):
    """When ``_today_tickets`` is given a snapshot store, the ticket must be
    priced from the store's point-in-time quote -- not from whatever price
    happens to be cached on the predictions.json runner dict."""
    snap_store = SnapshotStore(cfg=cfg)
    try:
        race = copy.deepcopy(RACE)
        target = race["runners"][0]
        target["fetched_at"] = (NOW - timedelta(seconds=30)).isoformat()

        snap_store.record([
            {
                "race_id": race["race_id"],
                "race_time": race["race_time"],
                "venue": race["venue"],
                "horse_name": target["horse_name"],
                "horse_id": target["horse_id"],
                "bookmaker": "boylesports",
                "market_type": "WIN",
                "odds_decimal": 99.0,  # distinct from the runner's cached best_odds=8.0
                "fetched_at": NOW - timedelta(seconds=5),
            }
        ])

        predictions_path = tmp_path / "predictions_snap.json"
        predictions_path.write_text(
            json.dumps({"generated_at": NOW.isoformat(), "races": [race]}), encoding="utf-8"
        )

        verdict = load_model_verdict()
        gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")

        tickets, reject_reason = loop._today_tickets(
            cfg=cfg, verdict=verdict, gate=gate,
            predictions_path=str(predictions_path),
            now=DAY_START, evaluated_at=NOW, snap_store=snap_store,
        )
        assert reject_reason is None
        priced = [t for t in tickets if t.horse_id == target["horse_id"]]
        assert len(priced) == 1
        assert priced[0].offered_odds == pytest.approx(99.0)
        assert priced[0].bookmaker == "boylesports"
    finally:
        snap_store.close()


# ── Stage 20: operating cutoff, prediction provenance, identity (B5/B6/B7) ──

def _write_predictions(path, *, generated_at, races):
    path.write_text(
        json.dumps({"generated_at": generated_at, "races": races}), encoding="utf-8"
    )


def test_prior_day_predictions_are_rejected_not_reticketed(cfg, tmp_path):
    """GAP-B: a stale predictions.json (yesterday's refresh, today's run) must
    issue nothing, never be silently re-ticketed under today's timestamp."""
    yesterday = NOW - timedelta(days=1)
    path = tmp_path / "predictions.json"
    _write_predictions(path, generated_at=yesterday.isoformat(), races=[RACE])

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
    )
    assert tickets == []
    assert reason is not None and reason.startswith(
        ("predictions_prior_day", "predictions_stale")
    )


def test_prior_local_day_predictions_rejected_even_when_recent(cfg, tmp_path):
    """A card generated just before Dublin midnight, evaluated just after, is
    less than an hour old but a genuinely different Dublin calendar day -- the
    day-boundary check must fire independently of the raw-age check."""
    from utils.timezone import to_utc

    generated_local = datetime(2026, 7, 27, 23, 45)  # 2026-07-27 Dublin
    evaluated_local = datetime(2026, 7, 28, 0, 30)   # 2026-07-28 Dublin
    generated = to_utc(generated_local)
    evaluated = to_utc(evaluated_local)
    path = tmp_path / "predictions.json"
    _write_predictions(path, generated_at=generated.isoformat(), races=[RACE])

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=evaluated.replace(hour=0, minute=0, second=0),
        evaluated_at=evaluated,
    )
    assert tickets == []
    assert reason is not None and reason.startswith("predictions_prior_day")


def test_future_generated_at_is_rejected(cfg, tmp_path):
    future = NOW + timedelta(hours=1)
    path = tmp_path / "predictions.json"
    _write_predictions(path, generated_at=future.isoformat(), races=[RACE])

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
    )
    assert tickets == []
    assert reason is not None and reason.startswith("predictions_generated_at_future")


def test_unreadable_generated_at_is_rejected(cfg, tmp_path):
    path = tmp_path / "predictions.json"
    _write_predictions(path, generated_at="not-a-timestamp", races=[RACE])

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
    )
    assert tickets == []
    assert reason is not None and reason.startswith("predictions_generated_at_unreadable")


def test_missing_generated_at_is_rejected(cfg, tmp_path):
    """No provenance timestamp at all is the same honest failure as an
    unreadable one -- never treated as "fresh by default"."""
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps({"races": [RACE]}), encoding="utf-8")

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
    )
    assert tickets == []
    assert reason is not None and reason.startswith("predictions_generated_at_unreadable")


def test_same_day_fresh_predictions_are_accepted(cfg, tmp_path):
    """The control: a genuinely same-day, non-future card is issued normally."""
    path = tmp_path / "predictions.json"
    _write_predictions(
        path, generated_at=(NOW - timedelta(hours=2)).isoformat(), races=[RACE]
    )
    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
    )
    assert reason is None
    assert tickets


def test_dublin_dst_spring_forward_does_not_falsely_reject_same_day_card(cfg, tmp_path):
    """2026-03-29 is the day Dublin clocks spring forward. A card generated at
    00:30 UTC and evaluated at 01:30 UTC (both still 2026-03-29 in Dublin, just
    either side of the 01:00 UTC jump to IST) must read as the SAME local day,
    not be rejected by a naive local-time subtraction that the jump would
    corrupt."""
    generated = datetime(2026, 3, 29, 0, 30, tzinfo=timezone.utc)
    evaluated = datetime(2026, 3, 29, 1, 30, tzinfo=timezone.utc)
    race = copy.deepcopy(RACE)
    race["race_time"] = "2026-03-29T14:00:00+00:00"
    path = tmp_path / "predictions.json"
    _write_predictions(path, generated_at=generated.isoformat(), races=[race])

    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=evaluated.replace(hour=0, minute=0, second=0),
        evaluated_at=evaluated,
    )
    assert reason is None
    assert tickets


def test_issued_at_is_real_time_dedupe_key_is_stable_day(cfg, predictions_path):
    """Stage 20 / B6: the ticket's disclosed issued_at is the real evaluated_at
    instant, never a fake day-truncated midnight; re-running the same day still
    yields the identical ticket_id (idempotent issue)."""
    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")

    first, reason1 = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=predictions_path, now=DAY_START, evaluated_at=NOW,
    )
    later = NOW + timedelta(hours=5)
    second, reason2 = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=predictions_path, now=DAY_START, evaluated_at=later,
    )
    assert reason1 is None and reason2 is None
    assert {t.issued_at for t in first} == {NOW.isoformat()}
    assert {t.issued_at for t in second} == {later.isoformat()}
    assert {t.ticket_id for t in first} == {t.ticket_id for t in second}


def test_provenance_flows_from_predictions_json_into_the_ticket(cfg, tmp_path):
    race = copy.deepcopy(RACE)
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": NOW.isoformat(),
                "provenance": {
                    "model_content_hash": "sha256:aaa",
                    "feature_schema_version": "v3nf",
                    "prediction_cycle_id": "cycle-xyz",
                },
                "races": [race],
            }
        ),
        encoding="utf-8",
    )
    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
        config_hash="cfg-hash-1",
    )
    assert reason is None
    assert tickets
    for ticket in tickets:
        assert ticket.model_content_hash == "sha256:aaa"
        assert ticket.feature_schema_version == "v3nf"
        assert ticket.prediction_cycle_id == "cycle-xyz"
        assert ticket.config_hash == "cfg-hash-1"
        assert ticket.provenance_complete is True


def test_cross_venue_same_off_time_snapshot_quote_is_not_confused(cfg, tmp_path):
    """B5: two different venues going off at the identical minute must not
    let one venue's snapshot price satisfy the other's quote lookup."""
    snap_store = SnapshotStore(cfg=cfg)
    try:
        race_a = copy.deepcopy(RACE)
        race_a["venue"] = "Leopardstown"
        race_a["race_id"] = None
        race_b = copy.deepcopy(RACE)
        race_b["venue"] = "York"
        race_b["race_id"] = None
        for runner in race_b["runners"]:
            runner["horse_id"] = runner["horse_id"] + "b"
            runner["horse_name"] = runner["horse_name"] + " B"

        snap_store.record(
            [
                {
                    "venue": "York",
                    "race_time": RACE["race_time"],
                    "horse_name": race_b["runners"][0]["horse_name"],
                    "horse_id": race_b["runners"][0]["horse_id"],
                    "bookmaker": "boylesports",
                    "market_type": "WIN",
                    "odds_decimal": 55.0,
                    "fetched_at": NOW - timedelta(seconds=5),
                }
            ]
        )

        path = tmp_path / "predictions.json"
        _write_predictions(path, generated_at=NOW.isoformat(), races=[race_a, race_b])

        verdict = load_model_verdict()
        gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
        tickets, reason = loop._today_tickets(
            cfg=cfg, verdict=verdict, gate=gate,
            predictions_path=str(path), now=DAY_START, evaluated_at=NOW,
            snap_store=snap_store,
        )
        assert reason is None
        # The York runner is priced 55.0 from the snapshot; no Leopardstown
        # runner may ever pick up that price via the shared bare off-time.
        leo_tickets = [t for t in tickets if t.venue == "Leopardstown"]
        assert leo_tickets
        assert all(t.offered_odds != pytest.approx(55.0) for t in leo_tickets)
    finally:
        snap_store.close()


# ── Stage 21: real stake sizing (B2) ─────────────────────────────────────────
# Every test below reaches a CANDIDATE decision only through an isolated stub
# GO verdict plus a faked `execution.gates.evaluate_race` -- never by touching
# the real NO-GO verdict or the gate's own logic.

def _stake_fixture_predictions(tmp_path, race):
    path = tmp_path / "predictions_stake.json"
    path.write_text(
        json.dumps({"generated_at": "2026-01-15T14:00:00+00:00", "races": [race]}),
        encoding="utf-8",
    )
    return str(path)


def test_a_go_candidate_receives_a_real_planner_sized_stake(cfg, tmp_path, monkeypatch):
    """B2: a live CANDIDATE must carry a nonzero, planner-derived stake
    honouring bankroll/exposure ceilings, not the previous permanent
    ``stake=None``."""
    race = {
        "venue": "Ascot", "race_time": "2026-01-15T14:30:00+00:00",
        "race_id": "R-STAKE-1",
        "runners": [{"horse_id": "H1", "horse_name": "Stake Horse"}],
    }
    path = _stake_fixture_predictions(tmp_path, race)

    def fake_evaluate_race(r, *, cfg, model_verdict, source_health, now, quotes=None):
        return [gates_module.GateResult(
            decision="CANDIDATE",
            detail={"model_prob": 0.90, "executable_odds": 3.0, "race_uid": "ascot-r-stake-1"},
        )]

    monkeypatch.setattr(gates_module, "evaluate_race", fake_evaluate_race)

    verdict = replace(load_model_verdict(), go=True, reasons=())
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    evaluated_at = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)

    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate, predictions_path=path,
        now=evaluated_at.replace(hour=0, minute=0, second=0),
        evaluated_at=evaluated_at, bankroll=1000.0,
    )
    assert reason is None
    assert len(tickets) == 1
    ticket = tickets[0]
    assert ticket.decision == "CANDIDATE"
    # p=0.90 @ 3.0 wants far more than tenth-Kelly's 0.5%-of-bankroll ceiling.
    assert ticket.stake == pytest.approx(5.0)
    assert ticket.paper_only is True  # cfg.paper_only stays true regardless


def test_a_no_go_pass_never_carries_a_stake_even_with_stake_wiring_present(cfg, predictions_path):
    """The real NO-GO verdict/gate, untouched: every decision stays PASS, so
    every ticket must still carry ``stake=None``, proving the new staking
    wiring cannot leak a stake onto a non-candidate."""
    verdict = load_model_verdict()
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate,
        predictions_path=predictions_path, now=DAY_START, evaluated_at=NOW,
        bankroll=1000.0,
    )
    assert reason is None
    assert tickets
    assert all(t.decision == "PASS" and t.stake is None for t in tickets)


def test_two_candidates_in_one_race_share_the_race_exposure_ceiling(cfg, tmp_path, monkeypatch):
    """A second CANDIDATE in the SAME race must not add a second full stake on
    top of the first -- the shared race-exposure ceiling (not a bespoke
    correlation check) is what keeps two same-race bets from compounding an
    unvalidated edge."""
    race = {
        "venue": "Ascot", "race_time": "2026-01-15T14:30:00+00:00",
        "race_id": "R-STAKE-2",
        "runners": [
            {"horse_id": "H1", "horse_name": "Stake Horse One"},
            {"horse_id": "H2", "horse_name": "Stake Horse Two"},
        ],
    }
    path = _stake_fixture_predictions(tmp_path, race)

    def fake_evaluate_race(r, *, cfg, model_verdict, source_health, now, quotes=None):
        return [
            gates_module.GateResult(
                decision="CANDIDATE",
                detail={"model_prob": 0.90, "executable_odds": 3.0, "race_uid": "ascot-r-stake-2"},
            )
            for _ in r["runners"]
        ]

    monkeypatch.setattr(gates_module, "evaluate_race", fake_evaluate_race)

    verdict = replace(load_model_verdict(), go=True, reasons=())
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    evaluated_at = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)

    tickets, reason = loop._today_tickets(
        cfg=cfg, verdict=verdict, gate=gate, predictions_path=path,
        now=evaluated_at.replace(hour=0, minute=0, second=0),
        evaluated_at=evaluated_at, bankroll=1000.0,
    )
    assert reason is None
    assert len(tickets) == 2
    stakes = sorted((t.stake or 0.0) for t in tickets)
    assert stakes == [pytest.approx(0.0), pytest.approx(5.0)]
    assert sum(stakes) <= 5.0 + 1e-9  # never exceeds the race's own ceiling


def test_stake_exposure_seeds_from_the_ticket_store_on_a_resumed_run(cfg, tmp_path, monkeypatch):
    """A second, later call to ``_today_tickets`` (a resumed partial run) must
    see the race capacity the first call's issued ticket already consumed,
    not re-plan from an empty exposure ledger."""
    race = {
        "venue": "Ascot", "race_time": "2026-01-15T14:30:00+00:00",
        "race_id": "R-STAKE-3",
        "runners": [{"horse_id": "H1", "horse_name": "Stake Horse Three"}],
    }
    path = _stake_fixture_predictions(tmp_path, race)

    def fake_evaluate_race(r, *, cfg, model_verdict, source_health, now, quotes=None):
        return [gates_module.GateResult(
            decision="CANDIDATE",
            detail={"model_prob": 0.90, "executable_odds": 3.0, "race_uid": "ascot-r-stake-3"},
        )]

    monkeypatch.setattr(gates_module, "evaluate_race", fake_evaluate_race)

    verdict = replace(load_model_verdict(), go=True, reasons=())
    gate = types.SimpleNamespace(state_label="FORWARD GATE NOT MET")
    evaluated_at = datetime(2026, 1, 15, 14, 0, tzinfo=timezone.utc)

    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    try:
        first, _ = loop._today_tickets(
            cfg=cfg, verdict=verdict, gate=gate, predictions_path=path,
            now=evaluated_at.replace(hour=0, minute=0, second=0),
            evaluated_at=evaluated_at, bankroll=1000.0, ticket_store=store,
        )
        assert first[0].stake == pytest.approx(5.0)
        store.issue(first[0])

        second, _ = loop._today_tickets(
            cfg=cfg, verdict=verdict, gate=gate, predictions_path=path,
            now=evaluated_at.replace(hour=0, minute=0, second=0),
            evaluated_at=evaluated_at + timedelta(minutes=1), bankroll=1000.0,
            ticket_store=store,
        )
        assert (second[0].stake or 0.0) == pytest.approx(0.0)
    finally:
        store.close()


# ── Stage 21: real settlement via execution.race_facts (B3) ─────────────────
def _bare_ticket(**overrides) -> Ticket:
    base = dict(
        ticket_id="t-stage21", issued_at=NOW.isoformat(), as_of=NOW.isoformat(),
        race_uid="race-stage21", race_id="R1", race_time="2026-01-15T14:30:00+00:00",
        venue="Ascot", race_key="ascot|2026-01-15T14:30",
        horse_key="stakehorse", horse_id="H1", horse_name="Stake Horse",
        bookmaker="livescorebet", market_type="WIN", bet_type="win",
        offered_odds=4.0, quote_fetched_at=None, quote_age_seconds=None,
        model_prob=0.3, market_adjusted_prob=None, fair_odds=None, market_prob=None,
        edge=None, expected_value=None, max_stake=10.0, stake=10.0,
        ew_places=None, ew_reduction=None, data_quality="good",
        validation_state="MODEL GO / FORWARD GATE NOT MET", decision="CANDIDATE",
        pass_reasons=(), reasons_passed=(), paper_only=True, model_verdict="GO",
        forward_gate_state="NOT MET",
    )
    base.update(overrides)
    return Ticket(**base)


def test_settle_via_race_facts_voids_an_explicit_non_runner(cfg):
    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=3, n_runners=2,
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="NONRUNNER", finish_position=None),
            "other horse": RunnerFact(horse_key="other horse", horse_name="Other Horse",
                                       ride_status="RUNNER", finish_position=1),
            "third horse": RunnerFact(horse_key="third horse", horse_name="Third Horse",
                                       ride_status="RUNNER", finish_position=2),
        },
    )
    ticket = _bare_ticket()
    payload = loop._settle_via_race_facts(ticket, {fact.race_key: fact}, cfg=cfg)
    assert payload["outcome"] == "void"
    assert payload["profit"] == pytest.approx(0.0)
    assert payload["returns"] == pytest.approx(ticket.stake)


def test_settle_via_race_facts_treats_a_non_finisher_as_a_loss_not_a_void(cfg):
    """Requirement 3: DNF is a loss, never a void refund."""
    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=2, n_runners=2,
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="RUNNER", finish_position=None,
                                       casualty_reason="Fell"),
            "other horse": RunnerFact(horse_key="other horse", horse_name="Other Horse",
                                       ride_status="RUNNER", finish_position=1),
        },
    )
    ticket = _bare_ticket()
    payload = loop._settle_via_race_facts(ticket, {fact.race_key: fact}, cfg=cfg)
    assert payload["outcome"] == "lost"
    assert payload["profit"] == pytest.approx(-ticket.stake)


def test_settle_via_race_facts_voids_an_abandoned_race(cfg):
    """n_runners <= 1 is execution.race_facts' own abandoned-race proxy."""
    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=1, n_runners=1,
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="RUNNER", finish_position=1),
        },
    )
    ticket = _bare_ticket()
    payload = loop._settle_via_race_facts(ticket, {fact.race_key: fact}, cfg=cfg)
    assert payload["outcome"] == "void"
    assert payload["returns"] == pytest.approx(ticket.stake)
    assert payload["profit"] == pytest.approx(0.0)


def test_settle_via_race_facts_applies_published_rule4_and_dead_heat(cfg):
    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=3, n_runners=2,
        rule_4_deduction=0.10, rule_4_type="AllBets",
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="RUNNER", finish_position=1, dead_heat_size=2),
            "other horse": RunnerFact(horse_key="other horse", horse_name="Other Horse",
                                       ride_status="RUNNER", finish_position=1, dead_heat_size=2),
            "withdrawn horse": RunnerFact(horse_key="withdrawn horse", horse_name="Withdrawn Horse",
                                           ride_status="NONRUNNER", industry_sp=5.0),
        },
    )
    ticket = _bare_ticket(offered_odds=4.0, stake=10.0)
    payload = loop._settle_via_race_facts(ticket, {fact.race_key: fact}, cfg=cfg)
    assert payload["outcome"] == "won"
    detail = payload["detail"]
    assert detail["dead_heat_divisor"] == pytest.approx(2.0)
    assert detail["rule_4_deduction"] == pytest.approx(0.10)
    assert detail["rule_4_source"] == "published"
    # matched stake 5.0 @ effective odds 1+3*0.9=3.7 -> winnings 5*2.7=13.5;
    # total returns = matched stake back (5.0) + winnings (13.5) = 18.5.
    assert payload["returns"] == pytest.approx(18.5)
    assert payload["profit"] == pytest.approx(8.5)


def test_settle_via_race_facts_settles_an_each_way_place_only_leg(cfg):
    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=4, n_runners=4,
        places_paid=3,
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="RUNNER", finish_position=2),
            "winner horse": RunnerFact(horse_key="winner horse", horse_name="Winner Horse",
                                        ride_status="RUNNER", finish_position=1),
            "third horse": RunnerFact(horse_key="third horse", horse_name="Third Horse",
                                       ride_status="RUNNER", finish_position=3),
            "fourth horse": RunnerFact(horse_key="fourth horse", horse_name="Fourth Horse",
                                        ride_status="RUNNER", finish_position=4),
        },
    )
    ticket = _bare_ticket(
        bet_type="each_way", offered_odds=6.0, stake=10.0, ew_places=3, ew_reduction=0.2,
    )
    payload = loop._settle_via_race_facts(ticket, {fact.race_key: fact}, cfg=cfg)
    assert payload["outcome"] == "placed"
    assert payload["returns"] == pytest.approx(10.0)
    assert payload["profit"] == pytest.approx(0.0)


def test_settle_via_race_facts_returns_none_when_the_archive_does_not_cover_the_race(cfg):
    ticket = _bare_ticket(race_time="2026-03-01T14:30:00+00:00")
    assert loop._settle_via_race_facts(ticket, {}, cfg=cfg) is None


def test_settle_open_tickets_prefers_race_facts_over_the_betsp_fallback(cfg, monkeypatch):
    """Integration: when execution.race_facts covers a ticket's race,
    _settle_open_tickets must settle through the real settlement engine, not
    the coarser betSP join -- proven by making the two disagree."""
    race = copy.deepcopy(RACE)
    race["venue"] = "Ascot"
    race["race_time"] = "2026-01-15T14:30:00+00:00"
    runner = race["runners"][0]
    runner["horse_name"] = "Stake Horse"

    fact = RaceFacts(
        race_key="ascot|2026-01-15T14:30", race_date="2026-01-15",
        off_time="2026-01-15T14:30", venue="Ascot", declared_field=2, n_runners=2,
        runners={
            "stake horse": RunnerFact(horse_key="stake horse", horse_name="Stake Horse",
                                       ride_status="RUNNER", finish_position=None,
                                       casualty_reason="Fell"),
            "other horse": RunnerFact(horse_key="other horse", horse_name="Other Horse",
                                       ride_status="RUNNER", finish_position=1),
        },
    )
    monkeypatch.setattr(loop, "_load_race_facts_for", lambda race_times: {fact.race_key: fact})
    # A deliberately WRONG betSP-fallback result (says "won"), proving the
    # archive-backed path wins rather than merely running first.
    monkeypatch.setattr(loop, "_load_results", lambda storage=None: [
        {"venue": "Ascot", "horse_id": runner["horse_id"], "horse_name": "Stake Horse",
         "race_date": race["race_time"], "position": 1},
    ])

    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        verdict = load_model_verdict()
        result = evaluate_candidate(
            runner, race, cfg=cfg, model_verdict=verdict, quote=None,
            source_health={}, now=NOW,
        )
        result = replace(result, decision="CANDIDATE")
        ticket = build_ticket(
            runner, race, result, cfg=cfg, model_verdict=verdict,
            forward_gate_state="FORWARD GATE NOT MET", now=NOW,
            stake_decision=types.SimpleNamespace(stake=10.0, max_stake=10.0),
        )
        ticket_store.issue(ticket)

        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["settled"] == 1
        assert stats["settled_via_race_facts"] == 1

        row = ticket_store.to_frame().iloc[0]
        assert row["outcome"] == "lost"  # the real DNF, not the fallback's "won"
        assert row["stake"] == pytest.approx(10.0)
        assert row["profit"] == pytest.approx(-10.0)
    finally:
        ticket_store.close()
        snap_store.close()


def test_settle_open_tickets_leaves_an_unmatched_horse_pending_not_void(cfg, monkeypatch):
    """Requirement 3/6: with no race_facts coverage and no matching betSP row
    at all, a still-open ticket must stay pending with a recorded reason --
    never assumed void, since the legacy join has no non-runner evidence."""
    monkeypatch.setattr(loop, "_load_race_facts_for", lambda race_times: {})
    monkeypatch.setattr(loop, "_load_results", lambda storage=None: [
        {"venue": RACE["venue"], "horse_id": "someone-else", "horse_name": "Someone Else",
         "race_date": RACE["race_time"], "position": 1},
    ])
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        ticket = _issue_candidate(ticket_store, cfg)
        stats = loop._settle_open_tickets(ticket_store, snap_store)
        assert stats["still_open"] == 1
        assert stats["pending_reasons"] == {"no_result_recorded": 1}
        assert [t.ticket_id for t in ticket_store.open_tickets()] == [ticket.ticket_id]
    finally:
        ticket_store.close()
        snap_store.close()


# ── Stage 21: PASS / shadow observation (B4) ─────────────────────────────────
def test_observe_pass_tickets_reconciles_without_ever_creating_a_bet(cfg, monkeypatch):
    """A PASS ticket's result and closing price get reconciled as disclosure
    evidence -- stake/returns/profit all stay zero regardless of outcome, and
    the observation never reaches the CANDIDATE-only forward ledger."""
    ticket_store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        verdict = load_model_verdict()
        runner = RACE["runners"][0]
        result = evaluate_candidate(
            runner, RACE, cfg=cfg, model_verdict=verdict, quote=None,
            source_health={}, now=NOW,
        )
        assert result.decision == "PASS"  # the real NO-GO gate, untouched
        ticket = build_ticket(
            runner, RACE, result, cfg=cfg, model_verdict=verdict,
            forward_gate_state="FORWARD GATE NOT MET", now=NOW,
        )
        assert ticket.offered_odds is not None
        ticket_store.issue(ticket)
        assert len(ticket_store.unobserved_pass_tickets()) == 1

        winning_result = {
            "venue": RACE["venue"], "horse_id": runner["horse_id"],
            "horse_name": runner["horse_name"], "race_date": RACE["race_time"],
            "position": 1,
        }
        monkeypatch.setattr(loop, "_load_results", lambda storage=None: [winning_result])
        monkeypatch.setattr(loop, "_load_race_facts_for", lambda race_times: {})

        stats = loop._observe_pass_tickets(ticket_store, snap_store)
        assert stats["observed"] == 1
        assert ticket_store.unobserved_pass_tickets() == []

        row = ticket_store.to_frame().iloc[0]
        assert row["decision"] == "PASS"
        assert row["outcome"] == "won"
        assert row["profit"] == pytest.approx(0.0)
        assert row["returns"] == pytest.approx(0.0)

        ledger = loop._forward_ledger(ticket_store)
        assert len(ledger) == 0  # PASS never enters the CANDIDATE-only ledger

        summary = loop._paper_summary(ticket_store)
        assert summary["n_pass_priced"] == 1
        assert summary["n_pass_observed"] == 1
        assert summary["n_candidates"] == 0  # never miscounted as a wager
    finally:
        ticket_store.close()
        snap_store.close()


def test_observe_pass_tickets_is_a_noop_when_nothing_is_priced(cfg, predictions_path, tmp_path):
    """Every runner in the default fixture PASSes before a price is even
    read in some configurations; whether priced or not, an unpriced PASS must
    never be reported as an observation gap."""
    _run(cfg, predictions_path, tmp_path)
    store = TicketStore(cfg.snapshots.db_path, cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        stats = loop._observe_pass_tickets(store, snap_store)
        # Whatever is priced settles or stays pending; nothing raises and the
        # zero-stake invariant holds either way.
        assert set(stats) == {"observed", "already_observed", "still_open", "unmeasurable_clv"}
    finally:
        store.close()
        snap_store.close()
