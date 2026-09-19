"""Requirement 9: one test per block code, plus the fail-closed paths.

The source-health lookup is always injected here, so no test reads or writes the
real ``data/source_health.json`` and no test touches the network.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from execution.config import ExecutionConfig
from execution.safeguards import (
    BLOCK_CODES,
    PAPER_ONLY_CODE,
    SafeguardCheck,
    Safeguards,
    SafeguardState,
    SafeguardViolation,
)

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
RACE_UID = "2026-07-27 13:00"


def healthy(_source=None):
    return {"status": "ok", "age_seconds": 30.0, "last_row_count": 12}


@pytest.fixture
def cfg():
    return ExecutionConfig.from_config({})


@pytest.fixture
def guards(cfg):
    return Safeguards(cfg, source_health_fn=healthy, now_fn=lambda: NOW)


@pytest.fixture
def state():
    return SafeguardState(
        bankroll=1000.0,
        initial_bankroll=1000.0,
        day_profit=0.0,
        day_staked=0.0,
        open_tickets=0,
        trading_day="2026-07-27",
    )


@pytest.fixture
def candidate():
    return {
        "race_uid": RACE_UID,
        "race_time": (NOW + timedelta(minutes=60)).isoformat(),
        "horse_key": "HORSE-1",
        "bet_type": "win",
        "stake": 5.0,
        "sources": ["betfair"],
    }


def _blocks(guards, candidate, state, **kw):
    check = guards.check(candidate, state, **kw)
    assert isinstance(check, SafeguardCheck)
    return check


# ── the allowed case ─────────────────────────────────────────────────────────
def test_clean_candidate_is_allowed(guards, candidate, state):
    check = _blocks(guards, candidate, state)
    assert check.allowed is True
    assert check.blocks == ()
    assert check.messages == ()
    # Paper-only rides along as disclosure, never as a block.
    assert check.detail["paper_only"]["paper_only"] is True
    assert check.detail["paper_only"]["code"] == PAPER_ONLY_CODE
    assert PAPER_ONLY_CODE not in BLOCK_CODES
    assert check.to_dict()["allowed"] is True


# ── one test per block code ──────────────────────────────────────────────────
def test_bankroll_stop_loss_blocks(guards, candidate, state):
    broke = SafeguardState(bankroll=800.0, initial_bankroll=1000.0)
    check = _blocks(guards, candidate, broke)
    assert check.allowed is False
    assert "bankroll_stop_loss" in check.blocks


def test_daily_loss_limit_blocks(guards, candidate, state):
    losing = SafeguardState(bankroll=980.0, initial_bankroll=1000.0, day_profit=-20.0)
    check = _blocks(guards, candidate, losing)
    assert check.allowed is False
    assert "daily_loss_limit" in check.blocks


def test_daily_exposure_limit_blocks(guards, candidate, state):
    spent = SafeguardState(bankroll=1000.0, initial_bankroll=1000.0, day_staked=29.0)
    check = _blocks(guards, candidate, spent)  # 29.00 + 5.00 > 30.00
    assert check.allowed is False
    assert "daily_exposure_limit" in check.blocks


def test_race_started_blocks_inside_the_buffer(guards, candidate, state):
    candidate["race_time"] = (NOW + timedelta(seconds=30)).isoformat()
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "race_started" in check.blocks


def test_missing_race_time_blocks(guards, candidate, state):
    candidate.pop("race_time")
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "race_started" in check.blocks

    candidate["race_time"] = "not-a-time"
    assert "race_started" in _blocks(guards, candidate, state).blocks


def test_duplicate_ticket_blocks(guards, candidate, state):
    existing = [
        {"race_uid": RACE_UID, "horse_key": "HORSE-1", "bet_type": "win", "stake": 5.0}
    ]
    check = _blocks(guards, candidate, state, existing_tickets=existing)
    assert check.allowed is False
    assert "duplicate_ticket" in check.blocks


def test_settled_ticket_is_not_a_duplicate(guards, candidate, state):
    existing = [
        {
            "race_uid": RACE_UID,
            "horse_key": "HORSE-1",
            "bet_type": "win",
            "stake": 5.0,
            "status": "settled",
        }
    ]
    check = _blocks(guards, candidate, state, existing_tickets=existing)
    assert check.allowed is True


def test_unidentifiable_candidate_blocks_as_duplicate(guards, candidate, state):
    candidate.pop("horse_key")
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "duplicate_ticket" in check.blocks


def test_correlated_bet_blocks_a_second_runner_in_the_same_race(guards, candidate, state):
    existing = [
        {"race_uid": RACE_UID, "horse_key": "HORSE-2", "bet_type": "win", "stake": 5.0}
    ]
    check = _blocks(guards, candidate, state, existing_tickets=existing)
    assert check.allowed is False
    assert "correlated_bet" in check.blocks
    assert "duplicate_ticket" not in check.blocks  # different runner


def test_stale_source_blocks(cfg, candidate, state):
    stale = Safeguards(
        cfg,
        source_health_fn=lambda src: {"status": "ok", "age_seconds": 1200.0},
        now_fn=lambda: NOW,
    )
    check = _blocks(stale, candidate, state)
    assert check.allowed is False
    assert "stale_source" in check.blocks


def test_source_without_a_known_age_blocks_as_stale(cfg, candidate, state):
    unknown_age = Safeguards(
        cfg,
        source_health_fn=lambda src: {"status": "ok", "age_seconds": None},
        now_fn=lambda: NOW,
    )
    assert "stale_source" in _blocks(unknown_age, candidate, state).blocks


def test_unhealthy_source_blocks(cfg, candidate, state):
    failing = Safeguards(
        cfg,
        source_health_fn=lambda src: {"status": "failing", "age_seconds": 10.0},
        now_fn=lambda: NOW,
    )
    check = _blocks(failing, candidate, state)
    assert check.allowed is False
    assert "unhealthy_source" in check.blocks


def test_missing_source_record_blocks(cfg, candidate, state):
    absent = Safeguards(cfg, source_health_fn=lambda src: {}, now_fn=lambda: NOW)
    assert "unhealthy_source" in _blocks(absent, candidate, state).blocks


def test_candidate_without_any_source_blocks(guards, candidate, state):
    candidate.pop("sources")
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "unhealthy_source" in check.blocks


def test_raising_source_health_fn_blocks(cfg, candidate, state):
    def boom(_source):
        raise OSError("source_health.json is unreadable")

    guards = Safeguards(cfg, source_health_fn=boom, now_fn=lambda: NOW)
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "unhealthy_source" in check.blocks


def test_max_open_tickets_blocks(guards, candidate, state):
    saturated = SafeguardState(
        bankroll=1000.0, initial_bankroll=1000.0, open_tickets=25
    )
    check = _blocks(guards, candidate, saturated)
    assert check.allowed is False
    assert "max_open_tickets" in check.blocks


def test_every_documented_block_code_is_reachable():
    """The vocabulary is a contract; no code may be documented but unreachable."""
    covered = {
        "bankroll_stop_loss",
        "daily_loss_limit",
        "daily_exposure_limit",
        "race_started",
        "duplicate_ticket",
        "correlated_bet",
        "stale_source",
        "unhealthy_source",
        "max_open_tickets",
    }
    assert set(BLOCK_CODES) == covered


# ── fail-closed and multi-block behaviour ────────────────────────────────────
def test_unreadable_stake_blocks_on_exposure(guards, candidate, state):
    candidate["stake"] = "five euro"
    check = _blocks(guards, candidate, state)
    assert check.allowed is False
    assert "daily_exposure_limit" in check.blocks


def test_unusable_bankroll_state_blocks(guards, candidate):
    check = _blocks(guards, candidate, SafeguardState(bankroll=None, initial_bankroll=0.0))
    assert check.allowed is False
    assert "bankroll_stop_loss" in check.blocks


def test_all_simultaneous_blocks_are_reported(cfg, state):
    """A candidate that is wrong four ways reports four codes, not the first one."""
    guards = Safeguards(
        cfg,
        source_health_fn=lambda src: {"status": "failing", "age_seconds": 10.0},
        now_fn=lambda: NOW,
    )
    candidate = {
        "race_uid": RACE_UID,
        "race_time": (NOW - timedelta(minutes=5)).isoformat(),  # already off
        "horse_key": "HORSE-1",
        "bet_type": "win",
        "stake": 5.0,
        "sources": ["betfair"],
    }
    broke = SafeguardState(
        bankroll=700.0, initial_bankroll=1000.0, day_profit=-300.0, open_tickets=30
    )
    existing = [
        {"race_uid": RACE_UID, "horse_key": "HORSE-1", "bet_type": "win", "stake": 5.0}
    ]
    check = _blocks(guards, candidate, broke, existing_tickets=existing)
    assert check.allowed is False
    assert {
        "bankroll_stop_loss",
        "daily_loss_limit",
        "race_started",
        "duplicate_ticket",
        "correlated_bet",
        "unhealthy_source",
        "max_open_tickets",
    } <= set(check.blocks)
    assert len(check.messages) == len(check.blocks)


def test_explicit_now_overrides_the_clock(guards, candidate, state):
    late = NOW + timedelta(minutes=61)  # the race has gone off by then
    check = _blocks(guards, candidate, state, now=late)
    assert "race_started" in check.blocks


# ── the paper-only boundary ──────────────────────────────────────────────────
def test_paper_only_is_disclosed_not_blocked(guards, candidate, state):
    check = _blocks(guards, candidate, state)
    assert guards.paper_only() is True
    assert PAPER_ONLY_CODE not in check.blocks


def test_assert_paper_only_raises_under_model_no_go(guards):
    with pytest.raises(SafeguardViolation) as excinfo:
        guards.assert_paper_only(model_go=False, forward_gate_passed=False)
    assert excinfo.value.code == PAPER_ONLY_CODE
    assert "PAPER-ONLY" in excinfo.value.message


def test_assert_paper_only_raises_even_with_a_model_go_while_paper_only(guards):
    """paper_only:true is a kill switch; evidence elsewhere cannot override it."""
    with pytest.raises(SafeguardViolation):
        guards.assert_paper_only(model_go=True, forward_gate_passed=True)


def test_assert_paper_only_still_raises_without_the_forward_gate():
    live = ExecutionConfig.from_config({"execution": {"paper_only": False}})
    guards = Safeguards(live, source_health_fn=healthy, now_fn=lambda: NOW)
    with pytest.raises(SafeguardViolation):
        guards.assert_paper_only(model_go=True, forward_gate_passed=False)
    with pytest.raises(SafeguardViolation):
        guards.assert_paper_only(model_go=False, forward_gate_passed=True)
    # Only the full evidence set is permitted through.
    guards.assert_paper_only(model_go=True, forward_gate_passed=True)
