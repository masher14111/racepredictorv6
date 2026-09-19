"""Gap accounting (Stage 6, requirement 5).

A day with no capture is a **gap**, not a day with zero qualifying bets. These
tests pin: a gap is recorded with its cause; a captured day is sticky against a
later failed poll; ``qualifying_weeks`` counts only captured days, gap-adjusted,
never wall-clock elapsed; and ``reconcile_gaps`` back-annotates days the loop
never ran on without touching days already recorded.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from execution import gap_ledger as gl

NOW = datetime(2026, 7, 28, 16, 0, tzinfo=timezone.utc)


def _snapshot_db(tmp_path, dates: list[str]) -> str:
    """A minimal odds_snapshots table with one row per given calendar date."""
    db_path = str(tmp_path / "snap.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE odds_snapshots (fetched_at TEXT)")
    conn.executemany(
        "INSERT INTO odds_snapshots (fetched_at) VALUES (?)",
        [(f"{d}T12:00:00+00:00",) for d in dates],
    )
    conn.commit()
    conn.close()
    return db_path


# ── record_day ────────────────────────────────────────────────────────────────
def test_record_day_captured_has_no_cause(tmp_path):
    path = str(tmp_path / "gap.json")
    days = gl.record_day(path, day=date(2026, 7, 28), captured=True, now=NOW)
    entry = days["2026-07-28"]
    assert entry["status"] == gl.CAPTURED
    assert entry["cause"] is None


def test_record_day_gap_carries_its_cause(tmp_path):
    path = str(tmp_path / "gap.json")
    days = gl.record_day(
        path, day=date(2026, 7, 28), captured=False, cause=gl.CAUSE_NO_RACING, now=NOW
    )
    entry = days["2026-07-28"]
    assert entry["status"] == gl.GAP
    assert entry["cause"] == gl.CAUSE_NO_RACING


def test_record_day_defaults_gap_cause_to_scraper_outage(tmp_path):
    path = str(tmp_path / "gap.json")
    days = gl.record_day(path, day=date(2026, 7, 28), captured=False, now=NOW)
    assert days["2026-07-28"]["cause"] == gl.CAUSE_SCRAPER_OUTAGE


def test_a_captured_day_is_sticky_against_a_later_failed_poll(tmp_path):
    """One successful poll is real evidence; a later failed poll the same day
    must not retroactively erase it."""
    path = str(tmp_path / "gap.json")
    gl.record_day(path, day=date(2026, 7, 28), captured=True, now=NOW)
    days = gl.record_day(path, day=date(2026, 7, 28), captured=False, now=NOW)
    assert days["2026-07-28"]["status"] == gl.CAPTURED


def test_gap_does_not_satisfy_the_gate_qualifying_weeks(tmp_path):
    """A recorded gap contributes zero qualifying weeks — it must never be
    interchangeable with a clean no-bet day."""
    path = str(tmp_path / "gap.json")
    gl.record_day(path, day=date(2026, 7, 28), captured=False, now=NOW)
    days = gl.load_ledger(path)
    assert gl.qualifying_weeks(days) == 0.0
    assert gl.is_qualifying_day(days, date(2026, 7, 28)) is False


# ── _captured_dates_from_snapshots ───────────────────────────────────────────
def test_captured_dates_from_snapshots_reads_distinct_dates(tmp_path):
    db = _snapshot_db(tmp_path, ["2026-07-27", "2026-07-27", "2026-07-28"])
    assert gl._captured_dates_from_snapshots(db) == {"2026-07-27", "2026-07-28"}


def test_captured_dates_from_snapshots_missing_db_is_empty(tmp_path):
    assert gl._captured_dates_from_snapshots(str(tmp_path / "nope.db")) == set()


# ── reconcile_gaps ────────────────────────────────────────────────────────────
def test_reconcile_gaps_backfills_unrecorded_days_only(tmp_path):
    db = _snapshot_db(tmp_path, ["2026-07-26"])  # only one of three days captured
    path = str(tmp_path / "gap.json")
    # 2026-07-27 already has an explicit (gap) entry; reconcile must not touch it.
    gl.record_day(path, day=date(2026, 7, 27), captured=False,
                   cause=gl.CAUSE_MACHINE_OFF, now=NOW)

    days = gl.reconcile_gaps(
        path, window_start=date(2026, 7, 26), db_path=db, today=date(2026, 7, 28)
    )

    assert days["2026-07-26"]["status"] == gl.CAPTURED  # backfilled from snapshots
    assert days["2026-07-27"]["cause"] == gl.CAUSE_MACHINE_OFF  # untouched
    # `today` itself is never back-annotated -- the loop only covers days
    # strictly before it, so a day still in progress is never guessed at.
    assert "2026-07-28" not in days


def test_reconcile_gaps_never_reaches_today(tmp_path):
    db = _snapshot_db(tmp_path, [])
    path = str(tmp_path / "gap.json")
    days = gl.reconcile_gaps(
        path, window_start=date(2026, 7, 28), db_path=db, today=date(2026, 7, 28)
    )
    assert days == {}  # window_start == today: nothing strictly before it


def test_reconcile_gaps_uses_no_run_recorded_cause(tmp_path):
    db = _snapshot_db(tmp_path, [])
    path = str(tmp_path / "gap.json")
    days = gl.reconcile_gaps(
        path, window_start=date(2026, 7, 27), db_path=db, today=date(2026, 7, 28)
    )
    assert days["2026-07-27"]["cause"] == gl.CAUSE_NO_RUN_RECORDED


# ── qualifying_weeks / qualifying_dates ──────────────────────────────────────
def test_qualifying_weeks_counts_only_captured_days(tmp_path):
    path = str(tmp_path / "gap.json")
    for i, captured in enumerate([True, True, False, True, True, True, True]):
        gl.record_day(path, day=date(2026, 7, 22 + i), captured=captured, now=NOW)
    days = gl.load_ledger(path)
    assert gl.qualifying_weeks(days) == 6 / 7.0


def test_qualifying_weeks_respects_window_start_filter(tmp_path):
    path = str(tmp_path / "gap.json")
    gl.record_day(path, day=date(2026, 7, 1), captured=True, now=NOW)  # before window
    gl.record_day(path, day=date(2026, 7, 28), captured=True, now=NOW)  # in window
    days = gl.load_ledger(path)
    assert gl.qualifying_weeks(days, window_start=date(2026, 7, 20)) == 1 / 7.0
    assert gl.qualifying_weeks(days) == 2 / 7.0
