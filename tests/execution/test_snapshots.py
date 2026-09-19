"""Point-in-time guarantees of the append-only odds snapshot store.

Each test pins one of the properties that make a replay honest: history cannot be
rewritten, a read can never see a price that did not exist yet, and a stale quote
is never handed back as if it were executable. The closing price is reachable by
exactly one function, and only for measuring CLV.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from execution.config import ExecutionConfig
from execution.snapshots import (
    AppendOnlyViolation,
    FuturePriceError,
    SnapshotStore,
    assert_point_in_time,
    horse_key,
    race_uid,
)
from utils.storage.migrations import current_version, target_version

T0 = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)
RACE_TIME = datetime(2026, 7, 27, 13, 15, 0, tzinfo=timezone.utc)
UID = race_uid(race_time=RACE_TIME)
HORSE = horse_key("Little Lady Karen")


def _cfg(max_age_seconds: float = 300.0) -> ExecutionConfig:
    return ExecutionConfig.from_config(
        {"execution": {"snapshots": {"max_age_seconds": max_age_seconds}}}
    )


def _row(*, minutes: float, odds: float, book: str = "livescorebet", horse: str = "Little Lady Karen") -> dict:
    return {
        "race_id": "SBTE_2_1028515026",
        "race_time": RACE_TIME,
        "venue": "Ayr",
        "horse_name": horse,
        "source": book,
        "market_type": "WIN",
        "odds_decimal": odds,
        "ew_places": 2,
        "ew_reduction": 0.25,
        "field_size": 5,
        "booksum": 1.158789,
        "validation_status": "VALID",
        "fetched_at": T0 + timedelta(minutes=minutes),
    }


@pytest.fixture
def store(tmp_path):
    s = SnapshotStore(db_path=str(tmp_path / "snap.db"), cfg=_cfg())
    yield s
    s.close()


# 1 ── schema ────────────────────────────────────────────────────────────────
def test_migrations_reach_version_5_with_both_tables(store):
    conn = sqlite3.connect(store.db_path)
    try:
        # >= 5, not ==: later stages (e.g. Stage 14's text_archive, migration 6)
        # add migrations to the same shared db without changing this contract.
        assert target_version() >= 5
        assert current_version(conn) == target_version()

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"odds_snapshots", "paper_tickets"} <= tables

        triggers = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='trigger'"
            ).fetchall()
        }
        assert {"odds_snapshots_no_update", "odds_snapshots_no_delete"} <= triggers
    finally:
        conn.close()


def test_paper_tickets_cannot_hold_a_real_money_row(store):
    """The CHECK makes a non-paper ticket unrepresentable, not merely refused."""
    conn = sqlite3.connect(store.db_path)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO paper_tickets (ticket_id, issued_at, race_uid, horse_key,"
                " validation_state, decision, paper_only) VALUES (?,?,?,?,?,?,0)",
                ("t1", "2026-07-27T12:00:00.000000+00:00", UID, HORSE, "OK", "PASS"),
            )
    finally:
        conn.close()


# 2 ── append-only ───────────────────────────────────────────────────────────
def test_update_and_delete_are_refused_by_the_triggers(store):
    store.record([_row(minutes=0, odds=2.2)])
    conn = sqlite3.connect(store.db_path)
    try:
        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as err:
            conn.execute("UPDATE odds_snapshots SET odds_decimal = 99.0")
        assert "append-only" in str(err.value)

        with pytest.raises((sqlite3.IntegrityError, sqlite3.OperationalError)) as err:
            conn.execute("DELETE FROM odds_snapshots")
        assert "append-only" in str(err.value)
    finally:
        conn.close()
    assert store.count() == 1


def test_store_translates_a_rewrite_attempt_into_append_only_violation(store):
    store.record([_row(minutes=0, odds=2.2)])
    with pytest.raises(AppendOnlyViolation):
        store.execute("UPDATE odds_snapshots SET odds_decimal = 99.0")
    with pytest.raises(AppendOnlyViolation):
        store.execute("DELETE FROM odds_snapshots")


# 3 ── the point-in-time contract ────────────────────────────────────────────
def test_quotes_as_of_never_returns_a_future_price(store):
    store.record(
        [
            _row(minutes=0, odds=2.20),
            _row(minutes=10, odds=2.50),
            _row(minutes=20, odds=3.00),  # after as_of
            _row(minutes=40, odds=5.00),  # after as_of
        ]
    )
    as_of = T0 + timedelta(minutes=12)
    frame = store.quotes_as_of(UID, as_of)

    assert not frame.empty
    stamps = [pd.Timestamp(v).to_pydatetime() for v in frame["fetched_at"]]
    assert max(stamps) <= as_of
    # The point-in-time view is the newest row per (horse, bookmaker).
    assert list(frame["odds_decimal"]) == [2.50]
    assert_point_in_time(frame, as_of)


def test_latest_quote_picks_newest_at_or_before_as_of(store):
    store.record(
        [
            _row(minutes=0, odds=2.20),
            _row(minutes=3, odds=2.60),
            _row(minutes=9, odds=9.99),  # newest overall, but in the future
        ]
    )
    quote = store.latest_quote(UID, HORSE, T0 + timedelta(minutes=5))

    assert quote is not None
    assert quote.odds_decimal == 2.60
    assert quote.fetched_at == T0 + timedelta(minutes=3)
    assert quote.age_seconds == pytest.approx(120.0)
    assert quote.is_stale is False
    assert quote.to_dict()["odds_decimal"] == 2.60


def test_best_quote_compares_only_in_age_quotes(store):
    """A cold book must not win "best available" just by having stopped updating."""
    store.record(
        [
            _row(minutes=0, odds=8.00, book="boylesports"),  # stale at as_of
            _row(minutes=8, odds=2.50, book="livescorebet"),
            _row(minutes=9, odds=3.20, book="paddy_power"),
        ]
    )
    as_of = T0 + timedelta(minutes=10)  # max_age 300s -> the 10-min-old row is stale

    best = store.best_quote(UID, HORSE, as_of)
    assert best is not None
    assert best.bookmaker == "paddy_power"
    assert best.odds_decimal == 3.20

    # Widening the age window lets the older, longer price back in.
    wide = store.best_quote(UID, HORSE, as_of, max_age_seconds=3600)
    assert wide is not None and wide.bookmaker == "boylesports"


def test_book_as_of_returns_one_row_per_runner_for_one_book(store):
    store.record(
        [
            _row(minutes=0, odds=2.20, horse="Little Lady Karen"),
            _row(minutes=2, odds=2.40, horse="Little Lady Karen"),
            _row(minutes=2, odds=3.25, horse="Bymiddaytomorrow"),
            _row(minutes=2, odds=7.00, horse="March Lilly", book="paddy_power"),
        ]
    )
    board = store.book_as_of(UID, T0 + timedelta(minutes=3), "livescorebet")

    assert set(board["bookmaker"]) == {"livescorebet"}
    assert sorted(board["odds_decimal"]) == [2.40, 3.25]


def test_book_as_of_reflects_an_incomplete_mixed_source_book_honestly(store):
    """A captured board that is short of the declared field must stay visibly
    short -- ``book_as_of`` never invents a row for a runner it never observed,
    so a consumer comparing row count against the recorded ``field_size`` can
    fail closed on a partial capture instead of de-vigging a hole."""
    store.record(
        [
            _row(minutes=0, odds=2.20, horse="Little Lady Karen"),
            _row(minutes=0, odds=3.25, horse="Bymiddaytomorrow"),
            # "March Lilly" (the declared field_size=5's 3rd+ runner) never
            # arrives from this book -- e.g. a partial scrape or a runner this
            # source has not (yet) priced.
        ]
    )
    board = store.book_as_of(UID, T0 + timedelta(minutes=1), "livescorebet")

    assert len(board) == 2
    # field_size is what the source reported at capture time (see _row); the
    # gap between it and the observed row count is the completeness signal.
    assert int(board.iloc[0]["field_size"]) == 5
    assert set(board["horse_key"]) == {horse_key("Little Lady Karen"), horse_key("Bymiddaytomorrow")}


def test_a_runner_withdrawn_mid_capture_is_never_silently_still_priced(store):
    """The store has no explicit non-runner flag; a withdrawal is only visible
    as a runner that stops being re-captured. That must resolve to the
    conservative outcome -- the runner's last price ages into stale and is
    refused as executable -- rather than staying "fresh" forever because
    nothing ever contradicts it."""
    withdrawn = "Bymiddaytomorrow"
    store.record(
        [
            _row(minutes=0, odds=2.20, horse="Little Lady Karen"),
            _row(minutes=0, odds=6.00, horse=withdrawn),
        ]
    )
    # Only the still-running horse gets re-polled -- the withdrawn one is gone
    # from every subsequent board, exactly as a bookmaker drops it once
    # withdrawn.
    store.record([_row(minutes=4, odds=2.10, horse="Little Lady Karen")])

    as_of = T0 + timedelta(minutes=6)  # max_age 300s -> the 6-minute-old row is stale
    running = store.latest_quote(UID, HORSE, as_of)
    assert running is not None and running.odds_decimal == 2.10

    stale_withdrawn = store.latest_quote(UID, horse_key(withdrawn), as_of)
    assert stale_withdrawn is None  # refused: last price is now stale, never a fill

    frame = store.quotes_as_of(UID, as_of)
    withdrawn_row = frame[frame["horse_key"] == horse_key(withdrawn)].iloc[0]
    assert bool(withdrawn_row["is_stale"]) is True


# 5b ── market type isolation (D23) ──────────────────────────────────────────
def test_win_and_place_snapshots_are_independent_price_books(store):
    """WIN and PLACE are different price books for the same runner (D23); the
    store must key and read them independently, never conflating the two."""
    store.record(
        [
            _row(minutes=0, odds=4.00),  # WIN, from _row's default market_type
            dict(_row(minutes=0, odds=1.80), market_type="PLACE"),
        ]
    )
    win = store.latest_quote(UID, HORSE, T0 + timedelta(minutes=1), market_type="WIN")
    place = store.latest_quote(UID, HORSE, T0 + timedelta(minutes=1), market_type="PLACE")

    assert win is not None and win.odds_decimal == 4.00 and win.market_type == "WIN"
    assert place is not None and place.odds_decimal == 1.80 and place.market_type == "PLACE"

    win_only = store.quotes_as_of(UID, T0 + timedelta(minutes=1), market_type="WIN")
    assert set(win_only["market_type"]) == {"WIN"}
    assert len(win_only) == 1


# 6 ── staleness ─────────────────────────────────────────────────────────────
def test_stale_quote_is_flagged_and_never_returned_as_executable(store):
    store.record([_row(minutes=0, odds=2.20)])
    as_of = T0 + timedelta(minutes=30)  # 1800s old, max_age 300s

    assert store.latest_quote(UID, HORSE, as_of) is None
    assert store.best_quote(UID, HORSE, as_of) is None

    frame = store.quotes_as_of(UID, as_of)
    assert len(frame) == 1
    assert bool(frame.iloc[0]["is_stale"]) is True
    assert frame.iloc[0]["age_seconds"] == pytest.approx(1800.0)


# 7 ── closing price ─────────────────────────────────────────────────────────
def test_closing_quote_returns_the_final_price_and_is_documented_clv_only(store):
    store.record(
        [
            _row(minutes=0, odds=2.20),
            _row(minutes=10, odds=2.50),
            _row(minutes=25, odds=1.90),
        ]
    )
    closing = store.closing_quote(UID, HORSE)
    assert closing is not None
    assert closing.odds_decimal == 1.90
    assert closing.fetched_at == T0 + timedelta(minutes=25)
    assert closing.age_seconds == 0.0

    # A decision at T0+11m must still see 2.50 — the closing price is not
    # retro-fitted onto the earlier decision.
    earlier = store.latest_quote(UID, HORSE, T0 + timedelta(minutes=11))
    assert earlier is not None and earlier.odds_decimal == 2.50

    doc = (SnapshotStore.closing_quote.__doc__ or "").upper()
    assert "CLOSING-LINE-VALUE MEASUREMENT ONLY" in doc
    assert "MUST NEVER" in doc


# 8 ── idempotency ───────────────────────────────────────────────────────────
def test_duplicate_record_is_idempotent_and_preserves_the_first_price(store):
    first = store.record([_row(minutes=0, odds=2.20)])
    assert (first.inserted, first.duplicates, first.rejected) == (1, 0, 0)

    # Same natural key, a different price: the re-scrape must not rewrite history.
    again = store.record([_row(minutes=0, odds=4.00)])
    assert (again.inserted, again.duplicates, again.rejected) == (0, 1, 0)
    assert store.count() == 1

    quote = store.latest_quote(UID, HORSE, T0 + timedelta(minutes=1))
    assert quote is not None and quote.odds_decimal == 2.20
    assert again.to_dict()["duplicates"] == 1


def test_fetch_times_lists_distinct_observation_instants(store):
    store.record([_row(minutes=0, odds=2.2), _row(minutes=0, odds=2.2, book="paddy_power"),
                  _row(minutes=5, odds=2.4)])
    assert store.fetch_times(UID) == [T0, T0 + timedelta(minutes=5)]


# 9 ── rejections ────────────────────────────────────────────────────────────
def test_bad_rows_are_rejected_with_counted_reasons(store):
    good = _row(minutes=0, odds=2.2)

    no_race = dict(good, race_id=None, race_time=None, venue=None)
    no_horse = dict(good, horse_name=None, horse_id=None)
    no_book = dict(good, source=None)
    no_time = dict(good, fetched_at=None)
    evens_or_worse = dict(good, odds_decimal=1.0, horse_name="Priced At One")
    negative = dict(good, odds_decimal=-3.0, horse_name="Negative Price")

    result = store.record(
        [good, no_race, no_horse, no_book, no_time, evens_or_worse, negative]
    )

    assert result.inserted == 1
    assert result.rejected == 6
    assert result.reasons == {
        "missing_race": 1,
        "missing_horse": 1,
        "missing_bookmaker": 1,
        "missing_fetched_at": 1,
        "invalid_odds": 2,
    }
    assert store.count() == 1


def test_record_frame_matches_record(tmp_path):
    s = SnapshotStore(db_path=str(tmp_path / "frame.db"), cfg=_cfg())
    try:
        frame = pd.DataFrame([_row(minutes=0, odds=2.2), _row(minutes=1, odds=2.3)])
        result = s.record_frame(frame)
        assert result.inserted == 2
        assert s.count() == 2
    finally:
        s.close()


# 10 ── the leak guard ───────────────────────────────────────────────────────
def test_assert_point_in_time_raises_on_a_leaking_frame():
    as_of = T0 + timedelta(minutes=5)
    leaking = pd.DataFrame(
        {
            "fetched_at": [T0, as_of, T0 + timedelta(minutes=6)],
            "odds_decimal": [2.2, 2.3, 2.4],
        }
    )
    with pytest.raises(FuturePriceError) as err:
        assert_point_in_time(leaking, as_of)
    assert "as_of" in str(err.value)

    clean = leaking.iloc[:2]
    assert_point_in_time(clean, as_of) is None
    assert_point_in_time([{"fetched_at": T0}], as_of) is None
    with pytest.raises(FuturePriceError):
        assert_point_in_time([{"fetched_at": T0 + timedelta(hours=1)}], as_of)


# ── identity helpers ─────────────────────────────────────────────────────────
def test_race_uid_prefers_race_time_over_source_local_race_id():
    """Each book mints its own race_id; only the off-time is shared."""
    a = race_uid(race_id="SBTE_2_1028491168", race_time=RACE_TIME, venue="Ayr")
    b = race_uid(race_id="45869743.10", race_time="2026-07-27 13:15:00+00:00", venue="Ayr")
    assert a == b == "2026-07-27T13:15:00+00:00"

    assert race_uid(race_id="45869743.10") == "45869743.10"
    assert race_uid(venue="Ayr") == "venue:ayr"
    assert race_uid() == ""


def test_horse_key_normalises_punctuation_and_case():
    assert horse_key("O'Brien's Pride") == horse_key("obriens pride") == "obrienspride"
    assert horse_key(None, horse_id="ABC-123") == "abc123"
    assert horse_key(None, None) == ""


# ── no backfill, ever (Stage 6 requirement 6) ────────────────────────────────
def test_the_live_capture_entry_point_has_no_timestamp_override():
    """``ingest_live_odds_parquet`` is the only capture path wired into the
    daily loop (via ``scripts.refresh._capture_snapshots``). The "no backdated
    fetched_at" guarantee holds because this entry point has no parameter a
    caller could use to override it -- every row's ``fetched_at`` comes solely
    from the scraper's own parquet output, never from caller-supplied "now"."""
    import inspect

    from execution.snapshots import ingest_live_odds_parquet

    params = inspect.signature(ingest_live_odds_parquet).parameters
    assert not any(
        "now" in name or "time" in name or "stamp" in name for name in params
    )
