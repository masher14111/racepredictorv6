"""Point-in-time / append-only guarantees of ``llm.text_archive``.

Mirrors ``tests/execution/test_snapshots.py``'s pattern for the same contract
applied to free text instead of prices: history cannot be rewritten, an
edited comment produces a new retrievable version rather than replacing the
old one, and an unknown publication time stays unknown rather than being
guessed from scrape/write time.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from llm.text_archive import AppendOnlyViolation, TextArchive, content_hash, runner_key

T0 = datetime(2026, 6, 17, 8, 0, 0, tzinfo=timezone.utc)
RACE_UID = "2026-06-17T14:30:00+00:00"


@pytest.fixture
def archive(tmp_path):
    a = TextArchive(db_path=str(tmp_path / "text.db"))
    yield a
    a.close()


def _record(archive, *, text, fetched_at, published_at=None, horse="Little Lady Karen"):
    return archive.record(
        text=text, source="sporting_life_spotlight", race_uid=RACE_UID,
        horse_name=horse, fetched_at=fetched_at, published_at=published_at,
    )


# ── basic append + retrieval ─────────────────────────────────────────────────

def test_record_then_read_back(archive):
    row = _record(archive, text="Needs better ground, may find this too sharp.", fetched_at=T0)
    assert row.content_hash == content_hash("Needs better ground, may find this too sharp.")
    assert row.race_uid == RACE_UID
    assert row.runner_key == runner_key("Little Lady Karen")
    got = archive.as_of(RACE_UID, "Little Lady Karen", as_of=T0 + timedelta(hours=1))
    assert got is not None
    assert got.text == row.text


def test_missing_race_uid_or_runner_rejected(archive):
    with pytest.raises(ValueError):
        archive.record(text="x", source="s", race_uid="", horse_name="Horse")
    with pytest.raises(ValueError):
        archive.record(text="x", source="s", race_uid=RACE_UID)
    with pytest.raises(ValueError):
        archive.record(text="   ", source="s", race_uid=RACE_UID, horse_name="Horse")


# ── two versions of a comment both remain retrievable ───────────────────────

def test_two_versions_of_a_comment_both_retrievable(archive):
    v1 = _record(archive, text="First cut: needs better ground.", fetched_at=T0)
    v2 = _record(
        archive, text="Updated: has since improved and handles any ground.",
        fetched_at=T0 + timedelta(hours=6),
    )
    assert v1.content_hash != v2.content_hash

    history = archive.history(RACE_UID, "Little Lady Karen")
    assert [h.text for h in history] == [v1.text, v2.text]
    # Both original hashes still independently retrievable.
    assert archive.by_hash(v1.content_hash)[0].text == v1.text
    assert archive.by_hash(v2.content_hash)[0].text == v2.text


def test_later_edit_cannot_alter_a_historical_as_of_read(archive):
    """The core immutability claim: reading 'as of before the edit' must keep
    returning the pre-edit text forever, no matter what is archived later."""
    v1 = _record(archive, text="First cut: needs better ground.", fetched_at=T0)
    cutoff = T0 + timedelta(minutes=30)

    before_edit = archive.as_of(RACE_UID, "Little Lady Karen", as_of=cutoff)
    assert before_edit.text == v1.text

    # A "later edit" (new scrape of changed commentary) arrives after the cutoff.
    _record(
        archive, text="Now: has since improved and handles any ground.",
        fetched_at=T0 + timedelta(hours=6),
    )

    after_edit_reread = archive.as_of(RACE_UID, "Little Lady Karen", as_of=cutoff)
    assert after_edit_reread.text == v1.text  # unchanged historical read

    latest = archive.as_of(RACE_UID, "Little Lady Karen")
    assert latest.text != v1.text


# ── append-only enforcement ──────────────────────────────────────────────────

def test_update_is_physically_refused(archive):
    _record(archive, text="Needs better ground.", fetched_at=T0)
    conn = sqlite3.connect(archive.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE text_archive SET text='tampered' WHERE id=1")
    conn.close()


def test_delete_is_physically_refused(archive):
    _record(archive, text="Needs better ground.", fetched_at=T0)
    conn = sqlite3.connect(archive.db_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM text_archive WHERE id=1")
    conn.close()


def test_store_level_wrapper_translates_to_append_only_violation(archive):
    _record(archive, text="Needs better ground.", fetched_at=T0)
    with pytest.raises(AppendOnlyViolation):
        archive._execute("UPDATE text_archive SET text='tampered' WHERE id=1")


# ── unknown publication time stays unknown ──────────────────────────────────

def test_unknown_publication_time_stays_unknown(archive):
    row = _record(archive, text="No publish date on this source.", fetched_at=T0, published_at=None)
    assert row.published_at is None
    reread = archive.as_of(RACE_UID, "Little Lady Karen", as_of=T0 + timedelta(days=1))
    assert reread.published_at is None  # never backfilled from fetched_at/archived_at


def test_known_publication_time_is_preserved(archive):
    published = T0 - timedelta(hours=2)
    row = _record(archive, text="Has a real publish timestamp.", fetched_at=T0, published_at=published)
    assert row.published_at is not None
    assert row.published_at.startswith("2026-06-17T06:00:00")


# ── content hash identity ────────────────────────────────────────────────────

def test_content_hash_is_stable_and_whitespace_normalised():
    a = content_hash("Needs  better   ground.")
    b = content_hash("Needs better ground.")
    assert a == b  # whitespace-insensitive identity
    assert content_hash("Different text.") != a


def test_idempotent_exact_refetch_does_not_duplicate(archive):
    row1 = _record(archive, text="Needs better ground.", fetched_at=T0)
    row2 = _record(archive, text="Needs better ground.", fetched_at=T0)  # identical refetch
    assert row1.id == row2.id
    assert len(archive.history(RACE_UID, "Little Lady Karen")) == 1


# ── honest coverage reporting ────────────────────────────────────────────────

def test_coverage_reports_only_what_is_actually_archived(archive):
    assert archive.coverage()["rows"] == 0
    _record(archive, text="Needs better ground.", fetched_at=T0, published_at=T0 - timedelta(hours=1))
    _record(
        archive, text="Second runner, no publish time known.", fetched_at=T0,
        horse="Another Horse", published_at=None,
    )
    cov = archive.coverage()
    assert cov["rows"] == 2
    assert cov["races"] == 1
    assert cov["unknown_published_at"] == 1
