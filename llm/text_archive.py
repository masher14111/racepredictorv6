"""Append-only archive for incoming free race/runner text (Stage-14 requirement 3).

Why this module exists
-----------------------
``llm/text_features.py`` turns free text into features, but until now nothing
kept the text itself: a re-scrape of an edited Spotlight comment silently
overwrote the previous version in ``data/spotlight.parquet``
(``drop_duplicates(..., keep="last")``), so a past prediction's actual input
could never be reproduced or audited after the fact.

``text_archive`` (migration 6, same ``data/races.db`` as
``execution.snapshots.odds_snapshots``) fixes that the same way: every
observation is a new row, keyed by its own content hash, and SQLite
``RAISE(ABORT)`` triggers make UPDATE/DELETE physically refused. Two versions
of the same runner's comment (original + a later edit) both stay retrievable
forever.

Timestamp honesty
------------------
* ``fetched_at`` — the instant our own scraper retrieved the text. Always
  supplied, always the point-in-time anchor for "was this available yet".
* ``published_at`` — the SOURCE's own claimed publication time. Left ``NULL``
  whenever the source does not supply one; never backfilled from
  ``fetched_at``/``archived_at``, which would fabricate a precision the source
  never gave. A row with unknown ``published_at`` stays unknown forever —
  ``as_of`` reads never guess it into existence.
* ``archived_at`` — when this process wrote the row; server-set, informational.

This module never computes a probability or feature; it only stores and
retrieves text. Feature extraction (``llm.text_features``) reads from here.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional

from utils.logger import get_logger
from utils.storage import DEFAULT_DB_PATH
from utils.storage.migrations import apply_migrations
from utils.storage.pool import ConnectionPool

logger = get_logger(__name__)

_TS_FMT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
_PUNCT = re.compile(r"[^a-z0-9]+")

_COLUMNS = (
    "content_hash", "source", "text_kind", "race_uid", "runner_key",
    "horse_id", "horse_name", "text", "published_at", "fetched_at",
    "archived_at", "metadata",
)
_INSERT_SQL = (
    "INSERT OR IGNORE INTO text_archive (" + ", ".join(_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" * len(_COLUMNS)) + ")"
)


class AppendOnlyViolation(RuntimeError):
    """An UPDATE or DELETE against ``text_archive`` was attempted and refused."""


def content_hash(text: str) -> str:
    """Stable identity for one exact piece of text (whitespace-normalised)."""
    normalised = " ".join(str(text).split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def runner_key(horse_name: Any = None, horse_id: Any = None) -> str:
    """Cross-source runner identity — normalised name, id only as a fallback.

    Mirrors ``execution.snapshots.horse_key`` deliberately (same reasoning: the
    name is what sources agree on, the id is source-local) so a text row and an
    odds/result row for the same runner reconcile to the same key.
    """
    name = "" if horse_name is None else str(horse_name).strip()
    if name:
        key = _PUNCT.sub("", name.lower())
        if key:
            return key
    hid = "" if horse_id is None else str(horse_id).strip()
    return _PUNCT.sub("", hid.lower()) if hid else ""


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            import pandas as pd
            ts = pd.Timestamp(value)
            if ts is pd.NaT:
                return None
            dt = ts.to_pydatetime()
        except Exception:
            return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(_TS_FMT)


def _iso_or_none(value: Any) -> Optional[str]:
    dt = _parse_ts(value)
    return _iso(dt) if dt is not None else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ArchivedText:
    """One immutable row read back from ``text_archive``."""

    id: int
    content_hash: str
    source: str
    text_kind: str
    race_uid: str
    runner_key: str
    horse_id: Optional[str]
    horse_name: Optional[str]
    text: str
    published_at: Optional[str]  # None means genuinely unknown, never guessed
    fetched_at: str
    archived_at: str
    metadata: dict

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> "ArchivedText":
        meta_raw = row["metadata"]
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
        except (TypeError, ValueError):
            meta = {}
        return cls(
            id=row["id"], content_hash=row["content_hash"], source=row["source"],
            text_kind=row["text_kind"], race_uid=row["race_uid"],
            runner_key=row["runner_key"], horse_id=row["horse_id"],
            horse_name=row["horse_name"], text=row["text"],
            published_at=row["published_at"], fetched_at=row["fetched_at"],
            archived_at=row["archived_at"], metadata=meta,
        )


class TextArchive:
    """Reader/writer for the append-only ``text_archive`` table."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self.db_path = db_path or DEFAULT_DB_PATH
        self._pool = ConnectionPool(self.db_path)
        apply_migrations(self._pool)

    def close(self) -> None:
        self._pool.close_all()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        conn = self._pool.connection()
        try:
            with self._pool.write_lock():
                cur = conn.execute(sql, tuple(params))
                conn.commit()
                return cur
        except sqlite3.Error as exc:
            if "append-only" in str(exc):
                raise AppendOnlyViolation(str(exc)) from exc
            raise

    # -- writing --------------------------------------------------------------
    def record(
        self,
        *,
        text: str,
        source: str,
        race_uid: str,
        horse_name: Any = None,
        horse_id: Any = None,
        text_kind: str = "spotlight",
        published_at: Any = None,
        fetched_at: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ArchivedText:
        """Append one text observation. Never mutates an existing row.

        Returns the row as archived (which may be a pre-existing row with the
        identical ``(content_hash, source, race_uid, runner_key, fetched_at)``
        key — the insert is idempotent on exact re-fetch, not a silent update).
        """
        text = str(text)
        if not text.strip():
            raise ValueError("text_archive: cannot archive empty text")
        if not race_uid:
            raise ValueError("text_archive: race_uid is required")
        rkey = runner_key(horse_name, horse_id)
        if not rkey:
            raise ValueError("text_archive: horse_name or horse_id is required")
        h = content_hash(text)
        fetched = _iso_or_none(fetched_at) or _iso(_now())
        published = _iso_or_none(published_at)  # None stays None — never guessed
        archived = _iso(_now())
        params = (
            h, str(source), str(text_kind), str(race_uid), rkey,
            None if horse_id is None else str(horse_id),
            None if horse_name is None else str(horse_name),
            text, published, fetched, archived,
            json.dumps(dict(metadata)) if metadata else None,
        )
        self._execute(_INSERT_SQL, params)
        row = self._execute(
            "SELECT * FROM text_archive WHERE content_hash=? AND source=? "
            "AND race_uid=? AND runner_key=? AND fetched_at=?",
            (h, str(source), str(race_uid), rkey, fetched),
        ).fetchone()
        return ArchivedText._from_row(row)

    # -- reading ----------------------------------------------------------------
    def history(self, race_uid: str, horse_name: Any = None, horse_id: Any = None) -> list[ArchivedText]:
        """Every archived version for one runner/race, oldest first. Nothing is hidden."""
        rkey = runner_key(horse_name, horse_id)
        cur = self._pool.connection().execute(
            "SELECT * FROM text_archive WHERE race_uid=? AND runner_key=? "
            "ORDER BY archived_at ASC, id ASC",
            (str(race_uid), rkey),
        )
        return [ArchivedText._from_row(r) for r in cur.fetchall()]

    def as_of(
        self, race_uid: str, horse_name: Any = None, horse_id: Any = None, *, as_of: Any = None
    ) -> Optional[ArchivedText]:
        """The most recent version actually available (``fetched_at``) at or
        before ``as_of`` (default: now).

        Filters on ``fetched_at`` (when our scraper actually saw the text), not
        ``archived_at`` (when this process happened to write the row) — a
        batched/delayed write must not make text look available earlier or
        later than it really was. A later edit (a new row) can never change
        what this returns for a past ``as_of`` instant — the same
        point-in-time contract as ``execution.snapshots.SnapshotStore``.
        """
        rkey = runner_key(horse_name, horse_id)
        cutoff = _iso(_parse_ts(as_of) or _now())
        cur = self._pool.connection().execute(
            "SELECT * FROM text_archive WHERE race_uid=? AND runner_key=? "
            "AND fetched_at<=? ORDER BY fetched_at DESC, id DESC LIMIT 1",
            (str(race_uid), rkey, cutoff),
        )
        row = cur.fetchone()
        return ArchivedText._from_row(row) if row else None

    def known_race_uids(self) -> set[str]:
        """Every ``race_uid`` with at least one archived row.

        A cheap pre-filter for bulk joins (``features.text_features_v1``): most
        callers join against a large matrix where only a tiny fraction of races
        have any archived text at all, so this lets a caller skip the per-row
        ``as_of`` lookup entirely for every other row.
        """
        cur = self._pool.connection().execute("SELECT DISTINCT race_uid FROM text_archive")
        return {r["race_uid"] for r in cur.fetchall()}

    def by_hash(self, h: str) -> list[ArchivedText]:
        cur = self._pool.connection().execute(
            "SELECT * FROM text_archive WHERE content_hash=? ORDER BY archived_at ASC", (h,)
        )
        return [ArchivedText._from_row(r) for r in cur.fetchall()]

    def coverage(self) -> dict:
        """Honest summary of what is actually archived — never a claimed corpus size."""
        row = self._pool.connection().execute(
            "SELECT COUNT(*) AS n, COUNT(DISTINCT race_uid) AS races, "
            "COUNT(DISTINCT source) AS sources, MIN(fetched_at) AS earliest, "
            "MAX(fetched_at) AS latest, "
            "SUM(CASE WHEN published_at IS NULL THEN 1 ELSE 0 END) AS unknown_published "
            "FROM text_archive"
        ).fetchone()
        return {
            "rows": row["n"], "races": row["races"], "sources": row["sources"],
            "earliest_fetched_at": row["earliest"], "latest_fetched_at": row["latest"],
            "unknown_published_at": row["unknown_published"],
        }


_SINGLETON: Optional[TextArchive] = None


def get_text_archive() -> TextArchive:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TextArchive()
    return _SINGLETON
