"""Append-only point-in-time odds snapshots (Stage-5 requirement 1).

Why this module exists
----------------------
The Stage-4 calibration audit (``reports/calibration_audit_20260727.md``) issued
**MODEL NO-GO**: on 904 untouched races the price-free line loses to the de-vigged
pre-off market by 0.186 race log-loss and closing-line value is -13.4%. A large
part of what made earlier numbers look better than reality was *price* look-ahead,
not model look-ahead: the ``live_odds`` table from migration 1 is an UPSERT cache
keyed ``(source, race_id, horse_id, market_type)``, so it retains only the newest
observation. Anything replaying history out of it sees, for every runner that was
re-scraped, a price that was **not** available at the moment the decision was made
— usually the closing price, which is exactly the price a real bettor cannot have.

``odds_snapshots`` (migration 4) fixes that by construction:

* every observation is stored, keyed by ``fetched_at``;
* SQLite ``RAISE(ABORT)`` triggers forbid UPDATE and DELETE, so price history is
  physically unrewritable — a past decision can never be re-priced;
* every read path here filters ``fetched_at <= as_of`` **in SQL**. That filter is
  the contract of this module, not an optimisation.

Staleness is treated the same way. A quote older than
``execution.snapshots.max_age_seconds`` is not an executable price, so
:meth:`SnapshotStore.latest_quote` and :meth:`SnapshotStore.best_quote` return
``None`` for one rather than handing back an unexecutable number that a caller
would inevitably treat as a fill. Missing data blocks; it never permits.

THE ONE EXCEPTION: :meth:`SnapshotStore.closing_quote` deliberately looks at the
newest row regardless of ``as_of``. IT EXISTS FOR CLOSING-LINE-VALUE MEASUREMENT
ONLY AND MUST NEVER BE USED TO PRICE A FILL, SIZE A STAKE, OR COMPUTE AN EDGE.
Any such use silently reintroduces precisely the look-ahead this module was
written to eliminate.
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Optional, Sequence

import pandas as pd

from execution.config import ExecutionConfig
from utils.logger import get_logger
from utils.storage.migrations import apply_migrations
from utils.storage.pool import ConnectionPool
from utils.text_norm import norm_venue as _norm_venue

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed-width UTC rendering so the SQL string comparison ``fetched_at <= as_of``
# is a true chronological comparison. Never change this format: the stored rows
# are immutable, so a re-render would break ordering against existing history.
_TS_FMT = "%Y-%m-%dT%H:%M:%S.%f+00:00"
# Race off-times are minute-precision identities, rendered without microseconds so
# the uid matches models.predictor._race_uid for a tz-aware whole-second stamp.
_RACE_TIME_FMT = "%Y-%m-%dT%H:%M:%S+00:00"

_INSERT_COLUMNS = (
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
    "odds_decimal",
    "sp",
    "ew_places",
    "ew_reduction",
    "ew_margin",
    "field_size",
    "booksum",
    "validation_status",
    "fetched_at",
    "ingested_at",
)

_INSERT_SQL = (
    "INSERT OR IGNORE INTO odds_snapshots (" + ", ".join(_INSERT_COLUMNS) + ") "
    "VALUES (" + ", ".join("?" * len(_INSERT_COLUMNS)) + ")"
)

_PUNCT = re.compile(r"[^a-z0-9]+")


class FuturePriceError(RuntimeError):
    """A row priced after the decision instant leaked into a point-in-time view."""


class AppendOnlyViolation(RuntimeError):
    """An UPDATE or DELETE against ``odds_snapshots`` was attempted and refused."""


# ── timestamp helpers ────────────────────────────────────────────────────────
def _parse_ts(value: Any) -> Optional[datetime]:
    """Coerce anything timestamp-ish to a tz-aware UTC datetime.

    Naive input is read as UTC — the scrapers stamp ``fetched_at`` in UTC, and
    guessing a local zone here would shift prices across the ``as_of`` boundary.
    """
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            if pd.isna(value):  # NaT / pd.NA / NaN
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, str) and not value.strip():
            return None
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError):
            return None
        if ts is pd.NaT:
            return None
        dt = ts.to_pydatetime()
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


# ── scalar coercion ──────────────────────────────────────────────────────────
def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return "" if text.lower() in ("nan", "nat", "none", "<na>") else text


def _as_opt_float(value: Any) -> Optional[float]:
    text = _clean_str(value)
    if not text:
        return None
    try:
        out = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _as_opt_int(value: Any) -> Optional[int]:
    out = _as_opt_float(value)
    return None if out is None else int(round(out))


# ── identity ─────────────────────────────────────────────────────────────────
def race_uid(
    race_id: Any = None, race_time: Any = None, venue: Any = None
) -> str:
    """Cross-bookmaker race identity — prefers ``race_time``.

    Deliberately identical in preference order to ``models.predictor._race_uid``:
    every bookmaker mints its own ``race_id`` for the same physical race
    (livescorebet ``SBTE_2_1028491168`` vs boylesports ``45869743.10``), so keying
    on ``race_id`` fragments the cross-book view and collapses each race to a
    single source. The off-time is the one identifier all sources agree on, and it
    still separates two races at the same venue on the same day.

    Duplicated rather than imported: ``models.predictor`` imports CatBoost and the
    feature builder at module scope, which is far too heavy a side effect for the
    storage layer. Keep the two in sync — the preference order is the contract.

    Unlike ``_race_uid`` the off-time is normalised to a canonical UTC rendering,
    so a ``datetime``, a ``pd.Timestamp`` and an ISO string for the same race all
    produce one uid. ``venue`` is a last-resort fallback only.
    """
    dt = _parse_ts(race_time)
    if dt is not None:
        return dt.astimezone(timezone.utc).strftime(_RACE_TIME_FMT)
    raw_time = _clean_str(race_time)
    if raw_time:
        return raw_time
    rid = _clean_str(race_id)
    if rid:
        return rid
    # Neither identifier survived. A bare venue cannot separate two races on the
    # same card, so it is namespaced to make the weakness visible to any reader.
    place = _clean_str(venue)
    return f"venue:{_PUNCT.sub('-', place.lower()).strip('-')}" if place else ""


def race_key(venue: Any = None, race_time: Any = None) -> str:
    """Venue-qualified, minute-precision physical-race identity (Stage 20 / B5).

    ``race_uid`` above drops venue the moment an off-time parses — two
    different venues going off at the identical minute collide onto one
    identity (reproduced on the live store: cross-venue collisions, D37). It
    also renders at second precision, so sub-minute jitter between two sources
    scraping the same physical race fragments it into two. Neither defect is
    safe to fix by changing ``race_uid`` itself: that function's exact output
    is the persisted, immutable value of every historical
    ``odds_snapshots``/``paper_tickets`` row, and this project never rewrites
    history (DESIGN.md). ``race_key`` is instead an ADDITIVE companion,
    ``norm_venue(venue)|YYYY-MM-DDTHH:MM`` (UTC) — the same scheme
    ``execution.race_facts.race_facts_key`` already uses for settlement, so a
    live decision and its eventual settlement resolve the same physical race
    the same way. Empty when either component cannot be recovered: a
    degraded key must never look venue-qualified when it is not.
    """
    v = _norm_venue(venue) if venue else ""
    dt = _parse_ts(race_time)
    if not v or dt is None:
        return ""
    return f"{v}|{dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M')}"


def horse_key(horse_name: Any = None, horse_id: Any = None) -> str:
    """Cross-bookmaker runner identity: normalised name, id only as a fallback.

    Same reasoning as :func:`race_uid` — ``horse_id`` is source-local, the name is
    what the books agree on. Normalisation lowercases and strips every non
    alphanumeric character so ``"O'Brien's Pride"``, ``"Obriens Pride"`` and
    ``"o brien s pride"`` collapse to one key.
    """
    name = _clean_str(horse_name)
    if name:
        key = _PUNCT.sub("", name.lower())
        if key:
            return key
    hid = _clean_str(horse_id)
    return _PUNCT.sub("", hid.lower()) if hid else ""


# ── value objects ────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Quote:
    """One price observation, always carrying the instant it was observed.

    ``as_of`` is the decision instant the quote was retrieved for and
    ``age_seconds`` the gap between the two, so a caller can never look at a price
    without also seeing how old it was when it was used.
    """

    race_uid: str
    horse_key: str
    bookmaker: str
    market_type: str
    odds_decimal: float
    fetched_at: datetime
    as_of: datetime
    age_seconds: float
    is_stale: bool
    sp: float | None = None
    ew_places: int | None = None
    ew_reduction: float | None = None
    horse_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "race_uid": self.race_uid,
            "horse_key": self.horse_key,
            "bookmaker": self.bookmaker,
            "market_type": self.market_type,
            "odds_decimal": self.odds_decimal,
            "fetched_at": _iso(self.fetched_at),
            "as_of": _iso(self.as_of),
            "age_seconds": self.age_seconds,
            "is_stale": self.is_stale,
            "sp": self.sp,
            "ew_places": self.ew_places,
            "ew_reduction": self.ew_reduction,
            "horse_name": self.horse_name,
        }


@dataclass(frozen=True)
class SnapshotWriteResult:
    """Outcome of a write: what landed, what was already there, what was refused."""

    inserted: int
    duplicates: int
    rejected: int
    reasons: dict[str, int]

    @property
    def considered(self) -> int:
        return self.inserted + self.duplicates + self.rejected

    def to_dict(self) -> dict:
        return {
            "inserted": self.inserted,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "considered": self.considered,
            "reasons": dict(self.reasons),
        }


# ── the store ────────────────────────────────────────────────────────────────
def _resolve_db_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


class SnapshotStore:
    """Reader/writer for the append-only ``odds_snapshots`` table.

    Every read is point-in-time: the caller must state the ``as_of`` instant and
    only rows with ``fetched_at <= as_of`` are considered. The single exception is
    :meth:`closing_quote` — see its docstring.
    """

    def __init__(self, db_path: str | None = None, cfg: ExecutionConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ExecutionConfig.from_config()
        self.db_path = _resolve_db_path(str(db_path or self.cfg.snapshots.db_path))
        self.max_age_seconds = float(self.cfg.snapshots.max_age_seconds)
        self._pool = ConnectionPool(self.db_path)
        apply_migrations(self._pool)

    # -- plumbing ------------------------------------------------------------
    def close(self) -> None:
        self._pool.close_all()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        """Run a statement, translating the append-only trigger into a Python error.

        The escape hatch is public on purpose: anything that tries to mutate price
        history through this store gets :class:`AppendOnlyViolation` rather than a
        bare SQLite message, so the refusal is legible in a traceback.
        """
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

    # -- writing -------------------------------------------------------------
    def record(
        self, rows: Iterable[Mapping], *, ingested_at: Any = None
    ) -> SnapshotWriteResult:
        """Append observations. Idempotent on the natural key, fail-closed on junk.

        Re-recording the same
        ``(race_uid, horse_key, bookmaker, market_type, fetched_at)`` is a
        ``duplicate``, not an error, and ``INSERT OR IGNORE`` guarantees the stored
        odds are left exactly as first observed — a re-scrape can never overwrite
        the price a decision was made on.
        """
        stamped = _parse_ts(ingested_at) or _now()
        ingested_iso = _iso(stamped)

        payload: list[tuple] = []
        rejected = 0
        reasons: dict[str, int] = {}

        def _reject(reason: str) -> None:
            nonlocal rejected
            rejected += 1
            reasons[reason] = reasons.get(reason, 0) + 1

        for row in rows:
            if not isinstance(row, Mapping):
                _reject("not_a_mapping")
                continue

            uid = _clean_str(row.get("race_uid")) or race_uid(
                race_id=row.get("race_id"),
                race_time=row.get("race_time"),
                venue=row.get("venue"),
            )
            if not uid:
                _reject("missing_race")
                continue

            hkey = _clean_str(row.get("horse_key")) or horse_key(
                horse_name=row.get("horse_name"), horse_id=row.get("horse_id")
            )
            if not hkey:
                _reject("missing_horse")
                continue

            # ``source`` is what the scrapers call it; ``bookmaker`` is the column.
            book = _clean_str(row.get("bookmaker")) or _clean_str(row.get("source"))
            if not book:
                _reject("missing_bookmaker")
                continue

            fetched = _parse_ts(row.get("fetched_at"))
            if fetched is None:
                _reject("missing_fetched_at")
                continue

            odds = _as_opt_float(row.get("odds_decimal"))
            if odds is None or odds <= 1.0:
                # <= 1.0 is not a price: it cannot return the stake.
                _reject("invalid_odds")
                continue

            market = (_clean_str(row.get("market_type")) or "WIN").upper()
            rkey = _clean_str(row.get("race_key")) or race_key(
                venue=row.get("venue"), race_time=row.get("race_time")
            )
            payload.append(
                (
                    uid,
                    rkey or None,
                    _clean_str(row.get("race_id")) or None,
                    _iso_or_none(row.get("race_time")) or _clean_str(row.get("race_time")) or None,
                    _clean_str(row.get("venue")) or None,
                    hkey,
                    _clean_str(row.get("horse_id")) or None,
                    _clean_str(row.get("horse_name")) or None,
                    book.lower(),
                    market,
                    odds,
                    _as_opt_float(row.get("sp")),
                    _as_opt_int(row.get("ew_places")),
                    _as_opt_float(row.get("ew_reduction")),
                    _as_opt_float(row.get("ew_margin")),
                    _as_opt_int(row.get("field_size")),
                    _as_opt_float(row.get("booksum")),
                    _clean_str(row.get("validation_status")) or None,
                    _iso(fetched),
                    ingested_iso,
                )
            )

        inserted = 0
        if payload:
            conn = self._pool.connection()
            try:
                with self._pool.write_lock():
                    # total_changes, not cursor.rowcount: SQLite leaves rowcount at
                    # the previous statement's value when OR IGNORE swallows a row,
                    # which would count duplicates as inserts.
                    before = conn.total_changes
                    conn.executemany(_INSERT_SQL, payload)
                    inserted = conn.total_changes - before
                    conn.commit()
            except sqlite3.Error as exc:
                conn.rollback()
                if "append-only" in str(exc):
                    raise AppendOnlyViolation(str(exc)) from exc
                raise

        result = SnapshotWriteResult(
            inserted=inserted,
            duplicates=len(payload) - inserted,
            rejected=rejected,
            reasons=reasons,
        )
        if rejected:
            logger.info("snapshots: %s", result.to_dict())
        return result

    def record_frame(self, df: pd.DataFrame, *, ingested_at: Any = None) -> SnapshotWriteResult:
        """:meth:`record` for a DataFrame of scraped odds."""
        if df is None or len(df) == 0:
            return SnapshotWriteResult(0, 0, 0, {})
        return self.record(df.to_dict("records"), ingested_at=ingested_at)

    # -- reading (point-in-time) --------------------------------------------
    def quotes_as_of(
        self,
        race_uid: str,
        as_of: Any,
        *,
        horse_key: str | None = None,
        bookmaker: str | None = None,
        market_type: str = "WIN",
        max_age_seconds: float | None = None,
        race_key: str | None = None,
    ) -> pd.DataFrame:
        """The market as it stood at ``as_of``: newest row per (horse, bookmaker).

        Only rows with ``fetched_at <= as_of`` are considered — enforced in SQL,
        twice (inside the newest-row subquery and again on the outer select). Stale
        rows are returned but flagged ``is_stale``; they are kept visible so a
        caller can *see* that the book had gone cold rather than silently receiving
        a shorter list. Callers pricing a fill must use :meth:`latest_quote` or
        :meth:`best_quote`, which refuse stale quotes outright.

        ``race_key`` (Stage 20 / B5) is an optional EXTRA filter on the
        venue-qualified, minute-precision identity from :func:`race_key` above —
        narrowing, never widening. Passing it stops two different venues that
        happen to share a bare ``race_uid`` off-time from ever satisfying the
        same lookup; omitting it (the default) preserves every existing caller's
        behaviour exactly.
        """
        moment = _parse_ts(as_of)
        if moment is None:
            raise ValueError("quotes_as_of requires an as_of instant")
        cutoff = _iso(moment)
        limit = self.max_age_seconds if max_age_seconds is None else float(max_age_seconds)

        where = ["race_uid = ?", "market_type = ?", "fetched_at <= ?"]
        params: list[Any] = [str(race_uid), str(market_type).upper(), cutoff]
        if horse_key:
            where.append("horse_key = ?")
            params.append(str(horse_key))
        if bookmaker:
            where.append("bookmaker = ?")
            params.append(str(bookmaker).strip().lower())
        if race_key:
            where.append("race_key = ?")
            params.append(str(race_key))
        inner_where = " AND ".join(where)
        outer_race_key_clause = " AND s.race_key = ?" if race_key else ""

        sql = f"""
            SELECT s.* FROM odds_snapshots AS s
            JOIN (
                SELECT race_uid, horse_key, bookmaker, market_type,
                       MAX(fetched_at) AS newest
                FROM odds_snapshots
                WHERE {inner_where}
                GROUP BY race_uid, horse_key, bookmaker, market_type
            ) AS latest
              ON s.race_uid = latest.race_uid
             AND s.horse_key = latest.horse_key
             AND s.bookmaker = latest.bookmaker
             AND s.market_type = latest.market_type
             AND s.fetched_at = latest.newest
            WHERE s.fetched_at <= ?{outer_race_key_clause}
            ORDER BY s.horse_key, s.bookmaker
        """
        conn = self._pool.connection()
        outer_params = (cutoff, str(race_key)) if race_key else (cutoff,)
        rows = conn.execute(sql, tuple(params) + outer_params).fetchall()
        frame = pd.DataFrame([dict(r) for r in rows], columns=_frame_columns())
        if frame.empty:
            frame["as_of"] = pd.Series(dtype="object")
            frame["age_seconds"] = pd.Series(dtype="float64")
            frame["is_stale"] = pd.Series(dtype="bool")
            return frame

        fetched = frame["fetched_at"].map(_parse_ts)
        frame["as_of"] = cutoff
        frame["age_seconds"] = [
            (moment - ts).total_seconds() if ts is not None else float("inf")
            for ts in fetched
        ]
        frame["is_stale"] = frame["age_seconds"] > limit
        # Belt and braces: the SQL already excludes the future, so a hit here is a
        # bug in this module rather than bad data.
        assert_point_in_time(frame, moment)
        return frame

    def latest_quote(
        self,
        race_uid: str,
        horse_key: str,
        as_of: Any,
        *,
        bookmaker: str | None = None,
        market_type: str = "WIN",
        max_age_seconds: float | None = None,
        race_key: str | None = None,
    ) -> Quote | None:
        """Newest quote at-or-before ``as_of``; ``None`` if there is none or it is stale.

        Returning ``None`` for a stale quote is the whole point: an old price is not
        an executable price, and handing one back would let it be treated as a fill.
        With no ``bookmaker`` given, ties on ``fetched_at`` break to the *worst*
        (lowest) price — the conservative reading of an ambiguous market.
        ``race_key`` narrows to the venue-qualified identity (Stage 20 / B5);
        see :meth:`quotes_as_of`.
        """
        frame = self.quotes_as_of(
            race_uid,
            as_of,
            horse_key=horse_key,
            bookmaker=bookmaker,
            market_type=market_type,
            max_age_seconds=max_age_seconds,
            race_key=race_key,
        )
        if frame.empty:
            return None
        ordered = frame.sort_values(
            ["fetched_at", "odds_decimal"], ascending=[False, True]
        )
        row = ordered.iloc[0]
        if bool(row["is_stale"]):
            logger.debug(
                "snapshots: stale quote refused (%s/%s, age %.0fs)",
                race_uid,
                horse_key,
                float(row["age_seconds"]),
            )
            return None
        return _quote_from_row(row)

    def best_quote(
        self,
        race_uid: str,
        horse_key: str,
        as_of: Any,
        *,
        market_type: str = "WIN",
        max_age_seconds: float | None = None,
        race_key: str | None = None,
    ) -> Quote | None:
        """Best price across bookmakers at ``as_of``, considering in-age quotes only.

        A stale quote is not a price you could take, so it cannot win the
        comparison — otherwise "best available" would drift towards whichever book
        stopped being scraped. ``race_key`` narrows to the venue-qualified
        identity (Stage 20 / B5); see :meth:`quotes_as_of`.
        """
        frame = self.quotes_as_of(
            race_uid,
            as_of,
            horse_key=horse_key,
            market_type=market_type,
            max_age_seconds=max_age_seconds,
            race_key=race_key,
        )
        if frame.empty:
            return None
        live = frame[~frame["is_stale"].astype(bool)]
        if live.empty:
            return None
        ordered = live.sort_values(["odds_decimal", "fetched_at"], ascending=[False, False])
        return _quote_from_row(ordered.iloc[0])

    def book_as_of(
        self,
        race_uid: str,
        as_of: Any,
        bookmaker: str,
        *,
        market_type: str = "WIN",
        max_age_seconds: float | None = None,
    ) -> pd.DataFrame:
        """One bookmaker's whole board at ``as_of`` — one row per runner.

        Used for overround / de-vig work, which needs a *single* book's complete
        field. Stale rows stay in the frame with ``is_stale`` set so a completeness
        check fails closed on a partially cold board instead of quietly de-vigging
        a hole.
        """
        return self.quotes_as_of(
            race_uid,
            as_of,
            bookmaker=bookmaker,
            market_type=market_type,
            max_age_seconds=max_age_seconds,
        )

    # -- reading (NOT point-in-time) ----------------------------------------
    def closing_quote(
        self,
        race_uid: str,
        horse_key: str,
        *,
        market_type: str = "WIN",
        bookmaker: str | None = None,
    ) -> Quote | None:
        """Last observed price for a runner, ignoring any ``as_of``.

        FOR CLOSING-LINE-VALUE MEASUREMENT ONLY. THIS IS THE ONE FUNCTION IN THIS
        MODULE THAT LOOKS INTO THE FUTURE RELATIVE TO A DECISION, AND IT MUST NEVER
        BE USED TO PRICE A FILL, SIZE A STAKE, COMPUTE AN EDGE OR EXPECTED VALUE,
        OR STAND IN FOR THE PRICE AVAILABLE EARLIER. Using it that way recreates
        exactly the look-ahead that made the pre-Stage-4 backtests look profitable.

        The returned :class:`Quote` reports ``as_of == fetched_at`` and
        ``age_seconds == 0`` because it has no decision instant: it is a
        measurement, not an offer.
        """
        where = ["race_uid = ?", "horse_key = ?", "market_type = ?"]
        params: list[Any] = [str(race_uid), str(horse_key), str(market_type).upper()]
        if bookmaker:
            where.append("bookmaker = ?")
            params.append(str(bookmaker).strip().lower())
        sql = (
            "SELECT * FROM odds_snapshots WHERE "
            + " AND ".join(where)
            + " ORDER BY fetched_at DESC, id DESC LIMIT 1"
        )
        conn = self._pool.connection()
        row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        record = dict(row)
        fetched = _parse_ts(record["fetched_at"])
        return Quote(
            race_uid=record["race_uid"],
            horse_key=record["horse_key"],
            bookmaker=record["bookmaker"],
            market_type=record["market_type"],
            odds_decimal=float(record["odds_decimal"]),
            fetched_at=fetched,
            as_of=fetched,
            age_seconds=0.0,
            is_stale=False,
            sp=_as_opt_float(record.get("sp")),
            ew_places=_as_opt_int(record.get("ew_places")),
            ew_reduction=_as_opt_float(record.get("ew_reduction")),
            horse_name=record.get("horse_name"),
        )

    # -- introspection -------------------------------------------------------
    def fetch_times(self, race_uid: str) -> list[datetime]:
        """Every distinct observation instant recorded for a race, oldest first."""
        conn = self._pool.connection()
        rows = conn.execute(
            "SELECT DISTINCT fetched_at FROM odds_snapshots WHERE race_uid = ? "
            "ORDER BY fetched_at",
            (str(race_uid),),
        ).fetchall()
        return [ts for ts in (_parse_ts(r[0]) for r in rows) if ts is not None]

    def count(self) -> int:
        conn = self._pool.connection()
        return int(conn.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0])


def _frame_columns() -> list[str]:
    return ["id", *_INSERT_COLUMNS]


def _quote_from_row(row: pd.Series) -> Quote:
    return Quote(
        race_uid=str(row["race_uid"]),
        horse_key=str(row["horse_key"]),
        bookmaker=str(row["bookmaker"]),
        market_type=str(row["market_type"]),
        odds_decimal=float(row["odds_decimal"]),
        fetched_at=_parse_ts(row["fetched_at"]),
        as_of=_parse_ts(row["as_of"]),
        age_seconds=float(row["age_seconds"]),
        is_stale=bool(row["is_stale"]),
        sp=_as_opt_float(row.get("sp")),
        ew_places=_as_opt_int(row.get("ew_places")),
        ew_reduction=_as_opt_float(row.get("ew_reduction")),
        horse_name=(str(row["horse_name"]) if _clean_str(row.get("horse_name")) else None),
    )


# ── guards and ingestion ─────────────────────────────────────────────────────
def assert_point_in_time(frame: Any, as_of: Any) -> None:
    """Raise :class:`FuturePriceError` if anything in ``frame`` postdates ``as_of``.

    Cheap enough to assert on every point-in-time read. Prefer calling it at the
    boundary of any new code path that builds a price view by hand: a leak here is
    silent and inflates every downstream number.
    """
    moment = _parse_ts(as_of)
    if moment is None:
        raise ValueError("assert_point_in_time requires an as_of instant")

    if isinstance(frame, pd.DataFrame):
        if frame.empty or "fetched_at" not in frame.columns:
            return
        stamps = [_parse_ts(v) for v in frame["fetched_at"]]
    elif isinstance(frame, Mapping):
        stamps = [_parse_ts(frame.get("fetched_at"))]
    else:
        stamps = [
            _parse_ts(r.get("fetched_at") if isinstance(r, Mapping) else r)
            for r in (frame or [])
        ]

    future = [ts for ts in stamps if ts is not None and ts > moment]
    if future:
        raise FuturePriceError(
            f"{len(future)} row(s) priced after as_of={_iso(moment)} "
            f"(latest {_iso(max(future))}) — a later price cannot stand in for "
            "the price available at the decision instant"
        )


def ingest_live_odds_parquet(
    path: str = "data/live_odds.parquet",
    *,
    store: SnapshotStore | None = None,
    cfg: ExecutionConfig | None = None,
) -> SnapshotWriteResult:
    """Append the scraped ``data/live_odds.parquet`` board into the snapshot store.

    Only ``validation_status == 'VALID'`` rows are ingested; everything else is
    counted as rejected with reason ``invalid_status``. Prices the market-validation
    layer already distrusted must not become the historical record a backtest then
    treats as executable.
    """
    resolved = _resolve_db_path(path)
    if not os.path.exists(resolved):
        logger.warning("snapshots: %s does not exist; nothing ingested", resolved)
        return SnapshotWriteResult(0, 0, 0, {"missing_file": 1})

    frame = pd.read_parquet(resolved)
    total = len(frame)
    if "validation_status" in frame.columns:
        status = frame["validation_status"].astype("string").str.strip().str.upper()
        valid = frame[status == "VALID"]
    else:
        # No status column means nothing vouched for these prices — reject all.
        valid = frame.iloc[0:0]
    invalid = total - len(valid)

    owned = store is None
    target = store if store is not None else SnapshotStore(cfg=cfg)
    try:
        result = target.record_frame(valid)
    finally:
        if owned:
            target.close()

    if not invalid:
        return result
    reasons = dict(result.reasons)
    reasons["invalid_status"] = reasons.get("invalid_status", 0) + invalid
    return SnapshotWriteResult(
        inserted=result.inserted,
        duplicates=result.duplicates,
        rejected=result.rejected + invalid,
        reasons=reasons,
    )


__all__ = [
    "AppendOnlyViolation",
    "FuturePriceError",
    "Quote",
    "SnapshotStore",
    "SnapshotWriteResult",
    "assert_point_in_time",
    "horse_key",
    "ingest_live_odds_parquet",
    "race_key",
    "race_uid",
]
