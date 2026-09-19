"""Versioned SQLite migrations driven by PRAGMA user_version."""
import re
import sqlite3
import sys
from datetime import datetime, timezone

from utils.logger import get_logger
from utils.storage.pool import ConnectionPool

logger = get_logger(__name__)

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def _norm_venue(value) -> str:
    """Local copy of ``utils.text_norm.norm_venue`` — a migration must not import
    application feature code, only stdlib, so this schema change survives even if
    that module's import graph shifts."""
    return _PUNCT_RE.sub("", str(value or "").lower())


def _minute_key_utc(value) -> str:
    """Best-effort UTC minute-precision key for an already-stored timestamp
    string. Mirrors ``execution.snapshots._parse_ts`` — naive input is read as
    UTC (the convention every writer of these two tables already uses), so the
    backfilled ``race_key`` agrees with what new rows compute going forward."""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Tolerate a bare "YYYY-MM-DD HH:MM[:SS[.ffffff]]" with no offset.
        try:
            dt = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _race_key_for(venue, race_time) -> str:
    v = _norm_venue(venue)
    t = _minute_key_utc(race_time)
    return f"{v}|{t}" if v and t else ""


def _migration_1(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE scrape_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            dataset TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            row_count INTEGER,
            error TEXT,
            host_pid INTEGER
        );
        CREATE TABLE race_catalog (
            race_id TEXT PRIMARY KEY,
            source TEXT,
            venue TEXT,
            race_time TEXT,
            race_date TEXT,
            status TEXT,
            market_type TEXT,
            parquet_path TEXT,
            partition TEXT,
            updated_at TEXT
        );
        CREATE TABLE live_odds (
            source TEXT NOT NULL,
            race_id TEXT NOT NULL,
            horse_id TEXT NOT NULL,
            horse_name TEXT,
            market_type TEXT NOT NULL,
            odds_decimal REAL,
            sp REAL,
            currency TEXT,
            fetched_at TEXT,
            PRIMARY KEY (source, race_id, horse_id, market_type)
        );
        CREATE TABLE cache_meta (
            key TEXT PRIMARY KEY,
            source TEXT,
            fetched_at TEXT,
            ttl_seconds INTEGER,
            fresh_until TEXT
        );
        CREATE TABLE locks (
            name TEXT PRIMARY KEY,
            owner TEXT,
            acquired_at TEXT,
            expires_at TEXT
        );
        CREATE INDEX idx_catalog_venue_status ON race_catalog(venue, status);
        CREATE INDEX idx_runs_source ON scrape_runs(source, dataset);
        """
    )


def _migration_2(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placed_at TEXT NOT NULL,
            race_id TEXT,
            horse_id TEXT,
            horse_name TEXT NOT NULL,
            venue TEXT,
            race_time TEXT,
            bet_type TEXT NOT NULL DEFAULT 'win',
            stake REAL NOT NULL,
            odds_decimal REAL NOT NULL,
            composite_score REAL,
            won_prob REAL,
            strategy TEXT NOT NULL DEFAULT 'flat',
            bankroll_before REAL,
            outcome TEXT,
            settled_at TEXT,
            gross_return REAL,
            profit REAL,
            notes TEXT
        );
        CREATE TABLE bankroll_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            bet_id INTEGER REFERENCES bets(id),
            amount REAL NOT NULL,
            balance REAL NOT NULL
        );
        CREATE INDEX idx_bets_placed_at ON bets(placed_at);
        CREATE INDEX idx_bets_outcome ON bets(outcome);
        CREATE INDEX idx_bets_horse ON bets(horse_id);
        """
    )


def _migration_3(conn: sqlite3.Connection) -> None:
    # Paper-betting CLV capture: the model's edge at bet time plus the closing
    # price (Betfair SP) and the resulting closing-line value, filled at settlement.
    conn.executescript(
        """
        ALTER TABLE bets ADD COLUMN value_edge REAL;
        ALTER TABLE bets ADD COLUMN closing_odds REAL;
        ALTER TABLE bets ADD COLUMN clv_pct REAL;
        """
    )


def _migration_4(conn: sqlite3.Connection) -> None:
    # Append-only point-in-time odds store (Stage-5 requirement 1). The live_odds
    # table from migration 1 is an UPSERT cache: it keeps only the newest price per
    # (source, race, horse), so a backtest reading it silently sees the *closing*
    # price wherever an earlier one existed. That is the look-ahead the Stage-4
    # audit (reports/calibration_audit_20260727.md) punished. This table keeps every
    # observation keyed by fetched_at instead, and the RAISE(ABORT) triggers make the
    # history physically unrewritable — no code path, however well intentioned, can
    # retro-fit a better price onto a past decision.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS odds_snapshots (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            race_uid          TEXT NOT NULL,
            race_id           TEXT,
            race_time         TEXT,
            venue             TEXT,
            horse_key         TEXT NOT NULL,
            horse_id          TEXT,
            horse_name        TEXT,
            bookmaker         TEXT NOT NULL,
            market_type       TEXT NOT NULL DEFAULT 'WIN',
            odds_decimal      REAL,
            sp                REAL,
            ew_places         INTEGER,
            ew_reduction      REAL,
            ew_margin         REAL,
            field_size        INTEGER,
            booksum           REAL,
            validation_status TEXT,
            fetched_at        TEXT NOT NULL,
            ingested_at       TEXT NOT NULL,
            UNIQUE (race_uid, horse_key, bookmaker, market_type, fetched_at)
        );
        CREATE INDEX IF NOT EXISTS idx_odds_snapshots_race_time ON odds_snapshots(race_uid, fetched_at);
        CREATE INDEX IF NOT EXISTS idx_odds_snapshots_fetched ON odds_snapshots(fetched_at);
        CREATE TRIGGER IF NOT EXISTS odds_snapshots_no_update
            BEFORE UPDATE ON odds_snapshots
            BEGIN SELECT RAISE(ABORT, 'odds_snapshots is append-only: UPDATE forbidden'); END;
        CREATE TRIGGER IF NOT EXISTS odds_snapshots_no_delete
            BEFORE DELETE ON odds_snapshots
            BEGIN SELECT RAISE(ABORT, 'odds_snapshots is append-only: DELETE forbidden'); END;
        """
    )


def _migration_5(conn: sqlite3.Connection) -> None:
    # Paper ticket storage (Stage-5 requirement 8). Every candidate — including the
    # PASS/"no bet" ones, which are the expected default under the Stage-4 MODEL
    # NO-GO — is recorded with the full disclosure record: the quote used, its age,
    # the gate decision and why, plus the model/forward-gate state at issue time.
    # `CHECK (paper_only = 1)` is deliberate: a real-money ticket is unrepresentable
    # in the storage layer itself, not merely refused by application code.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS paper_tickets (
            ticket_id            TEXT PRIMARY KEY,
            issued_at            TEXT NOT NULL,
            as_of                TEXT,
            race_uid             TEXT NOT NULL,
            race_id              TEXT,
            race_time            TEXT,
            venue                TEXT,
            horse_key            TEXT NOT NULL,
            horse_id             TEXT,
            horse_name           TEXT,
            bookmaker            TEXT,
            market_type          TEXT NOT NULL DEFAULT 'WIN',
            bet_type             TEXT NOT NULL DEFAULT 'win',
            offered_odds         REAL,
            quote_fetched_at     TEXT,
            quote_age_seconds    REAL,
            model_prob           REAL,
            market_adjusted_prob REAL,
            fair_odds            REAL,
            market_prob          REAL,
            edge                 REAL,
            expected_value       REAL,
            max_stake            REAL,
            stake                REAL,
            ew_places            INTEGER,
            ew_reduction         REAL,
            data_quality         TEXT,
            validation_state     TEXT NOT NULL,
            decision             TEXT NOT NULL,
            pass_reasons         TEXT,
            reasons_passed       TEXT,
            paper_only           INTEGER NOT NULL DEFAULT 1 CHECK (paper_only = 1),
            model_verdict        TEXT,
            forward_gate_state   TEXT,
            settled_at           TEXT,
            outcome              TEXT,
            returns              REAL,
            profit               REAL,
            closing_odds         REAL,
            clv_pct              REAL,
            settlement_detail    TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_paper_tickets_race ON paper_tickets(race_uid, horse_key, bet_type);
        CREATE INDEX IF NOT EXISTS idx_paper_tickets_issued ON paper_tickets(issued_at);
        CREATE INDEX IF NOT EXISTS idx_paper_tickets_open ON paper_tickets(settled_at);
        """
    )


def _migration_6(conn: sqlite3.Connection) -> None:
    # Append-only free-text archive (Stage-14 requirement). Mirrors migration 4's
    # odds_snapshots contract exactly: every incoming comment is kept, never
    # overwritten, keyed by its own content hash plus the observed row so a later
    # re-scrape of an EDITED comment becomes a new row rather than silently
    # replacing the version a past prediction actually saw. RAISE(ABORT) triggers
    # make that physically true, not just a code-review convention.
    # `published_at` is the source's OWN claimed publication time and is left
    # NULL whenever the source does not supply one — it is never backfilled from
    # `fetched_at` (our scrape time) or `archived_at` (our write time), which
    # would fabricate a precision the source never gave. `fetched_at` is the
    # honest point-in-time anchor: the instant our own scraper actually saw the
    # text, always known, always <= `archived_at`.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS text_archive (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash  TEXT NOT NULL,
            source        TEXT NOT NULL,
            text_kind     TEXT NOT NULL DEFAULT 'spotlight',
            race_uid      TEXT NOT NULL,
            runner_key    TEXT NOT NULL,
            horse_id      TEXT,
            horse_name    TEXT,
            text          TEXT NOT NULL,
            published_at  TEXT,
            fetched_at    TEXT NOT NULL,
            archived_at   TEXT NOT NULL,
            metadata      TEXT,
            UNIQUE (content_hash, source, race_uid, runner_key, fetched_at)
        );
        CREATE INDEX IF NOT EXISTS idx_text_archive_lookup
            ON text_archive(race_uid, runner_key, archived_at);
        CREATE INDEX IF NOT EXISTS idx_text_archive_hash ON text_archive(content_hash);
        CREATE TRIGGER IF NOT EXISTS text_archive_no_update
            BEFORE UPDATE ON text_archive
            BEGIN SELECT RAISE(ABORT, 'text_archive is append-only: UPDATE forbidden'); END;
        CREATE TRIGGER IF NOT EXISTS text_archive_no_delete
            BEFORE DELETE ON text_archive
            BEGIN SELECT RAISE(ABORT, 'text_archive is append-only: DELETE forbidden'); END;
        """
    )


def _migration_7(conn: sqlite3.Connection) -> None:
    # Append-only hosted-LLM shadow-extraction ledger (Stage-19 requirement).
    # Mirrors migration 6's text_archive contract: every extraction attempt is a
    # new row, never overwritten, RAISE(ABORT) triggers make that physically
    # true. Distinct from ``data/cache/llm``'s hosted-adapter cache (a TTL'd
    # performance cache the adapter itself manages) and from
    # ``llm/hosted_budget.py``'s spend ledger (a running total, not a per-row
    # audit trail) -- this table is the durable, queryable record of WHICH
    # archived comment was extracted, by which model/prompt/schema version,
    # and WHEN the extraction itself became available, kept separate from
    # ``text_archive.fetched_at`` (when the underlying text was scraped) so a
    # downstream reader can never conflate "the text existed pre-race" with
    # "the extracted feature existed pre-race" -- an extraction run well after
    # a race went off must never be readable as though it were available at
    # scrape time.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS llm_shadow_extractions (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            archive_row_id     INTEGER NOT NULL REFERENCES text_archive(id),
            content_hash       TEXT NOT NULL,
            source             TEXT NOT NULL,
            race_uid           TEXT NOT NULL,
            runner_key         TEXT NOT NULL,
            horse_id           TEXT,
            horse_name         TEXT,
            text_published_at  TEXT,
            text_fetched_at    TEXT NOT NULL,
            extraction_time    TEXT NOT NULL,
            provider           TEXT NOT NULL,
            model              TEXT NOT NULL,
            backend            TEXT NOT NULL,
            schema_version     INTEGER NOT NULL,
            prompt_version     INTEGER NOT NULL,
            schema_valid       INTEGER NOT NULL,
            is_cache_hit       INTEGER NOT NULL,
            failure_reason     TEXT,
            cost_usd           REAL NOT NULL DEFAULT 0.0,
            features           TEXT NOT NULL,
            raw_response_hash  TEXT,
            UNIQUE (archive_row_id, model, schema_version, prompt_version)
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_extractions_race
            ON llm_shadow_extractions(race_uid, runner_key);
        CREATE INDEX IF NOT EXISTS idx_shadow_extractions_archive_row
            ON llm_shadow_extractions(archive_row_id);
        CREATE INDEX IF NOT EXISTS idx_shadow_extractions_extraction_time
            ON llm_shadow_extractions(extraction_time);
        CREATE TRIGGER IF NOT EXISTS llm_shadow_extractions_no_update
            BEFORE UPDATE ON llm_shadow_extractions
            BEGIN SELECT RAISE(ABORT, 'llm_shadow_extractions is append-only: UPDATE forbidden'); END;
        CREATE TRIGGER IF NOT EXISTS llm_shadow_extractions_no_delete
            BEFORE DELETE ON llm_shadow_extractions
            BEGIN SELECT RAISE(ABORT, 'llm_shadow_extractions is append-only: DELETE forbidden'); END;
        """
    )


def _migration_8(conn: sqlite3.Connection) -> None:
    # Stage-20 identity/provenance repair. Two real, reproduced defects:
    #
    # (B5) ``execution.snapshots.race_uid`` drops venue the moment an off-time
    # parses — two different venues going off at the identical minute collide
    # onto one identity (reproduced on the live store, D37). It also renders at
    # SECOND precision while every other identity function in the codebase
    # (``features.derive.add_race_key``, ``execution.race_facts.race_facts_key``)
    # uses MINUTE precision, so sub-minute jitter between two sources scraping
    # the same physical race also fragments it.
    #
    # The fix is additive, not a rewrite of the immutable ``race_uid`` column:
    # a new ``race_key`` column, ALWAYS venue-qualified and minute-precision
    # (``norm_venue(venue)|YYYY-MM-DDTHH:MM``, matching
    # ``execution.race_facts.race_facts_key``'s scheme), computed from each
    # row's OWN already-stored ``venue``/``race_time`` columns — no external
    # mapping table is needed because both source columns already exist on
    # every row of both tables. Old rows keep their original ``race_uid``
    # forever (append-only, never rewritten); new code additionally
    # reads/writes ``race_key`` for grouping, duplicate-detection and quote
    # lookups, so a same-off-time different-venue race can no longer be
    # confused with another, while every existing ``race_uid``-keyed join
    # keeps working unchanged. Recomputing race_key from those two stored
    # columns is pure and deterministic, so re-running this migration's
    # backfill is idempotent.
    #
    # ``odds_snapshots`` carries its own ``RAISE(ABORT)`` triggers forbidding
    # UPDATE (migration 4) — that is the whole point of the table, so this
    # migration must NOT (and structurally cannot) backfill ``race_key`` onto
    # existing snapshot rows; they honestly keep a NULL ``race_key`` forever,
    # exactly like a pre-Stage-20 ticket's NULL provenance below. Every row
    # written by ``execution.snapshots.SnapshotStore.record`` AFTER this
    # migration computes its own ``race_key`` at write time (application code,
    # not this migration), so the gap is bounded to rows captured before this
    # deploy and never silently mis-resolved. ``paper_tickets`` carries no
    # such trigger (``TicketStore.settle`` already updates it), so its
    # backfill below is safe.
    #
    # (B7) ``paper_tickets`` carries no model/feature/config identity at all —
    # a ticket cannot be traced back to the artifact bundle or configuration
    # that produced it. Four new nullable columns record that provenance for
    # every ticket built from now on; old rows correctly stay NULL (they
    # predate the contract, not a fabricated backfill) and
    # ``provenance_complete`` (set by application code, never by this
    # migration) is what a forward-gate eligibility check actually reads.
    # ``ALTER TABLE ... ADD COLUMN`` has no ``IF NOT EXISTS`` in SQLite, and a
    # partially-applied prior attempt (e.g. this same process crashing between
    # adding columns and the backfill below) must be resumable without a
    # "duplicate column" error — so each ADD COLUMN is guarded by an explicit
    # existence check against the table's OWN current schema, not just tried
    # and swallowed blind.
    def _add_column_if_missing(table: str, column: str, coltype: str) -> None:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

    _add_column_if_missing("odds_snapshots", "race_key", "TEXT")
    _add_column_if_missing("paper_tickets", "race_key", "TEXT")
    _add_column_if_missing("paper_tickets", "model_content_hash", "TEXT")
    _add_column_if_missing("paper_tickets", "feature_schema_version", "TEXT")
    _add_column_if_missing("paper_tickets", "config_hash", "TEXT")
    _add_column_if_missing("paper_tickets", "prediction_cycle_id", "TEXT")
    _add_column_if_missing("paper_tickets", "provenance_complete", "INTEGER")
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_odds_snapshots_race_key
            ON odds_snapshots(race_key, fetched_at);
        CREATE INDEX IF NOT EXISTS idx_paper_tickets_race_key
            ON paper_tickets(race_key, horse_key, bet_type);
        """
    )
    # Backfill (paper_tickets only — see above): pure function of each row's
    # own (venue, race_time), so a re-run (this same restart-safety concern)
    # simply recomputes the identical values — safe to repeat unconditionally,
    # and cheap enough (thousands, not millions, of rows) not to bother
    # skipping already-filled rows.
    rows = conn.execute("SELECT rowid, venue, race_time FROM paper_tickets").fetchall()
    updates = [
        (_race_key_for(venue, race_time), rowid)
        for rowid, venue, race_time in rows
    ]
    conn.executemany(
        "UPDATE paper_tickets SET race_key = ? WHERE rowid = ?",
        [(key, rowid) for key, rowid in updates if key],
    )


# Ordered: (version, apply_fn). Append new migrations; never edit shipped ones.
MIGRATIONS = [
    (1, _migration_1),
    (2, _migration_2),
    (3, _migration_3),
    (4, _migration_4),
    (5, _migration_5),
    (6, _migration_6),
    (7, _migration_7),
    (8, _migration_8),
]


def target_version() -> int:
    return MIGRATIONS[-1][0]


def current_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def apply_migrations(pool: ConnectionPool) -> int:
    conn = pool.connection()
    with pool.write_lock():
        version = current_version(conn)
        for ver, fn in MIGRATIONS:
            if ver > version:
                logger.info("storage: applying migration %d", ver)
                fn(conn)
                conn.execute(f"PRAGMA user_version={ver}")
                conn.commit()
                version = ver
    return version


def _main(argv):
    from utils.storage import DEFAULT_DB_PATH
    pool = ConnectionPool(DEFAULT_DB_PATH)
    cmd = argv[0] if argv else "migrate"
    if cmd == "version":
        print(f"current={current_version(pool.connection())} target={target_version()}")
    elif cmd == "migrate":
        v = apply_migrations(pool)
        print(f"migrated to version {v}")
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    pool.close_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
