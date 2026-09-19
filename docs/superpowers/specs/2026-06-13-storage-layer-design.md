# Storage Layer Design — `utils/storage`

**Date:** 2026-06-13
**Status:** Approved (pending spec review)
**Component:** `utils/storage/` — unified persistence layer (SQLite live metadata + Parquet historical/feature data)

## Goal

Provide one storage facade for Race Predictor v3 that:

- Holds **live metadata** in SQLite (`data/races.db`): scrape-run tracking, a race/runner catalog index, a queryable live-odds snapshot, and cache/lock coordination.
- Holds **historical/feature data** in Parquet, centralizing the read-merge-write and year-partition logic currently duplicated across scraper writers.
- Is **thread- and process-safe** for concurrent scraper runs (WAL + busy_timeout + DB-level advisory locks, plus a per-process thread-local connection pool).
- Supports **versioned schema migrations** via `PRAGMA user_version`.

Existing per-module parquet writers (`scraper/timeform/writer.py`, `scraper/betsp/writer.py`, boylesports `_merge_parquet`, livescorebet) are **migrated** to thin wrappers over this layer; their existing test suites must remain green as the regression guard.

## Approach

Chosen: a public facade (`utils/storage/__init__.py` exposing `Storage`/`get_storage`) over a small internal package of focused units — mirroring the codebase's existing `client/parser/writer` separation. Rejected: a single god-module (would mix four concerns and grow past maintainability), and two unconnected public modules (callers should not care which backend a dataset lives in).

## Module Structure

```
utils/storage/
  __init__.py        — public API: Storage facade, get_storage() singleton, StorageError
  pool.py            — ConnectionPool: thread-local SQLite connections (WAL, busy_timeout, synchronous=NORMAL)
  sqlite_store.py    — live-metadata CRUD over the five tables
  parquet_store.py   — read-merge-write, year-partition, dedupe helpers (centralized)
  migrations.py      — ordered MIGRATIONS list + apply(); PRAGMA user_version
  locks.py           — DB-level advisory locks for cross-process dataset coordination
```

## SQLite Schema (`data/races.db`, WAL mode)

`PRAGMA user_version` tracks schema version and drives migrations.

- **`scrape_runs`** — `id INTEGER PK, source TEXT, dataset TEXT, started_at TEXT, finished_at TEXT, status TEXT(running|ok|error), row_count INTEGER, error TEXT, host_pid INTEGER`
- **`race_catalog`** — `race_id TEXT PK, source TEXT, venue TEXT, race_time TEXT, race_date TEXT, status TEXT(open|resulted), market_type TEXT, parquet_path TEXT, partition TEXT, updated_at TEXT`. The fast "what's live now" index.
- **`live_odds`** — `source TEXT, race_id TEXT, horse_id TEXT, horse_name TEXT, market_type TEXT, odds_decimal REAL, sp REAL, currency TEXT, fetched_at TEXT`, PK `(source, race_id, horse_id, market_type)`, upsert on conflict. Queryable snapshot in addition to `live_odds.parquet` (confirmed wanted, despite duplicating the parquet data).
- **`cache_meta`** — `key TEXT PK, source TEXT, fetched_at TEXT, ttl_seconds INTEGER, fresh_until TEXT`
- **`locks`** — `name TEXT PK, owner TEXT, acquired_at TEXT, expires_at TEXT`

All timestamps are Europe/Dublin ISO strings via `utils/timezone.py` (`now`, `to_local`), consistent with the rest of the project.

## Connection Pool & Thread/Process Safety

- One `sqlite3.Connection` per thread (thread-local store). Connections opened with `check_same_thread=False` (access mediated by the pool), `PRAGMA journal_mode=WAL`, `PRAGMA busy_timeout=5000`, `PRAGMA synchronous=NORMAL`, `PRAGMA foreign_keys=ON`.
- Within a process, writes are serialized by a `threading.RLock`.
- Across processes, safety comes from WAL + busy_timeout (concurrent readers + single writer), plus the `locks` table for coarse "source X is writing dataset Y" advisory coordination with expiry (stale-lock reclaim).
- Parquet read-merge-write is guarded by an advisory lock keyed on the dataset path so two processes do not clobber each other's merge.

## Public API (`Storage` facade)

```python
s = get_storage()                      # singleton per process; resolves data/races.db, applies migrations

# --- SQLite live metadata ---
run_id = s.start_run(source, dataset)
s.finish_run(run_id, status, row_count=None, error=None)
s.upsert_races(rows: list[dict])
s.get_live_races(venue=None, status="open") -> list[dict]
s.upsert_odds(rows: list[dict])
s.query_odds(venue=None, horse_id=None) -> list[dict]
s.cache_fresh(key) -> bool
s.touch_cache(key, ttl_seconds, source=None)
with s.lock("dataset:live_odds", ttl=60): ...   # contextmanager, cross-process

# --- Parquet historical/feature ---
s.write_parquet(df, dataset, partition_by=None, dedupe_key=None, columns=None)
s.append_parquet(df, dataset, partition_by=None, dedupe_key=None, columns=None)  # read-merge-write
s.read_parquet(dataset, filters=None) -> pd.DataFrame
```

Datasets are resolved by **logical name** → path via a registry (`live_odds`, `betsp`, `timeform`, `training`, `unified`, `races_db`), so callers never hardcode paths. Paths come from existing config/conventions used today.

## Migrations

- `migrations.py` holds `MIGRATIONS`: an ordered list of `(version, apply_fn)` (or SQL) steps.
- On `get_storage()`, the layer reads `PRAGMA user_version` and applies any pending migrations in a transaction, then bumps the version. Idempotent — re-running is a no-op.
- CLI: `python -m utils.storage migrate` applies pending migrations manually; `python -m utils.storage version` prints current/target versions.
- Migration `1` creates all five tables described above.

## Migrating Existing Writers

`scraper/timeform/writer.py`, `scraper/betsp/writer.py`, and boylesports' `_merge_parquet` are refactored to delegate to `s.write_parquet(...)` / `s.append_parquet(...)`, preserving their existing `FINAL_COLUMNS` / `_PARQUET_COLUMNS` and `_DEDUPE_KEY`. The `shutil.rmtree`-before-partitioned-write workaround moves into `parquet_store.py`. Each writer's existing tests must pass unchanged.

## Error Handling

- DB locked beyond `busy_timeout` → raise `StorageError` (wraps `sqlite3.OperationalError`).
- Corrupt DB on open → raise `StorageError` with guidance; migrations never partially apply (transactional).
- Parquet read failure during merge → log warning and overwrite, matching current writer behavior (no data-loss regression vs today).
- Advisory lock held by a dead owner past `expires_at` → reclaimable.

## Testing

- **Pool/thread-safety:** concurrent threads writing runs/odds; assert no corruption, all rows present.
- **Migrations:** fresh DB applies to latest version; re-apply is no-op; `user_version` correct.
- **SQLite store:** each method (start/finish run, upsert/get races, upsert/query odds, cache freshness, lock acquire/release/expiry).
- **Locks:** mutual exclusion within and across simulated owners; stale-lock reclaim.
- **Parquet store:** round-trip, dedupe by key, year-partition write, rmtree-before-partition correctness.
- **Regression:** existing `tests/scraper/timeform/test_writer.py`, `tests/scraper/betsp/test_writer.py`, boylesports/livescorebet writer paths pass unchanged after migration.

## Out of Scope (YAGNI)

- No external migration framework (Alembic) — `user_version` suffices for a single-file SQLite DB.
- No connection pooling library — thread-local sqlite3 connections are sufficient.
- No async API.
