# Storage Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `utils/storage/` — a unified persistence facade with a SQLite live-metadata store (WAL, pooled, migrated) and a centralized Parquet read-merge-write store, then migrate existing scraper writers onto it.

**Architecture:** A `Storage` facade (`utils/storage/__init__.py`) over focused internal units: `pool.py` (thread-local SQLite connections), `migrations.py` (`PRAGMA user_version`), `sqlite_store.py` (five live-metadata tables), `locks.py` (cross-process advisory locks), `parquet_store.py` (read-merge-write + year-partition). Existing per-module writers become thin wrappers; their tests are the regression guard.

**Tech Stack:** Python 3.14, `sqlite3` (stdlib), `pandas` + `pyarrow`, `pytest`. Timestamps via `utils/timezone.py`. Logging via `utils/logger.py`.

**No git repo:** This project has no git repo, so "Commit" steps are replaced by a `pytest -q` full-suite-green checkpoint. Run from the project root `C:\Users\mshr\Desktop\Race Predictor v3`.

---

## File Structure

```
utils/storage/
  __init__.py        — Storage facade, get_storage() singleton, StorageError, DATASETS registry
  pool.py            — ConnectionPool: thread-local connections, PRAGMAs, write RLock
  migrations.py      — MIGRATIONS list, apply_migrations(), CLI entry (migrate/version)
  sqlite_store.py    — SqliteStore: runs, race_catalog, live_odds, cache_meta CRUD
  locks.py           — advisory_lock() contextmanager over the locks table
  parquet_store.py   — write_parquet/append_parquet/read_parquet helpers
tests/utils/storage/
  test_pool.py
  test_migrations.py
  test_sqlite_store.py
  test_locks.py
  test_parquet_store.py
  test_facade.py
```

`utils/storage.py` does not exist as a file — the package `utils/storage/` provides the public API via `__init__.py`. (No existing module imports `utils.storage` today, so there is no collision.)

---

### Task 1: Connection pool

**Files:**

- Create: `utils/storage/pool.py`
- Test: `tests/utils/storage/test_pool.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_pool.py
import threading
from utils.storage.pool import ConnectionPool


def test_same_thread_returns_same_connection(tmp_path):
    pool = ConnectionPool(str(tmp_path / "t.db"))
    a = pool.connection()
    b = pool.connection()
    assert a is b
    pool.close_all()


def test_wal_and_busy_timeout_set(tmp_path):
    pool = ConnectionPool(str(tmp_path / "t.db"))
    conn = pool.connection()
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    pool.close_all()


def test_different_threads_get_different_connections(tmp_path):
    pool = ConnectionPool(str(tmp_path / "t.db"))
    seen = {}

    def grab(name):
        seen[name] = id(pool.connection())

    main_id = id(pool.connection())
    t = threading.Thread(target=grab, args=("worker",))
    t.start(); t.join()
    assert seen["worker"] != main_id
    pool.close_all()


def test_write_lock_serializes(tmp_path):
    pool = ConnectionPool(str(tmp_path / "t.db"))
    pool.connection().execute("CREATE TABLE c(n INTEGER)")
    pool.connection().commit()

    def bump():
        for _ in range(50):
            with pool.write_lock():
                conn = pool.connection()
                conn.execute("INSERT INTO c VALUES (1)")
                conn.commit()

    threads = [threading.Thread(target=bump) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert pool.connection().execute("SELECT COUNT(*) FROM c").fetchone()[0] == 200
    pool.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.storage'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/pool.py
"""Thread-local SQLite connection pool with WAL and a per-process write lock."""
import os
import sqlite3
import threading
from contextlib import contextmanager

from utils.logger import get_logger

logger = get_logger(__name__)

_BUSY_TIMEOUT_MS = 5000


class ConnectionPool:
    """One sqlite3.Connection per thread; writes serialized by a shared RLock."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._all_conns = []
        self._all_lock = threading.Lock()

    def connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
            with self._all_lock:
                self._all_conns.append(conn)
        return conn

    @contextmanager
    def write_lock(self):
        with self._write_lock:
            yield

    def close_all(self):
        with self._all_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._all_conns.clear()
        self._local = threading.local()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_pool.py -v`
Expected: PASS (4 passed). Also create empty `tests/utils/storage/__init__.py` if the suite uses package test dirs (match existing `tests/utils/` convention — check whether `tests/utils/__init__.py` exists and mirror it).

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite still green (191 + 4 new).

---

### Task 2: Migrations

**Files:**

- Create: `utils/storage/migrations.py`
- Test: `tests/utils/storage/test_migrations.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_migrations.py
from utils.storage.pool import ConnectionPool
from utils.storage.migrations import apply_migrations, target_version, current_version


def test_fresh_db_migrates_to_target(tmp_path):
    pool = ConnectionPool(str(tmp_path / "m.db"))
    apply_migrations(pool)
    conn = pool.connection()
    assert current_version(conn) == target_version()
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert {"scrape_runs", "race_catalog", "live_odds", "cache_meta", "locks"} <= names
    pool.close_all()


def test_reapply_is_noop(tmp_path):
    pool = ConnectionPool(str(tmp_path / "m.db"))
    apply_migrations(pool)
    v1 = current_version(pool.connection())
    apply_migrations(pool)  # second run must not error or change version
    assert current_version(pool.connection()) == v1
    pool.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_migrations.py -v`
Expected: FAIL — `No module named 'utils.storage.migrations'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/migrations.py
"""Versioned SQLite migrations driven by PRAGMA user_version."""
import sqlite3
import sys

from utils.logger import get_logger
from utils.storage.pool import ConnectionPool

logger = get_logger(__name__)


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


# Ordered: (version, apply_fn). Append new migrations; never edit shipped ones.
MIGRATIONS = [
    (1, _migration_1),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_migrations.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 3: SQLite store (runs, catalog, odds, cache)

**Files:**

- Create: `utils/storage/sqlite_store.py`
- Test: `tests/utils/storage/test_sqlite_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_sqlite_store.py
from utils.storage.pool import ConnectionPool
from utils.storage.migrations import apply_migrations
from utils.storage.sqlite_store import SqliteStore


def _store(tmp_path):
    pool = ConnectionPool(str(tmp_path / "s.db"))
    apply_migrations(pool)
    return SqliteStore(pool), pool


def test_run_lifecycle(tmp_path):
    store, pool = _store(tmp_path)
    rid = store.start_run("boylesports", "live_odds")
    store.finish_run(rid, "ok", row_count=42)
    row = store.connection().execute(
        "SELECT status, row_count, finished_at FROM scrape_runs WHERE id=?", (rid,)
    ).fetchone()
    assert row["status"] == "ok" and row["row_count"] == 42 and row["finished_at"]
    pool.close_all()


def test_upsert_and_get_live_races(tmp_path):
    store, pool = _store(tmp_path)
    store.upsert_races([
        {"race_id": "r1", "source": "boylesports", "venue": "Ascot",
         "race_time": "2026-06-13T14:00:00+01:00", "race_date": "2026-06-13",
         "status": "open", "market_type": "WIN"},
        {"race_id": "r2", "source": "boylesports", "venue": "Ascot",
         "race_time": "2026-06-13T13:00:00+01:00", "race_date": "2026-06-13",
         "status": "resulted", "market_type": "WIN"},
    ])
    store.upsert_races([{"race_id": "r1", "source": "boylesports", "venue": "Ascot",
                         "status": "resulted"}])  # update existing
    open_races = store.get_live_races(venue="Ascot", status="open")
    assert open_races == []
    resulted = store.get_live_races(venue="Ascot", status="resulted")
    assert {r["race_id"] for r in resulted} == {"r1", "r2"}
    pool.close_all()


def test_upsert_and_query_odds(tmp_path):
    store, pool = _store(tmp_path)
    store.upsert_odds([
        {"source": "boylesports", "race_id": "r1", "horse_id": "h1",
         "horse_name": "Bold", "market_type": "WIN", "odds_decimal": 5.0},
    ])
    store.upsert_odds([
        {"source": "boylesports", "race_id": "r1", "horse_id": "h1",
         "horse_name": "Bold", "market_type": "WIN", "odds_decimal": 4.5},
    ])  # same PK -> update price
    rows = store.query_odds(horse_id="h1")
    assert len(rows) == 1 and rows[0]["odds_decimal"] == 4.5
    pool.close_all()


def test_cache_freshness(tmp_path):
    store, pool = _store(tmp_path)
    assert store.cache_fresh("paddy") is False
    store.touch_cache("paddy", ttl_seconds=3600, source="paddy_power")
    assert store.cache_fresh("paddy") is True
    pool.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_sqlite_store.py -v`
Expected: FAIL — `No module named 'utils.storage.sqlite_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/sqlite_store.py
"""CRUD over the live-metadata SQLite tables."""
import os
from datetime import timedelta

from utils.logger import get_logger
from utils.storage.pool import ConnectionPool
from utils.timezone import now

logger = get_logger(__name__)


class SqliteStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def connection(self):
        return self._pool.connection()

    # --- scrape runs ---
    def start_run(self, source: str, dataset: str) -> int:
        with self._pool.write_lock():
            conn = self._pool.connection()
            cur = conn.execute(
                "INSERT INTO scrape_runs(source, dataset, started_at, status, host_pid) "
                "VALUES (?,?,?,?,?)",
                (source, dataset, now().isoformat(), "running", os.getpid()),
            )
            conn.commit()
            return cur.lastrowid

    def finish_run(self, run_id: int, status: str, row_count=None, error=None) -> None:
        with self._pool.write_lock():
            conn = self._pool.connection()
            conn.execute(
                "UPDATE scrape_runs SET finished_at=?, status=?, row_count=?, error=? "
                "WHERE id=?",
                (now().isoformat(), status, row_count, error, run_id),
            )
            conn.commit()

    # --- race catalog ---
    _CATALOG_COLS = ("race_id", "source", "venue", "race_time", "race_date",
                     "status", "market_type", "parquet_path", "partition")

    def upsert_races(self, rows) -> None:
        if not rows:
            return
        with self._pool.write_lock():
            conn = self._pool.connection()
            for row in rows:
                cols = [c for c in self._CATALOG_COLS if c in row]
                placeholders = ",".join("?" for _ in cols)
                updates = ",".join(f"{c}=excluded.{c}" for c in cols if c != "race_id")
                conn.execute(
                    f"INSERT INTO race_catalog({','.join(cols)}, updated_at) "
                    f"VALUES ({placeholders}, ?) "
                    f"ON CONFLICT(race_id) DO UPDATE SET {updates}, updated_at=excluded.updated_at",
                    tuple(row[c] for c in cols) + (now().isoformat(),),
                )
            conn.commit()

    def get_live_races(self, venue=None, status="open"):
        sql = "SELECT * FROM race_catalog WHERE 1=1"
        params = []
        if status is not None:
            sql += " AND status=?"; params.append(status)
        if venue is not None:
            sql += " AND venue=?"; params.append(venue)
        return [dict(r) for r in self._pool.connection().execute(sql, params).fetchall()]

    # --- live odds snapshot ---
    _ODDS_COLS = ("source", "race_id", "horse_id", "horse_name", "market_type",
                  "odds_decimal", "sp", "currency", "fetched_at")

    def upsert_odds(self, rows) -> None:
        if not rows:
            return
        with self._pool.write_lock():
            conn = self._pool.connection()
            stamp = now().isoformat()
            for row in rows:
                values = {c: row.get(c) for c in self._ODDS_COLS}
                if values["fetched_at"] is None:
                    values["fetched_at"] = stamp
                updates = ",".join(
                    f"{c}=excluded.{c}" for c in self._ODDS_COLS
                    if c not in ("source", "race_id", "horse_id", "market_type")
                )
                conn.execute(
                    f"INSERT INTO live_odds({','.join(self._ODDS_COLS)}) "
                    f"VALUES ({','.join('?' for _ in self._ODDS_COLS)}) "
                    f"ON CONFLICT(source, race_id, horse_id, market_type) DO UPDATE SET {updates}",
                    tuple(values[c] for c in self._ODDS_COLS),
                )
            conn.commit()

    def query_odds(self, venue=None, horse_id=None):
        # venue lives on race_catalog; join when filtering by it.
        if venue is not None:
            sql = ("SELECT o.* FROM live_odds o JOIN race_catalog c ON o.race_id=c.race_id "
                   "WHERE c.venue=?")
            params = [venue]
            if horse_id is not None:
                sql += " AND o.horse_id=?"; params.append(horse_id)
        else:
            sql = "SELECT * FROM live_odds WHERE 1=1"
            params = []
            if horse_id is not None:
                sql += " AND horse_id=?"; params.append(horse_id)
        return [dict(r) for r in self._pool.connection().execute(sql, params).fetchall()]

    # --- cache freshness ---
    def touch_cache(self, key: str, ttl_seconds: int, source=None) -> None:
        with self._pool.write_lock():
            conn = self._pool.connection()
            stamp = now()
            fresh_until = (stamp + timedelta(seconds=ttl_seconds)).isoformat()
            conn.execute(
                "INSERT INTO cache_meta(key, source, fetched_at, ttl_seconds, fresh_until) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET source=excluded.source, "
                "fetched_at=excluded.fetched_at, ttl_seconds=excluded.ttl_seconds, "
                "fresh_until=excluded.fresh_until",
                (key, source, stamp.isoformat(), ttl_seconds, fresh_until),
            )
            conn.commit()

    def cache_fresh(self, key: str) -> bool:
        row = self._pool.connection().execute(
            "SELECT fresh_until FROM cache_meta WHERE key=?", (key,)
        ).fetchone()
        if row is None or row["fresh_until"] is None:
            return False
        return row["fresh_until"] > now().isoformat()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_sqlite_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 4: Cross-process advisory locks

**Files:**

- Create: `utils/storage/locks.py`
- Test: `tests/utils/storage/test_locks.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_locks.py
import pytest
from utils.storage.pool import ConnectionPool
from utils.storage.migrations import apply_migrations
from utils.storage.locks import advisory_lock, LockHeldError


def _pool(tmp_path):
    pool = ConnectionPool(str(tmp_path / "l.db"))
    apply_migrations(pool)
    return pool


def test_lock_acquire_and_release(tmp_path):
    pool = _pool(tmp_path)
    with advisory_lock(pool, "dataset:live_odds", ttl=60, owner="A"):
        held = pool.connection().execute(
            "SELECT owner FROM locks WHERE name=?", ("dataset:live_odds",)).fetchone()
        assert held["owner"] == "A"
    # released after context exit
    assert pool.connection().execute(
        "SELECT 1 FROM locks WHERE name=?", ("dataset:live_odds",)).fetchone() is None
    pool.close_all()


def test_second_holder_blocked(tmp_path):
    pool = _pool(tmp_path)
    with advisory_lock(pool, "x", ttl=60, owner="A"):
        with pytest.raises(LockHeldError):
            with advisory_lock(pool, "x", ttl=60, owner="B", wait=False):
                pass
    pool.close_all()


def test_expired_lock_is_reclaimable(tmp_path):
    pool = _pool(tmp_path)
    # Insert an already-expired lock directly.
    conn = pool.connection()
    conn.execute("INSERT INTO locks(name, owner, acquired_at, expires_at) "
                 "VALUES ('x','dead','2000-01-01T00:00:00','2000-01-01T00:00:00')")
    conn.commit()
    with advisory_lock(pool, "x", ttl=60, owner="B", wait=False):
        assert pool.connection().execute(
            "SELECT owner FROM locks WHERE name='x'").fetchone()["owner"] == "B"
    pool.close_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_locks.py -v`
Expected: FAIL — `No module named 'utils.storage.locks'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/locks.py
"""Cross-process advisory locks backed by the `locks` table (with expiry)."""
import time
from contextlib import contextmanager
from datetime import timedelta

from utils.logger import get_logger
from utils.storage.pool import ConnectionPool
from utils.timezone import now

logger = get_logger(__name__)


class LockHeldError(RuntimeError):
    pass


def _try_acquire(pool: ConnectionPool, name: str, owner: str, ttl: int) -> bool:
    with pool.write_lock():
        conn = pool.connection()
        stamp = now()
        # Clear an expired holder first.
        conn.execute("DELETE FROM locks WHERE name=? AND expires_at <= ?",
                     (name, stamp.isoformat()))
        try:
            conn.execute(
                "INSERT INTO locks(name, owner, acquired_at, expires_at) VALUES (?,?,?,?)",
                (name, owner, stamp.isoformat(),
                 (stamp + timedelta(seconds=ttl)).isoformat()),
            )
            conn.commit()
            return True
        except Exception:  # noqa: BLE001 — PK conflict means held
            conn.rollback()
            return False


@contextmanager
def advisory_lock(pool: ConnectionPool, name: str, ttl: int = 60,
                  owner: str = "", wait: bool = True, poll: float = 0.1,
                  timeout: float = 30.0):
    deadline = None
    acquired = _try_acquire(pool, name, owner, ttl)
    waited = 0.0
    while not acquired:
        if not wait:
            raise LockHeldError(f"lock '{name}' is held")
        if waited >= timeout:
            raise LockHeldError(f"timed out waiting for lock '{name}'")
        time.sleep(poll)
        waited += poll
        acquired = _try_acquire(pool, name, owner, ttl)
    try:
        yield
    finally:
        with pool.write_lock():
            conn = pool.connection()
            conn.execute("DELETE FROM locks WHERE name=? AND owner=?", (name, owner))
            conn.commit()
```

Note: `time.sleep`/`time.time` are used (not `Date.now`); fine in app code. The `wait=False` tests never sleep.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_locks.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 5: Parquet store

**Files:**

- Create: `utils/storage/parquet_store.py`
- Test: `tests/utils/storage/test_parquet_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_parquet_store.py
import pandas as pd
from utils.storage.parquet_store import write_parquet, append_parquet, read_parquet


def test_write_then_read_roundtrip(tmp_path):
    path = str(tmp_path / "d.parquet")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_parquet(df, path)
    out = read_parquet(path)
    assert set(out["a"]) == {1, 2}


def test_append_dedupes_on_key(tmp_path):
    path = str(tmp_path / "d.parquet")
    write_parquet(pd.DataFrame({"k": [1], "v": ["old"]}), path)
    append_parquet(pd.DataFrame({"k": [1], "v": ["new"]}), path, dedupe_key=["k"])
    out = read_parquet(path)
    assert len(out) == 1 and out.iloc[0]["v"] == "new"


def test_year_partition_write(tmp_path):
    path = str(tmp_path / "part")  # directory dataset
    df = pd.DataFrame({"race_date": ["2025-01-01", "2026-01-01"], "v": [1, 2]})
    df["year"] = [2025, 2026]
    write_parquet(df, path, partition_by="year")
    out = read_parquet(path)
    assert set(out["year"]) == {2025, 2026}


def test_read_missing_returns_empty(tmp_path):
    out = read_parquet(str(tmp_path / "nope.parquet"))
    assert out.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_parquet_store.py -v`
Expected: FAIL — `No module named 'utils.storage.parquet_store'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/parquet_store.py
"""Centralized Parquet read-merge-write + year-partition helpers."""
import os
import shutil

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def read_parquet(path: str, filters=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.isdir(path) and not os.listdir(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, engine="pyarrow", filters=filters)
    except Exception as exc:  # noqa: BLE001
        logger.warning("read_parquet failed for %s (%s)", path, exc)
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, path: str, partition_by=None,
                  dedupe_key=None, columns=None) -> None:
    if df is None or df.empty:
        logger.debug("write_parquet: empty df, skipping %s", path)
        return
    df = df.copy()
    if dedupe_key:
        df = df.drop_duplicates(subset=dedupe_key, keep="last")
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[columns]
    if partition_by:
        # partition_cols append stale files — clear the dir first.
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False,
                      partition_cols=[partition_by] if isinstance(partition_by, str)
                      else list(partition_by))
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        df.to_parquet(path, engine="pyarrow", index=False)


def append_parquet(df: pd.DataFrame, path: str, partition_by=None,
                   dedupe_key=None, columns=None) -> None:
    """Read-merge-write: concat existing + new, dedupe, write back."""
    if df is None or df.empty:
        return
    existing = read_parquet(path)
    merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    write_parquet(merged, path, partition_by=partition_by,
                  dedupe_key=dedupe_key, columns=columns)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_parquet_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 6: Storage facade + dataset registry

**Files:**

- Create: `utils/storage/__init__.py`
- Test: `tests/utils/storage/test_facade.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/utils/storage/test_facade.py
import pandas as pd
import utils.storage as storage_mod
from utils.storage import Storage, get_storage


def test_facade_run_and_parquet(tmp_path, monkeypatch):
    db = str(tmp_path / "races.db")
    s = Storage(db_path=db, dataset_paths={"live_odds": str(tmp_path / "lo.parquet")})
    rid = s.start_run("boylesports", "live_odds")
    s.finish_run(rid, "ok", row_count=3)
    s.append_parquet(pd.DataFrame({"k": [1], "v": ["a"]}), "live_odds", dedupe_key=["k"])
    assert not s.read_parquet("live_odds").empty
    s.close()


def test_get_storage_is_singleton(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_mod, "_SINGLETON", None)
    monkeypatch.setattr(storage_mod, "DEFAULT_DB_PATH", str(tmp_path / "races.db"))
    a = get_storage()
    b = get_storage()
    assert a is b
    a.close()
    monkeypatch.setattr(storage_mod, "_SINGLETON", None)


def test_lock_contextmanager(tmp_path):
    s = Storage(db_path=str(tmp_path / "races.db"))
    with s.lock("dataset:live_odds", ttl=30, owner="t"):
        pass  # acquire + release without error
    s.close()


def test_unknown_dataset_raises(tmp_path):
    s = Storage(db_path=str(tmp_path / "races.db"))
    try:
        s.read_parquet("does_not_exist")
        assert False, "expected StorageError"
    except storage_mod.StorageError:
        pass
    s.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/utils/storage/test_facade.py -v`
Expected: FAIL — `cannot import name 'Storage'`

- [ ] **Step 3: Write minimal implementation**

```python
# utils/storage/__init__.py
"""Unified storage facade: SQLite live metadata + Parquet historical/feature data."""
import os
from contextlib import contextmanager

from utils.logger import get_logger
from utils.storage.pool import ConnectionPool
from utils.storage.migrations import apply_migrations
from utils.storage.sqlite_store import SqliteStore
from utils.storage import parquet_store
from utils.storage.locks import advisory_lock

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "races.db")

# Logical dataset name -> path (relative to project root unless absolute).
DATASETS = {
    "live_odds": os.path.join(_PROJECT_ROOT, "data", "live_odds.parquet"),
    "betsp": os.path.join(_PROJECT_ROOT, "data", "historical", "betsp.parquet"),
    "timeform": os.path.join(_PROJECT_ROOT, "data", "historical", "timeform.parquet"),
    "unified": os.path.join(_PROJECT_ROOT, "data", "unified_races.parquet"),
    "training": os.path.join(_PROJECT_ROOT, "data", "features", "training.parquet"),
}

_PARTITIONED = {"betsp", "timeform", "unified"}


class StorageError(RuntimeError):
    pass


class Storage:
    def __init__(self, db_path: str = DEFAULT_DB_PATH, dataset_paths=None):
        self._pool = ConnectionPool(db_path)
        apply_migrations(self._pool)
        self._sqlite = SqliteStore(self._pool)
        self._datasets = dict(DATASETS)
        if dataset_paths:
            self._datasets.update(dataset_paths)

    # --- SQLite delegation ---
    def start_run(self, source, dataset):
        return self._sqlite.start_run(source, dataset)

    def finish_run(self, run_id, status, row_count=None, error=None):
        self._sqlite.finish_run(run_id, status, row_count, error)

    def upsert_races(self, rows):
        self._sqlite.upsert_races(rows)

    def get_live_races(self, venue=None, status="open"):
        return self._sqlite.get_live_races(venue, status)

    def upsert_odds(self, rows):
        self._sqlite.upsert_odds(rows)

    def query_odds(self, venue=None, horse_id=None):
        return self._sqlite.query_odds(venue, horse_id)

    def cache_fresh(self, key):
        return self._sqlite.cache_fresh(key)

    def touch_cache(self, key, ttl_seconds, source=None):
        self._sqlite.touch_cache(key, ttl_seconds, source)

    @contextmanager
    def lock(self, name, ttl=60, owner="", wait=True):
        with advisory_lock(self._pool, name, ttl=ttl, owner=owner or str(os.getpid()),
                           wait=wait):
            yield

    # --- Parquet delegation ---
    def _resolve(self, dataset):
        path = self._datasets.get(dataset)
        if path is None:
            raise StorageError(f"unknown dataset '{dataset}'")
        return path

    def _partition_for(self, dataset):
        return "year" if dataset in _PARTITIONED else None

    def write_parquet(self, df, dataset, partition_by=None, dedupe_key=None, columns=None):
        path = self._resolve(dataset)
        parquet_store.write_parquet(
            df, path, partition_by=partition_by or self._partition_for(dataset),
            dedupe_key=dedupe_key, columns=columns)

    def append_parquet(self, df, dataset, partition_by=None, dedupe_key=None, columns=None):
        path = self._resolve(dataset)
        with self.lock(f"dataset:{dataset}", ttl=120):
            parquet_store.append_parquet(
                df, path, partition_by=partition_by or self._partition_for(dataset),
                dedupe_key=dedupe_key, columns=columns)

    def read_parquet(self, dataset, filters=None):
        return parquet_store.read_parquet(self._resolve(dataset), filters=filters)

    def close(self):
        self._pool.close_all()


_SINGLETON = None


def get_storage() -> Storage:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Storage(DEFAULT_DB_PATH)
    return _SINGLETON
```

Note: the facade's `read_parquet`/`append_parquet` accept a **logical dataset name**, while the lower-level `parquet_store` functions take a raw **path**. The facade test for "unknown dataset" exercises `_resolve` raising `StorageError`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/utils/storage/test_facade.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 7: Migrate `timeform/writer.py` onto the parquet store

**Files:**

- Modify: `scraper/timeform/writer.py`
- Regression test (unchanged): `tests/scraper/timeform/test_writer.py`

- [ ] **Step 1: Read the existing test to confirm the contract**

Run: `pytest tests/scraper/timeform/test_writer.py -v`
Expected: currently PASS — this is the contract to preserve.

- [ ] **Step 2: Refactor `write()` to delegate**

Replace the body of `write()` (keeping `FINAL_COLUMNS`, `_DEDUPE_KEY`, `_add_year`, and the signature `write(df, path)`) with delegation to the shared store:

```python
# scraper/timeform/writer.py  — replace the write() function only
from utils.storage import parquet_store

def write(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        logger.debug("timeform writer: empty df, nothing to write")
        return
    df = _add_year(df)
    parquet_store.append_parquet(
        df, path, partition_by="year", dedupe_key=_DEDUPE_KEY, columns=FINAL_COLUMNS)
    logger.debug("timeform writer: wrote %d rows", len(df))
```

Remove the now-unused `import os` / `import shutil` only if nothing else in the file uses them (check first; leave them if used elsewhere).

- [ ] **Step 3: Run the regression test**

Run: `pytest tests/scraper/timeform/test_writer.py -v`
Expected: PASS (unchanged behavior). If a test asserted on intermediate `shutil`/`os` calls, update it to assert on output rows instead — but only if it fails; do not touch passing tests.

- [ ] **Step 4: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 8: Migrate `betsp/writer.py` and boylesports `_merge_parquet`

**Files:**

- Modify: `scraper/betsp/writer.py`
- Modify: `scraper/boylesports.py` (the `_merge_parquet` function, ~line 442)
- Regression tests (unchanged): `tests/scraper/betsp/test_writer.py`, `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Confirm contracts**

Run: `pytest tests/scraper/betsp/test_writer.py tests/scraper/test_boylesports.py -v`
Expected: currently PASS.

- [ ] **Step 2: Refactor `betsp/writer.py`**

Read `scraper/betsp/writer.py` first to capture its `FINAL_COLUMNS`/dedupe-key names, then delegate its write to `parquet_store.append_parquet(df, path, partition_by="year", dedupe_key=<its key>, columns=<its columns>)`, preserving the public function name/signature. (betsp dedupe key per spec/overview: `(race_date, venue, horse_id, market_type)`.)

- [ ] **Step 3: Refactor boylesports `_merge_parquet`**

Replace the body of `_merge_parquet(df, path=_PARQUET_PATH)` with:

```python
def _merge_parquet(df: pd.DataFrame, path: str = _PARQUET_PATH) -> None:
    """Read-merge-write the live-odds parquet (now via the shared store)."""
    from utils.storage import parquet_store
    parquet_store.append_parquet(df, path, columns=_PARQUET_COLUMNS)
```

(Live odds parquet is a single file, not partitioned; boylesports merges by source via its own row construction, so no dedupe_key change — match current behavior. If the current `_merge_parquet` dropped the source's old rows before concat, replicate that by filtering `df`/existing on `source` BEFORE calling `append_parquet`; read the current implementation to confirm whether it deduped and preserve exactly.)

- [ ] **Step 4: Run regression tests**

Run: `pytest tests/scraper/betsp/test_writer.py tests/scraper/test_boylesports.py -v`
Expected: PASS unchanged.

- [ ] **Step 5: Checkpoint** — Run `pytest -q`. Expected: full suite green.

---

### Task 9: Migration CLI smoke + final verification

**Files:**

- Test: `tests/utils/storage/test_cli.py`

- [ ] **Step 1: Write the CLI test**

```python
# tests/utils/storage/test_cli.py
import utils.storage.migrations as m
from utils.storage.pool import ConnectionPool


def test_cli_migrate_and_version(tmp_path, monkeypatch, capsys):
    import utils.storage as storage_mod
    monkeypatch.setattr(storage_mod, "DEFAULT_DB_PATH", str(tmp_path / "races.db"))
    assert m._main(["migrate"]) == 0
    out = capsys.readouterr().out
    assert "migrated to version" in out
    assert m._main(["version"]) == 0
    assert "target=" in capsys.readouterr().out
```

- [ ] **Step 2: Run it**

Run: `pytest tests/utils/storage/test_cli.py -v`
Expected: PASS. If `_main` imports `DEFAULT_DB_PATH` at call-time (it does, inside the function), the monkeypatch takes effect.

- [ ] **Step 3: Manual CLI smoke**

Run: `python -m utils.storage.migrations version`
Expected: prints `current=1 target=1` (after first migrate) — creates `data/races.db` if absent.

- [ ] **Step 4: Final full-suite verification**

Run: `pytest -q`
Expected: all tests green (191 prior + ~21 new storage tests). Record the new total in the project memory.

---

## Self-Review Notes

- **Spec coverage:** module structure (Task 1–6), five SQLite tables (Task 2 migration), connection pool + thread-safety (Task 1), migrations + `user_version` + CLI (Tasks 2, 9), advisory locks incl. expiry (Task 4), parquet read-merge-write + partition (Task 5), facade + dataset registry (Task 6), writer migration with green regression tests (Tasks 7–8), error handling via `StorageError`/empty-df/empty-read (Tasks 5–6). All spec sections map to a task.
- **Type consistency:** `ConnectionPool.connection()`/`write_lock()`/`close_all()`, `SqliteStore` method names, `advisory_lock(pool, name, ttl, owner, wait)`, `parquet_store.{read,write,append}_parquet(... path ...)`, and facade dataset-name methods are used consistently across tasks.
- **Known integration risk:** Tasks 7–8 must preserve exact existing writer behavior; the rule is "regression tests stay green, don't edit passing tests." Read each target file before editing to confirm its real column/dedupe constants rather than assuming.
