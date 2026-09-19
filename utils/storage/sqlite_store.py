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
            sql += " AND status=?"
            params.append(status)
        if venue is not None:
            sql += " AND venue=?"
            params.append(venue)
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
        if venue is not None:
            sql = ("SELECT o.* FROM live_odds o JOIN race_catalog c ON o.race_id=c.race_id "
                   "WHERE c.venue=?")
            params = [venue]
            if horse_id is not None:
                sql += " AND o.horse_id=?"
                params.append(horse_id)
        else:
            sql = "SELECT * FROM live_odds WHERE 1=1"
            params = []
            if horse_id is not None:
                sql += " AND horse_id=?"
                params.append(horse_id)
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
