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

# Logical dataset name -> path.
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
