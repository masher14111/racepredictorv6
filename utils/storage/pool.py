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
