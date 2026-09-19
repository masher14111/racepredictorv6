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
