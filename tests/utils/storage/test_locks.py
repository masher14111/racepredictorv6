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
    conn = pool.connection()
    conn.execute("INSERT INTO locks(name, owner, acquired_at, expires_at) "
                 "VALUES ('x','dead','2000-01-01T00:00:00','2000-01-01T00:00:00')")
    conn.commit()
    with advisory_lock(pool, "x", ttl=60, owner="B", wait=False):
        assert pool.connection().execute(
            "SELECT owner FROM locks WHERE name='x'").fetchone()["owner"] == "B"
    pool.close_all()
