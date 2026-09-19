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
    t.start()
    t.join()
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
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert pool.connection().execute("SELECT COUNT(*) FROM c").fetchone()[0] == 200
    pool.close_all()
