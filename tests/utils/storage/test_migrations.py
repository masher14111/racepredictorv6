from utils.storage.pool import ConnectionPool
from utils.storage.migrations import MIGRATIONS, apply_migrations, target_version, current_version


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
    apply_migrations(pool)
    assert current_version(pool.connection()) == v1
    pool.close_all()


# ── Stage 20 (migration 8): race_key / provenance columns ──────────────────

def test_migration_8_adds_race_key_and_provenance_columns(tmp_path):
    pool = ConnectionPool(str(tmp_path / "m.db"))
    apply_migrations(pool)
    conn = pool.connection()
    snap_cols = {r[1] for r in conn.execute("PRAGMA table_info(odds_snapshots)").fetchall()}
    tix_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_tickets)").fetchall()}
    assert "race_key" in snap_cols
    assert {"race_key", "model_content_hash", "feature_schema_version",
            "config_hash", "prediction_cycle_id", "provenance_complete"} <= tix_cols
    pool.close_all()


def test_migration_8_backfills_race_key_on_pre_existing_tickets(tmp_path):
    """paper_tickets is NOT append-only, so a row written under migration 7
    must have its race_key computed retroactively from its own stored
    (venue, race_time) once migration 8 runs."""
    db = str(tmp_path / "m.db")
    pool = ConnectionPool(db)
    conn = pool.connection()
    for ver, fn in MIGRATIONS:
        if ver >= 8:
            break
        fn(conn)
        conn.execute(f"PRAGMA user_version={ver}")
        conn.commit()
    assert current_version(conn) == 7

    conn.execute(
        "INSERT INTO paper_tickets (ticket_id, issued_at, race_uid, venue, "
        "race_time, horse_key, market_type, bet_type, validation_state, "
        "decision, paper_only) VALUES (?, ?, ?, ?, ?, ?, 'WIN', 'win', 'x', "
        "'PASS', 1)",
        ("t1", "2026-07-27T12:00:00+00:00", "2026-07-27T15:40:00+00:00",
         "Naas", "2026-07-27T15:40:00+00:00", "h1"),
    )
    conn.commit()

    apply_migrations(pool)
    row = conn.execute(
        "SELECT race_key FROM paper_tickets WHERE ticket_id = 't1'"
    ).fetchone()
    assert row[0] == "naas|2026-07-27T15:40"
    pool.close_all()


def test_migration_8_never_updates_the_append_only_snapshots_table(tmp_path):
    """odds_snapshots forbids UPDATE (migration 4's RAISE(ABORT) triggers) —
    migration 8 must add its column without ever attempting to backfill a
    pre-existing snapshot row; the column stays honestly NULL for those.
    A prior version of this migration crashed here on a populated store."""
    db = str(tmp_path / "m.db")
    pool = ConnectionPool(db)
    conn = pool.connection()
    for ver, fn in MIGRATIONS:
        if ver >= 8:
            break
        fn(conn)
        conn.execute(f"PRAGMA user_version={ver}")
        conn.commit()

    conn.execute(
        "INSERT INTO odds_snapshots (race_uid, race_time, venue, horse_key, "
        "bookmaker, market_type, odds_decimal, fetched_at, ingested_at) "
        "VALUES (?, ?, ?, ?, ?, 'WIN', 4.0, ?, ?)",
        ("2026-07-27T15:40:00+00:00", "2026-07-27T15:40:00+00:00", "Naas",
         "h1", "boylesports",
         "2026-07-27T12:00:00.000000+00:00", "2026-07-27T12:00:00.000000+00:00"),
    )
    conn.commit()

    apply_migrations(pool)  # must not raise "append-only" here
    assert current_version(conn) == target_version()
    row = conn.execute(
        "SELECT race_key FROM odds_snapshots WHERE race_uid = ?",
        ("2026-07-27T15:40:00+00:00",),
    ).fetchone()
    assert row[0] is None
    pool.close_all()


def test_migration_8_resumes_after_a_partial_prior_attempt(tmp_path):
    """Simulates this process crashing after adding one column but before the
    rest of migration 8 (and the version bump) completed — restart-safety
    means the next call must not raise 'duplicate column'."""
    db = str(tmp_path / "m.db")
    pool = ConnectionPool(db)
    conn = pool.connection()
    for ver, fn in MIGRATIONS:
        if ver >= 8:
            break
        fn(conn)
        conn.execute(f"PRAGMA user_version={ver}")
        conn.commit()
    conn.execute("ALTER TABLE odds_snapshots ADD COLUMN race_key TEXT")
    conn.commit()
    assert current_version(conn) == 7  # crashed before the version bump

    apply_migrations(pool)  # resume: must not raise "duplicate column"
    assert current_version(conn) == target_version()
    pool.close_all()
