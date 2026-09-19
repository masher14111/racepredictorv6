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
                         "status": "resulted"}])
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
    ])
    rows = store.query_odds(horse_id="h1")
    assert len(rows) == 1 and rows[0]["odds_decimal"] == 4.5
    pool.close_all()


def test_cache_freshness(tmp_path):
    store, pool = _store(tmp_path)
    assert store.cache_fresh("paddy") is False
    store.touch_cache("paddy", ttl_seconds=3600, source="paddy_power")
    assert store.cache_fresh("paddy") is True
    pool.close_all()
