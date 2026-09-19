import pandas as pd

import utils.storage as storage_mod
from utils.storage import Storage, get_storage


def test_facade_run_and_parquet(tmp_path):
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
        pass
    s.close()


def test_unknown_dataset_raises(tmp_path):
    s = Storage(db_path=str(tmp_path / "races.db"))
    try:
        s.read_parquet("does_not_exist")
        assert False, "expected StorageError"
    except storage_mod.StorageError:
        pass
    s.close()
