"""Tests for utils/cache.py — Cache class and get_cache() singleton."""
import json
import threading
from datetime import datetime, timedelta, timezone



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_legacy(path, fetched_at_iso: str, data: dict = None):
    """Write a legacy-format cache file (fetched_at at top level, no envelope)."""
    payload = {"fetched_at": fetched_at_iso, **(data or {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _write_envelope(path, expires_at_iso: str, data: dict = None):
    """Write a new-format envelope cache file directly."""
    entry = {
        "key": path.stem,
        "cached_at": datetime.now(tz=timezone.utc).isoformat(),
        "expires_at": expires_at_iso,
        "ttl": 3600,
        "data": data or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(entry, fh)


# ---------------------------------------------------------------------------
# Cache.get — fresh and expired
# ---------------------------------------------------------------------------

def test_get_returns_none_when_key_missing(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    assert c.get("nope") is None


def test_get_returns_value_when_fresh(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path, default_ttl=3600)
    c.set("k", {"hello": "world"})
    assert c.get("k") == {"hello": "world"}


def test_get_returns_none_when_expired(tmp_path):
    from utils.cache import Cache
    future = (datetime.now(tz=timezone.utc) - timedelta(seconds=1)).isoformat()
    _write_envelope(tmp_path / "expired.json", expires_at_iso=future, data={"x": 1})
    c = Cache(cache_dir=tmp_path)
    assert c.get("expired") is None


def test_get_handles_corrupt_file(tmp_path):
    from utils.cache import Cache
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    c = Cache(cache_dir=tmp_path)
    assert c.get("bad") is None


# ---------------------------------------------------------------------------
# Cache.set — write behaviour
# ---------------------------------------------------------------------------

def test_set_creates_json_file(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path, default_ttl=60)
    c.set("mykey", [1, 2, 3])
    assert (tmp_path / "mykey.json").exists()


def test_set_creates_parent_dirs(tmp_path):
    from utils.cache import Cache
    deep = tmp_path / "a" / "b" / "c"
    c = Cache(cache_dir=deep, default_ttl=60)
    c.set("z", "value")
    assert (deep / "z.json").exists()


def test_set_per_call_ttl_overrides_default(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path, default_ttl=3600)
    c.set("short", "v", ttl=1)
    raw = json.loads((tmp_path / "short.json").read_text())
    assert raw["ttl"] == 1


def test_set_no_leftover_tmp_file(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.set("t", {"a": 1})
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------------------
# Cache.delete
# ---------------------------------------------------------------------------

def test_delete_removes_file(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.set("d", 42)
    c.delete("d")
    assert not (tmp_path / "d.json").exists()


def test_delete_missing_key_is_silent(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.delete("ghost")  # must not raise


# ---------------------------------------------------------------------------
# Cache.is_fresh — new format
# ---------------------------------------------------------------------------

def test_is_fresh_true_within_ttl(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path, default_ttl=3600)
    c.set("f", {})
    assert c.is_fresh("f") is True


def test_is_fresh_false_when_missing(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    assert c.is_fresh("absent") is False


def test_is_fresh_false_when_expired_envelope(tmp_path):
    from utils.cache import Cache
    past = (datetime.now(tz=timezone.utc) - timedelta(seconds=1)).isoformat()
    _write_envelope(tmp_path / "old.json", expires_at_iso=past)
    c = Cache(cache_dir=tmp_path)
    assert c.is_fresh("old") is False


# ---------------------------------------------------------------------------
# Cache.is_fresh — legacy format (fetched_at only)
# ---------------------------------------------------------------------------

def test_is_fresh_legacy_true_within_ttl(tmp_path):
    from utils.cache import Cache
    now_iso = datetime.now(tz=timezone.utc).isoformat()
    _write_legacy(tmp_path / "leg.json", fetched_at_iso=now_iso, data={"rows": []})
    c = Cache(cache_dir=tmp_path, default_ttl=3600)
    assert c.is_fresh("leg", ttl=3600) is True


def test_is_fresh_legacy_false_when_stale(tmp_path):
    from utils.cache import Cache
    old_iso = (datetime.now(tz=timezone.utc) - timedelta(seconds=5000)).isoformat()
    _write_legacy(tmp_path / "stale.json", fetched_at_iso=old_iso)
    c = Cache(cache_dir=tmp_path, default_ttl=3600)
    assert c.is_fresh("stale", ttl=3600) is False


def test_is_fresh_legacy_false_when_no_fetched_at(tmp_path):
    from utils.cache import Cache
    (tmp_path / "nots.json").write_text('{"rows": []}', encoding="utf-8")
    c = Cache(cache_dir=tmp_path)
    assert c.is_fresh("nots") is False


# ---------------------------------------------------------------------------
# Cache.get_stale
# ---------------------------------------------------------------------------

def test_get_stale_returns_value_after_expiry(tmp_path):
    from utils.cache import Cache
    past = (datetime.now(tz=timezone.utc) - timedelta(seconds=1)).isoformat()
    _write_envelope(tmp_path / "exp.json", expires_at_iso=past, data={"k": "v"})
    c = Cache(cache_dir=tmp_path)
    assert c.get_stale("exp") == {"k": "v"}


def test_get_stale_returns_legacy_dict_as_is(tmp_path):
    from utils.cache import Cache
    iso = datetime.now(tz=timezone.utc).isoformat()
    _write_legacy(tmp_path / "leg2.json", fetched_at_iso=iso, data={"rows": [1, 2]})
    c = Cache(cache_dir=tmp_path)
    result = c.get_stale("leg2")
    assert result["rows"] == [1, 2]


def test_get_stale_returns_none_when_missing(tmp_path):
    from utils.cache import Cache
    assert Cache(cache_dir=tmp_path).get_stale("x") is None


# ---------------------------------------------------------------------------
# Cache.get_meta
# ---------------------------------------------------------------------------

def test_get_meta_returns_envelope_fields(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path, default_ttl=60)
    c.set("m", {"payload": True})
    meta = c.get_meta("m")
    assert meta is not None
    assert "cached_at" in meta and "expires_at" in meta and "ttl" in meta
    assert "payload" not in meta


def test_get_meta_returns_none_when_missing(tmp_path):
    from utils.cache import Cache
    assert Cache(cache_dir=tmp_path).get_meta("missing") is None


def test_get_meta_legacy_returns_fetched_at(tmp_path):
    from utils.cache import Cache
    iso = datetime.now(tz=timezone.utc).isoformat()
    _write_legacy(tmp_path / "lm.json", fetched_at_iso=iso)
    meta = Cache(cache_dir=tmp_path).get_meta("lm")
    assert meta is not None
    assert meta.get("legacy") is True
    assert meta.get("fetched_at") == iso


# ---------------------------------------------------------------------------
# Cache.clear and Cache.keys
# ---------------------------------------------------------------------------

def test_clear_removes_all_entries(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.set("a", 1)
    c.set("b", 2)
    count = c.clear()
    assert count == 2
    assert c.keys() == []


def test_clear_preserves_gitkeep(tmp_path):
    from utils.cache import Cache
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    c = Cache(cache_dir=tmp_path)
    c.set("x", 0)
    c.clear()
    assert (tmp_path / ".gitkeep").exists()


def test_keys_lists_all_entries(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.set("alpha", 1)
    c.set("beta", 2)
    assert sorted(c.keys()) == ["alpha", "beta"]


def test_keys_includes_expired(tmp_path):
    from utils.cache import Cache
    past = (datetime.now(tz=timezone.utc) - timedelta(seconds=1)).isoformat()
    _write_envelope(tmp_path / "stale_key.json", expires_at_iso=past)
    assert "stale_key" in Cache(cache_dir=tmp_path).keys()


# ---------------------------------------------------------------------------
# Key sanitisation
# ---------------------------------------------------------------------------

def test_key_with_slashes_creates_safe_filename(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    c.set("a/b:c", {"ok": True})
    # Should write a file, not a subdirectory
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    assert files[0].name != "a/b:c.json"


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

def test_concurrent_writes_do_not_corrupt(tmp_path):
    from utils.cache import Cache
    c = Cache(cache_dir=tmp_path)
    errors = []

    def writer(i):
        try:
            c.set("shared", {"i": i})
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # File must be valid JSON after all writes
    raw = json.loads((tmp_path / "shared.json").read_text())
    assert "data" in raw


# ---------------------------------------------------------------------------
# get_cache singleton
# ---------------------------------------------------------------------------

def test_get_cache_returns_same_instance(monkeypatch):
    import utils.cache as cache_mod
    monkeypatch.setattr(cache_mod, "_instance", None)
    from utils.cache import get_cache, Cache
    c1 = get_cache()
    c2 = get_cache()
    assert c1 is c2
    assert isinstance(c1, Cache)


def test_get_cache_uses_default_cache_dir(monkeypatch):
    import utils.cache as cache_mod
    monkeypatch.setattr(cache_mod, "_instance", None)
    from utils.cache import get_cache, _DEFAULT_CACHE_DIR
    c = get_cache()
    assert c._dir == _DEFAULT_CACHE_DIR
