"""Disk-backed JSON cache with per-entry TTL expiration.

Public API
----------
Cache(cache_dir, default_ttl)
    .get(key)             -> Any | None   — None if missing or expired
    .set(key, value, ttl) -> None         — write; ttl overrides default
    .delete(key)          -> None         — remove entry (silent if absent)
    .is_fresh(key, ttl)   -> bool         — True iff present and unexpired
    .get_stale(key)       -> Any | None   — value regardless of expiry
    .get_meta(key)        -> dict | None  — {cached_at, expires_at, ttl}
    .clear()              -> int          — delete all entries, return count
    .keys()               -> list[str]    — all keys including expired

get_cache() -> Cache  — module-level singleton (data/cache/, TTL from config)

File format
-----------
Each entry is stored as ``{cache_dir}/{key}.json``::

    {
      "key":        "<str>",
      "cached_at":  "<ISO-8601>",
      "expires_at": "<ISO-8601>",
      "ttl":        <int seconds>,
      "data":       <any JSON-serialisable value>
    }

Legacy files (written before this module) store data directly at the top
level with a ``fetched_at`` timestamp.  ``is_fresh`` and ``get_stale`` handle
both formats transparently so existing scraper caches need no migration.
"""
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from utils.logger import get_logger

_log = get_logger(__name__)

_BASE = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE_DIR = _BASE / "data" / "cache"
_DEFAULT_TTL = 3600

_instance: Optional["Cache"] = None
_instance_lock = threading.Lock()


class Cache:
    """Thread-safe, disk-backed JSON cache with per-entry TTL."""

    def __init__(
        self,
        cache_dir: Optional[Any] = None,
        default_ttl: int = _DEFAULT_TTL,
    ) -> None:
        self._dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self._default_ttl = default_ttl
        self._lock = threading.Lock()

    # ── private helpers ───────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._dir / f"{safe}.json"

    @staticmethod
    def _is_envelope(entry: dict) -> bool:
        """True when entry uses the new {data, expires_at} envelope format."""
        return "data" in entry and "expires_at" in entry

    def _read_raw(self, key: str) -> Optional[dict]:
        path = self._path(key)
        try:
            with self._lock:
                if not path.exists():
                    return None
                with path.open(encoding="utf-8") as fh:
                    return json.load(fh)
        except Exception as exc:
            _log.warning("cache: read error for %r: %s", key, exc)
            return None

    def _expiry(self, entry: dict, ttl: Optional[int] = None) -> Optional[datetime]:
        """Return expiry as UTC-aware datetime, or None if undetermined."""
        if self._is_envelope(entry):
            try:
                return datetime.fromisoformat(entry["expires_at"]).astimezone(timezone.utc)
            except (ValueError, KeyError):
                return None
        # Legacy format: derive expiry from fetched_at + ttl
        raw_ts = entry.get("fetched_at")
        if not raw_ts:
            return None
        try:
            fetched = datetime.fromisoformat(raw_ts).astimezone(timezone.utc)
            effective = ttl if ttl is not None else self._default_ttl
            return fetched + timedelta(seconds=effective)
        except (ValueError, TypeError):
            return None

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value if present and unexpired; else None."""
        entry = self._read_raw(key)
        if entry is None:
            return None
        expiry = self._expiry(entry)
        if expiry is None or datetime.now(tz=timezone.utc) > expiry:
            _log.debug("cache: miss (expired) — %r", key)
            return None
        return entry["data"] if self._is_envelope(entry) else entry

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Persist value with TTL-based expiry.  Atomic write via .tmp → replace."""
        effective_ttl = ttl if ttl is not None else self._default_ttl
        now = datetime.now(tz=timezone.utc)
        entry = {
            "key": key,
            "cached_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=effective_ttl)).isoformat(),
            "ttl": effective_ttl,
            "data": value,
        }
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(entry, fh, indent=2, default=str)
                tmp.replace(path)
            _log.debug("cache: set %r (ttl=%ds)", key, effective_ttl)
        except Exception as exc:
            _log.warning("cache: write error for %r: %s", key, exc)
            try:
                with self._lock:
                    if tmp.exists():
                        tmp.unlink()
            except OSError:
                pass

    def delete(self, key: str) -> None:
        """Remove a cache entry.  Silently ignores missing keys."""
        path = self._path(key)
        try:
            with self._lock:
                if path.exists():
                    path.unlink()
            _log.debug("cache: deleted %r", key)
        except Exception as exc:
            _log.warning("cache: delete error for %r: %s", key, exc)

    def is_fresh(self, key: str, ttl: Optional[int] = None) -> bool:
        """True iff an entry exists and has not yet expired.

        ``ttl`` overrides the instance default when checking legacy-format files
        that store ``fetched_at`` instead of ``expires_at``.
        """
        entry = self._read_raw(key)
        if entry is None:
            return False
        expiry = self._expiry(entry, ttl=ttl)
        if expiry is None:
            return False
        return datetime.now(tz=timezone.utc) <= expiry

    def get_stale(self, key: str) -> Optional[Any]:
        """Return value regardless of expiry.  Useful as fallback after live failure."""
        entry = self._read_raw(key)
        if entry is None:
            return None
        return entry["data"] if self._is_envelope(entry) else entry

    def get_meta(self, key: str) -> Optional[dict]:
        """Return entry metadata without the data payload.

        Returns ``None`` if the entry is absent or unreadable.
        """
        entry = self._read_raw(key)
        if entry is None:
            return None
        if self._is_envelope(entry):
            return {k: entry[k] for k in ("key", "cached_at", "expires_at", "ttl") if k in entry}
        fetched = entry.get("fetched_at")
        return {"fetched_at": fetched, "legacy": True} if fetched else None

    def clear(self) -> int:
        """Delete all cache entries.  Returns the number removed."""
        removed = 0
        try:
            with self._lock:
                for p in sorted(self._dir.glob("*.json")):
                    if p.name == ".gitkeep":
                        continue
                    p.unlink()
                    removed += 1
        except Exception as exc:
            _log.warning("cache: clear error: %s", exc)
        return removed

    def keys(self) -> list:
        """List all cache keys (file stems), including expired entries."""
        try:
            return [p.stem for p in self._dir.glob("*.json") if p.name != ".gitkeep"]
        except Exception:
            return []


# ── module-level singleton ─────────────────────────────────────────────────────

def get_cache(
    cache_dir: Optional[Any] = None,
    default_ttl: Optional[int] = None,
) -> Cache:
    """Return the shared Cache singleton backed by ``data/cache/``.

    First-call arguments win; subsequent calls with different values are ignored.
    Use ``Cache(...)`` directly when you need an ad-hoc instance at a specific path.
    TTL defaults to ``scrape_interval`` from config.yaml (fallback: 3600 s).
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                ttl = default_ttl
                if ttl is None:
                    try:
                        from utils.config_loader import get_config
                        ttl = int(get_config().get("scrape_interval", _DEFAULT_TTL))
                    except Exception:
                        ttl = _DEFAULT_TTL
                _instance = Cache(
                    cache_dir=cache_dir if cache_dir is not None else _DEFAULT_CACHE_DIR,
                    default_ttl=ttl,
                )
    return _instance
