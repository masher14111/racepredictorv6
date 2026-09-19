"""Disk-backed per-source health telemetry for live-odds scrapers.

Tracks operational status (success rate, last fetch, rejection reasons — not
per-proxy stats, see utils.proxy_manager for that) per bookmaker source. It is
disk-backed rather than an in-memory singleton because `scripts/refresh.py`
runs as a separate process from the Streamlit UI; only a file both can read
makes a cron-triggered refresh visible on the dashboard.

Public API
----------
record_success(source, rows=0, races=0)    -> None
record_failure(source, reason="other")     -> None
record_unavailable(source, reason="other") -> None  # proxy/gateway gone
set_proxy_reachable(source, reachable)     -> None
get_health(source=None) -> dict | dict[str, dict]   # includes derived age_seconds

File format
-----------
``data/source_health.json``::

    {
      "<source>": {
        "status": "ok" | "failing" | "unavailable",
        "last_attempt_at": "<ISO-8601>",
        "last_success_at": "<ISO-8601> | null",
        "total_successes": <int>,
        "total_failures": <int>,
        "success_rate": <float 0..1 | null>,
        "last_row_count": <int | null>,
        "last_race_count": <int | null>,
        "last_rejection_reason": "<str | null>",
        "proxy_reachable": <bool | null>
      },
      ...
    }
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = Path(__file__).resolve().parents[1]
_PATH = _BASE / "data" / "source_health.json"
_lock = threading.Lock()

STATUS_OK = "ok"
STATUS_FAILING = "failing"
STATUS_UNAVAILABLE = "unavailable"

_DEFAULT_RECORD = {
    "status": None,
    "last_attempt_at": None,
    "last_success_at": None,
    "total_successes": 0,
    "total_failures": 0,
    "success_rate": None,
    "last_row_count": None,
    "last_race_count": None,
    "last_rejection_reason": None,
    "proxy_reachable": None,
}


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _read_all() -> dict:
    try:
        with _lock:
            if not _PATH.exists():
                return {}
            with _PATH.open(encoding="utf-8") as fh:
                return json.load(fh)
    except Exception as exc:
        logger.warning("source_health: read error: %s", exc)
        return {}


def _write_all(data: dict) -> None:
    tmp = _PATH.with_suffix(".tmp")
    try:
        with _lock:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
            tmp.replace(_PATH)
    except Exception as exc:
        logger.warning("source_health: write error: %s", exc)
        try:
            with _lock:
                if tmp.exists():
                    tmp.unlink()
        except OSError:
            pass


def _record_for(data: dict, source: str) -> dict:
    return dict(_DEFAULT_RECORD, **data.get(source, {}))


def record_success(source: str, rows: int = 0, races: int = 0) -> None:
    data = _read_all()
    rec = _record_for(data, source)
    now = _now_iso()
    rec["status"] = STATUS_OK
    rec["last_attempt_at"] = now
    rec["last_success_at"] = now
    rec["total_successes"] = int(rec["total_successes"]) + 1
    total = rec["total_successes"] + int(rec["total_failures"])
    rec["success_rate"] = rec["total_successes"] / total if total else None
    rec["last_row_count"] = rows
    rec["last_race_count"] = races
    rec["last_rejection_reason"] = None
    data[source] = rec
    _write_all(data)


def _record_failure_like(source: str, reason: str, status: str) -> None:
    data = _read_all()
    rec = _record_for(data, source)
    rec["status"] = status
    rec["last_attempt_at"] = _now_iso()
    rec["total_failures"] = int(rec["total_failures"]) + 1
    total = int(rec["total_successes"]) + rec["total_failures"]
    rec["success_rate"] = rec["total_successes"] / total if total else None
    rec["last_rejection_reason"] = reason
    data[source] = rec
    _write_all(data)


def record_failure(source: str, reason: str = "other") -> None:
    _record_failure_like(source, reason, STATUS_FAILING)


def record_unavailable(source: str, reason: str = "other") -> None:
    """Record a hard proxy/gateway-unavailable failure (never a direct fallback)."""
    _record_failure_like(source, reason, STATUS_UNAVAILABLE)


def set_proxy_reachable(source: str, reachable: bool) -> None:
    data = _read_all()
    rec = _record_for(data, source)
    rec["proxy_reachable"] = reachable
    data[source] = rec
    _write_all(data)


def _with_age(rec: dict) -> dict:
    rec = dict(rec)
    last_success = rec.get("last_success_at")
    age = None
    if last_success:
        try:
            dt = datetime.fromisoformat(last_success)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age = (datetime.now(tz=timezone.utc) - dt).total_seconds()
        except (ValueError, TypeError):
            age = None
    rec["age_seconds"] = age
    return rec


def get_health(source: Optional[str] = None) -> Union[dict, dict[str, dict]]:
    """Return one source's health record, or all sources keyed by name.

    Each record includes a derived ``age_seconds`` (time since last success),
    computed fresh rather than stored so it's always accurate at read time.
    """
    data = _read_all()
    if source is not None:
        return _with_age(_record_for(data, source))
    return {name: _with_age(dict(_DEFAULT_RECORD, **rec)) for name, rec in data.items()}
