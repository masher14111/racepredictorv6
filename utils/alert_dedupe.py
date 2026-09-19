"""Remembers which alerts have already been sent, so a rebuild stays quiet.

``models.predictor`` re-scores every race in the cache on each run, including
races that have already gone off. Without this, any prediction run re-fires the
same top-pick and odds-drop alerts, which is why a rebuild produced a burst of
notifications about races that finished hours ago.

A fingerprint is the alert's *content*, not just its subject — an odds-drop
fingerprint carries the new price, so a horse that shortens again does notify,
while the same drop seen on five consecutive rebuilds notifies once. That keeps
genuine price movement (the thing actually worth a phone buzz) while dropping
the repeats.

State lives in ``data/cache/sent_alerts.json`` and self-prunes, so it never has
to be cleaned up by hand and a deleted file just means "alert once more".
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timedelta, timezone

from utils.logger import get_logger

logger = get_logger(__name__)

_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "sent_alerts.json"
)

# Long enough to cover a full race day of repeated rebuilds; short enough that
# the file stays small and tomorrow's identical price is allowed to alert again.
_TTL = timedelta(hours=12)

_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _read() -> dict:
    try:
        with open(_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _write(data: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_PATH), exist_ok=True)
        tmp = _PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, _PATH)
    except OSError as exc:  # noqa: BLE001 — never let bookkeeping break a run
        logger.warning("alert_dedupe: could not persist state (%s)", exc)


def _prune(data: dict) -> dict:
    cutoff = _now() - _TTL
    out = {}
    for key, stamp in data.items():
        try:
            if datetime.fromisoformat(stamp) >= cutoff:
                out[key] = stamp
        except (TypeError, ValueError):
            continue  # unparseable entry — drop it
    return out


def seen(fingerprint: str) -> bool:
    """True if this exact alert already went out; otherwise record it and return False.

    Check-and-mark is one locked operation so a repeated fingerprint inside a
    single run cannot send twice.
    """
    if not fingerprint:
        return False
    with _lock:
        data = _prune(_read())
        if fingerprint in data:
            return True
        data[fingerprint] = _now().isoformat()
        _write(data)
        return False


def reset() -> None:
    """Forget everything — test hook, and a manual "alert me again" escape hatch."""
    with _lock:
        _write({})


def fingerprint(kind: str, *parts) -> str:
    """Stable id for an alert. Include whatever must change to re-notify."""
    return "|".join([kind, *(str(p) for p in parts)])
