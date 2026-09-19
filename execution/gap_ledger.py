"""Gap accounting for the daily paper loop (Stage 6, requirement 5).

A day with no capture is a **gap**, not a day with zero qualifying bets — the
two look identical in a bet ledger but mean very different things about how
much the forward window has actually proven. Conflating them lets a machine
that was off for a month pass the same "0 bets that day" accounting as a day
the model legitimately declined every runner, which would silently inflate
the evidence base. This module keeps a day-by-day record of which is which.

The ledger only ever *appends or updates today's own entry* going forward;
:func:`reconcile_gaps` additionally back-annotates entries for **past** days
that have no entry at all, using the snapshot store's own row timestamps as
the evidence. That is bookkeeping about which days had capture, not a
snapshot or settlement — it never writes a price or a bet outcome, so it does
not conflict with the "no backfill, ever" rule in requirement 6, which is
about snapshot ``fetched_at`` values and settlement prices specifically.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_GAP_LEDGER_PATH = os.path.join("data", "execution", "gap_ledger.json")

CAPTURED = "captured"
GAP = "gap"

# Recognised gap causes. ``other`` covers anything a caller passes that is not
# one of these; it is still recorded, just not specially reasoned about here.
CAUSE_SCRAPER_OUTAGE = "scraper_outage"
CAUSE_NO_RACING = "no_racing"
CAUSE_PARTIAL_CARD = "partial_card"
CAUSE_MACHINE_OFF = "machine_off"
CAUSE_NO_RUN_RECORDED = "no_run_recorded"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_date(d: date) -> str:
    return d.isoformat()


@dataclass(frozen=True)
class DayRecord:
    day: str
    status: str
    cause: Optional[str]
    recorded_at: str


def load_ledger(path: str = DEFAULT_GAP_LEDGER_PATH) -> dict:
    """Return ``{iso_date: {"status", "cause", "recorded_at"}}``. Missing -> empty."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("gap_ledger: %s unreadable (%s); treating as empty", path, exc)
        return {}
    return raw.get("days") or {} if isinstance(raw, dict) else {}


def _write(path: str, days: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump({"days": days}, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def record_day(
    path: str = DEFAULT_GAP_LEDGER_PATH,
    *,
    day: Optional[date] = None,
    captured: bool,
    cause: Optional[str] = None,
    now: Optional[datetime] = None,
) -> dict:
    """Upsert today's capture status. Idempotent and safe to call many times a day.

    Once a day is marked ``captured`` it never reverts to ``gap`` on a later
    call the same day, even if that later call itself failed — one successful
    poll is enough to make the day real evidence; a later failed poll does not
    retroactively un-capture it.
    """
    now = now or _now()
    d = day or now.date()
    key = _iso_date(d)
    days = load_ledger(path)
    existing = days.get(key)

    if existing and existing.get("status") == CAPTURED:
        return days

    status = CAPTURED if captured else GAP
    days[key] = {
        "status": status,
        "cause": None if captured else (cause or CAUSE_SCRAPER_OUTAGE),
        "recorded_at": now.isoformat(),
    }
    _write(path, days)
    return days


def _captured_dates_from_snapshots(db_path: str) -> set:
    """Distinct calendar dates with at least one snapshot row, read directly.

    A read-only query against ``odds_snapshots`` rather than going through
    :class:`execution.snapshots.SnapshotStore` — this module only ever needs a
    distinct-date projection for reconciliation, not the store's point-in-time
    read/write contract.
    """
    if not os.path.exists(db_path):
        return set()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT DISTINCT substr(fetched_at, 1, 10) FROM odds_snapshots"
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("gap_ledger: could not read %s (%s)", db_path, exc)
        return set()
    return {r[0] for r in rows if r and r[0]}


def reconcile_gaps(
    path: str = DEFAULT_GAP_LEDGER_PATH,
    *,
    window_start: date,
    db_path: str,
    today: Optional[date] = None,
) -> dict:
    """Back-annotate any past day in ``[window_start, today)`` with no ledger entry.

    Only ever adds ``gap`` entries for days strictly before ``today`` that have
    neither an existing ledger entry nor any captured snapshot row — i.e. days
    the daily loop plainly never ran on. Cause is ``no_run_recorded``: this
    module cannot distinguish a dead machine from a skipped cron job, so it
    names the fact (nothing ran) rather than guessing why.
    """
    today = today or _now().date()
    days = load_ledger(path)
    captured_dates = _captured_dates_from_snapshots(db_path)

    d = window_start
    changed = False
    while d < today:
        key = _iso_date(d)
        if key not in days:
            if key in captured_dates:
                days[key] = {
                    "status": CAPTURED,
                    "cause": None,
                    "recorded_at": _now().isoformat(),
                }
            else:
                days[key] = {
                    "status": GAP,
                    "cause": CAUSE_NO_RUN_RECORDED,
                    "recorded_at": _now().isoformat(),
                }
            changed = True
        d += timedelta(days=1)

    if changed:
        _write(path, days)
    return days


def qualifying_dates(days: dict) -> list:
    """Sorted ISO dates whose status is ``captured`` — the only ones that count."""
    return sorted(k for k, v in days.items() if v.get("status") == CAPTURED)


def gap_entries(days: dict) -> list:
    """Sorted ``(date, cause)`` pairs for every recorded gap, for the report."""
    return sorted(
        (k, v.get("cause")) for k, v in days.items() if v.get("status") == GAP
    )


def qualifying_weeks(days: dict, *, window_start: Optional[date] = None) -> float:
    """Gap-adjusted weeks of evidence: captured days / 7, not wall-clock elapsed.

    This is the number that must feed the forward gate's ``min_weeks``
    criterion (via ``evaluate_forward_gate``'s ``metrics`` override) instead of
    a wall-clock span, so a machine that was off for a month cannot make an
    eight-week gate pass on two real weeks of data.
    """
    dates = qualifying_dates(days)
    if window_start is not None:
        dates = [d for d in dates if date.fromisoformat(d) >= window_start]
    return len(dates) / 7.0


def is_qualifying_day(days: dict, day: date) -> bool:
    rec = days.get(_iso_date(day))
    return bool(rec and rec.get("status") == CAPTURED)
