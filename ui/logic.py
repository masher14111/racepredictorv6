"""Pure, Streamlit-free UI logic — extracted so it can be unit-tested headless.

These functions encode the two pieces of behaviour that caused the original
"GUI shows nothing" symptom (Task 03):

* the date-filter default (must fall back to "All upcoming" when the cache has
  no race dated today, otherwise a stale cache hides behind an empty "Today"), and
* the empty-state message selection (distinguish "no cache" / "no models" /
  "0 upcoming races" / "no races match filters").

``ui/app.py`` imports these so the rendering layer and the decision logic stay
in sync, and tests can exercise the logic without importing Streamlit.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from utils.timezone import to_local

# Sidebar date-filter options, in display order.
DATE_LABELS = ["Today", "Tomorrow", "All upcoming"]


def parse_race_date(iso: Optional[str]) -> Optional[date]:
    """Parse an ISO race_time/generated_at string to a local calendar date.

    Returns None for empty/unparseable input. Aware timestamps are converted to
    local time first so the date matches the Europe/Dublin "today"."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = to_local(dt)
        return dt.date()
    except (ValueError, TypeError):
        return None


def has_today_race(races: list[dict], today: date) -> bool:
    """True if any race in the cache is dated `today` (local)."""
    return any(parse_race_date(r.get("race_time")) == today for r in races or [])


def default_date_index(races: list[dict], today: date) -> int:
    """Index into DATE_LABELS for the default date filter.

    "Today" (0) only when the cache actually holds a race for today; otherwise
    "All upcoming" so a stale cache is never hidden behind an empty Today view."""
    if has_today_race(races, today):
        return 0
    return DATE_LABELS.index("All upcoming")


def empty_state(
    preds: Optional[dict],
    races_all: list[dict],
    races_filtered: list[dict],
) -> Optional[tuple[str, str]]:
    """Pick the empty-state (title, reason_html) when there is nothing to show.

    Returns None when there ARE races to render. Mirrors the branch order in
    ``app.main``: no cache → no models → 0 upcoming races → filters too narrow."""
    if races_filtered:
        return None
    model_targets = (preds or {}).get("model_targets") or []
    if not preds:
        return (
            "No prediction cache found",
            "Click <strong>Refresh predictions</strong> in the sidebar,"
            " or run <code>python -m models.predictor</code> from the terminal.",
        )
    if not model_targets:
        return (
            "No models loaded",
            "The prediction cache has no model targets. "
            "Run <code>python -m models.train</code>, then refresh.",
        )
    if not races_all:
        return (
            "0 upcoming races found",
            "The predictor ran but produced no races — there are no upcoming "
            "racecards in the data. Scrape today's racecards, then "
            "<strong>Refresh predictions</strong>.",
        )
    return (
        "No races match the current filters",
        "Try widening the date to <strong>All upcoming</strong> or "
        "selecting more venues in the sidebar.",
    )
