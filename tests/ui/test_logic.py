"""Tests for ui/logic.py — the headless GUI decision logic (Task 03 regressions).

Covers the date-filter default (stale cache must not hide behind an empty
"Today") and the empty-state branch selection.
"""
from __future__ import annotations

from datetime import date

from ui.logic import (
    DATE_LABELS,
    default_date_index,
    empty_state,
    has_today_race,
    parse_race_date,
)


# ── parse_race_date ───────────────────────────────────────────────────────────

class TestParseRaceDate:
    def test_none_for_empty(self):
        assert parse_race_date(None) is None
        assert parse_race_date("") is None

    def test_none_for_garbage(self):
        assert parse_race_date("not-a-date") is None

    def test_naive_iso(self):
        assert parse_race_date("2026-06-14 14:30:00") == date(2026, 6, 14)

    def test_aware_iso_converted_to_local(self):
        # Aware UTC timestamp → Europe/Dublin date.
        assert parse_race_date("2026-06-14T13:00:00+00:00") == date(2026, 6, 14)


# ── date-filter default (the "GUI shows nothing" fix) ─────────────────────────

class TestDefaultDateIndex:
    def test_today_when_cache_has_today(self):
        today = date(2026, 6, 14)
        races = [{"race_time": "2026-06-14T13:00:00+00:00"}]
        assert has_today_race(races, today) is True
        assert default_date_index(races, today) == 0
        assert DATE_LABELS[0] == "Today"

    def test_all_upcoming_when_cache_is_stale(self):
        today = date(2026, 6, 14)
        races = [{"race_time": "2026-06-13T13:00:00+00:00"}]  # yesterday only
        assert has_today_race(races, today) is False
        idx = default_date_index(races, today)
        assert DATE_LABELS[idx] == "All upcoming"

    def test_all_upcoming_when_no_races(self):
        today = date(2026, 6, 14)
        assert default_date_index([], today) == DATE_LABELS.index("All upcoming")


# ── empty-state branch selection ──────────────────────────────────────────────

class TestEmptyState:
    def test_none_when_races_present(self):
        assert empty_state({"model_targets": ["won"]}, [{"venue": "X"}], [{"venue": "X"}]) is None

    def test_no_cache(self):
        title, _ = empty_state(None, [], [])
        assert title == "No prediction cache found"

    def test_no_models(self):
        preds = {"model_targets": [], "races": []}
        title, _ = empty_state(preds, [], [])
        assert title == "No models loaded"

    def test_zero_upcoming_races(self):
        preds = {"model_targets": ["won"], "races": []}
        title, _ = empty_state(preds, [], [])
        assert title == "0 upcoming races found"

    def test_filters_too_narrow(self):
        # cache has races + models, but the current filter removed them all.
        preds = {"model_targets": ["won"], "races": [{"venue": "Ascot"}]}
        races_all = [{"venue": "Ascot", "race_time": "2026-06-14T13:00:00+00:00"}]
        title, _ = empty_state(preds, races_all, [])
        assert title == "No races match the current filters"
