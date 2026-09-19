"""Tests for ui/_form.py — historical form + feature-row resolution.

These lock the data → view contract for the detail pages headless (no Streamlit,
no parquet on disk): history ordering, connection aggregates, the SHAP
feature-row resolution order, and — crucially — the regression that a horse with
history no longer resolves to an empty feature row (the old silently-blank
breakdown bug, caused by a synthetic features.parquet whose ids never matched).
"""
from __future__ import annotations

import pandas as pd

from ui import _form


def _matrix() -> pd.DataFrame:
    """A tiny two-horse matrix standing in for training_full.parquet."""
    return pd.DataFrame([
        # horse h1 — two runs, most recent 2025-10
        {"horse_id": "h1", "horse_name": "Alpha", "race_date": "2025-04-09",
         "venue": "Kempton", "position": 5, "field_size": 13, "going": "Soft",
         "race_class": "Class 4", "distance": "7f", "horse_speed": 80.0,
         "horse_speed_rank": 6, "sp": 9.0, "won": 0, "placed": 0,
         "jockey_name": "J One", "trainer_name": "T One", "jockey_win_rate": 0.1,
         "trainer_win_rate": 0.05, "historical_place_rate": 0.2,
         "horse_career_runs": 10, "speed_trend": 1.4, "field_size_dup": 0},
        {"horse_id": "h1", "horse_name": "Alpha", "race_date": "2025-10-22",
         "venue": "Ascot", "position": 1, "field_size": 8, "going": "Good",
         "race_class": "Class 3", "distance": "1m", "horse_speed": 95.0,
         "horse_speed_rank": 1, "sp": 3.5, "won": 1, "placed": 1,
         "jockey_name": "J One", "trainer_name": "T One", "jockey_win_rate": 0.15,
         "trainer_win_rate": 0.08, "historical_place_rate": 0.5,
         "horse_career_runs": 11, "speed_trend": 2.0},
        # horse h2 — one run
        {"horse_id": "h2", "horse_name": "Bravo", "race_date": "2025-09-01",
         "venue": "Naas", "position": 3, "field_size": 10, "going": "Yielding",
         "race_class": "Class 5", "distance": "6f", "horse_speed": 70.0,
         "horse_speed_rank": 4, "sp": 12.0, "won": 0, "placed": 1,
         "jockey_name": "J Two", "trainer_name": "T Two", "jockey_win_rate": 0.2,
         "trainer_win_rate": 0.1, "historical_place_rate": 1.0,
         "horse_career_runs": 1, "speed_trend": -1.0},
    ])


# ── recent form ───────────────────────────────────────────────────────────────

def test_history_is_most_recent_first():
    runs = _form.horse_history_from(_matrix(), "h1")
    assert [r["date"] for r in runs] == ["2025-10-22", "2025-04-09"]
    assert runs[0]["position"] == 1 and runs[0]["won"] is True
    assert runs[0]["venue"] == "Ascot" and runs[0]["field_size"] == 8


def test_history_respects_limit_and_unknown_horse():
    assert len(_form.horse_history_from(_matrix(), "h1", limit=1)) == 1
    assert _form.horse_history_from(_matrix(), "nope") == []
    assert _form.horse_history_from(pd.DataFrame(), "h1") == []


def test_run_record_distance_falls_back_to_furlongs():
    m = pd.DataFrame([{"horse_id": "x", "race_date": "2025-01-01",
                       "distance_furlongs": 7.0}])
    rec = _form.horse_history_from(m, "x")[0]
    assert rec["distance"] == "7f"


# ── connection / aggregate stats ──────────────────────────────────────────────

def test_connection_stats_read_off_latest_row():
    stats = _form.connection_stats_from(_matrix(), "h1")
    # latest row (2025-10) values, not the older ones
    assert stats["jockey_name"] == "J One"
    assert stats["jockey_win_rate"] == 0.15
    assert stats["historical_place_rate"] == 0.5
    assert stats["runs"] == 2


def test_connection_stats_empty_for_unknown():
    assert _form.connection_stats_from(_matrix(), "nope") == {}


# ── feature-row resolution (the empty-breakdown regression) ───────────────────

def test_feature_row_resolves_from_history_for_known_horse():
    """Regression: a horse present in the matrix must resolve a NON-empty feature
    row from its latest run — the path the breakdown was silently missing before."""
    resolved = _form.feature_row_from(_matrix(), None, "h1")
    assert resolved is not None
    row, source = resolved
    assert source == "history"
    # latest row → its speed/feature values are present (not the synthetic fixture)
    assert row["horse_speed"] == 95.0
    assert row["horse_id"] == "h1"


def test_feature_row_prefers_live_inference_over_history():
    inference = pd.DataFrame([{"horse_id": "h1", "horse_speed": 111.0,
                               "field_size": 9}])
    row, source = _form.feature_row_from(_matrix(), inference, "h1")
    assert source == "live"
    assert row["horse_speed"] == 111.0


def test_feature_row_none_when_nowhere():
    assert _form.feature_row_from(_matrix(), None, "ghost") is None
    assert _form.feature_row_from(pd.DataFrame(), None, "h1") is None


# ── derivations ───────────────────────────────────────────────────────────────

def test_trend_label_signs():
    assert _form.trend_label(2.0) == "Improving"
    assert _form.trend_label(-2.0) == "Declining"
    assert _form.trend_label(0.0) == "Steady"
    assert _form.trend_label(None) is None


def test_race_shape_clear_favourite():
    sels = [{"rank": 1, "horse_name": "Fav", "won_prob": 0.55, "value_bet": False},
            {"rank": 2, "horse_name": "Two", "won_prob": 0.20}]
    s = _form.race_shape(sels)
    assert s["top_name"] == "Fav"
    assert "clear favourite" in s["shape"]
    assert s["n_value"] == 0


def test_race_shape_wide_open_and_value_count():
    sels = [{"rank": 1, "horse_name": "A", "won_prob": 0.16, "value_bet": True},
            {"rank": 2, "horse_name": "B", "won_prob": 0.15, "value_bet": True}]
    s = _form.race_shape(sels)
    assert "wide-open" in s["shape"]
    assert s["n_value"] == 2


def test_race_shape_no_scored_runners():
    s = _form.race_shape([{"rank": 1, "horse_name": "A", "won_prob": None}])
    assert s["top_name"] is None
