import pandas as pd

from scraper.timeform import features


def test_going_speed_maps_via_config():
    df = pd.DataFrame({"going": ["Good", "good-to-soft", "Heavy", "unknown"]})
    out = features.add_going_speed(df, {"good": 3, "good-to-soft": 2, "heavy": 0})
    assert list(out["going_speed"]) == [3, 2, 0, None]


def test_class_change_vs_previous_run():
    df = pd.DataFrame({
        "horse_name": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "race_class": [4, 3, 3],
    })
    out = features.add_class_change(df).sort_values("race_date")
    # first run has no prior -> NA; then 4->3 = -1 (up in class); 3->3 = 0
    assert pd.isna(out.iloc[0]["class_change"])
    assert out.iloc[1]["class_change"] == -1
    assert out.iloc[2]["class_change"] == 0


def test_trailing_win_rate_excludes_same_day_row():
    # A won on 2026-03-01. Its OWN win must not count toward its 2026-03-01 rate.
    df = pd.DataFrame({
        "horse_name": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = features.add_trailing_rates(
        df, entity="horse_name", win_col="historical_win_rate",
        place_col="historical_place_rate", lookback_months=12,
        lookback_runs=20, place_positions=3).sort_values("race_date")
    # On 2026-03-01: prior runs are pos 1 and pos 2 -> 1 win of 2 = 0.5, not 2/3.
    assert out.iloc[2]["historical_win_rate"] == 0.5
    assert out.iloc[2]["runs_in_window"] == 2


def test_trailing_rate_cold_start_is_null():
    df = pd.DataFrame({
        "horse_name": ["A"],
        "race_date": pd.to_datetime(["2026-03-01"], utc=True),
        "position": [1],
    })
    out = features.add_trailing_rates(
        df, entity="horse_name", win_col="historical_win_rate",
        place_col="historical_place_rate", lookback_months=12,
        lookback_runs=20, place_positions=3)
    assert pd.isna(out.iloc[0]["historical_win_rate"])
    assert out.iloc[0]["runs_in_window"] == 0
