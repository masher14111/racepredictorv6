from datetime import date

import pandas as pd

from tools.coverage_report import build_coverage_report


def test_detects_all_null_columns_and_watched_fill():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "source": ["betsp", "betsp"],
        "position": [1, None],
        "timeform_rating": [None, None],
    })
    report = build_coverage_report(df, "test", watch_cols=["position", "timeform_rating", "missing_col"])
    assert report["all_null_columns"] == ["timeform_rating"]
    assert report["watched_fill"]["position"]["fill_fraction"] == 0.5
    assert report["watched_fill"]["timeform_rating"]["fill_fraction"] == 0.0
    assert report["watched_fill"]["missing_col"] == {"present": False}


def test_detects_staleness_against_today():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01"], utc=True),
        "source": ["betsp"],
    })
    report = build_coverage_report(df, "test", today=date(2026, 1, 10), stale_after_days=2)
    assert report["days_stale"] == 9
    assert report["is_stale"] is True


def test_not_stale_when_within_threshold():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-09"], utc=True),
        "source": ["betsp"],
    })
    report = build_coverage_report(df, "test", today=date(2026, 1, 10), stale_after_days=2)
    assert report["is_stale"] is False


def test_flags_a_missing_calendar_day_in_the_recent_window():
    dates = ["2026-01-01", "2026-01-02", "2026-01-04"]  # 01-03 missing
    df = pd.DataFrame({
        "race_date": pd.to_datetime(dates, utc=True),
        "source": ["betsp"] * 3,
    })
    report = build_coverage_report(df, "test")
    assert "2026-01-03" in report["missing_calendar_days_recent"]


def test_flags_a_sudden_provider_share_shift():
    rows = []
    for i in range(10):
        rows.append({"race_date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                     "source": "betsp"})
        rows.append({"race_date": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=i),
                     "source": "timeform"})
    # Day 10: timeform goes dark entirely (0% instead of its steady ~50%).
    rows.append({"race_date": pd.Timestamp("2026-01-11", tz="UTC"), "source": "betsp"})
    df = pd.DataFrame(rows)
    report = build_coverage_report(df, "test")
    alerts = report["provider_shift_alerts"]
    assert any(a["date"] == "2026-01-11" and a["source"] == "timeform" for a in alerts)


def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame({"race_date": pd.Series([], dtype="datetime64[ns, UTC]"), "source": pd.Series([], dtype="object")})
    report = build_coverage_report(df, "empty")
    assert report["rows"] == 0
    assert report["all_null_columns"] == []
