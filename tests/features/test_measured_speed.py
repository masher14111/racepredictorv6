"""Step 11: par fitting and the measured pre-race feature family.

The load-bearing properties here are the leak ones:
  * a par is fitted on strictly EARLIER races only
  * appending future races cannot change a historical figure (append invariance)
  * frame order cannot change a figure (row-order invariance)
  * a race never contributes to its own pre-race feature
  * a non-finisher gets a null figure, never a slow one
"""
import numpy as np
import pandas as pd
import pytest

from features import measured_speed as ms


def _races(rows):
    """rows = (race_key, off_time, venue, distance_yards, winner_speed_yps)."""
    return pd.DataFrame([{
        "race_key": k, "off_time": t, "venue": v, "surface": "TURF",
        "going": "Good", "race_type": "flat", "distance_yards": d,
        "winner_speed_yps": s,
    } for k, t, v, d, s in rows])


def _series(n, speed=17.0, venue="ascot", start_hour=12):
    rows = []
    for i in range(n):
        day = f"2024-01-{(i // 8) + 1:02d}"
        clock = f"{start_hour + (i % 8):02d}:00"
        rows.append((f"{venue}|{day}T{clock}", f"{day}T{clock}", venue, 1760.0,
                     speed + 0.001 * i))
    return _races(rows)


# ───────────────────────────── par fitting ─────────────────────────────
def test_par_uses_only_strictly_earlier_races():
    races = _series(40)
    out = ms.add_race_pars(races, min_prior=1).sort_values("off_time").reset_index(drop=True)
    # The very first race has no prior at any level, so it gets no par at all.
    assert pd.isna(out.loc[0, "par_mean"])
    # The second race's par is exactly the first race's log speed.
    assert out.loc[1, "par_mean"] == pytest.approx(np.log(races["winner_speed_yps"].iloc[0]))
    assert out.loc[1, "par_n"] == 1


def test_par_never_includes_a_same_instant_race():
    """Two races at the same instant must not inform each other's par -- that is
    what makes the result independent of frame order."""
    races = _races([
        ("a|2024-01-01T12:00", "2024-01-01T12:00", "ascot", 1760.0, 17.0),
        ("b|2024-01-01T12:00", "2024-01-01T12:00", "ascot", 1760.0, 19.0),
        ("c|2024-01-01T13:00", "2024-01-01T13:00", "ascot", 1760.0, 18.0),
    ])
    out = ms.add_race_pars(races, min_prior=1).set_index("race_key")
    assert pd.isna(out.loc["a|2024-01-01T12:00", "par_mean"])
    assert pd.isna(out.loc["b|2024-01-01T12:00", "par_mean"])
    assert out.loc["c|2024-01-01T13:00", "par_n"] == 2


def test_par_is_invariant_to_appended_future_races():
    """The headline acceptance check: historical values must not move when
    later records arrive."""
    early = _series(30)
    late = _series(30)
    late["off_time"] = late["off_time"].str.replace("2024-01", "2025-06", regex=False)
    late["race_key"] = late["race_key"].str.replace("2024-01", "2025-06", regex=False)

    before = ms.add_race_pars(early, min_prior=1).set_index("race_key")
    after = ms.add_race_pars(pd.concat([early, late], ignore_index=True),
                             min_prior=1).set_index("race_key")
    shared = before.index
    pd.testing.assert_series_equal(before["par_mean"], after.loc[shared, "par_mean"])
    pd.testing.assert_series_equal(before["par_n"], after.loc[shared, "par_n"])


def test_par_is_invariant_to_row_order():
    races = _series(40)
    straight = ms.add_race_pars(races, min_prior=1).set_index("race_key")["par_mean"]
    shuffled = ms.add_race_pars(
        races.sample(frac=1.0, random_state=7).reset_index(drop=True),
        min_prior=1).set_index("race_key")["par_mean"]
    pd.testing.assert_series_equal(straight, shuffled.loc[straight.index])


def test_backoff_prefers_the_most_specific_sufficient_level():
    races = _series(40, venue="ascot")
    out = ms.add_race_pars(races, min_prior=15).sort_values("off_time")
    tail = out.iloc[20:]
    assert (tail["par_level"] == 0).all(), "a well-populated course/distance cell wins"
    head = out.iloc[:3]
    assert head["par_mean"].isna().all(), "no level has enough prior races yet"


def test_backoff_falls_through_when_the_specific_cell_is_thin():
    """One race at a brand-new course still gets a par, from a coarser level."""
    common = _series(40, venue="ascot")
    rare = _races([("york|2024-01-09T12:00", "2024-01-09T12:00", "york", 1760.0, 17.5)])
    out = ms.add_race_pars(pd.concat([common, rare], ignore_index=True), min_prior=15)
    row = out.loc[out["venue"] == "york"].iloc[0]
    assert row["par_level"] > 1, "must back off past the empty course-specific cells"
    assert not pd.isna(row["par_mean"])


# ───────────────────────────── runner figures ─────────────────────────────
def _runners(rows, race_key="ascot|2024-02-01T12:00"):
    return pd.DataFrame([{
        "race_key": race_key, "horse_key": h, "horse_id": h,
        "finished": fin, "est_speed_yps": sp,
    } for h, fin, sp in rows])


def test_figure_is_null_for_a_non_finisher_never_slow():
    races = _series(40)
    extra = _races([("ascot|2024-02-01T12:00", "2024-02-01T12:00", "ascot", 1760.0, 17.0)])
    allraces = pd.concat([races, extra], ignore_index=True)
    runners = _runners([("winner", True, 17.0), ("faller", False, np.nan)])
    out = ms.add_speed_figures(runners, allraces, min_prior=1).set_index("horse_key")
    assert not pd.isna(out.loc["winner", "measured_speed_z"])
    assert pd.isna(out.loc["faller", "measured_speed_pct"])
    assert pd.isna(out.loc["faller", "measured_speed_z"])


def test_faster_than_par_is_positive():
    races = _series(40, speed=17.0)
    extra = _races([("ascot|2024-02-01T12:00", "2024-02-01T12:00", "ascot", 1760.0, 17.0)])
    allraces = pd.concat([races, extra], ignore_index=True)
    out = ms.add_speed_figures(_runners([("quick", True, 18.5), ("slow", True, 15.5)]),
                               allraces, min_prior=1).set_index("horse_key")
    assert out.loc["quick", "measured_speed_pct"] > 0
    assert out.loc["slow", "measured_speed_pct"] < 0


# ───────────────────────── pre-race feature family ─────────────────────────
def _history(n_runs, figure_start=0.0):
    rows = []
    for i in range(n_runs):
        day = f"2024-{(i % 12) + 1:02d}-15"
        uid = f"Ascot|{day}T14:00"
        rows.append({"race_uid": uid, "race_date": pd.Timestamp(day, tz="UTC"),
                     "venue": "Ascot", "market_type": "WIN", "horse_id": "h1",
                     "distance_furlongs": 8.0, "going_band": "good",
                     "measured_speed_z": figure_start + i})
    return pd.DataFrame(rows)


def test_pre_race_feature_excludes_the_races_own_figure():
    hist = _history(5)
    out = ms.add_measured_pre_race_features(hist)
    # The first run has no prior figure at all.
    assert pd.isna(out.loc[0, "msf_last"])
    # The second run's msf_last is the FIRST run's figure, not its own.
    assert out.loc[1, "msf_last"] == pytest.approx(hist.loc[0, "measured_speed_z"])
    assert out.loc[1, "msf_last"] != hist.loc[1, "measured_speed_z"]


def test_pre_race_features_are_invariant_to_appended_future_runs():
    hist = _history(6)
    future = _history(4, figure_start=100.0)
    future["race_date"] = future["race_date"] + pd.DateOffset(years=1)
    future["race_uid"] = future["race_uid"].str.replace("2024", "2025", regex=False)

    before = ms.add_measured_pre_race_features(hist)
    after = ms.add_measured_pre_race_features(
        pd.concat([hist, future], ignore_index=True)).iloc[: len(hist)]
    for col in ("msf_last", "msf_mean3", "msf_mean6"):
        pd.testing.assert_series_equal(before[col], after[col], check_names=False)


def test_trend_is_short_window_minus_long_window():
    out = ms.add_measured_pre_race_features(_history(8))
    row = out.iloc[7]
    assert row["msf_trend"] == pytest.approx(row["msf_mean3"] - row["msf_mean6"])


def test_missing_figure_column_yields_null_family_not_an_error():
    frame = _history(3).drop(columns=["measured_speed_z"])
    out = ms.add_measured_pre_race_features(frame)
    for col in ms.MEASURED_FEATURE_COLS:
        assert col in out.columns and out[col].isna().all()


def test_rank_splits_win_and_place_books():
    """D23: a per-race grouping that ignores market_type mixes two price books."""
    rows = []
    # An earlier run per horse, so each one carries a prior figure to rank on.
    for market in ("WIN", "PLACE"):
        for i, horse in enumerate(["a", "b", "c"]):
            rows.append({"race_uid": "Ascot|2024-04-01T14:00",
                         "race_date": pd.Timestamp("2024-04-01", tz="UTC"),
                         "venue": "Ascot", "market_type": market, "horse_id": horse,
                         "distance_furlongs": 8.0, "going_band": "good",
                         "measured_speed_z": float(i)})
    for market in ("WIN", "PLACE"):
        for i, horse in enumerate(["a", "b", "c"]):
            rows.append({"race_uid": "Ascot|2024-05-01T14:00",
                         "race_date": pd.Timestamp("2024-05-01", tz="UTC"),
                         "venue": "Ascot", "market_type": market, "horse_id": horse,
                         "distance_furlongs": 8.0, "going_band": "good",
                         "measured_speed_z": float(i)})
    out = ms.add_measured_pre_race_features(pd.DataFrame(rows))
    target = out.loc[out["race_uid"] == "Ascot|2024-05-01T14:00"]
    # Each book ranks its own three runners 1..3, rather than one pooled 1..6.
    assert set(target.groupby("market_type")["msf_rank"].count()) == {3}
    assert set(target.loc[target["market_type"] == "WIN", "msf_rank"]) == {1.0, 2.0, 3.0}
    assert set(target.loc[target["market_type"] == "PLACE", "msf_rank"]) == {1.0, 2.0, 3.0}
