import math

import pandas as pd

from features import derive


# --- relocated leak-safe primitives ---

def test_going_speed_maps_via_config():
    df = pd.DataFrame({"going": ["Good", "good-to-soft", "Heavy", "unknown"]})
    out = derive.add_going_speed(df, {"good": 3, "good-to-soft": 2, "heavy": 0})
    assert list(out["going_speed"]) == [3, 2, 0, None]


_GOING_MAP = {
    "firm": 5, "good-to-firm": 4, "good": 3, "good-to-soft": 2, "soft": 1, "heavy": 0,
}


def test_going_speed_v2_matches_the_original_bug_for_reference():
    """Documents the bug going_speed_v2 fixes: the space/hyphen mismatch means
    add_going_speed (unchanged, frozen for existing bundles) never matches a
    two-word band at all, only the 4 single-word exact strings."""
    df = pd.DataFrame({"going": ["Good to Firm", "Good to Soft", "Good", "Heavy"]})
    out = derive.add_going_speed(df, _GOING_MAP)
    assert list(out["going_speed"]) == [None, None, 3, 0]


def test_going_speed_v2_matches_composite_and_irish_going_strings():
    df = pd.DataFrame({"going": [
        "Good to Firm",                          # space/hyphen mismatch (the bug)
        "Good (Good to Soft in places)",         # parenthetical qualifier
        "Standard / Slow",                       # slash-alt AW reading
        "Yielding",                              # Irish equivalent of soft
        "Good to Yielding (Yielding in places)", # Irish + parenthetical
        "Soft to Heavy",                         # two-band transition string
        "unrecognised nonsense",
        None,
    ]})
    out = derive.add_going_speed_v2(df, _GOING_MAP)
    # "Good (Good to Soft in places)" -> headline "Good" (3): the parenthetical
    # qualifier is dropped entirely, same rule as the two-band transition case.
    assert list(out["going_speed_v2"]) == [4, 3, None, 1, 2, 1, None, None]
    assert list(out["going_is_all_weather"]) == [False, False, True, False, False, False, False, None]


def test_going_speed_v2_never_invents_a_value_off_the_frozen_scale():
    df = pd.DataFrame({"going": ["Firm", "Soft", "Heavy", "Good"]})
    out = derive.add_going_speed_v2(df, _GOING_MAP)
    assert set(v for v in out["going_speed_v2"] if v is not None) <= set(_GOING_MAP.values())


def test_trailing_win_rate_excludes_same_day_row():
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = derive.add_trailing_rates(
        df, entity="horse_id", win_col="historical_win_rate",
        place_col="historical_place_rate", lookback_months=12,
        lookback_runs=20, place_positions=3).sort_values("race_date")
    assert out.iloc[2]["historical_win_rate"] == 0.5
    assert out.iloc[2]["runs_in_window"] == 2


# --- odds features ---

def test_odds_features_implied_prob_and_rank():
    df = pd.DataFrame({
        "race_date": ["2026-06-13"] * 3, "venue": ["X"] * 3,
        "horse_id": ["a", "b", "c"],
        "odds_decimal": [2.0, 4.0, 5.0], "sp": [None, None, None],
        "odds_finish": [None, None, None],
    })
    out = derive.add_odds_features(df)
    assert out.iloc[0]["implied_prob"] == 0.5
    assert out.iloc[0]["market_rank"] == 1  # shortest odds = rank 1
    assert out.iloc[0]["field_size"] == 3
    # overround-normalized probs sum to 1 within the race
    assert abs(out["overround_norm_prob"].sum() - 1.0) < 1e-9
    assert abs(out.iloc[0]["log_odds"] - math.log(2.0)) < 1e-9


def test_odds_features_fall_back_to_sp_then_morningwap():
    df = pd.DataFrame({
        "race_date": ["2026-06-13"] * 2, "venue": ["X"] * 2,
        "horse_id": ["a", "b"],
        "odds_decimal": [None, None], "sp": [3.0, None], "morningwap": [None, 6.0],
    })
    out = derive.add_odds_features(df)
    assert out.iloc[0]["implied_prob"] == 1 / 3.0
    assert out.iloc[1]["implied_prob"] == 1 / 6.0


def test_odds_features_never_use_finishing_sp():
    """C2: odds_finish (returned/finishing SP) must NOT feed market features — it
    leaks the outcome. A row priced only by odds_finish yields a null implied_prob."""
    df = pd.DataFrame({
        "race_date": ["2026-06-13"] * 2, "venue": ["X"] * 2,
        "horse_id": ["a", "b"],
        "odds_decimal": [None, None], "sp": [None, None],
        "morningwap": [4.0, None], "odds_finish": [2.0, 6.0],
    })
    out = derive.add_odds_features(df)
    assert out.iloc[0]["implied_prob"] == 1 / 4.0   # morningwap used, not odds_finish=2.0
    assert pd.isna(out.iloc[1]["implied_prob"])      # only odds_finish present -> null


def test_odds_decimal_drives_implied_prob_regression():
    """Task 08 regression: the live `odds_decimal` column (what the scrapers write)
    must derive implied_prob = 1/odds for every priced runner."""
    df = pd.DataFrame({
        "race_date": ["2026-06-14"] * 3, "venue": ["Sandown"] * 3,
        "horse_id": ["a", "b", "c"],
        "odds_decimal": [2.5, 4.0, 10.0], "sp": [None, None, None],
        "odds_finish": [None, None, None],
    })
    out = derive.add_odds_features(df)
    assert out["implied_prob"].notna().all()
    assert out.iloc[0]["implied_prob"] == 1 / 2.5
    assert out.iloc[2]["implied_prob"] == 1 / 10.0


def test_odds_features_null_when_no_usable_odds():
    df = pd.DataFrame({
        "race_date": ["2026-06-13"], "venue": ["X"], "horse_id": ["a"],
        "odds_decimal": [None], "sp": [None], "odds_finish": [None],
    })
    out = derive.add_odds_features(df)
    assert pd.isna(out.iloc[0]["implied_prob"])


def test_odds_features_do_not_mix_win_and_place_books():
    """Step 03: a runner now carries one row per market (features/fuse.py). A
    race's WIN book and PLACE book are disjoint populations for field_size/
    market_rank/overround_norm_prob — grouping on race_uid alone (ignoring
    market_type) would rank a WIN row's implied_prob against the PLACE book's
    prices and double the field size."""
    df = pd.DataFrame({
        "race_date": ["2026-06-13"] * 4, "venue": ["X"] * 4,
        "horse_id": ["a", "b", "a", "b"],
        "market_type": ["WIN", "WIN", "PLACE", "PLACE"],
        # WIN book: a is favourite. PLACE book: b is the shorter price.
        "odds_decimal": [2.0, 4.0, 1.5, 1.2],
        "sp": [None, None, None, None], "odds_finish": [None, None, None, None],
    })
    out = derive.add_odds_features(df)
    win = out[out["market_type"] == "WIN"].set_index("horse_id")
    place = out[out["market_type"] == "PLACE"].set_index("horse_id")
    assert win.loc["a", "field_size"] == 2
    assert place.loc["a", "field_size"] == 2  # not 4 (the two books combined)
    assert win.loc["a", "market_rank"] == 1     # favourite in the WIN book
    assert place.loc["b", "market_rank"] == 1   # favourite in the PLACE book
    # Each book's overround-normalized probs sum to 1 independently.
    assert abs(win["overround_norm_prob"].sum() - 1.0) < 1e-9
    assert abs(place["overround_norm_prob"].sum() - 1.0) < 1e-9


# --- per-race key (model-10: recover the off-time so field_size is per-race) ---

def test_race_key_uses_off_time_to_separate_races_in_a_venue_day():
    # Two races at the same venue/day with different off-times must NOT collapse.
    df = pd.DataFrame({
        "venue": ["Ascot"] * 4,
        "race_date": pd.to_datetime(["2026-06-13"] * 4, utc=True),
        "race_time": ["2026-06-13T14:00", "2026-06-13T14:00",
                      "2026-06-13T14:30", "2026-06-13T14:30"],
        "horse_id": ["a", "b", "c", "d"],
    })
    out = derive.add_race_key(df)
    assert out["race_uid"].nunique() == 2
    # field_size (via add_odds_features) is now per-race (2), not venue-day (4)
    out["odds_decimal"] = [2.0, 3.0, 4.0, 5.0]
    out["sp"] = None
    out["odds_finish"] = None
    feats = derive.add_odds_features(out)
    assert list(feats["field_size"]) == [2, 2, 2, 2]


def test_race_key_falls_back_to_venue_day_without_off_time():
    df = pd.DataFrame({
        "venue": ["Ascot", "Ascot"],
        "race_date": pd.to_datetime(["2026-06-13", "2026-06-13"], utc=True),
        "horse_id": ["a", "b"],
    })
    out = derive.add_race_key(df)
    assert out["race_uid"].nunique() == 1


# --- going band (model-10: derive a coarse band from the raw going string) ---

def test_going_band_maps_primary_descriptor():
    df = pd.DataFrame({"going": [
        "Good", "Good to Firm", "Soft (Heavy in places)", "Standard / Slow",
        "Heavy", "Yielding", None, ""]})
    out = derive.add_going_band(df)
    assert list(out["going_band"]) == [
        "good", "good", "soft", "standard", "heavy", "soft", None, None]


# --- freshness / experience (model-10) ---

def test_days_since_last_run_is_prior_gap_and_null_first_run():
    df = pd.DataFrame({
        "horse_id": ["A", "A", "B"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-15", "2026-02-01"], utc=True),
        "position": [1, 2, 3],
    })
    out = derive.add_days_since_last_run(df)
    assert pd.isna(out.iloc[0]["days_since_last_run"])      # A's first run
    assert out.iloc[1]["days_since_last_run"] == 14.0       # 14 days later
    assert pd.isna(out.iloc[2]["days_since_last_run"])      # B's first run


def test_career_runs_counts_strictly_prior_known_runs():
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, None, 2],   # middle run has no result -> not counted
    })
    out = derive.add_career_runs(df)
    assert list(out["horse_career_runs"]) == [0, 1, 1]


# --- step 04 / DECISIONS D25-D26: market-duplicate (WIN+PLACE) rows of the
# SAME real race must count as ONE prior run, never two, and must never leak
# into each other as a spurious same-race "prior" result. ---


def test_days_since_last_run_market_duplicate_rows_share_identical_gap():
    # Horse A: race1 (WIN+PLACE), race2 31 days later (WIN+PLACE). Both of
    # race2's rows must show the SAME 31-day gap, and neither of race1's rows
    # may see its own market sibling as a spurious prior (must stay NaN).
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A", "A"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    })
    out = derive.add_days_since_last_run(df)
    assert pd.isna(out.iloc[0]["days_since_last_run"])
    assert pd.isna(out.iloc[1]["days_since_last_run"])
    assert out.iloc[2]["days_since_last_run"] == 31.0
    assert out.iloc[3]["days_since_last_run"] == 31.0


def test_career_runs_market_duplicate_rows_share_identical_count():
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A", "A"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    })
    out = derive.add_career_runs(df)
    assert list(out["horse_career_runs"]) == [0, 0, 1, 1]


def test_class_change_market_duplicate_rows_share_identical_value():
    df = pd.DataFrame({
        "horse_name": ["Dobbin", "Dobbin", "Dobbin", "Dobbin"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "race_class": [4, 4, 2, 2],   # stepped UP (lower number = higher class)
    })
    out = derive.add_class_change(df)
    assert pd.isna(out.iloc[0]["class_change"])
    assert pd.isna(out.iloc[1]["class_change"])
    assert out.iloc[2]["class_change"] == -2
    assert out.iloc[3]["class_change"] == -2


def test_days_since_last_run_append_future_race_does_not_change_earlier_rows():
    """Append-future invariance: adding a LATER race for the same horse must not
    change any earlier admissible row's already-computed gap."""
    base = pd.DataFrame({
        "horse_id": ["A", "A"],
        "race_uid": ["R1", "R2"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
        "position": [1, 2],
    })
    future_row = pd.DataFrame({
        "horse_id": ["A"], "race_uid": ["R3"],
        "race_date": pd.to_datetime(["2026-06-01"], utc=True), "position": [3],
    })
    out_base = derive.add_days_since_last_run(base)
    out_appended = derive.add_days_since_last_run(
        pd.concat([base, future_row], ignore_index=True))
    pd.testing.assert_series_equal(
        out_base["days_since_last_run"], out_appended["days_since_last_run"].iloc[:2],
        check_names=False)


# --- distance ---

def test_distance_furlongs_parses_miles_furlongs_yards():
    df = pd.DataFrame({"distance": ["1m2f188y", "5f", "2m", None]})
    out = derive.add_distance_furlongs(df)
    assert out.iloc[0]["distance_furlongs"] == 10 + 188 / 220.0
    assert out.iloc[1]["distance_furlongs"] == 5.0
    assert out.iloc[2]["distance_furlongs"] == 16.0
    assert pd.isna(out.iloc[3]["distance_furlongs"])


# --- recent form ---

def test_recent_form_parses_figures():
    df = pd.DataFrame({"recent_form": ["50318-1", "PF-", None]})
    out = derive.add_recent_form(df)
    assert out.iloc[0]["recent_form_runs"] == 6
    assert out.iloc[0]["recent_form_wins"] == 2  # two '1's
    assert out.iloc[0]["recent_form_avg"] == (5 + 0 + 3 + 1 + 8 + 1) / 6.0
    # all non-numeric -> runs counts figures but avg/wins null/zero
    assert out.iloc[1]["recent_form_wins"] == 0
    assert pd.isna(out.iloc[1]["recent_form_avg"])
    assert out.iloc[2]["recent_form_runs"] == 0


# --- rating rank ---

def test_rating_rank_descending_within_race():
    df = pd.DataFrame({
        "race_date": ["2026-06-13"] * 3, "venue": ["X"] * 3,
        "horse_id": ["a", "b", "c"], "timeform_rating": [120, 100, 130],
    })
    out = derive.add_rating_rank(df)
    # highest rating -> rank 1
    assert out.set_index("horse_id").loc["c", "rating_rank"] == 1
    assert out.set_index("horse_id").loc["b", "rating_rank"] == 3


# --- trailing rates convenience wrapper ---

def test_add_all_trailing_rates_covers_horse_jockey_trainer():
    df = pd.DataFrame({
        "horse_id": ["A", "A"], "jockey_id": ["J", "J"], "trainer_id": ["T", "T"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
        "position": [1, 2],
    })
    cfg = {"lookback_months": 12, "lookback_runs": 20, "place_positions": 3}
    out = derive.add_all_trailing_rates(df, cfg).sort_values("race_date")
    # second run: horse/jockey/trainer each had one prior win -> rate 1.0
    assert out.iloc[1]["historical_win_rate"] == 1.0
    assert out.iloc[1]["jockey_win_rate"] == 1.0
    assert out.iloc[1]["trainer_win_rate"] == 1.0
