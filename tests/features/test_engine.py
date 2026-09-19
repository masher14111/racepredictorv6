# tests/features/test_engine.py

import math

import pandas as pd

from features import engine

CFG = {"lookback_months": 12, "lookback_runs": 20, "place_positions": 3}


def test_trailing_rate_is_leak_safe_and_groups_by_key():
    df = pd.DataFrame({
        "k": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = engine._trailing_rate(
        df, ["k"], "win_rate", "place_rate", "runs", CFG).sort_values("race_date")
    # first row has no prior runs -> null
    assert pd.isna(out.iloc[0]["win_rate"])
    # third row: two prior runs (one win) -> 0.5, never sees its own result
    assert out.iloc[2]["win_rate"] == 0.5
    assert out.iloc[2]["runs"] == 2


def test_trailing_rate_excludes_same_date_runs():
    # Two runs on the SAME date (both with positions). The later-sorted same-date
    # row must NOT count its same-date sibling -> NULL trailing rate.
    df = pd.DataFrame({
        "k": ["A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "position": [1, 2],
    })
    out = engine._trailing_rate(df, ["k"], "win_rate", "place_rate", "runs", CFG)
    # neither same-date row may see the other -> both NULL, 0 runs
    assert out["win_rate"].isna().all()
    assert (out["runs"] == 0).all()


def test_trailing_rate_predicate_filters_prior_runs():
    df = pd.DataFrame({
        "k": ["A", "A", "A"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 5, 1],
        "going_speed": [3, 2, 3],
    })
    # predicate: only prior runs whose going_speed equals the current row's
    out = engine._trailing_rate(
        df, ["k"], "win_rate", "place_rate", "runs", CFG,
        predicate=lambda prior, cur: prior["going_speed"] == cur["going_speed"]
    ).sort_values("race_date")
    # 3rd row going_speed=3; only the 2026-01-01 run (going_speed=3, pos 1) qualifies
    assert out.iloc[2]["win_rate"] == 1.0
    assert out.iloc[2]["runs"] == 1


def test_trailing_rate_null_when_no_position_column():
    df = pd.DataFrame({"k": ["A"], "race_date": pd.to_datetime(["2026-01-01"], utc=True)})
    out = engine._trailing_rate(df, ["k"], "win_rate", "place_rate", "runs", CFG)
    assert pd.isna(out.iloc[0]["win_rate"])
    assert out.iloc[0]["runs"] == 0


def test_combo_win_rate_keys_on_jockey_trainer_pair():
    df = pd.DataFrame({
        "jockey_id": ["J", "J", "J"],
        "trainer_id": ["T", "T", "X"],   # third row is a DIFFERENT pairing
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "position": [1, 2, 1],
    })
    out = engine.add_combo_win_rate(df, CFG).sort_values(["trainer_id", "race_date"])
    jt = out[(out["jockey_id"] == "J") & (out["trainer_id"] == "T")].sort_values("race_date")
    # second J/T run: one prior win for the pair -> 1.0 over 1 run
    assert jt.iloc[1]["jt_combo_win_rate"] == 1.0
    assert jt.iloc[1]["jt_combo_runs"] == 1
    # the J/X pairing has no prior runs -> null
    jx = out[(out["jockey_id"] == "J") & (out["trainer_id"] == "X")].iloc[0]
    assert pd.isna(jx["jt_combo_win_rate"])
    assert "_jt_combo_place" not in out.columns


def test_going_preference_uses_only_matching_going_band():
    df = pd.DataFrame({
        "horse_id": ["H", "H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "going_speed": [3, 2, 3],   # today's run (3rd) is fast going, like run 1
        "position": [1, 4, 1],
    })
    out = engine.add_going_preference(df, CFG).sort_values("race_date")
    # 3rd row: only the going_speed=3 prior run (pos 1) counts -> win rate 1.0
    assert out.iloc[2]["going_pref_win_rate"] == 1.0
    assert out.iloc[2]["going_pref_place_rate"] == 1.0
    assert "_going_pref_runs" not in out.columns


def test_going_preference_null_when_today_going_unknown():
    df = pd.DataFrame({
        "horse_id": ["H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
        "going_speed": [3, None],
        "position": [1, None],
    })
    out = engine.add_going_preference(df, CFG).sort_values("race_date")
    assert pd.isna(out.iloc[1]["going_pref_win_rate"])


def test_going_preference_null_when_going_speed_column_absent():
    # No going_speed column at all → going-band predicate can't be evaluated,
    # so both rates fall back to null rather than raising KeyError.
    df = pd.DataFrame({
        "horse_id": ["H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
        "position": [1, 2],
    })
    out = engine.add_going_preference(df, CFG)
    assert out["going_pref_win_rate"].isna().all()
    assert out["going_pref_place_rate"].isna().all()


def test_going_preference_prefers_going_band_over_going_speed():
    # going_band present -> grouping keys on it (raw-string band), not going_speed.
    df = pd.DataFrame({
        "horse_id": ["H", "H", "H"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "going_band": ["soft", "good", "soft"],
        "going_speed": [None, None, None],   # would give null if it were used
        "position": [1, 4, 1],
    })
    out = engine.add_going_preference(df, CFG).sort_values("race_date")
    # 3rd row on soft: only the prior soft run (pos 1) counts -> win rate 1.0
    assert out.iloc[2]["going_pref_win_rate"] == 1.0


def test_course_suitability_is_leak_safe_per_venue():
    df = pd.DataFrame({
        "horse_id": ["H", "H", "H"],
        "venue": ["Ascot", "York", "Ascot"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 5, 2],
    })
    out = engine.add_course_suitability(df, CFG).sort_values("race_date")
    # 3rd run at Ascot: only the prior Ascot run (pos 1) counts -> win 1.0, 1 run
    assert out.iloc[2]["course_win_rate"] == 1.0
    assert out.iloc[2]["course_runs"] == 1
    assert pd.isna(out.iloc[0]["course_win_rate"])    # first ever run -> null


def test_distance_suitability_pools_within_band():
    # 6f (sprint) then 16f (staying) then 5f (sprint): the 5f run only sees the 6f.
    df = pd.DataFrame({
        "horse_id": ["H", "H", "H"],
        "distance_furlongs": [6.0, 16.0, 5.0],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-03-01"], utc=True),
        "position": [1, 8, 3],
    })
    out = engine.add_distance_suitability(df, CFG).sort_values("race_date")
    assert out.iloc[2]["distance_runs"] == 1            # only the prior sprint run
    assert out.iloc[2]["distance_win_rate"] == 1.0
    assert "_dist_band" not in out.columns


def test_speed_trend_positive_when_recent_form_improves():
    # Four prior runs improving over time (poor -> winning). The short 3-run mean
    # (recent) then exceeds the longer all-runs baseline -> positive trend. Needs
    # >3 priors so the short and long windows actually cover different run sets.
    df = pd.DataFrame({
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01"],
            utc=True),
        "venue": list("VWXYZ"), "horse_id": ["H"] * 5,
        "timeform_rating": [None] * 5,
        "position": [10, 9, 1, 1, None], "field_size": [10] * 5,
    })
    out = engine.add_speed_figures(df, CFG).sort_values("race_date")
    assert "speed_trend" in out.columns
    assert out.iloc[4]["speed_trend"] > 0


def test_speed_figures_prefer_tfr_when_present():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13", "2026-06-13"], utc=True),
        "venue": ["X", "X"], "horse_id": ["a", "b"],
        "timeform_rating": [120, 100], "position": [None, None], "field_size": [2, 2],
    })
    out = engine.add_speed_figures(df, CFG).set_index("horse_id")
    assert out.loc["a", "horse_speed"] == 120
    assert out.loc["a", "horse_speed_rank"] == 1  # higher figure -> rank 1


def test_speed_figures_fall_back_to_leak_safe_proxy():
    # No TFR. Horse H finished 1st of 4 then races today; proxy = trailing mean of
    # the prior run's field-relative finish percentile = (4-1)/(4-1)*100 = 100.
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01", "2026-06-13"], utc=True),
        "venue": ["X", "Y"], "horse_id": ["H", "H"],
        "timeform_rating": [None, None], "position": [1, None], "field_size": [4, 6],
    })
    out = engine.add_speed_figures(df, CFG).sort_values("race_date")
    # first run has no prior -> null; today's run sees the prior 1st-of-4 -> 100
    assert pd.isna(out.iloc[0]["horse_speed"])
    assert out.iloc[1]["horse_speed"] == 100.0


def test_speed_figures_excludes_same_date_prior_runs():
    # Two runs dated the SAME day, no TFR. The same-date prior run must NOT
    # produce a horse_speed proxy for a row dated that same day.
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13", "2026-06-13"], utc=True),
        "venue": ["X", "Y"], "horse_id": ["H", "H"],
        "timeform_rating": [None, None], "position": [1, None], "field_size": [4, 6],
    })
    out = engine.add_speed_figures(df, CFG).sort_values("venue")
    # row dated same day as the only prior run -> no proxy -> NULL horse_speed
    assert out["horse_speed"].isna().all()


def test_pace_bias_passes_through_pace_rating():
    df = pd.DataFrame({"pace_rating": [55, None]})
    out = engine.add_pace_bias(df)
    assert out.iloc[0]["pace_bias"] == 55
    assert pd.isna(out.iloc[1]["pace_bias"])


def test_pace_bias_null_when_column_absent():
    df = pd.DataFrame({"horse_id": ["a"]})
    out = engine.add_pace_bias(df)
    assert pd.isna(out.iloc[0]["pace_bias"])


def test_ew_value_index_matches_hand_computation():
    # odds=5.0, fair win prob=0.25, ew_places=3, reduction=0.25
    # place prob proxy = min(1, 0.25*3) = 0.75
    # place_odds = 1 + (5-1)*0.25 = 2.0
    # win_ev = 0.25*5 = 1.25 ; place_ev = 0.75*2.0 = 1.5 ; index = 1.25+1.5-2 = 0.75
    df = pd.DataFrame({
        "odds_decimal": [5.0], "sp": [None],
        "ew_places": [3], "ew_reduction": [0.25],
        "overround_norm_prob": [0.25],
    })
    out = engine.add_ew_value_index(df)
    assert abs(out.iloc[0]["ew_value_index"] - 0.75) < 1e-9


def test_ew_value_index_null_without_terms_or_prob():
    df = pd.DataFrame({
        "odds_decimal": [5.0], "sp": [None],
        "ew_places": [None], "ew_reduction": [None], "overround_norm_prob": [0.25],
    })
    out = engine.add_ew_value_index(df)
    assert pd.isna(out.iloc[0]["ew_value_index"])


def test_odds_delta_drift_and_value():
    df = pd.DataFrame({
        "morningwap": [6.0, None],
        "sp": [4.0, 4.0],
        "implied_prob": [0.25, 0.10],
        "overround_norm_prob": [0.20, 0.10],
    })
    out = engine.add_odds_delta(df)
    # drift = (6-4)/6 = 0.3333...
    assert abs(out.iloc[0]["odds_drift"] - (2.0 / 6.0)) < 1e-9
    # second row has no morningwap -> drift null
    assert pd.isna(out.iloc[1]["odds_drift"])
    # value delta = implied_prob - overround_norm_prob
    assert abs(out.iloc[0]["odds_value_delta"] - 0.05) < 1e-9
    assert abs(out.iloc[1]["odds_value_delta"] - 0.0) < 1e-9


def test_race_complexity_broadcasts_per_race_and_skips_null_components():
    # Two races. timeform_rating all-null (component skipped gracefully).
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 5, utc=True),
        "venue": ["A", "A", "A", "B", "B"],
        "horse_id": ["a1", "a2", "a3", "b1", "b2"],
        "overround_norm_prob": [0.34, 0.33, 0.33, 0.80, 0.20],
        "race_class": [3, 5, 4, 3, 3],
        "recent_form_avg": [2.0, 5.0, 3.0, 1.0, 1.0],
        "timeform_rating": [None, None, None, None, None],
    })
    out = engine.add_race_complexity(df)
    # every runner in race A shares one complexity value
    a_vals = out[out["venue"] == "A"]["race_complexity"].unique()
    assert len(a_vals) == 1
    # race A (3 runners, even market, wide class spread) is more complex than race B
    a = out[out["venue"] == "A"]["race_complexity"].iloc[0]
    b = out[out["venue"] == "B"]["race_complexity"].iloc[0]
    assert a > b


def test_race_complexity_skips_present_but_all_null_component():
    # A present-but-all-null component (race_class) must be skipped entirely,
    # yielding the SAME race_complexity as omitting the column outright.
    base = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 5, utc=True),
        "venue": ["A", "A", "A", "B", "B"],
        "horse_id": ["a1", "a2", "a3", "b1", "b2"],
        "overround_norm_prob": [0.34, 0.33, 0.33, 0.80, 0.20],
    })
    with_null = base.copy()
    with_null["race_class"] = [None, None, None, None, None]
    omitted = base.copy()

    out_null = engine.add_race_complexity(with_null).sort_values("horse_id")
    out_omit = engine.add_race_complexity(omitted).sort_values("horse_id")
    pd.testing.assert_series_equal(
        out_null["race_complexity"].reset_index(drop=True),
        out_omit["race_complexity"].reset_index(drop=True),
    )


def test_race_complexity_all_na_object_column_does_not_crash():
    # Regression: the normalizer fills absent columns with pd.NA (NAType), not
    # Python None. A grouped std over an object/NAType column raised
    # "float() argument must be ... not 'NAType'" in pandas' cython path. The
    # spread components must coerce to numeric first. betsp supplies no
    # race_class, so this is the real production shape.
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 4, utc=True),
        "venue": ["A", "A", "B", "B"],
        "horse_id": ["a1", "a2", "b1", "b2"],
        "overround_norm_prob": [0.55, 0.45, 0.6, 0.4],
        "race_class": pd.array([pd.NA, pd.NA, pd.NA, pd.NA], dtype="object"),
    })
    out = engine.add_race_complexity(df)  # must not raise
    assert "race_complexity" in out.columns
    assert len(out) == 4


def test_engine_feature_columns_are_numeric_dtype():
    # ew_value_index, odds_drift, pace_bias must end up numeric (not object),
    # even when some rows are null. Downstream model code wants numeric dtype.
    ew_df = pd.DataFrame({
        "odds_decimal": [5.0, 5.0], "sp": [None, None],
        "ew_places": [3, None], "ew_reduction": [0.25, None],
        "overround_norm_prob": [0.25, 0.25],
    })
    ew_out = engine.add_ew_value_index(ew_df)
    assert pd.api.types.is_numeric_dtype(ew_out["ew_value_index"])
    assert abs(ew_out.iloc[0]["ew_value_index"] - 0.75) < 1e-9
    assert pd.isna(ew_out.iloc[1]["ew_value_index"])

    odds_df = pd.DataFrame({
        "morningwap": [6.0, None],
        "sp": [4.0, 4.0],
        "implied_prob": [0.25, 0.10],
        "overround_norm_prob": [0.20, 0.10],
    })
    odds_out = engine.add_odds_delta(odds_df)
    assert pd.api.types.is_numeric_dtype(odds_out["odds_drift"])
    assert abs(odds_out.iloc[0]["odds_drift"] - (2.0 / 6.0)) < 1e-9
    assert pd.isna(odds_out.iloc[1]["odds_drift"])

    pace_df = pd.DataFrame({"pace_rating": [55, None]})
    pace_out = engine.add_pace_bias(pace_df)
    assert pd.api.types.is_numeric_dtype(pace_out["pace_bias"])
    assert pace_out.iloc[0]["pace_bias"] == 55
    assert pd.isna(pace_out.iloc[1]["pace_bias"])


# --- race_complexity_v2 / race_market_entropy (step 04 / DECISIONS D27) ---
# race_complexity (legacy, above) is left byte-identical for existing model
# bundles; these are the corrected, additive replacements.

def _rc2_df():
    return pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 5, utc=True),
        "venue": ["A", "A", "A", "B", "B"],
        "horse_id": ["a1", "a2", "a3", "b1", "b2"],
        "overround_norm_prob": [0.34, 0.33, 0.33, 0.80, 0.20],
        "race_class": [3, 5, 4, 3, 3],
        "recent_form_avg": [2.0, 5.0, 3.0, 1.0, 1.0],
        "timeform_rating": [None] * 5,
    })


def test_race_complexity_v2_has_no_market_input():
    """Unlike the legacy race_complexity, v2 must not read overround_norm_prob
    (or any other market/odds column) at all — verified by mutating the market
    column and checking the output is unchanged."""
    df = _rc2_df()
    scale = engine.fit_race_complexity_v2_scale(df)
    out1 = engine.add_race_complexity_v2(df, scale=scale)

    df2 = df.copy()
    df2["overround_norm_prob"] = [0.99, 0.001, 0.001, 0.5, 0.5]  # wildly different book
    out2 = engine.add_race_complexity_v2(df2, scale=scale)

    pd.testing.assert_series_equal(
        out1["race_complexity_v2"], out2["race_complexity_v2"])


def test_race_complexity_v2_uses_frozen_scale_not_call_frame():
    """The legacy race_complexity's defect: standardizing against the mean/std
    of whatever frame is passed in makes an admissible row's value depend on
    which OTHER rows share the call. v2 must give the SAME value for the SAME
    race regardless of what else is in the call frame, given a fixed scale."""
    df = _rc2_df()
    scale = engine.fit_race_complexity_v2_scale(df)

    race_a_only = df[df["venue"] == "A"].reset_index(drop=True)
    out_full = engine.add_race_complexity_v2(df, scale=scale)
    out_solo = engine.add_race_complexity_v2(race_a_only, scale=scale)

    full_a = out_full[out_full["venue"] == "A"]["race_complexity_v2"].reset_index(drop=True)
    solo_a = out_solo["race_complexity_v2"].reset_index(drop=True)
    pd.testing.assert_series_equal(full_a, solo_a)


def test_race_complexity_v2_missing_scale_degrades_to_null_not_crash(tmp_path, monkeypatch):
    from features import _feature_scale
    monkeypatch.setattr(_feature_scale, "DEFAULT_SCALE_PATH",
                        str(tmp_path / "does_not_exist.json"))
    df = _rc2_df()
    out = engine.add_race_complexity_v2(df)  # scale=None -> loads (missing) default
    assert "race_complexity_v2" in out.columns
    assert out["race_complexity_v2"].isna().all()


def test_race_complexity_v2_reuses_field_size_column_surviving_a_later_filter():
    """field_size (features/derive.py::add_odds_features) is computed ONCE,
    over the FULL per-race row set, before features/builder.py's later
    position.notna() label filter can drop a non-runner/unresolved result.
    _race_complexity_v2_raw_components must REUSE that column rather than
    recounting rows-per-race itself — recounting after such a filter already
    ran would silently undercount every survivor's field size (found this
    stage: an earlier evidence-script shortcut recomputed post-filter and hit
    exactly this)."""
    full = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 4 + ["2026-06-14"] * 4, utc=True),
        "venue": ["A"] * 4 + ["B"] * 4,
        "horse_id": ["a1", "a2", "a3", "a4", "b1", "b2", "b3", "b4"],
        # field_size as add_odds_features would have set it: the TRUE field,
        # unaffected by which rows later survive a label filter.
        "field_size": [4, 4, 4, 4, 4, 4, 4, 4],
        # race A: a4 is a non-runner/unresolved result -> true field size stays 4.
        "position": [1.0, 2.0, 3.0, None, 1.0, 2.0, 3.0, 4.0],
    })
    labelled_only = full[full["position"].notna()].reset_index(drop=True)

    scale = engine.fit_race_complexity_v2_scale(full)
    assert scale["field_size"]["mean"] == 4.0

    out_full = engine.add_race_complexity_v2(full, scale=scale)
    out_filtered = engine.add_race_complexity_v2(labelled_only, scale=scale)
    # Same scale, same real races -> race A's 3 labelled runners must see the
    # SAME true field size (4) whether or not its 4th (unlabelled) row is
    # present in the call frame.
    pd.testing.assert_series_equal(
        out_full.loc[out_full["position"].notna(), "race_complexity_v2"].reset_index(drop=True),
        out_filtered["race_complexity_v2"].reset_index(drop=True))


def test_fit_race_complexity_v2_scale_uses_one_row_per_race():
    """A 3-runner race must not out-weight a 2-runner race 3x in the fit just
    because it has more rows — the fit collapses to one row per race first."""
    df = _rc2_df()
    scale = engine.fit_race_complexity_v2_scale(df)
    assert scale["field_size"]["n"] == 2   # 2 races, not 5 runner-rows


def test_race_market_entropy_matches_shannon_entropy_of_book():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 2, utc=True),
        "venue": ["A", "A"], "horse_id": ["a1", "a2"],
        "overround_norm_prob": [0.5, 0.5],
    })
    out = engine.add_race_market_entropy(df)
    expected = -(0.5 * math.log(0.5) * 2)
    assert abs(out["race_market_entropy"].iloc[0] - expected) < 1e-9


def test_race_market_entropy_null_without_market_column():
    df = pd.DataFrame({"horse_id": ["a1"]})
    out = engine.add_race_market_entropy(df)
    assert out["race_market_entropy"].isna().all()


# --- trainer/jockey form: market-sibling must not leak into its own rate ---

def test_trainer_form_market_sibling_rows_do_not_leak_into_each_other():
    from features._trainer_form import add_trainer_form
    df = pd.DataFrame({
        "trainer_id": ["T", "T", "T", "T"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-01-05", "2026-01-05"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    })
    out = add_trainer_form(df, {"min_runners_in_window": 1})
    r1 = out[out["race_uid"] == "R1"]
    r2 = out[out["race_uid"] == "R2"]
    # R1's two rows must NOT see each other -> null (no prior race exists yet)
    assert r1["trainer_hot_strike_rate"].isna().all()
    # R2's two rows both see R1's single win once -> 1.0, and agree with each other
    assert (r2["trainer_hot_strike_rate"] == 1.0).all()


def test_jockey_form_market_sibling_rows_do_not_leak_into_each_other():
    from features._jockey_form import add_jockey_form
    df = pd.DataFrame({
        "jockey_id": ["J", "J", "J", "J"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-01-05", "2026-01-05"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    })
    out = add_jockey_form(df, {"min_runners_in_window": 1})
    r1 = out[out["race_uid"] == "R1"]
    r2 = out[out["race_uid"] == "R2"]
    assert r1["jockey_hot_strike_rate"].isna().all()
    assert (r2["jockey_hot_strike_rate"] == 1.0).all()


def test_race_complexity_single_race_does_not_crash():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-06-13"] * 2, utc=True),
        "venue": ["A", "A"], "horse_id": ["a1", "a2"],
        "overround_norm_prob": [0.6, 0.4], "race_class": [3, 4],
        "recent_form_avg": [2.0, 3.0], "timeform_rating": [None, None],
    })
    out = engine.add_race_complexity(df)
    assert "race_complexity" in out.columns
    assert out["race_complexity"].notna().all()
