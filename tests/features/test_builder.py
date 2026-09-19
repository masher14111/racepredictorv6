import pandas as pd
import pytest

from features.builder import build_training_matrix, build_inference_matrix
from utils.timezone import now

# Anchor the fixture to wall-clock so live rows are always "upcoming" relative to
# now(): inference counts a row as live only when race_date >= today (see
# builder._live_mask), so a hardcoded past date would make the live rows vanish.
_TODAY = now().date().isoformat()
_PAST_1 = (now() - pd.DateOffset(months=5)).date().isoformat()
_PAST_2 = (now() - pd.DateOffset(months=4)).date().isoformat()


def _hist_row(horse, date, pos, venue="Ascot", source="betsp", **kw):
    base = {
        "race_date": date, "venue": venue, "horse_id": horse, "horse_name": horse,
        "jockey_id": "J", "trainer_id": "T", "source": source, "position": pos,
        "odds_decimal": 4.0, "sp": 4.0, "odds_finish": 4.0, "going": "Good",
        "distance": "1m", "timeform_rating": 100, "race_class": 3,
        "recent_form": "11", "market_type": "WIN",
    }
    base.update(kw)
    return base


def _unified():
    # Horse h1 ran twice (history) and races again today (live, no position).
    return pd.DataFrame([
        _hist_row("h1", _PAST_1, 1),
        _hist_row("h1", _PAST_2, 2),
        _hist_row("h2", _PAST_1, 3),
        # today's live race
        _hist_row("h1", _TODAY, None, source="boylesports",
                  position=None, odds_finish=None),
        _hist_row("h2", _TODAY, None, source="boylesports",
                  position=None, odds_finish=None),
    ])


def test_training_matrix_is_labelled_and_drops_null_positions(tmp_path):
    out_path = tmp_path / "training.parquet"
    df = build_training_matrix(unified=_unified(), write=True, output_path=str(out_path))
    # only the 3 historical rows (with positions) survive
    assert len(df) == 3
    assert df["position"].notna().all()
    assert "won" in df.columns and "placed" in df.columns
    assert out_path.exists()
    assert len(pd.read_parquet(out_path)) == 3


def test_inference_matrix_is_live_only_and_unlabelled():
    df = build_inference_matrix(unified=_unified())
    # only today's 2 live runners
    assert len(df) == 2
    assert df["position"].isna().all()
    assert "won" not in df.columns
    # h1 had a prior win+place in history; trailing rate must be computed (not null)
    h1 = df.set_index("horse_id").loc["h1"]
    assert h1["historical_win_rate"] is not None


def test_train_and_inference_share_feature_columns():
    train = build_training_matrix(unified=_unified(), write=False)
    infer = build_inference_matrix(unified=_unified())
    feature_cols = lambda d: set(d.columns) - {"won", "placed"}
    assert feature_cols(train) == feature_cols(infer)
    # the new engine columns exist on BOTH paths (train/serve parity)
    for col in ["horse_speed", "horse_speed_rank", "jt_combo_win_rate",
                "jt_combo_runs", "going_pref_win_rate", "going_pref_place_rate",
                "pace_bias", "ew_value_index", "odds_drift", "odds_value_delta",
                "race_complexity"]:
        assert col in feature_cols(train)
        assert col in feature_cols(infer)


def test_inference_matrix_returns_live_rows_when_upcoming_exist():
    """Task 09/10 regression: when upcoming races exist, the inference matrix must
    return exactly those live runners (non-empty), carrying their odds-derived
    features."""
    df = build_inference_matrix(unified=_unified())
    assert not df.empty
    assert len(df) == 2
    # live odds attached → implied_prob derived for every live runner
    assert df["implied_prob"].notna().all()
    assert set(df["horse_id"]) == {"h1", "h2"}


def test_inference_matrix_empty_when_only_past_null_position_rows():
    """Task 08 regression: historical rows with null position (results that never
    joined) are NOT 'live'. Keying inference off position.isna() alone scored
    tens of thousands of phantom races — the date guard must drop them."""
    past_join_misses = pd.DataFrame([
        _hist_row("h1", _PAST_1, None, source="boylesports",
                  position=None, odds_finish=None),
        _hist_row("h2", _PAST_2, None, source="boylesports",
                  position=None, odds_finish=None),
    ])
    df = build_inference_matrix(unified=past_join_misses)
    assert df.empty


def test_inference_counts_todays_already_run_races_as_history():
    """Step 09 audit F1: the training matrix counts an earlier-that-day result
    as a prior (the cut is the real off-time), so the serving path has to see
    the same history — it used to hard-exclude every same-day row, which made
    the identical feature mean two different things in training and live."""
    rows = [
        _hist_row("hA", _TODAY, 1, race_time=f"{_TODAY}T13:30"),
        _hist_row("hB", _TODAY, 4, race_time=f"{_TODAY}T15:00"),
        _hist_row("hC", _TODAY, None, position=None, odds_finish=None,
                  race_time=f"{_TODAY}T20:20"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert list(df["horse_id"]) == ["hC"]
    # trainer T has run twice already today (one win) before the 20:20
    assert df["trainer_win_rate"].iloc[0] == pytest.approx(0.5)


def test_inference_ignores_a_later_race_today_that_has_not_run():
    """The same-day admission above is gated on a KNOWN result, so a race that
    goes off after the one being scored can never enter its history."""
    rows = [
        _hist_row("hA", _TODAY, None, position=None, odds_finish=None,
                  race_time=f"{_TODAY}T20:20"),
        _hist_row("hC", _TODAY, None, position=None, odds_finish=None,
                  race_time=f"{_TODAY}T13:30"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert sorted(df["horse_id"]) == ["hA", "hC"]
    assert df["trainer_win_rate"].isna().all()


# ── Stage 20 (B5 / GAP-C): live off-time source disagreement reconciliation ──

def test_live_off_time_disagreement_is_reconciled_to_one_race():
    """Two odds sources reporting the SAME physical race a minute apart used
    to fragment into two race_uid values downstream — reproduced live at
    Dundalk (19:30 vs 19:31). build_inference_matrix must resolve them to one
    off-time (via derive.add_race_key's downstream race_uid) when the runner
    names genuinely overlap between the two source rows."""
    rows = [
        _hist_row("h1", _TODAY, None, position=None, odds_finish=None,
                  source="betsp", race_time=f"{_TODAY}T19:30:00+00:00"),
        _hist_row("h2", _TODAY, None, position=None, odds_finish=None,
                  source="betsp", race_time=f"{_TODAY}T19:30:00+00:00"),
        _hist_row("h1", _TODAY, None, position=None, odds_finish=None,
                  source="boylesports", race_time=f"{_TODAY}T19:31:00+00:00"),
        _hist_row("h2", _TODAY, None, position=None, odds_finish=None,
                  source="boylesports", race_time=f"{_TODAY}T19:31:00+00:00"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert df["race_uid"].nunique() == 1
    assert sorted(df["horse_id"]) == ["h1", "h2"]


def test_adjacent_distinct_races_at_the_same_venue_stay_separate():
    """Two genuinely different races at one venue, 35 minutes apart with
    disjoint fields, must never be merged by the same reconciliation."""
    rows = [
        _hist_row("hA", _TODAY, None, position=None, odds_finish=None,
                  source="betsp", race_time=f"{_TODAY}T15:40:00+00:00"),
        _hist_row("hB", _TODAY, None, position=None, odds_finish=None,
                  source="betsp", race_time=f"{_TODAY}T16:15:00+00:00"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert df["race_uid"].nunique() == 2


def test_same_off_time_different_venues_stay_distinct_races():
    """Same instant, different venues (a real, reproduced event: Leopardstown
    and York both went off at 16:15 on 2024-05-17) must never collapse into
    one race_uid."""
    rows = [
        _hist_row("hA", _TODAY, None, position=None, odds_finish=None,
                  venue="Leopardstown", source="betsp",
                  race_time=f"{_TODAY}T16:15:00+00:00"),
        _hist_row("hB", _TODAY, None, position=None, odds_finish=None,
                  venue="York", source="betsp",
                  race_time=f"{_TODAY}T16:15:00+00:00"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert df["race_uid"].nunique() == 2
    assert set(df["venue"]) == {"Leopardstown", "York"}


def test_ambiguous_close_off_time_with_no_runner_overlap_is_quarantined(tmp_path, monkeypatch):
    """Close off-times at one venue with NO confirmed shared runner must be
    excluded from the live matrix rather than guessed either way."""
    import features.builder as builder_mod

    quarantine_path = tmp_path / "identity_quarantine.json"
    monkeypatch.setattr(builder_mod, "_IDENTITY_QUARANTINE_PATH", str(quarantine_path))

    rows = [
        _hist_row("x1", _TODAY, None, position=None, odds_finish=None,
                  source="betsp", race_time=f"{_TODAY}T19:30:00+00:00"),
        _hist_row("y1", _TODAY, None, position=None, odds_finish=None,
                  source="boylesports", race_time=f"{_TODAY}T19:31:30+00:00"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    assert df.empty
    assert quarantine_path.exists()


def _live_only_row(horse, venue="Ascot", **kw):
    """A live runner the way a bookmaker odds feed emits it: horse + price only, no
    jockey/trainer/going/distance/ratings, null position."""
    base = {
        "race_date": _TODAY, "venue": venue, "horse_id": horse, "horse_name": horse,
        "source": "boylesports", "position": None, "odds_decimal": 6.0,
        "market_type": "WIN",
    }
    base.update(kw)
    return base


def test_horse_speed_uses_true_field_size_after_pruning():
    """Task 29: the horse-speed proxy is a finish percentile whose denominator is the
    race's field size. build_inference_matrix prunes history to the live horses' own
    rows, so a 6-runner historical race collapses to the 2 live horses — a field_size
    recomputed over that pruned frame (=2) makes the percentile negative. The
    true-field-size precompute must keep it on the trained 0..100 scale."""
    field = {"h1": 5, "h2": 6, "o1": 1, "o2": 2, "o3": 3, "o4": 4}
    rows = [_hist_row(h, _PAST_1, pos, venue="Big") for h, pos in field.items()]
    rows += [_live_only_row("h1", venue="Big"), _live_only_row("h2", venue="Big")]
    df = build_inference_matrix(unified=pd.DataFrame(rows))

    hs = df.set_index("horse_id")["horse_speed"]
    # h1 finished 5th of 6 → percentile (6-5)/(6-1)*100 = 20 (NOT -300, which the
    # pruned 2-runner field_size would have produced).
    assert hs["h1"] == pytest.approx(20.0, abs=1e-6)
    assert hs["h2"] == pytest.approx(0.0, abs=1e-6)
    assert (df["horse_speed"].dropna().between(0, 100)).all()


def test_live_connections_filled_from_history():
    """Task 29: live odds feeds carry no jockey/trainer, so the jockey/trainer/jt
    trailing features are dead for every live runner. The last-known connections are
    carried forward from the horse's own history so those features can compute."""
    rows = [
        _hist_row("h1", _PAST_1, 1, jockey_id="JK", trainer_id="TR"),
        _hist_row("h1", _PAST_2, 2, jockey_id="JK", trainer_id="TR"),
        _live_only_row("h1"),  # no jockey/trainer on the live row
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    h1 = df.set_index("horse_id").loc["h1"]
    assert h1["jockey_id"] == "JK"
    assert h1["trainer_id"] == "TR"
    # connections carried forward → their trailing win rates are now defined
    assert pd.notna(h1["jockey_win_rate"])
    assert pd.notna(h1["trainer_win_rate"])


def test_live_connections_source_flags_declared_vs_fallback_vs_missing():
    """Stage 05: today's declared jockey/trainer must not be indistinguishable
    from an invented last-known guess. h1 has history -> its live row's missing
    jockey/trainer gets a "historical_fallback" flag; h2 has no history at all
    -> "missing"; h3's live row already carries its own declared jockey/trainer
    (a racecard source, e.g. Timeform, reported it) -> "declared" and the fill
    must not override it."""
    rows = [
        _hist_row("h1", _PAST_1, 1, jockey_id="JK", jockey_name="J K",
                  trainer_id="TR", trainer_name="T R"),
        _live_only_row("h1"),
        _live_only_row("h2"),  # never raced before -> no fallback available
        _live_only_row("h3", jockey_id="DECLARED_J", jockey_name="Declared Jockey",
                       trainer_id="DECLARED_T", trainer_name="Declared Trainer"),
    ]
    df = build_inference_matrix(unified=pd.DataFrame(rows))
    by_horse = df.set_index("horse_id")

    assert by_horse.loc["h1", "jockey_connections_source"] == "historical_fallback"
    assert by_horse.loc["h1", "jockey_id"] == "JK"
    assert by_horse.loc["h1", "trainer_connections_source"] == "historical_fallback"

    assert by_horse.loc["h2", "jockey_connections_source"] == "missing"
    assert pd.isna(by_horse.loc["h2", "jockey_id"])

    assert by_horse.loc["h3", "jockey_connections_source"] == "declared"
    assert by_horse.loc["h3", "jockey_id"] == "DECLARED_J"
    assert by_horse.loc["h3", "trainer_connections_source"] == "declared"


def test_training_matrix_connections_are_declared_not_fallback():
    """Historical (labelled) rows are real recorded results, never a last-known
    guess, and train/infer must share the same feature-column schema."""
    df = build_training_matrix(unified=_unified(), write=False)
    assert (df["jockey_connections_source"] == "declared").all()
    assert (df["trainer_connections_source"] == "declared").all()


def test_builder_emits_full_feature_matrix_and_new_columns(tmp_path):
    out_path = tmp_path / "training.parquet"
    features_path = tmp_path / "features.parquet"
    build_training_matrix(
        unified=_unified(), write=True,
        output_path=str(out_path), features_path=str(features_path))
    full = pd.read_parquet(features_path)
    # full matrix keeps ALL rows (3 history + 2 live), unlike training (history only)
    assert len(full) == 5
    # engine columns are present on the full matrix
    for col in ["horse_speed", "jt_combo_win_rate", "going_pref_win_rate",
                "pace_bias", "ew_value_index", "odds_drift", "odds_value_delta",
                "race_complexity"]:
        assert col in full.columns
