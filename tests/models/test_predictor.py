"""Tests for models/predictor.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.predictor import (
    Predictor,
    RacePrediction,
    RunnerPrediction,
    _attach_odds_by_book,
    _best_of_books,
    _decimal_odds,
    _effective_decimal,
    _odds_by_book_map,
    _prob,
    _runner_dict,
    _str,
    _write_cache,
    check_output_invariants,
    predict as module_predict,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _live_df(n_races: int = 2, runners_per_race: int = 5, seed: int = 0) -> pd.DataFrame:
    """Synthetic inference matrix: no position column, all FEATURE_COLS present.

    Dates are pinned to *tomorrow* so the rows are genuinely upcoming and survive
    Predictor._guard_upcoming (the historical-join-miss guard), regardless of the
    UTC/Dublin offset on the day the suite runs."""
    rng = np.random.default_rng(seed)
    rows = []
    base = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=1, hours=13)
    for r in range(n_races):
        race_date = base + pd.Timedelta(hours=r)
        venue = f"Venue{r}"
        for i in range(runners_per_race):
            row = {col: rng.random() for col in FEATURE_COLS}
            row["implied_prob"] = rng.uniform(0.05, 0.50)
            row["race_date"] = race_date
            row["venue"] = venue
            row["horse_id"] = f"h{r}_{i}"
            row["horse_name"] = f"Horse {r} {i}"
            row["jockey"] = f"Jockey{i}"
            row["trainer"] = f"Trainer{i}"
            row["position"] = pd.NA
            rows.append(row)
    return pd.DataFrame(rows)


def _mock_catboost(prob_value: float = 0.20) -> MagicMock:
    """A CatBoostClassifier mock whose predict_proba returns a fixed probability."""
    m = MagicMock()
    m.predict_proba.side_effect = lambda X: np.column_stack(
        [1 - np.full(len(X), prob_value), np.full(len(X), prob_value)]
    )
    return m


def _predictor_with_mocks(
    live: pd.DataFrame,
    won_prob: float = 0.18,
    placed_prob: float = 0.40,
    showed_prob: float = 0.60,
    min_odds: float = 2.50,
) -> tuple[Predictor, dict]:
    """Return a Predictor with injected mocks; skip disk I/O entirely."""
    p = Predictor(min_selection_odds=min_odds)
    p._models = {
        "won": _mock_catboost(won_prob),
        "placed_2": _mock_catboost(placed_prob),
        "showed": _mock_catboost(showed_prob),
    }
    return p, p._models


# ── _decimal_odds ─────────────────────────────────────────────────────────────

class TestDecimalOdds:
    def test_from_implied_prob(self):
        row = pd.Series({"implied_prob": 0.25})
        assert _decimal_odds(row) == pytest.approx(4.0, rel=1e-3)

    def test_from_decimal_odds_column(self):
        row = pd.Series({"implied_prob": None, "decimal_odds": 5.0})
        assert _decimal_odds(row) == pytest.approx(5.0)

    def test_implied_prob_takes_priority(self):
        row = pd.Series({"implied_prob": 0.5, "decimal_odds": 10.0})
        assert _decimal_odds(row) == pytest.approx(2.0)

    def test_nan_when_neither_available(self):
        row = pd.Series({"implied_prob": None, "decimal_odds": None})
        assert pd.isna(_decimal_odds(row))

    def test_nan_for_zero_implied_prob(self):
        row = pd.Series({"implied_prob": 0.0})
        assert pd.isna(_decimal_odds(row))

    def test_nan_for_negative_implied_prob(self):
        row = pd.Series({"implied_prob": -0.1})
        assert pd.isna(_decimal_odds(row))

    def test_nan_for_decimal_odds_below_1(self):
        row = pd.Series({"implied_prob": None, "decimal_odds": 0.5})
        assert pd.isna(_decimal_odds(row))


# ── _prob / _str helpers ──────────────────────────────────────────────────────

class TestHelpers:
    def test_prob_rounds_to_4dp(self):
        assert _prob(0.123456) == pytest.approx(0.1235)

    def test_prob_none_on_nan(self):
        assert _prob(float("nan")) is None

    def test_prob_none_on_none(self):
        assert _prob(None) is None

    def test_str_empty_on_none(self):
        assert _str(None) == ""

    def test_str_isoformat_for_timestamp(self):
        ts = pd.Timestamp("2024-06-01 13:00:00+01:00")
        result = _str(ts)
        assert "2024-06-01" in result

    def test_str_plain_string_passthrough(self):
        assert _str("Leopardstown") == "Leopardstown"


# ── _runner_dict ──────────────────────────────────────────────────────────────

class TestRunnerDict:
    def _row(self, **kwargs) -> pd.Series:
        defaults = {
            "horse_id": "h1",
            "horse_name": "Test Horse",
            "jockey": "J Smith",
            "trainer": "T Jones",
            "implied_prob": 0.20,
            "_dec_odds": 5.0,
            "won_prob": 0.18,
            "placed_2_prob": 0.42,
            "showed_prob": 0.61,
            "composite_score": 0.35,
            "each_way_value": True,
            "low_odds": False,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_keys_present(self):
        d = _runner_dict(self._row(), rank=1)
        for key in ("rank", "horse_id", "horse_name", "decimal_odds", "implied_prob",
                    "won_prob", "placed_2_prob", "showed_prob", "composite_score",
                    "each_way_value", "low_odds"):
            assert key in d

    def test_rank_preserved(self):
        assert _runner_dict(self._row(), rank=2)["rank"] == 2

    def test_won_prob_normalized_key_present_defaults_none(self):
        d = _runner_dict(self._row(), rank=1)
        assert "won_prob_normalized" in d
        assert d["won_prob_normalized"] is None  # absent in this row → None

    def test_won_prob_normalized_emitted_when_present(self):
        d = _runner_dict(self._row(won_prob_normalized=0.25), rank=1)
        assert d["won_prob_normalized"] == pytest.approx(0.25)

    def test_rank_none_for_excluded(self):
        assert _runner_dict(self._row(), rank=None)["rank"] is None

    def test_decimal_odds_none_when_nan(self):
        d = _runner_dict(self._row(_dec_odds=np.nan), rank=1)
        assert d["decimal_odds"] is None

    def test_each_way_value_bool(self):
        d = _runner_dict(self._row(each_way_value=True), rank=1)
        assert d["each_way_value"] is True


# ── _write_cache ──────────────────────────────────────────────────────────────

class TestWriteCache:
    def test_creates_file(self, tmp_path):
        cache = tmp_path / "data" / "predictions.json"
        payload = {"generated_at": "2024-01-01", "races": []}
        with patch("models.predictor._CACHE_PATH", cache), patch(

            "models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"

        ):
            _write_cache(payload)
        assert cache.exists()

    def test_valid_json(self, tmp_path):
        cache = tmp_path / "data" / "predictions.json"
        payload = {"generated_at": "2024-01-01", "races": [{"venue": "X"}]}
        with patch("models.predictor._CACHE_PATH", cache), patch(

            "models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"

        ):
            _write_cache(payload)
        data = json.loads(cache.read_text())
        assert data["races"][0]["venue"] == "X"

    def test_no_tmp_file_left_behind(self, tmp_path):
        cache = tmp_path / "data" / "predictions.json"
        with patch("models.predictor._CACHE_PATH", cache), patch(

            "models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"

        ):
            _write_cache({"races": []})
        tmp = cache.with_suffix(".tmp")
        assert not tmp.exists()

    def test_overwrites_previous(self, tmp_path):
        cache = tmp_path / "data" / "predictions.json"
        with patch("models.predictor._CACHE_PATH", cache), patch(

            "models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"

        ):
            _write_cache({"races": [], "v": 1})
            _write_cache({"races": [], "v": 2})
        data = json.loads(cache.read_text())
        assert data["v"] == 2


# ── Predictor.load ────────────────────────────────────────────────────────────

class TestPredictorLoad:
    def test_returns_false_when_no_bins(self, tmp_path):
        p = Predictor(model_dir=str(tmp_path))
        assert p.load() is False

    def test_returns_true_when_bin_present(self, tmp_path):
        # Write a real minimal CatBoost model to disk
        from catboost import CatBoostClassifier
        import numpy as np
        m = CatBoostClassifier(iterations=1, verbose=False, allow_writing_files=False)
        X = np.random.default_rng(0).random((20, len(FEATURE_COLS)))
        y = (X[:, 0] > 0.5).astype(int)
        m.fit(X, y, verbose=False)
        m.save_model(str(tmp_path / "catboost_won_v3.bin"))

        p = Predictor(model_dir=str(tmp_path))
        assert p.load() is True
        assert "won" in p._models

    def test_reads_feature_cols_from_meta(self, tmp_path):
        meta = {"feature_cols": ["implied_prob", "market_rank"], "targets": {}}
        (tmp_path / "catboost_v3_meta.json").write_text(json.dumps(meta))
        p = Predictor(model_dir=str(tmp_path))
        p.load()
        assert p._feature_cols == ["implied_prob", "market_rank"]

    def test_partial_load_returns_true(self, tmp_path):
        from catboost import CatBoostClassifier
        import numpy as np
        m = CatBoostClassifier(iterations=1, verbose=False, allow_writing_files=False)
        X = np.random.default_rng(1).random((20, len(FEATURE_COLS)))
        y = (X[:, 0] > 0.5).astype(int)
        m.fit(X, y, verbose=False)
        m.save_model(str(tmp_path / "catboost_won_v3.bin"))
        # placed_2 and showed are absent — should still return True

        p = Predictor(model_dir=str(tmp_path))
        assert p.load() is True


# ── Predictor._score ──────────────────────────────────────────────────────────

class TestPredictorScore:
    def test_prob_columns_added(self):
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert "won_prob" in scored.columns
        assert "placed_2_prob" in scored.columns
        assert "showed_prob" in scored.columns

    def test_composite_score_column_added(self):
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert "composite_score" in scored.columns

    def test_composite_score_in_range(self):
        live = _live_df(n_races=1, runners_per_race=6)
        p, _ = _predictor_with_mocks(live, won_prob=0.3, placed_prob=0.5, showed_prob=0.7)
        scored = p._score(live)
        assert (scored["composite_score"] >= 0.0).all()
        assert (scored["composite_score"] <= 1.0).all()

    def test_composite_score_weighted(self):
        """composite = 0.5*won + 0.3*placed + 0.2*showed → 0.5*0.1+0.3*0.4+0.2*0.6 = 0.29"""
        live = _live_df(n_races=1, runners_per_race=2)
        p, _ = _predictor_with_mocks(live, won_prob=0.10, placed_prob=0.40, showed_prob=0.60)
        scored = p._score(live)
        expected = 0.5 * 0.10 + 0.3 * 0.40 + 0.2 * 0.60
        assert scored["composite_score"].iloc[0] == pytest.approx(expected, rel=1e-4)

    def test_no_models_gives_zero_composite(self):
        live = _live_df(n_races=1, runners_per_race=3)
        p = Predictor()
        p._models = {}
        scored = p._score(live)
        assert (scored["composite_score"] == 0.0).all()

    def test_failed_predict_proba_fills_nan(self):
        live = _live_df(n_races=1, runners_per_race=3)
        p = Predictor()
        bad_model = MagicMock()
        bad_model.predict_proba.side_effect = RuntimeError("kaboom")
        p._models = {"won": bad_model}
        scored = p._score(live)
        assert scored["won_prob"].isna().all()

    def test_won_prob_normalized_field_sums_to_one(self):
        """won_prob_normalized: each race's field win probabilities sum to 1."""
        live = _live_df(n_races=2, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        for _, grp in scored.groupby([scored["venue"], scored["race_date"]]):
            assert grp["won_prob_normalized"].sum() == pytest.approx(1.0)

    def test_won_prob_itself_not_overwritten(self):
        """The headline calibrated won_prob is preserved; normalization is a
        separate column (marginal calibration is not sacrificed for sum-to-1)."""
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live, won_prob=0.18)
        scored = p._score(live)
        assert np.allclose(scored["won_prob"].values, 0.18)

    def test_normalized_preserves_within_race_win_ranking(self):
        """Normalization divides a race by a constant → win-prob order is unchanged."""
        live = _live_df(n_races=1, runners_per_race=6)
        p, _ = _predictor_with_mocks(live)
        p._models["won"].predict_proba.side_effect = lambda X: np.column_stack(
            [1 - X[:, 0] * 0.4, X[:, 0] * 0.4]
        )
        scored = p._score(live)
        raw = scored[FEATURE_COLS[0]].values * 0.4
        assert (np.argsort(scored["won_prob_normalized"].values).tolist()
                == np.argsort(raw).tolist())


# ── headline win prob (calib-fl-01) ───────────────────────────────────────────

class TestHeadlineWinProb:
    """The headline win prob the UI/API present is ``won_prob_normalized`` — a
    field-coherent, well-calibrated number — NOT the raw calibrated marginal
    ``won_prob``, which saturates badly out-of-sample (calib-fl-01)."""

    def test_headline_total_equals_expected_winner_count(self):
        """Summed over the whole card, the headline equals the number of races —
        i.e. the expected number of winners (exactly one per race)."""
        n_races = 4
        live = _live_df(n_races=n_races, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert scored["won_prob_normalized"].sum() == pytest.approx(n_races)

    def test_headline_reanchors_a_saturating_marginal(self):
        """Regression guard for the calib-fl-01 bug: when the raw calibrated
        marginal is overconfident/saturated (every runner pinned high), the
        headline (won_prob_normalized) must re-anchor to the base rate
        (~1/field_size) and never echo the saturating marginal value."""
        runners = 5
        live = _live_df(n_races=1, runners_per_race=runners)
        p, _ = _predictor_with_mocks(live, won_prob=0.70)  # saturated ceiling
        scored = p._score(live)
        # The marginal is preserved untouched as the raw/debug column …
        assert np.allclose(scored["won_prob"].values, 0.70)
        # … but the headline re-anchors: sums to 1, mean = 1/field_size, and is
        # nowhere near the saturating 0.70.
        assert scored["won_prob_normalized"].sum() == pytest.approx(1.0)
        assert scored["won_prob_normalized"].mean() == pytest.approx(1.0 / runners)
        assert scored["won_prob_normalized"].max() < scored["won_prob"].max()

    def test_runner_dict_emits_distinct_headline_and_raw_marginal(self):
        """_runner_dict surfaces BOTH: won_prob (raw marginal, debug) and
        won_prob_normalized (headline). For a saturated marginal they differ."""
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live, won_prob=0.70)
        scored = p._score(live)
        d = _runner_dict(scored.iloc[0], rank=1)
        assert d["won_prob"] == pytest.approx(0.70, abs=1e-3)
        assert d["won_prob_normalized"] == pytest.approx(0.25, abs=1e-3)  # 1/4
        assert d["won_prob_normalized"] != d["won_prob"]


# ── missing / partial features (live messy data) ──────────────────────────────

class TestMissingFeatures:
    """Live feeds emit debut horses, unknown jockeys, and no form. The scorer must
    feed CatBoost a full-width, correctly-ordered matrix (NaN for absent features,
    matching how it trained) and never silently emit no probabilities."""

    def test_feature_frame_full_width_and_order(self):
        """A live matrix missing some feature cols still yields a matrix with one
        column per trained feature, in order — absent ones NaN-filled."""
        live = _live_df(n_races=1, runners_per_race=3)
        # Drop a middle feature entirely (simulates a source that didn't populate it)
        dropped = FEATURE_COLS[5]
        live = live.drop(columns=[dropped])
        p, _ = _predictor_with_mocks(live)
        X = p._feature_frame(live, FEATURE_COLS)
        assert X.shape == (len(live), len(FEATURE_COLS))
        assert np.isnan(X[:, 5]).all()  # dropped column → NaN slot, position preserved

    def test_scores_despite_missing_columns(self):
        """predict_proba is still called (full width) when columns are absent —
        the old code dropped them and CatBoost would raise a shape mismatch."""
        live = _live_df(n_races=1, runners_per_race=4)
        live = live.drop(columns=[FEATURE_COLS[2], FEATURE_COLS[7]])
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert scored["won_prob"].notna().all()

    def test_all_null_feature_passes_through_as_nan(self):
        """A feature present but entirely null (e.g. paywalled rating) must not
        crash and must not be fabricated to a number."""
        live = _live_df(n_races=1, runners_per_race=3)
        live[FEATURE_COLS[10]] = np.nan
        p, _ = _predictor_with_mocks(live)
        X = p._feature_frame(live, FEATURE_COLS)
        assert np.isnan(X[:, 10]).all()

    def test_debut_runner_flagged_and_low_confidence(self):
        """First-time runner (no career runs / no trailing place rate) → flagged and
        confidence 'low' even with a complete market row."""
        live = _live_df(n_races=1, runners_per_race=2)
        live.loc[0, "horse_career_runs"] = 0
        live.loc[0, "historical_place_rate"] = np.nan
        live.loc[1, "horse_career_runs"] = 12  # established runner
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert bool(scored.loc[0, "first_time_runner"]) is True
        assert scored.loc[0, "confidence"] == "low"
        assert bool(scored.loc[1, "first_time_runner"]) is False

    def test_data_completeness_full_when_all_signal_present(self):
        live = _live_df(n_races=1, runners_per_race=3)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        # _live_df populates every FEATURE_COL → all signal features present.
        assert scored["data_completeness"].tolist() == pytest.approx([1.0] * len(scored))

    def test_data_completeness_drops_with_missing_signal(self):
        live = _live_df(n_races=1, runners_per_race=3)
        # Null out several signal features for runner 0.
        for c in ("jockey_win_rate", "trainer_win_rate", "course_win_rate"):
            live.loc[0, c] = np.nan
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        assert scored.loc[0, "data_completeness"] < 1.0

    def test_runner_dict_exposes_quality_fields(self):
        live = _live_df(n_races=1, runners_per_race=2)
        live.loc[0, "horse_career_runs"] = 0
        live.loc[0, "historical_place_rate"] = np.nan
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        d = _runner_dict(scored.iloc[0], rank=1)
        assert d["first_time_runner"] is True
        assert d["confidence"] == "low"
        assert 0.0 <= d["data_completeness"] <= 1.0

    def test_runner_dict_quality_defaults_safe_without_columns(self):
        """Mock rows / old caches lacking the quality columns get safe defaults."""
        row = pd.Series({"horse_id": "h", "horse_name": "H", "_dec_odds": 5.0})
        d = _runner_dict(row, rank=1)
        assert d["confidence"] == "unknown"
        assert d["first_time_runner"] is False
        assert d["data_completeness"] is None

    def test_value_layer_aligned_when_columns_missing(self):
        """The value model is also fed a full-width price-free matrix."""
        live = _live_df(n_races=1, runners_per_race=3)
        live = live.drop(columns=[PRICE_FREE_FEATURE_COLS[1]])
        p, _ = _predictor_with_mocks(live)
        p._value_model = _mock_catboost(0.25)
        scored = p._score(live)
        assert scored["value_win_prob"].notna().all()


# ── place/show coherence ──────────────────────────────────────────────────────

class TestTargetCoherence:
    """P(won) ≤ P(placed_2) ≤ P(showed): independent binary models can violate
    this; the scorer clips downstream targets up so they stay coherent."""

    def test_incoherent_models_clipped_to_monotonic(self):
        # won=0.50 but placed=0.30, showed=0.20 → must become 0.50, 0.50, 0.50
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live, won_prob=0.50, placed_prob=0.30, showed_prob=0.20)
        scored = p._score(live)
        assert (scored["placed_2_prob"] >= scored["won_prob"] - 1e-9).all()
        assert (scored["showed_prob"] >= scored["placed_2_prob"] - 1e-9).all()
        assert scored["placed_2_prob"].iloc[0] == pytest.approx(0.50)
        assert scored["showed_prob"].iloc[0] == pytest.approx(0.50)

    def test_already_coherent_unchanged(self):
        live = _live_df(n_races=1, runners_per_race=3)
        p, _ = _predictor_with_mocks(live, won_prob=0.18, placed_prob=0.40, showed_prob=0.60)
        scored = p._score(live)
        assert scored["won_prob"].iloc[0] == pytest.approx(0.18)
        assert scored["placed_2_prob"].iloc[0] == pytest.approx(0.40)
        assert scored["showed_prob"].iloc[0] == pytest.approx(0.60)


# ── Predictor._build_race ─────────────────────────────────────────────────────

class TestBuildRace:
    def _scored_grp(self, n: int = 6, implied_probs=None) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        rows = []
        for i in range(n):
            ip = implied_probs[i] if implied_probs else rng.uniform(0.05, 0.30)
            rows.append({
                "horse_id": f"h{i}",
                "horse_name": f"Horse{i}",
                "jockey": "J",
                "trainer": "T",
                "venue": "Fairyhouse",
                "race_date": pd.Timestamp("2024-06-01 14:00", tz="UTC"),
                "implied_prob": ip,
                "won_prob": rng.uniform(0.05, 0.35),
                "placed_2_prob": rng.uniform(0.15, 0.55),
                "showed_prob": rng.uniform(0.30, 0.75),
                "composite_score": rng.uniform(0.10, 0.50),
            })
        return pd.DataFrame(rows)

    def test_selections_max_top_n(self):
        p = Predictor()
        grp = self._scored_grp(8)
        race = p._build_race(grp)
        assert len(race["selections"]) <= 3

    def test_selections_sorted_by_composite(self):
        p = Predictor()
        grp = self._scored_grp(6)
        race = p._build_race(grp)
        scores = [s["composite_score"] for s in race["selections"]]
        assert scores == sorted(scores, reverse=True)

    def test_low_odds_runner_excluded(self):
        # Runner 0 at 1.50 (implied_prob=0.667) is below min_odds=2.50
        ips = [0.667, 0.20, 0.15, 0.12, 0.10, 0.08]
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(6, implied_probs=ips)
        # give the favourite the highest composite so it would rank #1 if not excluded
        grp.loc[0, "composite_score"] = 0.99
        race = p._build_race(grp)
        selection_ids = [s["horse_id"] for s in race["selections"]]
        assert "h0" not in selection_ids
        excluded_ids = [r["horse_id"] for r in race["excluded_low_odds"]]
        assert "h0" in excluded_ids

    def test_low_odds_flag_set_on_excluded_runner(self):
        ips = [0.60, 0.20, 0.15, 0.12, 0.10, 0.08]
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(6, implied_probs=ips)
        race = p._build_race(grp)
        excl = {r["horse_id"]: r for r in race["excluded_low_odds"]}
        assert excl["h0"]["low_odds"] is True

    def test_each_way_value_when_field_gte_8_and_odds_gte_threshold(self):
        # 8 runners; runner 0 at 10.0 (ip=0.10) → above each_way_threshold (8.0)
        ips = [0.10, 0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07]
        p = Predictor()
        grp = self._scored_grp(8, implied_probs=ips)
        race = p._build_race(grp)
        all_runners = race["selections"] + race["excluded_low_odds"]
        runner_0 = next(r for r in all_runners if r["horse_id"] == "h0")
        assert runner_0["each_way_value"] is True

    def test_no_each_way_when_field_lt_8(self):
        ips = [0.10, 0.15, 0.12, 0.11, 0.10]
        p = Predictor()
        grp = self._scored_grp(5, implied_probs=ips)
        race = p._build_race(grp)
        all_runners = race["selections"] + race["excluded_low_odds"]
        assert all(not r["each_way_value"] for r in all_runners)
        assert race["each_way_available"] is False

    def test_each_way_available_set_for_8_plus(self):
        p = Predictor()
        grp = self._scored_grp(8)
        race = p._build_race(grp)
        assert race["each_way_available"] is True

    def test_unknown_odds_runner_not_excluded(self):
        grp = self._scored_grp(4)
        grp.loc[0, "implied_prob"] = None  # unknown price
        p = Predictor(min_selection_odds=2.50)
        race = p._build_race(grp)
        excluded_ids = [r["horse_id"] for r in race["excluded_low_odds"]]
        assert "h0" not in excluded_ids

    def test_field_size_correct(self):
        p = Predictor()
        grp = self._scored_grp(7)
        race = p._build_race(grp)
        assert race["field_size"] == 7

    def test_venue_and_race_time_populated(self):
        p = Predictor()
        grp = self._scored_grp(5)
        race = p._build_race(grp)
        assert race["venue"] == "Fairyhouse"
        assert "2024-06-01" in race["race_time"]

    def test_non_runner_excluded_entirely(self):
        """A 'Non Runner' row must never appear in selections, excluded_low_odds,
        or count toward field_size."""
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(4)
        # jockey_name is the real schema column the filter inspects.
        grp["jockey_name"] = "Real Jockey"
        # Mark one runner as a non-runner with a high composite so it would
        # otherwise top the selections.
        grp.loc[1, "jockey_name"] = "Non Runner"
        grp.loc[1, "horse_name"] = "QUARTERMASTER (IRE)"
        grp.loc[1, "horse_id"] = "nr0"
        grp.loc[1, "composite_score"] = 0.99
        grp.loc[1, "implied_prob"] = 0.05  # high odds, not low-odds-excluded

        race = p._build_race(grp)

        all_ids = [r["horse_id"] for r in race["selections"]] + [
            r["horse_id"] for r in race["excluded_low_odds"]
        ]
        all_names = [r["horse_name"] for r in race["selections"]] + [
            r["horse_name"] for r in race["excluded_low_odds"]
        ]
        assert "nr0" not in all_ids
        assert "QUARTERMASTER (IRE)" not in all_names
        assert race["field_size"] == 3

    def test_declared_runner_status_non_runner_is_excluded(self):
        """Step 09 audit F5: the racecard's own runner_status reached the live
        matrix (step 05) but nothing read it, so a declared withdrawal stayed in
        the field whenever the odds feed still carried a normal jockey name."""
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(4)
        grp["jockey_name"] = "Real Jockey"
        grp["runner_status"] = "RUNNER"
        grp.loc[1, "runner_status"] = "NON_RUNNER"
        grp.loc[1, "horse_id"] = "nr0"
        grp.loc[1, "composite_score"] = 0.99
        grp.loc[1, "implied_prob"] = 0.05

        race = p._build_race(grp)

        all_ids = [r["horse_id"] for r in race["selections"]] + [
            r["horse_id"] for r in race["excluded_low_odds"]]
        assert "nr0" not in all_ids
        assert race["field_size"] == 3

    def test_null_runner_status_keeps_the_whole_field(self):
        """runner_status is absent/null for every historical and odds-only row,
        so a null must never be read as a withdrawal."""
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(4)
        grp["jockey_name"] = "Real Jockey"
        grp["runner_status"] = None
        race = p._build_race(grp)
        assert race["field_size"] == 4

    def test_all_null_runner_status_as_pyarrow_null_dtype_does_not_crash(self):
        """Step 18 live-refresh crash: when no declared card has arrived (B8),
        runner_status is null on every row; pandas' pyarrow backend infers such
        a column as null[pyarrow], and the old `.fillna("")` raised ArrowInvalid
        instead of treating it as an absent status."""
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(4)
        grp["jockey_name"] = "Real Jockey"
        grp["runner_status"] = pd.Series([None] * len(grp), index=grp.index).convert_dtypes(
            dtype_backend="pyarrow"
        )
        race = p._build_race(grp)
        assert race["field_size"] == 4


# ── audit regressions + output invariants (probability→EV path) ───────────────

class TestProbabilityToEvAuditRegressions:
    """Regression guards for the two structural defects the probability→EV audit
    fixed, plus the honesty invariants (audit req 1, 2, 9)."""

    def test_non_runner_dropped_before_within_race_normalization(self):
        """Regression (req 1): a non-runner must be removed BEFORE within-race
        normalization, not after selection.

        Old bug: normalization ran over the full frame *including* the withdrawn
        horse, so that horse absorbed a share of the sum-to-1 mass; only later did
        selection drop it, leaving the real field's displayed win probabilities
        summing to < 1. After the fix the non-runner never reaches normalization,
        so the surviving field sums to exactly 1 and the withdrawn row is gone.
        """
        live = _live_df(n_races=1, runners_per_race=5)
        live["jockey_name"] = "Real Jockey"
        # One withdrawn runner, given the top raw win prob so — if it were still in
        # the field at normalization time — it would swallow the largest mass share.
        live.loc[0, "jockey_name"] = "Non Runner"
        live.loc[0, "horse_id"] = "nr0"
        p = Predictor()
        p._models = {"won": _mock_catboost(0.30)}

        scored = p._score(live)

        assert "nr0" not in set(scored["horse_id"])          # dropped pre-normalize
        assert len(scored) == 4                               # only the real field
        # The four survivors carry the *entire* normalized mass — proof the divisor
        # was the cleaned field, not the 5-runner frame (which would give 0.8).
        assert scored["won_prob_normalized"].sum() == pytest.approx(1.0)

    def test_value_engine_receives_complete_field_not_top_three(self):
        """Regression (req 2): the value engine must see EVERY valid runner, not the
        display-only top three.

        A genuine value bet can sit outside the top-three selections (a mid-field
        runner the market over-prices). If ``find_value_bets`` is fed only
        ``selections`` it never sees that runner. Fed the full ``runners`` field it
        finds it. This reproduces the partial-field bug and its fix.
        """
        from models.value import find_value_bets, ValueConfig

        cfg = ValueConfig(devig=False, require_support=False, min_confidence=0.0,
                          min_prob=0.0, min_odds=2.0, max_odds=6.0, min_ev=0.05)
        top3 = [{"horse_id": f"fav{i}", "horse_name": f"Fav{i}",
                 "model_prob": 0.50, "best_odds": 1.5} for i in range(3)]
        # EV = 0.30 * 5.0 − 1 = +0.5, inside the [2.0, 6.0] band — a clear value bet,
        # but ranked below the three favourites, so absent from a top-three list.
        value_pick = {"horse_id": "val", "horse_name": "Value",
                      "model_prob": 0.30, "best_odds": 5.0}

        # Partial field (display top three only) → the value pick is invisible.
        assert find_value_bets({"selections": top3}, config=cfg) == []
        # Complete field via `runners` → the value pick is found.
        picks = find_value_bets({"runners": top3 + [value_pick]}, config=cfg)
        assert any(pk["horse_id"] == "val" for pk in picks)

    def test_build_race_emits_full_runners_field(self):
        """req 2: `_build_race` exposes the complete field under `runners` while
        `selections` stays the top three."""
        p = Predictor()
        grp = TestBuildRace()._scored_grp(6)
        race = p._build_race(grp)
        assert len(race["runners"]) == 6
        assert len(race["selections"]) <= 3
        # every selection is part of the full field
        sel_ids = {r["horse_id"] for r in race["selections"]}
        run_ids = {r["horse_id"] for r in race["runners"]}
        assert sel_ids.issubset(run_ids)

    def test_invariants_pass_on_a_clean_race(self):
        norm = 1.0 / 3
        race = {
            "venue": "Ascot", "race_time": "2026-07-25T14:00:00+00:00",
            "runners": [
                {"horse_id": f"h{i}", "horse_name": f"H{i}",
                 "won_prob_normalized": norm, "market_prob": norm,
                 "decimal_odds": 3.0, "best_odds": 3.1}
                for i in range(3)
            ],
        }
        assert check_output_invariants([race]) == []

    def test_invariants_flag_probabilities_not_summing_to_one(self):
        race = {
            "venue": "Ascot", "race_time": "t",
            "runners": [
                {"horse_id": "a", "horse_name": "A", "won_prob_normalized": 0.2},
                {"horse_id": "b", "horse_name": "B", "won_prob_normalized": 0.2},
            ],
        }
        viols = check_output_invariants([race])
        assert any("normalized win probs sum" in v for v in viols)

    def test_invariants_flag_invalid_odds_and_phantom_runner(self):
        race = {
            "venue": "Ascot", "race_time": "t",
            "runners": [
                {"horse_id": "", "horse_name": "", "best_odds": 0.9},
            ],
        }
        viols = check_output_invariants([race])
        assert any("phantom runner" in v for v in viols)
        assert any("≤1" in v for v in viols)

    def test_invariants_flag_value_bet_without_valid_price(self):
        race = {
            "venue": "Ascot", "race_time": "t",
            "runners": [
                {"horse_id": "a", "horse_name": "A", "value_bet": True,
                 "expected_value": None, "best_odds": None, "decimal_odds": None},
            ],
        }
        viols = check_output_invariants([race])
        assert any("without a finite EV" in v for v in viols)

    def test_invariants_flag_ev_leak_on_ineligible_race(self):
        """Invariant 6: a race stamped ev_eligible=False must carry no EV/value."""
        race = {
            "venue": "Ascot", "race_time": "t", "ev_eligible": False,
            "runners": [
                {"horse_id": "a", "horse_name": "A", "value_bet": False,
                 "expected_value": 0.4, "decimal_odds": 3.0},
            ],
        }
        viols = check_output_invariants([race])
        assert any("EV-ineligible race" in v for v in viols)

    def test_invariants_flag_partial_market_line_on_eligible_race(self):
        """Invariant 6: an eligible race where only some runners carry market_prob
        is a partial de-vig — exactly the defect the audit exists to prevent."""
        race = {
            "venue": "Ascot", "race_time": "t", "ev_eligible": True,
            "runners": [
                {"horse_id": "a", "horse_name": "A", "market_prob": 0.5,
                 "decimal_odds": 2.0},
                {"horse_id": "b", "horse_name": "B", "market_prob": None,
                 "decimal_odds": 2.0},
            ],
        }
        viols = check_output_invariants([race])
        assert any("partial de-vig" in v for v in viols)


# ── race-level EV eligibility gate (audit req 4/6/9, handoff issue #6) ─────────

class TestRaceEvGate:
    """One race-level EV decision, computed in _build_race, persisted as
    ``ev_eligible``/``ev_gate``, enforced across every EV surface."""

    def _grp(self, n: int = 4) -> pd.DataFrame:
        rows = []
        for i in range(n):
            rows.append({
                "horse_id": f"h{i}", "horse_name": f"Horse{i}", "jockey": "J",
                "trainer": "T", "venue": "Ascot",
                "race_date": pd.Timestamp("2026-06-01 14:00", tz="UTC"),
                "implied_prob": 0.25,          # → decimal 4.0, inside [2.0, 4.0]
                "won_prob": 0.2, "placed_2_prob": 0.4, "showed_prob": 0.6,
                "composite_score": 0.5 - i * 0.05,
                "value_win_prob": 0.30, "value_edge": 0.05,
                "expected_value": 0.30,        # clears min_ev 0.05
                "value_supported": True,
            })
        return pd.DataFrame(rows)

    def test_fully_priced_race_is_eligible(self):
        p = Predictor()
        race = p._build_race(self._grp())
        assert race["ev_eligible"] is True
        gate = race["ev_gate"]
        assert gate["reasons"] == []
        assert gate["reference_book_complete"] is True
        assert gate["reference_source"] == "fused_consensus"
        assert gate["computed_at"]
        # 4 runners at implied 0.25 → booksum 1.0 → overround 0.0
        assert gate["reference_overround"] == pytest.approx(0.0, abs=1e-3)
        assert any(r["value_bet"] for r in race["runners"])

    def test_unpriced_runner_fails_race_closed(self):
        """req 4/9: one runner with no price → no complete reference book → the
        whole race PASSes and every EV/market field is withdrawn."""
        grp = self._grp()
        grp.loc[2, "implied_prob"] = np.nan
        p = Predictor()
        race = p._build_race(grp)
        assert race["ev_eligible"] is False
        joined = "; ".join(race["ev_gate"]["reasons"])
        assert "incomplete_reference_book:3/4" in joined
        assert "unpriced_runners:1" in joined
        for r in race["runners"] + race["selections"] + race["excluded_low_odds"]:
            assert r["value_bet"] is False
            assert r["expected_value"] is None
            assert r["value_edge"] is None
        assert check_output_invariants([race]) == []

    def test_stale_provenance_fails_race_closed_at_cache_boundary(self):
        """req 6/9: fetched_at beyond the TTL → PASS, even if upstream gates were
        somehow bypassed. The cache boundary is the last line of defence."""
        grp = self._grp()
        grp["fetched_at"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
        p = Predictor()
        race = p._build_race(grp)
        assert race["ev_eligible"] is False
        assert any(r.startswith("odds_age_exceeds_ttl")
                   for r in race["ev_gate"]["reasons"])
        assert all(not r["value_bet"] for r in race["runners"])
        assert all(r["expected_value"] is None for r in race["runners"])

    def test_a_stale_row_hidden_by_mixed_timestamp_precision_still_fails_closed(self):
        """Step 17: ``fetched_at`` strings of mixed precision (whole-second vs
        sub-second, as different scrapers emit) made pandas infer ONE format and
        return NaT for the rest; ``ages.max()`` skipped the NaT, so the one STALE row
        below was invisible and the race stayed eligible — the gate failed OPEN."""
        grp = self._grp()
        fresh = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=30)
        stale = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
        grp["fetched_at"] = [
            fresh.strftime("%Y-%m-%dT%H:%M:%S+00:00"),          # whole-second: sets the inferred format
            stale.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00"),       # sub-second AND two hours old
            fresh.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            fresh.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        ]
        race = Predictor()._build_race(grp)
        assert race["ev_eligible"] is False
        assert any(r.startswith("odds_age_exceeds_ttl") for r in race["ev_gate"]["reasons"])

    def test_an_unreadable_price_timestamp_fails_race_closed(self):
        """A timestamp that is present but unparseable is an age we cannot compute —
        never a freshness pass."""
        grp = self._grp()
        fresh = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=30)).isoformat()
        grp["fetched_at"] = [fresh, "not-a-timestamp", fresh, fresh]
        race = Predictor()._build_race(grp)
        assert race["ev_eligible"] is False
        assert "unreadable_price_timestamp:1" in race["ev_gate"]["reasons"]
        assert all(r["expected_value"] is None for r in race["runners"])

    def test_explicit_stale_flag_fails_race_closed(self):
        grp = self._grp()
        grp["stale"] = [False, True, False, False]
        p = Predictor()
        race = p._build_race(grp)
        assert race["ev_eligible"] is False
        assert "stale_price_rows:1" in race["ev_gate"]["reasons"]

    def test_fresh_provenance_keeps_race_eligible(self):
        grp = self._grp()
        grp["fetched_at"] = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=30)
        grp["stale"] = False
        p = Predictor()
        race = p._build_race(grp)
        assert race["ev_eligible"] is True
        assert race["ev_gate"]["odds_max_age_seconds"] is not None
        assert race["ev_gate"]["odds_max_age_seconds"] < 900

    def test_ineligible_race_passes_value_and_suggestion_engines(self):
        """req 7 / issue #6 — the persisted race-level PASS is honoured by
        find_value_bets and suggest_bets: no surface may rebuild an EV the cache
        refused to carry (the UI/backend rule-disagreement regression)."""
        from models.suggestions import suggest_bets
        from models.value import ValueConfig, find_value_bets

        grp = self._grp()
        grp.loc[2, "implied_prob"] = np.nan
        p = Predictor()
        race = p._build_race(grp)
        assert race["ev_eligible"] is False
        # Deliberately permissive config: without the race-level gate these
        # settings WOULD emit picks from the three priced runners.
        cfg = ValueConfig(devig=False, require_support=False, min_confidence=0.0)
        assert find_value_bets(race, config=cfg) == []
        book = suggest_bets([race], value_config=cfg)
        assert book.suggestions == []
        assert book.n_races_with_value == 0

    def test_eligible_race_flags_agree_across_surfaces(self):
        """req 7: every find_value_bets pick is a runner the cache flags
        value_bet=True — the shared core gate keeps the surfaces consistent."""
        from models.value import ValueConfig, find_value_bets

        p = Predictor()
        race = p._build_race(self._grp())
        flagged = {r["horse_id"] for r in race["runners"] if r["value_bet"]}
        assert flagged, "fixture must be a value race"
        picks = find_value_bets(race, config=ValueConfig(min_confidence=0.0))
        assert picks, "same race must clear the value engine's core gates"
        assert {pk["horse_id"] for pk in picks} <= flagged

    def test_race_prediction_roundtrip_preserves_runners_and_gate(self):
        """predict_typed's typed view must not silently drop the full field or
        the race-level gate the JSON cache carries."""
        p = Predictor()
        race = p._build_race(self._grp())
        rp = RacePrediction.from_dict(race)
        d = rp.to_dict()
        assert len(d["runners"]) == len(race["runners"])
        assert d["ev_eligible"] is True
        assert d["ev_gate"]["reference_book_complete"] is True


# ── reference-book selection (audit req 5) ─────────────────────────────────────

class TestReferenceBookSelection:
    """The fair line de-vigs ONE complete bookmaker board when any source prices
    the whole field, else the documented fused consensus — never the synthetic
    best-price overlay."""

    def test_single_complete_book_preferred_over_fused_mixture(self):
        from models.predictor import _reference_decimals
        df = pd.DataFrame({
            "implied_prob": [0.25, 0.25, 0.25],
            "odds_by_book": [
                {"livescorebet": 3.8, "boylesports": 3.9},
                {"boylesports": 4.1},                       # livescorebet hole
                {"livescorebet": 4.0, "boylesports": 4.2},
            ],
        })
        ref, src = _reference_decimals(df, [("V", "t")] * 3)
        # livescorebet misses runner 1 → boylesports is the first COMPLETE book
        assert list(src) == ["boylesports"] * 3
        assert list(ref) == [3.9, 4.1, 4.2]

    def test_no_complete_book_falls_back_to_fused_consensus(self):
        from models.predictor import _reference_decimals
        df = pd.DataFrame({
            "implied_prob": [0.25, 0.20],
            "odds_by_book": [
                {"livescorebet": 3.8},   # neither book prices both runners
                {"boylesports": 5.1},
            ],
        })
        ref, src = _reference_decimals(df, [("V", "t")] * 2)
        assert list(src) == ["fused_consensus"] * 2
        assert list(ref) == [4.0, 5.0]  # 1/0.25, 1/0.20

    def test_two_races_choose_books_independently(self):
        from models.predictor import _reference_decimals
        df = pd.DataFrame({
            "implied_prob": [0.5, 0.5, 0.5, 0.5],
            "odds_by_book": [
                {"livescorebet": 2.0}, {"livescorebet": 2.1},   # race A: lsb complete
                {"boylesports": 2.2}, {"boylesports": 2.3},     # race B: boyle complete
            ],
        })
        gids = [("V", "13:00"), ("V", "13:00"), ("V", "13:30"), ("V", "13:30")]
        ref, src = _reference_decimals(df, gids)
        assert list(src) == ["livescorebet", "livescorebet",
                             "boylesports", "boylesports"]
        assert list(ref) == [2.0, 2.1, 2.2, 2.3]

    def test_best_price_overlay_never_used_as_reference(self):
        from models.predictor import _reference_decimals
        df = pd.DataFrame({
            "implied_prob": [0.25, 0.25],
            "best_odds": [9.9, 9.8],     # synthetic overlay — must be ignored
            "odds_by_book": [{}, {}],
        })
        ref, src = _reference_decimals(df, [("V", "t")] * 2)
        assert list(src) == ["fused_consensus"] * 2
        assert list(ref) == [4.0, 4.0]


# ── exact EV reproducibility from the persisted cache (audit req 8/10) ─────────

class TestExactEvRecompute:
    """Every cached EV/edge must be EXACTLY recomputable from cached inputs:
    ev == round(prob * price - 1, 4), edge == round(prob - implied, 4). The
    quantisation happens before the multiply, so there is no hidden precision."""

    def test_cached_value_ev_and_edge_reproduce_exactly(self):
        live = _live_df(n_races=1, runners_per_race=5, seed=7)
        p, _ = _predictor_with_mocks(live)
        p._value_model = _mock_catboost(0.2734567)   # deliberately awkward float
        p._value_calibrator = None
        p._value_fl_calibrator = None
        scored = p._score(live)
        race = p._build_race(scored)
        checked = 0
        for r in race["runners"]:
            if r["expected_value"] is None or r["best_odds"] is None:
                continue
            assert round(r["value_win_prob"] * r["best_odds"] - 1.0, 4) \
                == r["expected_value"]
            assert round(r["value_win_prob"] - r["implied_prob"], 4) \
                == r["value_edge"]
            checked += 1
        assert checked >= 4

    def test_cached_line_evs_reproduce_exactly(self):
        live = _live_df(n_races=1, runners_per_race=5, seed=11)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        race = p._build_race(scored)
        checked = 0
        for r in race["runners"]:
            exe = r["best_odds"] if r["best_odds"] is not None else r["decimal_odds"]
            if r.get("ev_catboost") is None or exe is None:
                continue
            assert round(r["catboost_win_prob"] * exe - 1.0, 4) == r["ev_catboost"]
            checked += 1
        assert checked >= 4

    def test_cached_decimal_odds_reproduce_from_cached_implied_prob(self):
        live = _live_df(n_races=1, runners_per_race=5, seed=13)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        race = p._build_race(scored)
        for r in race["runners"]:
            if r["decimal_odds"] is None or not r["implied_prob"]:
                continue
            assert r["decimal_odds"] == round(1.0 / r["implied_prob"], 3)


# ── Predictor.predict (full pipeline) ─────────────────────────────────────────

class TestPredictorPredict:
    def test_returns_list(self, tmp_path):
        live = _live_df(n_races=2, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        assert isinstance(races, list)
        assert len(races) == 2

    def test_each_race_has_selections(self, tmp_path):
        live = _live_df(n_races=3, runners_per_race=6)
        p, _ = _predictor_with_mocks(live)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        for race in races:
            assert len(race["selections"]) > 0
            assert len(race["selections"]) <= 3

    def test_cache_file_written(self, tmp_path):
        live = _live_df(n_races=2, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        cache = tmp_path / "data" / "predictions.json"
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", cache),

            patch("models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"),
        ):
            p.predict()
        assert cache.exists()
        data = json.loads(cache.read_text())
        assert "generated_at" in data
        assert "races" in data
        assert data["total_races"] == 2

    def test_returns_empty_on_empty_inference_matrix(self, tmp_path):
        empty = pd.DataFrame()
        p, _ = _predictor_with_mocks(empty)
        with (
            patch("models.predictor.build_inference_matrix", return_value=empty),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        assert races == []

    def test_auto_loads_models_if_none_loaded(self, tmp_path):
        """predict() calls load() lazily when _models is empty."""
        live = _live_df(n_races=1, runners_per_race=3)
        p = Predictor()
        assert not p._models  # nothing loaded yet
        with (
            patch.object(p, "load", return_value=False) as mock_load,
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        mock_load.assert_called_once()
        assert races == []

    def test_races_sorted_by_race_time(self, tmp_path):
        live = _live_df(n_races=4, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        times = [r["race_time"] for r in races]
        assert times == sorted(times)

    def test_same_venue_multiple_post_times_not_collapsed(self, tmp_path):
        """Regression: two races at the SAME venue/day but different post-times must
        stay separate. Grouping by venue-day collapsed every race at a track into one
        inflated card (field_size summed, ranks across the whole meeting)."""
        base = pd.Timestamp("2024-06-01 13:00:00", tz="UTC")
        r1 = _live_df(n_races=1, runners_per_race=5, seed=0)
        r2 = _live_df(n_races=1, runners_per_race=5, seed=1)
        for d in (r1, r2):
            d["venue"] = "Ascot"
        r1["race_time"] = base.isoformat()
        r2["race_time"] = (base + pd.Timedelta(minutes=35)).isoformat()
        r2["horse_id"] = "g" + r2["horse_id"].astype(str)  # distinct ids
        combined = pd.concat([r1, r2], ignore_index=True)
        p, _ = _predictor_with_mocks(combined)
        with (
            patch("models.predictor.build_inference_matrix", return_value=combined),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict()
        assert len(races) == 2                              # not 1 (collapsed)
        assert all(r["venue"] == "Ascot" for r in races)
        assert all(r["field_size"] == 5 for r in races)     # not 10
        assert len({r["race_time"] for r in races}) == 2

    def test_model_targets_in_cache(self, tmp_path):
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live)
        cache = tmp_path / "data" / "predictions.json"
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", cache),

            patch("models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"),
        ):
            p.predict()
        data = json.loads(cache.read_text())
        assert set(data["model_targets"]) == {"won", "placed_2", "showed"}


# ── upcoming-races guard (historical-join-miss regression) ────────────────────

class TestUpcomingGuard:
    """Predictor._guard_upcoming must drop any non-upcoming row before scoring, so
    a builder regression can never resurrect the historical-join-miss bug."""

    def test_guard_drops_result_bearing_rows(self):
        live = _live_df(n_races=1, runners_per_race=4)
        live.loc[0, "position"] = 3  # a finished runner is history, never live
        p, _ = _predictor_with_mocks(live)
        kept = p._guard_upcoming(live)
        assert len(kept) == 3
        assert (kept["position"].isna()).all()

    def test_guard_drops_past_dated_join_misses(self):
        live = _live_df(n_races=1, runners_per_race=4)
        # Two null-position rows pinned to the past = the join-miss signature.
        live.loc[0, "race_date"] = pd.Timestamp("2020-01-01", tz="UTC")
        live.loc[1, "race_date"] = pd.Timestamp("2019-05-05", tz="UTC")
        p, _ = _predictor_with_mocks(live)
        kept = p._guard_upcoming(live)
        assert len(kept) == 2

    def test_guard_keeps_genuine_upcoming(self):
        live = _live_df(n_races=2, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        kept = p._guard_upcoming(live)
        assert len(kept) == len(live)

    def test_predict_excludes_non_upcoming_rows(self, tmp_path):
        """End-to-end: corrupt rows injected into the live matrix never reach the
        cache — total_runners reflects only genuine upcoming runners."""
        live = _live_df(n_races=1, runners_per_race=5)
        live.loc[0, "race_date"] = pd.Timestamp("2018-01-01", tz="UTC")  # join-miss
        live.loc[1, "position"] = 1                                       # finished
        p, _ = _predictor_with_mocks(live)
        cache = tmp_path / "data" / "predictions.json"
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", cache),

            patch("models.predictor._MODEL_MANIFEST_PATH", cache.parent / "execution" / "served_model_manifest.json"),
        ):
            p.predict()
        data = json.loads(cache.read_text())
        assert data["total_runners"] == 3


# ── typed result objects ──────────────────────────────────────────────────────

class TestTypedResults:
    def test_predict_typed_returns_dataclasses(self, tmp_path):
        live = _live_df(n_races=2, runners_per_race=5)
        p, _ = _predictor_with_mocks(live)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
        ):
            races = p.predict_typed()
        assert all(isinstance(r, RacePrediction) for r in races)
        assert all(
            isinstance(s, RunnerPrediction) for r in races for s in r.selections
        )

    def test_runner_round_trips_through_dict(self):
        live = _live_df(n_races=1, runners_per_race=2)
        p, _ = _predictor_with_mocks(live)
        scored = p._score(live)
        d = _runner_dict(scored.iloc[0], rank=1)
        runner = RunnerPrediction.from_dict(d)
        # The typed dataclass round-trips its own fields exactly. The unified
        # multi-line keys (catboost/lgbm/market prob + per-line EV) are an additive
        # JSON-only layer outside the dataclass, so they are excluded from the
        # round-trip comparison (and the lgbm_* keys may be omitted entirely).
        typed_keys = set(runner.to_dict())
        assert runner.to_dict() == {k: v for k, v in d.items() if k in typed_keys}
        assert {"catboost_win_prob", "market_prob", "ev_catboost"} <= d.keys()

    def test_from_dict_tolerates_old_cache(self):
        """A cache predating the data-quality fields rehydrates with safe defaults."""
        old = {"rank": 1, "horse_id": "h", "horse_name": "H", "won_prob": 0.2}
        runner = RunnerPrediction.from_dict(old)
        assert runner.horse_name == "H"
        assert runner.confidence is None
        assert runner.data_completeness is None


# ── Predictor.refresh ─────────────────────────────────────────────────────────

class TestPredictorRefresh:
    def test_refresh_delegates_to_predict(self, tmp_path):
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
            patch.object(p, "predict", wraps=p.predict) as spy,
        ):
            p.refresh()
        spy.assert_called_once_with(unified=None)


# ── module-level predict() ────────────────────────────────────────────────────

class TestModuleLevelPredict:
    def test_returns_list(self, tmp_path):
        live = _live_df(n_races=1, runners_per_race=4)
        with (
            patch("models.predictor.build_inference_matrix", return_value=live),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
            patch("models.predictor.Predictor.load", return_value=True),
        ):
            p_inst = Predictor()
            p_inst._models = {
                "won": _mock_catboost(0.15),
                "placed_2": _mock_catboost(0.38),
                "showed": _mock_catboost(0.60),
            }
            with patch("models.predictor.Predictor", return_value=p_inst):
                result = module_predict()
        assert isinstance(result, list)

    def test_accepts_preloaded_unified(self, tmp_path):
        live = _live_df(n_races=1, runners_per_race=3)
        captured = {}

        def fake_build(unified=None):
            captured["unified"] = unified
            return live

        p_inst = Predictor()
        p_inst._models = {
            "won": _mock_catboost(0.20),
            "placed_2": _mock_catboost(0.40),
            "showed": _mock_catboost(0.60),
        }
        with (
            patch("models.predictor.build_inference_matrix", side_effect=fake_build),
            patch("models.predictor._CACHE_PATH", tmp_path / "data" / "predictions.json"),

            patch("models.predictor._MODEL_MANIFEST_PATH", tmp_path / "data" / "execution" / "served_model_manifest.json"),
            patch("models.predictor.Predictor", return_value=p_inst),
        ):
            module_predict(unified=live)
        assert captured["unified"] is live


# ── min_odds boundary ─────────────────────────────────────────────────────────

class TestMinOddsBoundary:
    def _grp(self, implied_prob: float) -> pd.DataFrame:
        return pd.DataFrame([{
            "horse_id": "h0", "horse_name": "A", "jockey": "J", "trainer": "T",
            "venue": "V", "race_date": pd.Timestamp("2024-06-01", tz="UTC"),
            "implied_prob": implied_prob,
            "won_prob": 0.20, "placed_2_prob": 0.40, "showed_prob": 0.60,
            "composite_score": 0.30,
        }])

    def test_exactly_at_min_odds_is_eligible(self):
        # ip = 1/2.50 = 0.40 → decimal = exactly 2.50 → NOT low_odds
        p = Predictor(min_selection_odds=2.50)
        grp = self._grp(implied_prob=1.0 / 2.50)
        race = p._build_race(grp)
        assert len(race["selections"]) == 1
        assert race["excluded_low_odds"] == []

    def test_just_below_min_odds_is_excluded(self):
        # ip = 1/2.49 → decimal ≈ 2.49 → low_odds
        p = Predictor(min_selection_odds=2.50)
        grp = self._grp(implied_prob=1.0 / 2.49)
        race = p._build_race(grp)
        assert race["selections"] == []
        assert len(race["excluded_low_odds"]) == 1


# ── value-betting layer ─────────────────────────────────────────────────────────

class TestValueLayer:
    """Price-free value model → value_win_prob / value_edge / expected_value /
    value_bet. The layer is additive and must default safely when absent."""

    def _value_predictor(self, live, value_prob=0.30, calibrator=None):
        """Predictor with the three main mocks + a mock value model."""
        p, _ = _predictor_with_mocks(live)
        p._value_model = _mock_catboost(value_prob)
        p._value_calibrator = calibrator
        # __init__ already set _value_feature_cols = PRICE_FREE_FEATURE_COLS,
        # all of which _live_df provides.
        return p

    # ── _runner_dict defaults ──
    def test_runner_dict_has_value_keys_by_default(self):
        """TestBuildRace-style rows carry no value cols → keys present & safe."""
        row = pd.Series({
            "horse_id": "h1", "horse_name": "H", "implied_prob": 0.2,
            "_dec_odds": 5.0, "composite_score": 0.3,
        })
        d = _runner_dict(row, rank=1)
        assert d["value_win_prob"] is None
        assert d["value_edge"] is None
        assert d["expected_value"] is None
        assert d["value_bet"] is False

    # ── _score with value model absent ──
    def test_score_no_value_columns_when_model_absent(self):
        live = _live_df(n_races=1, runners_per_race=4)
        p, _ = _predictor_with_mocks(live)  # no value model
        scored = p._score(live)
        assert "value_win_prob" not in scored.columns
        assert "expected_value" not in scored.columns

    # ── _score with value model present ──
    def test_score_adds_value_columns(self):
        live = _live_df(n_races=1, runners_per_race=4)
        p = self._value_predictor(live, value_prob=0.30)
        scored = p._score(live)
        assert "value_win_prob" in scored.columns
        assert np.allclose(scored["value_win_prob"].values, 0.30)

    def test_value_edge_is_prob_minus_implied(self):
        live = _live_df(n_races=1, runners_per_race=3)
        live["implied_prob"] = 0.20
        p = self._value_predictor(live, value_prob=0.35)
        scored = p._score(live)
        assert scored["value_edge"].iloc[0] == pytest.approx(0.35 - 0.20)

    def test_expected_value_formula(self):
        # implied_prob=0.20 → decimal=5.0; value_prob=0.30 → EV = 0.30*5 - 1 = 0.5
        live = _live_df(n_races=1, runners_per_race=3)
        live["implied_prob"] = 0.20
        p = self._value_predictor(live, value_prob=0.30)
        scored = p._score(live)
        assert scored["expected_value"].iloc[0] == pytest.approx(0.30 * 5.0 - 1.0)

    def test_value_calibrator_applied(self):
        live = _live_df(n_races=1, runners_per_race=3)
        # calibrator that halves the probability
        calib = MagicMock()
        calib.predict.side_effect = lambda p: np.asarray(p) * 0.5
        p = self._value_predictor(live, value_prob=0.40, calibrator=calib)
        scored = p._score(live)
        assert np.allclose(scored["value_win_prob"].values, 0.20)

    def test_composite_score_unchanged_by_value_layer(self):
        """Adding the value layer must not perturb composite ranking."""
        live = _live_df(n_races=1, runners_per_race=2)
        p = self._value_predictor(live, value_prob=0.99)
        p_no_val, _ = _predictor_with_mocks(live)
        a = p._score(live)["composite_score"].values
        b = p_no_val._score(live)["composite_score"].values
        assert a == pytest.approx(b)

    # ── _build_race value_bet flag ──
    def _scored_grp(self, implied_prob, expected_value):
        return pd.DataFrame([{
            "horse_id": "h0", "horse_name": "A", "jockey": "J", "trainer": "T",
            "venue": "V", "race_date": pd.Timestamp("2024-06-01", tz="UTC"),
            "implied_prob": implied_prob,
            "won_prob": 0.20, "placed_2_prob": 0.40, "showed_prob": 0.60,
            "composite_score": 0.30, "expected_value": expected_value,
        }])

    def test_value_bet_true_when_ev_positive_in_band(self):
        # ip=0.30 → dec≈3.33 (within 2.0–4.0); EV=0.5 ≥ 0.05
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.30, expected_value=0.5)
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is True
        assert all_r[0]["expected_value"] == pytest.approx(0.5)

    def test_value_bet_false_when_ev_below_threshold(self):
        # ip=0.30 → dec≈3.33 stays in-band, so only the EV gate (0.01 < 0.05) fires
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.30, expected_value=0.01)
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is False

    def test_value_bet_false_when_odds_below_band(self):
        # ip=0.60 → dec≈1.667 < value_min_odds(2.0) → no value bet despite high EV
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.60, expected_value=0.9)
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is False

    def test_value_bet_false_when_odds_above_band(self):
        # ip=0.10 → dec=10.0 > value_max_odds(4.0) → longshot gated out
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.10, expected_value=0.9)
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is False

    # ── informed-only support gate (Task 29) ──
    def test_value_bet_requires_feature_support(self):
        """A formless runner (historical_place_rate NaN → value_supported False) is
        NOT a value bet even with strong in-band EV — guards the longshot trap where
        an unraced/foreign runner gets base-rate × long odds = spurious positive EV."""
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.25, expected_value=0.9)  # dec=4.0, in band
        grp["value_supported"] = False
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is False

    def test_value_bet_allowed_when_supported(self):
        p = Predictor()
        grp = self._scored_grp(implied_prob=0.25, expected_value=0.5)  # dec=4.0, in band
        grp["value_supported"] = True
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is True

    def test_score_value_sets_supported_from_history(self):
        """_score_value marks runners with prior form supported, formless ones not."""
        live = _live_df(n_races=1, runners_per_race=2)
        live["historical_place_rate"] = [0.5, float("nan")]
        p = self._value_predictor(live, value_prob=0.30)
        scored = p._score(live)
        assert list(scored["value_supported"]) == [True, False]

    def test_value_bet_independent_of_top3_gate(self):
        """A short-priced favourite (below the 2.5 selection gate) can still be a
        value bet if its EV clears the value band (min_odds 2.0)."""
        # ip=0.45 → dec≈2.22: below the 2.5 selection min_odds (excluded_low_odds)
        # but inside the value band [2.0, 4.0].
        p = Predictor(min_selection_odds=2.50)
        grp = self._scored_grp(implied_prob=0.45, expected_value=0.3)
        race = p._build_race(grp)
        excl = race["excluded_low_odds"]
        assert len(excl) == 1
        assert excl[0]["value_bet"] is True


# ── favourite-longshot recalibration ─────────────────────────────────────────────

class TestFLRecalibration:
    """The F-L recalibrator remaps value_win_prob against the market price, so the
    whole value chain (edge / EV) reflects the corrected prob. Absent artifact or a
    disabled flag is a graceful no-op; odds-missing rows keep their original prob."""

    def _fake_recal(self):
        """OddsBandCalibrator-like stub: scales favourites (price<5) up by 1.5 and
        longshots (price>=5) down by 0.5 — a deterministic stand-in whose direction
        mirrors the real artifact without depending on the saved .pkl."""
        m = MagicMock()

        def predict(prob, odds):
            prob = np.asarray(prob, dtype=float)
            odds = np.asarray(odds, dtype=float)
            return np.clip(prob * np.where(odds < 5.0, 1.5, 0.5), 0.0, 1.0)

        m.predict.side_effect = predict
        return m

    def _predictor(self, live, value_prob=0.20, recal=None):
        p, _ = _predictor_with_mocks(live)
        p._value_model = _mock_catboost(value_prob)
        p._value_calibrator = None
        p._value_fl_calibrator = recal
        return p

    def test_recalibration_revises_favourite_up_longshot_down(self):
        live = _live_df(n_races=1, runners_per_race=2)
        # row0 favourite (implied 0.40 → dec 2.5), row1 longshot (implied 0.05 → dec 20)
        live["implied_prob"] = [0.40, 0.05]
        p = self._predictor(live, value_prob=0.20, recal=self._fake_recal())
        scored = p._score(live)
        vw = scored["value_win_prob"].to_numpy()
        assert vw[0] == pytest.approx(0.30)   # 0.20 * 1.5 (favourite, dec 2.5)
        assert vw[1] == pytest.approx(0.10)   # 0.20 * 0.5 (longshot, dec 20)
        # everything derived uses the recalibrated prob
        assert scored["value_edge"].iloc[0] == pytest.approx(0.30 - 0.40)
        assert scored["expected_value"].iloc[0] == pytest.approx(0.30 * 2.5 - 1.0)

    def test_no_op_when_calibrator_absent(self):
        live = _live_df(n_races=1, runners_per_race=3)
        live["implied_prob"] = 0.20
        p = self._predictor(live, value_prob=0.30, recal=None)
        scored = p._score(live)
        assert np.allclose(scored["value_win_prob"].values, 0.30)

    def test_missing_odds_row_keeps_prob(self):
        live = _live_df(n_races=1, runners_per_race=2)
        live["implied_prob"] = [0.40, np.nan]  # row1 has no usable price
        p = self._predictor(live, value_prob=0.20, recal=self._fake_recal())
        scored = p._score(live)
        vw = scored["value_win_prob"].to_numpy()
        assert vw[0] == pytest.approx(0.30)   # favourite recalibrated
        assert vw[1] == pytest.approx(0.20)   # no price → original prob preserved
        assert np.isfinite(vw).all()          # never degraded to NaN

    def test_apply_helper_is_identity_without_calibrator(self):
        p = Predictor()
        p._value_fl_calibrator = None
        out = p._apply_fl_recalibration(np.array([0.2, 0.3]), pd.Series([2.5, 20.0]))
        assert np.allclose(out, [0.2, 0.3])

    def test_config_flag_default_on(self):
        assert Predictor()._value_fl_recalibration is True

    # ── Stage-4 audit: independent vs market-adjusted persisted separately ────

    def test_independent_prob_persisted_alongside_market_adjusted(self):
        """value_win_prob_independent is the pre-F-L price-free line; when the
        recalibrator fires the two columns MUST differ and the derived EV/edge
        must use the market-adjusted one (live-repair-04 requirement 3)."""
        live = _live_df(n_races=1, runners_per_race=2)
        live["implied_prob"] = [0.40, 0.05]
        p = self._predictor(live, value_prob=0.20, recal=self._fake_recal())
        scored = p._score(live)
        ind = scored["value_win_prob_independent"].to_numpy()
        adj = scored["value_win_prob"].to_numpy()
        assert np.allclose(ind, 0.20)            # price never touched it
        assert adj[0] == pytest.approx(0.30)     # favourite revised up
        assert adj[1] == pytest.approx(0.10)     # longshot revised down
        assert not np.allclose(ind, adj)
        # derived quantities come from the ADJUSTED prob
        assert scored["expected_value"].iloc[0] == pytest.approx(0.30 * 2.5 - 1.0)

    def test_independent_equals_adjusted_when_recalibrator_absent(self):
        live = _live_df(n_races=1, runners_per_race=3)
        live["implied_prob"] = 0.20
        p = self._predictor(live, value_prob=0.30, recal=None)
        scored = p._score(live)
        assert np.allclose(scored["value_win_prob_independent"].values,
                           scored["value_win_prob"].values)

    def test_adjusted_reproducible_from_persisted_independent(self):
        """Exact-reproducibility invariant: value_win_prob recomputes from the
        PERSISTED (4dp) independent prob and the effective price."""
        live = _live_df(n_races=1, runners_per_race=2)
        live["implied_prob"] = [0.40, 0.05]
        recal = self._fake_recal()
        p = self._predictor(live, value_prob=0.2077, recal=recal)
        scored = p._score(live)
        ind = scored["value_win_prob_independent"].to_numpy()
        dec = 1.0 / live["implied_prob"].to_numpy()
        expect = np.round(np.asarray(
            recal.predict(ind, np.round(dec, 3)), dtype=float), 4)
        assert np.allclose(scored["value_win_prob"].to_numpy(), expect)

    def test_runner_dict_carries_both_probability_lines(self):
        row = pd.Series({
            "horse_id": "h1", "horse_name": "A", "implied_prob": 0.4,
            "value_win_prob": 0.31, "value_win_prob_independent": 0.22,
        })
        d = _runner_dict(row, rank=1)
        assert d["value_win_prob"] == pytest.approx(0.31)
        assert d["value_win_prob_independent"] == pytest.approx(0.22)

    def test_old_cache_without_independent_field_still_loads(self):
        """Artifact-compatibility: pre-Stage-4 cached runner dicts (no
        value_win_prob_independent key) must rehydrate with None, not raise."""
        d = {"rank": 1, "horse_id": "h", "horse_name": "n", "jockey": "",
             "trainer": "", "decimal_odds": 3.0, "odds_by_book": {},
             "best_odds": 3.0, "best_book": "b", "implied_prob": 0.33,
             "won_prob": 0.2, "won_prob_normalized": 0.2, "placed_2_prob": 0.4,
             "showed_prob": 0.5, "composite_score": 0.5, "each_way_value": False,
             "low_odds": False, "value_win_prob": 0.25, "value_edge": 0.01,
             "expected_value": 0.05, "value_bet": True, "value_supported": True,
             "data_completeness": 0.9, "confidence": "high",
             "first_time_runner": False}
        r = RunnerPrediction.from_dict(d)
        assert r.value_win_prob_independent is None
        assert r.value_win_prob == pytest.approx(0.25)

    def test_flag_off_skips_recalibrator_load(self):
        """With the flag off, _load_value_model must not load the recalibrator even
        when the artifact is on disk."""
        p = Predictor()
        if not (p._dir / f"catboost_won_{p._value_model_tag}.bin").exists():
            pytest.skip("value model artifact not present")
        p._value_fl_recalibration = False
        p._load_value_model()
        assert p._value_fl_calibrator is None

    def test_flag_on_loads_real_recalibrator(self):
        """The real saved artifact loads and recalibrates a favourite upward."""
        p = Predictor()
        if not (p._dir / f"fl_oddsband_{p._value_model_tag}_calib.pkl").exists() or \
           not (p._dir / f"catboost_won_{p._value_model_tag}.bin").exists():
            pytest.skip("recalibrator / value model artifact not present")
        p._value_fl_recalibration = True
        p._load_value_model()
        assert p._value_fl_calibrator is not None
        # favourite at 2.0 with a low raw prob is revised upward (F-L correction)
        out = np.asarray(p._value_fl_calibrator.predict([0.22], [2.0]), dtype=float)
        assert np.isfinite(out).all() and out[0] > 0.22


# ── odds-by-book (per-bookmaker prices + best) ───────────────────────────────────

class TestOddsByBook:
    """Each bookmaker's own board price is carried per runner; the best price is
    surfaced and used for value/EV. The layer defaults safely when absent."""

    def _raw(self) -> pd.DataFrame:
        """Un-fused unified rows: one runner across 3 books + a non-bookmaker
        (betsp) row that must be ignored + a second runner with a single book."""
        fresh = pd.Timestamp.now(tz="UTC").isoformat()
        return pd.DataFrame([
            {"race_date": "2026-06-15", "venue": "Ascot", "horse_id": "h1",
             "source": "livescorebet", "odds_decimal": 4.0, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh, "stale": False},
            {"race_date": "2026-06-15", "venue": "Ascot", "horse_id": "h1",
             "source": "paddy_power", "odds_decimal": 4.2, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh, "stale": False},
            {"race_date": "2026-06-15", "venue": "Ascot", "horse_id": "h1",
             "source": "boylesports", "odds_decimal": 3.9, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh, "stale": False},
            {"race_date": "2026-06-15", "venue": "Ascot", "horse_id": "h1",
             "source": "betsp", "odds_decimal": 5.0, "market_type": "WIN"},
            {"race_date": "2026-06-15", "venue": "Ascot", "horse_id": "h2",
             "source": "livescorebet", "odds_decimal": 7.5, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh, "stale": False},
        ])

    def test_map_collects_per_book(self):
        # No race_id/race_time on these rows → race_uid component is "".
        m = _odds_by_book_map(self._raw())
        assert m[("2026-06-15", "Ascot", "", "h1")] == {
            "livescorebet": 4.0, "paddy_power": 4.2, "boylesports": 3.9,
        }

    def test_map_excludes_non_bookmaker_sources(self):
        m = _odds_by_book_map(self._raw())
        assert "betsp" not in m[("2026-06-15", "Ascot", "", "h1")]

    def test_map_single_source_runner(self):
        m = _odds_by_book_map(self._raw())
        assert m[("2026-06-15", "Ascot", "", "h2")] == {"livescorebet": 7.5}

    def test_map_aggregates_books_across_per_source_race_ids(self):
        """Regression (req 3): each bookmaker mints its OWN race_id for the same
        physical race, so the cross-book identity must key on the shared race_time,
        not race_id — otherwise every runner collapses to a single book and
        best-price aggregation is lost."""
        fresh = pd.Timestamp.now(tz="UTC").isoformat()
        raw = pd.DataFrame([
            {"race_date": "2026-07-26", "venue": "Gowran Park", "horse_id": "h1",
             "race_id": "SBTE_2_1028491168", "race_time": "2026-07-26T16:05:00+00:00",
             "source": "livescorebet", "odds_decimal": 5.0, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh},
            {"race_date": "2026-07-26", "venue": "Gowran Park", "horse_id": "h1",
             "race_id": "45869743.10", "race_time": "2026-07-26T16:05:00+00:00",
             "source": "boylesports", "odds_decimal": 7.5, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh},
            {"race_date": "2026-07-26", "venue": "Gowran Park", "horse_id": "h1",
             "race_id": "pp-xyz", "race_time": "2026-07-26T16:05:00+00:00",
             "source": "paddy_power", "odds_decimal": 5.5, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh},
        ])
        m = _odds_by_book_map(raw)
        assert len(m) == 1                       # one race, not three
        (key, books), = m.items()
        assert books == {"livescorebet": 5.0, "boylesports": 7.5, "paddy_power": 5.5}

    def test_map_separates_same_venue_day_different_race_time(self):
        """req 3: two races at the same venue on the same day (distinct off-times)
        must NOT be collapsed — a bare venue-day key would merge them."""
        fresh = pd.Timestamp.now(tz="UTC").isoformat()
        raw = pd.DataFrame([
            {"race_date": "2026-07-26", "venue": "Ascot", "horse_id": "h1",
             "race_time": "2026-07-26T14:00:00+00:00",
             "source": "livescorebet", "odds_decimal": 3.0, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh},
            {"race_date": "2026-07-26", "venue": "Ascot", "horse_id": "h1",
             "race_time": "2026-07-26T15:30:00+00:00",
             "source": "livescorebet", "odds_decimal": 9.0, "market_type": "WIN",
             "validation_status": "VALID", "fetched_at": fresh},
        ])
        m = _odds_by_book_map(raw)
        assert len(m) == 2                       # not merged into one venue-day key
        assert {list(v.values())[0] for v in m.values()} == {3.0, 9.0}

    def test_best_of_books(self):
        assert _best_of_books({"livescorebet": 4.0, "paddy_power": 4.2,
                               "boylesports": 3.9}) == ("paddy_power", 4.2)

    def test_best_of_books_empty(self):
        assert _best_of_books({}) == (None, None)

    def test_attach_and_runner_dict_emits_map_and_best(self):
        live = pd.DataFrame([{
            "race_date": pd.Timestamp("2026-06-15", tz="UTC"), "venue": "Ascot",
            "horse_id": "h1", "horse_name": "Multi Book", "implied_prob": 0.25,
        }])
        live = _attach_odds_by_book(live, _odds_by_book_map(self._raw()))
        d = _runner_dict(live.iloc[0], rank=1)
        assert d["odds_by_book"] == {
            "livescorebet": 4.0, "paddy_power": 4.2, "boylesports": 3.9,
        }
        assert d["best_book"] == "paddy_power"
        assert d["best_odds"] == 4.2

    def test_runner_dict_defaults_safe_without_map(self):
        # Row with only a fused price → empty map, best_odds falls back to it.
        row = pd.Series({"horse_id": "h", "horse_name": "H", "_dec_odds": 5.0})
        d = _runner_dict(row, rank=1)
        assert d["odds_by_book"] == {}
        assert d["best_book"] is None
        assert d["best_odds"] == 5.0

    def test_effective_decimal_prefers_best(self):
        # implied 0.50 → fused decimal 2.0, but best board price is 6.0.
        assert _effective_decimal(pd.Series({"best_odds": 6.0, "implied_prob": 0.50})) == 6.0

    def test_effective_decimal_falls_back_to_fused(self):
        assert _effective_decimal(pd.Series({"implied_prob": 0.25})) == 4.0

    def test_value_bet_and_ev_use_best_price(self):
        """A favourite below the 2.5 selection gate whose best board price lands in
        the value band is a value bet — proving EV/value use best, not fused odds."""
        p = Predictor()
        grp = pd.DataFrame([{
            "horse_id": "h0", "horse_name": "A", "jockey": "J", "trainer": "T",
            "venue": "V", "race_date": pd.Timestamp("2026-06-15", tz="UTC"),
            "implied_prob": 0.55,  # fused decimal ≈ 1.82, below the 2.0 value band
            "best_odds": 3.0,       # best board price inside [2.0, 4.0]
            "won_prob": 0.20, "placed_2_prob": 0.40, "showed_prob": 0.60,
            "composite_score": 0.30, "expected_value": 0.3,
        }])
        race = p._build_race(grp)
        all_r = race["selections"] + race["excluded_low_odds"]
        assert all_r[0]["value_bet"] is True
        assert all_r[0]["best_odds"] == 3.0


def test_market_integrity_guard_drops_entire_contaminated_race_before_scoring():
    from models.predictor import Predictor

    live = pd.DataFrame([
        {
            "race_date": "2999-01-01", "race_time": "2999-01-01T14:00:00+00:00",
            "venue": "Ascot", "horse_id": "h1", "position": None,
            "validation_status": "INVALID",
        },
        {
            "race_date": "2999-01-01", "race_time": "2999-01-01T14:00:00+00:00",
            "venue": "Ascot", "horse_id": "h2", "position": None,
            "validation_status": "VALID",
        },
        {
            "race_date": "2999-01-01", "race_time": "2999-01-01T15:00:00+00:00",
            "venue": "Ascot", "horse_id": "h3", "position": None,
            "validation_status": "VALID",
        },
    ])
    guarded = Predictor.__new__(Predictor)._guard_upcoming(live)
    assert set(guarded["horse_id"]) == {"h3"}


def test_odds_by_book_map_ignores_invalid_market_rows():
    from models.predictor import _odds_by_book_map

    fresh = pd.Timestamp.now(tz="UTC").isoformat()
    raw = pd.DataFrame([
        {
            "race_date": "2026-07-25", "venue": "Ascot", "horse_id": "h1",
            "source": "livescorebet", "market_type": "WIN",
            "validation_status": "INVALID", "odds_decimal": 10.0,
            "fetched_at": fresh,
        },
        {
            "race_date": "2026-07-25", "venue": "Ascot", "horse_id": "h1",
            "source": "paddy_power", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 4.0,
            "fetched_at": fresh,
        },
    ])
    mapping = _odds_by_book_map(raw)
    assert mapping[("2026-07-25", "Ascot", "", "h1")] == {"paddy_power": 4.0}


def test_odds_by_book_map_rejects_entire_stale_source_race():
    """A stale quote must not survive as the executable best-price overlay.

    One stale row poisons the whole source race; salvaging its fresh-looking
    sibling would create a partial snapshot and could manufacture EV.
    """
    fresh = pd.Timestamp.now(tz="UTC").isoformat()
    raw = pd.DataFrame([
        {
            "race_date": "2026-07-25", "race_id": "race-a", "venue": "Ascot",
            "horse_id": "h1", "source": "boylesports", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 10.0,
            "fetched_at": fresh, "stale": True,
        },
        {
            "race_date": "2026-07-25", "race_id": "race-a", "venue": "Ascot",
            "horse_id": "h2", "source": "boylesports", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 4.0,
            "fetched_at": fresh, "stale": False,
        },
        {
            "race_date": "2026-07-25", "race_id": "race-b", "venue": "Ascot",
            "horse_id": "h1", "source": "paddy_power", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 3.5,
            "fetched_at": fresh, "stale": False,
        },
    ])
    mapping = _odds_by_book_map(raw)
    assert all("boylesports" not in books for books in mapping.values())
    assert mapping[("2026-07-25", "Ascot", "race-b", "h1")] == {
        "paddy_power": 3.5
    }


def test_odds_by_book_map_fails_closed_without_freshness_metadata():
    raw = pd.DataFrame([{
        "race_date": "2026-07-25", "venue": "Ascot", "horse_id": "h1",
        "source": "paddy_power", "market_type": "WIN",
        "validation_status": "VALID", "odds_decimal": 12.0,
    }])
    assert _odds_by_book_map(raw) == {}


def test_stale_price_cannot_reach_value_bet_or_suggestion_end_to_end():
    """Stage 2 acceptance item 5, proven across the whole chain in one test.

    A single runner has two bookmaker quotes for the same race: a stale
    boylesports quote at 50.0 (which would manufacture a huge, obviously fake
    EV if it were ever used) and a fresh paddy_power quote at 3.0 that carries
    no real edge against the model's price-free probability. This walks the
    exact path live data takes in production --
    ``_odds_by_book_map`` -> ``_attach_odds_by_book`` -> value-layer EV ->
    ``Predictor._build_race`` -> ``models.value.find_value_bets`` ->
    ``models.suggestions.suggest_bets`` -- and proves the stale 50.0 never
    surfaces as ``best_odds``, never drives ``value_bet``/``expected_value``,
    and never produces a bet suggestion. (``_runner_dict`` does not carry
    ``stale``/``fetched_at`` into the race dict, so ``find_value_bets``' own
    belt-and-suspenders staleness gate is a no-op on this path -- the whole of
    the protection here is ``_odds_by_book_map``'s whole-group rejection.)
    """
    from models.suggestions import suggest_bets
    from models.value import find_value_bets

    fresh = pd.Timestamp.now(tz="UTC").isoformat()
    raw = pd.DataFrame([
        {
            "race_date": "2026-07-25", "race_id": "race-a", "venue": "Ascot",
            "horse_id": "h1", "source": "boylesports", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 50.0,
            "fetched_at": fresh, "stale": True,
        },
        {
            "race_date": "2026-07-25", "race_id": "race-a", "venue": "Ascot",
            "horse_id": "h1", "source": "paddy_power", "market_type": "WIN",
            "validation_status": "VALID", "odds_decimal": 3.0,
            "fetched_at": fresh, "stale": False,
        },
    ])
    mapping = _odds_by_book_map(raw)
    # The stale boylesports quote never survives into the executable map.
    assert mapping == {("2026-07-25", "Ascot", "race-a", "h1"): {"paddy_power": 3.0}}

    live = pd.DataFrame([{
        "race_date": pd.Timestamp("2026-07-25", tz="UTC"), "venue": "Ascot",
        "horse_id": "h1", "horse_name": "Runner", "jockey": "J", "trainer": "T",
        "race_id": "race-a", "implied_prob": 1.0 / 3.0,
        "won_prob": 0.30, "placed_2_prob": 0.50, "showed_prob": 0.70,
        "composite_score": 0.30,
    }])
    live = _attach_odds_by_book(live, mapping)
    assert live.loc[0, "best_odds"] == 3.0
    assert live.loc[0, "best_book"] == "paddy_power"
    assert "boylesports" not in live.loc[0, "odds_by_book"]

    # Replicate Predictor._add_value_layer's EV formula off the price this test
    # controls: a model prob (0.30) with no real edge over the survivor's 3.0
    # price. If the stale 50.0 ever leaked through, this would compute EV=14.0
    # (0.30*50-1) instead of a small negative number.
    dec = live.apply(_effective_decimal, axis=1)
    assert dec.iloc[0] == 3.0
    live["value_win_prob"] = 0.30
    live["value_edge"] = live["value_win_prob"] - live["implied_prob"]
    live["expected_value"] = live["value_win_prob"] * dec - 1.0
    live["value_supported"] = True

    p = Predictor()
    race = p._build_race(live)
    all_r = race["selections"] + race["excluded_low_odds"]
    assert len(all_r) == 1
    runner = all_r[0]
    assert runner["best_odds"] == 3.0
    assert runner["expected_value"] < 0  # no manufactured edge from the stale price
    assert runner["value_bet"] is False

    violations = check_output_invariants([race])
    assert violations == []

    picks = find_value_bets(race)
    assert picks == []

    book = suggest_bets([race])
    assert book.suggestions == []
