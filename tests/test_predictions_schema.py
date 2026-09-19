"""Schema contract for data/predictions.json after the unified multi-line wiring.

Verifies that the additive enrichment in :mod:`models.predict_unified`:
  * keeps every pre-existing top-level and per-runner key intact (additive only),
  * adds the top-level ``verdict`` block with a GO/NO-GO entry per model line,
  * adds the per-runner ``catboost_win_prob`` / ``market_prob`` / ``ev_catboost``
    keys, and the ``lgbm_*`` keys when the LightGBM line is available,
  * degrades gracefully (omits the ``lgbm_*`` keys) when the model file is absent.

Drives a synthetic live frame through a loaded ``Predictor`` so the contract is
exercised deterministically regardless of whether live racecards are present.
"""
from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

import models.predictor as predictor_mod
from models.predictor import Predictor

# Per-runner keys that existed before the unified wiring — must never disappear.
_LEGACY_RUNNER_KEYS = {
    "rank", "horse_id", "horse_name", "jockey", "trainer", "decimal_odds",
    "odds_by_book", "best_odds", "best_book", "implied_prob", "won_prob",
    "won_prob_normalized", "placed_2_prob", "showed_prob", "composite_score",
    "each_way_value", "low_odds", "value_win_prob", "value_edge",
    "expected_value", "value_bet", "value_supported", "data_completeness",
    "confidence", "first_time_runner",
}
_LEGACY_TOP_KEYS = {
    "generated_at", "model_targets", "total_races", "total_runners", "races",
}


def _synthetic_live() -> pd.DataFrame:
    """Two races of fully-priced runners, dated today so the upcoming-guard keeps them."""
    today = pd.Timestamp(date.today(), tz="UTC")
    rows = []
    for ri, (venue, rt) in enumerate(
        [("Ascot", "2026-06-19T14:00"), ("York", "2026-06-19T15:30")]
    ):
        for hi, imp in enumerate([0.45, 0.30, 0.18, 0.12]):
            rows.append(
                dict(
                    venue=venue, race_time=rt, race_date=today,
                    race_uid=f"{venue}_{rt}",
                    horse_id=f"h{ri}{hi}", horse_name=f"Horse{ri}{hi}",
                    jockey_name="J Smith", trainer_name="T Jones",
                    implied_prob=imp, morningwap=round(1.0 / imp, 2),
                    ppwap=round(1.0 / imp, 2),
                    historical_place_rate=0.3, horse_career_runs=10,
                )
            )
    return pd.DataFrame(rows)


@pytest.fixture
def payload(tmp_path, monkeypatch):
    """Run the predictor against a synthetic frame, returning the written payload.

    The cache / inference-feature paths are redirected to ``tmp_path`` so the test
    never clobbers the real data/ artefacts.
    """
    monkeypatch.setattr(predictor_mod, "_CACHE_PATH", tmp_path / "predictions.json")
    monkeypatch.setattr(
        predictor_mod, "_INFERENCE_FEATURES_PATH", tmp_path / "inference_features.parquet"
    )
    p = Predictor()
    if not p.load():
        pytest.skip("CatBoost models not available on disk")
    live = _synthetic_live()
    # unified=empty so the odds-by-book map / disk read is skipped; effective price
    # falls back to the fused odds derived from implied_prob.
    p.predict(unified=live.iloc[0:0], live=live)
    written = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    return written


def test_legacy_top_level_keys_intact(payload):
    assert _LEGACY_TOP_KEYS.issubset(payload.keys())
    assert payload["total_races"] == len(payload["races"]) == 2


def test_verdict_block_present(payload):
    verdict = payload["verdict"]
    assert isinstance(verdict, dict)
    # An entry per model line; lgbm carries the real holdout GO/NO-GO.
    assert "lgbm" in verdict and "catboost" in verdict
    lgbm = verdict["lgbm"]
    if lgbm.get("available") is not False:
        assert "model_beats_market_logloss" in lgbm
        assert "go" in lgbm


def test_legacy_runner_keys_intact(payload):
    for race in payload["races"]:
        for runner in race["selections"] + race["excluded_low_odds"]:
            assert _LEGACY_RUNNER_KEYS.issubset(runner.keys()), (
                f"missing legacy keys: {_LEGACY_RUNNER_KEYS - runner.keys()}"
            )


def test_additive_unified_keys_present(payload):
    seen_runner = False
    for race in payload["races"]:
        for runner in race["selections"] + race["excluded_low_odds"]:
            seen_runner = True
            # Always-present additive keys (value may be null).
            for k in ("catboost_win_prob", "market_prob", "ev_catboost"):
                assert k in runner, f"{k} missing from runner"
    assert seen_runner, "no runners in payload to validate"


def test_market_prob_is_devigged(payload):
    """De-vigged market probs sum to ~1.0 within a complete-book race."""
    race = payload["races"][0]
    runners = race["selections"] + race["excluded_low_odds"]
    mkt = [r["market_prob"] for r in runners if r.get("market_prob") is not None]
    assert len(mkt) == len(runners) > 1
    assert sum(mkt) == pytest.approx(1.0, abs=1e-6)


def test_ev_consistency(payload):
    """ev_catboost = catboost_win_prob * price - 1 at the executable board price."""
    for race in payload["races"]:
        for r in race["selections"] + race["excluded_low_odds"]:
            cb = r.get("catboost_win_prob")
            price = r.get("best_odds") or r.get("decimal_odds")
            if cb is None or price is None or r.get("ev_catboost") is None:
                continue
            assert r["ev_catboost"] == pytest.approx(cb * price - 1.0, abs=1e-3)


def test_lgbm_keys_paired(payload):
    """lgbm_win_prob and ev_lgbm appear together, or neither does (graceful omit)."""
    for race in payload["races"]:
        for r in race["selections"] + race["excluded_low_odds"]:
            assert ("lgbm_win_prob" in r) == ("ev_lgbm" in r)


def test_lgbm_omitted_when_model_missing(tmp_path, monkeypatch):
    """With no LightGBM model file on the model dir, lgbm_* keys are omitted."""
    monkeypatch.setattr(predictor_mod, "_CACHE_PATH", tmp_path / "predictions.json")
    monkeypatch.setattr(
        predictor_mod, "_INFERENCE_FEATURES_PATH", tmp_path / "inf.parquet"
    )
    p = Predictor()
    if not p.load():
        pytest.skip("CatBoost models not available on disk")
    # Point the predictor at a model dir with the CatBoost models but no LightGBM.
    import models.predict_unified as pu
    monkeypatch.setattr(pu, "_LGBM_MODEL_FILENAME", "does_not_exist_lgbm.txt")
    live = _synthetic_live()
    p.predict(unified=live.iloc[0:0], live=live)
    written = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))
    for race in written["races"]:
        for r in race["selections"] + race["excluded_low_odds"]:
            assert "lgbm_win_prob" not in r
            assert "ev_lgbm" not in r
            # Non-lgbm additive keys still present.
            assert "catboost_win_prob" in r and "market_prob" in r
