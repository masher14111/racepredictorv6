"""End-to-end smoke test for the rebuild — the gate that says "the chain wires up".

Cheap, CatBoost-free, network-free. It proves the new model line and honesty
machinery import and connect end to end:

  1. Every new module imports (the LightGBM v3 line, the de-vig / head-to-head /
     integrity / holdout backtest stack, and the UI honesty / compare panels).
  2. :func:`backtest.holdout.run_holdout` produces a GO/NO-GO **verdict dict** on a
     tiny in-memory fixture, driven by a stub model so CatBoost never boots.
  3. The verdict block surfaced to the UI (``ui.model_honesty.verdict_lines``)
     parses that summary into a graded GO/NO-GO line.
  4. ``models.predict_unified`` enriches a synthetic predictions payload with the
     additive ``verdict`` block and per-runner ``catboost_*`` / ``market_prob``
     keys (skipped when the CatBoost models are not on disk).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest


def test_new_modules_import():
    """Every module the rebuild added must import without side effects."""
    for name in (
        "models.lgbm_softmax",
        "models.train_lgbm",
        "models.devig",
        "models.head_to_head",
        "models.predict_unified",
        "backtest.integrity",
        "backtest.holdout",
        "ui.model_honesty",
        "ui.model_compare",
    ):
        assert importlib.import_module(name) is not None


class _StubModel:
    """Fixed P(win) per row — stands in for the frozen CatBoost model."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        p = self._probs[: len(X)]
        return np.column_stack([1.0 - p, p])


def _panel() -> pd.DataFrame:
    """Two 2-runner races inside the holdout window, pre-off + closing prices."""
    return pd.DataFrame({
        "race_date": pd.to_datetime(
            ["2026-05-25", "2026-05-25", "2026-05-26", "2026-05-26"], utc=True),
        "venue": ["Ascot", "Ascot", "York", "York"],
        "horse_id": [1, 2, 3, 4],
        "horse_name": ["A", "B", "C", "D"],
        "race_uid": ["Ascot|2026-05-25", "Ascot|2026-05-25",
                     "York|2026-05-26", "York|2026-05-26"],
        "market_type": ["WIN"] * 4,
        "won": [1, 0, 0, 1],
        "field_size": [2, 2, 2, 2],
        "ppwap": [2.0, 4.0, 5.0, 1.5],
        "morningwap": [2.0, 4.0, 5.0, 1.5],
        "odds_finish": [1.8, 4.5, 6.0, 1.4],
    })


def test_holdout_produces_verdict_and_ui_grades_it(monkeypatch, tmp_path):
    """run_holdout → verdict dict on disk → the UI parses it into a graded line."""
    from backtest.holdout import FrozenModel, run_holdout
    from ui import model_honesty as MH

    frozen = FrozenModel(
        model=_StubModel([0.6, 0.3, 0.2, 0.7]), calibrator=None, fl_calibrator=None,
        feature_cols=["field_size"], target="won", tag="stub",
        meta={}, train_cutoff=pd.Timestamp("2026-05-01"),
    )
    monkeypatch.setattr("backtest.holdout.load_frozen_model", lambda _p: frozen)

    summary = run_holdout(
        model_path="catboost_won_stub.bin",
        holdout_start="2026-05-22", holdout_end="2026-05-30",
        out_dir=str(tmp_path), panel_df=_panel())

    # (1) a verdict dict was produced with the load-bearing GO/NO-GO gate.
    assert isinstance(summary, dict)
    assert "model_beats_market_logloss" in summary
    assert isinstance(summary["model_beats_market_logloss"], bool)
    assert (tmp_path / "summary.json").exists()
    on_disk = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert on_disk["n_races"] == 2

    # (2) the UI turns a predictions-style verdict block into a graded line.
    h2h = summary["head_to_head"]
    verdict = {"catboost": {
        "model_beats_market_logloss": summary["model_beats_market_logloss"],
        "go": summary["model_beats_market_logloss"],
        "model_log_loss": h2h["model"]["log_loss"],
        "market_log_loss": h2h["market"]["log_loss"],
        "mean_clv_log": summary["betting"].get("mean_clv_log"),
    }}
    lines = MH.verdict_lines(verdict)
    assert len(lines) == 1
    assert lines[0]["status"] in ("go", "nogo")


def test_predictions_json_gets_additive_keys(tmp_path, monkeypatch):
    """models.predict_unified adds the verdict block + per-runner additive keys."""
    import models.predictor as predictor_mod
    from models.predictor import Predictor

    monkeypatch.setattr(predictor_mod, "_CACHE_PATH", tmp_path / "predictions.json")
    monkeypatch.setattr(
        predictor_mod, "_INFERENCE_FEATURES_PATH", tmp_path / "inf.parquet")

    p = Predictor()
    if not p.load():
        pytest.skip("CatBoost models not available on disk")

    today = pd.Timestamp.now("UTC").normalize()
    rows = []
    for ri, venue in enumerate(["Ascot", "York"]):
        rt = f"{today.date()}T1{ri}:00"
        for hi, imp in enumerate([0.45, 0.30, 0.18, 0.12]):
            rows.append(dict(
                venue=venue, race_time=rt, race_date=today,
                race_uid=f"{venue}_{rt}",
                horse_id=f"h{ri}{hi}", horse_name=f"Horse{ri}{hi}",
                jockey_name="J Smith", trainer_name="T Jones",
                implied_prob=imp, morningwap=round(1.0 / imp, 2),
                ppwap=round(1.0 / imp, 2),
                historical_place_rate=0.3, horse_career_runs=10,
            ))
    live = pd.DataFrame(rows)
    p.predict(unified=live.iloc[0:0], live=live)
    written = json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))

    assert isinstance(written.get("verdict"), dict)
    seen = False
    for race in written["races"]:
        for r in race["selections"] + race["excluded_low_odds"]:
            seen = True
            for k in ("catboost_win_prob", "market_prob", "ev_catboost"):
                assert k in r, f"{k} missing from runner"
    assert seen, "no runners in payload"
