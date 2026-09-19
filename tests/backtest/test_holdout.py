"""Tests for backtest.holdout — the honest GO/NO-GO holdout harness.

The load-bearing guarantees: (1) the leakage guard HARD-FAILS when the holdout
starts on/before the model's recorded train cutoff and degrades to a warning when
no cutoff is recorded; (2) EV>0 selection/settlement uses the PRE-OFF price with
2% commission on winnings and log CLV against the closing line; (3) the whole run
wires score → head-to-head → simulate → integrity → summary.json + ledger.csv.

CatBoost is never booted: a tiny stub with ``predict_proba`` stands in for the
frozen model so the orchestration is exercised in milliseconds.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from backtest.holdout import (
    FrozenModel,
    _train_cutoff_from_meta,
    run_holdout,
    score,
    simulate_ev_bets,
)


class _StubModel:
    """Returns a fixed P(win) per row from the first feature column."""

    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        p = self._probs[: len(X)]
        return np.column_stack([1.0 - p, p])


def _frozen(probs, *, cutoff=None, calibrator=None, fl=None):
    return FrozenModel(
        model=_StubModel(probs), calibrator=calibrator, fl_calibrator=fl,
        feature_cols=["field_size"], target="won", tag="stub",
        meta={}, train_cutoff=cutoff,
    )


def _panel():
    """Two 2-runner races inside the holdout window, with pre-off + closing prices."""
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


# ── cutoff parsing ────────────────────────────────────────────────────────────

class TestTrainCutoff:
    def test_flat_key(self):
        assert _train_cutoff_from_meta({"train_cutoff": "2026-02-28"}) == pd.Timestamp("2026-02-28")

    def test_nested_data_window(self):
        meta = {"data_window": {"max_race_date": "2026-03-01"}}
        assert _train_cutoff_from_meta(meta) == pd.Timestamp("2026-03-01")

    def test_absent_returns_none(self):
        assert _train_cutoff_from_meta({"feature_cols": ["x"]}) is None

    def test_unparseable_returns_none(self):
        assert _train_cutoff_from_meta({"train_cutoff": "not-a-date"}) is None


# ── leakage guard ─────────────────────────────────────────────────────────────

class TestLeakageGuard:
    def _run(self, monkeypatch, cutoff, start="2026-05-22"):
        frozen = _frozen([0.6, 0.3, 0.2, 0.7], cutoff=cutoff)
        monkeypatch.setattr("backtest.holdout.load_frozen_model", lambda _p: frozen)
        return run_holdout(
            model_path="catboost_won_stub.bin",
            holdout_start=start, holdout_end="2026-05-30",
            out_dir=str(self.tmp), panel_df=_panel())

    def test_hard_fails_when_holdout_on_or_before_cutoff(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        with pytest.raises(SystemExit, match="LEAKAGE GUARD"):
            self._run(monkeypatch, cutoff=pd.Timestamp("2026-06-01"))

    def test_runs_when_holdout_strictly_after_cutoff(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        summary = self._run(monkeypatch, cutoff=pd.Timestamp("2026-05-01"))
        assert summary["model"]["leakage_verified"] is True

    def test_degrades_to_warning_without_cutoff(self, monkeypatch, tmp_path):
        self.tmp = tmp_path
        summary = self._run(monkeypatch, cutoff=None)
        assert summary["model"]["leakage_verified"] is False


# ── EV simulation ─────────────────────────────────────────────────────────────

class TestSimulateEvBets:
    def test_only_positive_ev_is_bet(self):
        df = _panel()
        df["bet_price"] = df["ppwap"]
        df["close_price"] = df["odds_finish"]
        # norm_prob * price - 1: runner A 0.6*2-1=+0.2 (bet), B 0.2*4-1=-0.2 (skip),
        # C 0.1*5-1=-0.5 (skip), D 0.8*1.5-1=+0.2 (bet).
        ledger = simulate_ev_bets(df, [0.6, 0.2, 0.1, 0.8])
        assert len(ledger) == 2
        assert set(ledger["horse_name"]) == {"A", "D"}

    def test_winner_profit_nets_commission(self):
        df = _panel().iloc[[0]].copy()  # winner at ppwap 2.0
        df["bet_price"] = df["ppwap"]
        df["close_price"] = df["odds_finish"]
        ledger = simulate_ev_bets(df, [0.6], stake=10.0, commission=0.02)
        # gross = 10*(2-1)=10; net = 10*0.98 = 9.8
        assert ledger["profit"].iloc[0] == pytest.approx(9.8)

    def test_loser_forfeits_stake(self):
        df = _panel().iloc[[1]].copy()  # B loses at 4.0; force a bet with high prob
        df["bet_price"] = df["ppwap"]
        df["close_price"] = df["odds_finish"]
        ledger = simulate_ev_bets(df, [0.9], stake=10.0)
        assert ledger["profit"].iloc[0] == pytest.approx(-10.0)

    def test_clv_log_is_log_price_over_close(self):
        df = _panel().iloc[[0]].copy()
        df["bet_price"] = df["ppwap"]      # 2.0
        df["close_price"] = df["odds_finish"]  # 1.8
        ledger = simulate_ev_bets(df, [0.6])
        assert ledger["clv_log"].iloc[0] == pytest.approx(np.log(2.0 / 1.8))


# ── scoring ───────────────────────────────────────────────────────────────────

class TestScore:
    def test_calibrator_applied(self):
        df = _panel()
        df["bet_price"] = df["ppwap"]
        frozen = _frozen([0.6, 0.3, 0.2, 0.7],
                         calibrator=type("C", (), {"predict": staticmethod(lambda p: p * 0.5)})())
        out = score(frozen, df)
        assert out == pytest.approx([0.3, 0.15, 0.1, 0.35])


# ── end-to-end orchestration ──────────────────────────────────────────────────

class TestRunHoldout:
    def test_writes_summary_and_ledger(self, monkeypatch, tmp_path):
        frozen = _frozen([0.6, 0.3, 0.2, 0.7], cutoff=pd.Timestamp("2026-05-01"))
        monkeypatch.setattr("backtest.holdout.load_frozen_model", lambda _p: frozen)
        summary = run_holdout(
            model_path="catboost_won_stub.bin",
            holdout_start="2026-05-22", holdout_end="2026-05-30",
            out_dir=str(tmp_path), panel_df=_panel())

        assert "model_beats_market_logloss" in summary
        assert (tmp_path / "summary.json").exists()
        assert (tmp_path / "ledger.csv").exists()
        on_disk = json.loads((tmp_path / "summary.json").read_text())
        assert "model_beats_market_logloss" in on_disk
        assert on_disk["n_races"] == 2
        # integrity suite ran and includes the look-ahead guard
        names = {c["name"] for c in summary["integrity"]}
        assert "lookahead_bias" in names

    def test_empty_window_raises(self, monkeypatch, tmp_path):
        frozen = _frozen([0.6, 0.3, 0.2, 0.7])
        monkeypatch.setattr("backtest.holdout.load_frozen_model", lambda _p: frozen)
        with pytest.raises(SystemExit, match="No runners"):
            run_holdout(
                model_path="catboost_won_stub.bin",
                holdout_start="2030-01-01", holdout_end="2030-01-31",
                out_dir=str(tmp_path), panel_df=_panel())
