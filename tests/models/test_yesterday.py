"""Tests for the Yesterday's Bet Predictor settlement engine.

The model-scoring step (``_score_day``) loads the real CatBoost models, so the
end-to-end tests monkeypatch it with a deterministic scorer — that keeps the focus
on the engine's selection, settlement, and accounting logic (which is where the
correctness risk is) without depending on trained artefacts.
"""
import numpy as np
import pandas as pd
import pytest

from models import yesterday as Y
from models.yesterday import (
    YesterdayConfig,
    _price,
    _settle_bet,
    _settled_days,
    _summarise,
    _top_pick_rows,
    run_yesterday,
)


# ── fixtures ────────────────────────────────────────────────────────────────────

def _matrix() -> pd.DataFrame:
    """Two settled races on 2026-06-12 + one unsettled (no position) day."""
    rows = []
    # Race A — Cork, 3 runners. Favourite (winner) at 2.0, two losers.
    rows += [
        dict(race_date="2026-06-12", race_time="2026-06-12 14:00", venue="Cork",
             horse_id="a1", horse_name="Alpha", odds_finish=2.0, implied_prob=0.5,
             position=1, won_prob=0.55, value_win_prob=0.60, value_supported=True,
             historical_place_rate=0.4, horse_career_runs=10),
        dict(race_date="2026-06-12", race_time="2026-06-12 14:00", venue="Cork",
             horse_id="a2", horse_name="Bravo", odds_finish=4.0, implied_prob=0.25,
             position=2, won_prob=0.30, value_win_prob=0.20, value_supported=True,
             historical_place_rate=0.3, horse_career_runs=8),
        dict(race_date="2026-06-12", race_time="2026-06-12 14:00", venue="Cork",
             horse_id="a3", horse_name="Charlie", odds_finish=6.0, implied_prob=0.166,
             position=3, won_prob=0.15, value_win_prob=0.10, value_supported=True,
             historical_place_rate=0.2, horse_career_runs=5),
    ]
    # Race B — Naas, 2 runners. Winner is the longer price (5.0).
    rows += [
        dict(race_date="2026-06-12", race_time="2026-06-12 15:30", venue="Naas",
             horse_id="b1", horse_name="Delta", odds_finish=1.5, implied_prob=0.66,
             position=2, won_prob=0.62, value_win_prob=0.55, value_supported=True,
             historical_place_rate=0.5, horse_career_runs=12),
        dict(race_date="2026-06-12", race_time="2026-06-12 15:30", venue="Naas",
             horse_id="b2", horse_name="Echo", odds_finish=5.0, implied_prob=0.20,
             position=1, won_prob=0.40, value_win_prob=0.45, value_supported=True,
             historical_place_rate=0.35, horse_career_runs=9),
    ]
    # Unsettled day (today) — no positions.
    rows += [
        dict(race_date="2026-06-17", race_time="2026-06-17 14:00", venue="Cork",
             horse_id="c1", horse_name="Foxtrot", odds_finish=np.nan, implied_prob=0.3,
             position=np.nan, won_prob=0.5, value_win_prob=0.5, value_supported=True,
             historical_place_rate=0.3, horse_career_runs=7),
    ]
    return pd.DataFrame(rows)


def _fake_score(day_df: pd.DataFrame) -> pd.DataFrame:
    """Stand-in for _score_day: the fixture already carries the score columns."""
    return day_df.copy()


# ── pure helpers ────────────────────────────────────────────────────────────────

def test_settled_days_lists_only_days_with_positions():
    days = _settled_days(_matrix())
    assert days == ["2026-06-12"]


def test_settled_days_empty_when_no_positions():
    df = _matrix()
    df["position"] = np.nan
    assert _settled_days(df) == []


def test_price_prefers_starting_price_then_implied():
    df = pd.DataFrame({
        "odds_finish": [3.0, np.nan, 0.0],
        "implied_prob": [0.2, 0.25, 0.5],
    })
    out = _price(df).tolist()
    assert out[0] == pytest.approx(3.0)        # SP wins
    assert out[1] == pytest.approx(4.0)        # 1/0.25 fallback
    assert out[2] == pytest.approx(2.0)        # SP invalid -> 1/0.5


def test_settle_bet_win_returns_full_odds():
    row = pd.Series(dict(venue="Cork", horse_name="Alpha", horse_id="a1",
                         position=1, race_time="t"))
    bet = _settle_bet(row, 2.0, YesterdayConfig(bet_type="win", stake=10),
                      is_value=True, model_prob=0.6)
    assert bet["outcome"] == "win"
    assert bet["gross_return"] == pytest.approx(20.0)
    assert bet["profit"] == pytest.approx(10.0)


def test_settle_bet_loss_loses_stake():
    row = pd.Series(dict(venue="Cork", horse_name="Bravo", horse_id="a2",
                         position=4, race_time="t"))
    bet = _settle_bet(row, 4.0, YesterdayConfig(bet_type="win", stake=10),
                      is_value=False, model_prob=0.3)
    assert bet["outcome"] == "lose"
    assert bet["profit"] == pytest.approx(-10.0)


def test_settle_bet_each_way_place_pays_place_leg_only():
    row = pd.Series(dict(venue="Cork", horse_name="Charlie", horse_id="a3",
                         position=3, race_time="t"))
    cfg = YesterdayConfig(bet_type="each_way", stake=10, ew_places=3, ew_fraction=0.2)
    bet = _settle_bet(row, 6.0, cfg, is_value=True, model_prob=0.2)
    # place-only: half stake on the place leg at (odds-1)*0.2 + 1 = 2.0 -> 5*2.0 = 10.0
    assert bet["outcome"] == "place"
    assert bet["gross_return"] == pytest.approx(10.0)
    assert bet["profit"] == pytest.approx(0.0)


def test_summarise_computes_roi_and_rates():
    bets = [
        {"outcome": "win", "stake": 10.0, "profit": 10.0},
        {"outcome": "lose", "stake": 10.0, "profit": -10.0},
        {"outcome": "place", "stake": 10.0, "profit": 2.0},
    ]
    s = _summarise(bets, YesterdayConfig(stake=10, bankroll=1000))
    assert s["n_bets"] == 3
    assert s["n_winners"] == 1
    assert s["win_rate"] == pytest.approx(33.33, abs=0.01)
    assert s["place_rate"] == pytest.approx(66.67, abs=0.01)
    assert s["total_profit"] == pytest.approx(2.0)
    assert s["roi_pct"] == pytest.approx(6.67, abs=0.01)
    assert s["final_bankroll"] == pytest.approx(1002.0)


def test_summarise_empty_is_safe():
    s = _summarise([], YesterdayConfig(bankroll=500))
    assert s["n_bets"] == 0
    assert s["final_bankroll"] == pytest.approx(500.0)


def test_top_pick_rows_ranks_by_win_prob_and_gates_min_odds():
    grp = pd.DataFrame({
        "won_prob": [0.6, 0.5, 0.4],
        "_price": [1.2, 3.0, 5.0],
        "horse_name": ["x", "y", "z"],
    })
    cfg = YesterdayConfig(strategy="top_pick", min_odds=2.0, top_n=1)
    picked = _top_pick_rows(grp, cfg)
    # 0.6 is highest but priced 1.2 < min 2.0, so the 0.5 @ 3.0 runner is taken.
    assert list(picked["horse_name"]) == ["y"]


# ── end-to-end (scoring monkeypatched) ───────────────────────────────────────────

def test_run_yesterday_no_results():
    df = _matrix()
    df["position"] = np.nan
    res = run_yesterday(YesterdayConfig(), df=df)
    assert res.has_results is False
    assert res.date is None
    assert "No settled results" in res.message


def test_run_yesterday_top_pick_settles_blind(monkeypatch):
    monkeypatch.setattr(Y, "_score_day", _fake_score)
    res = run_yesterday(YesterdayConfig(strategy="top_pick", bet_type="win", stake=10,
                                        min_odds=1.01), df=_matrix())
    assert res.has_results is True
    assert res.date == "2026-06-12"
    assert res.n_races == 2
    # one pick per race -> 2 bets
    assert res.summary["n_bets"] == 2
    # Race A top pick (Alpha @2.0) won; Race B top pick (Delta @1.5) lost.
    by_horse = {b["horse_name"]: b for b in res.bets}
    assert by_horse["Alpha"]["outcome"] == "win"
    assert by_horse["Delta"]["outcome"] == "lose"
    assert res.summary["total_profit"] == pytest.approx(10.0 - 10.0)


def test_run_yesterday_value_strategy_uses_value_gate(monkeypatch):
    monkeypatch.setattr(Y, "_score_day", _fake_score)
    res = run_yesterday(YesterdayConfig(strategy="value", bet_type="win", stake=10),
                        df=_matrix())
    assert res.has_results is True
    # Every settled bet must be flagged as a value pick and have a real result.
    assert all(b["is_value"] for b in res.bets)
    assert all(b["outcome"] in ("win", "place", "lose") for b in res.bets)
    # value_races counter never exceeds races scanned
    assert 0 <= res.n_value_races <= res.n_races


def test_run_yesterday_respects_target_date(monkeypatch):
    monkeypatch.setattr(Y, "_score_day", _fake_score)
    res = run_yesterday(YesterdayConfig(strategy="top_pick", target_date="2026-06-12"),
                        df=_matrix())
    assert res.date == "2026-06-12"
    # An out-of-range date falls back to the most recent settled day.
    res2 = run_yesterday(YesterdayConfig(strategy="top_pick", target_date="1999-01-01"),
                         df=_matrix())
    assert res2.date == "2026-06-12"
