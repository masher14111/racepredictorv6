"""Tests for the live-CLV harness (scripts/clv_live).

Covers the three load-bearing behaviours: (1) only the system's value picks are
selected and at the best board price; (2) logging is idempotent and bets carry
the experiment tag + the price taken; (3) the report computes CLV (price taken /
SP - 1) only over the tagged, settled picks.
"""
from __future__ import annotations

import json
from argparse import Namespace

import pytest

import scripts.clv_live as clv
from utils.bet_tracker import BetTracker


def _predictions(tmp_path):
    payload = {
        "races": [{
            "venue": "Ascot", "race_uid": "Ascot_1", "race_time": "2099-01-01T14:00:00+00:00",
            "selections": [
                {"race_uid": "Ascot_1", "horse_id": "h1", "horse_name": "Valuer",
                 "value_bet": True, "best_odds": 3.5, "best_book": "boylesports",
                 "decimal_odds": 3.2, "value_win_prob": 0.35, "value_edge": 0.1},
                {"race_uid": "Ascot_1", "horse_id": "h2", "horse_name": "NoValue",
                 "value_bet": False, "best_odds": 2.0, "decimal_odds": 2.0},
            ],
            "excluded_low_odds": [],
        }]
    }
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    t = BetTracker(db_path=str(tmp_path / "bets.db"), flat_stake=10.0)
    monkeypatch.setattr(clv, "_tracker", lambda: t)
    return t


def test_value_picks_selects_only_flagged_at_board_price(tmp_path):
    picks = clv._value_picks(_predictions(tmp_path))
    assert len(picks) == 1
    assert picks[0]["horse_name"] == "Valuer"
    assert picks[0]["odds"] == 3.5  # best board price, not decimal_odds


def test_log_places_tagged_bet_and_is_idempotent(tmp_path, tracker, capsys):
    args = Namespace(predictions=str(_predictions(tmp_path)), bet_type="win")
    assert clv.cmd_log(args) == 0
    bets = tracker.all_bets()
    assert len(bets) == 1
    b = bets[0]
    assert b["odds_decimal"] == 3.5 and b["horse_name"] == "Valuer"
    assert str(b["notes"]).startswith(clv._TAG)

    # Second run logs nothing new (duplicate guard) — idempotent.
    clv.cmd_log(args)
    assert len(tracker.all_bets()) == 1


def test_report_computes_clv_over_tagged_settled(tmp_path, tracker, capsys):
    clv.cmd_log(Namespace(predictions=str(_predictions(tmp_path)), bet_type="win"))
    bet_id = tracker.all_bets()[0]["id"]
    # Settle a winner whose SP (3.0) is shorter than the 3.5 taken → positive CLV.
    tracker.settle_bet(bet_id, "win", closing_odds=3.0)

    clv.cmd_report(Namespace())
    out = capsys.readouterr().out
    assert "settled picks : 1" in out
    # CLV = 3.5/3.0 - 1 = +16.67%
    assert "+16.67%" in out
    assert "POSITIVE" in out
