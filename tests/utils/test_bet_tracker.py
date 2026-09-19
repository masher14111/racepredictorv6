"""Tests for utils/bet_tracker.py."""
from datetime import datetime

import pytest

from utils.bet_tracker import (
    BetTracker,
    DuplicateBetError,
    RaceStartedError,
    StopLossError,
    Strategy,
)


def _tracker(tmp_path, **kwargs) -> BetTracker:
    return BetTracker(db_path=str(tmp_path / "test.db"), **kwargs)


# ------------------------------------------------------------------
# Bankroll initialisation
# ------------------------------------------------------------------

def test_initial_bankroll(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=500.0)
    assert bt.bankroll == 500.0
    assert bt.initial_bankroll == 500.0


def test_initial_bankroll_not_reset_on_second_init(tmp_path):
    _tracker(tmp_path, initial_bankroll=500.0)
    bt2 = _tracker(tmp_path, initial_bankroll=9999.0)
    assert bt2.initial_bankroll == 500.0  # first value wins


def test_bankroll_history_has_init_row(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=200.0)
    hist = bt.bankroll_history()
    assert len(hist) == 1
    assert hist.iloc[0]["event_type"] == "init"
    assert hist.iloc[0]["balance"] == 200.0


# ------------------------------------------------------------------
# Stop-loss
# ------------------------------------------------------------------

def test_not_stopped_initially(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0, stop_loss_pct=0.20)
    assert not bt.is_stopped()


def test_stopped_after_loss(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0, stop_loss_pct=0.20, flat_stake=10.0)
    # floor=800; after 21 losses bankroll=790 < 800
    for _ in range(21):
        bid = bt.record_bet("Horse A", 5.0, 10.0, strategy=Strategy.FLAT)
        bt.settle_bet(bid, "lose")
    assert bt.is_stopped()


def test_stop_loss_blocks_new_bets(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=100.0, stop_loss_pct=0.20, flat_stake=10.0)
    for _ in range(3):
        bid = bt.record_bet("H", 5.0, 10.0)
        bt.settle_bet(bid, "lose")
    # bankroll = 70 < floor 80
    assert bt.is_stopped()
    with pytest.raises(StopLossError):
        bt.record_bet("H", 5.0, 10.0)


def test_recommend_stake_zero_when_stopped(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=100.0, stop_loss_pct=0.20, flat_stake=10.0)
    for _ in range(3):
        bid = bt.record_bet("H", 5.0, 10.0)
        bt.settle_bet(bid, "lose")
    assert bt.recommend_stake(5.0, 0.3) == 0.0


# ------------------------------------------------------------------
# Stake sizing
# ------------------------------------------------------------------

def test_flat_stake(tmp_path):
    bt = _tracker(tmp_path, flat_stake=15.0)
    assert bt.recommend_stake(5.0, 0.3, Strategy.FLAT) == 15.0


def test_kelly_no_edge_returns_zero(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    # implied prob = 1/5 = 0.2, win_prob = 0.15 → negative edge
    assert bt.kelly_stake(5.0, 0.15) == 0.0


def test_kelly_with_edge(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    # b=4, p=0.3, q=0.7 → f = (4*0.3 - 0.7)/4 = 0.125 → stake = 125
    stake = bt.kelly_stake(5.0, 0.3)
    assert abs(stake - 125.0) < 0.01


def test_fractional_kelly(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0, kelly_fraction=0.25)
    full = bt.kelly_stake(5.0, 0.3)
    frac = bt.recommend_stake(5.0, 0.3, Strategy.FRACTIONAL_KELLY)
    assert abs(frac - full * 0.25) < 0.01


def test_kelly_invalid_odds(tmp_path):
    bt = _tracker(tmp_path)
    assert bt.kelly_stake(1.0, 0.5) == 0.0  # b=0
    assert bt.kelly_stake(0.5, 0.5) == 0.0  # b<0


# ------------------------------------------------------------------
# Record bets
# ------------------------------------------------------------------

def test_record_bet_deducts_bankroll(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.record_bet("Dancer", 4.0, 50.0)
    assert bt.bankroll == 950.0


def test_record_bet_returns_incrementing_ids(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    id1 = bt.record_bet("A", 4.0, 10.0)
    id2 = bt.record_bet("B", 5.0, 10.0)
    assert id2 == id1 + 1


def test_record_bet_invalid_bet_type(tmp_path):
    bt = _tracker(tmp_path)
    with pytest.raises(ValueError, match="bet_type"):
        bt.record_bet("X", 4.0, 10.0, bet_type="bogus")


def test_record_bet_stores_metadata(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=500.0)
    bid = bt.record_bet(
        "Sea Breeze", 6.0, 20.0,
        bet_type="each_way",
        venue="Leopardstown",
        composite_score=0.75,
        won_prob=0.18,
        strategy=Strategy.FLAT,
        notes="test note",
    )
    row = bt.get_bet(bid)
    assert row["horse_name"] == "Sea Breeze"
    assert row["bet_type"] == "each_way"
    assert row["venue"] == "Leopardstown"
    assert row["composite_score"] == pytest.approx(0.75)
    assert row["notes"] == "test note"
    assert row["outcome"] is None


def test_pending_bets_lists_unsettled(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.record_bet("A", 4.0, 10.0)
    bt.record_bet("B", 5.0, 10.0)
    assert len(bt.pending_bets()) == 2


# ------------------------------------------------------------------
# Settle bets — win bet
# ------------------------------------------------------------------

def test_settle_win_bet_win(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("Dancer", 5.0, 10.0)
    result = bt.settle_bet(bid, "win")
    assert result["outcome"] == "win"
    assert result["gross_return"] == pytest.approx(50.0)
    assert result["profit"] == pytest.approx(40.0)
    assert bt.bankroll == pytest.approx(1000.0 - 10.0 + 50.0)


def test_settle_win_bet_lose(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("Dancer", 5.0, 10.0)
    result = bt.settle_bet(bid, "lose")
    assert result["outcome"] == "lose"
    assert result["gross_return"] == pytest.approx(0.0)
    assert result["profit"] == pytest.approx(-10.0)
    assert bt.bankroll == pytest.approx(990.0)


def test_settle_win_bet_place_is_loss(tmp_path):
    # A win-only bet that places pays nothing
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("X", 8.0, 10.0, bet_type="win")
    result = bt.settle_bet(bid, "place")
    assert result["gross_return"] == pytest.approx(0.0)
    assert result["profit"] == pytest.approx(-10.0)


# ------------------------------------------------------------------
# Settle bets — each-way
# ------------------------------------------------------------------

def test_settle_ew_win(tmp_path):
    # 10.0 EW stake (£5 win + £5 place), odds 10.0, ew_fraction 0.2
    # win leg: 5 * 10 = 50; place leg: 5 * ((10-1)*0.2+1) = 5 * 2.8 = 14; total = 64
    bt = _tracker(tmp_path, initial_bankroll=1000.0, ew_fraction=0.2)
    bid = bt.record_bet("Star", 10.0, 10.0, bet_type="each_way")
    result = bt.settle_bet(bid, "win")
    assert result["gross_return"] == pytest.approx(64.0)
    assert result["profit"] == pytest.approx(54.0)


def test_settle_ew_place(tmp_path):
    # Only place leg wins: 5 * 2.8 = 14
    bt = _tracker(tmp_path, initial_bankroll=1000.0, ew_fraction=0.2)
    bid = bt.record_bet("Star", 10.0, 10.0, bet_type="each_way")
    result = bt.settle_bet(bid, "place")
    assert result["gross_return"] == pytest.approx(14.0)
    assert result["profit"] == pytest.approx(4.0)


def test_settle_ew_lose(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0, ew_fraction=0.2)
    bid = bt.record_bet("Star", 10.0, 10.0, bet_type="each_way")
    result = bt.settle_bet(bid, "lose")
    assert result["gross_return"] == pytest.approx(0.0)
    assert result["profit"] == pytest.approx(-10.0)


# ------------------------------------------------------------------
# Settle error cases
# ------------------------------------------------------------------

def test_settle_invalid_outcome(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("X", 4.0, 10.0)
    with pytest.raises(ValueError, match="outcome"):
        bt.settle_bet(bid, "bogus")


def test_settle_already_settled(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("X", 4.0, 10.0)
    bt.settle_bet(bid, "lose")
    with pytest.raises(ValueError, match="already settled"):
        bt.settle_bet(bid, "win")


def test_settle_missing_bet(tmp_path):
    bt = _tracker(tmp_path)
    with pytest.raises(ValueError, match="not found"):
        bt.settle_bet(9999, "win")


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def _populated_tracker(tmp_path) -> BetTracker:
    bt = _tracker(tmp_path, initial_bankroll=1000.0, ew_fraction=0.2)
    # 3 settled bets
    b1 = bt.record_bet("A", 4.0, 10.0, venue="Ascot", strategy=Strategy.FLAT)
    bt.settle_bet(b1, "win")  # profit +30
    b2 = bt.record_bet("B", 6.0, 10.0, venue="Ascot", strategy=Strategy.FLAT)
    bt.settle_bet(b2, "lose")  # profit -10
    b3 = bt.record_bet("C", 8.0, 10.0, venue="Naas", bet_type="each_way", strategy=Strategy.FLAT)
    bt.settle_bet(b3, "place")  # 5*((8-1)*0.2+1)=5*2.4=12; profit +2
    # 1 pending
    bt.record_bet("D", 5.0, 10.0, venue="Naas", strategy=Strategy.FLAT)
    return bt


def test_summary_totals(tmp_path):
    bt = _populated_tracker(tmp_path)
    s = bt.summary()
    assert s["total_bets"] == 3
    assert s["pending_bets"] == 1
    assert s["total_staked"] == pytest.approx(30.0)
    assert s["total_profit"] == pytest.approx(22.0)  # 30 - 10 + 2
    assert s["win_rate"] == pytest.approx(1 / 3, abs=0.001)
    assert not s["stop_loss_active"]


def test_summary_empty(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    s = bt.summary()
    assert s["total_bets"] == 0
    assert s["total_profit"] == 0.0


def test_pl_series_cumulative(tmp_path):
    bt = _populated_tracker(tmp_path)
    df = bt.pl_series()
    assert len(df) == 3
    assert "cumulative_profit" in df.columns
    assert df["cumulative_profit"].iloc[-1] == pytest.approx(22.0)


def test_pl_series_empty(tmp_path):
    bt = _tracker(tmp_path)
    df = bt.pl_series()
    assert df.empty


def test_breakdown_by_venue(tmp_path):
    bt = _populated_tracker(tmp_path)
    df = bt.breakdown(by="venue")
    ascot = df[df["venue"] == "Ascot"].iloc[0]
    assert ascot["bets"] == 2
    assert ascot["wins"] == 1
    naas = df[df["venue"] == "Naas"].iloc[0]
    assert naas["bets"] == 1


def test_breakdown_unknown_column(tmp_path):
    bt = _populated_tracker(tmp_path)
    df = bt.breakdown(by="nonexistent")
    assert df.empty


def test_bankroll_history_tracks_changes(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=500.0)
    bid = bt.record_bet("A", 4.0, 50.0)
    bt.settle_bet(bid, "win")
    hist = bt.bankroll_history()
    assert len(hist) == 3  # init, bet_placed, bet_settled
    assert hist["balance"].iloc[-1] == pytest.approx(500.0 - 50.0 + 50.0 * 4.0)


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------

def test_export_csv_creates_file(tmp_path):
    bt = _populated_tracker(tmp_path)
    out = bt.export_csv(tmp_path / "export.csv")
    assert out.exists()
    lines = out.read_text().splitlines()
    assert len(lines) == 5  # header + 4 bets


def test_export_csv_empty(tmp_path):
    bt = _tracker(tmp_path)
    out = bt.export_csv(tmp_path / "empty.csv")
    assert out.exists()
    assert "no bets" in out.read_text()


def test_export_csv_default_path(tmp_path, monkeypatch):
    import utils.bet_tracker as bt_mod
    monkeypatch.setattr(bt_mod, "_DEFAULT_CSV", tmp_path / "default.csv")
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.record_bet("A", 5.0, 10.0)
    out = bt.export_csv()
    assert out == tmp_path / "default.csv"
    assert out.exists()


# ------------------------------------------------------------------
# all_bets filter
# ------------------------------------------------------------------

def test_all_bets_settled_only(tmp_path):
    bt = _populated_tracker(tmp_path)
    settled = bt.all_bets(settled_only=True)
    assert len(settled) == 3
    assert all(b["outcome"] is not None for b in settled)


def test_all_bets_includes_pending(tmp_path):
    bt = _populated_tracker(tmp_path)
    all_b = bt.all_bets()
    assert len(all_b) == 4


# ------------------------------------------------------------------
# Paper-betting: value-edge capture + CLV
# ------------------------------------------------------------------

def test_record_bet_stores_value_edge(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("A", 5.0, 10.0, won_prob=0.30, value_edge=0.10)
    row = bt.get_bet(bid)
    assert row["won_prob"] == pytest.approx(0.30)
    assert row["value_edge"] == pytest.approx(0.10)


def test_settle_captures_positive_clv(tmp_path):
    # Took 6.0, closed at 4.0 → beat the line. CLV% = (6/4 - 1)*100 = 50.
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("A", 6.0, 10.0)
    row = bt.settle_bet(bid, "win", closing_odds=4.0)
    assert row["closing_odds"] == pytest.approx(4.0)
    assert row["clv_pct"] == pytest.approx(50.0)


def test_settle_captures_negative_clv(tmp_path):
    # Took 4.0, closed at 5.0 → worse than the line. CLV% = (4/5 - 1)*100 = -20.
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("A", 4.0, 10.0)
    row = bt.settle_bet(bid, "lose", closing_odds=5.0)
    assert row["clv_pct"] == pytest.approx(-20.0)


def test_settle_without_closing_odds_has_null_clv(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("A", 4.0, 10.0)
    row = bt.settle_bet(bid, "win")
    assert row["clv_pct"] is None


def test_summary_avg_clv(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    b1 = bt.record_bet("A", 6.0, 10.0)
    bt.settle_bet(b1, "win", closing_odds=4.0)   # +50
    b2 = bt.record_bet("B", 4.0, 10.0)
    bt.settle_bet(b2, "lose", closing_odds=5.0)  # -20
    assert bt.summary()["avg_clv_pct"] == pytest.approx(15.0)


# ------------------------------------------------------------------
# Paper-betting: void settlement
# ------------------------------------------------------------------

def test_settle_void_refunds_stake(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.record_bet("A", 5.0, 40.0)
    assert bt.bankroll == pytest.approx(960.0)
    row = bt.settle_bet(bid, "void")
    assert row["outcome"] == "void"
    assert row["gross_return"] == pytest.approx(40.0)
    assert row["profit"] == pytest.approx(0.0)
    assert bt.bankroll == pytest.approx(1000.0)


# ------------------------------------------------------------------
# Paper-betting: set bankroll
# ------------------------------------------------------------------

def test_set_bankroll_before_bets_resets_baseline(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.set_bankroll(250.0)
    assert bt.bankroll == pytest.approx(250.0)
    assert bt.initial_bankroll == pytest.approx(250.0)  # baseline moved
    assert len(bt.bankroll_history()) == 1  # still just the init row


def test_set_bankroll_after_bets_adjusts(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.record_bet("A", 5.0, 10.0)  # bankroll 990
    bt.set_bankroll(2000.0)
    assert bt.bankroll == pytest.approx(2000.0)
    assert bt.initial_bankroll == pytest.approx(1000.0)  # baseline preserved
    hist = bt.bankroll_history()
    assert hist["event_type"].iloc[-1] == "adjust"


def test_set_bankroll_rejects_nonpositive(tmp_path):
    bt = _tracker(tmp_path)
    with pytest.raises(ValueError):
        bt.set_bankroll(0.0)


# ------------------------------------------------------------------
# Paper-betting: guards (started race + double-betting)
# ------------------------------------------------------------------

def test_place_paper_bet_success(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bid = bt.place_paper_bet(
        "Dancer", 5.0, 10.0, race_id="ascot-1430", horse_id="h1",
        race_time="2999-01-01T14:30:00+00:00", won_prob=0.3, value_edge=0.1,
    )
    assert bt.get_bet(bid)["horse_name"] == "Dancer"
    assert bt.bankroll == pytest.approx(990.0)


def test_place_paper_bet_blocks_started_race(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    fixed_now = datetime.fromisoformat("2026-06-16T15:00:00+00:00")
    with pytest.raises(RaceStartedError):
        bt.place_paper_bet(
            "Dancer", 5.0, 10.0, race_id="r1", horse_id="h1",
            race_time="2026-06-16T14:30:00+00:00", _now=fixed_now,
        )


def test_place_paper_bet_allows_future_race(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    fixed_now = datetime.fromisoformat("2026-06-16T14:00:00+00:00")
    bid = bt.place_paper_bet(
        "Dancer", 5.0, 10.0, race_id="r1", horse_id="h1",
        race_time="2026-06-16T14:30:00+00:00", _now=fixed_now,
    )
    assert bid > 0


def test_place_paper_bet_blocks_duplicate(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.place_paper_bet("Dancer", 5.0, 10.0, race_id="r1", horse_id="h1",
                       race_time="2999-01-01T14:30:00+00:00")
    with pytest.raises(DuplicateBetError):
        bt.place_paper_bet("Dancer", 5.0, 10.0, race_id="r1", horse_id="h1",
                           race_time="2999-01-01T14:30:00+00:00")


def test_place_paper_bet_allows_different_market_same_horse(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.place_paper_bet("Dancer", 5.0, 10.0, bet_type="win", race_id="r1",
                       horse_id="h1", race_time="2999-01-01T14:30:00+00:00")
    # each-way is a distinct market — allowed
    bid = bt.place_paper_bet("Dancer", 5.0, 10.0, bet_type="each_way", race_id="r1",
                             horse_id="h1", race_time="2999-01-01T14:30:00+00:00")
    assert bid > 0


def test_has_open_bet_ignores_missing_ids(tmp_path):
    bt = _tracker(tmp_path, initial_bankroll=1000.0)
    bt.record_bet("A", 5.0, 10.0)  # no race_id/horse_id
    assert not bt.has_open_bet(None, None)


def test_race_has_started_unknown_time_is_false():
    assert BetTracker.race_has_started(None) is False
    assert BetTracker.race_has_started("not-a-date") is False
