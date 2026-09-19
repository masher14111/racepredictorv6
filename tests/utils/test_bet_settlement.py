"""Tests for utils/bet_settlement.py — auto-settling paper bets from results."""
import pytest

from utils.bet_tracker import BetTracker
from utils.bet_settlement import settle_from_results


def _tracker(tmp_path, **kwargs) -> BetTracker:
    return BetTracker(db_path=str(tmp_path / "test.db"), initial_bankroll=1000.0, **kwargs)


def _place(bt, **kw):
    """Place a future-dated paper bet (so the started-race guard never fires).

    Far-future date so this is robust to the real wall clock; the result fixture
    shares the same day so name/void matching lines up.
    """
    kw.setdefault("race_time", "2999-01-01T14:30:00+00:00")
    kw.setdefault("venue", "Ascot")
    return bt.place_paper_bet(**kw)


def _result(horse_name="Dancer", horse_id="h1", venue="Ascot",
            race_date="2999-01-01", position=1, odds_finish=4.0):
    return dict(horse_name=horse_name, horse_id=horse_id, venue=venue,
                race_date=race_date, position=position, odds_finish=odds_finish)


# ------------------------------------------------------------------
# Win / lose / place by finishing position
# ------------------------------------------------------------------

def test_win_bet_winner_settled_win(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=10.0,
           race_id="r1", horse_id="h1", won_prob=0.3)
    settled = settle_from_results(bt, [_result(position=1, odds_finish=4.0)])
    assert len(settled) == 1
    assert settled[0]["outcome"] == "win"
    assert settled[0]["profit"] == pytest.approx(40.0)
    assert settled[0]["clv_pct"] == pytest.approx(25.0)  # (5/4-1)*100


def test_win_bet_loser_settled_lose(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=10.0,
           race_id="r1", horse_id="h1")
    settled = settle_from_results(bt, [_result(position=4)])
    assert settled[0]["outcome"] == "lose"
    assert settled[0]["profit"] == pytest.approx(-10.0)


def test_each_way_places_within_paid_places(tmp_path):
    bt = _tracker(tmp_path, ew_fraction=0.2, ew_places=3)
    _place(bt, horse_name="Dancer", odds_decimal=10.0, stake=10.0,
           bet_type="each_way", race_id="r1", horse_id="h1")
    settled = settle_from_results(bt, [_result(position=2, odds_finish=8.0)])
    assert settled[0]["outcome"] == "place"
    # place leg: 5 * ((10-1)*0.2+1)=5*2.8=14 → profit +4
    assert settled[0]["profit"] == pytest.approx(4.0)


def test_each_way_outside_places_loses(tmp_path):
    bt = _tracker(tmp_path, ew_places=3)
    _place(bt, horse_name="Dancer", odds_decimal=10.0, stake=10.0,
           bet_type="each_way", race_id="r1", horse_id="h1")
    settled = settle_from_results(bt, [_result(position=5)])
    assert settled[0]["outcome"] == "lose"


# ------------------------------------------------------------------
# Void / non-runner handling
# ------------------------------------------------------------------

def test_void_when_horse_absent_but_race_ran(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=20.0,
           race_id="r1", horse_id="h1")
    # the race ran (a different horse has a result) but our horse is absent
    other = _result(horse_name="Rival", horse_id="h2", position=1)
    settled = settle_from_results(bt, [other])
    assert settled[0]["outcome"] == "void"
    assert bt.bankroll == pytest.approx(1000.0)  # stake refunded


def test_void_when_horse_ran_without_position(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=20.0,
           race_id="r1", horse_id="h1")
    settled = settle_from_results(bt, [_result(position=None)])
    assert settled[0]["outcome"] == "void"


def test_pending_when_race_not_in_results(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=20.0, venue="Naas",
           race_id="r1", horse_id="h1")
    # results only cover a different race (other venue, other horse) → stay pending
    settled = settle_from_results(
        bt, [_result(venue="Ascot", horse_id="h2", horse_name="Rival")])
    assert settled == []
    assert len(bt.pending_bets()) == 1


# ------------------------------------------------------------------
# Matching fallback (by normalized name when ids differ)
# ------------------------------------------------------------------

def test_match_by_name_when_no_horse_id(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Sea The Stars (IRE)", odds_decimal=3.0, stake=10.0,
           race_id="r1", horse_id=None)
    # result has a different id space but the normalized name matches
    res = _result(horse_name="Sea the Stars", horse_id="zzz", position=1,
                  odds_finish=2.5)
    settled = settle_from_results(bt, [res])
    assert settled[0]["outcome"] == "win"


def test_idempotent_second_run_settles_nothing(tmp_path):
    bt = _tracker(tmp_path)
    _place(bt, horse_name="Dancer", odds_decimal=5.0, stake=10.0,
           race_id="r1", horse_id="h1")
    results = [_result(position=1)]
    assert len(settle_from_results(bt, results)) == 1
    assert settle_from_results(bt, results) == []  # already settled → no-op


# ── venue aliases: the meeting name vs the course name ────────────────────────

def test_royal_ascot_result_settles_an_ascot_bet():
    """The results feed files the June meeting as "Royal Ascot"; every placed bet
    says "Ascot". Without the alias the bet never matches its own result and sits
    pending forever — which is what happened to 11 bets from 19 Jun 2026."""
    from utils.bet_settlement import _venue_key
    assert _venue_key("Royal Ascot") == _venue_key("Ascot") == "ascot"


def test_down_royal_is_not_collapsed_by_the_alias():
    """A "strip the Royal prefix" rule would eat Down Royal, a different course
    entirely. The alias is an explicit map for exactly this reason."""
    from utils.bet_settlement import _venue_key
    assert _venue_key("Down Royal") == "downroyal"
    assert _venue_key("Down Royal") != _venue_key("Ascot")
    assert _venue_key("Downpatrick") == "downpatrick"


def test_alias_lets_a_royal_ascot_result_settle_end_to_end():
    from utils.bet_settlement import settle_from_results

    class _Tracker:
        ew_places = 3
        def __init__(self):
            self.settled = []
        def pending_bets(self):
            return [{"id": 1, "venue": "Ascot", "horse_name": "Causeway",
                     "race_time": "2026-06-19T15:05:00+01:00", "bet_type": "win",
                     "horse_id": "b4dd7477c23249de"}]
        def settle_bet(self, bet_id, outcome, closing_odds=None):
            row = {"id": bet_id, "outcome": outcome, "closing_odds": closing_odds}
            self.settled.append(row)
            return row

    t = _Tracker()
    out = settle_from_results(t, [{
        "venue": "Royal Ascot", "horse_name": "Causeway",
        "race_date": "2026-06-19T15:05:00+01:00", "position": 1,
        "odds_finish": 4.2, "horse_id": 452611,
    }])
    assert len(out) == 1 and out[0]["outcome"] == "win"
    assert out[0]["closing_odds"] == 4.2
