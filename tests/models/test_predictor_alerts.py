"""Alert gating in models.predictor._fire_alerts.

The reported symptom: every rebuild sent a burst of notifications about racing
that had already been run. predictions.json holds the whole day and only the
race-soon alert was time-gated, so top-pick and odds-drop alerts re-fired for
finished races on every run.
"""

from datetime import timedelta

import pytest

import models.predictor as predictor
from utils import alert_dedupe
from utils.timezone import now


class FakeNotifier:
    """Records calls instead of sending. Mirrors the attributes _fire_alerts reads."""

    race_soon_minutes = 10
    odds_drop_threshold_pct = 10.0

    def __init__(self):
        self.soon, self.top, self.drop = [], [], []

    def notify_race_soon(self, venue, race_time, horse, score, odds, mins):
        self.soon.append((venue, horse, mins))

    def notify_new_top_pick(self, venue, race_time, horse, score, odds):
        self.top.append((venue, horse))

    def notify_odds_drop(self, horse, venue, old, new, pct):
        self.drop.append((horse, old, new, round(pct, 1)))


@pytest.fixture
def notifier(monkeypatch):
    fake = FakeNotifier()
    monkeypatch.setattr(predictor, "get_notifier", lambda: fake)
    alert_dedupe.reset()
    yield fake
    alert_dedupe.reset()


def _race(venue="Dundalk", minutes_from_now=30, runners=None, top="Alpha"):
    rt = (now() + timedelta(minutes=minutes_from_now)).isoformat()
    sels = runners if runners is not None else [
        {"horse_name": top, "decimal_odds": 4.0, "composite_score": 0.8},
    ]
    return {"venue": venue, "race_time": rt, "selections": sels,
            "excluded_low_odds": []}


def _key(race):
    return f"{race['venue']}|{race['race_time']}"


# ---------------------------------------------------------------------------
# The bug: finished races must never alert
# ---------------------------------------------------------------------------
def test_started_race_sends_nothing(notifier):
    """A race 2h in the past must be silent on all three alert types."""
    race = _race(minutes_from_now=-120, top="Beta")
    predictor._fire_alerts(
        [race],
        {_key(race): "Alpha"},                       # top pick changed
        {(_key(race), "Beta"): 8.0},                 # and the price halved
    )
    assert notifier.soon == [] and notifier.top == [] and notifier.drop == []


def test_race_just_gone_off_is_silent(notifier):
    race = _race(minutes_from_now=-1, top="Beta")
    predictor._fire_alerts([race], {_key(race): "Alpha"},
                           {(_key(race), "Beta"): 8.0})
    assert not (notifier.soon or notifier.top or notifier.drop)


def test_upcoming_race_still_alerts(notifier):
    """The gate must not silence the alerts that matter."""
    race = _race(minutes_from_now=30, top="Beta")
    predictor._fire_alerts([race], {_key(race): "Alpha"},
                           {(_key(race), "Beta"): 8.0})
    assert notifier.top == [("Dundalk", "Beta")]
    assert notifier.drop == [("Beta", 8.0, 4.0, 50.0)]


def test_unparseable_race_time_is_treated_as_not_upcoming(notifier):
    """Silence beats guessing — an unreadable time is not evidence of a future race."""
    race = _race()
    race["race_time"] = "not-a-timestamp"
    predictor._fire_alerts([race], {_key(race): "Alpha"},
                           {(_key(race), "Alpha"): 8.0})
    assert not (notifier.soon or notifier.top or notifier.drop)


def test_races_beyond_the_window_are_skipped(notifier, monkeypatch):
    """Tomorrow's card should not buzz the phone all evening."""
    race = _race(minutes_from_now=600, top="Beta")   # 10h out, window is 180m
    predictor._fire_alerts([race], {_key(race): "Alpha"},
                           {(_key(race), "Beta"): 8.0})
    assert not (notifier.top or notifier.drop)


# ---------------------------------------------------------------------------
# De-duplication across rebuilds
# ---------------------------------------------------------------------------
def test_same_odds_drop_is_sent_once_across_rebuilds(notifier):
    race = _race(minutes_from_now=30, top="Beta")
    old = {(_key(race), "Beta"): 8.0}
    for _ in range(4):                     # four rebuilds, same prices
        predictor._fire_alerts([race], {}, old)
    assert len(notifier.drop) == 1


def test_a_further_drop_does_alert_again(notifier):
    """Genuine new price movement is the whole point — it must still get through."""
    race1 = _race(minutes_from_now=30, top="Beta")
    predictor._fire_alerts([race1], {}, {(_key(race1), "Beta"): 8.0})

    race2 = dict(race1, selections=[
        {"horse_name": "Beta", "decimal_odds": 3.0, "composite_score": 0.8}])
    predictor._fire_alerts([race2], {}, {(_key(race2), "Beta"): 8.0})

    assert len(notifier.drop) == 2
    assert notifier.drop[1][2] == 3.0


def test_repeated_top_pick_change_is_sent_once(notifier):
    race = _race(minutes_from_now=30, top="Beta")
    for _ in range(3):
        predictor._fire_alerts([race], {_key(race): "Alpha"}, {})
    assert len(notifier.top) == 1


# ---------------------------------------------------------------------------
# Odds map keyed by race, not by bare horse name
# ---------------------------------------------------------------------------
def test_same_horse_name_at_another_meeting_is_not_a_price_move(notifier):
    """A bare horse-name key made an unrelated runner look like a big drop."""
    race = _race(venue="Dundalk", minutes_from_now=30, top="Alpha")
    stale_other_meeting = {("Ayr|2026-01-01T13:00:00+00:00", "Alpha"): 20.0}
    predictor._fire_alerts([race], {}, stale_other_meeting)
    assert notifier.drop == []


def test_drop_below_threshold_is_ignored(notifier):
    race = _race(minutes_from_now=30, top="Beta")
    predictor._fire_alerts([race], {}, {(_key(race), "Beta"): 4.2})  # ~5%
    assert notifier.drop == []


def test_lengthening_price_is_not_a_drop(notifier):
    race = _race(minutes_from_now=30, top="Beta")
    predictor._fire_alerts([race], {}, {(_key(race), "Beta"): 2.0})
    assert notifier.drop == []


def test_excluded_low_odds_runners_are_also_checked(notifier):
    race = _race(minutes_from_now=30, top="Alpha")
    race["excluded_low_odds"] = [{"horse_name": "Odds On", "decimal_odds": 1.2}]
    predictor._fire_alerts([race], {}, {(_key(race), "Odds On"): 1.8})
    assert [d[0] for d in notifier.drop] == ["Odds On"]


def test_dispatch_never_raises_on_malformed_input(notifier):
    """Alerting is best-effort; it must not be able to fail a prediction run."""
    predictor._fire_alerts([{"venue": "X"}], {}, {})   # no race_time at all
    assert not (notifier.soon or notifier.top or notifier.drop)
