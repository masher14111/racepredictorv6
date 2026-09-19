"""Recommended paper stake — fractional Kelly, clamped to the configured band.

The behaviours that matter are the refusals: no edge must mean no bet, an
uncalibrated or impossible input must mean no bet, and the ceiling must only
ever clamp a stake Kelly independently asked for.
"""

import pytest

from utils.stake_advice import recommend_stake

BANK = dict(bankroll=1000.0, kelly_fraction=0.25, min_stake=0.50, max_stake=800.0)


# ---------------------------------------------------------------------------
# The core refusal: a price shorter than the model justifies is not a bet
# ---------------------------------------------------------------------------
def test_no_edge_is_no_bet():
    # 10% chance at 5.0 (needs >20% to break even)
    a = recommend_stake(0.10, 5.0, **BANK)
    assert a.stake == 0.0
    assert not a.is_bet
    assert a.band == "No bet"
    assert a.edge < 0
    assert "shorter than" in a.reason


def test_exactly_fair_price_is_no_bet():
    # 25% at 4.0 is a zero-edge price — Kelly stakes nothing.
    a = recommend_stake(0.25, 4.0, **BANK)
    assert a.stake == 0.0
    assert a.edge == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def test_quarter_kelly_sizing_is_exact():
    # p=0.5, dec=3.0 -> b=2, edge=0.5, full Kelly=0.25 of bank -> quarter = 62.50
    a = recommend_stake(0.5, 3.0, **BANK)
    assert a.stake == pytest.approx(62.50)
    assert a.kelly_fraction == pytest.approx(0.25)
    assert a.edge == pytest.approx(0.5)
    assert not a.capped


def test_smaller_edge_gives_smaller_stake():
    # p=0.2, dec=6.0 -> b=5, edge=0.2, full=0.04 -> 0.04*0.25*1000 = 10.00
    a = recommend_stake(0.2, 6.0, **BANK)
    assert a.stake == pytest.approx(10.00)
    assert a.band == "Small"


def test_bigger_edge_gives_bigger_stake():
    small = recommend_stake(0.2, 6.0, **BANK).stake
    big = recommend_stake(0.4, 6.0, **BANK).stake
    assert big > small


# ---------------------------------------------------------------------------
# The band clamps
# ---------------------------------------------------------------------------
def test_ceiling_clamps_and_is_reported():
    a = recommend_stake(0.5, 3.0, **{**BANK, "bankroll": 100_000.0})
    assert a.stake == 800.0
    assert a.capped is True
    assert "capped" in a.reason


def test_floor_lifts_a_real_but_tiny_edge():
    a = recommend_stake(0.5, 3.0, **{**BANK, "bankroll": 1.0})
    assert a.stake == 0.50
    assert a.is_bet
    assert "minimum" in a.reason


def test_floor_never_rescues_a_losing_bet():
    """The floor applies to real edges only — it must not invent a bet."""
    a = recommend_stake(0.05, 3.0, **{**BANK, "bankroll": 1.0})
    assert a.stake == 0.0


def test_default_settings_cannot_reach_the_ceiling():
    """Full Kelly cannot exceed the bankroll, so quarter-Kelly on 1000 caps at 250.

    Documents that the 800 ceiling is a guard rail, not a reachable target at
    the shipped defaults.
    """
    best = max(
        recommend_stake(p / 100, dec, **BANK).stake
        for p in range(1, 100)
        for dec in (1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0)
    )
    assert best <= 250.0


# ---------------------------------------------------------------------------
# The value-layer veto — the card must never show a stake beside a negative EV
# ---------------------------------------------------------------------------
def test_negative_market_ev_vetoes_a_kelly_positive_bet():
    """Kelly likes this price; the value layer does not. The value layer wins."""
    unvetoed = recommend_stake(0.5, 3.0, **BANK)
    assert unvetoed.stake > 0

    a = recommend_stake(0.5, 3.0, market_ev=-0.11, **BANK)
    assert a.stake == 0.0
    assert a.band == "No bet"
    assert "Value layer" in a.reason


def test_zero_market_ev_is_also_a_veto():
    assert recommend_stake(0.5, 3.0, market_ev=0.0, **BANK).stake == 0.0


def test_positive_market_ev_allows_the_bet():
    a = recommend_stake(0.5, 3.0, market_ev=0.12, **BANK)
    assert a.stake == pytest.approx(62.50)


@pytest.mark.parametrize("ev", [None, float("nan"), "n/a"])
def test_absent_or_unusable_market_ev_does_not_veto(ev):
    assert recommend_stake(0.5, 3.0, market_ev=ev, **BANK).stake > 0


# ---------------------------------------------------------------------------
# Unusable inputs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prob,odds",
    [
        (None, 3.0), (0.5, None), (None, None),
        (0.0, 3.0), (1.0, 3.0), (-0.2, 3.0), (1.5, 3.0),
        (0.5, 1.0), (0.5, 0.0), (0.5, -2.0),
        (float("nan"), 3.0), (0.5, float("nan")),
        ("x", 3.0), (0.5, "x"),
    ],
)
def test_unusable_inputs_give_no_bet(prob, odds):
    a = recommend_stake(prob, odds, **BANK)
    assert a.stake == 0.0
    assert not a.is_bet


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_bankroll_or_fraction_gives_no_bet(bad):
    assert recommend_stake(0.5, 3.0, **{**BANK, "bankroll": bad}).stake == 0.0
    assert recommend_stake(0.5, 3.0, **{**BANK, "kelly_fraction": bad}).stake == 0.0


def test_a_live_stake_is_never_labelled_no_bet():
    """Regression: a tiny-but-real edge printed "1.00" beside the label "No bet"."""
    a = recommend_stake(0.087, 12.0, **BANK)
    assert a.stake > 0
    assert a.band == "Token"

    for p in range(1, 100):
        for dec in (1.2, 1.5, 2.0, 3.0, 6.0, 12.0, 34.0):
            adv = recommend_stake(p / 100, dec, **BANK)
            assert (adv.band == "No bet") == (adv.stake == 0.0), (p, dec, adv)


def test_stake_is_rounded_to_cents():
    a = recommend_stake(0.37, 3.3, **BANK)
    assert a.stake == round(a.stake, 2)
