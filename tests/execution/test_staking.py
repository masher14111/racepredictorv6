"""Requirement 6: the ceilings, not Kelly, are the control.

Every test here asserts a *refusal* or a *shrinkage*. Nothing in this module may
ever produce a larger stake than the smallest applicable ceiling, and a loosened
config must not change that (ExecutionConfig clamps it back).
"""
from __future__ import annotations

import pytest

from execution.config import ExecutionConfig
from execution.staking import (
    AccumulatorBlocked,
    CorrelatedBetBlocked,
    ExposureState,
    StakePlanner,
    reject_accumulator,
    reject_correlated,
    selection_lock_odds_band,
)

BANKROLL = 1000.0
RACE = "2026-07-27 14:30"
DAY = "2026-07-27"


@pytest.fixture
def cfg():
    """Defaults only — an absent execution block yields the strictest config."""
    return ExecutionConfig.from_config({})


@pytest.fixture
def planner(cfg):
    return StakePlanner(cfg, BANKROLL)


def _plan(planner, prob, odds, *, race=RACE, day=DAY, exposure=None):
    return planner.plan(
        prob=prob, decimal_odds=odds, race_uid=race, race_date=day, exposure=exposure
    )


# ── Kelly ────────────────────────────────────────────────────────────────────
def test_tenth_kelly_is_applied(planner):
    # p=0.36 @ 3.0 -> EV 0.08, full Kelly 0.04 of bankroll = 40.00.
    decision = _plan(planner, 0.36, 3.0)
    assert decision.full_kelly_stake == pytest.approx(40.0)
    assert decision.kelly_stake == pytest.approx(4.0)  # a tenth of full Kelly
    assert decision.stake == pytest.approx(4.0)
    assert decision.binding_constraint == "kelly"
    assert decision.is_bet


def test_per_bet_cap_binds_before_kelly_on_a_large_edge(planner):
    # p=0.90 @ 3.0 is an absurd edge; tenth-Kelly still asks for 85.00.
    decision = _plan(planner, 0.90, 3.0)
    assert decision.kelly_stake == pytest.approx(85.0)
    assert decision.stake == pytest.approx(5.0)  # 0.5% of a 1000 bankroll
    assert decision.max_stake == pytest.approx(5.0)
    assert decision.binding_constraint == "per_bet_cap"


def test_book_liability_clamps_the_per_bet_cap(cfg):
    # 0.5% of a 100k bankroll is 500, but the book will only lay 50.
    planner = StakePlanner(cfg, 100_000.0)
    decision = _plan(planner, 0.90, 3.0)
    assert planner.per_bet_cap() == pytest.approx(cfg.frictions.max_stake_per_bet)
    assert decision.stake == pytest.approx(50.0)
    assert decision.binding_constraint == "per_bet_cap"


# ── exposure ceilings ────────────────────────────────────────────────────────
def test_race_cap_stops_the_second_bet_in_the_same_race(planner):
    exposure = ExposureState(race_staked={RACE: 3.0}, day_staked={DAY: 3.0})
    partial = _plan(planner, 0.90, 3.0, exposure=exposure)
    assert partial.stake == pytest.approx(2.0)  # 5.00 race ceiling less 3.00 staked
    assert partial.binding_constraint == "race_cap"

    exhausted = ExposureState(race_staked={RACE: 5.0}, day_staked={DAY: 5.0})
    blocked = _plan(planner, 0.90, 3.0, exposure=exhausted)
    assert blocked.stake == 0.0
    assert blocked.binding_constraint == "race_cap"


def test_exhausted_capacity_is_exactly_zero_never_negative(planner):
    over = ExposureState(race_staked={RACE: 50.0}, day_staked={DAY: 50.0})
    assert planner.remaining_race_capacity(RACE, over) == 0.0
    assert planner.remaining_daily_capacity(DAY, over) == 0.0
    decision = _plan(planner, 0.90, 3.0, exposure=over)
    assert decision.stake == 0.0
    assert decision.max_stake == 0.0
    assert not decision.is_bet


def test_daily_cap_stops_the_day(planner):
    # A fresh race, so only the day ceiling can bind: 30.00 total, 29.50 used.
    nearly = ExposureState(day_staked={DAY: 29.5})
    decision = _plan(planner, 0.90, 3.0, race="2026-07-27 15:05", exposure=nearly)
    assert decision.stake == pytest.approx(0.5)
    assert decision.binding_constraint == "daily_cap"

    spent = ExposureState(day_staked={DAY: 30.0})
    blocked = _plan(planner, 0.90, 3.0, race="2026-07-27 15:05", exposure=spent)
    assert blocked.stake == 0.0
    assert blocked.binding_constraint == "daily_cap"


def test_exposure_from_a_different_day_does_not_bind(planner):
    other_day = ExposureState(day_staked={"2026-07-26": 30.0})
    decision = _plan(planner, 0.90, 3.0, exposure=other_day)
    assert decision.stake == pytest.approx(5.0)
    assert decision.binding_constraint == "per_bet_cap"


# ── rounding / minimum ───────────────────────────────────────────────────────
def test_rounding_is_down_never_up(planner):
    # p=0.3624 @ 3.0 -> tenth-Kelly 4.36, which must round DOWN to 4.30.
    decision = _plan(planner, 0.3624, 3.0)
    assert decision.kelly_stake == pytest.approx(4.36)
    assert decision.stake == pytest.approx(4.30)
    assert decision.stake <= decision.max_stake


def test_sub_minimum_stake_is_dropped_not_rounded_up(cfg):
    # 100 bankroll, p=0.34 @ 3.0 -> tenth-Kelly 0.10, under the 0.50 minimum.
    planner = StakePlanner(cfg, 100.0)
    decision = _plan(planner, 0.34, 3.0)
    assert decision.max_stake == pytest.approx(0.10)
    assert decision.stake == 0.0
    assert decision.binding_constraint == "min_stake"


# ── no edge ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "prob, odds",
    [
        (0.30, 3.0),  # negative EV
        (1.0 / 3.0, 3.0),  # exactly fair -> still not a bet
        (0.90, 1.0),  # invalid price -> NaN Kelly
        (0.90, None),  # missing price
        (float("nan"), 3.0),  # unusable probability
    ],
)
def test_non_positive_or_unreadable_edge_never_stakes(planner, prob, odds):
    decision = _plan(planner, prob, odds)
    assert decision.stake == 0.0
    assert decision.max_stake == 0.0
    assert decision.binding_constraint == "no_edge"


def test_no_edge_beats_available_capacity(planner):
    """Room under every ceiling does not license a bet without an edge."""
    assert planner.remaining_race_capacity(RACE) == pytest.approx(5.0)
    assert _plan(planner, 0.20, 3.0).binding_constraint == "no_edge"


# ── reporting ────────────────────────────────────────────────────────────────
def test_every_ceiling_is_reported_in_reasons(planner):
    decision = _plan(planner, 0.90, 3.0)
    joined = " | ".join(decision.reasons)
    for label in ("kelly:", "per_bet_cap:", "race_cap:", "daily_cap:", "binding:"):
        assert label in joined
    payload = decision.to_dict()
    assert payload["binding_constraint"] == "per_bet_cap"
    assert payload["stake"] == pytest.approx(5.0)
    assert isinstance(payload["reasons"], list) and payload["reasons"]


# ── exposure bookkeeping ─────────────────────────────────────────────────────
def test_exposure_state_from_tickets():
    exposure = ExposureState.from_tickets(
        [
            {"race_uid": RACE, "race_date": DAY, "stake": 2.5, "status": "open"},
            {"race_uid": RACE, "race_date": DAY, "stake": 1.0, "status": "settled"},
            {"race_uid": "other", "race_date": DAY, "stake": 4.0},
        ]
    )
    # Settled turnover still counts against exposure; only open tickets count as open.
    assert exposure.staked_in_race(RACE) == pytest.approx(3.5)
    assert exposure.staked_on_day(DAY) == pytest.approx(7.5)
    assert exposure.open_tickets == 2
    assert exposure.staked_in_race("unknown") == 0.0


def test_unreadable_ticket_stake_raises_rather_than_freeing_capacity():
    with pytest.raises(ValueError):
        ExposureState.from_tickets([{"race_uid": RACE, "stake": "not-a-number"}])
    with pytest.raises(ValueError):
        ExposureState.from_tickets([{"race_uid": RACE, "stake": -1.0}])


def test_exposure_prefers_race_key_over_race_uid(planner):
    """Stage 21: two different venues sharing an exact off-time collapse onto
    one ``race_uid`` (D37/D53) -- exposure must key on the venue-qualified
    ``race_key`` instead so they get independent race-exposure ceilings."""
    from execution.staking import ticket_race_uid

    same_time_different_venue_a = {"race_uid": "shared-off-time", "race_key": "leopardstown|2026-07-27T14:30"}
    same_time_different_venue_b = {"race_uid": "shared-off-time", "race_key": "york|2026-07-27T14:30"}
    assert ticket_race_uid(same_time_different_venue_a) != ticket_race_uid(same_time_different_venue_b)

    exposure = ExposureState.from_tickets(
        [{"race_key": "leopardstown|2026-07-27T14:30", "race_uid": "shared-off-time", "stake": 5.0}]
    )
    # A ticket for the OTHER venue sharing that off-time must see full capacity.
    decision = planner.plan(
        prob=0.90, decimal_odds=3.0, race_uid="york|2026-07-27T14:30",
        race_date=DAY, exposure=exposure,
    )
    assert decision.stake == pytest.approx(5.0)


def test_with_stake_matches_from_tickets_day_key_convention(planner):
    """``ExposureState.with_stake`` must accumulate onto the SAME bucket a
    fresh ``from_tickets`` read of an already-issued ticket would produce --
    otherwise a resumed run's freshly-planned exposure and its seeded
    exposure silently drift apart."""
    seeded = ExposureState.from_tickets(
        [{"race_key": RACE, "race_uid": "irrelevant", "race_date": DAY, "stake": 5.0}]
    )
    incremental = ExposureState().with_stake(race_uid=RACE, race_date=DAY, stake=5.0)
    assert seeded.staked_in_race(RACE) == incremental.staked_in_race(RACE) == pytest.approx(5.0)
    assert seeded.staked_on_day(DAY) == incremental.staked_on_day(DAY) == pytest.approx(5.0)


def test_with_stake_is_a_noop_for_a_non_positive_stake(planner):
    base = ExposureState(race_staked={RACE: 1.0})
    assert base.with_stake(race_uid=RACE, race_date=DAY, stake=0.0) is base
    assert base.with_stake(race_uid=RACE, race_date=DAY, stake=-5.0) is base


# ── selection-lock odds band (Stage 21 / B10) ────────────────────────────────
def test_a_price_outside_the_selection_lock_band_is_never_staked(planner):
    """A real edge at a real price is still refused when that price falls
    outside the ONE strategy actually validated on the held-out window --
    the band is a hard eligibility gate, not merely another ceiling."""
    decision = _plan(planner, 0.90, 8.0, exposure=None)  # unrestricted: a real stake
    assert decision.is_bet

    banded = planner.plan(
        prob=0.90, decimal_odds=8.0, race_uid=RACE, race_date=DAY, odds_band=(1.0, 4.0),
    )
    assert banded.stake == 0.0
    assert banded.binding_constraint == "outside_selection_lock_band"
    assert not banded.is_bet


def test_a_price_inside_the_selection_lock_band_stakes_normally(planner):
    inside = planner.plan(
        prob=0.36, decimal_odds=3.0, race_uid=RACE, race_date=DAY, odds_band=(1.0, 4.0),
    )
    unrestricted = _plan(planner, 0.36, 3.0)
    assert inside.stake == pytest.approx(unrestricted.stake)
    assert inside.binding_constraint == unrestricted.binding_constraint


def test_an_unreadable_price_is_refused_by_the_band_check(planner):
    banded = planner.plan(
        prob=0.90, decimal_odds=None, race_uid=RACE, race_date=DAY, odds_band=(1.0, 4.0),
    )
    assert banded.stake == 0.0
    assert banded.binding_constraint == "outside_selection_lock_band"


def test_selection_lock_odds_band_reads_the_real_frozen_lock(cfg):
    """Live-config smoke check: the real lock on disk today freezes
    ``short_lt_4`` (odds 1.0-4.0) -- this must not silently read as
    unrestricted (``None``) while a real lock file exists."""
    band = selection_lock_odds_band(cfg)
    assert band == (1.0, 4.0)


def test_selection_lock_odds_band_is_none_when_the_lock_is_absent(tmp_path):
    from dataclasses import replace

    cfg = ExecutionConfig.from_config()
    cfg = replace(cfg, selection=replace(cfg.selection, lock_file=str(tmp_path / "no_such_lock.json")))
    assert selection_lock_odds_band(cfg) is None


# ── structurally unavailable bet shapes ──────────────────────────────────────
def test_accumulators_are_blocked(cfg):
    reject_accumulator([], cfg=cfg)  # nothing to build
    reject_accumulator([{"race_uid": RACE}], cfg=cfg)  # a single is not a multiple
    with pytest.raises(AccumulatorBlocked):
        reject_accumulator([{"race_uid": RACE}, {"race_uid": "other"}], cfg=cfg)


def test_correlated_same_race_bets_are_blocked(cfg):
    existing = [{"race_uid": RACE, "horse_key": "A"}]
    reject_correlated({"race_uid": "other", "horse_key": "B"}, existing, cfg=cfg)
    with pytest.raises(CorrelatedBetBlocked):
        reject_correlated({"race_uid": RACE, "horse_key": "B"}, existing, cfg=cfg)


def test_unidentifiable_race_is_treated_as_correlated(cfg):
    with pytest.raises(CorrelatedBetBlocked):
        reject_correlated({"horse_key": "B"}, [], cfg=cfg)


# ── a loosened config cannot loosen anything ─────────────────────────────────
def test_loose_config_is_clamped_and_cannot_raise_the_ceilings():
    loose = ExecutionConfig.from_config(
        {
            "execution": {
                "paper_only": False,
                "staking": {
                    "kelly_fraction": 1.0,
                    "max_stake_pct_bankroll": 0.5,
                    "max_race_exposure_pct": 0.9,
                    "max_daily_exposure_pct": 0.9,
                    "allow_accumulators": True,
                    "allow_correlated_bets": True,
                },
            }
        }
    )
    st = loose.staking
    assert st.kelly_fraction == 0.10
    assert st.max_stake_pct_bankroll == 0.005
    assert st.max_race_exposure_pct == 0.005
    assert st.max_daily_exposure_pct == 0.03
    assert st.allow_accumulators is False
    assert st.allow_correlated_bets is False
    assert loose.clamps_applied  # the tightening is reported, not silent

    planner = StakePlanner(loose, BANKROLL)
    decision = _plan(planner, 0.90, 3.0)
    assert decision.stake == pytest.approx(5.0)
    assert decision.binding_constraint == "per_bet_cap"

    # ...and the structurally unavailable shapes stay unavailable.
    with pytest.raises(AccumulatorBlocked):
        reject_accumulator([{"race_uid": RACE}, {"race_uid": "other"}], cfg=loose)
    with pytest.raises(CorrelatedBetBlocked):
        reject_correlated(
            {"race_uid": RACE}, [{"race_uid": RACE, "horse_key": "A"}], cfg=loose
        )
