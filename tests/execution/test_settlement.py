"""Settlement tests: Rule 4, dead heats, voids, each-way and commission.

These deductions are the difference between a simulated edge and a real one. The
Stage-4 audit put the model 0.186 race log-loss *behind* the de-vigged market, so
a settlement layer that quietly skipped a 75% Rule 4 deduction or paid an
each-way leg on terms captured at settlement rather than at bet time would be
manufacturing exactly the edge the audit ruled out.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import pytest

from backtest.metrics import settle as metrics_settle
from execution.config import ExecutionConfig, FrictionConfig
from execution.settlement import (
    MAX_COMBINED_RULE_4,
    RULE_4_TABLE,
    EachWayTerms,
    RaceResult,
    combined_rule_4,
    rule_4_deduction,
    settle_ticket,
)


def make_cfg(**friction_overrides) -> ExecutionConfig:
    return replace(
        ExecutionConfig(), frictions=replace(FrictionConfig(), **friction_overrides)
    )


def result(positions=None, **kw) -> RaceResult:
    return RaceResult(
        race_uid="2026-07-20T14:10:00+navan",
        positions=positions if positions is not None else {"ours": 1},
        **kw,
    )


def settle(**kw):
    defaults = dict(
        stake=10.0,
        decimal_odds=5.0,
        bet_type="win",
        horse_key="ours",
        race_result=result(),
    )
    defaults.update(kw)
    return settle_ticket(**defaults)


# ── Rule 4 table ─────────────────────────────────────────────────────────────
def test_every_rule_4_band_boundary_is_inclusive():
    for upper, deduction in RULE_4_TABLE:
        assert rule_4_deduction(upper) == deduction, f"{upper} should deduct {deduction}"


def test_just_above_each_boundary_falls_into_the_next_band():
    nexts = [d for _, d in RULE_4_TABLE[1:]] + [0.0]
    for (upper, _), expected in zip(RULE_4_TABLE, nexts):
        assert rule_4_deduction(upper + 1e-6) == expected, f"just above {upper}"


def test_long_prices_attract_no_deduction():
    assert rule_4_deduction(11.01) == 0.0
    assert rule_4_deduction(51.0) == 0.0


def test_unusable_withdrawal_price_takes_the_maximum_band():
    # Never the flattering assumption of "no deduction".
    assert rule_4_deduction(None) == 0.75
    assert rule_4_deduction(1.0) == 0.75  # the feeds' missing-price sentinel
    assert rule_4_deduction("not a price") == 0.75


def test_combined_withdrawals_sum_then_cap_at_ninety_percent():
    assert combined_rule_4([]) == 0.0
    assert combined_rule_4([12.0, 20.0]) == 0.0
    assert combined_rule_4([2.50, 6.00]) == pytest.approx(0.55)  # 0.40 + 0.15
    assert combined_rule_4([1.30, 1.30]) == pytest.approx(MAX_COMBINED_RULE_4)
    assert combined_rule_4([1.30, 1.30, 1.30]) == pytest.approx(0.90)


# ── Rule 4 applied to a ticket ───────────────────────────────────────────────
def test_rule_4_deducts_from_winnings_never_from_the_returned_stake():
    res = settle(
        race_result=result(
            positions={"ours": 1},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 4.0},  # 4.00 band -> 0.25
        )
    )
    # €10 at 5.0 wins €40; 25% of the winnings is deducted, the stake is not.
    assert res.rule_4_deduction == pytest.approx(0.25)
    assert res.returns == pytest.approx(40.0)  # 10 + 40*0.75
    assert res.profit == pytest.approx(30.0)
    assert res.status == "WIN"


def test_rule_4_is_not_applied_to_a_losing_ticket():
    res = settle(
        race_result=result(
            positions={"ours": 4},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 1.30},
        )
    )
    assert res.status == "LOSE"
    assert res.returns == 0.0
    assert res.profit == pytest.approx(-10.0)


def test_rule_4_can_be_disabled_by_config():
    rr = result(
        positions={"ours": 1},
        non_runners=frozenset({"scratched"}),
        withdrawn_odds={"scratched": 1.30},
    )
    on = settle(race_result=rr, cfg=make_cfg(rule_4_enabled=True))
    off = settle(race_result=rr, cfg=make_cfg(rule_4_enabled=False))
    assert on.rule_4_deduction == pytest.approx(0.75)
    assert on.returns == pytest.approx(20.0)  # 10 + 40*0.25
    assert off.rule_4_deduction == 0.0
    assert off.returns == pytest.approx(50.0)


def test_published_rule_4_override_beats_the_derived_table():
    """A deduction the racecourse published is evidence; a derived one is a guess."""
    res = settle(
        race_result=result(
            positions={"ours": 1},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 4.0},  # table would say 0.25
            rule_4_override=0.40,
        )
    )
    assert res.rule_4_deduction == pytest.approx(0.40)
    assert res.detail["rule_4_source"] == "published"
    assert res.returns == pytest.approx(34.0)  # 10 + 40*0.60


def test_rule_4_override_is_clamped_to_the_legal_range():
    high = settle(race_result=result(positions={"ours": 1}, rule_4_override=1.5))
    low = settle(race_result=result(positions={"ours": 1}, rule_4_override=-0.2))
    assert high.rule_4_deduction == pytest.approx(MAX_COMBINED_RULE_4)
    assert low.rule_4_deduction == 0.0


def test_rule_4_override_is_ignored_when_rule_4_is_disabled():
    res = settle(
        race_result=result(positions={"ours": 1}, rule_4_override=0.40),
        cfg=make_cfg(rule_4_enabled=False),
    )
    assert res.rule_4_deduction == 0.0
    assert res.detail["rule_4_source"] == "none"


def test_derived_deduction_is_labelled_as_derived():
    res = settle(
        race_result=result(
            positions={"ours": 1},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 4.0},
        )
    )
    assert res.detail["rule_4_source"] == "derived"


def test_our_own_withdrawal_price_never_deducts_from_our_own_ticket():
    """A void is a void; the horse we backed cannot Rule-4 itself."""
    res = settle(
        race_result=result(
            positions={"ours": None},
            non_runners=frozenset({"ours"}),
            withdrawn_odds={"ours": 1.30},
        )
    )
    assert res.status == "VOID"
    assert res.rule_4_deduction == 0.0


# ── voids ────────────────────────────────────────────────────────────────────
def test_our_horse_a_non_runner_voids_with_the_stake_returned():
    res = settle(race_result=result(positions={}, non_runners=frozenset({"ours"})))
    assert res.status == "VOID"
    assert res.returns == pytest.approx(10.0)
    assert res.profit == 0.0
    assert res.commission_paid == 0.0


def test_void_race_voids_every_ticket():
    res = settle(race_result=result(positions={"ours": 1}, void_race=True))
    assert res.status == "VOID"
    assert res.returns == pytest.approx(10.0)
    assert res.profit == 0.0


def test_void_non_runners_disabled_loses_the_stake_rather_than_guessing():
    res = settle(
        race_result=result(positions={}, non_runners=frozenset({"ours"})),
        cfg=make_cfg(void_non_runners=False),
    )
    assert res.status == "LOSE"
    assert res.profit == pytest.approx(-10.0)


# ── dead heats ───────────────────────────────────────────────────────────────
def test_two_horse_dead_heat_halves_the_winning_portion():
    res = settle(race_result=result(positions={"ours": 1}, dead_heat_counts={"ours": 2}))
    # €5 wins at 5.0 (=€25), €5 loses.
    assert res.dead_heat_divisor == 2.0
    assert res.returns == pytest.approx(25.0)
    assert res.profit == pytest.approx(15.0)
    assert res.status == "WIN"


def test_three_horse_dead_heat_thirds_the_winning_portion():
    res = settle(race_result=result(positions={"ours": 1}, dead_heat_counts={"ours": 3}))
    assert res.dead_heat_divisor == 3.0
    assert res.returns == pytest.approx(50.0 / 3.0)
    assert res.profit == pytest.approx(50.0 / 3.0 - 10.0)


def test_dead_heat_can_be_disabled_by_config():
    rr = result(positions={"ours": 1}, dead_heat_counts={"ours": 2})
    off = settle(race_result=rr, cfg=make_cfg(dead_heat_enabled=False))
    assert off.dead_heat_divisor == 1.0
    assert off.returns == pytest.approx(50.0)


def test_dead_heat_and_rule_4_compose():
    res = settle(
        race_result=result(
            positions={"ours": 1},
            dead_heat_counts={"ours": 2},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 4.0},  # 0.25
        )
    )
    # effective price 1 + 4*0.75 = 4.0; €5 matched -> €20 back, €5 lost.
    assert res.returns == pytest.approx(20.0)
    assert res.profit == pytest.approx(10.0)


# ── each-way ─────────────────────────────────────────────────────────────────
EW = EachWayTerms(places=3, fraction=0.2)


def test_each_way_winner_pays_both_legs():
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EW,
        race_result=result(positions={"ours": 1}),
    )
    # win leg: €5 @ 11.0 = 55; place leg: €5 @ (10*0.2+1)=3.0 = 15.
    assert res.status == "WIN"
    assert res.win_leg_return == pytest.approx(55.0)
    assert res.place_leg_return == pytest.approx(15.0)
    assert res.returns == pytest.approx(70.0)
    assert res.profit == pytest.approx(60.0)


def test_each_way_placed_pays_the_place_leg_only():
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EW,
        race_result=result(positions={"ours": 3}),
    )
    assert res.status == "PLACE"
    assert res.win_leg_return == 0.0
    assert res.place_leg_return == pytest.approx(15.0)
    assert res.returns == pytest.approx(15.0)
    assert res.profit == pytest.approx(5.0)


def test_each_way_outside_the_places_loses_both_legs():
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EW,
        race_result=result(positions={"ours": 4}),
    )
    assert res.status == "LOSE"
    assert res.returns == 0.0
    assert res.profit == pytest.approx(-10.0)


def test_each_way_terms_come_from_bet_time_not_from_the_settled_race():
    """The race ended up paying 4 places; our ticket was struck on 2."""
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EachWayTerms(places=2, fraction=0.2),
        race_result=result(positions={"ours": 3}, places_paid=4),
    )
    assert res.status == "LOSE"
    assert res.detail["places_used"] == 2


def test_fewer_places_actually_paid_binds_downwards():
    """Non-runners cut the race to 2 places; a 3-place ticket cannot outrank that."""
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EW,
        race_result=result(positions={"ours": 3}, places_paid=2),
    )
    assert res.status == "LOSE"
    assert res.detail["places_used"] == 2


def test_each_way_without_captured_terms_refuses_to_settle():
    with pytest.raises(ValueError, match="captured at bet time"):
        settle(bet_type="each_way", ew_terms=None)


def test_each_way_place_odds_match_bet_tracker_convention():
    assert EW.place_odds(11.0) == pytest.approx(3.0)  # (11-1)*0.2 + 1
    assert EachWayTerms(places=4, fraction=0.25).place_odds(9.0) == pytest.approx(3.0)


# ── each-way terms capture ───────────────────────────────────────────────────
@dataclass(frozen=True)
class FakeQuote:
    ew_places: Optional[int] = None
    ew_reduction: Optional[object] = None


def test_terms_from_quote_parses_every_feed_convention():
    assert EachWayTerms.from_quote(FakeQuote(3, "1/5")) == EachWayTerms(3, 0.2)
    assert EachWayTerms.from_quote(FakeQuote(3, 5)) == EachWayTerms(3, 0.2)
    assert EachWayTerms.from_quote(FakeQuote(3, 0.2)) == EachWayTerms(3, 0.2)
    assert EachWayTerms.from_quote(FakeQuote(4, "1/4")) == EachWayTerms(4, 0.25)


def test_terms_from_quote_never_invents_missing_terms():
    assert EachWayTerms.from_quote(FakeQuote(None, "1/5")) is None
    assert EachWayTerms.from_quote(FakeQuote(3, None)) is None
    assert EachWayTerms.from_quote(FakeQuote(0, "1/5")) is None
    assert EachWayTerms.from_quote(FakeQuote(3, "rubbish")) is None
    assert EachWayTerms.from_quote(FakeQuote(3, 0)) is None


# ── commission ───────────────────────────────────────────────────────────────
def test_commission_is_charged_on_net_winnings_only():
    res = settle(commission_rate=0.02)
    # gross 50, net winnings 40, commission 0.80.
    assert res.commission_paid == pytest.approx(0.8)
    assert res.returns == pytest.approx(49.2)
    assert res.profit == pytest.approx(39.2)


def test_commission_is_not_charged_on_a_loser_or_a_void():
    loser = settle(commission_rate=0.02, race_result=result(positions={"ours": 5}))
    void = settle(
        commission_rate=0.02,
        race_result=result(positions={}, non_runners=frozenset({"ours"})),
    )
    assert loser.commission_paid == 0.0
    assert loser.profit == pytest.approx(-10.0)
    assert void.commission_paid == 0.0
    assert void.profit == 0.0


def test_commission_comes_after_rule_4_and_the_dead_heat():
    res = settle(
        commission_rate=0.05,
        race_result=result(
            positions={"ours": 1},
            dead_heat_counts={"ours": 2},
            non_runners=frozenset({"scratched"}),
            withdrawn_odds={"scratched": 4.0},
        ),
    )
    # gross 20 (see the compose test), net winnings 10, commission 0.50.
    assert res.commission_paid == pytest.approx(0.5)
    assert res.returns == pytest.approx(19.5)
    assert res.profit == pytest.approx(9.5)


def test_plain_win_bet_agrees_exactly_with_backtest_metrics_settle():
    """The shared betting maths must not fork between modules."""
    for odds in (1.5, 2.0, 5.0, 21.0):
        for won, position in ((1, 1), (0, 6)):
            for commission in (0.0, 0.02, 0.05):
                res = settle(
                    decimal_odds=odds,
                    commission_rate=commission,
                    race_result=result(positions={"ours": position}),
                )
                expected = float(metrics_settle(10.0, odds, won, commission))
                assert res.profit == pytest.approx(expected), (odds, won, commission)


# ── missing results ──────────────────────────────────────────────────────────
def test_missing_position_is_no_result_not_a_loser():
    res = settle(race_result=result(positions={"ours": None}))
    assert res.status == "NO_RESULT"
    assert res.returns == 0.0
    assert res.profit == 0.0


def test_horse_absent_from_the_result_is_no_result():
    res = settle(race_result=result(positions={"someone_else": 1}))
    assert res.status == "NO_RESULT"
    assert res.profit == 0.0


# ── input validation and serialisation ───────────────────────────────────────
def test_an_unbackable_price_cannot_be_settled():
    with pytest.raises(ValueError, match="executable price"):
        settle(decimal_odds=1.0)


def test_unsupported_bet_type_is_refused():
    with pytest.raises(ValueError, match="unsupported bet_type"):
        settle(bet_type="forecast")


def test_leg_returns_and_commission_reconstruct_the_returns():
    res = settle(
        stake=10.0,
        decimal_odds=11.0,
        bet_type="each_way",
        ew_terms=EW,
        commission_rate=0.02,
        race_result=result(positions={"ours": 1}),
    )
    assert res.returns == pytest.approx(
        res.win_leg_return + res.place_leg_return - res.commission_paid
    )


def test_to_dict_is_json_friendly():
    d = settle().to_dict()
    assert d["status"] == "WIN"
    assert d["returns"] == pytest.approx(50.0)
    assert d["detail"]["position"] == 1
    assert d["detail"]["bet_type"] == "win"
