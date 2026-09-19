"""execution/baselines.py — three selection rules, one matched race set.

The point of these tests is not that the baselines are profitable (Stage 4 says
nothing here is). It is that they are *comparable*: identical race sets, gates
actually enforced, and "no bet" preserved as the default answer.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from execution.baselines import (
    BASELINES,
    LEDGER_COLUMNS,
    assert_same_races,
    build_baseline_ledgers,
    devigged_market_selection,
    eligible_race_ids,
    favourite_selection,
    model_only_selection,
)
from execution.baselines import _fair_probs, _normalise
from execution.config import ExecutionConfig
from execution.evaluation import evaluate_ledger

# ── fixture field ────────────────────────────────────────────────────────────
# Five runners; reference book totals 1.1324 (inside the 1.25 ceiling). Runner
# "b" is available at 6.0 on the board versus a 4.5 consensus, which is the only
# way the de-vigged-market null can ever show an edge. Derived numbers:
#   fair            [0.46479, 0.19624, 0.16056, 0.09812, 0.08028]
#   market EV       [-0.117, +0.177, -0.117, -0.117, -0.117]  -> backs "b"
#   model EV        [-0.335, +0.140, +0.650, -0.280, -0.120]
#   model edge      [-0.115, -0.006, +0.139, -0.018, -0.000]  -> backs "c" only
#   shortest ref    "a" at 1.9                                -> favourite
REF_ODDS = [1.9, 4.5, 5.5, 9.0, 11.0]
EXEC_ODDS = [1.9, 6.0, 5.5, 9.0, 11.0]
MODEL_PROB = [0.35, 0.19, 0.30, 0.08, 0.08]
HORSES = ["a", "b", "c", "d", "e"]


def make_race(race_uid: str, date: str, winner: str = "c", **overrides) -> pd.DataFrame:
    race = pd.DataFrame({
        "race_uid": race_uid,
        "race_date": pd.to_datetime([date] * 5),
        "horse_key": HORSES,
        "model_prob": MODEL_PROB,
        "decimal_odds": EXEC_ODDS,
        "reference_odds": REF_ODDS,
        "closing_odds": REF_ODDS,
        "won": [1 if h == winner else 0 for h in HORSES],
    })
    for col, value in overrides.items():
        race[col] = value
    return race


@pytest.fixture
def cfg() -> ExecutionConfig:
    """Default execution config — no config.yaml read, no disk, no network."""
    return ExecutionConfig.from_config({})


@pytest.fixture
def scored() -> pd.DataFrame:
    return pd.concat([
        make_race("R1", "2026-06-01", winner="c"),
        make_race("R2", "2026-06-02", winner="a"),
        make_race("R3", "2026-06-03", winner="e"),
    ], ignore_index=True)


# ── eligibility ──────────────────────────────────────────────────────────────


def test_all_races_eligible_in_the_clean_fixture(cfg, scored):
    assert eligible_race_ids(scored, cfg=cfg) == {"R1", "R2", "R3"}


def test_short_field_is_ineligible(cfg, scored):
    trimmed = pd.concat([
        scored[scored.race_uid == "R1"].iloc[:4],   # 4 runners < min_field_size 5
        scored[scored.race_uid != "R1"],
    ], ignore_index=True)
    assert eligible_race_ids(trimmed, cfg=cfg) == {"R2", "R3"}


def test_incomplete_reference_book_is_ineligible(cfg, scored):
    """A partial book de-vigs to a fair line that is simply wrong — PASS."""
    frame = scored.copy()
    frame.loc[(frame.race_uid == "R2") & (frame.horse_key == "d"),
              "reference_odds"] = np.nan
    assert eligible_race_ids(frame, cfg=cfg) == {"R1", "R3"}


def test_unpriced_runner_is_ineligible(cfg, scored):
    frame = scored.copy()
    frame.loc[(frame.race_uid == "R3") & (frame.horse_key == "e"),
              "decimal_odds"] = np.nan
    assert eligible_race_ids(frame, cfg=cfg) == {"R1", "R2"}


def test_loose_book_is_ineligible(cfg, scored):
    """sum(1/reference) above gates.max_overround (1.25) → PASS."""
    frame = scored.copy()
    mask = frame.race_uid == "R1"
    frame.loc[mask, "reference_odds"] = [1.5, 3.0, 4.0, 6.0, 8.0]  # book ~1.60
    assert eligible_race_ids(frame, cfg=cfg) == {"R2", "R3"}


def test_persisted_ev_ineligible_flag_is_obeyed(cfg, scored):
    frame = scored.copy()
    frame["ev_eligible"] = True
    frame.loc[frame.race_uid == "R2", "ev_eligible"] = False
    assert eligible_race_ids(frame, cfg=cfg) == {"R1", "R3"}


def test_unsettled_race_is_ineligible(cfg, scored):
    frame = scored.copy()
    frame["won"] = frame["won"].astype(float)
    frame.loc[(frame.race_uid == "R1") & (frame.horse_key == "b"), "won"] = np.nan
    assert eligible_race_ids(frame, cfg=cfg) == {"R2", "R3"}


def test_missing_model_probability_is_ineligible(cfg, scored):
    """Applied to all three strategies so the race set stays matched."""
    frame = scored.copy()
    frame.loc[(frame.race_uid == "R3") & (frame.horse_key == "a"),
              "model_prob"] = np.nan
    assert eligible_race_ids(frame, cfg=cfg) == {"R1", "R2"}


def test_short_card_is_ineligible(cfg, scored):
    """The declared field says 7 but only 5 runners arrived — not the race the
    market priced."""
    frame = scored.copy()
    frame["field_size"] = 5
    frame.loc[frame.race_uid == "R3", "field_size"] = 7
    assert eligible_race_ids(frame, cfg=cfg) == {"R1", "R2"}


def test_runner_without_history_makes_the_race_ineligible(cfg, scored):
    frame = scored.copy()
    frame["value_supported"] = True
    frame.loc[(frame.race_uid == "R1") & (frame.horse_key == "e"),
              "value_supported"] = False
    assert eligible_race_ids(frame, cfg=cfg) == {"R2", "R3"}


def test_absent_gate_evidence_is_not_treated_as_satisfied_or_as_a_block(cfg, scored):
    """No history / field-size columns at all: the clean fixture stays eligible
    (the condition is unverified, documented as an upper bound), and disabling
    the gate changes nothing."""
    assert "value_supported" not in scored.columns
    relaxed = replace(cfg, gates=replace(cfg.gates, require_runner_history=False))
    assert eligible_race_ids(scored, cfg=cfg) == eligible_race_ids(scored, cfg=relaxed)


def test_empty_frame_has_no_eligible_races(cfg):
    assert eligible_race_ids(pd.DataFrame(), cfg=cfg) == set()


# ── the three selection rules ────────────────────────────────────────────────


def test_favourite_backs_the_shortest_price(cfg, scored):
    ledger = favourite_selection(scored, cfg=cfg)
    assert len(ledger) == 3
    assert set(ledger["horse_key"]) == {"a"}
    assert ledger["decimal_odds"].tolist() == [1.9, 1.9, 1.9]
    # its recorded probability is the de-vigged market prob, so A/E is defined
    assert ledger["model_prob"].iloc[0] == pytest.approx(0.464789, abs=1e-5)


def test_favourite_is_defined_by_the_reference_line_not_the_board_price(cfg):
    """A big board overlay on an outsider must not make it the favourite."""
    race = make_race("R1", "2026-06-01")
    race.loc[race.horse_key == "e", "decimal_odds"] = 1.5   # freak board price
    ledger = favourite_selection(race, cfg=cfg)
    assert ledger["horse_key"].tolist() == ["a"]


def test_model_only_backs_the_best_ev_runner_clearing_both_gates(cfg, scored):
    ledger = model_only_selection(scored, cfg=cfg)
    assert len(ledger) == 3
    # "b" has EV +0.14 but edge -0.006 (below min_edge 0.02) so it is skipped;
    # "c" clears both and has the higher EV anyway.
    assert set(ledger["horse_key"]) == {"c"}
    assert ledger["model_prob"].iloc[0] == pytest.approx(0.30)


def test_model_only_respects_the_ev_gate(cfg, scored):
    strict = replace(cfg, gates=replace(cfg.gates, min_expected_value=0.90))
    assert len(model_only_selection(scored, cfg=strict)) == 0


def test_model_only_respects_the_edge_gate(cfg, scored):
    strict = replace(cfg, gates=replace(cfg.gates, min_edge=0.20))
    assert len(model_only_selection(scored, cfg=strict)) == 0


# ── the odds band ────────────────────────────────────────────────────────────
# The fixture backs "c" at an executable 5.5. A band is a restriction on which
# runner may be *backed*; it must never reach the de-vig or the race set.


def test_a_band_containing_the_pick_changes_nothing(cfg, scored):
    banded = model_only_selection(scored, cfg=cfg, odds_band=(4.0, 12.0))
    plain = model_only_selection(scored, cfg=cfg)
    assert set(banded["horse_key"]) == set(plain["horse_key"]) == {"c"}
    assert len(banded) == len(plain) == 3


def test_a_band_excluding_the_pick_declines_rather_than_substituting(cfg, scored):
    """No second-choice runner is promoted; the strategy simply does not bet.

    "b" would be the next-best EV, but it fails min_edge. Backing it because the
    band removed "c" would report a bet the strategy's own gates refuse.
    """
    assert len(model_only_selection(scored, cfg=cfg, odds_band=(1.0, 4.0))) == 0


def test_the_upper_bound_is_exclusive_and_the_lower_inclusive(cfg, scored):
    """5.5 is inside [5.5, ...) and outside [..., 5.5). Bands must not overlap."""
    assert len(model_only_selection(scored, cfg=cfg, odds_band=(5.5, None))) == 3
    assert len(model_only_selection(scored, cfg=cfg, odds_band=(None, 5.5))) == 0


def test_a_band_does_not_move_the_devigged_fair_line(cfg, scored):
    """The regression this parameter exists for.

    Pre-filtering the frame renormalised the survivors' fair probabilities
    against a book no bookmaker offered, so ``edge = prob - fair`` was measured
    against a fiction. Here the same runner is backed under a band that drops
    three of the five runners, and its recorded probability is unchanged.
    """
    wide = model_only_selection(scored, cfg=cfg)
    narrow = model_only_selection(scored, cfg=cfg, odds_band=(5.0, 6.5))
    assert set(narrow["horse_key"]) == {"c"}
    assert narrow["model_prob"].iloc[0] == pytest.approx(wide["model_prob"].iloc[0])

    # And the direct comparison: pre-filtering *does* shift the fair line, which
    # is exactly why the band is not implemented that way.
    field = scored[scored.race_uid == "R1"]
    in_band = field[(field.decimal_odds >= 5.0) & (field.decimal_odds < 6.5)]
    assert _fair_probs(in_band).sum() == pytest.approx(1.0)
    full_fair = _fair_probs(field)[HORSES.index("c")]
    assert _fair_probs(in_band)[0] != pytest.approx(full_fair)


def test_a_band_does_not_change_which_races_are_eligible(cfg, scored):
    """Strategies scored on different race sets cannot be compared at all.

    Banding the frame made a short-price band report zero eligible races — the
    surviving book sums far below 1.0, so the completeness and overround checks
    both reject it. Eligibility must be a property of the race, not of the
    strategy looking at it.
    """
    everything = eligible_race_ids(scored, cfg=cfg)
    assert everything == {"R1", "R2", "R3"}
    short_only = scored[scored.decimal_odds < 4.0]
    assert eligible_race_ids(short_only, cfg=cfg) == set(), "the old behaviour"
    # With the band passed through instead, the race set is untouched.
    assert eligible_race_ids(scored, cfg=cfg) == everything


def test_no_band_means_the_whole_book(cfg, scored):
    for empty in (None, (), (None, None)):
        assert len(model_only_selection(scored, cfg=cfg, odds_band=empty)) == 3


def test_no_bet_is_the_default_when_nothing_clears(cfg, scored):
    """A gate that nothing clears yields an empty, correctly-shaped ledger —
    never a relaxed threshold and never an exception."""
    strict = replace(cfg, gates=replace(cfg.gates, min_expected_value=5.0))
    ledger = model_only_selection(scored, cfg=strict)
    assert list(ledger.columns) == list(LEDGER_COLUMNS)
    assert len(ledger) == 0


def test_devigged_market_backs_the_board_overlay(cfg, scored):
    ledger = devigged_market_selection(scored, cfg=cfg)
    assert set(ledger["horse_key"]) == {"b"}
    assert ledger["decimal_odds"].iloc[0] == pytest.approx(6.0)
    assert ledger["model_prob"].iloc[0] == pytest.approx(0.196244, abs=1e-5)


def test_devigged_market_bets_nothing_on_a_single_book_field(cfg):
    """The null hypothesis has no edge by construction: when the executable
    price IS the reference price, every de-vigged EV is 1/overround - 1 < 0."""
    race = make_race("R1", "2026-06-01")
    race["decimal_odds"] = REF_ODDS
    assert len(devigged_market_selection(race, cfg=cfg)) == 0


def test_devigged_probabilities_sum_to_one_per_race(cfg, scored):
    frame = _normalise(scored)
    for _, grp in frame.groupby("race_uid", sort=False):
        fair = _fair_probs(grp)
        assert fair.sum() == pytest.approx(1.0)
        assert (fair > 0).all()


# ── matching ─────────────────────────────────────────────────────────────────


def test_all_three_baselines_score_the_same_races(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    assert set(ledgers) == set(BASELINES)
    race_sets = [set(df["race_uid"]) for df in ledgers.values()]
    assert race_sets[0] == race_sets[1] == race_sets[2] == {"R1", "R2", "R3"}
    for df in ledgers.values():
        assert list(df.columns) == list(LEDGER_COLUMNS)


def test_matching_drops_races_a_strategy_passed_on(cfg, scored):
    """R2 is engineered so the model has no qualifier; all three ledgers must
    then drop R2, not just the model's."""
    frame = scored.copy()
    mask = frame.race_uid == "R2"
    # Flatten the model's view onto the fair line -> every edge below min_edge.
    frame.loc[mask, "model_prob"] = [0.46479, 0.19624, 0.16056, 0.09812, 0.08028]

    ledgers = build_baseline_ledgers(frame, cfg=cfg)
    for name, df in ledgers.items():
        assert set(df["race_uid"]) == {"R1", "R3"}, name
    assert_same_races(ledgers)


def test_empty_eligible_set_gives_three_empty_ledgers(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg, eligible=set())
    assert set(ledgers) == set(BASELINES)
    for df in ledgers.values():
        assert len(df) == 0
        assert list(df.columns) == list(LEDGER_COLUMNS)
    assert_same_races(ledgers)   # three empties still match


def test_assert_same_races_raises_on_a_mismatch(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    ledgers["favourite"] = ledgers["favourite"].iloc[:2]
    with pytest.raises(ValueError, match="race sets differ"):
        assert_same_races(ledgers)


def test_assert_same_races_accepts_a_matched_set_and_an_empty_mapping(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    assert_same_races(ledgers)      # no raise
    assert_same_races({})           # no raise


def test_assert_same_races_rejects_a_ledger_without_race_uid():
    bad = pd.DataFrame({"stake": [1.0]})
    with pytest.raises(ValueError, match="race_uid"):
        assert_same_races({"a": pd.DataFrame({"race_uid": ["R1"], "stake": [1.0]}),
                           "b": bad})


# ── contract with evaluate_ledger ────────────────────────────────────────────


def test_baseline_ledgers_settle_correctly(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    fav = ledgers["favourite"].set_index("race_uid")
    # "a" wins only R2, at 1.9 on a flat 1.0 unit.
    assert fav.loc["R2", "profit"] == pytest.approx(0.9)
    assert fav.loc["R2", "returns"] == pytest.approx(1.9)
    assert fav.loc["R1", "profit"] == pytest.approx(-1.0)
    assert fav.loc["R1", "returns"] == pytest.approx(0.0)
    assert bool(fav.loc["R1", "voided"]) is False


def test_baseline_ledgers_feed_evaluate_ledger(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    metrics = {name: evaluate_ledger(df, name=name, n_boot=100)
               for name, df in ledgers.items()}
    for name, m in metrics.items():
        assert m.n_bets == 3, name
        assert m.n_races == 3, name
        assert m.n_days == 3, name
        assert m.turnover == pytest.approx(3.0), name
    # model_only backs "c", which wins R1 at 5.5 -> +4.5, and loses R2/R3.
    assert metrics["model_only"].profit == pytest.approx(2.5)
    assert metrics["model_only"].roi == pytest.approx(2.5 / 3.0)
    assert metrics["favourite"].hit_rate == pytest.approx(1 / 3)


def test_baselines_registry_documents_every_strategy(cfg, scored):
    ledgers = build_baseline_ledgers(scored, cfg=cfg)
    assert set(BASELINES) == set(ledgers)
    assert all(isinstance(v, str) and v for v in BASELINES.values())


def test_column_aliases_are_accepted(cfg, scored):
    """A backtest frame using race_id / best_odds / close_price still works."""
    aliased = scored.rename(columns={
        "race_uid": "race_id",
        "decimal_odds": "best_odds",
        "closing_odds": "close_price",
        "horse_key": "horse_name",
    })
    ledgers = build_baseline_ledgers(aliased, cfg=cfg)
    assert set(ledgers["favourite"]["race_uid"]) == {"R1", "R2", "R3"}
    assert set(ledgers["favourite"]["horse_key"]) == {"a"}


def test_closing_price_never_reaches_a_selection(cfg, scored):
    """Mangling the close must not change a single pick — it is CLV-only."""
    baseline = build_baseline_ledgers(scored, cfg=cfg)
    mangled = scored.copy()
    mangled["closing_odds"] = 1.01
    after = build_baseline_ledgers(mangled, cfg=cfg)
    for name in BASELINES:
        assert (baseline[name]["horse_key"].tolist()
                == after[name]["horse_key"].tolist()), name
