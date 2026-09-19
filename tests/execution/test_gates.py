"""The candidate gate must answer PASS unless every condition is proved met.

The tests are written from the Stage-4 position: with the real MODEL NO-GO
verdict on disk, *no* runner may ever be issued as a candidate, however perfect
its data. The synthetic-GO fixtures below exist only to prove the remaining
conditions are wired up — they are not a claim that the model is bettable.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from execution.config import ExecutionConfig
from execution.gates import (
    CONDITIONS,
    GateResult,
    evaluate_candidate,
    evaluate_race,
    horse_key,
    race_key,
    race_uid,
)
from execution.model_gate import LineVerdict, ModelVerdict, load_model_verdict

NOW = datetime(2026, 7, 27, 12, 0, 0, tzinfo=timezone.utc)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg() -> ExecutionConfig:
    return ExecutionConfig.from_config()


@pytest.fixture
def go_verdict() -> ModelVerdict:
    """A synthetic GO. Nothing on disk produces this; Stage 4 says NO-GO."""
    line = LineVerdict(
        name="audit_independent",
        n_races=904,
        model_log_loss=1.60,
        market_log_loss=1.69,
        logloss_delta=0.09,
        logloss_delta_ci95=(0.04, 0.14),
        beats_market_logloss=True,
        beats_best_devig=True,
    )
    return ModelVerdict(
        go=True,
        available=True,
        lines={"audit_independent": line},
        mean_clv_log=0.02,
        clv_ci95=(0.01, 0.03),
        clv_positive=True,
        drift_ok=True,
    )


@pytest.fixture
def health() -> dict:
    return {
        "livescorebet": {"status": "ok", "age_seconds": 60.0},
        "paddy_power": {"status": "ok", "age_seconds": 90.0},
    }


@pytest.fixture
def quote() -> dict:
    """A fresh, named-bookmaker quote (duck-typed like execution.snapshots.Quote)."""
    return {
        "race_uid": "2026-07-27T15:40:00Z",
        "horse_key": "littleladykaren",
        "bookmaker": "livescorebet",
        "market_type": "WIN",
        "odds_decimal": 4.4,
        "fetched_at": NOW - timedelta(seconds=45),
        "as_of": NOW,
        "age_seconds": 45.0,
        "is_stale": False,
    }


def _race() -> dict:
    """An eight-runner card whose reference book sums to 1.20 (a 20% book).

    Every runner is priced at 6.0 on the reference line, so ``fair_prob`` is
    ``(1/6)/1.20 == 0.13889`` for each of them.
    """
    runners = [
        {
            "horse_id": f"H{i}",
            "horse_name": f"Runner {i}",
            "reference_odds": 6.0,
            "best_odds": 4.4,
            "best_book": "livescorebet",
            "reference_source": "livescorebet",
            "value_supported": True,
            "value_win_prob_independent": 0.10,
        }
        for i in range(8)
    ]
    # The runner under test: a big independent edge at a takeable price.
    runners[0].update(
        {
            "horse_id": "H0",
            "horse_name": "Little Lady Karen",
            "value_win_prob_independent": 0.30,
            "value_win_prob": 0.22,
        }
    )
    return {
        "venue": "Naas",
        "race_time": "2026-07-27T15:40:00+00:00",
        "race_id": "SBTE_2_1028491168",
        "field_size": 8,
        "ev_eligible": True,
        "ev_gate": {
            "eligible": True,
            "reasons": [],
            "reference_source": "livescorebet",
            "reference_book_complete": True,
            "reference_overround": 0.20,
            "odds_max_age_seconds": 45.0,
            "price_sources": ["livescorebet", "paddy_power"],
        },
        "runners": runners,
    }


@pytest.fixture
def race() -> dict:
    return _race()


def _evaluate(race, cfg, verdict, quote, health, runner=None, **kw):
    return evaluate_candidate(
        runner if runner is not None else race["runners"][0],
        race,
        cfg=cfg,
        model_verdict=verdict,
        quote=quote,
        source_health=health,
        now=NOW,
        **kw,
    )


# ── the Stage-4 position ─────────────────────────────────────────────────────

def test_real_stage4_verdict_passes_every_runner(cfg, race, quote, health):
    """The controlling fact: MODEL NO-GO means no candidate, ever."""
    verdict = load_model_verdict()
    assert verdict.go is False, "Stage 4 issued MODEL NO-GO; this test assumes it"

    results = evaluate_race(
        race, cfg=cfg, model_verdict=verdict, source_health=health, now=NOW
    )
    assert len(results) == 8
    for result in results:
        assert result.decision == "PASS"
        assert result.is_candidate is False
        assert "model_validation:NO-GO" in result.pass_reasons
        assert "model_validation" not in result.passed


def test_no_go_reasons_carry_the_verdicts_own_reasons(cfg, race, quote, health):
    verdict = load_model_verdict()
    result = _evaluate(race, cfg, verdict, quote, health)

    model_reasons = [r for r in result.pass_reasons if r.startswith("model_validation:")]
    assert model_reasons[0] == "model_validation:NO-GO"
    for reason in verdict.reasons:
        assert f"model_validation:{reason}" in model_reasons
    assert any("does not beat the de-vigged market" in m for m in result.messages)


def test_a_perfect_runner_still_passes_under_the_real_verdict(cfg, race, quote, health):
    """Perfect data does not buy a candidate while the model is NO-GO."""
    result = _evaluate(race, cfg, load_model_verdict(), quote, health)
    assert result.decision == "PASS"
    # Every other condition was met — the model gate is the only thing blocking.
    assert set(result.pass_reasons) == {
        f"model_validation:{r}" for r in load_model_verdict().reasons
    } | {"model_validation:NO-GO"}


# ── the happy path (synthetic GO only) ───────────────────────────────────────

def test_perfect_runner_with_a_go_verdict_is_a_candidate(cfg, race, go_verdict, quote, health):
    result = _evaluate(race, cfg, go_verdict, quote, health)

    assert result.decision == "CANDIDATE"
    assert result.is_candidate is True
    assert result.pass_reasons == ()
    assert set(result.passed) == set(CONDITIONS)
    assert result.detail["probability_key"] == "value_win_prob_independent"
    assert result.detail["model_prob"] == pytest.approx(0.30)
    assert result.detail["fair_prob"] == pytest.approx((1 / 6.0) / 1.20)
    assert result.detail["edge"] == pytest.approx(0.30 - (1 / 6.0) / 1.20)
    assert result.detail["expected_value"] == pytest.approx(0.30 * 4.4 - 1)
    assert result.detail["reference_book_sum"] == pytest.approx(1.20)


def test_to_dict_round_trips_the_decision(cfg, race, go_verdict, quote, health):
    payload = _evaluate(race, cfg, go_verdict, quote, health).to_dict()
    assert payload["decision"] == "CANDIDATE"
    assert payload["is_candidate"] is True
    assert payload["pass_reasons"] == []
    assert sorted(payload["passed"]) == sorted(CONDITIONS)


# ── the race-level gate is authoritative ─────────────────────────────────────

def test_ev_ineligible_race_short_circuits_with_race_gate_reasons(
    cfg, race, go_verdict, quote, health
):
    race["ev_eligible"] = False
    race["ev_gate"]["eligible"] = False
    race["ev_gate"]["reasons"] = ["incomplete_reference_book:6/8", "stale_price_rows:2"]

    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.decision == "PASS"
    assert result.pass_reasons == (
        "race_gate:incomplete_reference_book:6/8",
        "race_gate:stale_price_rows:2",
    )
    assert result.passed == ()
    # No condition was even reached — we do not second-guess the race gate.
    assert set(result.detail["conditions"].values()) == {"not_evaluated"}


def test_ev_ineligible_race_short_circuits_even_with_every_flag_disabled(
    cfg, race, go_verdict, quote, health
):
    """A disabled require_* flag waives a *condition*, never the race decision."""
    open_gates = replace(
        cfg.gates,
        require_complete_card=False,
        require_runner_history=False,
        require_calibrated_probability=False,
        require_reference_market=False,
        require_executable_price=False,
        require_source_health=False,
        require_model_validation=False,
    )
    loose = replace(cfg, gates=open_gates)
    race["ev_eligible"] = False
    race["ev_gate"]["reasons"] = ["unpriced_runners:3"]

    result = _evaluate(race, loose, go_verdict, quote, health)
    assert result.decision == "PASS"
    assert result.pass_reasons == ("race_gate:unpriced_runners:3",)


def test_ev_ineligible_race_with_no_reasons_still_passes(cfg, race, go_verdict, quote, health):
    race["ev_eligible"] = False
    race["ev_gate"] = {}
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.pass_reasons == ("race_gate:unspecified",)


# ── one condition at a time ──────────────────────────────────────────────────
#
# Each entry flips exactly one input and declares the condition prefixes the
# result may carry. Most flips are isolated; two are coupled by the spec itself
# and say so:
#   * calibrated_probability — removing the independent probability necessarily
#     leaves edge and EV uncomputable, and an uncomputable threshold is a PASS
#     reason, never a waiver;
#   * complete_card — it is defined as "ev_eligible AND a complete reference
#     book", so an incomplete book fails reference_market too.

def _drop_independent(race):
    race["runners"][0].pop("value_win_prob_independent")


def _mark_book_incomplete(race):
    race["ev_gate"]["reference_book_complete"] = False


CONDITION_FLIPS = {
    "complete_card": (
        lambda race, quote, health: race.update({"ev_eligible": None}),
        {"complete_card"},
    ),
    "runner_history": (
        lambda race, quote, health: race["runners"][0].update({"value_supported": False}),
        {"runner_history"},
    ),
    "calibrated_probability": (
        lambda race, quote, health: _drop_independent(race),
        {"calibrated_probability", "min_edge", "min_expected_value"},
    ),
    "reference_market": (
        lambda race, quote, health: _mark_book_incomplete(race),
        {"reference_market", "complete_card"},
    ),
    "executable_price": (
        lambda race, quote, health: quote.update(
            {"age_seconds": 4000.0, "fetched_at": NOW - timedelta(seconds=4000)}
        ),
        {"executable_price"},
    ),
    "source_health": (
        lambda race, quote, health: health.update(
            {"livescorebet": {"status": "failing", "age_seconds": 30.0}}
        ),
        {"source_health"},
    ),
    "model_validation": (
        lambda race, quote, health: None,  # handled by swapping the verdict
        {"model_validation"},
    ),
    "operating_cutoff": (
        # NOW is 12:00; move the off-time to 12:00:30 — 30s away, inside the
        # (default 60s) started-race buffer.
        lambda race, quote, health: race.update(
            {"race_time": (NOW + timedelta(seconds=30)).isoformat()}
        ),
        {"operating_cutoff"},
    ),
    "field_size": (
        lambda race, quote, health: race.update({"field_size": 3}),
        {"field_size"},
    ),
    "overround": (
        lambda race, quote, health: race["ev_gate"].update({"reference_overround": 0.60}),
        {"overround"},
    ),
    "min_edge": (
        # fair == 0.13889; a 0.15 model prob leaves edge 0.011 (< 0.02) while EV
        # at 8.0 is +0.20, so only the edge gate can fail.
        lambda race, quote, health: (
            race["runners"][0].update({"value_win_prob_independent": 0.15}),
            quote.update({"odds_decimal": 8.0}),
        ),
        {"min_edge"},
    ),
    "min_expected_value": (
        # edge stays 0.161 but the executable price is short enough that
        # 0.30 * 3.4 - 1 == +0.02, below the 0.05 EV floor.
        lambda race, quote, health: quote.update({"odds_decimal": 3.4}),
        {"min_expected_value"},
    ),
}


@pytest.mark.parametrize("condition", sorted(CONDITION_FLIPS))
def test_each_condition_flipped_to_unmet_yields_pass(
    condition, cfg, race, go_verdict, quote, health
):
    flip, expected_prefixes = CONDITION_FLIPS[condition]
    flip(race, quote, health)
    verdict = load_model_verdict() if condition == "model_validation" else go_verdict

    result = _evaluate(race, cfg, verdict, quote, health)

    assert result.decision == "PASS"
    assert result.is_candidate is False
    prefixes = {reason.split(":", 1)[0] for reason in result.pass_reasons}
    assert condition in prefixes, result.pass_reasons
    assert prefixes == expected_prefixes, result.pass_reasons
    assert condition not in result.passed
    # Everything not implicated still passed — one flip breaks one thing.
    assert set(result.passed) == set(CONDITIONS) - expected_prefixes


def test_every_condition_has_a_flip_case():
    """A new condition must arrive with a test that can fail it."""
    assert set(CONDITION_FLIPS) == set(CONDITIONS)


# ── the calibrated-probability rule ──────────────────────────────────────────

def test_market_adjusted_probability_alone_does_not_calibrate(
    cfg, race, go_verdict, quote, health
):
    runner = race["runners"][0]
    runner.pop("value_win_prob_independent")
    runner["value_win_prob"] = 0.30  # a price echo, however large

    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.decision == "PASS"
    assert "calibrated_probability:market_adjusted_only" in result.pass_reasons
    assert result.detail["model_prob"] is None
    assert result.detail["market_adjusted_prob"] == pytest.approx(0.30)
    assert result.detail["probability_key"] is None
    # It must not have been smuggled into the numbers either.
    assert result.detail["edge"] is None
    assert result.detail["expected_value"] is None


def test_the_independent_key_actually_used_is_recorded(cfg, race, go_verdict, quote, health):
    runner = race["runners"][0]
    runner.pop("value_win_prob_independent")
    runner["model_win_prob_independent"] = 0.30

    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.decision == "CANDIDATE"
    assert result.detail["probability_key"] == "model_win_prob_independent"


@pytest.mark.parametrize(
    "bad",
    [
        0.0, 1.0, -0.2, 1.4, float("nan"), None,
        # A textual probability is refused rather than coerced: it means the
        # number did not come from the calibrated layer we validated.
        "0.3", b"0.3", "",
    ],
)
def test_an_out_of_range_probability_never_calibrates(
    bad, cfg, race, go_verdict, quote, health
):
    race["runners"][0]["value_win_prob_independent"] = bad
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.decision == "PASS"
    assert any(r.startswith("calibrated_probability:") for r in result.pass_reasons)


# ── disabled flags skip, they do not pass ────────────────────────────────────

@pytest.mark.parametrize(
    "condition,flag",
    [
        ("complete_card", "require_complete_card"),
        ("runner_history", "require_runner_history"),
        ("calibrated_probability", "require_calibrated_probability"),
        ("reference_market", "require_reference_market"),
        ("executable_price", "require_executable_price"),
        ("source_health", "require_source_health"),
        ("model_validation", "require_model_validation"),
    ],
)
def test_a_disabled_condition_is_skipped_not_passed(
    condition, flag, cfg, race, go_verdict, quote, health
):
    loose = replace(cfg, gates=replace(cfg.gates, **{flag: False}))
    result = _evaluate(race, loose, go_verdict, quote, health)

    assert condition not in result.passed
    assert not any(r.startswith(f"{condition}:") for r in result.pass_reasons)
    assert result.detail["skipped"] == [condition]
    assert result.detail["conditions"][condition] == "skipped"


@pytest.mark.parametrize(
    "condition", ["field_size", "overround", "min_edge", "min_expected_value"]
)
def test_numeric_thresholds_have_no_off_switch(condition, cfg, race, go_verdict, quote, health):
    """There is no require_* flag for a risk ceiling — it is always evaluated."""
    open_gates = replace(
        cfg.gates,
        require_complete_card=False,
        require_runner_history=False,
        require_calibrated_probability=False,
        require_reference_market=False,
        require_executable_price=False,
        require_source_health=False,
        require_model_validation=False,
    )
    result = _evaluate(race, replace(cfg, gates=open_gates), go_verdict, quote, health)
    assert condition in result.passed
    assert condition not in result.detail["skipped"]


def test_disabling_model_validation_cannot_issue_under_the_real_verdict(
    cfg, race, quote, health
):
    """Even with the flag off, the numeric gates still stand between us and a bet."""
    loose = replace(cfg, gates=replace(cfg.gates, require_model_validation=False))
    race["runners"][0]["value_win_prob_independent"] = 0.14  # edge 0.001

    result = _evaluate(race, loose, load_model_verdict(), quote, health)
    assert result.decision == "PASS"
    assert any(r.startswith("min_edge:") for r in result.pass_reasons)


def test_a_config_file_cannot_switch_model_validation_off(cfg, race, quote, health):
    """``gates.require_model_validation: false`` in YAML is clamped back on at load.

    Step 17 reproduced a CANDIDATE issued under the real NO-GO verdict from that one
    line, on a runner with a genuine edge — so no numeric gate stood in the way, unlike
    the contrived 0.001 edge above. The loader now treats it like
    ``forward_gate.require_model_go``: forced true, and the tightening is reported.
    """
    # Control: this race is otherwise clean, so an IN-MEMORY override really would issue.
    # (``replace`` is a programmatic/analysis path; the clamp guards operator config.)
    in_memory = replace(cfg, gates=replace(cfg.gates, require_model_validation=False))
    assert _evaluate(race, in_memory, load_model_verdict(), quote, health).decision == "CANDIDATE"

    loaded = ExecutionConfig.from_config(
        {"execution": {"gates": {"require_model_validation": False}}}
    )
    assert loaded.gates.require_model_validation is True
    assert any("require_model_validation" in line for line in loaded.clamps_applied)

    result = _evaluate(race, loaded, load_model_verdict(), quote, health)
    assert result.decision == "PASS"
    assert any(r.startswith("model_validation:") for r in result.pass_reasons)


# ── missing data always passes ───────────────────────────────────────────────

def test_an_empty_runner_and_race_pass_on_everything(cfg, go_verdict):
    result = evaluate_candidate(
        {}, {}, cfg=cfg, model_verdict=go_verdict, source_health={}, now=NOW
    )
    assert result.decision == "PASS"
    # model_validation is the one condition that says nothing about the runner —
    # it reads the frozen verdict, which here is a synthetic GO. Every condition
    # that depends on data about *this* bet has nothing to work with, so every
    # one of them is a PASS reason.
    assert result.passed == ("model_validation",)
    prefixes = {r.split(":", 1)[0] for r in result.pass_reasons}
    assert prefixes == set(CONDITIONS) - {"model_validation"}


def test_an_empty_runner_under_the_real_verdict_passes_on_all_conditions(cfg):
    """With Stage 4's actual NO-GO there is no condition left that can be met."""
    result = evaluate_candidate(
        {}, {}, cfg=cfg, model_verdict=load_model_verdict(), source_health={}, now=NOW
    )
    assert result.decision == "PASS"
    assert result.passed == ()
    prefixes = {r.split(":", 1)[0] for r in result.pass_reasons}
    assert prefixes == set(CONDITIONS)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        # The eligibility stamp, not ``ev_gate``: dropping the gate block still
        # leaves the book derivable from the runners' own reference prices (see
        # test_the_book_sum_is_derived_when_the_race_makes_no_claim), whereas an
        # unstamped race is one nothing has certified as complete.
        (lambda race, q: race.pop("ev_eligible"), "complete_card"),
        (lambda race, q: race["runners"][0].pop("value_supported"), "runner_history"),
        (lambda race, q: q.clear(), "executable_price"),
        (lambda race, q: race.pop("field_size"), None),
    ],
)
def test_missing_inputs_are_pass_reasons_not_waivers(
    mutate, expected, cfg, race, go_verdict, quote, health
):
    mutate(race, quote)
    result = _evaluate(race, cfg, go_verdict, quote, health)
    if expected is None:
        # field_size falls back to counting the declared field — that is a
        # measurement, not an assumption, so the runner may still qualify.
        assert result.detail["field_size"] == 8
        return
    assert result.decision == "PASS"
    assert any(r.startswith(f"{expected}:") for r in result.pass_reasons)


def test_absent_source_health_record_is_not_met(cfg, race, go_verdict, quote):
    result = _evaluate(race, cfg, go_verdict, quote, {})
    assert result.decision == "PASS"
    reasons = [r for r in result.pass_reasons if r.startswith("source_health:")]
    assert any("no_record" in r for r in reasons)


def test_source_health_without_an_age_is_not_fresh(cfg, race, go_verdict, quote):
    health = {
        "livescorebet": {"status": "ok", "age_seconds": None},
        "paddy_power": {"status": "ok", "age_seconds": 10.0},
    }
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "source_health:livescorebet:age_unknown" in result.pass_reasons


def test_a_stale_source_is_not_healthy(cfg, race, go_verdict, quote):
    health = {
        "livescorebet": {"status": "ok", "age_seconds": 9_000.0},
        "paddy_power": {"status": "ok", "age_seconds": 10.0},
    }
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert any(
        r.startswith("source_health:livescorebet:stale=") for r in result.pass_reasons
    )


# ── the executable price ─────────────────────────────────────────────────────

def test_a_consensus_line_is_not_an_executable_price(cfg, race, go_verdict, quote, health):
    quote["bookmaker"] = "fused_consensus"
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "executable_price:no_named_bookmaker" in result.pass_reasons


def test_a_quote_flagged_stale_by_the_store_is_refused(cfg, race, go_verdict, quote, health):
    quote["is_stale"] = True  # fresh by our arithmetic, stale per the store
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "executable_price:flagged_stale" in result.pass_reasons


def test_a_future_dated_price_is_refused(cfg, race, go_verdict, quote, health):
    quote["fetched_at"] = NOW + timedelta(seconds=120)
    quote.pop("age_seconds")
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "executable_price:timestamp_in_future" in result.pass_reasons


def test_a_price_without_a_timestamp_has_an_unknowable_age(cfg, race, go_verdict, health):
    quote = {"bookmaker": "livescorebet", "odds_decimal": 4.4}
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "executable_price:age_unknown" in result.pass_reasons


def test_a_supplied_quote_overrides_the_runners_cached_price(
    cfg, race, go_verdict, quote, health
):
    """The snapshot is the record of what was offered; the cache is not."""
    race["runners"][0]["best_odds"] = 12.0
    quote["odds_decimal"] = 4.4
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.detail["executable_odds"] == pytest.approx(4.4)
    assert result.detail["quote_origin"] == "quote"


def test_without_a_quote_the_runners_own_price_needs_a_timestamp(
    cfg, race, go_verdict, health
):
    result = _evaluate(race, cfg, go_verdict, None, health)
    assert "executable_price:age_unknown" in result.pass_reasons

    race["runners"][0]["fetched_at"] = (NOW - timedelta(seconds=30)).isoformat()
    ok = _evaluate(race, cfg, go_verdict, None, health)
    assert ok.decision == "CANDIDATE"
    assert ok.detail["quote_origin"] == "runner"


# ── the overround conversion ─────────────────────────────────────────────────

def test_overround_is_read_as_an_excess_and_compared_as_a_book_sum(
    cfg, race, go_verdict, quote, health
):
    race["ev_gate"]["reference_overround"] = 0.24  # book sums to 1.24, under 1.25
    assert _evaluate(race, cfg, go_verdict, quote, health).decision == "CANDIDATE"

    race["ev_gate"]["reference_overround"] = 0.26  # 1.26, over the ceiling
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert any(r.startswith("overround:") for r in result.pass_reasons)


def test_a_large_excess_is_never_reinterpreted_as_a_book_sum(
    cfg, race, go_verdict, quote, health
):
    """1.10 must mean a 110% *excess* (a 2.10 book), not a passable 1.10 book."""
    race["ev_gate"]["reference_overround"] = 1.10
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.detail["reference_book_sum"] == pytest.approx(2.10)
    assert any(r.startswith("overround:") for r in result.pass_reasons)


def test_the_book_sum_is_derived_when_the_race_makes_no_claim(
    cfg, race, go_verdict, quote, health
):
    """No ev_gate: completeness must be demonstrated by the runners themselves."""
    race.pop("ev_gate")
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.detail["reference_book_sum"] == pytest.approx(8 / 6.0)
    assert "reference_market" in result.passed
    # ...but the card still cannot be called complete without the race gate.
    assert "complete_card:ev_eligible_unknown" not in result.pass_reasons
    assert "complete_card" in result.passed


def test_a_partially_priced_field_is_not_a_reference_book(
    cfg, race, go_verdict, quote, health
):
    race.pop("ev_gate")
    race["runners"][3].pop("reference_odds")
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "reference_market:incomplete_reference_book" in result.pass_reasons
    assert result.detail["reference_book_sum"] is None


# ── evaluate_race ────────────────────────────────────────────────────────────

def test_evaluate_race_covers_the_complete_declared_field(cfg, race, go_verdict, health):
    race["selections"] = race["runners"][:3]
    results = evaluate_race(
        race, cfg=cfg, model_verdict=go_verdict, source_health=health, now=NOW
    )
    assert len(results) == 8, "the full field, never the display top-N"


def test_evaluate_race_routes_quotes_by_horse_key(cfg, race, go_verdict, quote, health):
    key = horse_key(race["runners"][0])
    results = evaluate_race(
        race,
        cfg=cfg,
        model_verdict=go_verdict,
        quotes={key: quote},
        source_health=health,
        now=NOW,
    )
    by_key = {r.detail["horse_key"]: r for r in results}
    assert by_key[key].detail["quote_origin"] == "quote"
    assert by_key[key].decision == "CANDIDATE"
    # Nobody else got a price, so nobody else can be a candidate.
    others = [r for k, r in by_key.items() if k != key]
    assert all(r.decision == "PASS" for r in others)


def test_evaluate_race_accepts_a_callable_quote_source(cfg, race, go_verdict, quote, health):
    results = evaluate_race(
        race,
        cfg=cfg,
        model_verdict=go_verdict,
        quotes=lambda runner: quote if runner["horse_id"] == "H0" else None,
        source_health=health,
        now=NOW,
    )
    assert results[0].decision == "CANDIDATE"


def test_evaluate_race_falls_back_to_selections_when_runners_is_absent(
    cfg, race, go_verdict, health
):
    race["selections"] = race["runners"][:3]
    race["excluded_low_odds"] = race["runners"][3:]
    race.pop("runners")
    results = evaluate_race(
        race, cfg=cfg, model_verdict=go_verdict, source_health=health, now=NOW
    )
    assert len(results) == 8


def test_evaluate_race_on_an_empty_card_returns_nothing(cfg, go_verdict):
    assert evaluate_race({}, cfg=cfg, model_verdict=go_verdict, now=NOW) == []


# ── result shape ─────────────────────────────────────────────────────────────

def test_gate_result_is_frozen(cfg, race, go_verdict, quote, health):
    result = _evaluate(race, cfg, go_verdict, quote, health)
    with pytest.raises(Exception):
        result.decision = "CANDIDATE"  # type: ignore[misc]


def test_pass_messages_are_human_sentences(cfg, race, quote, health):
    result = _evaluate(race, cfg, load_model_verdict(), quote, health)
    assert result.messages
    for message in result.messages:
        assert message[0].isupper()
        assert message.rstrip().endswith(".")


def test_the_race_is_not_mutated_by_evaluation(cfg, race, go_verdict, quote, health):
    before = copy.deepcopy(race)
    _evaluate(race, cfg, go_verdict, quote, health)
    assert race == before


# ── Stage 20: canonical identity (B5) and operating cutoff (B6) ─────────────

def test_race_key_disambiguates_same_off_time_different_venues():
    """race_uid alone drops venue once an off-time parses (B5) — two venues at
    the identical minute must not collide once race_key is also considered."""
    a = {"venue": "Leopardstown", "race_time": "2024-05-17T15:15:00+00:00"}
    b = {"venue": "York", "race_time": "2024-05-17T15:15:00+00:00"}
    assert race_uid(a) == race_uid(b)  # the known B5 collision, unchanged
    assert race_key(a) != race_key(b)  # the new identity tells them apart
    assert race_key(a) == "leopardstown|2024-05-17T15:15"
    assert race_key(b) == "york|2024-05-17T15:15"


def test_race_key_is_stable_across_naive_and_aware_timestamps():
    from execution.snapshots import race_key as _snap_race_key

    assert _snap_race_key("Naas", "2026-07-27T15:40:00+00:00") == _snap_race_key(
        "Naas", datetime(2026, 7, 27, 15, 40, 0, tzinfo=timezone.utc)
    )


def test_race_key_empty_when_venue_missing():
    from execution.snapshots import race_key as _snap_race_key

    assert _snap_race_key(None, "2026-07-27T15:40:00+00:00") == ""
    assert _snap_race_key("Naas", None) == ""


def test_operating_cutoff_blocks_a_race_already_off(cfg, race, go_verdict, quote, health):
    race["race_time"] = (NOW - timedelta(minutes=1)).isoformat()
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert result.decision == "PASS"
    assert any(r.startswith("operating_cutoff:started_or_within_buffer") for r in result.pass_reasons)


def test_operating_cutoff_blocks_inside_the_configured_buffer(cfg, race, go_verdict, quote, health):
    buffered = replace(
        cfg, safeguards=replace(cfg.safeguards, started_race_buffer_seconds=120)
    )
    race["race_time"] = (NOW + timedelta(seconds=90)).isoformat()
    result = _evaluate(race, buffered, go_verdict, quote, health)
    assert "operating_cutoff" not in result.passed
    assert any(r.startswith("operating_cutoff:") for r in result.pass_reasons)


def test_operating_cutoff_allows_a_race_safely_in_the_future(
    cfg, race, go_verdict, quote, health
):
    race["race_time"] = (NOW + timedelta(minutes=30)).isoformat()
    result = _evaluate(race, cfg, go_verdict, quote, health)
    assert "operating_cutoff" in result.passed


def test_operating_cutoff_can_be_disabled_by_config(cfg, race, go_verdict, quote, health):
    open_cfg = replace(cfg, safeguards=replace(cfg.safeguards, block_started_races=False))
    race["race_time"] = (NOW - timedelta(minutes=5)).isoformat()
    result = _evaluate(race, open_cfg, go_verdict, quote, health)
    assert "operating_cutoff" not in result.passed
    assert not any(r.startswith("operating_cutoff:") for r in result.pass_reasons)
    assert "operating_cutoff" in result.detail["skipped"]


def test_operating_cutoff_across_the_dublin_dst_spring_forward(cfg, go_verdict, quote, health):
    """2026-03-29 01:00 UTC is the moment Dublin clocks jump 01:00->02:00 IST.
    The cutoff must be judged on the real UTC instant, never on a naive local
    wall-clock subtraction that a DST jump would make nonsensical."""
    dst_now = datetime(2026, 3, 29, 0, 30, 0, tzinfo=timezone.utc)
    dst_race = _race()
    dst_race["race_time"] = "2026-03-29T01:00:00+00:00"
    # off time 30 minutes after dst_now, unaffected by the local clock jump
    # because everything here is compared in UTC.
    result = evaluate_candidate(
        dst_race["runners"][0], dst_race, cfg=cfg, model_verdict=go_verdict,
        quote=quote, source_health=health, now=dst_now,
    )
    assert "operating_cutoff" in result.passed
