"""The two dated reports — mostly tests of what they refuse to say.

A report is the only Stage-5 artifact a person reads without running anything,
so the failure mode that matters is not a crash: it is a report that renders
cleanly while implying something the evidence does not support. These tests
therefore assert on prose as much as on structure — that a NO-GO short-circuits
before any runner table, that "no candidates" is stated as a valid answer rather
than an absence, and that the real-execution lane is present and refusing even
when there is nothing to put in it.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from execution import report as R

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

NO_GO = {
    "verdict": "NO-GO",
    "go": False,
    "reasons": [
        "logloss_edge_not_significant:delta=-0.18621 ci95=[-0.22507,-0.14620]",
        "clv_not_positive:mean=-0.13417 ci95_lower=-0.13888",
    ],
}
GO = {"verdict": "GO", "go": True, "reasons": []}

GATE_FAILED = {
    "state_label": "FORWARD GATE NOT MET (9 of 9 criteria failed)",
    "passed": False,
    "evidence_kind": "forward",
    "weeks_elapsed": 0.0,
    "n_qualified_bets": 0,
    "n_qualified_races": 0,
    "failed": ["model_go", "min_weeks"],
    "criteria": [
        {"name": "model_go", "passed": False, "observed": "NO-GO", "required": "GO"},
        {"name": "min_weeks", "passed": False, "observed": 0.0, "required": ">= 8"},
    ],
}
GATE_PASSED = {**GATE_FAILED, "state_label": "FORWARD GATE MET", "passed": True,
               "failed": [], "criteria": [
                   {"name": "model_go", "passed": True, "observed": "GO", "required": "GO"}]}

TICKET = {
    "horse_name": "Runner 0", "race_uid": "naas|2026-07-27T15:40",
    "bookmaker": "livescorebet", "offered_odds": 8.0,
    "quote_fetched_at": "2026-07-27T11:59:20+00:00", "quote_age_seconds": 40.0,
    "model_prob": 0.20, "market_adjusted_prob": 0.16, "fair_odds": 8.26,
    "market_prob": 0.1428, "edge": 0.079, "expected_value": 0.6,
    "max_stake": 5.0, "data_quality": "ok",
    "validation_state": "MODEL NO-GO / FORWARD GATE NOT MET",
    "pass_reasons": ["model_validation:NO-GO", "min_edge:0.079<0.080"],
}


# ── the candidates report under NO-GO ────────────────────────────────────────
def test_a_no_go_report_says_no_real_money_recommendations_before_any_runner():
    text = R.candidates_report(
        candidates=[TICKET], model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW
    )
    assert "no real-money recommendations" in text.lower()
    # Order matters: the refusal must precede the table, or a reader skimming
    # for the table sees prices first and the caveat second.
    assert text.lower().index("no real-money recommendations") < text.index("Runner 0")


def test_a_no_go_report_names_the_criteria_that_failed():
    text = R.candidates_report(
        model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW
    )
    for reason in NO_GO["reasons"]:
        assert reason in text
    assert "`model_go`" in text
    assert "`min_weeks`" in text


def test_runners_under_a_no_go_are_labelled_paper_candidates():
    text = R.candidates_report(
        candidates=[TICKET], model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW
    )
    assert "## Paper candidates" in text
    assert "## Candidates" not in text.replace("## Paper candidates", "")
    assert "no money is staked" in text


def test_only_a_go_and_a_met_gate_drop_the_refusal():
    text = R.candidates_report(
        candidates=[TICKET], model_verdict=GO, forward_gate=GATE_PASSED,
        paper_only=False, now=NOW,
    )
    assert "no real-money recommendations" not in text.lower()
    assert "## Candidates" in text
    # Even then, the footer still states the system stakes nothing.
    assert "paper-only" in text.lower()


def test_paper_only_forces_the_deployment_cell_regardless_of_the_gates():
    text = R.candidates_report(
        model_verdict=GO, forward_gate=GATE_PASSED, paper_only=True, now=NOW
    )
    assert "| Deployment | **PAPER-ONLY** |" in text


@pytest.mark.parametrize("verdict, gate", [
    (NO_GO, GATE_PASSED), (GO, GATE_FAILED), (NO_GO, GATE_FAILED), (None, None),
])
def test_any_unmet_gate_produces_the_refusal(verdict, gate):
    text = R.candidates_report(model_verdict=verdict, forward_gate=gate, now=NOW)
    assert "no real-money recommendations" in text.lower()


def test_an_absent_verdict_defaults_to_no_go_not_to_go():
    text = R.candidates_report(now=NOW)
    assert "| Model | **NO-GO** |" in text


# ── the honest PASS ──────────────────────────────────────────────────────────
def test_no_candidates_is_stated_as_a_valid_answer():
    text = R.candidates_report(model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW)
    assert "## No candidates" in text
    assert '"No bet" is a valid' in text


def test_the_pass_section_counts_conditions_rather_than_repeating_reasons():
    """356 identical paragraphs is not a disclosure; it is a place to hide one."""
    passes = [dict(TICKET, horse_name=f"Runner {i}") for i in range(50)]
    text = R.candidates_report(
        passes=passes, model_verdict=NO_GO, forward_gate=GATE_FAILED,
        pass_detail_path="reports/candidate_decisions_20260727.csv", now=NOW,
    )
    assert "### By condition" in text
    assert "### Distinct reasons" in text
    assert "### Per runner" in text
    # The reason string appears once in the distinct table, not 50 times.
    assert text.count("model_validation:NO-GO") < 5
    # But every runner is still named.
    for i in range(50):
        assert f"Runner {i}" in text


def test_the_pass_section_names_the_csv_that_holds_the_full_text():
    text = R.candidates_report(
        passes=[TICKET], model_verdict=NO_GO, forward_gate=GATE_FAILED,
        pass_detail_path="reports/candidate_decisions_20260727.csv", now=NOW,
    )
    assert "`reports/candidate_decisions_20260727.csv`" in text


def test_a_missing_detail_path_still_says_where_the_detail_lives():
    text = R.candidates_report(
        passes=[TICKET], model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW
    )
    assert "run artifact" in text


def test_a_pass_summary_stays_small_enough_to_read():
    """The regression: 356 runners once produced a 209 KB report."""
    passes = [dict(TICKET, horse_name=f"Runner {i}") for i in range(400)]
    text = R.candidates_report(
        passes=passes, model_verdict=NO_GO, forward_gate=GATE_FAILED,
        pass_detail_path="x.csv", now=NOW,
    )
    assert len(text) < 120_000, f"{len(text)} chars"


# ── requirement 8's disclosure columns ───────────────────────────────────────
def test_every_disclosure_column_has_a_header():
    text = R.candidates_report(
        candidates=[TICKET], model_verdict=GO, forward_gate=GATE_PASSED, now=NOW
    )
    for _, label in R._TICKET_COLUMNS:
        assert f" {label} " in text or f"| {label}" in text, label


def test_a_missing_field_renders_as_na_not_as_a_blank_cell():
    """A blank cell reads as zero or as "not applicable"; n/a reads as absent."""
    text = R.candidates_report(
        candidates=[{**TICKET, "edge": None}], model_verdict=GO,
        forward_gate=GATE_PASSED, now=NOW,
    )
    assert "n/a" in text


def test_a_candidates_pass_reasons_are_printed_under_the_table():
    text = R.candidates_report(
        candidates=[TICKET], model_verdict=GO, forward_gate=GATE_PASSED, now=NOW
    )
    assert "Why each passed the gate" in text
    assert "min_edge:0.079<0.080" in text


# ── the forward-validation report ────────────────────────────────────────────
def test_all_three_lanes_are_always_present_in_order():
    text = R.forward_validation_report(
        model_verdict=NO_GO, forward_gate=GATE_FAILED, now=NOW
    )
    i1 = text.index("Lane 1 — historical backtest")
    i2 = text.index("Lane 2 — paper / shadow bets")
    i3 = text.index("Lane 3 — real-money execution")
    assert i1 < i2 < i3


def test_the_real_lane_refuses_rather_than_reporting_no_data_yet():
    text = R.forward_validation_report(now=NOW)
    assert R.REAL_EXECUTION_LANE in text
    assert "not a lane pending data" in text or "not empty pending data" in text


def test_the_report_disclaims_short_backtests_and_staking_ceilings():
    text = R.forward_validation_report(now=NOW)
    assert "cannot show that a strategy is safe or profitable" in text
    assert "risk ceilings, not profitability claims" in text
    assert '"No bet" is a valid and expected output.' in text


def test_the_backtest_lane_is_disclaimed_even_when_it_has_no_data():
    text = R.forward_validation_report(now=NOW)
    assert "cannot satisfy the forward gate" in text
    assert "_No backtest ledger._" in text


# ── uncertainty intervals (requirement 3) ────────────────────────────────────
# The regression: ``_interval`` read ``low``/``high`` where the evaluation layer
# emits ``lower``/``upper``, so every CI in the 2026-07-27 run rendered "n/a" and
# a 31% ROI on 63 bets appeared with nothing beside it.
METRICS = {
    "model_only": {
        "n_bets": 63, "n_races": 63, "roi": 0.311,
        "roi_ci": {"lower": -0.0907, "upper": 0.7089, "excludes_zero": False},
        "mean_clv_log": -0.0937, "mean_clv_pct": -0.0825,
        "clv_ci": {"lower": -0.1248, "upper": -0.0644, "excludes_zero": True},
        "ae_ratio": 1.066, "ae_ci": {"lower": 0.737, "upper": 1.403},
        "hit_rate": 0.397, "max_drawdown_pct": 0.026,
        "longest_losing_streak": 7, "turnover": 302.6,
    }
}


def test_the_evaluation_layers_interval_keys_actually_render():
    lines = "\n".join(R.strategy_table(METRICS))
    assert "n/a" not in lines
    assert "[-9.07%, 70.89%]" in lines
    assert "[-0.1248, -0.0644]" in lines


def test_a_dataclass_interval_renders_the_same_as_its_dict():
    class Interval:
        lower, upper = -0.09, 0.71
    assert R._interval(Interval(), pct=True) == R._interval(
        {"lower": -0.09, "upper": 0.71}, pct=True
    )


def test_an_roi_whose_interval_spans_zero_is_called_indistinguishable_from_chance():
    text = "\n".join(R._roi_reading(METRICS))
    assert "not distinguishable" in text.lower()
    assert "63 bet(s)" in text
    assert "must not be reported as a return" in text


def test_an_roi_entirely_below_zero_is_called_a_loss():
    metrics = {"model_only": {"n_bets": 40, "roi": -0.2,
                              "roi_ci": {"lower": -0.31, "upper": -0.09}}}
    assert "lost money, not noise" in "\n".join(R._roi_reading(metrics))


def test_an_roi_entirely_above_zero_still_refuses_to_be_sufficient():
    metrics = {"model_only": {"n_bets": 400, "roi": 0.2,
                              "roi_ci": {"lower": 0.05, "upper": 0.35}}}
    text = "\n".join(R._roi_reading(metrics))
    assert "nowhere near a sufficient one" in text


def test_no_roi_produces_no_reading_rather_than_a_zero():
    assert R._roi_reading({}) == []
    assert R._roi_reading({"model_only": {"n_bets": 0}}) == []


def test_a_measurable_negative_clv_is_never_reported_as_unmeasurable():
    text = "\n".join(R._clv_reading(METRICS))
    assert "not measurable" not in text
    assert "never a green light to bet" in text


def test_an_absent_clv_interval_is_still_not_a_pass():
    text = "\n".join(R._clv_reading({"model_only": {"n_bets": 3}}))
    assert "Absence of a CLV read is not a pass" in text


def test_both_readings_appear_in_the_backtest_lane():
    text = R.forward_validation_report(backtest_metrics=METRICS, now=NOW)
    assert "**ROI read:**" in text
    assert "**CLV read:**" in text


# ── the walk-config attribution ──────────────────────────────────────────────
def test_the_walk_block_names_the_strategy_the_numbers_belong_to():
    text = R.forward_validation_report(walk_config={
        "source": "selected strategy", "strategy": "edge0.080_ev0.050_mid_4_12",
        "min_edge": 0.08, "min_expected_value": 0.05, "odds_band": [4.0, 12.0],
        "band_applies_to": "model_only only — baselines bet the whole book",
    }, now=NOW)
    assert "edge0.080_ev0.050_mid_4_12" in text
    assert "4.00–12.00" in text
    assert "baselines bet the whole book" in text


def test_a_default_walk_says_so_rather_than_implying_a_selection():
    text = R.forward_validation_report(walk_config={
        "source": "config defaults", "strategy": None,
        "min_edge": 0.02, "min_expected_value": 0.05, "odds_band": None,
        "band_applies_to": "",
    }, now=NOW)
    assert "config defaults" in text
    assert "whole book" in text


def test_no_walk_config_adds_nothing():
    assert R.walk_config_block(None) == []
    assert R.walk_config_block({}) == []


# ── the selection block ──────────────────────────────────────────────────────
def test_the_selection_block_reads_positional_windows():
    """``selection._window`` returns ``[start, end, n_rows]``, not a mapping."""
    lines = "\n".join(R.selection_block({
        "n_strategies_tried": 64, "correction": "sidak", "alpha": 0.05,
        "adjusted_alpha": 0.0008, "selected": {"name": "w", "params": {}},
        "windows": {"train": ["2025-01-01", "2025-11-05", 44584],
                    "validation": ["2025-11-06", "2026-02-20", 13204],
                    "test": ["2026-02-22", "2026-05-25", 14229]},
    }))
    assert "2026-02-22" in lines and "14229" in lines


def test_the_selection_block_also_reads_mapping_windows():
    """A lock round-tripped through JSON may arrive either way."""
    lines = "\n".join(R.selection_block({
        "windows": {"test": {"start": "2026-02-22", "end": "2026-05-25", "n_rows": 14229}},
    }))
    assert "2026-02-22" in lines and "14229" in lines


def test_the_selection_block_surfaces_the_bias_note_and_the_count_tried():
    lines = "\n".join(R.selection_block({
        "n_strategies_tried": 64,
        "selection_bias_note": "64 strategy variant(s) were scored...",
    }))
    assert "**64**" in lines
    assert "64 strategy variant(s)" in lines


def test_a_reused_lock_is_labelled_as_not_re_scored():
    lines = "\n".join(R.selection_block({
        "test_scored_at": "2026-07-27T21:10:20+00:00", "reused_lock": True,
    }))
    assert "NOT re-scored" in lines


def test_a_reset_lock_is_disclosed_as_extra_draws_on_the_test_window():
    """The bias note's "scored exactly once" is true per-lock, false per-programme."""
    lines = "\n".join(R.selection_block({
        "n_strategies_tried": 64, "correction": "sidak",
        "selection_bias_note": "...scored on the test window exactly once...",
        "superseded_locks": [{
            "path": "data/execution/selection_lock_superseded_20260727.json",
            "selected": {"name": "edge0.080_ev0.050_mid_4_12"},
            "test_score": -0.14455661150878496,
            "test_scored_at": "2026-07-27T21:10:20+00:00",
            "reason": "the odds-band filter corrupted the de-vig",
        }],
    }))
    assert "reset 1 time(s)" in lines
    assert "understates" in lines
    assert "edge0.080_ev0.050_mid_4_12" in lines
    assert "-0.14456" in lines
    assert "more optimistic of several draws" in lines


def test_no_reset_adds_nothing():
    assert R.superseded_locks_block(None) == []
    assert R.superseded_locks_block([]) == []


def test_a_reset_is_disclosed_even_with_nothing_recorded_about_it():
    lines = "\n".join(R.superseded_locks_block([{"path": "x.json"}]))
    assert "reset 1 time(s)" in lines
    assert "not recorded" in lines


def test_no_selection_says_the_protocol_did_not_run():
    """Silence would read as "nothing to correct for"; the truth is "unknown"."""
    lines = "\n".join(R.selection_block(None))
    assert "did not run" in lines
    assert "strategies tried" not in lines


# ── writing ──────────────────────────────────────────────────────────────────
def test_write_report_uses_a_dated_filename(tmp_path):
    path = R.write_report("hello", name="forward_validation",
                          report_dir=str(tmp_path), now=NOW)
    assert path.endswith("forward_validation_20260727.md")
    assert (tmp_path / "forward_validation_20260727.md").read_text(
        encoding="utf-8") == "hello"


def test_write_json_round_trips_and_survives_non_serialisable_values(tmp_path):
    path = str(tmp_path / "sub" / "run.json")
    R.write_json({"when": NOW, "n": 3}, path=path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["n"] == 3
    assert payload["when"].startswith("2026-07-27")
