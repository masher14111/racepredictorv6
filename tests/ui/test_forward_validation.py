"""The forward-validation page's data layer, exercised without Streamlit.

The page's job is not to display numbers — it is to stop a reader drawing a
conclusion the numbers do not support. So the tests are mostly about what the
layer refuses to say: no ratio without its sample size, no CLV read stated as a
bare number, no lane collapsed into another, and no path by which an unmet gate
reads as a green light.
"""
from __future__ import annotations

import json

import pytest

from ui import forward_validation as FV


# ── loading ──────────────────────────────────────────────────────────────────
# load_run checks two artifacts (the Stage 6 daily-loop run takes priority over
# the legacy Stage 5 run); both must be pointed at missing/corrupt paths for
# these "nothing has ever run" tests to mean what they say.
def test_a_missing_run_artifact_is_none_not_an_exception(tmp_path):
    assert FV.load_run(tmp_path / "nope.json",
                        daily_loop_path=tmp_path / "nope2.json") is None


def test_a_corrupt_run_artifact_is_none(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("{not json", encoding="utf-8")
    assert FV.load_run(path, daily_loop_path=tmp_path / "nope2.json") is None


def test_a_json_list_is_refused(tmp_path):
    """The page indexes the payload by key; a list would fail deep inside render."""
    path = tmp_path / "run.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    assert FV.load_run(path, daily_loop_path=tmp_path / "nope2.json") is None


def test_latest_reports_picks_the_newest_dated_file(tmp_path):
    for name in ("forward_validation_20260701.md", "forward_validation_20260727.md",
                 "todays_candidates_20260727.md"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    found = FV.latest_reports(tmp_path)
    assert found["forward_validation"].endswith("forward_validation_20260727.md")
    assert found["candidates"].endswith("todays_candidates_20260727.md")


def test_an_empty_report_dir_yields_nulls_not_errors(tmp_path):
    assert FV.latest_reports(tmp_path) == {"forward_validation": None, "candidates": None}


# ── formatting ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("value", [None, float("nan")])
def test_an_absent_number_renders_as_a_dash_never_as_zero(value):
    """A dash reads as "unknown"; a 0.00% reads as "measured, and it was zero"."""
    assert FV.fmt_num(value) == "—"
    assert FV.fmt_num(value, pct=True) == "—"


def test_a_half_open_interval_is_not_rendered_as_an_interval():
    assert FV.fmt_interval({"low": 0.1}) == "—"
    assert FV.fmt_interval({"high": 0.1}) == "—"
    assert FV.fmt_interval(None) == "—"
    assert FV.fmt_interval({"low": -0.02, "high": 0.05}) == "-0.0200 … 0.0500"


# ── the sample-size caveat ───────────────────────────────────────────────────
def test_zero_bets_says_undefined_not_zero():
    text = FV.sample_caveat(0)
    assert "undefined, not zero" in text


@pytest.mark.parametrize("n", [1, 12, 49])
def test_a_tiny_sample_is_always_caveated(n):
    assert str(n) in FV.sample_caveat(n)


def test_a_mid_sample_is_caveated_differently_from_a_tiny_one():
    assert FV.sample_caveat(20) != FV.sample_caveat(120)
    assert "noise" in FV.sample_caveat(120)


def test_a_large_sample_carries_no_caveat():
    assert FV.sample_caveat(1000) is None


def test_an_unparseable_sample_size_is_treated_as_unknown_not_large():
    """Fail loud. A caveat that disappears on bad input is worse than none."""
    assert FV.sample_caveat("many") is not None


# ── the gate and strategy tables ─────────────────────────────────────────────
def test_gate_rows_put_failures_first():
    gate = {"criteria": [
        {"name": "min_weeks", "passed": True, "observed": 9, "required": ">= 8"},
        {"name": "model_go", "passed": False, "observed": "NO-GO", "required": "GO"},
    ]}
    assert [r["name"] for r in FV.gate_rows(gate)] == ["model_go", "min_weeks"]


def test_gate_rows_survive_a_malformed_criterion():
    gate = {"criteria": [None, "nonsense", {"name": "model_go", "passed": False}]}
    assert [r["name"] for r in FV.gate_rows(gate)] == ["model_go"]


def test_no_gate_yields_no_rows_rather_than_an_empty_pass():
    assert FV.gate_rows(None) == []
    assert FV.gate_rows({}) == []


def test_strategy_rows_lead_with_the_model():
    """The model line is the claim under test; baselines are the control."""
    rows = FV.strategy_rows({
        "favourite": {"n_bets": 153},
        "devigged_market": {"n_bets": 74},
        "model_only": {"n_bets": 61},
    })
    assert rows[0]["strategy"] == "model_only"


def test_every_strategy_row_carries_its_own_caveat():
    rows = FV.strategy_rows({"model_only": {"n_bets": 61, "roi": -0.2}})
    assert rows[0]["caveat"] is not None, "a 61-bet ROI must never travel bare"


# ── the CLV read ─────────────────────────────────────────────────────────────
def test_negative_clv_is_stated_as_paper_only_not_as_a_number():
    read = FV.clv_verdict({"model_only": {
        "mean_clv_log": -0.134, "clv_ci": {"low": -0.139, "high": -0.129},
    }})
    assert read["tone"] == "bad"
    assert "paper-only" in read["text"]
    assert "never a green light" in read["text"]


def test_positive_mean_with_an_interval_spanning_zero_is_not_a_pass():
    read = FV.clv_verdict({"model_only": {
        "mean_clv_log": 0.01, "clv_ci": {"low": -0.03, "high": 0.05},
    }})
    assert read["tone"] == "warn"
    assert "noise" in read["text"]


def test_only_an_interval_above_zero_earns_the_ok_tone():
    read = FV.clv_verdict({"model_only": {
        "mean_clv_log": 0.04, "clv_ci": {"low": 0.01, "high": 0.07},
    }})
    assert read["tone"] == "ok"
    # Even the good case refuses to license betting on its own.
    assert "forward window" in read["text"]


def test_an_unmeasurable_clv_is_a_warning_not_a_silent_pass():
    read = FV.clv_verdict({"model_only": {"n_bets": 3}})
    assert read["tone"] == "warn"
    assert "not a pass" in read["text"]


def test_no_metrics_gives_no_read():
    assert FV.clv_verdict(None) is None
    assert FV.clv_verdict({}) is None


# ── the deployment line ──────────────────────────────────────────────────────
def test_the_default_deployment_line_is_the_locked_down_one():
    """An empty payload must not read as GO / MET / eligible."""
    state = FV.deployment_line(None)
    assert state == {
        "model": "NO-GO",
        "gate": "FORWARD GATE NOT MET",
        "deployment": "PAPER-ONLY",
        "releasable": False,
        "paper_only": True,
        "generated_at": None,
    }


def test_releasable_needs_both_gates():
    both = {"model_verdict": {"go": True}, "forward_gate": {"passed": True}}
    assert FV.deployment_line(both)["releasable"] is True
    for one in ({"model_verdict": {"go": True}, "forward_gate": {"passed": False}},
                {"model_verdict": {"go": False}, "forward_gate": {"passed": True}}):
        assert FV.deployment_line(one)["releasable"] is False


def test_the_real_run_artifact_reads_as_paper_only():
    """The committed artifact, not a fixture: this is the system's actual state."""
    run = FV.load_run()
    if run is None:
        pytest.skip("no Stage-5 run artifact recorded yet")
    state = FV.deployment_line(run)
    assert state["model"] == "NO-GO"
    assert state["releasable"] is False
    assert state["deployment"] == "PAPER-ONLY"


# ── the selection block and the lane list ────────────────────────────────────
def test_selection_rows_surface_the_bias_note_and_the_lock_state():
    row = FV.selection_rows({"selection": {
        "n_strategies_tried": 64, "correction": "sidak",
        "selected": {"name": "edge0.080_ev0.050_mid_4_12", "params": {"min_edge": 0.08}},
        "test_scored_at": "2026-07-27T21:10:20+00:00", "reused_lock": True,
        "selection_bias_note": "64 strategy variant(s) were scored...",
    }})
    assert row["tried"] == 64
    assert row["reused_lock"] is True
    assert row["note"]


def test_no_selection_block_when_the_backtest_was_skipped():
    assert FV.selection_rows({"paper": {}}) is None
    assert FV.selection_rows(None) is None


def test_not_exercised_reports_what_the_backtest_could_not_cover():
    limits = FV.not_exercised({"backtest_summaries": {
        "model_only": {"not_exercised": ["dead heats", "each-way settlement"]},
    }})
    assert limits == ["dead heats", "each-way settlement"]


def test_the_three_lanes_are_fixed_and_named():
    assert FV.LANES == ("backtest", "paper", "real")
    assert set(FV.LANE_BLURB) == set(FV.LANES)


def test_the_real_lane_says_it_is_structurally_empty_not_merely_unpopulated():
    blurb = FV.LANE_BLURB["real"].lower()
    assert "no real-money execution exists" in blurb
    assert "nothing can fill it" in blurb


def test_the_backtest_lane_disclaims_the_release_gate():
    assert "cannot satisfy the release" in FV.LANE_BLURB["backtest"].lower()


# ── interval keys (the same defect the report layer had) ─────────────────────
def test_the_evaluation_layers_interval_keys_render_on_the_page():
    """``lower``/``upper`` is what ``execution.evaluation.Interval`` emits.

    Reading ``low``/``high`` produced an em-dash where the interval belongs, so
    the page showed a point estimate with nothing qualifying it.
    """
    assert FV.fmt_interval({"lower": -0.09, "upper": 0.71}, pct=True) == (
        "-9.00% … 71.00%"
    )
    assert FV.interval_bounds({"lower": 1.0, "upper": 2.0}) == (1.0, 2.0)


def test_the_legacy_spelling_still_renders():
    assert FV.fmt_interval({"low": 1.0, "high": 2.0}, 2) == "1.00 … 2.00"


def test_a_measurable_clv_is_not_reported_as_unmeasurable_on_the_page():
    verdict = FV.clv_verdict({"model_only": {
        "mean_clv_log": -0.0937, "mean_clv_pct": -0.0825,
        "clv_ci": {"lower": -0.1248, "upper": -0.0644},
    }})
    assert verdict["tone"] == "bad"
    assert "not measurable" not in verdict["text"]
    assert "never a green light to bet" in verdict["text"]
