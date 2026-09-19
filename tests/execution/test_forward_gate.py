"""Requirement 7: the forward-release gate must be unclearable by today's evidence.

The gate's job is to say "not yet" for as long as the evidence says so. These tests
pin down the ways it could wrongly say "yes": accepting a backtest, passing an empty
ledger by vacuous truth, reading a positive mean CLV as a positive interval, or
letting a missing metric count as a satisfied one.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from execution.config import ExecutionConfig
from execution.forward_gate import evaluate_forward_gate
from execution.model_gate import ModelVerdict, load_model_verdict

ALL_CRITERIA = {
    "model_go",
    "min_weeks",
    "min_qualified_bets",
    "min_qualified_races",
    "positive_mean_clv",
    "clv_ci_lower",
    "ae_stable",
    "calibration",
    "drawdown",
}

GO_VERDICT = ModelVerdict(go=True, source="stub", available=True, drift_ok=True)
NO_GO_VERDICT = ModelVerdict(go=False, source="stub", available=True, reasons=("stub",))


@pytest.fixture
def cfg():
    return ExecutionConfig.from_config({})


def make_forward_ledger(
    *,
    n_bets: int = 220,
    n_races: int = 200,
    span_days: int = 63,
    clv_ratio: float = 1.08,
    prob: float = 0.20,
    win_every: int = 5,
    bet_price: float = 5.0,
    start_bankroll: float = 1000.0,
    stake: float = 1.0,
) -> pd.DataFrame:
    """A synthetic forward ledger that satisfies every numeric criterion.

    ``win_every=5`` with ``prob=0.20`` puts A/E at exactly 1.0 and ECE at 0, and the
    win/loss cycle keeps the bankroll drawdown far inside the 10% ceiling.
    """
    start = datetime(2026, 3, 2)
    rows = []
    bankroll = start_bankroll
    for i in range(n_bets):
        won = 1 if i % win_every == win_every - 1 else 0
        profit = stake * (bet_price - 1.0) if won else -stake
        before = bankroll
        bankroll += profit
        rows.append(
            {
                "race_date": start + timedelta(days=(i * span_days) // max(1, n_bets - 1)),
                "race_uid": f"R{i % n_races}",
                "prob": prob,
                "won": won,
                "stake": stake,
                "bet_price": bet_price,
                "close_price": bet_price / clv_ratio,
                "profit": profit,
                "bankroll_before": before,
                "bankroll_after": bankroll,
            }
        )
    return pd.DataFrame(rows)


def failed_names(result) -> set[str]:
    return set(result.failed)


def criterion(result, name):
    matches = [c for c in result.criteria if c.name == name]
    assert matches, f"criterion {name!r} missing from {[c.name for c in result.criteria]}"
    return matches[0]


# ── the controlling fact: today's real Stage-4 verdict ───────────────────────
def test_real_stage4_verdict_fails_the_gate_on_model_go(cfg):
    verdict = load_model_verdict()
    assert verdict.go is False, "this test encodes the real Stage-4 MODEL NO-GO"

    result = evaluate_forward_gate(
        make_forward_ledger(), cfg=cfg, model_verdict=verdict, evidence_kind="forward"
    )

    assert result.passed is False
    assert "model_go" in failed_names(result)
    model_go = criterion(result, "model_go")
    assert model_go.observed == "NO-GO"
    assert model_go.detail["reasons"], "the NO-GO reasons must travel with the gate"
    assert "NOT met" in result.summary
    assert "paper-only" in result.summary


# ── evidence kind ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("kind", ["backtest", "holdout", "", "Forward-ish"])
def test_non_forward_evidence_fails_immediately_with_a_single_criterion(cfg, kind):
    result = evaluate_forward_gate(
        make_forward_ledger(), cfg=cfg, model_verdict=GO_VERDICT, evidence_kind=kind
    )

    assert result.passed is False
    assert [c.name for c in result.criteria] == ["forward_evidence"]
    assert result.failed == ("forward_evidence",)
    assert "backtest" in criterion(result, "forward_evidence").detail["message"].lower()
    assert result.state_label == "FORWARD GATE NOT MET (1 of 1 criteria failed)"


def test_forward_evidence_kind_is_case_insensitive(cfg):
    result = evaluate_forward_gate(
        pd.DataFrame(), cfg=cfg, model_verdict=NO_GO_VERDICT, evidence_kind="Forward"
    )
    assert "forward_evidence" not in failed_names(result)
    assert failed_names(result) == ALL_CRITERIA


# ── empty / missing ledger ────────────────────────────────────────────────────
@pytest.mark.parametrize("ledger", [None, pd.DataFrame()])
def test_empty_or_missing_ledger_fails_every_criterion(cfg, ledger):
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=NO_GO_VERDICT)

    assert result.passed is False
    assert failed_names(result) == ALL_CRITERIA
    assert len(result.criteria) == 9
    assert result.state_label == "FORWARD GATE NOT MET (9 of 9 criteria failed)"
    assert (result.n_qualified_bets, result.n_qualified_races) == (0, 0)
    assert result.weeks_elapsed == 0.0
    # Nothing is computable, and nothing computable is a pass.
    for name in ("positive_mean_clv", "clv_ci_lower", "ae_stable", "calibration", "drawdown"):
        assert criterion(result, name).observed is None


def test_empty_ledger_fails_every_evidence_criterion_even_under_a_go_verdict(cfg):
    """A GO verdict is a precondition, never a substitute for forward evidence."""
    result = evaluate_forward_gate(pd.DataFrame(), cfg=cfg, model_verdict=GO_VERDICT)

    assert criterion(result, "model_go").passed is True
    assert failed_names(result) == ALL_CRITERIA - {"model_go"}
    assert result.passed is False
    # 0.0 drawdown over 0 bets is vacuous and must not clear the ceiling.
    assert criterion(result, "drawdown").passed is False


def test_ledger_without_a_race_key_reports_no_races_and_no_interval(cfg):
    ledger = make_forward_ledger().drop(columns=["race_uid"])
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    # One cluster per bet would inflate the race count AND shrink the interval.
    assert result.n_qualified_races == 0
    assert "min_qualified_races" in failed_names(result)
    assert criterion(result, "clv_ci_lower").observed is None
    assert criterion(result, "clv_ci_lower").passed is False


# ── individual criteria ───────────────────────────────────────────────────────
def test_ledger_shorter_than_eight_weeks_fails_min_weeks(cfg):
    ledger = make_forward_ledger(span_days=27)  # ~3.9 weeks
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    assert "min_weeks" in failed_names(result)
    assert result.weeks_elapsed == pytest.approx(27 / 7.0, abs=0.2)
    assert criterion(result, "min_weeks").required.startswith(">= 8")
    # Sample-size criteria are unaffected by the short window.
    assert "min_qualified_bets" not in failed_names(result)


def test_too_few_bets_and_races_fails_the_sample_criteria(cfg):
    ledger = make_forward_ledger(n_bets=40, n_races=30)
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    assert {"min_qualified_bets", "min_qualified_races"} <= failed_names(result)
    assert result.n_qualified_bets == 40
    assert result.n_qualified_races == 30
    assert criterion(result, "min_qualified_bets").required == ">= 200 qualified bets"


def test_negative_clv_fails_both_clv_criteria(cfg):
    # bet_price below the close: the Stage-4 shape (mean CLV -13.4%).
    ledger = make_forward_ledger(clv_ratio=1.0 / 1.134)
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    assert {"positive_mean_clv", "clv_ci_lower"} <= failed_names(result)
    assert criterion(result, "positive_mean_clv").observed < 0
    assert criterion(result, "clv_ci_lower").observed < 0


def test_ci_straddling_zero_fails_clv_ci_lower_even_with_a_positive_mean(cfg):
    result = evaluate_forward_gate(
        make_forward_ledger(),
        cfg=cfg,
        model_verdict=GO_VERDICT,
        metrics={"mean_clv": 0.021, "clv_ci95": (-0.018, 0.061)},
    )

    assert criterion(result, "positive_mean_clv").passed is True
    assert criterion(result, "clv_ci_lower").passed is False
    assert criterion(result, "clv_ci_lower").observed == pytest.approx(-0.018)
    assert result.passed is False


def test_ae_outside_the_band_fails(cfg):
    # Over-confident: 220 bets at p=0.20 expects 44 winners, this ledger has 22.
    ledger = make_forward_ledger(win_every=10)
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    ae = criterion(result, "ae_stable")
    assert ae.passed is False
    assert ae.observed == pytest.approx(0.5, abs=0.05)
    assert ae.required == "0.9 <= A/E <= 1.1"
    # An A/E that far off is a calibration failure too.
    assert "calibration" in failed_names(result)


def test_drawdown_beyond_the_ceiling_fails(cfg):
    result = evaluate_forward_gate(
        make_forward_ledger(),
        cfg=cfg,
        model_verdict=GO_VERDICT,
        metrics={"max_drawdown_pct": 0.18},
    )
    assert criterion(result, "drawdown").passed is False
    assert criterion(result, "drawdown").observed == pytest.approx(0.18)


def test_ledger_without_a_bankroll_column_fails_drawdown_rather_than_skipping_it(cfg):
    ledger = make_forward_ledger().drop(columns=["bankroll_before", "bankroll_after"])
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    drawdown = criterion(result, "drawdown")
    assert drawdown.passed is False
    assert drawdown.observed is None


# ── the only way through ──────────────────────────────────────────────────────
def test_all_passing_forward_ledger_with_a_go_verdict_passes(cfg):
    ledger = make_forward_ledger()
    result = evaluate_forward_gate(
        ledger,
        cfg=cfg,
        model_verdict=GO_VERDICT,
        evidence_kind="forward",
        now=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )

    assert result.failed == (), f"unexpected failures: {result.failed}"
    assert result.passed is True
    assert result.state_label == "FORWARD GATE MET (9 of 9 criteria passed)"
    assert result.n_qualified_bets == 220
    assert result.n_qualified_races == 200
    assert result.weeks_elapsed >= 8.0
    assert result.evaluated_at.startswith("2026-05-10")

    # Even a pass never asserts profitability.
    lowered = result.summary.lower()
    assert "not a claim about future results" in lowered
    assert "profitable" not in lowered
    assert "paper-only override" in lowered

    payload = result.to_dict()
    assert payload["passed"] is True
    assert len(payload["criteria"]) == 9
    assert payload["state_label"] == result.state_label


def test_qualified_flag_filters_the_ledger(cfg):
    ledger = make_forward_ledger()
    ledger["qualified"] = [i < 100 for i in range(len(ledger))]
    result = evaluate_forward_gate(ledger, cfg=cfg, model_verdict=GO_VERDICT)

    assert result.n_qualified_bets == 100
    assert "min_qualified_bets" in failed_names(result)


def test_require_toggles_drop_their_criteria(cfg):
    """The clamps force both requirements on; a hand-built cfg proves the wiring."""
    relaxed = replace(
        cfg,
        forward_gate=replace(
            cfg.forward_gate, require_model_go=False, require_positive_mean_clv=False
        ),
    )
    result = evaluate_forward_gate(
        pd.DataFrame(), cfg=relaxed, model_verdict=NO_GO_VERDICT
    )

    names = {c.name for c in result.criteria}
    assert "model_go" not in names
    assert "positive_mean_clv" not in names
    assert len(result.criteria) == 7
    # clv_ci_lower is not optional: the interval criterion always applies.
    assert "clv_ci_lower" in names
