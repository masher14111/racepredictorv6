"""Tests for scripts/reconcile_prob_to_ev.py (the Stage 3 audit harness core).

Offline: exercises the reconciliation over hand-built payloads only — the
synthetic model-run path is covered by the smoke/e2e suites.
"""
from __future__ import annotations

import pytest

from scripts.reconcile_prob_to_ev import (
    _identity_errors,
    _representative_races,
    reconcile_payload,
)


def _runner(hid="a", p=0.3, odds=4.0, **kw):
    ev = round(p * odds - 1.0, 4)
    base = {
        "horse_id": hid, "horse_name": hid.upper(),
        "implied_prob": round(1.0 / odds, 4),
        "decimal_odds": round(1.0 / round(1.0 / odds, 4), 3),
        "best_odds": odds, "best_book": "livescorebet",
        "reference_odds": odds, "reference_source": "livescorebet",
        "market_prob": round(1.0 / odds, 4),
        "won_prob_normalized": p, "catboost_win_prob": p,
        "value_win_prob": p,
        "value_edge": round(p - round(1.0 / odds, 4), 4),
        "expected_value": ev, "ev_catboost": ev,
        "value_bet": False,
    }
    base.update(kw)
    return base


def _race(runners, eligible=True, reasons=None, venue="Ascot", t="2026-07-27T14:00"):
    return {
        "venue": venue, "race_time": t, "field_size": len(runners),
        "ev_eligible": eligible,
        "ev_gate": {"eligible": eligible, "reasons": reasons or [],
                    "reference_source": "livescorebet",
                    "reference_book_complete": eligible,
                    "reference_overround": 0.2, "odds_max_age_seconds": 30.0,
                    "price_sources": ["livescorebet"],
                    "computed_at": "2026-07-27T13:55:00+01:00"},
        "runners": runners, "selections": runners[:3], "excluded_low_odds": [],
    }


class TestIdentityErrors:
    def test_clean_runner_has_no_errors(self):
        assert _identity_errors(_runner()) == []

    def test_broken_ev_is_caught(self):
        r = _runner(expected_value=0.9999)  # not p*odds-1
        errs = _identity_errors(r)
        assert any("expected_value" in e for e in errs)

    def test_broken_line_ev_is_caught(self):
        r = _runner(ev_catboost=0.0001)
        assert any("ev_catboost" in e for e in _identity_errors(r))

    def test_broken_edge_is_caught(self):
        r = _runner(value_edge=0.4242)
        assert any("value_edge" in e for e in _identity_errors(r))

    def test_inconsistent_decimal_odds_caught(self):
        r = _runner(decimal_odds=5.55)  # != 1/implied_prob
        assert any("decimal_odds" in e for e in _identity_errors(r))

    def test_null_fields_make_no_claim(self):
        r = _runner(expected_value=None, ev_catboost=None, value_edge=None,
                    value_win_prob=None)
        assert _identity_errors(r) == []


class TestReconcilePayload:
    def test_clean_eligible_race(self):
        # 4 runners at 4.0 → market book sums to 1 after devig-by-construction
        runners = [_runner(hid=f"h{i}", p=0.25, odds=4.0) for i in range(4)]
        for r in runners:
            r["market_prob"] = 0.25
        payload = {"total_runners": 4, "races": [_race(runners)]}
        res = reconcile_payload(payload)
        assert res["identity_failures"] == []
        assert res["invariant_violations"] == []
        row = res["table"].iloc[0]
        assert row["ev_eligible"] and row["win_sums_to_1"]
        assert row["market_book"] == "complete" and row["market_sums_to_1"]
        assert row["ev_identity_exact"]

    def test_pass_race_reports_reasons_and_leak_is_flagged(self):
        # An ineligible race that STILL carries an EV → invariant violation.
        leaked = [_runner(hid=f"h{i}", p=0.25, odds=4.0,
                          market_prob=None, won_prob_normalized=None)
                  for i in range(2)]
        race = _race(leaked, eligible=False, reasons=["stale_price_rows:2"])
        res = reconcile_payload({"total_runners": 2, "races": [race]})
        row = res["table"].iloc[0]
        assert not bool(row["ev_eligible"])  # pandas may store np.False_
        assert "stale_price_rows:2" in row["pass_reasons"]
        assert any("EV-ineligible race" in v for v in res["invariant_violations"])

    def test_suppressed_pass_race_is_clean(self):
        # A properly suppressed PASS race (no EV/value fields) has no violations.
        clean = [_runner(hid=f"h{i}", p=0.25, odds=4.0, market_prob=None,
                         value_edge=None, expected_value=None, ev_catboost=None,
                         won_prob_normalized=None, value_win_prob=None)
                 for i in range(2)]
        race = _race(clean, eligible=False, reasons=["incomplete_reference_book:1/2"])
        res = reconcile_payload({"total_runners": 2, "races": [race]})
        assert res["invariant_violations"] == []
        assert res["identity_failures"] == []

    def test_ev_identity_failure_reported_per_race(self):
        bad = [_runner(hid="h0", expected_value=0.5)]
        bad[0]["market_prob"] = None
        race = _race(bad, eligible=True)
        race["ev_eligible"] = None  # keep invariant 6 out of this test's way
        res = reconcile_payload({"total_runners": 1, "races": [race]})
        assert len(res["identity_failures"]) == 1
        assert not res["table"].iloc[0]["ev_identity_exact"]


class TestRepresentativeRaces:
    def test_picks_eligible_ineligible_and_value(self):
        eligible = _race([_runner()], venue="A", t="13:00")
        ineligible = _race([_runner()], eligible=False,
                           reasons=["unpriced_runners:1"], venue="B", t="13:30")
        value = _race([_runner(value_bet=True)], venue="C", t="14:00")
        picks = _representative_races(
            {"races": [eligible, ineligible, value]})
        venues = [r["venue"] for r in picks]
        assert "A" in venues and "B" in venues and "C" in venues

    def test_deduplicates(self):
        one = _race([_runner(value_bet=True)], venue="A", t="13:00")
        picks = _representative_races({"races": [one]})
        assert len(picks) == 1
