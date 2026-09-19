"""Tests for the bet-suggestion engine (models/suggestions.py).

Covers the three things the curation layer adds on top of models.value:
  * **tiering** — Strong needs all three bars (edge / confidence / data); anything
    short is a Lean; a debutant or unknown completeness is never Strong;
  * **rationale** — plain-English "Backed by …" from SHAP positives, with an
    honest value-only fallback when no explanation exists;
  * **curation** — at most ``max_per_race`` picks, ranked Strong-before-Lean then
    by edge, honest coverage counts, and an empty book when nothing has value.
"""
import pytest

from models.suggestions import (
    SuggestionConfig,
    TIER_LEAN,
    TIER_STRONG,
    assign_tier,
    build_rationale,
    suggest_bets,
)
from models.value import ValueConfig


# ── fixtures ─────────────────────────────────────────────────────────────────

def _runner(horse_id, name, prob, odds, *, completeness=0.8, confidence="high",
            first_time=False, jockey="A Jockey", trainer="A Trainer",
            place_rate=0.4):
    """A predictor runner dict carrying everything value + suggestions read."""
    return {
        "horse_id": horse_id,
        "horse_name": name,
        "value_win_prob": prob,
        "best_odds": odds,
        "best_book": "paddy_power",
        "data_completeness": completeness,
        "confidence": confidence,
        "first_time_runner": first_time,
        "jockey": jockey,
        "trainer": trainer,
        "historical_place_rate": place_rate,
    }


def _strong_race():
    """8-runner field with one clear value horse → full field factor (conf 1.0),
    big relative edge, complete data ⇒ a Strong suggestion."""
    sels = [
        _runner("h1", "Value King", 0.40, 3.0),       # the value pick
        _runner("h2", "Filler Two", 0.05, 4.0),
        _runner("h3", "Filler Three", 0.05, 5.0),
        _runner("h4", "Filler Four", 0.05, 6.0),
        _runner("h5", "Filler Five", 0.04, 8.0),
        _runner("h6", "Filler Six", 0.03, 10.0),
        _runner("h7", "Filler Seven", 0.03, 12.0),
        _runner("h8", "Filler Eight", 0.02, 15.0),
    ]
    return {"venue": "Ascot", "race_time": "2026-06-16T14:30:00+01:00",
            "each_way_available": True, "selections": sels}


def _lean_race():
    """4-runner field: qualifies on every value gate but the thin field caps the
    confidence at 0.5 (< 0.60 Strong bar) ⇒ a Lean suggestion."""
    sels = [
        _runner("g1", "Lean Leader", 0.50, 2.5),
        _runner("g2", "Lean Two", 0.10, 3.5),
        _runner("g3", "Lean Three", 0.08, 4.5),
        _runner("g4", "Lean Four", 0.06, 6.0),
    ]
    return {"venue": "Cork", "race_time": "2026-06-16T15:10:00+01:00",
            "each_way_available": False, "selections": sels}


def _no_value_race():
    """An odds-on favourite (below the band) and no priced edge elsewhere ⇒ no
    qualifying runner."""
    sels = [
        _runner("n1", "Short Price", 0.70, 1.4),   # below min_odds
        _runner("n2", "No Edge Two", 0.20, 4.0),   # fair ~ model, no EV edge
        _runner("n3", "No Edge Three", 0.10, 8.0),
    ]
    return {"venue": "Naas", "race_time": "2026-06-16T16:00:00+01:00",
            "selections": sels}


# ── tiering ──────────────────────────────────────────────────────────────────

class TestAssignTier:
    cfg = SuggestionConfig()

    def test_all_bars_met_is_strong(self):
        tier, reasons = assign_tier(
            edge_pct=0.30, value_confidence=0.80, completeness=0.9,
            data_confidence="high", first_time=False, cfg=self.cfg)
        assert tier == TIER_STRONG
        assert reasons  # always explained

    def test_low_edge_is_lean(self):
        tier, _ = assign_tier(
            edge_pct=0.05, value_confidence=0.80, completeness=0.9,
            data_confidence="high", first_time=False, cfg=self.cfg)
        assert tier == TIER_LEAN

    def test_low_confidence_is_lean(self):
        tier, _ = assign_tier(
            edge_pct=0.30, value_confidence=0.45, completeness=0.9,
            data_confidence="high", first_time=False, cfg=self.cfg)
        assert tier == TIER_LEAN

    def test_first_time_runner_never_strong(self):
        tier, reasons = assign_tier(
            edge_pct=0.30, value_confidence=0.80, completeness=0.9,
            data_confidence="high", first_time=True, cfg=self.cfg)
        assert tier == TIER_LEAN
        assert any("first-time" in r for r in reasons)

    def test_unknown_completeness_never_strong(self):
        tier, reasons = assign_tier(
            edge_pct=0.30, value_confidence=0.80, completeness=None,
            data_confidence="high", first_time=False, cfg=self.cfg)
        assert tier == TIER_LEAN
        assert any("completeness unknown" in r for r in reasons)

    def test_low_data_confidence_never_strong(self):
        tier, _ = assign_tier(
            edge_pct=0.30, value_confidence=0.80, completeness=0.9,
            data_confidence="low", first_time=False, cfg=self.cfg)
        assert tier == TIER_LEAN

    def test_below_completeness_bar_is_lean(self):
        tier, _ = assign_tier(
            edge_pct=0.30, value_confidence=0.80, completeness=0.30,
            data_confidence="high", first_time=False, cfg=self.cfg)
        assert tier == TIER_LEAN


# ── rationale ────────────────────────────────────────────────────────────────

class TestBuildRationale:
    def test_value_only_clause_when_no_explanation(self):
        text, drivers = build_rationale(
            None, model_prob=0.40, fair_prob=0.27, edge=0.13,
            offered_odds=3.5, expected_value=0.40)
        assert drivers == []
        assert "40%" in text and "27%" in text
        assert text.endswith(".")
        assert "Backed by" not in text

    def test_backed_by_names_positive_drivers(self):
        explanation = {"top_positive": [
            {"feature": "horse_speed", "label": "Speed", "contribution": 0.5},
            {"feature": "trainer_win_rate", "label": "Trainer", "contribution": 0.3},
        ]}
        text, drivers = build_rationale(
            explanation, model_prob=0.40, fair_prob=0.27, edge=0.13,
            offered_odds=3.5, expected_value=0.40, top_k=3)
        assert text.startswith("Backed by")
        assert "strong recent speed figures" in text
        assert "an in-form trainer" in text
        assert len(drivers) == 2

    def test_non_positive_drivers_skipped(self):
        explanation = {"top_positive": [
            {"feature": "horse_speed", "label": "Speed", "contribution": -0.1},
        ]}
        text, drivers = build_rationale(
            explanation, model_prob=0.40, fair_prob=0.27, edge=0.13,
            offered_odds=3.5, expected_value=0.40)
        assert drivers == []  # negative contribution is not a "backed by" reason
        assert "Backed by" not in text

    def test_top_k_caps_driver_count(self):
        explanation = {"top_positive": [
            {"feature": "horse_speed", "contribution": 0.5},
            {"feature": "trainer_win_rate", "contribution": 0.4},
            {"feature": "course_win_rate", "contribution": 0.3},
            {"feature": "distance_win_rate", "contribution": 0.2},
        ]}
        _, drivers = build_rationale(
            explanation, model_prob=0.40, fair_prob=0.27, edge=0.13,
            offered_odds=3.5, expected_value=0.40, top_k=2)
        assert len(drivers) == 2


# ── end-to-end curation ──────────────────────────────────────────────────────

class TestSuggestBets:
    def test_strong_race_yields_one_strong(self):
        book = suggest_bets([_strong_race()], bankroll=1000.0)
        assert len(book.suggestions) == 1
        s = book.suggestions[0]
        assert s.tier == TIER_STRONG
        assert s.horse_name == "Value King"
        assert s.rank == 1
        assert book.n_strong == 1
        assert book.n_races_with_value == 1

    def test_lean_race_yields_one_lean(self):
        book = suggest_bets([_lean_race()], bankroll=1000.0)
        assert len(book.suggestions) == 1
        assert book.suggestions[0].tier == TIER_LEAN
        assert book.n_lean == 1

    def test_no_value_race_yields_nothing(self):
        book = suggest_bets([_no_value_race()], bankroll=1000.0)
        assert book.suggestions == []
        assert book.n_races_with_value == 0
        assert book.n_races == 1

    def test_max_per_race_caps_picks(self):
        # Even a field with several qualifiers contributes at most max_per_race.
        cfg = SuggestionConfig(max_per_race=1)
        book = suggest_bets([_strong_race()], config=cfg, bankroll=1000.0)
        assert len(book.suggestions) == 1

    def test_ranking_strong_before_lean(self):
        book = suggest_bets([_lean_race(), _strong_race()], bankroll=1000.0)
        tiers = [s.tier for s in book.suggestions]
        assert tiers == [TIER_STRONG, TIER_LEAN]
        assert [s.rank for s in book.suggestions] == [1, 2]

    def test_max_suggestions_global_cap(self):
        cfg = SuggestionConfig(max_suggestions=1)
        book = suggest_bets([_lean_race(), _strong_race()], config=cfg,
                            bankroll=1000.0)
        assert len(book.suggestions) == 1
        assert book.suggestions[0].tier == TIER_STRONG  # Strong survives the cap

    def test_coverage_counts_are_honest(self):
        book = suggest_bets([_strong_race(), _no_value_race()], bankroll=1000.0)
        assert book.n_races == 2
        assert book.n_races_with_value == 1

    def test_suggestion_carries_value_maths_and_stake(self):
        book = suggest_bets([_strong_race()], bankroll=1000.0)
        s = book.suggestions[0]
        assert s.offered_odds == pytest.approx(3.0)
        assert s.model_prob == pytest.approx(0.40)
        assert s.edge is not None and s.edge > 0
        assert s.suggested_stake is not None and s.suggested_stake > 0
        assert s.fair_odds is not None
        assert s.best_book == "paddy_power"

    def test_rationale_present_without_explainer(self):
        book = suggest_bets([_strong_race()], bankroll=1000.0)
        s = book.suggestions[0]
        assert s.rationale  # value-only clause at minimum
        assert s.rationale_drivers == []

    def test_rationale_uses_explainer_when_supplied(self):
        feature_rows = {"h1": {"horse_speed": 95.0}}

        class _Explainer:
            def explain(self, row, top_k=6):
                return {"top_positive": [
                    {"feature": "horse_speed", "label": "Speed",
                     "value": row.get("horse_speed"), "contribution": 0.6}]}

        book = suggest_bets([_strong_race()], bankroll=1000.0,
                            feature_rows=feature_rows, explainer=_Explainer())
        s = book.suggestions[0]
        assert "strong recent speed figures" in s.rationale
        assert s.rationale_drivers

    def test_to_dict_roundtrip(self):
        book = suggest_bets([_strong_race()], bankroll=1000.0)
        d = book.to_dict()
        assert isinstance(d["suggestions"], list)
        assert isinstance(d["suggestions"][0], dict)
        assert d["suggestions"][0]["horse_name"] == "Value King"

    def test_empty_input_is_empty_book(self):
        book = suggest_bets([], bankroll=1000.0)
        assert book.suggestions == []
        assert book.n_races == 0


# ── config ───────────────────────────────────────────────────────────────────

class TestSuggestionConfig:
    def test_from_config_defaults(self):
        cfg = SuggestionConfig.from_config({})
        assert cfg.strong_edge_pct == pytest.approx(0.15)
        assert cfg.max_per_race == 1

    def test_from_config_reads_block(self):
        cfg = SuggestionConfig.from_config(
            {"suggestions": {"strong_edge_pct": 0.25, "max_per_race": 2,
                             "max_suggestions": 5}})
        assert cfg.strong_edge_pct == pytest.approx(0.25)
        assert cfg.max_per_race == 2
        assert cfg.max_suggestions == 5

    def test_value_config_band_still_governs_selection(self):
        # The suggestion layer must not widen the validated odds band: picks above
        # max_odds (the 7.0 and 5.0 here) are rejected by the value gate it inherits.
        race = {"venue": "X", "race_time": "2026-06-16T17:00:00+01:00",
                "selections": [
                    _runner("a", "Longshot", 0.30, 7.0),
                    _runner("b", "Two", 0.10, 3.0),
                    _runner("c", "Three", 0.10, 4.0),
                    _runner("d", "Four", 0.10, 5.0),
                ]}
        book = suggest_bets([race], value_config=ValueConfig(), bankroll=1000.0)
        assert all(s.offered_odds <= 4.0 for s in book.suggestions)
