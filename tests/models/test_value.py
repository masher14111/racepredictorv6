"""Tests for the value-detection layer (models/value.py).

Covers de-vig (over-round stripping), edge/EV/Kelly maths, every config-driven
gate, runner extraction from each supported input shape, ranking, and the
backtester validation replay (evaluate_filter).
"""
import numpy as np
import pandas as pd
import pytest

from models.value import (
    ValueConfig,
    _confidence,
    _extract_runners,
    devig_field,
    evaluate_filter,
    find_value_bets,
)


# ── de-vig ─────────────────────────────────────────────────────────────────--

class TestDevigField:
    def test_fair_probs_sum_to_one(self):
        fair, _ = devig_field([2.0, 4.0, 5.0])
        assert fair.sum() == pytest.approx(1.0)

    def test_overround_is_raw_sum_minus_one(self):
        # raw implied: .5 + .25 + .2 = .95 -> under-round (-0.05)
        _, ov = devig_field([2.0, 4.0, 5.0])
        assert ov == pytest.approx(-0.05)

    def test_positive_overround_for_real_book(self):
        # a margined book: 1/2 + 1/2 + 1/4 = 1.25 -> overround 0.25
        fair, ov = devig_field([2.0, 2.0, 4.0])
        assert ov == pytest.approx(0.25)
        assert fair.sum() == pytest.approx(1.0)

    def test_fair_below_raw_when_margined(self):
        # de-vigging a margined book pulls each fair prob below its raw implied.
        fair, _ = devig_field([2.0, 2.0, 4.0])
        assert fair[0] < 0.5  # raw implied was 0.5

    def test_invalid_price_excluded(self):
        # price <= 1 is invalid -> NaN fair, others still normalise over valid set
        fair, _ = devig_field([2.0, 4.0, 1.0])
        assert np.isnan(fair[2])
        assert np.nansum(fair) == pytest.approx(1.0)


# ── confidence ─────────────────────────────────────────────────────────────--

class TestConfidence:
    cfg = ValueConfig()

    def test_unsupported_runner_zero_confidence(self):
        c = _confidence(np.array([4.0]), n_priced=8,
                        supported=np.array([False]), cfg=self.cfg)
        assert c[0] == 0.0

    def test_full_confidence_supported_full_field_short_odds(self):
        c = _confidence(np.array([4.0]), n_priced=8,
                        supported=np.array([True]), cfg=self.cfg)
        assert c[0] == pytest.approx(1.0)

    def test_thin_field_reduces_confidence(self):
        c = _confidence(np.array([4.0]), n_priced=2,
                        supported=np.array([True]), cfg=self.cfg)
        assert c[0] == pytest.approx(0.3)  # clip(2/8, 0.3, 1)

    def test_longshot_odds_decay(self):
        # within (reliable_max=8, 24] odds-reliability decays below 1
        c = _confidence(np.array([16.0]), n_priced=8,
                        supported=np.array([True]), cfg=self.cfg)
        assert 0.0 < c[0] < 1.0

    def test_far_longshot_zero(self):
        c = _confidence(np.array([50.0]), n_priced=8,
                        supported=np.array([True]), cfg=self.cfg)
        assert c[0] == 0.0


# ── runner extraction ──────────────────────────────────────────────────────--

class TestExtractRunners:
    def test_predictor_race_dict(self):
        race = {
            "selections": [{"horse_name": "A", "value_win_prob": 0.3, "best_odds": 4.0}],
            "excluded_low_odds": [{"horse_name": "B", "value_win_prob": 0.5, "best_odds": 1.5}],
        }
        df = _extract_runners(race)
        assert len(df) == 2
        assert set(df["horse_name"]) == {"A", "B"}

    def test_single_runner_dict(self):
        df = _extract_runners({"horse_name": "A", "value_win_prob": 0.3, "best_odds": 4.0})
        assert len(df) == 1 and df["model_prob"].iat[0] == 0.3

    def test_list_of_runners(self):
        df = _extract_runners([{"value_win_prob": 0.2, "decimal_odds": 5.0}])
        assert df["decimal_odds"].iat[0] == 5.0

    def test_best_odds_preferred_over_decimal(self):
        df = _extract_runners([{"value_win_prob": 0.2, "best_odds": 6.0, "decimal_odds": 5.0}])
        assert df["decimal_odds"].iat[0] == 6.0

    def test_support_inferred_from_historical_place_rate(self):
        rows = [
            {"value_win_prob": 0.2, "decimal_odds": 5.0, "historical_place_rate": 0.3},
            {"value_win_prob": 0.2, "decimal_odds": 5.0, "historical_place_rate": None},
        ]
        df = _extract_runners(rows)
        assert list(df["supported"]) == [True, False]

    def test_dataframe_with_prob_and_bet_price(self):
        raw = pd.DataFrame({"prob": [0.3], "bet_price": [4.0], "horse_name": ["A"]})
        df = _extract_runners(raw)
        assert df["model_prob"].iat[0] == 0.3 and df["decimal_odds"].iat[0] == 4.0

    def test_dataframe_missing_columns_raises(self):
        with pytest.raises(ValueError):
            _extract_runners(pd.DataFrame({"foo": [1]}))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _extract_runners(42)


# ── find_value_bets ────────────────────────────────────────────────────────--

def _race(runners):
    return {"selections": [dict(value_supported=True, **r) for r in runners]}


class TestFindValueBets:
    # Gate-under-test configs disable de-vig (exact edge maths) and the confidence
    # gate (a small synthetic field would otherwise trip it); confidence/support
    # gates get their own dedicated tests below.
    def _cfg(self, **kw):
        kw.setdefault("devig", False)
        kw.setdefault("min_confidence", 0.0)
        return ValueConfig(**kw)

    def _even_field(self, probs, odds):
        return _race([{"horse_name": chr(65 + i), "horse_id": str(i),
                       "value_win_prob": pr, "best_odds": od}
                      for i, (pr, od) in enumerate(zip(probs, odds))])

    def test_returns_value_pick_when_model_beats_market(self):
        # runner priced 4.0 (raw implied .25) but model says .40 -> EV .6, in band
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg())
        assert picks and picks[0]["horse_name"] == "A"
        assert picks[0]["expected_value"] == pytest.approx(0.40 * 4.0 - 1.0)

    def test_edge_uses_fair_prob_not_raw(self):
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        # de-vig OFF: edge = p - raw_implied = .40 - .25 = .15
        off = find_value_bets(race, self._cfg(devig=False))[0]
        assert off["edge"] == pytest.approx(0.40 - 0.25)
        # de-vig ON: fair prob differs from the raw implied, so the edge differs.
        on = find_value_bets(race, self._cfg(devig=True))[0]
        assert on["edge"] != pytest.approx(off["edge"])
        assert on["fair_prob"] != pytest.approx(on["market_implied_prob"])

    def test_ev_gate_blocks_low_ev(self):
        # model .26 at 4.0 -> EV .04 < min_ev .05; B at .30/4.0 (EV .2) still passes
        race = self._even_field([0.26, 0.30, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg())
        names = {p["horse_name"] for p in picks}
        assert "A" not in names and "B" in names

    def test_min_odds_gate(self):
        # A: strong EV but priced 1.5 < min_odds 2.0; B at 5.0 passes
        race = self._even_field([0.80, 0.30, 0.10], [1.5, 5.0, 8.0])
        picks = find_value_bets(race, self._cfg(max_odds=10))
        names = {p["horse_name"] for p in picks}
        assert "A" not in names and "B" in names
        assert all(p["decimal_odds"] >= 2.0 for p in picks)

    def test_max_odds_longshot_gate(self):
        # huge EV at 12.0 but above default max_odds 4.0
        race = self._even_field([0.30, 0.10, 0.10], [12.0, 8.0, 8.0])
        picks = find_value_bets(race, self._cfg())
        assert picks == []

    def test_min_abs_edge_gate(self):
        # A raw edge .15 passes a .10 floor but B (.30-.25=.05) is dropped
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg(min_abs_edge=0.10))
        names = {p["horse_name"] for p in picks}
        assert "A" in names and "B" not in names

    def test_min_edge_pct_gate(self):
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        # A: p/raw_fair-1 = .40/.25-1 = .60; require 1.0 -> dropped
        picks = find_value_bets(race, self._cfg(min_edge_pct=1.0))
        assert all(p["horse_name"] != "A" for p in picks)

    def test_min_prob_gate(self):
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg(min_prob=0.45))
        assert picks == []

    def test_require_support_blocks_formless(self):
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0, "value_supported": False},
            {"horse_name": "B", "value_win_prob": 0.30, "best_odds": 4.0, "value_supported": True},
        ]}
        picks = find_value_bets(race, self._cfg())
        names = {p["horse_name"] for p in picks}
        assert "A" not in names and "B" in names

    def test_support_not_required_when_disabled(self):
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0, "value_supported": False},
        ]}
        # confidence still needs support; disabling require_support AND lowering
        # min_confidence (unsupported -> confidence 0) lets it through.
        picks = find_value_bets(race, self._cfg(require_support=False))
        assert picks and picks[0]["horse_name"] == "A"

    def test_min_confidence_gate(self):
        # one priced runner -> thin field; confidence = clip(1/8,0.3,1)=0.3 < .40
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0, "value_supported": True},
        ]}
        assert find_value_bets(race, ValueConfig(devig=False)) == []
        assert find_value_bets(race, ValueConfig(devig=False, min_confidence=0.2))

    def test_kelly_stake_capped(self):
        # massive edge -> full Kelly large, but suggested stake capped at 5% bankroll
        race = self._even_field([0.90, 0.05, 0.05], [3.0, 8.0, 8.0])
        picks = find_value_bets(race, self._cfg(kelly_cap=0.05), bankroll=1000)
        assert picks[0]["suggested_stake"] == pytest.approx(50.0)  # 5% of 1000

    def test_kelly_stake_uses_fraction(self):
        # full Kelly for p=.40,d=4.0 is (.4*4-1)/3 = .2; quarter-Kelly stake = .05*bank
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg(kelly_fraction=0.25, kelly_cap=1.0),
                                bankroll=1000)
        a = next(p for p in picks if p["horse_name"] == "A")
        assert a["kelly_fraction"] == pytest.approx(0.2)
        assert a["suggested_stake"] == pytest.approx(0.25 * 0.2 * 1000)

    def test_ranked_by_edge_descending(self):
        race = self._even_field([0.40, 0.35, 0.20], [4.0, 4.0, 5.0])
        picks = find_value_bets(race, self._cfg())
        edges = [p["edge"] for p in picks]
        assert edges == sorted(edges, reverse=True)
        assert [p["rank"] for p in picks] == list(range(1, len(picks) + 1))

    def test_empty_input_returns_empty(self):
        assert find_value_bets({"selections": []}) == []
        assert find_value_bets([]) == []

    def test_disabled_config_returns_empty(self):
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        assert find_value_bets(race, self._cfg(enabled=False)) == []

    def test_nan_price_skipped(self):
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": None, "value_supported": True},
            {"horse_name": "B", "value_win_prob": 0.40, "best_odds": 4.0, "value_supported": True},
        ]}
        picks = find_value_bets(race, self._cfg())
        names = {p["horse_name"] for p in picks}
        assert "A" not in names and "B" in names

    def test_bankroll_override(self):
        race = self._even_field([0.40, 0.30, 0.20], [4.0, 4.0, 5.0])
        small = find_value_bets(race, self._cfg(kelly_cap=1.0), bankroll=100)
        big = find_value_bets(race, self._cfg(kelly_cap=1.0), bankroll=1000)
        assert big[0]["suggested_stake"] == pytest.approx(10 * small[0]["suggested_stake"])


# ── staleness gate (belt-and-suspenders vs features/fuse.py) ────────────────--

class TestStalenessGate:
    """Belt-and-suspenders: a caller that bypasses features/fuse.py's fail-closed
    TTL drop still can't get a stale EV pick out of find_value_bets directly."""

    def _cfg(self, **kw):
        kw.setdefault("devig", False)
        kw.setdefault("min_confidence", 0.0)
        return ValueConfig(**kw)

    def test_explicit_stale_flag_excludes_runner(self):
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0,
             "value_supported": True, "stale": True,
             "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
        ]}
        assert find_value_bets(race, self._cfg()) == []

    def test_fetched_at_beyond_ttl_excludes_runner(self):
        old = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=1000)).isoformat()
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0,
             "value_supported": True, "stale": False, "fetched_at": old},
        ]}
        assert find_value_bets(race, self._cfg(max_stale_seconds=900)) == []

    def test_fetched_at_within_ttl_keeps_runner(self):
        recent = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=60)).isoformat()
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0,
             "value_supported": True, "stale": False, "fetched_at": recent},
        ]}
        picks = find_value_bets(race, self._cfg(max_stale_seconds=900))
        assert picks and picks[0]["horse_name"] == "A"

    def test_no_fetched_at_is_not_gated(self):
        # Callers without provenance (e.g. existing tests/mocks) keep working —
        # only an explicit stale signal excludes a runner.
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0,
             "value_supported": True},
        ]}
        picks = find_value_bets(race, self._cfg())
        assert picks and picks[0]["horse_name"] == "A"

    def test_zero_max_stale_seconds_disables_ttl_gate(self):
        old = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=99999)).isoformat()
        race = {"selections": [
            {"horse_name": "A", "value_win_prob": 0.40, "best_odds": 4.0,
             "value_supported": True, "stale": False, "fetched_at": old},
        ]}
        picks = find_value_bets(race, self._cfg(max_stale_seconds=0))
        assert picks and picks[0]["horse_name"] == "A"

    def test_dataframe_input_respects_stale_column(self):
        df = pd.DataFrame([
            {"horse_id": "1", "horse_name": "A", "value_win_prob": 0.40,
             "best_odds": 4.0, "stale": True,
             "fetched_at": pd.Timestamp.now(tz="UTC").isoformat()},
        ])
        assert find_value_bets(df, self._cfg()) == []


# ── evaluate_filter (backtester validation replay) ──────────────────────────--

class TestEvaluateFilter:
    def _scored(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        # favourites the model slightly under-rates win more often (mirrors OOS).
        price = rng.uniform(2.0, 12.0, n)
        prob = np.clip(1.0 / price + rng.normal(0, 0.02, n), 0.01, 0.9)
        won = (rng.uniform(size=n) < (1.0 / price)).astype(int)
        close = price * rng.uniform(0.9, 1.2, n)
        dates = pd.date_range("2025-01-01", periods=n, freq="h")
        return pd.DataFrame({"race_date": dates, "prob": prob, "bet_price": price,
                             "close_price": close, "won": won})

    def test_returns_expected_keys(self):
        m = evaluate_filter(self._scored(), ValueConfig())
        for k in ("n_bets", "yield_pct", "clv_pct_mean", "beat_close_rate",
                  "hit_rate", "avg_odds", "ae_by_odds", "flat_final_bankroll",
                  "kelly_final_bankroll"):
            assert k in m

    def test_respects_odds_band(self):
        m = evaluate_filter(self._scored(), ValueConfig(min_odds=2.0, max_odds=6.0))
        assert m["n_bets"] > 0
        assert m["avg_odds"] <= 6.0

    def test_empty_selection_when_ev_unreachable(self):
        m = evaluate_filter(self._scored(), ValueConfig(min_ev=10.0))
        assert m["n_bets"] == 0
        assert m["flat_final_bankroll"] == pytest.approx(ValueConfig().bankroll)

    def test_real_oos_frame_reproduces_validated_numbers(self):
        """Guard the headline validated result on the saved OOS run, if present."""
        import os
        path = "data/backtests/20260616_195957/scored.parquet"
        if not os.path.exists(path):
            pytest.skip("saved backtest run not present")
        s = pd.read_parquet(path)
        # Guards the model-16 RAW-prob baseline at the original [2,6] band, so it
        # stays an invariant independent of the shipped default (now [2,4]).
        m = evaluate_filter(s, ValueConfig(max_odds=6.0))  # [2,6], EV>=.05
        assert m["n_bets"] == 1665
        assert m["yield_pct"] == pytest.approx(5.34, abs=0.05)
        assert m["clv_pct_mean"] == pytest.approx(-7.19, abs=0.05)  # honest: CLV negative
        assert m["flat_final_bankroll"] > ValueConfig().bankroll     # bankroll grows


# ── F-L recalibrated probabilities flow through the value layer ──────────────--

class TestRecalibratedValueWinProb:
    """The predictor recalibrates ``value_win_prob`` (favourites up, longshots
    down) BEFORE find_value_bets sees it, so the value layer reflects the corrected
    probability with no change to its own logic: a recalibrated favourite surfaces
    as value, a deflated longshot drops out."""

    def test_recalibrated_favourite_surfaces_as_value(self):
        # Favourite at 4.0 (implied 0.25) whose price-free prob was recalibrated UP
        # to 0.32 → EV = 0.32*4 - 1 = +0.28, inside the [2,4] band.
        race = [{"horse_name": "Fav", "value_win_prob": 0.32, "best_odds": 4.0,
                 "value_supported": True}]
        picks = find_value_bets(race, ValueConfig(min_confidence=0.0))
        assert picks and picks[0]["horse_name"] == "Fav"
        assert picks[0]["model_prob"] == pytest.approx(0.32)
        assert picks[0]["expected_value"] == pytest.approx(0.28, abs=1e-6)

    def test_deflated_longshot_gated_out(self):
        # Longshot at 20.0 whose prob was recalibrated DOWN to 0.03 → EV negative and
        # outside max_odds; no value bet (the longshot trap the recalibration closes).
        race = [{"horse_name": "Long", "value_win_prob": 0.03, "best_odds": 20.0,
                 "value_supported": True}]
        assert find_value_bets(race, ValueConfig(min_confidence=0.0)) == []

    def test_same_runner_flips_with_recalibration(self):
        """One runner, two probs: the raw price-free prob misses the EV gate; the
        recalibrated (higher) prob clears it — proving the layer tracks the prob."""
        base = {"horse_name": "R", "best_odds": 4.0, "value_supported": True}
        cfg = ValueConfig(min_confidence=0.0)
        assert find_value_bets([{**base, "value_win_prob": 0.20}], cfg) == []  # EV=-0.2
        assert find_value_bets([{**base, "value_win_prob": 0.30}], cfg)        # EV=+0.2


# ── config wiring ──────────────────────────────────────────────────────────--

class TestValueConfigFromConfig:
    def test_reads_value_and_bet_tracker_sections(self):
        raw = {
            "value": {"enabled": True, "min_expected_value": 0.07, "min_odds": 2.0,
                      "max_odds": 6.0, "min_confidence": 0.4, "devig": True},
            "bet_tracker": {"kelly_fraction": 0.5, "initial_bankroll": 500.0},
        }
        cfg = ValueConfig.from_config(raw)
        assert cfg.min_ev == 0.07
        assert cfg.max_odds == 6.0
        assert cfg.kelly_fraction == 0.5
        assert cfg.bankroll == 500.0

    def test_defaults_when_sections_absent(self):
        cfg = ValueConfig.from_config({})
        assert cfg.min_odds == 2.0 and cfg.max_odds == 4.0 and cfg.require_support


# ── reference-book contract (audit req 4/5/8) ────────────────────────────────

class TestReferenceBookContract:
    """The fair line comes only from a COMPLETE reference book — never a partial
    field, never the synthetic best-price overlay — and every pick carries the
    typed calculation fields + provenance."""

    def _field(self, n=4, ref=4.0, best=None):
        rows = []
        for i in range(n):
            r = {"horse_id": str(i), "horse_name": chr(65 + i),
                 "value_win_prob": 0.30, "decimal_odds": ref,
                 "value_supported": True}
            if best is not None:
                r["best_odds"] = best
            rows.append(r)
        return rows

    def test_incomplete_reference_book_passes_entire_race(self):
        """req 4 regression: a runner with no price at all leaves the book
        partial — the WHOLE race PASSes instead of being scored off a
        misleading partial de-vig (the old behaviour skipped just that runner)."""
        runners = self._field(4)
        runners[2].pop("decimal_odds")
        picks = find_value_bets({"runners": runners},
                                ValueConfig(min_confidence=0.0))
        assert picks == []

    def test_same_race_fully_priced_scores(self):
        picks = find_value_bets({"runners": self._field(4)},
                                ValueConfig(min_confidence=0.0))
        assert len(picks) == 4  # sanity: only completeness separated the cases

    def test_best_price_overlay_never_devigged(self):
        """req 5 regression: inflating every runner's best (executable) price must
        change EV only — fair_prob/overround come from the reference line and
        must not move. De-vigging the synthetic best-price board would move them."""
        cfg = ValueConfig(min_confidence=0.0, max_odds=10.0)
        base = find_value_bets({"runners": self._field(4)}, cfg)
        inflated = find_value_bets({"runners": self._field(4, best=8.0)}, cfg)
        assert base and inflated
        assert base[0]["fair_prob"] == inflated[0]["fair_prob"] == pytest.approx(0.25)
        assert base[0]["overround"] == inflated[0]["overround"]
        assert inflated[0]["executable_odds"] == pytest.approx(8.0)
        assert inflated[0]["reference_odds"] == pytest.approx(4.0)
        assert inflated[0]["expected_value"] == pytest.approx(0.30 * 8.0 - 1.0)

    def test_race_level_ev_gate_honoured_from_cache(self):
        """issue #6: a race dict stamped ev_eligible=False PASSes outright, no
        matter how permissive the value config is."""
        race = {"ev_eligible": False,
                "ev_gate": {"reasons": ["stale_price_rows:2"]},
                "runners": self._field(4)}
        cfg = ValueConfig(devig=False, require_support=False, min_confidence=0.0)
        assert find_value_bets(race, config=cfg) == []

    def test_pick_carries_typed_fields_and_provenance(self):
        """req 8: separate typed fields for every quantity on the prob→EV path,
        plus source/age/timestamp diagnostics (req 6)."""
        runners = self._field(4, best=4.2)
        for r in runners:
            r["best_book"] = "paddy_power"
            r["reference_source"] = "livescorebet"
            r["empirical_win_rate"] = 0.18
        picks = find_value_bets({"runners": runners},
                                ValueConfig(min_confidence=0.0, max_odds=6.0))
        assert picks
        pk = picks[0]
        for key in ("model_win_prob", "empirical_win_rate", "raw_implied_prob",
                    "devigged_market_prob", "fair_decimal_odds", "reference_odds",
                    "executable_odds", "edge_pp", "relative_edge", "EV",
                    "expected_value", "computed_at", "executable_source",
                    "reference_source", "price_age_seconds"):
            assert key in pk, f"pick missing typed field {key}"
        assert pk["executable_source"] == "paddy_power"
        assert pk["reference_source"] == "livescorebet"
        assert pk["empirical_win_rate"] == pytest.approx(0.18)
        assert pk["EV"] == pk["expected_value"]
        assert pk["price_age_seconds"] is None  # no fetched_at supplied → honest None
