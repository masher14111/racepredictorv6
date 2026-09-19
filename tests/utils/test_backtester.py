"""Tests for utils/backtester.py."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from utils.backtester import (
    BacktestConfig,
    BacktestResult,
    Backtester,
    _classify_odds_band,
    _gross_return,
    _kelly_stake,
    _settle_outcome,
    run_backtest,
)


# ── fixtures ─────────────────────────────────────────────────────────────────────

def _minimal_df(n_races: int = 4, runners_per_race: int = 6) -> pd.DataFrame:
    """Synthetic labelled DataFrame with two venues and known positions."""
    rows = []
    for i in range(n_races):
        venue = "Leopardstown" if i % 2 == 0 else "Curragh"
        race_date = pd.Timestamp(f"2024-01-{1 + i:02d}", tz="UTC")
        for j in range(runners_per_race):
            position = j + 1
            implied_prob = 1.0 / (3.0 + j * 2.0)
            rows.append({
                "race_date": race_date,
                "venue": venue,
                "horse_name": f"Horse_{i}_{j}",
                "horse_id": f"h{i}{j}",
                "jockey": "J. Murphy",
                "trainer": "A. O'Brien",
                "position": position,
                "implied_prob": implied_prob,
                "decimal_odds": round(1.0 / implied_prob, 3),
                # feature columns (all zero — backtester doesn't need real values without models)
                "historical_win_rate": 0.1,
                "timeform_rating": 100,
            })
    return pd.DataFrame(rows)


def _cfg(**kwargs) -> BacktestConfig:
    defaults = dict(initial_bankroll=1000.0, flat_stake=10.0, train_split_pct=0.50)
    defaults.update(kwargs)
    return BacktestConfig(**defaults)


# ── _settle_outcome ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("position,bet_type,ew_places,expected", [
    (1, "win", 3, "win"),
    (2, "win", 3, "lose"),
    (3, "win", 3, "lose"),
    (1, "each_way", 3, "win"),
    (2, "each_way", 3, "place"),
    (3, "each_way", 3, "place"),
    (4, "each_way", 3, "lose"),
    (2, "each_way", 2, "place"),
    (3, "each_way", 2, "lose"),
])
def test_settle_outcome(position, bet_type, ew_places, expected):
    assert _settle_outcome(position, bet_type, ew_places) == expected


# ── _gross_return ────────────────────────────────────────────────────────────────

class TestGrossReturn:
    def test_win_bet_win(self):
        assert _gross_return("win", 10.0, 5.0, "win", 0.20) == pytest.approx(50.0)

    def test_win_bet_lose(self):
        assert _gross_return("lose", 10.0, 5.0, "win", 0.20) == 0.0

    def test_win_bet_place_is_loss(self):
        # place outcome on a win bet returns nothing
        assert _gross_return("place", 10.0, 5.0, "win", 0.20) == 0.0

    def test_ew_win_returns_both_legs(self):
        # stake=10 → each leg=5; decimal_odds=5.0, ew_fraction=0.20
        # win leg: 5 × 5.0 = 25; place leg: 5 × ((5-1)×0.20 + 1) = 5 × 1.80 = 9
        assert _gross_return("win", 10.0, 5.0, "each_way", 0.20) == pytest.approx(34.0)

    def test_ew_place_returns_place_leg_only(self):
        # place leg: 5 × 1.80 = 9
        assert _gross_return("place", 10.0, 5.0, "each_way", 0.20) == pytest.approx(9.0)

    def test_ew_lose_returns_nothing(self):
        assert _gross_return("lose", 10.0, 5.0, "each_way", 0.20) == 0.0


# ── _kelly_stake ─────────────────────────────────────────────────────────────────

class TestKellyStake:
    def test_no_edge_returns_zero(self):
        # implied_prob = 1/5 = 0.20; if model says win_prob = 0.10 → no edge
        assert _kelly_stake(5.0, 0.10, 1000.0, 0.25) == 0.0

    def test_positive_edge(self):
        # win_prob=0.25, odds=5.0 (b=4); f = (4×0.25 - 0.75)/4 = 0.0625
        stake = _kelly_stake(5.0, 0.25, 1000.0, 0.25)
        assert stake == pytest.approx(0.0625 * 1000.0 * 0.25, rel=1e-3)

    def test_zero_win_prob_returns_zero(self):
        assert _kelly_stake(5.0, 0.0, 1000.0, 0.25) == 0.0

    def test_full_certainty_returns_zero(self):
        assert _kelly_stake(5.0, 1.0, 1000.0, 0.25) == 0.0

    def test_odds_at_or_below_one_returns_zero(self):
        assert _kelly_stake(1.0, 0.50, 1000.0, 0.25) == 0.0


# ── _classify_odds_band ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("odds,expected", [
    (2.0,  "Favourite (≤4.0)"),
    (4.0,  "Favourite (≤4.0)"),
    (4.1,  "Mid (4.1–10.0)"),
    (10.0, "Mid (4.1–10.0)"),
    (10.1, "Longshot (>10.0)"),
    (33.0, "Longshot (>10.0)"),
])
def test_classify_odds_band(odds, expected):
    assert _classify_odds_band(odds) == expected


# ── BacktestConfig defaults ──────────────────────────────────────────────────────

def test_backtest_config_defaults():
    cfg = BacktestConfig()
    assert cfg.strategy == "flat"
    assert cfg.flat_stake == 10.0
    assert cfg.initial_bankroll == 1000.0
    assert cfg.top_n == 3
    assert cfg.bet_type == "win"
    assert cfg.train_split_pct == 0.80


# ── Backtester.run() ─────────────────────────────────────────────────────────────

class TestBacktesterRun:
    def test_empty_df_returns_zero_bets(self):
        cfg = _cfg()
        bt = Backtester(cfg)
        result = bt.run(pd.DataFrame())
        assert result.model_summary["total_bets"] == 0
        assert result.market_summary["total_bets"] == 0
        assert result.model_bets.empty
        assert result.market_bets.empty

    def test_flat_strategy_produces_bets(self):
        cfg = _cfg(strategy="flat", top_n=1)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        # test window is last 50% → 2 races × 1 selection = at most 2 bets
        assert result.market_summary["total_bets"] > 0

    def test_all_stakes_equal_flat_stake(self):
        cfg = _cfg(strategy="flat", flat_stake=15.0, top_n=2)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        if not result.market_bets.empty:
            assert (result.market_bets["stake"] == 15.0).all()

    def test_win_rate_between_zero_and_one(self):
        cfg = _cfg(top_n=3)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=6))
        ms = result.market_summary
        assert 0.0 <= ms["win_rate"] <= 100.0

    def test_roi_finite(self):
        cfg = _cfg(top_n=2)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=6))
        assert np.isfinite(result.market_summary["roi_pct"])

    def test_bankroll_sequence_consistent(self):
        cfg = _cfg(strategy="flat", flat_stake=10.0, top_n=2)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=6))
        bets = result.market_bets
        if len(bets) >= 2:
            # each row's bankroll = previous bankroll - stake + gross_return
            for idx in range(1, len(bets)):
                prev = bets.iloc[idx - 1]
                curr = bets.iloc[idx]
                expected = prev["bankroll"] - curr["stake"] + curr["gross_return"]
                assert curr["bankroll"] == pytest.approx(expected, abs=0.01)

    def test_cumulative_profit_monotone_tracking(self):
        cfg = _cfg(top_n=2)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=6))
        bets = result.market_bets
        if not bets.empty:
            assert "cumulative_profit" in bets.columns
            assert bets["cumulative_profit"].iloc[-1] == pytest.approx(
                bets["profit"].sum(), abs=0.01
            )

    def test_min_selection_odds_filter(self):
        cfg = _cfg(min_selection_odds=999.0, top_n=3)
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        # all runners have odds < 999 → no bets
        assert result.market_summary["total_bets"] == 0

    def test_stop_loss_halts_simulation(self):
        # Build a DataFrame where the top-ranked runner (by implied_prob) never wins.
        # The market simulation bets top-1 per race; every bet loses.
        # With stake=40, bankroll=100, floor=90 (10% stop-loss):
        # after the first loss bankroll=60 < 90 → stop-loss fires before race 2.
        rows = []
        for i in range(20):
            race_date = pd.Timestamp(f"2024-01-{i + 1:02d}", tz="UTC")
            for j in range(6):
                implied_prob = 1.0 / (3.0 + j * 2.0)
                rows.append({
                    "race_date": race_date,
                    "venue": "Leopardstown",
                    "horse_name": f"Horse_{i}_{j}",
                    "horse_id": f"h{i}{j}",
                    "jockey": "J. Murphy",
                    "trainer": "A. O'Brien",
                    # j=0 has highest implied_prob but position=6 (loses every time)
                    "position": 6 - j,
                    "implied_prob": implied_prob,
                    "decimal_odds": round(1.0 / implied_prob, 3),
                    "historical_win_rate": 0.1,
                    "timeform_rating": 100,
                })
        df = pd.DataFrame(rows)

        cfg = _cfg(
            initial_bankroll=100.0,
            flat_stake=40.0,
            stop_loss_pct=0.10,
            top_n=1,
            train_split_pct=0.0,
        )
        bt = Backtester(cfg)
        result = bt.run(df)
        # stop-loss fires after first loss — should have far fewer than 20 bets
        assert result.market_summary["total_bets"] < 20

    def test_metadata_keys_present(self):
        cfg = _cfg()
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        for key in ("has_models", "total_runners", "total_races", "date_range",
                    "train_rows", "test_rows"):
            assert key in result.metadata

    def test_has_models_false_without_bins(self, tmp_path):
        # Point at an isolated empty dir so the assertion holds regardless of
        # whether real trained .bin files exist in the project's models/ folder.
        cfg = _cfg(model_dir=str(tmp_path / "no_models"))
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        assert result.metadata["has_models"] is False


# ── summary metrics ──────────────────────────────────────────────────────────────

class TestComputeSummary:
    def _make_bets(self) -> pd.DataFrame:
        # 3 bets: win, lose, lose  stake=10, odds=5.0
        return pd.DataFrame([
            {"outcome": "win",  "stake": 10.0, "profit": 40.0, "bankroll": 1040.0},
            {"outcome": "lose", "stake": 10.0, "profit": -10.0, "bankroll": 1030.0},
            {"outcome": "lose", "stake": 10.0, "profit": -10.0, "bankroll": 1020.0},
        ])

    def test_win_rate(self):
        bets = self._make_bets()
        cfg = BacktestConfig()
        s = Backtester._compute_summary(bets, cfg)
        assert s["win_rate"] == pytest.approx(100 / 3, rel=1e-3)

    def test_roi(self):
        bets = self._make_bets()
        cfg = BacktestConfig()
        s = Backtester._compute_summary(bets, cfg)
        # profit=20, staked=30 → roi=66.67%
        assert s["roi_pct"] == pytest.approx(20 / 30 * 100, rel=1e-3)

    def test_max_drawdown(self):
        bets = self._make_bets()
        cfg = BacktestConfig()
        s = Backtester._compute_summary(bets, cfg)
        # peak=1040, low=1020 → dd = (1040-1020)/1040 ≈ 1.92%
        assert s["max_drawdown_pct"] == pytest.approx((1040 - 1020) / 1040 * 100, rel=1e-2)

    def test_empty_bets_returns_zero_totals(self):
        cfg = BacktestConfig(initial_bankroll=500.0)
        s = Backtester._compute_summary(pd.DataFrame(), cfg)
        assert s["total_bets"] == 0
        assert s["final_bankroll"] == 500.0


# ── breakdown helpers ────────────────────────────────────────────────────────────

def _sample_bets() -> pd.DataFrame:
    return pd.DataFrame([
        {"race_date": "2024-01-10", "venue": "Leopardstown", "outcome": "win",
         "decimal_odds": 3.0, "stake": 10.0, "profit": 20.0, "bankroll": 1020.0},
        {"race_date": "2024-01-10", "venue": "Curragh",      "outcome": "lose",
         "decimal_odds": 6.0, "stake": 10.0, "profit": -10.0, "bankroll": 1010.0},
        {"race_date": "2024-02-05", "venue": "Leopardstown", "outcome": "lose",
         "decimal_odds": 12.0, "stake": 10.0, "profit": -10.0, "bankroll": 1000.0},
    ])


def test_monthly_breakdown_groups_by_month():
    bets = _sample_bets()
    g = Backtester._monthly_breakdown(bets)
    assert len(g) == 2
    assert set(g["month"]) == {"2024-01", "2024-02"}


def test_odds_band_breakdown_classifies_correctly():
    bets = _sample_bets()
    g = Backtester._odds_band_breakdown(bets)
    assert "Favourite (≤4.0)" in g["odds_band"].values   # odds=3.0
    assert "Mid (4.1–10.0)" in g["odds_band"].values     # odds=6.0
    assert "Longshot (>10.0)" in g["odds_band"].values    # odds=12.0


def test_venue_breakdown_sorts_by_roi():
    bets = _sample_bets()
    g = Backtester._venue_breakdown(bets)
    assert g.iloc[0]["venue"] == "Leopardstown"  # net profit = 20-10=10 → higher ROI


# ── Backtester.report() ──────────────────────────────────────────────────────────

class TestBacktesterReport:
    def test_report_writes_file(self, tmp_path):
        cfg = _cfg(output_path=str(tmp_path / "test_report.md"))
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        md = bt.report(result, output_path=str(tmp_path / "test_report.md"))
        out = tmp_path / "test_report.md"
        assert out.exists()
        assert "# Backtest Report" in md

    def test_report_contains_key_sections(self, tmp_path):
        cfg = _cfg(output_path=str(tmp_path / "r.md"))
        bt = Backtester(cfg)
        result = bt.run(_minimal_df(n_races=4))
        md = bt.report(result, output_path=str(tmp_path / "r.md"))
        assert "## Configuration" in md
        assert "## Overall Performance" in md
        assert "## Notes" in md

    def test_report_empty_result_no_crash(self, tmp_path):
        cfg = _cfg(output_path=str(tmp_path / "r.md"))
        bt = Backtester(cfg)
        empty = BacktestResult(
            config=cfg,
            model_bets=pd.DataFrame(),
            market_bets=pd.DataFrame(),
            model_summary=Backtester._compute_summary(pd.DataFrame(), cfg),
            market_summary=Backtester._compute_summary(pd.DataFrame(), cfg),
            metadata={"has_models": False, "total_races": 0, "total_runners": 0,
                      "date_range": "N/A", "train_rows": 0, "test_rows": 0},
        )
        md = bt.report(empty, output_path=str(tmp_path / "r.md"))
        assert "# Backtest Report" in md
        assert "0" in md  # bets count


# ── run_backtest convenience ──────────────────────────────────────────────────────

def test_run_backtest_returns_result(tmp_path):
    cfg = _cfg(output_path=str(tmp_path / "bt.md"))
    result = run_backtest(cfg, df=_minimal_df(n_races=4))
    assert isinstance(result, BacktestResult)
    assert (tmp_path / "bt.md").exists()


# ── price resolution / CLV calc ────────────────────────────────────────────────────

class TestResolvePricesAndCLV:
    def test_bet_and_settle_resolve_from_distinct_columns(self):
        # board price = odds_decimal; closing/SP price = sp → both genuine, CLV measurable
        df = pd.DataFrame([
            {"odds_decimal": 6.0, "sp": 5.0, "implied_prob": 1 / 6, "position": 1},
            {"odds_decimal": 4.0, "sp": 5.0, "implied_prob": 1 / 4, "position": 2},
        ])
        out = Backtester._resolve_prices(df)
        assert out["bet_odds"].tolist() == [6.0, 4.0]
        assert out["settle_odds"].tolist() == [5.0, 5.0]
        assert out["has_clv"].all()

    def test_settles_at_sp_even_without_board_price_but_no_clv(self):
        # No board price; bet price derived from implied_prob (4.0), settle at sp (5.0).
        # Settlement uses SP, but CLV is NOT measurable (bet price isn't a real board quote).
        df = pd.DataFrame([{"sp": 5.0, "implied_prob": 1 / 4, "position": 1}])
        out = Backtester._resolve_prices(df)
        assert out["bet_odds"].iloc[0] == pytest.approx(4.0)
        assert out["settle_odds"].iloc[0] == 5.0
        assert not bool(out["has_clv"].iloc[0])

    def test_settle_falls_back_to_bet_when_no_sp(self):
        # Only a board price exists → settle collapses to bet price, CLV unavailable.
        df = pd.DataFrame([{"odds_decimal": 7.0, "implied_prob": 1 / 7, "position": 1}])
        out = Backtester._resolve_prices(df)
        assert out["bet_odds"].iloc[0] == 7.0
        assert out["settle_odds"].iloc[0] == 7.0
        assert not bool(out["has_clv"].iloc[0])

    def test_clv_value_matches_bet_over_settle(self):
        # CLV in the simulated bets = bet_odds / settle_odds - 1 (only where has_clv).
        bets = pd.DataFrame([
            {"outcome": "win",  "stake": 10.0, "profit": 40.0, "bankroll": 1040.0,
             "clv": 6.0 / 5.0 - 1.0, "has_clv": True},
            {"outcome": "lose", "stake": 10.0, "profit": -10.0, "bankroll": 1030.0,
             "clv": 4.0 / 5.0 - 1.0, "has_clv": True},
            {"outcome": "lose", "stake": 10.0, "profit": -10.0, "bankroll": 1020.0,
             "clv": np.nan, "has_clv": False},
        ])
        s = Backtester._compute_summary(bets, BacktestConfig())
        assert s["n_clv"] == 2                                  # nan/has_clv=False excluded
        assert s["clv_mean_pct"] == pytest.approx(0.0)          # mean(+0.20, -0.20) = 0
        assert s["clv_positive_rate"] == pytest.approx(50.0)    # one of two beat the close

    def test_summary_clv_keys_default_when_absent(self):
        bets = pd.DataFrame([
            {"outcome": "win", "stake": 10.0, "profit": 40.0, "bankroll": 1040.0},
        ])
        s = Backtester._compute_summary(bets, BacktestConfig())
        assert s["n_clv"] == 0 and s["clv_mean_pct"] == 0.0 and s["clv_positive_rate"] == 0.0

    def test_settlement_uses_sp_not_board_in_full_run(self):
        # One winning favourite per race, board=6.0 but SP=3.0. A €10 win must return
        # 10×3.0=30 (settled at SP), NOT 10×6.0=60 (the board price used to pick it).
        rows = []
        for i in range(4):
            race_date = pd.Timestamp(f"2024-01-{1 + i:02d}", tz="UTC")
            for j in range(4):
                rows.append({
                    "race_date": race_date, "venue": "Leopardstown",
                    "horse_name": f"H{i}{j}", "horse_id": f"h{i}{j}",
                    "position": j + 1,                       # j=0 wins
                    "implied_prob": 1.0 / (3.0 + j * 2.0),   # j=0 is the favourite (top score)
                    "odds_decimal": 6.0,                     # board price we'd take
                    "sp": 3.0,                               # executable settlement price
                })
        df = pd.DataFrame(rows)
        cfg = _cfg(strategy="flat", flat_stake=10.0, top_n=1, train_split_pct=0.0,
                   min_selection_odds=2.0)
        result = Backtester(cfg).run(df)
        bets = result.market_bets
        won = bets[bets["outcome"] == "win"]
        assert not won.empty
        n = len(won)
        assert won["gross_return"].tolist() == pytest.approx([30.0] * n)   # 10 × SP(3.0)
        assert (won["settle_odds"] == 3.0).all()
        assert (won["bet_odds"] == 6.0).all()
        # CLV = 6.0/3.0 - 1 = +1.0 (we beat the close by a mile)
        assert won["clv"].tolist() == pytest.approx([1.0] * n)


# ── per-band bootstrapped CI ───────────────────────────────────────────────────────

class TestBandCI:
    def _bets(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"decimal_odds": 3.0, "stake": 10.0, "profit": 20.0},
            {"decimal_odds": 3.5, "stake": 10.0, "profit": -10.0},
            {"decimal_odds": 3.2, "stake": 10.0, "profit": -10.0},
            {"decimal_odds": 12.0, "stake": 10.0, "profit": -10.0},
            {"decimal_odds": 15.0, "stake": 10.0, "profit": 110.0},
        ])

    def test_bands_and_point_roi(self):
        g = Backtester._band_ci(self._bets(), n_boot=500, seed=1)
        assert set(g["odds_band"]) >= {"Favourite (≤4.0)", "Longshot (>10.0)"}
        fav = g[g["odds_band"] == "Favourite (≤4.0)"].iloc[0]
        # favourite band: profit (20-10-10)=0 / staked 30 → 0.00%
        assert fav["roi_pct"] == pytest.approx(0.0, abs=1e-6)
        assert fav["bets"] == 3

    def test_ci_brackets_median(self):
        g = Backtester._band_ci(self._bets(), n_boot=500, seed=1)
        assert g["ci_low"].le(g["roi_median"]).all()
        assert g["roi_median"].le(g["ci_high"]).all()

    def test_deterministic_with_seed(self):
        a = Backtester._band_ci(self._bets(), n_boot=500, seed=7)
        b = Backtester._band_ci(self._bets(), n_boot=500, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_empty_returns_empty(self):
        assert Backtester._band_ci(pd.DataFrame()).empty


# ── longshot / robustness gate ─────────────────────────────────────────────────────

class TestRobustnessGate:
    def test_flags_longshot_dominated_edge(self):
        # 10 small losers + 1 huge longshot winner → positive ROI driven by one bet.
        bets = pd.DataFrame(
            [{"decimal_odds": 3.0, "stake": 10.0, "profit": -10.0} for _ in range(10)]
            + [{"decimal_odds": 51.0, "stake": 10.0, "profit": 500.0}]
        )
        gate = Backtester._robustness_gate(bets, top_k=3)
        assert gate["roi_full"] > 0
        assert gate["flagged"] is True
        assert gate["n_longshot_winners"] == 1
        assert gate["roi_ex_top_k"] <= 0

    def test_not_flagged_when_edge_is_broad(self):
        bets = pd.DataFrame([
            {"decimal_odds": 3.0, "stake": 10.0, "profit": 5.0} for _ in range(30)
        ])
        gate = Backtester._robustness_gate(bets, top_k=3)
        assert gate["roi_full"] > 0
        assert gate["flagged"] is False

    def test_empty_is_unflagged(self):
        gate = Backtester._robustness_gate(pd.DataFrame())
        assert gate["flagged"] is False
        assert gate["n_bets"] == 0


# ── report still renders the new sections ──────────────────────────────────────────

def test_report_includes_new_sections(tmp_path):
    # df with a genuine board price (odds_decimal) and SP (sp) → CLV + bands populated.
    rows = []
    for i in range(6):
        race_date = pd.Timestamp(f"2024-01-{1 + i:02d}", tz="UTC")
        for j in range(6):
            rows.append({
                "race_date": race_date,
                "venue": "Leopardstown" if i % 2 == 0 else "Curragh",
                "horse_name": f"H{i}{j}", "horse_id": f"h{i}{j}",
                "position": j + 1,
                "implied_prob": 1.0 / (3.0 + j * 2.0),
                "odds_decimal": round(3.0 + j * 2.0, 2),
                "sp": round((3.0 + j * 2.0) * 0.9, 2),     # SP slightly shorter than board
            })
    df = pd.DataFrame(rows)
    cfg = _cfg(output_path=str(tmp_path / "r.md"), top_n=2, train_split_pct=0.0,
               min_selection_odds=2.0)
    bt = Backtester(cfg)
    result = bt.run(df)
    md = bt.report(result, output_path=str(tmp_path / "r.md"))
    assert "Edge Robustness (Longshot Gate)" in md
    assert "bootstrapped" in md
    assert "CLV" in md
    assert result.metadata["clv_available"] is True
