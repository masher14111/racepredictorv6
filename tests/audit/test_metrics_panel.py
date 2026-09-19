"""Metric primitives for the Stage-4 audit: race-level scoring, bootstrap
head-to-head, A/E tables, CLV."""
import numpy as np
import pandas as pd
import pytest

from audit.metrics_panel import (
    ae_table,
    clv_stats,
    ece_equal_freq,
    head_to_head_ci,
    market_line,
    odds_band_series,
    per_race_logloss,
    wilson_interval,
)


def _frame():
    """Two races, complete books, known winners."""
    return pd.DataFrame({
        "race_uid": ["r1"] * 3 + ["r2"] * 3,
        "won": [1, 0, 0, 0, 1, 0],
        "bet_price": [2.0, 4.0, 4.0, 3.0, 3.0, 3.0],
        "close_price": [1.8, 4.5, 4.5, 3.2, 2.8, 3.1],
        "prob": [0.5, 0.25, 0.25, 1 / 3, 1 / 3, 1 / 3],
    })


class TestPrimitives:
    def test_per_race_logloss_is_winner_nll(self):
        ll = per_race_logloss(_frame(), "prob")
        assert ll.loc["r1"] == pytest.approx(-np.log(0.5))
        assert ll.loc["r2"] == pytest.approx(-np.log(1 / 3))

    def test_market_line_devigs_complete_books(self):
        m = market_line(_frame())
        # r1 booksum 1/2+1/4+1/4 = 1 → probs unchanged by proportional de-vig
        assert m[0] == pytest.approx(0.5)
        assert m[1] == pytest.approx(0.25)

    def test_market_line_nan_on_partial_book(self):
        df = _frame()
        df.loc[2, "bet_price"] = np.nan
        m = market_line(df)
        assert np.isnan(m[:3]).all()
        assert np.isfinite(m[3:]).all()

    def test_wilson_interval_contains_phat_and_bounds(self):
        lo, hi = wilson_interval(30, 100)
        assert lo < 0.30 < hi
        assert 0.0 <= lo and hi <= 1.0
        assert wilson_interval(0, 0) == (pytest.approx(float("nan"), nan_ok=True),) * 2 \
            or np.isnan(wilson_interval(0, 0)).all()

    def test_ece_small_for_well_calibrated_probs(self):
        rng = np.random.default_rng(5)
        p = np.where(rng.uniform(size=6000) < 0.5, 0.2, 0.4)
        y = (rng.uniform(size=6000) < p).astype(float)
        assert ece_equal_freq(p, y) < 0.03

    def test_ece_nan_for_constant_predictions(self):
        # A constant vector defines no bins — NaN by design, never a crash.
        assert np.isnan(ece_equal_freq(np.full(1000, 0.3),
                                       np.zeros(1000)))

    def test_clv_stats_signs(self):
        s = clv_stats(pd.Series([2.0, 3.0]), pd.Series([1.8, 3.3]))
        assert s["n"] == 2
        # log(2/1.8) > 0, log(3/3.3) < 0 → beat rate 0.5
        assert s["beat_close_rate"] == pytest.approx(0.5)


class TestHeadToHeadCI:
    def test_model_equal_to_market_gives_zero_delta(self):
        df = _frame()
        df["prob"] = market_line(df)
        out = head_to_head_ci(df, "prob", n_boot=200, seed=1)
        assert out["logloss_delta_market_minus_model"] == pytest.approx(0.0, abs=1e-12)
        lo, hi = out["logloss_delta_ci95"]
        assert lo <= 0.0 <= hi

    def test_sharper_model_beats_market_with_positive_delta(self):
        rng = np.random.default_rng(9)
        rows = []
        for r in range(400):
            true_p = rng.dirichlet([2.0, 1.0, 1.0])
            odds = 1.0 / np.maximum(true_p * 0.8 + 0.4 / 3, 1e-3)  # blunted market
            w = rng.choice(3, p=true_p)
            for i in range(3):
                rows.append({"race_uid": f"r{r}", "won": int(i == w),
                             "bet_price": float(odds[i]),
                             "close_price": float(odds[i]),
                             "prob": float(true_p[i])})
        df = pd.DataFrame(rows)
        out = head_to_head_ci(df, "prob", n_boot=300, seed=2)
        assert out["model_beats_market_logloss"]
        assert out["logloss_delta_market_minus_model"] > 0
        assert out["logloss_delta_ci95"][0] > 0  # significantly better

    def test_output_carries_ci_and_bootfrac_keys(self):
        out = head_to_head_ci(_frame(), "prob", n_boot=50, seed=3)
        for key in ("model_log_loss_ci95", "logloss_delta_ci95",
                    "logloss_delta_frac_boot_positive", "brier_delta_ci95",
                    "model_ece", "market_ece", "n_races", "n_runners"):
            assert key in out


class TestAETable:
    def test_ae_and_ci_scale(self):
        rng = np.random.default_rng(3)
        n = 4000
        p = np.full(n, 0.25)
        y = (rng.uniform(size=n) < 0.25).astype(int)
        df = pd.DataFrame({"prob": p, "won": y})
        t = ae_table(df, "prob", pd.Series(["all"] * n), min_rows=100)
        row = t.iloc[0]
        assert row["n"] == n
        assert row["ae"] == pytest.approx(y.mean() / 0.25, rel=1e-6)
        lo, hi = row["ae_ci95"]
        assert lo < row["ae"] < hi

    def test_small_groups_dropped(self):
        df = pd.DataFrame({"prob": [0.5] * 10, "won": [1] * 10})
        t = ae_table(df, "prob", pd.Series(["tiny"] * 10), min_rows=50)
        assert t.empty

    def test_odds_band_series_labels(self):
        bands = odds_band_series(pd.Series([1.5, 2.5, 100.0]))
        assert bands.iloc[0].startswith("<2.0")
        assert bands.iloc[1] == "2.0-3.0"
        assert bands.iloc[2] == "51.0+"
