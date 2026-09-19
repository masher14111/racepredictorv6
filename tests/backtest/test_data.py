"""Tests for backtest.data — building the leak-free bet panel.

The two load-bearing guarantees here are (1) the panel bets at a *pre-off*
execution price, never the leaky ``implied_prob``/``odds_finish`` finishing
price, and (2) the WIN market filter so there is exactly one win candidate per
runner.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.data import PanelConfig, load_panel


def _raw(**overrides):
    """A small training-matrix-shaped frame; override columns as needed."""
    base = pd.DataFrame({
        "race_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"], utc=True),
        "venue": ["Ascot", "Ascot", "York"],
        "horse_id": [1, 2, 3],
        "horse_name": ["A", "B", "C"],
        "market_type": ["WIN", "WIN", "WIN"],
        "won": [1, 0, 0],
        "ppwap": [4.0, 6.0, 3.0],          # pre-off execution price
        "morningwap": [4.2, 6.5, 3.1],
        "odds_finish": [3.5, 7.0, 2.8],    # BSP (closing line)
        "implied_prob": [1 / 3.5, 1 / 7.0, 1 / 2.8],  # leaky: == 1/odds_finish
    })
    for k, v in overrides.items():
        base[k] = v
    return base


class TestWinFilter:
    def test_keeps_only_win_market(self):
        df = _raw()
        place = df.copy()
        place["market_type"] = "PLACE"
        place["ppwap"] = place["ppwap"] / 2  # place-market prices differ
        both = pd.concat([df, place], ignore_index=True)
        panel = load_panel(df=both)
        assert len(panel) == 3  # the 3 PLACE duplicates are dropped

    def test_one_row_per_runner(self):
        panel = load_panel(df=_raw())
        keys = panel[["race_date", "venue", "horse_id"]]
        assert not keys.duplicated().any()


class TestExecutionPrice:
    def test_bet_price_is_preoff_not_bsp(self):
        # bet_price must come from ppwap, never odds_finish (the closing line).
        panel = load_panel(df=_raw())
        assert panel["bet_price"].tolist() == pytest.approx([4.0, 6.0, 3.0])

    def test_close_price_is_bsp(self):
        panel = load_panel(df=_raw())
        assert panel["close_price"].tolist() == pytest.approx([3.5, 7.0, 2.8])

    def test_falls_back_to_morningwap_when_primary_missing(self):
        df = _raw()
        df.loc[0, "ppwap"] = np.nan       # missing primary -> use morningwap 4.2
        df.loc[1, "ppwap"] = 1.0          # sentinel invalid -> use morningwap 6.5
        panel = load_panel(df=df).sort_values("horse_id").reset_index(drop=True)
        prices = dict(zip(panel["horse_id"], panel["bet_price"]))
        assert prices[1] == pytest.approx(4.2)
        assert prices[2] == pytest.approx(6.5)

    def test_row_dropped_when_no_usable_price(self):
        df = _raw()
        df.loc[0, ["ppwap", "morningwap"]] = [1.0, np.nan]  # no valid exec price
        panel = load_panel(df=df)
        assert 1 not in panel["horse_id"].tolist()


class TestNoLeakyImpliedProb:
    def test_implied_prob_column_not_carried_into_panel(self):
        # The training matrix's implied_prob is 1/odds_finish (a finishing price).
        # It must never reach the betting panel as a usable signal.
        panel = load_panel(df=_raw())
        assert "implied_prob" not in panel.columns

    def test_panel_implied_would_differ_from_leaky_column(self):
        # Sanity: an implied prob computed from the *execution* price is NOT the
        # leaky 1/odds_finish, proving we bet off a different (legitimate) price.
        from backtest import metrics
        panel = load_panel(df=_raw())
        exec_implied = metrics.implied_prob(panel["bet_price"].to_numpy())
        leaky = 1.0 / _raw()["odds_finish"].to_numpy()
        assert not np.allclose(np.sort(exec_implied), np.sort(leaky))


class TestOutcomeAndSorting:
    def test_won_derived_from_position_when_absent(self):
        df = _raw().drop(columns=["won"])
        df["position"] = [1, 4, 2]
        panel = load_panel(df=df).sort_values("horse_id").reset_index(drop=True)
        assert dict(zip(panel["horse_id"], panel["won"])) == {1: 1, 2: 0, 3: 0}

    def test_sorted_by_race_date(self):
        df = _raw()
        df["race_date"] = pd.to_datetime(["2024-03-01", "2024-01-01", "2024-02-01"], utc=True)
        panel = load_panel(df=df)
        assert panel["race_date"].is_monotonic_increasing

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_panel(path="does/not/exist.parquet")

    def test_absurd_price_treated_as_missing(self):
        df = _raw()
        df.loc[0, ["ppwap", "morningwap"]] = [5000.0, np.nan]  # above max_price
        panel = load_panel(df=df, config=PanelConfig(max_price=1000.0))
        assert 1 not in panel["horse_id"].tolist()
