"""Tests for the performance dashboard data + figure layer (``ui.dashboard``).

The Streamlit assembly (``ui/pages/10_Dashboard.py``) is not imported here — only
the pure builders, which is the whole point of keeping them Streamlit-free.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import pytest

from ui import dashboard as D


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def paper_df() -> pd.DataFrame:
    """A small, typed paper-bet frame shaped like ``load_paper_bets`` output."""
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "placed_at": pd.to_datetime(
            ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"],
            utc=True),
        "settled_at": pd.to_datetime(
            ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", None],
            utc=True),
        "horse_name": ["A", "B", "C", "D", "E"],
        "venue": ["Ascot", "Cork", "Ascot", "Cork", "Ascot"],
        "bet_type": ["win", "win", "each_way", "win", "win"],
        "stake": [10.0, 10.0, 10.0, 10.0, 10.0],
        "odds_decimal": [3.0, 12.0, 6.0, 1.8, 25.0],
        "closing_odds": [2.8, 14.0, 6.0, 1.9, 22.0],
        "clv_pct": [7.1, -14.3, 0.0, -5.3, 13.6],
        "won_prob": [0.33, 0.08, 0.16, 0.55, 0.04],
        "value_edge": [0.05, -0.01, 0.02, -0.02, 0.01],
        "outcome": ["win", "lose", "place", "lose", None],  # last is pending
        "profit": [20.0, -10.0, -4.0, -10.0, None],
    })
    df["placed_date"] = df["placed_at"].dt.date
    df["is_pick"] = df["value_edge"].fillna(0) > 0
    return df


@pytest.fixture
def bt_summary() -> dict:
    return {
        "model_metrics": {"auc": 0.668, "ece": 0.0015, "base_rate": 0.116, "n": 77323},
        "ae_by_prob": [
            {"bucket": "0.02-0.05", "n": 8133, "expected": 318.8, "actual": 305.0,
             "mean_pred": 0.039, "ae": 0.957},
            {"bucket": "0.05-0.10", "n": 5000, "expected": 350.0, "actual": 360.0,
             "mean_pred": 0.07, "ae": 1.03},
        ],
        "strategies": {
            "ev_flat": {"strategy": "ev_flat", "n_bets": 2296, "yield_pct": -4.36,
                        "hit_rate": 0.09, "clv_pct_mean": -11.99, "beat_close_rate": 0.237,
                        "max_drawdown_pct": 100.0, "bankroll_growth_pct": -100.0,
                        "avg_odds": 13.8},
            "edge10_flat": {"strategy": "edge10_flat", "n_bets": 500, "yield_pct": -2.1,
                            "hit_rate": 0.12, "clv_pct_mean": -9.0, "beat_close_rate": 0.27,
                            "max_drawdown_pct": 80.0, "bankroll_growth_pct": -90.0,
                            "avg_odds": 8.0},
        },
    }


# ── filtering ─────────────────────────────────────────────────────────────────

def test_apply_filters_market(paper_df):
    out = D.apply_filters(paper_df, markets=["each_way"])
    assert set(out["bet_type"]) == {"each_way"}
    assert len(out) == 1


def test_apply_filters_picks_only(paper_df):
    out = D.apply_filters(paper_df, picks_only=True)
    assert out["is_pick"].all()
    assert len(out) == 3  # value_edge > 0 rows


def test_apply_filters_odds_band(paper_df):
    out = D.apply_filters(paper_df, odds_bands=["Longshot (16+)"])
    assert (out["odds_decimal"] >= 16.0).all()
    assert len(out) == 1


def test_apply_filters_date_range(paper_df):
    import datetime as dt
    out = D.apply_filters(
        paper_df, date_range=(dt.date(2026, 6, 2), dt.date(2026, 6, 3)))
    assert len(out) == 2


def test_apply_filters_all_bands_is_noop(paper_df):
    out = D.apply_filters(paper_df, odds_bands=[b[0] for b in D.ODDS_BANDS])
    assert len(out) == len(paper_df)


def test_apply_filters_empty_frame():
    assert D.apply_filters(pd.DataFrame(), markets=["win"]).empty


def test_settled_only_excludes_pending(paper_df):
    s = D.settled_only(paper_df)
    assert s["outcome"].notna().all()
    assert len(s) == 4


# ── metrics ───────────────────────────────────────────────────────────────────

def test_paper_metrics_values(paper_df):
    s = D.settled_only(paper_df)
    pm = D.paper_metrics(s, 1000.0)
    assert pm["n_bets"] == 4
    assert pm["staked"] == 40.0
    assert pm["profit"] == -4.0          # 20 -10 -4 -10
    assert pm["roi_pct"] == round(-4.0 / 40.0 * 100, 2)
    assert pm["hit_rate"] == round(1 / 4, 4)   # one win of four decided
    assert pm["final_bankroll"] == 996.0
    assert pm["beat_close_rate"] == round(1 / 4, 4)   # one positive clv


def test_paper_metrics_empty():
    pm = D.paper_metrics(pd.DataFrame(), 1000.0)
    assert pm["n_bets"] == 0
    assert pm["roi_pct"] is None
    assert pm["final_bankroll"] == 1000.0


def test_paper_metrics_void_excluded_from_hit_rate():
    df = pd.DataFrame({
        "stake": [10.0, 10.0], "profit": [20.0, 0.0], "odds_decimal": [3.0, 2.0],
        "outcome": ["win", "void"], "settled_at": pd.to_datetime(["2026-06-01", "2026-06-02"], utc=True),
        "clv_pct": [5.0, None],
    })
    pm = D.paper_metrics(df, 1000.0)
    assert pm["n_void"] == 1
    assert pm["hit_rate"] == 1.0   # 1 win / 1 decided (void not counted)


def test_paper_ae_rows(paper_df):
    rows = D.paper_ae_rows(D.settled_only(paper_df))
    assert rows and all("ae" in r and "bucket" in r for r in rows)
    # every actual/expected pair is internally consistent
    for r in rows:
        if r["ae"] is not None and r["expected"]:
            assert r["ae"] == pytest.approx(r["actual"] / r["expected"], rel=1e-2)


# ── backtest loaders ──────────────────────────────────────────────────────────

def test_list_and_load_backtest(tmp_path, bt_summary):
    run = tmp_path / "20260101_000000"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps(bt_summary), encoding="utf-8")
    assert D.list_backtest_runs(tmp_path) == ["20260101_000000"]
    loaded = D.load_backtest_summary("20260101_000000", tmp_path)
    assert loaded["model_metrics"]["auc"] == 0.668


def test_list_backtest_runs_missing_dir(tmp_path):
    assert D.list_backtest_runs(tmp_path / "nope") == []


def test_load_backtest_summary_absent(tmp_path):
    assert D.load_backtest_summary("ghost", tmp_path) is None


def test_load_backtest_ledger_absent(tmp_path):
    assert D.load_backtest_ledger("ghost", "ev_flat", tmp_path).empty


def test_backtest_compare_table_sorted(bt_summary):
    cmp = D.backtest_compare_table(bt_summary)
    assert list(cmp["strategy"]) == ["edge10_flat", "ev_flat"]  # higher yield first
    assert "clv_pct_mean" in cmp.columns


def test_backtest_compare_table_empty():
    assert D.backtest_compare_table({}).empty


# ── figure builders ───────────────────────────────────────────────────────────

def test_equity_curve_fig(paper_df):
    fig = D.equity_curve_fig(D.settled_only(paper_df), 1000.0)
    assert isinstance(fig, go.Figure)
    # three traces: an invisible baseline at the starting bankroll, the curve
    # that fills down to it, and the single end marker + direct label
    assert len(fig.data) == 3
    curve = fig.data[1]
    # the fill must reference the BASELINE, not zero — `tozeroy` fills from 0,
    # which is off-scale for a bankroll that opens at 1,000
    assert curve.fill == "tonexty"
    # and it takes the line's own hue, not a fixed third colour
    r, g, b = (int(curve.line.color[i:i + 2], 16) for i in (1, 3, 5))
    assert curve.fillcolor == f"rgba({r}, {g}, {b}, 0.1)"


def test_equity_curve_fig_empty():
    fig = D.equity_curve_fig(pd.DataFrame(), 1000.0)
    assert isinstance(fig, go.Figure)  # honest empty state, not a crash


def test_calibration_fig_two_series(paper_df, bt_summary):
    fig = D.calibration_ae_fig(D.paper_ae_rows(D.settled_only(paper_df)),
                               bt_summary["ae_by_prob"])
    names = {t.name for t in fig.data}
    assert {"Paper", "Backtest"} <= names


def test_calibration_fig_empty():
    fig = D.calibration_ae_fig([], [])
    assert isinstance(fig, go.Figure)


def test_cumulative_profit_fig(paper_df):
    fig = D.cumulative_profit_fig(D.settled_only(paper_df))
    assert isinstance(fig, go.Figure) and len(fig.data) >= 1


def test_clv_fig(paper_df):
    fig = D.clv_fig(D.settled_only(paper_df))
    assert isinstance(fig, go.Figure) and len(fig.data) == 1


def test_clv_fig_no_clv():
    df = pd.DataFrame({"outcome": ["win"], "stake": [10.0], "profit": [20.0],
                       "odds_decimal": [3.0], "clv_pct": [None],
                       "settled_at": pd.to_datetime(["2026-06-01"], utc=True)})
    fig = D.clv_fig(df)
    assert isinstance(fig, go.Figure)  # falls back to empty state


def test_normalized_equity_fig(paper_df):
    led = pd.DataFrame({"bankroll_after": [1010.0, 1005.0, 1020.0],
                        "profit": [10.0, -5.0, 15.0]})
    fig = D.normalized_equity_fig(D.settled_only(paper_df), 1000.0, led)
    names = {t.name for t in fig.data}
    assert "Paper" in names and any("Backtest" in n for n in names)
