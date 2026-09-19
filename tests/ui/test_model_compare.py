"""Tests for the model-comparison page data + HTML layer (``ui.model_compare``).

A smoke import of the page module plus unit coverage of the Streamlit-free
builders — run discovery, model identification, the comparison set, the scorecard
rows / best-cell logic, the reliability-chart frame and the scorecard HTML — and
the compact verdict strip reused on the performance pages (``ui.model_honesty``).
The Streamlit assembly (``ui/pages/15_Model_Compare.py``) is byte-compiled only.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from ui import model_compare as MC
from ui import model_honesty as MH


# ── fixtures ──────────────────────────────────────────────────────────────────

def _summary(kind: str, *, log_loss: float, market_ll: float, roi: float,
             clv: float, beats: bool) -> dict:
    mt = "catboost" if kind == "catboost" else "lgbm_softmax"
    return {
        "model": {"model_type": mt, "leakage_verified": True},
        "window": {"start": "2026-05-23", "end": "2026-06-12"},
        "devig_method": "proportional",
        "head_to_head": {
            "model": {"log_loss": log_loss, "brier_runner_level": 0.080, "ece": 0.011},
            "market": {"log_loss": market_ll, "brier_runner_level": 0.0798, "ece": 0.012},
            "model_beats_market_logloss": beats,
        },
        "betting": {"n_bets": 1190, "roi": roi, "mean_clv_log": clv,
                    "clv_beat_rate": 0.30},
        "integrity": [],
    }


def _band_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"band": "<2.0 (odds-on)", "n_runners": 47, "actual_rate": 0.574,
         "model_mean": 0.634, "market_mean": 0.637},
        {"band": "2.0-3.0", "n_runners": 111, "actual_rate": 0.414,
         "model_mean": 0.392, "market_mean": 0.390},
        {"band": "ALL", "n_runners": 2972, "actual_rate": 0.105,
         "model_mean": 0.105, "market_mean": 0.105},
    ])


def _write_holdout(root, kind: str, stamp: str, *, summary: dict,
                   band: pd.DataFrame | None = None):
    d = root / f"holdout_{kind}_{stamp}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if band is not None:
        band.to_csv(d / "odds_band_calibration.csv", index=False)
    return d


@pytest.fixture
def two_models(tmp_path):
    """An lgbm run (beats, negative CLV) + a catboost run (no edge)."""
    _write_holdout(tmp_path, "lgbm", "20260618_212927",
                   summary=_summary("lgbm", log_loss=1.7135, market_ll=1.7285,
                                    roi=0.1587, clv=-0.117, beats=True),
                   band=_band_df())
    _write_holdout(tmp_path, "catboost", "20260617_100000",
                   summary=_summary("catboost", log_loss=1.7400, market_ll=1.7285,
                                    roi=-0.02, clv=-0.05, beats=False),
                   band=_band_df())
    return tmp_path


# ── smoke ─────────────────────────────────────────────────────────────────────

def test_module_exposes_builders():
    for name in ("discover_holdouts", "latest_by_kind", "comparison_set",
                 "model_metrics", "market_metrics", "compare_rows", "best_cells",
                 "calibration_chart_df", "compare_table_html", "reliability_chart",
                 "render"):
        assert callable(getattr(MC, name)), name


def test_page_module_compiles():
    import py_compile
    from pathlib import Path
    page = Path(MC.__file__).resolve().parent / "pages" / "15_Model_Compare.py"
    assert page.exists()
    py_compile.compile(str(page), doraise=True)


# ── model identification ──────────────────────────────────────────────────────

def test_model_kind_from_summary_and_name():
    assert MC.model_kind({"model": {"model_type": "lgbm_softmax"}}) == "lgbm"
    assert MC.model_kind({"model": {"model_type": "catboost"}}) == "catboost"
    assert MC.model_kind({}, "holdout_catboost_20260101_000000") == "catboost"
    assert MC.model_kind({}, "holdout_lgbm_20260101_000000") == "lgbm"
    assert MC.model_kind({}, "holdout_mystery_x") is None


# ── discovery ─────────────────────────────────────────────────────────────────

def test_discover_holdouts_newest_first(two_models):
    runs = MC.discover_holdouts(two_models)
    assert [r["kind"] for r in runs] == ["lgbm", "catboost"]  # newest stamp first


def test_discover_skips_runs_without_head_to_head(tmp_path):
    d = tmp_path / "holdout_lgbm_20260101_000000"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"model": {}}), encoding="utf-8")
    assert MC.discover_holdouts(tmp_path) == []


def test_discover_missing_dir(tmp_path):
    assert MC.discover_holdouts(tmp_path / "nope") == []


# ── comparison set ────────────────────────────────────────────────────────────

def test_latest_by_kind(two_models):
    runs = MC.discover_holdouts(two_models)
    by_kind = MC.latest_by_kind(runs)
    assert set(by_kind) == {"lgbm", "catboost"}


def test_comparison_set_defaults_to_newest(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    assert selected["kind"] == "lgbm"
    assert set(by_kind) == {"lgbm", "catboost"}


def test_comparison_set_honours_pick(two_models):
    runs = MC.discover_holdouts(two_models)
    cat = next(r for r in runs if r["kind"] == "catboost")
    selected, _ = MC.comparison_set(runs, cat["name"])
    assert selected["name"] == cat["name"]


# ── scorecard rows ────────────────────────────────────────────────────────────

def test_compare_rows_order_and_market(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    rows = MC.compare_rows(selected, by_kind)
    assert [r["label"] for r in rows] == [
        MC._MODEL_LABELS["catboost"], MC._MODEL_LABELS["lgbm"], MC._MARKET_LABEL]
    market = rows[-1]
    assert market["is_market"] and market["ev_roi"] is None and market["clv_pct"] is None


def test_clv_pct_conversion(two_models):
    runs = MC.discover_holdouts(two_models)
    lgbm = MC.latest_by_kind(runs)["lgbm"]
    row = MC.model_metrics(lgbm)
    assert row["clv_pct"] == pytest.approx((math.exp(-0.117) - 1.0) * 100.0)


def test_best_cells_lower_and_higher(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    rows = MC.compare_rows(selected, by_kind)
    best = MC.best_cells(rows)
    # lgbm (index 1) has the lower log-loss and the higher ROI of the two models.
    assert rows[best["log_loss"]]["kind"] == "lgbm"
    assert rows[best["ev_roi"]]["kind"] == "lgbm"
    # market is never the EV-ROI winner (it carries none).
    assert not rows[best["ev_roi"]]["is_market"]


# ── chart frame ───────────────────────────────────────────────────────────────

def test_calibration_chart_df_has_all_series(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    df = MC.calibration_chart_df(selected, by_kind)
    assert set(df["series"]) == {
        MC._MODEL_LABELS["catboost"], MC._MODEL_LABELS["lgbm"], MC._MARKET_LABEL}
    # aggregate band dropped
    assert "ALL" not in set(df["band"])


def test_calibration_chart_df_empty_when_no_bands(tmp_path):
    _write_holdout(tmp_path, "lgbm", "20260618_212927",
                   summary=_summary("lgbm", log_loss=1.7, market_ll=1.72,
                                    roi=0.1, clv=-0.1, beats=True))  # no band csv
    runs = MC.discover_holdouts(tmp_path)
    selected, by_kind = MC.comparison_set(runs)
    df = MC.calibration_chart_df(selected, by_kind)
    assert df.empty


# ── graceful single-model degrade ─────────────────────────────────────────────

def test_single_model_renders_model_and_market(tmp_path):
    _write_holdout(tmp_path, "lgbm", "20260618_212927",
                   summary=_summary("lgbm", log_loss=1.7135, market_ll=1.7285,
                                    roi=0.1587, clv=-0.117, beats=True),
                   band=_band_df())
    runs = MC.discover_holdouts(tmp_path)
    selected, by_kind = MC.comparison_set(runs)
    rows = MC.compare_rows(selected, by_kind)
    assert [r["label"] for r in rows] == [MC._MODEL_LABELS["lgbm"], MC._MARKET_LABEL]
    html = MC.compare_table_html(rows)
    assert "LightGBM" in html and MC._MARKET_LABEL in html
    df = MC.calibration_chart_df(selected, by_kind)
    assert set(df["series"]) == {MC._MODEL_LABELS["lgbm"], MC._MARKET_LABEL}


# ── HTML / chart builders ─────────────────────────────────────────────────────

def test_compare_table_marks_paper_only_for_negative_clv(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    rows = MC.compare_rows(selected, by_kind)
    html = MC.compare_table_html(rows)
    assert "paper-only" in html          # lgbm beats log-loss but CLV < 0
    assert "no edge" in html             # catboost does not beat the market


def test_compare_table_empty():
    assert "No model runs" in MC.compare_table_html([])


def test_reliability_chart_none_on_empty():
    assert MC.reliability_chart(pd.DataFrame(
        columns=["series", "band", "predicted", "actual"])) is None


def test_reliability_chart_builds(two_models):
    runs = MC.discover_holdouts(two_models)
    selected, by_kind = MC.comparison_set(runs)
    df = MC.calibration_chart_df(selected, by_kind)
    chart = MC.reliability_chart(df)
    assert chart is not None
    spec = chart.to_dict()  # serialises without error → valid Vega-Lite
    assert spec["layer"]


# ── verdict strip (reused on the performance pages) ───────────────────────────

def test_verdict_strip_go_with_negative_clv_flags_paper_only():
    verdict = {"lgbm": {"model_beats_market_logloss": True, "go": True,
                        "model_log_loss": 1.7135, "market_log_loss": 1.7285,
                        "mean_clv_log": -0.117}}
    html = MH.verdict_strip_html(verdict)
    assert "GO" in html and "paper-only" in html


def test_verdict_strip_nogo():
    verdict = {"lgbm": {"model_beats_market_logloss": False,
                        "model_log_loss": 2.0, "market_log_loss": 1.9}}
    html = MH.verdict_strip_html(verdict)
    assert "NO-GO" in html


def test_verdict_strip_unavailable():
    assert "No verdict" in MH.verdict_strip_html({})
    assert MH.latest_verdict_line({}) is None
