"""Tests for the model-honesty panel data + HTML layer (``ui.model_honesty``).

A smoke import of the page module plus unit coverage of the Streamlit-free
builders — discovery, loading, verdict normalisation, the GO/NO-GO banner, the
CLV read, the integrity list and the per-band calibration table. The Streamlit
assembly (``ui/pages/14_Model_Honesty.py``) is exercised only via the pure
functions it delegates to, which is the whole point of keeping them headless.
"""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from ui import model_honesty as MH


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def verdict() -> dict:
    """A verdict block shaped like the real predictions.json: a GO line whose
    closing-line value is negative (the honest-trap case) plus an unavailable
    second model."""
    return {
        "lgbm": {
            "go": True,
            "model_beats_market_logloss": True,
            "model_log_loss": 1.71355,
            "market_log_loss": 1.72846,
            "logloss_gap_market_minus_model": 0.0149,
            "n_races_kept": 313,
            "ev_roi": 0.1587,
            "ev_n_bets": 1190,
            "mean_clv_log": -0.1169,
            "clv_beat_rate": 0.3017,
            "holdout_window": {"start": "2026-05-23", "end": "2026-06-12"},
            "model_version": "v3-lgbm-20260618_212927",
        },
        "catboost": {"available": False, "note": "no CatBoost holdout verdict on disk"},
    }


@pytest.fixture
def summary() -> dict:
    return {
        "model": {"leakage_verified": True},
        "devig_method": "proportional",
        "betting": {"n_bets": 1190, "mean_clv_log": -0.1169, "clv_beat_rate": 0.3017},
        "integrity": [
            {"name": "lookahead_bias", "status": "OK", "message": "OK: first bet after cutoff."},
            {"name": "course_cherrypicking", "status": "WARN",
             "message": "WARN: top venue contributes 88.9% of profit."},
            {"name": "liquidity", "status": "FAIL", "message": "FAIL: no matched volume."},
        ],
    }


@pytest.fixture
def band_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"band": "<2.0 (odds-on)", "n_runners": 47, "n_wins": 27, "actual_rate": 0.5745,
         "model_mean": 0.6347, "market_mean": 0.6369, "model_gap": -0.060, "market_gap": -0.062,
         "ae_model": 0.905, "ae_market": 0.902},
        {"band": "2.0-3.0", "n_runners": 111, "n_wins": 46, "actual_rate": 0.4144,
         "model_mean": 0.3924, "market_mean": 0.3896, "model_gap": 0.022, "market_gap": 0.025,
         "ae_model": 1.056, "ae_market": 1.064},
        {"band": "51.0+", "n_runners": 0, "n_wins": 0, "actual_rate": float("nan"),
         "model_mean": float("nan"), "market_mean": float("nan"),
         "model_gap": float("nan"), "market_gap": float("nan"),
         "ae_model": float("nan"), "ae_market": float("nan")},
        {"band": "ALL", "n_runners": 2972, "n_wins": 313, "actual_rate": 0.1053,
         "model_mean": 0.1053, "market_mean": 0.1053, "model_gap": 0.0, "market_gap": 0.0,
         "ae_model": 1.0, "ae_market": 1.0},
    ])


def _write_holdout(root, stamp: str, *, summary: dict, band: pd.DataFrame | None = None):
    d = root / f"holdout_lgbm_{stamp}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if band is not None:
        band.to_csv(d / "odds_band_calibration.csv", index=False)
    return d


# ── smoke import ──────────────────────────────────────────────────────────────

def test_module_exposes_builders():
    for name in ("find_latest_holdout", "load_summary", "load_band_table",
                 "load_verdict", "verdict_lines", "banner_html", "clv_block_html",
                 "integrity_rows_html", "band_table_html", "render"):
        assert callable(getattr(MH, name)), name


def test_page_module_compiles():
    """The Streamlit page exists and is syntactically valid (smoke).

    It can't be imported under a numeric module name (and importing it would run
    the Streamlit assembly with no script context), so we byte-compile it — that
    catches syntax/typo regressions without standing up a Streamlit runtime."""
    import py_compile
    from pathlib import Path
    page = Path(MH.__file__).resolve().parent / "pages" / "14_Model_Honesty.py"
    assert page.exists()
    py_compile.compile(str(page), doraise=True)


# ── discovery / loading ───────────────────────────────────────────────────────

def test_find_latest_holdout_picks_newest(tmp_path, summary):
    _write_holdout(tmp_path, "20260101_000000", summary=summary)
    newest = _write_holdout(tmp_path, "20260618_212927", summary=summary)
    assert MH.find_latest_holdout(tmp_path) == newest


def test_find_latest_holdout_missing_dir(tmp_path):
    assert MH.find_latest_holdout(tmp_path / "nope") is None


def test_find_latest_holdout_ignores_dirs_without_summary(tmp_path):
    (tmp_path / "holdout_empty").mkdir()
    assert MH.find_latest_holdout(tmp_path) is None


def test_load_summary_and_band(tmp_path, summary, band_df):
    d = _write_holdout(tmp_path, "20260618_212927", summary=summary, band=band_df)
    loaded = MH.load_summary(d)
    assert loaded["devig_method"] == "proportional"
    bt = MH.load_band_table(d)
    assert list(bt["band"])[:2] == ["<2.0 (odds-on)", "2.0-3.0"]


def test_load_summary_absent(tmp_path):
    assert MH.load_summary(None) is None
    assert MH.load_summary(tmp_path / "ghost") is None


def test_load_band_table_absent(tmp_path, summary):
    d = _write_holdout(tmp_path, "20260618_212927", summary=summary)  # no csv
    assert MH.load_band_table(d) is None


def test_load_verdict_absent(tmp_path):
    assert MH.load_verdict(tmp_path / "ghost.json") == {}


def test_load_verdict_real(tmp_path, verdict):
    p = tmp_path / "predictions.json"
    p.write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
    assert MH.load_verdict(p)["lgbm"]["go"] is True


# ── verdict normalisation ─────────────────────────────────────────────────────

def test_verdict_lines_split(verdict):
    lines = {ln["key"]: ln for ln in MH.verdict_lines(verdict)}
    assert lines["lgbm"]["status"] == "go"
    assert lines["catboost"]["status"] == "unavailable"


def test_verdict_lines_nogo():
    lines = MH.verdict_lines({"lgbm": {"model_beats_market_logloss": False,
                                       "model_log_loss": 2.0, "market_log_loss": 1.9}})
    assert lines[0]["status"] == "nogo"


def test_verdict_lines_skips_non_dict():
    assert MH.verdict_lines({"lgbm": None}) == []


# ── banner ────────────────────────────────────────────────────────────────────

def test_banner_go_flags_negative_clv(verdict):
    line = MH.verdict_lines(verdict)[0]
    html = MH.banner_html(line)
    assert "Beats the market" in html
    # model vs market log-loss, formatted exactly as the code does (float-safe)
    assert MH._fmt_ll(1.71355) in html and MH._fmt_ll(1.72846) in html
    assert "Caution" in html and "paper-only" in html  # negative-CLV honesty flag


def test_banner_nogo_wording():
    line = MH.verdict_lines({"lgbm": {"model_beats_market_logloss": False,
                                      "model_log_loss": 2.0, "market_log_loss": 1.9}})[0]
    html = MH.banner_html(line)
    assert "Does NOT beat the market" in html
    assert "nogo" in html


def test_banner_unavailable():
    html = MH.banner_html({"key": "catboost", "label": "CatBoost",
                           "status": "unavailable", "note": "none on disk"})
    assert "No verdict yet" in html and "none on disk" in html


# ── CLV ───────────────────────────────────────────────────────────────────────

def test_clv_summary_negative():
    s = MH.clv_summary(-0.1169, 0.3017)
    assert s["positive"] is False
    assert s["mean_pct"] == pytest.approx((math.exp(-0.1169) - 1) * 100, rel=1e-6)
    assert "worse" in s["plain"]


def test_clv_summary_positive():
    s = MH.clv_summary(0.05, 0.6)
    assert s["positive"] is True and "beat the market" in s["plain"]


def test_clv_summary_none():
    assert MH.clv_summary(None, None) is None
    assert MH.clv_summary(float("nan"), 0.3) is None


def test_clv_block_marks_worse(verdict):
    html = MH.clv_block_html(MH.verdict_lines(verdict)[0])
    assert "Mean CLV" in html and "Beat-close rate" in html
    assert "30%" in html
    assert "<em>worse</em>" in html  # emphasised, not colour-alone


# ── integrity ─────────────────────────────────────────────────────────────────

def test_integrity_rows_badges(summary):
    html = MH.integrity_rows_html(summary["integrity"])
    assert 'mh-badge ok' in html and 'mh-badge warn' in html and 'mh-badge fail' in html
    assert "Lookahead Bias" in html  # name humanised


def test_integrity_rows_empty():
    assert "No integrity checks" in MH.integrity_rows_html([])


def test_integrity_unknown_status_defaults_warn():
    html = MH.integrity_rows_html([{"name": "x", "status": "WEIRD", "message": "m"}])
    assert "mh-badge warn" in html


# ── band table ────────────────────────────────────────────────────────────────

def test_band_table_renders_and_marks_better(band_df):
    html = MH.band_table_html(band_df)
    # band label is HTML-escaped ("<2.0…" → "&lt;2.0…"), as it should be
    assert "2.0 (odds-on)" in html and "Closer fit" in html
    assert "mh-better" in html  # at least one side flagged closer
    assert "mh-agg" in html     # the ALL aggregate row is de-emphasised


def test_band_table_handles_nan(band_df):
    html = MH.band_table_html(band_df)
    # the empty 51.0+ band must degrade to em-dashes, not 'nan'
    assert "nan" not in html.lower()


def test_band_table_none():
    assert "No per-band calibration" in MH.band_table_html(None)
    assert "No per-band calibration" in MH.band_table_html(pd.DataFrame())
