"""Tests for models.train_lgbm — the v3 LightGBM fresh-window retrain + GO/NO-GO.

Load-bearing guarantees:
  * the chronological windows resolve to the spec'd shape
    (train < split <= validation <= max_date < holdout_start <= holdout_end),
    with the most-recent ~3 weeks reserved as the holdout;
  * the split is leak-free: train+val never reach the holdout, and the three
    slices share no race;
  * end-to-end the trainer fits, scores the holdout through the shared harness,
    writes the booster + summary.json + a lgbm_v3_meta.json that carries a verdict.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from features.lgbm_adapter import SELECTED_V4_COLS
from models.train_lgbm import (
    Windows,
    chronological_split,
    resolve_windows,
    run_training,
)

_START = pd.Timestamp("2026-01-01")
_N_DAYS = 60  # → data max = 2026-03-01


def _make_df(n_days: int = _N_DAYS, races_per_day: int = 2, seed: int = 5) -> pd.DataFrame:
    """Synthetic WIN-market matrix spanning ``n_days`` days from 2026-01-01.

    One winner per race; real pre-off prices (morningwap/ppwap) and a finishing SP
    that differs; the v4 feature columns the adapter selects; plus a couple of
    post-off columns the adapter must drop."""
    rng = np.random.default_rng(seed)
    bands = ["unknown", "first_time", "long_absence", "layoff", "normal", "fresh"]
    rows = []
    hid = 0
    for d in range(n_days):
        day = _START + pd.Timedelta(days=d)
        for r in range(races_per_day):
            field = int(rng.integers(5, 10))
            mw = rng.uniform(2.0, 30.0, size=field)
            winner = int(rng.integers(field))
            ruid = f"VEN{r}|{day.date()} 13:{r:02d}"
            for i in range(field):
                row = {
                    "race_date": day,
                    "race_uid": ruid,
                    "venue": f"VEN{r}",
                    "market_type": "WIN",
                    "horse_id": f"h{hid}",
                    "horse_name": f"Horse{hid}",
                    "morningwap": float(mw[i]),
                    "ppwap": float(mw[i] * rng.uniform(0.92, 1.08)),
                    "odds_finish": float(mw[i] * rng.uniform(0.6, 1.5)),
                    "sp": np.nan,  # nulled (betSP backbone) — must never be read
                    "position": (1 if i == winner else i + 2),
                    "won": int(i == winner),
                    "odds_drift": float(rng.normal()),   # post-off → must be excluded
                    "ew_value_index": float(rng.normal()),  # post-off → must be excluded
                    "freshness_band": str(rng.choice(bands)),
                    "is_steaming": bool(rng.integers(0, 2)),
                    "is_drifting": bool(rng.integers(0, 2)),
                    "historical_win_rate": float(rng.uniform(0.0, 0.4)),
                }
                for c in SELECTED_V4_COLS:
                    row.setdefault(c, float(rng.normal()))
                rows.append(row)
                hid += 1
    return pd.DataFrame(rows)


# ── window resolution ───────────────────────────────────────────────────────


class TestResolveWindows:
    def test_default_three_week_holdout(self):
        df = _make_df()
        w = resolve_windows(df["race_date"])
        data_max = pd.Timestamp("2026-03-01")
        assert w.data_max_date == data_max
        assert w.holdout_end == data_max
        assert w.holdout_start == data_max - pd.Timedelta(days=20)
        assert w.max_date == w.holdout_start - pd.Timedelta(days=1)
        # spec ordering
        assert w.train_start <= w.split_date <= w.max_date < w.holdout_start <= w.holdout_end

    def test_explicit_max_date_derives_holdout_start(self):
        df = _make_df()
        w = resolve_windows(df["race_date"], max_date="2026-02-08")
        assert w.max_date == pd.Timestamp("2026-02-08")
        assert w.holdout_start == pd.Timestamp("2026-02-09")

    def test_explicit_holdout_start_derives_max_date(self):
        df = _make_df()
        w = resolve_windows(df["race_date"], holdout_start="2026-02-15")
        assert w.holdout_start == pd.Timestamp("2026-02-15")
        assert w.max_date == pd.Timestamp("2026-02-14")

    def test_explicit_split_date_honoured(self):
        df = _make_df()
        w = resolve_windows(df["race_date"], split_date="2026-01-20")
        assert w.split_date == pd.Timestamp("2026-01-20")

    def test_inconsistent_bounds_raise(self):
        df = _make_df()
        # max_date on/after holdout_start would let the holdout be trained on.
        with pytest.raises(SystemExit, match="strictly before"):
            resolve_windows(df["race_date"], max_date="2026-03-05",
                            holdout_start="2026-03-01")


# ── chronological split ──────────────────────────────────────────────────────


class TestChronologicalSplit:
    def _split(self):
        df = _make_df()
        df["_day"] = pd.to_datetime(df["race_date"]).dt.normalize().to_numpy()
        w = resolve_windows(df["race_date"])
        return (df, w, *chronological_split(df, w))

    def test_slices_are_date_ordered_and_leakfree(self):
        df, w, train_df, val_df, holdout_df, train_max = self._split()
        assert train_df["_day"].max() < w.split_date
        assert val_df["_day"].min() >= w.split_date
        assert val_df["_day"].max() <= w.max_date
        assert holdout_df["_day"].min() >= w.holdout_start
        # the load-bearing invariant
        assert train_max < w.holdout_start

    def test_no_race_spans_two_slices(self):
        df, w, train_df, val_df, holdout_df, train_max = self._split()
        tr, va, ho = (set(train_df["race_uid"]), set(val_df["race_uid"]),
                      set(holdout_df["race_uid"]))
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(ho)
        assert va.isdisjoint(ho)
        assert len(ho) > 0 and len(tr) > 0 and len(va) > 0


# ── end-to-end ───────────────────────────────────────────────────────────────


class TestRunTrainingEndToEnd:
    def test_trains_scores_and_writes_verdict(self, tmp_path):
        pytest.importorskip("lightgbm")
        df = _make_df()
        out_model = tmp_path / "lgbm_won_v3.txt"
        out_meta = tmp_path / "lgbm_v3_meta.json"
        holdout_out = tmp_path / "holdout"

        result = run_training(
            df=df,
            output_path=out_model,
            meta_path=out_meta,
            holdout_out=holdout_out,
            model_version="v3-lgbm-test",
            model_params={"n_estimators": 40, "num_leaves": 7,
                          "min_data_in_leaf": 2, "seed": 0},
            early_stopping_rounds=20,
            show_importance=True,
        )

        # artefacts exist
        assert out_model.exists()
        assert out_model.with_suffix(".meta.json").exists()  # LGBMSoftmaxModel sidecar
        assert out_meta.exists()
        assert (holdout_out / "summary.json").exists()
        assert (holdout_out / "ledger.csv").exists()

        # the meta carries a verdict with the GO/NO-GO boolean
        meta = json.loads(out_meta.read_text())
        assert "verdict" in meta
        assert isinstance(meta["verdict"]["model_beats_market_logloss"], bool)
        assert meta["verdict"]["model_beats_market_logloss"] == result["summary"][
            "model_beats_market_logloss"]

        # leakage discipline recorded and satisfied
        assert meta["leakage"]["assert_train_max_lt_holdout_start"] is True
        assert "morningwap" in meta["leakage"]["market_features"]

        # holdout summary is the shared-harness shape
        summary = json.loads((holdout_out / "summary.json").read_text())
        assert "model_beats_market_logloss" in summary
        assert summary["head_to_head"]["n_races_kept"] > 0
        names = {c["name"] for c in summary["integrity"]}
        assert "lookahead_bias" in names

        # feature importance was recorded (--importance)
        assert meta["feature_importance"]
        assert isinstance(result["windows"], Windows)

    def test_feature_cols_override_trains_the_independent_branch(self, tmp_path):
        """Step 10: an explicit feature_cols whitelist (e.g. INDEPENDENT_FEATURE_COLS)
        must actually constrain what the booster sees, end-to-end."""
        pytest.importorskip("lightgbm")
        from features.lgbm_adapter import INDEPENDENT_FEATURE_COLS

        df = _make_df()
        result = run_training(
            df=df,
            output_path=tmp_path / "lgbm_indep.txt",
            meta_path=tmp_path / "lgbm_indep_meta.json",
            holdout_out=tmp_path / "holdout_indep",
            model_version="v3-lgbm-indep-test",
            model_params={"n_estimators": 30, "num_leaves": 7,
                          "min_data_in_leaf": 2, "seed": 0},
            early_stopping_rounds=15,
            feature_cols=INDEPENDENT_FEATURE_COLS,
        )
        assert result["model"].feature_name == INDEPENDENT_FEATURE_COLS
        assert "implied_prob" not in result["model"].feature_name
        assert result["summary"]["head_to_head"]["n_races_kept"] > 0
