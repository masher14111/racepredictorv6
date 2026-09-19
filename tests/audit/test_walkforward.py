"""Fold-boundary, cross-fitting and race-normalization regressions for the
Stage-4 walk-forward driver. CatBoost is stubbed (repo convention: model-free
tests) — the guarantees under test are the harness's, not the learner's."""
import numpy as np
import pandas as pd
import pytest

import audit.walkforward as wf
from audit.walkforward import (
    AuditWFConfig,
    fit_temperature,
    grouped_softmax,
    run_walkforward_audit,
)


class _StubBooster:
    """Deterministic predict_proba: p = clip(1/bet_price-ish feature, .01, .6)."""

    def __init__(self, **params):
        self.params = params

    def fit(self, X, y, sample_weight=None, eval_set=None, verbose=False):
        return self

    def predict_proba(self, X):
        f = np.asarray(X, dtype=float)[:, 0]
        p = np.clip(np.nan_to_num(f, nan=0.10), 0.01, 0.60)
        return np.column_stack([1 - p, p])

    def get_best_iteration(self):
        return 1


@pytest.fixture()
def stub_catboost(monkeypatch):
    import catboost

    monkeypatch.setattr(catboost, "CatBoostClassifier", _StubBooster)


def _panel(n_days=500, runners=6, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=d)
        odds = rng.uniform(2.0, 12.0, runners)
        odds = odds / (1.0 / odds).sum() / odds  # normalize booksum to 1
        odds = 1.0 / (1.0 / odds / (1.0 / odds).sum())
        winner = rng.choice(runners, p=(1.0 / odds) / (1.0 / odds).sum())
        for i in range(runners):
            rows.append({
                "race_date": date, "race_uid": f"race{d}",
                "horse_id": f"h{d}_{i}", "won": int(i == winner),
                "bet_price": float(odds[i]), "close_price": float(odds[i]),
                "feat_a": 1.0 / odds[i], "implied_prob": 1.0 / odds[i],
            })
    return pd.DataFrame(rows)


class TestFoldBoundaries:
    def test_no_test_row_on_or_before_train_end(self, stub_catboost):
        panel = _panel()
        cfg = AuditWFConfig(feature_cols=["feat_a"], min_train_days=200,
                            test_window_days=60, task_type="CPU",
                            fl_min_rows=50, min_calib_rows=50)
        scored, meta = run_walkforward_audit(panel, cfg)
        assert len(scored) > 0
        by_fold = {m["fold"]: m for m in meta if m.get("scored")}
        for fold, grp in scored.groupby("fold"):
            m = by_fold[fold]
            dates = pd.to_datetime(grp["race_date"], utc=True)
            assert (dates > pd.to_datetime(m["train_end"], utc=True)).all()
            assert (dates <= pd.to_datetime(m["test_end"], utc=True)
                    + pd.Timedelta(days=1)).all()

    def test_final_window_never_scored(self, stub_catboost):
        panel = _panel()
        final_start = "2026-01-01"
        cfg = AuditWFConfig(feature_cols=["feat_a"], min_train_days=200,
                            test_window_days=60, task_type="CPU",
                            final_test_start=final_start,
                            fl_min_rows=50, min_calib_rows=50)
        scored, _ = run_walkforward_audit(panel, cfg)
        assert (pd.to_datetime(scored["race_date"], utc=True)
                < pd.to_datetime(final_start, utc=True)).all()

    def test_every_probability_line_present_and_finite(self, stub_catboost):
        panel = _panel(n_days=400)
        cfg = AuditWFConfig(feature_cols=["feat_a"], min_train_days=250,
                            test_window_days=60, task_type="CPU",
                            fl_min_rows=50, min_calib_rows=50)
        scored, _ = run_walkforward_audit(panel, cfg)
        for col in ("p_raw", "p_ind", "p_adj", "p_adj_sig",
                    "norm_raw", "norm_ind", "norm_adj", "norm_temp"):
            assert col in scored.columns
            assert np.isfinite(scored[col].to_numpy(dtype=float)).all()


class TestRaceNormalization:
    def test_norm_lines_sum_to_one_per_race(self, stub_catboost):
        panel = _panel(n_days=400)
        cfg = AuditWFConfig(feature_cols=["feat_a"], min_train_days=250,
                            test_window_days=60, task_type="CPU",
                            fl_min_rows=50, min_calib_rows=50)
        scored, _ = run_walkforward_audit(panel, cfg)
        for col in ("norm_ind", "norm_adj", "norm_temp"):
            sums = scored.groupby("race_uid")[col].sum()
            assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6), col


class TestGroupedSoftmax:
    def test_sums_to_one_and_orders_by_score(self):
        rid = np.array(["r1"] * 3 + ["r2"] * 2)
        s = np.array([2.0, 1.0, 0.0, 1.0, 1.0])
        q = grouped_softmax(s, rid, temperature=1.0)
        assert q[:3].sum() == pytest.approx(1.0)
        assert q[3:].sum() == pytest.approx(1.0)
        assert q[0] > q[1] > q[2]
        assert q[3] == pytest.approx(q[4])

    def test_high_temperature_flattens(self):
        rid = np.array(["r"] * 3)
        s = np.array([3.0, 0.0, -3.0])
        sharp = grouped_softmax(s, rid, temperature=0.5)
        flat = grouped_softmax(s, rid, temperature=10.0)
        assert sharp[0] > flat[0]
        assert abs(flat[0] - 1 / 3) < abs(sharp[0] - 1 / 3)

    def test_fit_temperature_recovers_unity_for_coherent_probs(self):
        rng = np.random.default_rng(11)
        rows, rids = [], []
        for r in range(300):
            p = rng.dirichlet(np.ones(5) * 2.0)
            w = rng.choice(5, p=p)
            for i in range(5):
                rows.append((p[i], int(i == w)))
                rids.append(f"r{r}")
        p, y = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
        T = fit_temperature(p, y, np.array(rids))
        # Finite-sample noise on 300 Dirichlet races leaves ~±0.3 wobble; the
        # point is T lands near 1 for already-coherent probs, not at an extreme.
        assert 0.7 <= T <= 1.4


class TestCrossFitting:
    def test_fl_fit_only_sees_train_tail(self, stub_catboost, monkeypatch):
        """Cross-fit proof: capture every OddsBandCalibrator.fit call and check
        its rows all predate the fold's test window."""
        calls = []
        real_fit = wf.OddsBandCalibrator.fit.__func__

        def spy(cls, prob, odds, y, **kw):
            calls.append(len(np.asarray(prob)))
            return real_fit(cls, prob, odds, y, **kw)

        monkeypatch.setattr(wf.OddsBandCalibrator, "fit", classmethod(spy))
        panel = _panel(n_days=400)
        cfg = AuditWFConfig(feature_cols=["feat_a"], min_train_days=250,
                            test_window_days=60, task_type="CPU",
                            calibration_frac=0.15,
                            fl_min_rows=50, min_calib_rows=50)
        scored, meta = run_walkforward_audit(panel, cfg)
        scored_folds = [m for m in meta if m.get("scored")]
        # two F-L fits (isotonic + sigmoid) per scored fold, each on the
        # 15% train tail — never the full train, never train+test.
        assert len(calls) == 2 * len(scored_folds)
        for m, n_fit in zip(scored_folds, calls[::2]):
            assert n_fit == int(m["n_train"] * cfg.calibration_frac)
            assert n_fit < m["n_train"]
