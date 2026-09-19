import numpy as np
import pytest
from catboost import CatBoostClassifier

from models.evaluate import report, _ae_by_band, _reliability_table


def _tiny_model(n_features=5, n_samples=120, seed=42):
    """Train a minimal CatBoost model on synthetic data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features)).astype(float)
    y = (X[:, 0] > 0.5).astype(int)
    model = CatBoostClassifier(
        iterations=10, verbose=False, allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    model.fit(X, y)
    return model, X, y


def test_report_returns_dict():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)  # all mid-band
    result = report(model, X, y, implied_prob, cols, "won")
    assert isinstance(result, dict)


def test_report_has_auc_key():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_report_has_band_aucs():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    # spread across all three bands
    rng = np.random.default_rng(0)
    implied_prob = rng.uniform(0.01, 0.5, len(y))
    result = report(model, X, y, implied_prob, cols, "won")
    assert "band_aucs" in result
    assert set(result["band_aucs"].keys()) == {"favourite", "mid", "underdog"}


def test_report_confusion_matrix_shape():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    cm = result["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_report_feature_importance_length():
    model, X, y = _tiny_model(n_features=5)
    cols = [f"f{i}" for i in range(5)]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    fi = result["feature_importance"]
    assert len(fi) == min(20, 5)  # 5 features → 5 entries


def test_report_feature_importance_sorted_descending():
    model, X, y = _tiny_model(n_features=5)
    cols = [f"f{i}" for i in range(5)]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    importances = [v for _, v in result["feature_importance"]]
    assert importances == sorted(importances, reverse=True)


def test_band_auc_none_when_single_class():
    """Band with only one class label -> AUC reported as None, not an error."""
    # Use a normally-trained model. We need two classes globally (so overall
    # AUC works), but only one class within the favourite band, so that band
    # returns None without raising.
    model, X, y = _tiny_model(n_samples=120)
    cols = [f"f{i}" for i in range(X.shape[1])]
    # Put first 10 samples in favourite band (>=0.25), rest in underdog (<0.10)
    implied_prob = np.full(len(y), 0.05)   # all underdog by default
    implied_prob[:10] = 0.5                 # first 10 → favourite band
    # Force first 10 labels all to 0 (single class in favourite band)
    y_mixed = y.copy()
    y_mixed[:10] = 0
    # Ensure at least one 1 exists outside the favourite band so overall AUC works
    y_mixed[10] = 1
    result = report(model, X, y_mixed, implied_prob, cols, "test")
    # favourite band has only class 0 → AUC should be None, not raise
    assert result["band_aucs"]["favourite"] is None


# --- calibration diagnostics --------------------------------------------------

def test_report_has_brier_keys():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    assert "brier" in result and "brier_baseline" in result
    assert 0.0 <= result["brier"] <= 1.0


def test_report_calibrator_changes_brier():
    """A calibrator that collapses probs to a constant changes the reported Brier
    (proving report scores the *calibrated* probabilities, not the raw ones)."""
    import types
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    raw = report(model, X, y, implied_prob, cols, "won")["brier"]
    calib = types.SimpleNamespace(predict=lambda p: np.full(len(p), float(y.mean())))
    cal = report(model, X, y, implied_prob, cols, "won", calibrator=calib)["brier"]
    assert cal != pytest.approx(raw)


def test_report_has_calibration_keys():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    rng = np.random.default_rng(0)
    implied_prob = rng.uniform(0.01, 0.5, len(y))
    result = report(model, X, y, implied_prob, cols, "won")
    assert "ae_by_band" in result
    assert "reliability" in result
    assert set(result["ae_by_band"].keys()) == {"favourite", "mid", "underdog"}


def test_ae_perfectly_calibrated_band():
    """Synthetic data where actual == expected in a band → A/E == 1.0, no flag."""
    # 100 favourite-band rows, model predicts 0.30 each → expected 30 wins;
    # make exactly 30 of them actual wins.
    y = np.zeros(100, dtype=int)
    y[:30] = 1
    y_prob = np.full(100, 0.30)
    ip = np.full(100, 0.30)  # all favourite band
    out = _ae_by_band(y, y_prob, ip)
    assert out["favourite"]["n"] == 100
    assert out["favourite"]["ae"] == pytest.approx(1.0)
    assert out["favourite"]["flagged"] is False
    assert out["mid"]["ae"] is None
    assert out["underdog"]["ae"] is None


def test_ae_overconfident_band_flagged():
    """Model expects 30 wins but only 10 happen → A/E ≈ 0.33, flagged."""
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    y_prob = np.full(100, 0.30)
    ip = np.full(100, 0.30)
    out = _ae_by_band(y, y_prob, ip)
    assert out["favourite"]["ae"] == pytest.approx(10.0 / 30.0)
    assert out["favourite"]["flagged"] is True


def test_ae_ignores_nan_implied_prob():
    y = np.array([1, 0, 1, 0])
    y_prob = np.array([0.3, 0.3, 0.3, 0.3])
    ip = np.array([0.3, 0.3, np.nan, np.nan])
    out = _ae_by_band(y, y_prob, ip)
    # only the two non-NaN rows (both favourite) count
    assert out["favourite"]["n"] == 2
    assert out["favourite"]["actual"] == pytest.approx(1.0)
    assert out["favourite"]["expected"] == pytest.approx(0.6)


def test_reliability_table_observed_matches_predicted():
    """Well-calibrated: observed freq ≈ mean predicted prob within each bucket."""
    # bucket [0.2,0.3): 100 rows at p=0.25, 25 wins → observed 0.25
    y = np.concatenate([np.ones(25, dtype=int), np.zeros(75, dtype=int)])
    y_prob = np.full(100, 0.25)
    table = _reliability_table(y, y_prob)
    assert len(table) == 1
    row = table[0]
    assert row["bin_lo"] == pytest.approx(0.2)
    assert row["bin_hi"] == pytest.approx(0.3)
    assert row["n"] == 100
    assert row["mean_pred"] == pytest.approx(0.25)
    assert row["observed_freq"] == pytest.approx(0.25)


def test_reliability_table_prob_one_lands_in_last_bucket():
    y = np.array([1, 1, 0])
    y_prob = np.array([1.0, 0.95, 0.92])
    table = _reliability_table(y, y_prob)
    # all three fall in the final [0.9,1.0) bucket; none spill past the edge
    assert len(table) == 1
    assert table[0]["n"] == 3
    assert table[0]["bin_hi"] == pytest.approx(1.0)
