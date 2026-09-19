"""Unit tests for the pure metric helpers in models.ensemble_experiment.

The experiment itself is a heavyweight research script (trains models on the full
matrix); these tests cover only its novel, side-effect-free pieces — the ECE
calculation and the metrics bundle — so the numbers reported in
memory/model-13-ensemble.md are reproducible. Importing the module is safe
without xgboost installed (xgboost is imported lazily inside fit_xgboost).
"""
import numpy as np
import pytest

from models.ensemble_experiment import ece, metrics


def test_ece_perfectly_calibrated_is_zero():
    # Predicted 0.25 four times; exactly one positive → observed freq 0.25.
    p = np.array([0.25, 0.25, 0.25, 0.25])
    y = np.array([1, 0, 0, 0])
    assert ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_constant_miscalibration():
    # Always predict 0.5, never win → gap of 0.5 in a single bin.
    p = np.full(10, 0.5)
    y = np.zeros(10, dtype=int)
    assert ece(y, p) == pytest.approx(0.5, abs=1e-12)


def test_ece_two_bins_weighted_average():
    # Bin [0.0,0.1): pred 0.05, obs 0.0 → gap 0.05; bin [0.9,1.0): pred 0.95,
    # obs 1.0 → gap 0.05. Equal counts → weighted mean gap 0.05.
    p = np.concatenate([np.full(10, 0.05), np.full(10, 0.95)])
    y = np.concatenate([np.zeros(10), np.ones(10)]).astype(int)
    assert ece(y, p) == pytest.approx(0.05, abs=1e-12)


def test_ece_ignores_empty_bins():
    # Only the two end bins are populated, and both are perfectly calibrated
    # (0.0→never, 1.0→always); empty bins must not dilute the zero result.
    p = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0, 0, 1, 1])
    assert ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_metrics_bundle_keys_and_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics(y, p)
    assert set(m) == {"auc", "logloss", "brier", "ece"}
    assert m["auc"] == pytest.approx(1.0)          # perfectly ranked
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
