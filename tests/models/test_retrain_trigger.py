import os
from unittest.mock import patch

import numpy as np
import pandas as pd

from models import retrain_trigger as rt
from models.features import FEATURE_COLS
from models.retrain_trigger import (
    DriftReport,
    FeatureDrift,
    _archive_current,
    _log_performance_decay,
    _prune_archive,
    _psi,
    check_and_retrain,
    check_drift,
    save_reference_snapshot,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _feature_df(n=200, seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.random(n) * scale for col in FEATURE_COLS})


_CFG = {"psi_threshold": 0.2, "ks_pvalue_threshold": 0.05, "min_drift_features": 1}


# ── _psi ───────────────────────────────────────────────────────────────────────

def test_psi_identical_returns_near_zero():
    arr = np.random.default_rng(0).random(300)
    assert _psi(arr, arr) < 0.01


def test_psi_very_different_distributions_exceeds_threshold():
    rng = np.random.default_rng(0)
    ref = rng.random(300)
    cur = rng.random(300) * 10 + 5  # completely shifted range
    assert _psi(ref, cur) > 0.2


def test_psi_too_few_ref_rows_returns_zero():
    ref = np.array([0.1, 0.2])   # fewer than bins=10
    cur = np.random.default_rng(0).random(100)
    assert _psi(ref, cur) == 0.0


def test_psi_all_nan_ref_returns_zero():
    assert _psi(np.full(50, np.nan), np.random.default_rng(0).random(100)) == 0.0


def test_psi_empty_cur_returns_zero():
    assert _psi(np.random.default_rng(0).random(100), np.array([])) == 0.0


def test_psi_single_unique_ref_value_returns_zero():
    ref = np.ones(100)
    cur = np.random.default_rng(0).random(100)
    assert _psi(ref, cur) == 0.0


# ── check_drift ────────────────────────────────────────────────────────────────

def test_check_drift_returns_drift_report():
    report = check_drift(_feature_df(300, seed=1), _feature_df(300, seed=2), _CFG)
    assert isinstance(report, DriftReport)
    assert len(report.features) == len(FEATURE_COLS)
    for fd in report.features:
        assert isinstance(fd, FeatureDrift)


def test_check_drift_no_drift_same_scale():
    # With 28 features tested at KS p<0.05, ~1-2 false positives are expected
    # by chance (multiple testing). Use min_drift_features=5 so truly stable
    # distributions don't trigger the alert.
    cfg = {**_CFG, "min_drift_features": 5}
    report = check_drift(_feature_df(500, seed=1), _feature_df(500, seed=2), cfg)
    assert not report.triggered


def test_check_drift_detects_large_scale_shift():
    ref = _feature_df(500, seed=0, scale=1.0)
    cur = _feature_df(500, seed=0, scale=100.0)
    report = check_drift(ref, cur, _CFG)
    assert report.triggered
    assert report.n_drifted > 0


def test_check_drift_min_drift_features_gate():
    ref = _feature_df(500, seed=0, scale=1.0)
    cur = _feature_df(500, seed=0, scale=100.0)
    cfg = {**_CFG, "min_drift_features": 999}
    report = check_drift(ref, cur, cfg)
    # many features drift, but triggered=False because n_drifted < 999
    assert not report.triggered
    assert report.n_drifted > 0


def test_check_drift_ks_stat_and_pvalue_populated():
    report = check_drift(_feature_df(300), _feature_df(300, seed=99), _CFG)
    for fd in report.features:
        assert fd.ks_stat is not None
        assert fd.ks_pvalue is not None


def test_check_drift_missing_col_in_cur_skipped():
    ref = _feature_df(200)
    cur = _feature_df(200).drop(columns=[FEATURE_COLS[0]])
    report = check_drift(ref, cur, _CFG)
    assert len(report.features) == len(FEATURE_COLS) - 1


def test_check_drift_n_drifted_matches_features():
    ref = _feature_df(500, seed=0, scale=1.0)
    cur = _feature_df(500, seed=0, scale=100.0)
    report = check_drift(ref, cur, _CFG)
    assert report.n_drifted == sum(1 for f in report.features if f.drifted)


# ── save_reference_snapshot ────────────────────────────────────────────────────

def test_save_reference_snapshot_writes_parquet(tmp_path):
    df = _feature_df(100)
    save_reference_snapshot(df, str(tmp_path))
    path = tmp_path / "reference_features.parquet"
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert list(loaded.columns) == [c for c in FEATURE_COLS if c in df.columns]
    assert len(loaded) == 100


def test_save_reference_snapshot_excludes_extra_columns(tmp_path):
    df = _feature_df(50)
    df["should_not_appear"] = 999
    save_reference_snapshot(df, str(tmp_path))
    loaded = pd.read_parquet(tmp_path / "reference_features.parquet")
    assert "should_not_appear" not in loaded.columns


def test_save_reference_snapshot_creates_dir(tmp_path):
    new_dir = tmp_path / "new_subdir"
    save_reference_snapshot(_feature_df(30), str(new_dir))
    assert (new_dir / "reference_features.parquet").exists()


def test_save_reference_snapshot_overwrites_existing(tmp_path):
    save_reference_snapshot(_feature_df(100, seed=1), str(tmp_path))
    save_reference_snapshot(_feature_df(50, seed=2), str(tmp_path))
    loaded = pd.read_parquet(tmp_path / "reference_features.parquet")
    assert len(loaded) == 50


# ── archiving ──────────────────────────────────────────────────────────────────

def test_archive_current_copies_bin_and_json(tmp_path):
    (tmp_path / "catboost_won_v3.bin").write_bytes(b"model")
    (tmp_path / "catboost_v3_meta.json").write_text('{"targets": {}}')
    _archive_current(str(tmp_path), "20260613T120000")
    dest = tmp_path / "archive" / "20260613T120000"
    assert (dest / "catboost_won_v3.bin").exists()
    assert (dest / "catboost_v3_meta.json").exists()


def test_archive_current_leaves_originals_in_place(tmp_path):
    (tmp_path / "catboost_won_v3.bin").write_bytes(b"model")
    _archive_current(str(tmp_path), "20260613T120000")
    assert (tmp_path / "catboost_won_v3.bin").exists()


def test_archive_current_copies_parquet(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    df.to_parquet(tmp_path / "reference_features.parquet")
    _archive_current(str(tmp_path), "20260613T130000")
    assert (tmp_path / "archive" / "20260613T130000" / "reference_features.parquet").exists()


def test_archive_current_skips_non_artifact_files(tmp_path):
    (tmp_path / "README.txt").write_text("notes")
    _archive_current(str(tmp_path), "20260613T140000")
    dest = tmp_path / "archive" / "20260613T140000"
    assert not (dest / "README.txt").exists()


def test_prune_archive_removes_oldest(tmp_path):
    archive = tmp_path / "archive"
    for ts in ["20260601T000000", "20260607T000000", "20260613T000000"]:
        (archive / ts).mkdir(parents=True)
    _prune_archive(str(tmp_path), keep=2)
    remaining = os.listdir(archive)
    assert len(remaining) == 2
    assert "20260601T000000" not in remaining
    assert "20260613T000000" in remaining


def test_prune_archive_no_op_within_limit(tmp_path):
    archive = tmp_path / "archive"
    for ts in ["20260601T000000", "20260607T000000"]:
        (archive / ts).mkdir(parents=True)
    _prune_archive(str(tmp_path), keep=5)
    assert len(os.listdir(archive)) == 2


def test_prune_archive_no_op_missing_dir(tmp_path):
    # should not raise if archive/ doesn't exist yet
    _prune_archive(str(tmp_path), keep=5)


# ── performance decay ──────────────────────────────────────────────────────────

def test_log_performance_decay_warns_on_large_drop(caplog):
    import logging
    old = {"targets": {"won": {"test_auc": 0.80}}}
    new = {"targets": {"won": {"test_auc": 0.70}}}  # 12.5% relative drop
    with caplog.at_level(logging.WARNING, logger="models.retrain_trigger"):
        _log_performance_decay(old, new, 0.05)
    assert any("decay" in r.message.lower() for r in caplog.records)


def test_log_performance_decay_no_warning_on_small_drop(caplog):
    import logging
    old = {"targets": {"won": {"test_auc": 0.80}}}
    new = {"targets": {"won": {"test_auc": 0.79}}}  # 1.25% drop — below threshold
    with caplog.at_level(logging.WARNING, logger="models.retrain_trigger"):
        _log_performance_decay(old, new, 0.05)
    assert not any("decay" in r.message.lower() for r in caplog.records)


def test_log_performance_decay_improvement_logs_info(caplog):
    import logging
    old = {"targets": {"won": {"test_auc": 0.70}}}
    new = {"targets": {"won": {"test_auc": 0.75}}}
    with caplog.at_level(logging.INFO, logger="models.retrain_trigger"):
        _log_performance_decay(old, new, 0.05)
    assert any("won" in r.message for r in caplog.records)


def test_log_performance_decay_missing_old_target_no_crash():
    _log_performance_decay({}, {"targets": {"won": {"test_auc": 0.7}}}, 0.05)


def test_log_performance_decay_missing_auc_no_crash():
    old = {"targets": {"won": {}}}
    new = {"targets": {"won": {"test_auc": 0.7}}}
    _log_performance_decay(old, new, 0.05)


# ── check_and_retrain (orchestration) ───────────────────────────────────────────

_FULL_CFG = {
    "psi_threshold": 0.2,
    "ks_pvalue_threshold": 0.05,
    "min_drift_features": 1,
    "archive_keep": 5,
    "auc_decay_threshold": 0.05,
    "model_dir": "models",
}


def test_check_and_retrain_force_check_only_skips_drift_and_train(tmp_path):
    """--force --check-only reports a trigger without ever calling train()."""
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch("models.train.train") as mock_train:
        result = check_and_retrain(force=True, check_only=True, model_dir=str(tmp_path))
    assert result is True
    mock_train.assert_not_called()


def test_check_and_retrain_no_reference_returns_false(tmp_path):
    """No baseline snapshot on disk → cannot measure drift → no retrain."""
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)):
        result = check_and_retrain(model_dir=str(tmp_path))
    assert result is False


def test_check_and_retrain_no_current_features_returns_false(tmp_path):
    """Reference exists but features.parquet is missing → no retrain."""
    _feature_df().to_parquet(tmp_path / "reference_features.parquet", index=False)
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch.object(rt, "_DEFAULT_FEATURES", str(tmp_path / "missing.parquet")):
        result = check_and_retrain(model_dir=str(tmp_path))
    assert result is False


def test_check_and_retrain_no_drift_returns_false_and_writes_report(tmp_path):
    """Same-scale current vs reference → no drift → no retrain, report still written."""
    cur = tmp_path / "current.parquet"
    _feature_df(seed=1).to_parquet(tmp_path / "reference_features.parquet", index=False)
    _feature_df(seed=2).to_parquet(cur, index=False)
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch.object(rt, "_DEFAULT_FEATURES", str(cur)), \
         patch("models.train.train") as mock_train:
        result = check_and_retrain(model_dir=str(tmp_path))
    assert result is False
    mock_train.assert_not_called()
    assert (tmp_path / "drift_report.json").exists()


def test_check_and_retrain_drift_triggers_retrain(tmp_path):
    """A large distribution shift drives PSI past threshold → train() runs."""
    cur = tmp_path / "current.parquet"
    _feature_df(seed=1, scale=1.0).to_parquet(tmp_path / "reference_features.parquet", index=False)
    _feature_df(seed=2, scale=50.0).to_parquet(cur, index=False)
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch.object(rt, "_DEFAULT_FEATURES", str(cur)), \
         patch("models.train.train", return_value={"targets": {}}) as mock_train:
        result = check_and_retrain(model_dir=str(tmp_path))
    assert result is True
    mock_train.assert_called_once()


def test_check_and_retrain_force_retrains(tmp_path):
    """--force skips drift entirely and retrains."""
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch("models.train.train", return_value={"targets": {}}) as mock_train:
        result = check_and_retrain(force=True, model_dir=str(tmp_path))
    assert result is True
    mock_train.assert_called_once()


def test_check_and_retrain_empty_matrix_aborts(tmp_path):
    """train() returning None (empty matrix) is reported as a failed retrain."""
    with patch.object(rt, "_load_cfg", return_value=dict(_FULL_CFG)), \
         patch("models.train.train", return_value=None) as mock_train:
        result = check_and_retrain(force=True, model_dir=str(tmp_path))
    assert result is False
    mock_train.assert_called_once()
