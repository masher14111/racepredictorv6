"""Re-baseline held-out metrics after the C2 leakage fix (morningwap market features).

Mirrors models.train's chronological split (sort by race_date, last test_size ->
test) and the persisted feature_cols/calibrators, then recomputes AUC, log-loss,
Brier and ECE per (version_tag, target) on the same held-out tail used by the
model-09 audit. Read-only: loads the saved .bin models and *_calib.pkl, never
retrains. Run AFTER models.train has been re-run on the regenerated matrix.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from models.calibration import brier_score
from models.targets import add_targets

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_DIR = os.path.join(_BASE, "models")
_TRAIN = os.path.join(_BASE, "data", "features", "training.parquet")
TARGETS = ["won", "placed_2", "showed"]
TEST_SIZE = 0.20


def _ece(y, p, n_bins=10):
    """Expected calibration error: weighted mean |mean_pred - obs_freq| over bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = len(y)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _load(vtag, target):
    bin_path = os.path.join(_MODEL_DIR, f"catboost_{target}_{vtag}.bin")
    model = CatBoostClassifier()
    model.load_model(bin_path)
    calib = None
    cal_path = os.path.join(_MODEL_DIR, f"catboost_{target}_{vtag}_calib.pkl")
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as fh:
            calib = pickle.load(fh)
    return model, calib


def _feature_cols(vtag):
    with open(os.path.join(_MODEL_DIR, f"catboost_{vtag}_meta.json"), encoding="utf-8") as fh:
        return json.load(fh)["feature_cols"]


def main():
    df = pd.read_parquet(_TRAIN)
    df = add_targets(df, 3)
    df = df.sort_values("race_date").reset_index(drop=True)
    split = int(len(df) * (1.0 - TEST_SIZE))
    test = df.iloc[split:].copy()
    print(f"held-out tail: {len(test)} rows  "
          f"{test['race_date'].min()} -> {test['race_date'].max()}\n")

    base = {t: float(pd.to_numeric(test[t], errors="coerce").mean()) for t in TARGETS}
    print("base rates:", {t: round(base[t], 4) for t in TARGETS}, "\n")

    hdr = f"{'vtag':5s} {'target':9s} {'AUC':>6s} {'logloss':>8s} {'Brier':>7s} {'ECE':>7s} {'calib':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for vtag in ("v3", "v3nf"):
        cols = _feature_cols(vtag)
        for target in TARGETS:
            model, calib = _load(vtag, target)
            mask = pd.to_numeric(test[target], errors="coerce").notna().values
            X = test.loc[mask, cols].astype(float).values
            y = pd.to_numeric(test.loc[mask, target], errors="coerce").astype(int).values
            p = model.predict_proba(X)[:, 1]
            if calib is not None:
                p = np.asarray(calib.predict(p), dtype=float)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            auc = roc_auc_score(y, p)
            ll = log_loss(y, p, labels=[0, 1])
            br = brier_score(y, p)
            ece = _ece(y, p)
            print(f"{vtag:5s} {target:9s} {auc:6.3f} {ll:8.3f} {br:7.3f} {ece:7.3f} "
                  f"{'yes' if calib is not None else 'NO':>6s}")
    print()


if __name__ == "__main__":
    main()
