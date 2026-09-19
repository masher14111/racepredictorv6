"""Step 13: chronological out-of-fold predictions for CatBoost, LightGBM and
XGBoost over dev_core (whole-race walk-forward, models.split_utils.
group_time_series_split), used ONLY to select simple ensemble-blend weights
(blend_dev window) and to fit the blended-probability calibrator (calib
window) — never the eval panel, never the final holdout.

Economisation (disclosed): hyperparameters are FIXED per model (the already-
tuned CatBoost/XGBoost params from steps 10/13's own dev_core fits; LightGBM's
existing untuned defaults) and NOT re-tuned per fold, and each fold's booster
uses a capped, fixed round count (no early stopping) — a full nested re-tune
across 3 GBDT families x 6 folds x 2 branches is outside this stage's CPU
budget, the same economisation step 12 disclosed for its own walk-forward
panel (reports/improvement/12/blend_manifest.json "disclosed_limitation").

Run: .venv/Scripts/python.exe reports/improvement/13/03_oof_blend_dev.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from features.lgbm_adapter import FINAL_FEATURE_COLS, INDEPENDENT_FEATURE_COLS, build_lgbm_matrix
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.lgbm_softmax import LGBMSoftmaxModel
from models.split_utils import group_time_series_split
from models.targets import add_targets
from models.train import _sample_weights
from models.tuner import _hardware_params as cb_hardware_params

MANIFEST = "reports/improvement/13/manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
OUT_PARQUET = "data/audit/13/oof_dev_core.parquet"
OUT_META = "reports/improvement/13/oof_meta.json"

N_SPLITS = 6
FOLD_ROUNDS = 500          # fixed, no early stopping (see module docstring)
RNG = 42

CB_META = {
    "indep": "data/audit/10/models_catboost/catboost_s10idp_meta.json",
    "market": "data/audit/10/models_catboost/catboost_s10mkt_meta.json",
}
XGB_META = {
    "indep": "data/audit/13/models_xgboost/xgboost_s13idp_meta.json",
    "market": "data/audit/13/models_xgboost/xgboost_s13mkt_meta.json",
}

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def _best_params(path: str) -> dict:
    meta = json.load(open(path, "r", encoding="utf-8"))
    return dict(meta["targets"]["won"]["best_params"])


def load_core(manifest: dict) -> pd.DataFrame:
    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    win = win[win["_day"].notna()].reset_index(drop=True)
    dev_core_end = pd.Timestamp(manifest["windows"]["dev_core"]["end_exclusive"])
    dev_core = win[win["_day"] < dev_core_end].reset_index(drop=True)
    del win, df

    dev_core = add_targets(dev_core, 3)
    keep = dev_core["won"].notna().to_numpy()
    core = dev_core.loc[keep].reset_index(drop=True)
    log(f"dev_core (labelled WIN rows): {len(core)} rows / {core['race_uid'].nunique()} races")
    return core


def _cb_predict(train_idx, val_idx, X, y, w, best_params):
    cfg = {"task_type": "GPU", "devices": "0", "thread_count": -1}
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() <= 0:
            cfg["task_type"] = "CPU"
    except Exception:
        cfg["task_type"] = "CPU"
    params = {
        "iterations": FOLD_ROUNDS, "loss_function": "Logloss", "verbose": False,
        "allow_writing_files": False, "random_seed": RNG,
        **cb_hardware_params(cfg), **best_params,
    }
    if "auto_class_weights" not in params and "scale_pos_weight" not in params:
        params["auto_class_weights"] = "Balanced"
    m = CatBoostClassifier(**params)
    m.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx], verbose=False)
    return m.predict_proba(X[val_idx])[:, 1]


def _xgb_predict(train_idx, val_idx, X, y, w, best_params):
    pos = float(np.sum(y[train_idx] == 1))
    neg = float(np.sum(y[train_idx] == 0))
    spw = (neg / pos) if pos > 0 else 1.0
    m = XGBClassifier(
        n_estimators=FOLD_ROUNDS, eval_metric="logloss", scale_pos_weight=spw,
        tree_method="hist", n_jobs=-1, random_state=RNG, verbosity=0, **best_params,
    )
    m.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx], verbose=False)
    return m.predict_proba(X[val_idx])[:, 1]


def _lgbm_predict(train_idx, val_idx, X_df, y, race_uid):
    m = LGBMSoftmaxModel(n_estimators=FOLD_ROUNDS, num_threads=-1, seed=RNG)
    m.fit(X_df.iloc[train_idx].reset_index(drop=True), y[train_idx], race_uid[train_idx])
    return m.predict_proba(X_df.iloc[val_idx].reset_index(drop=True), race_uid[val_idx])


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    core = load_core(manifest)

    y = core["won"].astype(int).to_numpy()
    w = _sample_weights(core, 20.0)
    race_uid = core["race_uid"].to_numpy()
    race_date = core["race_date"].to_numpy()

    X_cb_indep = core[PRICE_FREE_FEATURE_COLS].astype(float).to_numpy()
    X_cb_market = core[FEATURE_COLS].astype(float).to_numpy()
    X_xgb_indep = X_cb_indep
    X_xgb_market = X_cb_market

    X_lgbm_indep, y_lgbm_indep, rid_lgbm_indep = build_lgbm_matrix(
        core, inference=False, feature_cols=INDEPENDENT_FEATURE_COLS)
    X_lgbm_market, y_lgbm_market, rid_lgbm_market = build_lgbm_matrix(
        core, inference=False, feature_cols=FINAL_FEATURE_COLS)
    assert len(X_lgbm_indep) == len(core) and np.array_equal(rid_lgbm_indep, race_uid)
    assert len(X_lgbm_market) == len(core) and np.array_equal(rid_lgbm_market, race_uid)

    cb_params = {b: _best_params(CB_META[b]) for b in ("indep", "market")}
    xgb_params = {b: _best_params(XGB_META[b]) for b in ("indep", "market")}
    log(f"loaded fixed hyperparams: catboost={cb_params} xgboost={xgb_params}")

    oof = {k: np.full(len(core), np.nan) for k in
          ("cb_indep", "cb_market", "xgb_indep", "xgb_market", "lgbm_indep", "lgbm_market")}
    covered = np.zeros(len(core), dtype=bool)

    folds = list(group_time_series_split(race_uid, race_date, N_SPLITS))
    for i, (train_idx, val_idx) in enumerate(folds):
        if len(train_idx) == 0 or len(val_idx) == 0 or len(np.unique(y[train_idx])) < 2:
            log(f"fold {i+1}/{N_SPLITS}: skipped (degenerate)")
            continue
        log(f"fold {i+1}/{N_SPLITS}: train={len(train_idx)} val={len(val_idx)}")
        oof["cb_indep"][val_idx] = _cb_predict(train_idx, val_idx, X_cb_indep, y, w, cb_params["indep"])
        oof["cb_market"][val_idx] = _cb_predict(train_idx, val_idx, X_cb_market, y, w, cb_params["market"])
        oof["xgb_indep"][val_idx] = _xgb_predict(train_idx, val_idx, X_xgb_indep, y, w, xgb_params["indep"])
        oof["xgb_market"][val_idx] = _xgb_predict(train_idx, val_idx, X_xgb_market, y, w, xgb_params["market"])
        oof["lgbm_indep"][val_idx] = _lgbm_predict(train_idx, val_idx, X_lgbm_indep, y, race_uid)
        oof["lgbm_market"][val_idx] = _lgbm_predict(train_idx, val_idx, X_lgbm_market, y, race_uid)
        covered[val_idx] = True
        log(f"fold {i+1}/{N_SPLITS}: done")

    out = pd.DataFrame({
        "race_uid": race_uid, "horse_id": core["horse_id"].to_numpy(),
        "race_date": race_date, "won": y, "covered": covered,
        **oof,
    })
    os.makedirs(os.path.dirname(OUT_PARQUET), exist_ok=True)
    out.to_parquet(OUT_PARQUET, engine="pyarrow", index=False)
    log(f"wrote {OUT_PARQUET} ({len(out)} rows, covered={int(covered.sum())})")

    meta = {
        "n_splits": N_SPLITS, "fold_rounds": FOLD_ROUNDS,
        "n_rows": int(len(core)), "n_covered": int(covered.sum()),
        "catboost_params": cb_params, "xgboost_params": xgb_params,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }
    with open(OUT_META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log(f"wrote {OUT_META}")
    print("OOF_BLEND_DEV_13_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
