"""CLI entry point and programmatic API for training CatBoost race models.

Usage:
    python -m models.train [--targets won placed_2 showed]
                           [--trials N] [--no-tune] [--model-dir PATH]

The train() function accepts an optional pre-loaded df so callers (including
tests) never need to touch disk.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features.builder import build_training_matrix
from models.calibration import fit_calibrator
from models.evaluate import report
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.retrain_trigger import save_reference_snapshot
from models.split_utils import chronological_group_split
from models.targets import add_targets
from models.tuner import _hardware_params, run_study
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# Below these row counts a time-ordered calibration slice is too small to fit a
# trustworthy isotonic curve, so calibration is skipped (model still trains).
_MIN_CALIB_ROWS = 200
_MIN_CORE_ROWS = 200


def _resolve_task_type(requested: str) -> str:
    """Return 'GPU' only if a CatBoost-visible GPU exists, else 'CPU'.

    Lets config default to GPU on this machine while staying portable: CI and
    GPU-less machines transparently fall back to CPU instead of crashing.
    """
    if str(requested).upper() != "GPU":
        return "CPU"
    try:
        from catboost.utils import get_gpu_device_count

        if get_gpu_device_count() > 0:
            return "GPU"
        logger.warning("train: task_type=GPU requested but no GPU detected — using CPU")
    except Exception as exc:  # noqa: BLE001
        logger.warning("train: GPU detection failed (%s) — using CPU", exc)
    return "CPU"


def _load_existing_params(out_dir: str, vtag: str) -> dict:
    """Read best_params per target from a prior meta.json (for --reuse-params)."""
    path = os.path.join(out_dir, f"catboost_{vtag}_meta.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        return {t: d.get("best_params", {}) or {} for t, d in meta.get("targets", {}).items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("train: --reuse-params: could not read %s (%s)", path, exc)
        return {}


def _load_model_cfg():
    m = get_config().get("model", {})
    return {
        "test_size": float(m.get("test_size", 0.2)),
        "cv_folds": int(m.get("cv_folds", 5)),
        "optuna_trials": int(m.get("optuna_trials", 50)),
        "early_stopping_rounds": int(m.get("early_stopping_rounds", 50)),
        "iterations": int(m.get("iterations", 1000)),
        "show_positions": int(m.get("show_positions", 3)),
        "targets": list(m.get("targets", ["won", "placed_2", "showed"])),
        "model_dir": str(m.get("model_dir", "models")),
        "max_sample_weight": float(m.get("max_sample_weight", 20.0)),
        "version_tag": str(m.get("version_tag", "v3")),
        "calibration_method": str(m.get("calibration_method", "isotonic")),
        "calibration_size": float(m.get("calibration_size", 0.15)),
        "task_type": str(m.get("task_type", "CPU")),
        "devices": str(m.get("devices", "0")),
        "thread_count": int(m.get("thread_count", -1)),
        "random_seed": int(m.get("random_seed", 42)),
        "tuning": dict(m.get("tuning", {}) or {}),
    }


def _time_split(df, test_size, group_col="race_uid", date_col="race_date"):
    """Whole-race chronological split: last test_size fraction of ROWS -> test,
    rounded to whole race_uid groups so no race straddles the boundary.

    Row-count slicing on date-sorted rows (the pre-step-02 behaviour) can cut a
    single race's rows in half whenever several races share a race_date or a
    race's own rows aren't perfectly contiguous after the sort. Cutting on
    group boundaries instead (see models.split_utils) guarantees zero shared
    race_uid between the two returned frames, independent of input row order.
    """
    df = df.reset_index(drop=True)
    left_idx, right_idx = chronological_group_split(
        df[group_col].to_numpy(), df[date_col].to_numpy(), test_size)
    return df.iloc[left_idx].copy(), df.iloc[right_idx].copy()


def _group_carve(df, frac, group_col="race_uid", date_col="race_date"):
    """Whole-race split of an already-filtered frame; see _time_split."""
    df = df.reset_index(drop=True)
    left_idx, right_idx = chronological_group_split(
        df[group_col].to_numpy(), df[date_col].to_numpy(), frac)
    return df.iloc[left_idx].copy(), df.iloc[right_idx].copy()


def _sample_weights(df, max_weight):
    """Odds-inverse weights clipped to [1, max_weight]. NaN implied_prob -> 1.0."""
    ip = pd.to_numeric(df["implied_prob"], errors="coerce")
    ip = ip.fillna(1.0)                       # NaN -> neutral weight
    ip = ip.clip(lower=1.0 / max_weight)      # floor avoids weight > max_weight
    return (1.0 / ip).clip(upper=max_weight).to_numpy(dtype=float)


def train(df=None, targets=None, trials=None, no_tune=False, model_dir=None,
          feature_cols=None, version_tag=None, reuse_params=False):
    """Train CatBoost models and save artefacts.

    Parameters
    ----------
    df           : pre-loaded labelled DataFrame. If None, calls build_training_matrix().
    targets      : list of target names to train. Defaults to config value.
    trials       : override optuna_trials from config.
    no_tune      : if True, skip Optuna and use CatBoost defaults.
    model_dir    : override model_dir from config.
    feature_cols : input feature whitelist. Defaults to models.features.FEATURE_COLS.
                   Pass PRICE_FREE_FEATURE_COLS to train the price-free value-betting
                   variant. The chosen list is persisted in meta["feature_cols"], so the
                   predictor loads the matching columns automatically.
    version_tag  : filename tag for .bin / meta.json artefacts. Defaults to config value
                   (e.g. "v3"). Pass "v3nf" to write a parallel model without touching v3.
    reuse_params : if True, skip Optuna and reuse best_params from the existing
                   meta.json for this version_tag (refits + recalibrates fast).

    Returns
    -------
    dict of metadata, or None if the training matrix is empty.
    """
    cfg = _load_model_cfg()
    if targets is not None:
        cfg["targets"] = targets
    if trials is not None:
        cfg["optuna_trials"] = trials
    if model_dir is not None:
        cfg["model_dir"] = model_dir
    if version_tag is not None:
        cfg["version_tag"] = version_tag

    # Resolve GPU→CPU once; the effective value flows to the tuner and final fit.
    cfg["task_type"] = _resolve_task_type(cfg["task_type"])
    logger.info("train: task_type=%s", cfg["task_type"])

    calib_method = cfg["calibration_method"].strip().lower()
    calibrate = calib_method in ("isotonic", "sigmoid", "auto")
    calib_frac = cfg["calibration_size"]

    feat_cols = list(FEATURE_COLS if feature_cols is None else feature_cols)

    if df is None:
        logger.info("train: loading training matrix from disk")
        df = build_training_matrix(write=False)

    if len(df) == 0:
        logger.error(
            "train: training matrix is empty — no labelled data (finishing positions) "
            "available yet. Fetch historical results first."
        )
        return None

    df = add_targets(df, cfg["show_positions"])

    # WIN and PLACE are different price books (DESIGN.md); live inference only
    # ever serves WIN-market rows (utils/market_validation.LIVE_SOURCES emit
    # market_type=WIN exclusively), so a won/placed_2/showed model trained on a
    # mix of WIN- and PLACE-priced rows would see a feature distribution at
    # train time it never sees at serve time — and would silently disagree with
    # models.train_lgbm's WIN-only population on the same "eligible race" set.
    # Restrict to WIN rows for all three targets; PLACE rows stay in the source
    # feature matrix for place-market evaluation elsewhere (e.g. backtest/).
    if "market_type" in df.columns:
        n_before = len(df)
        df = df[df["market_type"].astype("string").str.upper() == "WIN"].copy()
        logger.info(
            "train: restricted to WIN-market rows (%d -> %d rows) to match "
            "live-serving and models.train_lgbm's eligible population",
            n_before, len(df))

    # Odds-inverse sample weights, attached as a column so they travel with their
    # rows through _time_split's race_date re-sort. Positional slicing of a
    # separate weight array would otherwise pair each row with another row's weight.
    df["_sample_weight"] = _sample_weights(df, cfg["max_sample_weight"])

    train_df, test_df = _time_split(df, cfg["test_size"])
    n_train = len(train_df)

    # Use model_dir as-is if absolute, otherwise join with _BASE
    if os.path.isabs(cfg["model_dir"]):
        out_dir = cfg["model_dir"]
    else:
        out_dir = os.path.join(_BASE, cfg["model_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # filter the chosen whitelist to columns actually present in the df
    avail_cols = [c for c in feat_cols if c in df.columns]
    missing = set(feat_cols) - set(avail_cols)
    if missing:
        logger.warning("train: %d FEATURE_COLS not in df, skipping: %s",
                       len(missing), sorted(missing))

    meta = {
        "targets": {},
        "feature_cols": avail_cols,
        "train_rows": n_train,
        "test_rows": len(test_df),
    }

    stored_params = _load_existing_params(out_dir, cfg["version_tag"]) if reuse_params else {}

    for target in cfg["targets"]:
        if target not in train_df.columns:
            logger.warning("train: target '%s' not in dataframe, skipping", target)
            continue

        logger.info("train: ── target=%s ──", target)

        # drop rows where this target is null
        tr_sub = train_df.loc[train_df[target].notna()].reset_index(drop=True)
        te_sub = test_df.loc[test_df[target].notna()].reset_index(drop=True)

        def _xy(frame):
            X = frame[avail_cols].fillna(np.nan).astype(float).values
            y = frame[target].astype(int).values
            w = frame["_sample_weight"].to_numpy(dtype=float)
            return X, y, w

        X_te, y_te, _w_te = _xy(te_sub)
        ip_te = pd.to_numeric(te_sub["implied_prob"], errors="coerce").to_numpy(dtype=float)

        # Carve a time-ordered calibration slice off the most-recent whole
        # races of train (closest to the live distribution). Cut on race_uid
        # group boundaries (models.split_utils), never mid-race, so tuning and
        # the final fit can never see a race whose other rows calibrate it.
        # Skipped when either resulting slice is too small to trust.
        if calibrate and len(tr_sub) > 0:
            core_df, cal_df = _group_carve(tr_sub, calib_frac)
        else:
            core_df, cal_df = tr_sub, tr_sub.iloc[0:0]
        if len(cal_df) < _MIN_CALIB_ROWS or len(core_df) < _MIN_CORE_ROWS:
            core_df, cal_df = tr_sub, tr_sub.iloc[0:0]
        n_cal = len(cal_df)

        X_core, y_core, w_core = _xy(core_df)
        if n_cal > 0:
            X_cal, y_cal, _w_cal = _xy(cal_df)
        else:
            X_cal = y_cal = None

        if no_tune:
            best_params = {}
            logger.info("train: --no-tune: using CatBoost defaults")
        elif reuse_params and target in stored_params:
            best_params = stored_params[target]
            logger.info("train: reusing stored params for %s (skipping Optuna): %s",
                        target, best_params)
        else:
            # Tuning CV must never touch the calibration or outer-test rows —
            # it only ever sees core_df (train minus the calibration carve).
            logger.info("train: running Optuna (%d trials)", cfg["optuna_trials"])
            best_params = run_study(
                X_core, y_core, w_core, cfg,
                race_ids=core_df["race_uid"].to_numpy(),
                order_keys=core_df["race_date"].to_numpy(),
            )

        # fit final model; hold out the most-recent whole races (~10%) of the
        # core for early stopping — same group-safe carve as calibration.
        fit_df, es_df = _group_carve(core_df, 0.10)
        X_fit, y_fit, w_fit = _xy(fit_df)
        if len(es_df) > 0:
            X_es, y_es, _w_es = _xy(es_df)
        else:
            # Only when core itself is a single race (can't be split further);
            # fall back to fitting without an eval_set rather than crashing.
            X_es = y_es = None
            logger.warning("train: %s: core has no separable early-stop slice "
                            "(single race group) — fitting without eval_set", target)

        final_params = {
            "iterations": cfg["iterations"],
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "early_stopping_rounds": cfg["early_stopping_rounds"],
            "verbose": False,
            "allow_writing_files": False,
            **_hardware_params(cfg),
            **best_params,
        }
        # Class-imbalance handling is now a tuned choice (auto_class_weights or
        # scale_pos_weight may arrive via best_params). Default to "Balanced"
        # only when tuning picked neither — preserves --no-tune / older
        # --reuse-params behaviour without double-setting (CatBoost rejects both).
        if "auto_class_weights" not in final_params and "scale_pos_weight" not in final_params:
            final_params["auto_class_weights"] = "Balanced"
        model = CatBoostClassifier(**final_params)
        if X_es is not None:
            model.fit(X_fit, y_fit, sample_weight=w_fit,
                      eval_set=(X_es, y_es), verbose=False)
        else:
            model.fit(X_fit, y_fit, sample_weight=w_fit, verbose=False)

        vtag = cfg["version_tag"]

        # Fit + persist a calibrator on the held-out slice (needs both classes
        # present). Maps raw→true-frequency; the chosen method is recorded in
        # meta. "auto" compares isotonic vs sigmoid on a sub-fold (see
        # models.calibration). Monotone either way, so AUC/ranking is preserved.
        calibrator = None
        method_used = None
        if X_cal is not None and len(np.unique(y_cal)) > 1:
            raw_cal = model.predict_proba(X_cal)[:, 1]
            calibrator, method_used = fit_calibrator(calib_method, raw_cal, y_cal)
            cal_path = os.path.join(out_dir, f"catboost_{target}_{vtag}_calib.pkl")
            with open(cal_path, "wb") as fh:
                pickle.dump(calibrator, fh)
            logger.info("train: saved %s calibrator %s (n_cal=%d)",
                        method_used, cal_path, n_cal)
        elif calibrate:
            logger.warning("train: calibration skipped for %s (insufficient slice)", target)

        metrics = report(model, X_te, y_te, ip_te, avail_cols, target,
                         calibrator=calibrator)

        model_path = os.path.join(out_dir, f"catboost_{target}_{vtag}.bin")
        model.save_model(model_path)
        logger.info("train: saved %s", model_path)

        meta["targets"][target] = {
            "best_params": best_params,
            "test_auc": metrics["auc"],
            "test_brier": metrics["brier"],
            "brier_baseline": metrics["brier_baseline"],
            "band_aucs": metrics["band_aucs"],
            "ae_by_band": metrics["ae_by_band"],
            "reliability": metrics["reliability"],
            "calibrated": calibrator is not None,
            "calibration_method": method_used,
        }

    meta_path = os.path.join(out_dir, f"catboost_{cfg['version_tag']}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("train: saved meta to %s", meta_path)

    save_reference_snapshot(train_df, out_dir)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost race outcome models (won / placed_2 / showed)"
    )
    parser.add_argument(
        "--targets", nargs="+", choices=["won", "placed_2", "showed"], default=None,
        help="Subset of targets to train (default: all from config)"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override optuna_trials from config (e.g. 5 for a smoke test)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip Optuna; use CatBoost default hyperparameters"
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Override model_dir from config"
    )
    parser.add_argument(
        "--price-free", action="store_true",
        help="Train the value-betting variant: drop all market-price features "
             "(implies --version-tag v3nf unless overridden)"
    )
    parser.add_argument(
        "--version-tag", default=None,
        help="Override version_tag from config (filename tag for artefacts)"
    )
    parser.add_argument(
        "--reuse-params", action="store_true",
        help="Skip Optuna; reuse best_params from the existing meta.json for this "
             "version-tag (fast refit + recalibrate)"
    )
    args = parser.parse_args()

    feature_cols = None
    version_tag = args.version_tag
    if args.price_free:
        feature_cols = PRICE_FREE_FEATURE_COLS
        if version_tag is None:
            version_tag = "v3nf"

    result = train(
        targets=args.targets,
        trials=args.trials,
        no_tune=args.no_tune,
        model_dir=args.model_dir,
        feature_cols=feature_cols,
        version_tag=version_tag,
        reuse_params=args.reuse_params,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
