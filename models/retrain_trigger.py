"""Drift-based and on-demand model retraining trigger.

Usage:
    python -m models.retrain_trigger                # check drift, retrain if triggered
    python -m models.retrain_trigger --force        # always retrain
    python -m models.retrain_trigger --check-only   # report drift, no retrain

A reference snapshot of the training feature distribution is saved automatically
when models.train.train() completes. Drift is measured per-feature using PSI
(primary) and a two-sample KS test (secondary). When enough features exceed
their thresholds the trigger fires, archives the current models, retrains, and
logs any AUC decay versus the previous run.

Archive: <model_dir>/archive/<YYYYMMDDTHHMMSS>/  — keeps last N versions.
Drift report: <model_dir>/drift_report.json      — overwritten on every check.
"""
import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from models.features import FEATURE_COLS
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.timezone import now as tz_now

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_FEATURES = os.path.join(_BASE, "data", "features.parquet")


# ── config ─────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    raw = get_config()
    r = raw.get("retrain", {})
    m = raw.get("model", {})
    return {
        "psi_threshold": float(r.get("psi_threshold", 0.2)),
        "ks_pvalue_threshold": float(r.get("ks_pvalue_threshold", 0.05)),
        "min_drift_features": int(r.get("min_drift_features", 1)),
        "archive_keep": int(r.get("archive_keep", 5)),
        "auc_decay_threshold": float(r.get("auc_decay_threshold", 0.05)),
        "model_dir": str(m.get("model_dir", "models")),
    }


def _resolve_dir(model_dir: Optional[str], cfg: dict) -> str:
    raw = model_dir if model_dir is not None else cfg["model_dir"]
    return raw if os.path.isabs(raw) else os.path.join(_BASE, raw)


# ── drift detection ────────────────────────────────────────────────────────────

@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    drifted: bool


@dataclass
class DriftReport:
    features: list = field(default_factory=list)  # list[FeatureDrift]
    n_drifted: int = 0
    triggered: bool = False


def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index for one numeric feature. NaNs stripped before binning."""
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < bins or len(cur) == 0:
        return 0.0
    edges = np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    # small epsilon keeps probabilities positive (avoids log(0))
    eps = 1e-8
    ref_pct = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(cur) + eps * len(cur_counts))
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_drift(ref_df: pd.DataFrame, cur_df: pd.DataFrame, cfg: dict) -> DriftReport:
    """Compute per-feature PSI and KS statistics and flag drifted features."""
    report = DriftReport()
    for col in FEATURE_COLS:
        if col not in ref_df.columns or col not in cur_df.columns:
            continue
        ref_vals = pd.to_numeric(ref_df[col], errors="coerce").to_numpy(dtype=float)
        cur_vals = pd.to_numeric(cur_df[col], errors="coerce").to_numpy(dtype=float)

        psi_val = _psi(ref_vals, cur_vals)

        ref_clean = ref_vals[~np.isnan(ref_vals)]
        cur_clean = cur_vals[~np.isnan(cur_vals)]
        if len(ref_clean) >= 2 and len(cur_clean) >= 2:
            ks = ks_2samp(ref_clean, cur_clean)
            ks_stat: Optional[float] = round(float(ks.statistic), 6)
            ks_pvalue: Optional[float] = round(float(ks.pvalue), 8)
        else:
            ks_stat = ks_pvalue = None

        # PSI is the robust trigger. KS is computed and reported as a diagnostic
        # only: its two-sample p-value shrinks toward 0 as sample size grows, so a
        # KS-OR clause with min_drift_features=1 would fire on nearly every check
        # at scale regardless of true drift.
        drifted = psi_val > cfg["psi_threshold"]

        fd = FeatureDrift(
            feature=col,
            psi=round(psi_val, 6),
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            drifted=drifted,
        )
        report.features.append(fd)

        if drifted:
            logger.warning(
                "drift: %-28s  PSI=%6.4f  KS=%s  KS-p=%s  ← DRIFTED",
                col, psi_val,
                f"{ks_stat:.4f}" if ks_stat is not None else "   n/a",
                f"{ks_pvalue:.3e}" if ks_pvalue is not None else "       n/a",
            )
        else:
            logger.debug(
                "drift: %-28s  PSI=%6.4f  KS=%s  KS-p=%s",
                col, psi_val,
                f"{ks_stat:.4f}" if ks_stat is not None else "   n/a",
                f"{ks_pvalue:.3e}" if ks_pvalue is not None else "       n/a",
            )

    report.n_drifted = sum(1 for f in report.features if f.drifted)
    report.triggered = report.n_drifted >= cfg["min_drift_features"]
    return report


# ── reference snapshot ─────────────────────────────────────────────────────────

def save_reference_snapshot(df: pd.DataFrame, model_dir: str) -> None:
    """Persist FEATURE_COLS distribution as the drift baseline.

    Called by models.train.train() after a successful training run so the
    reference always matches the data the current models were trained on.
    """
    out_dir = model_dir if os.path.isabs(model_dir) else os.path.join(_BASE, model_dir)
    os.makedirs(out_dir, exist_ok=True)
    cols = [c for c in FEATURE_COLS if c in df.columns]
    path = os.path.join(out_dir, "reference_features.parquet")
    df[cols].to_parquet(path, index=False)
    logger.info(
        "retrain_trigger: saved reference snapshot (%d rows, %d features) → %s",
        len(df), len(cols), path,
    )


# ── model archiving ────────────────────────────────────────────────────────────

def _archive_current(out_dir: str, timestamp: str) -> None:
    """Copy all .bin / .json / .parquet model artifacts into archive/<timestamp>/."""
    dest = os.path.join(out_dir, "archive", timestamp)
    os.makedirs(dest, exist_ok=True)
    archived = 0
    for fname in os.listdir(out_dir):
        src = os.path.join(out_dir, fname)
        if os.path.isfile(src) and fname.rsplit(".", 1)[-1] in {"bin", "json", "parquet"}:
            shutil.copy2(src, os.path.join(dest, fname))
            archived += 1
    if archived:
        logger.info("retrain_trigger: archived %d artifacts → %s", archived, dest)


def _prune_archive(out_dir: str, keep: int) -> None:
    """Remove oldest archive runs beyond the keep limit."""
    archive_root = os.path.join(out_dir, "archive")
    if not os.path.isdir(archive_root):
        return
    runs = sorted(
        d for d in os.listdir(archive_root)
        if os.path.isdir(os.path.join(archive_root, d))
    )
    for old_run in runs[: max(0, len(runs) - keep)]:
        shutil.rmtree(os.path.join(archive_root, old_run))
        logger.info("retrain_trigger: pruned archive run %s", old_run)


# ── performance decay ──────────────────────────────────────────────────────────

def _log_performance_decay(old_meta: dict, new_meta: dict, decay_threshold: float) -> None:
    """Compare per-target AUC between the old and new meta dicts and log changes."""
    for target, new_t in new_meta.get("targets", {}).items():
        old_t = old_meta.get("targets", {}).get(target, {})
        old_auc = old_t.get("test_auc")
        new_auc = new_t.get("test_auc")
        if old_auc is None or new_auc is None:
            continue
        delta = new_auc - old_auc
        rel = delta / old_auc if old_auc else 0.0
        if rel < -decay_threshold:
            logger.warning(
                "retrain_trigger: [%s] AUC %.4f → %.4f (%+.1f%%) — "
                "performance decay exceeds %.0f%% threshold",
                target, old_auc, new_auc, 100 * rel, 100 * decay_threshold,
            )
        else:
            logger.info(
                "retrain_trigger: [%s] AUC %.4f → %.4f (%+.1f%%)",
                target, old_auc, new_auc, 100 * rel,
            )


# ── main entry point ───────────────────────────────────────────────────────────

def check_and_retrain(
    force: bool = False,
    check_only: bool = False,
    model_dir: Optional[str] = None,
) -> bool:
    """Check feature drift and retrain if triggered.

    Parameters
    ----------
    force      : skip drift check and always retrain
    check_only : report drift but do not retrain even if triggered
    model_dir  : override model_dir from config

    Returns
    -------
    bool — True if retraining ran (or was triggered in check_only mode)
    """
    cfg = _load_cfg()
    out_dir = _resolve_dir(model_dir, cfg)

    if not force:
        ref_path = os.path.join(out_dir, "reference_features.parquet")
        if not os.path.exists(ref_path):
            logger.warning(
                "retrain_trigger: no reference snapshot at %s — "
                "train a model first to establish the baseline", ref_path,
            )
            return False

        if not os.path.exists(_DEFAULT_FEATURES):
            logger.warning(
                "retrain_trigger: current features not found at %s — "
                "run build_training_matrix() first", _DEFAULT_FEATURES,
            )
            return False

        ref_df = pd.read_parquet(ref_path)
        cur_df = pd.read_parquet(_DEFAULT_FEATURES)
        # The reference snapshot is the labelled training split. features.parquet
        # holds the FULL matrix (incl. unlabelled live/future rows), so compare
        # like-for-like by restricting current to labelled runners.
        if "position" in cur_df.columns:
            cur_df = cur_df[cur_df["position"].notna()]
        logger.info(
            "retrain_trigger: drift check — reference %d rows, current %d rows",
            len(ref_df), len(cur_df),
        )

        drift = check_drift(ref_df, cur_df, cfg)
        logger.info(
            "retrain_trigger: %d/%d features drifted "
            "(PSI>%.2f or KS-p<%.3f) — triggered=%s",
            drift.n_drifted, len(drift.features),
            cfg["psi_threshold"], cfg["ks_pvalue_threshold"], drift.triggered,
        )

        report_path = os.path.join(out_dir, "drift_report.json")
        os.makedirs(out_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "timestamp": tz_now().strftime("%Y-%m-%dT%H:%M:%S"),
                    "n_drifted": drift.n_drifted,
                    "triggered": drift.triggered,
                    "features": [asdict(f) for f in drift.features],
                },
                fh, indent=2,
            )
        logger.info("retrain_trigger: drift report → %s", report_path)

        if not drift.triggered:
            logger.info("retrain_trigger: no significant drift — retraining not triggered")
            return False

        logger.info(
            "retrain_trigger: drift detected (%d features) — triggering retrain",
            drift.n_drifted,
        )

    else:
        logger.info("retrain_trigger: --force — skipping drift check")

    if check_only:
        logger.info("retrain_trigger: --check-only — skipping retrain")
        return True

    # ── archive → retrain ───────────────────────────────────────────────────
    ts = tz_now().strftime("%Y%m%dT%H%M%S")
    _archive_current(out_dir, ts)
    _prune_archive(out_dir, cfg["archive_keep"])

    # read old metrics before train() overwrites catboost_v3_meta.json
    old_meta: dict = {}
    old_meta_path = os.path.join(out_dir, "catboost_v3_meta.json")
    if os.path.exists(old_meta_path):
        try:
            with open(old_meta_path, encoding="utf-8") as fh:
                old_meta = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("retrain_trigger: could not read old meta: %s", exc)

    logger.info("retrain_trigger: retraining (run %s)", ts)
    from models.train import train  # lazy import — avoids circular dependency at load time
    new_meta = train(model_dir=out_dir)

    if new_meta is None:
        logger.error(
            "retrain_trigger: retraining aborted — training matrix is empty "
            "(no finishing positions available yet)"
        )
        return False

    _log_performance_decay(old_meta, new_meta, cfg["auc_decay_threshold"])
    logger.info("retrain_trigger: done (run %s)", ts)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check feature drift and auto-retrain CatBoost models"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip drift check and retrain unconditionally",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="Report drift but do not retrain even if triggered",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Override model_dir from config",
    )
    args = parser.parse_args()
    check_and_retrain(force=args.force, check_only=args.check_only, model_dir=args.model_dir)


if __name__ == "__main__":
    main()
