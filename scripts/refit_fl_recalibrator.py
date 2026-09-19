"""Refit the favourite-longshot recalibrator with SIGMOID bands (Stage 4).

Why: the shipped isotonic-band artifact (`models/fl_oddsband_v3nf_calib.pkl`,
2026-06-17) clamps below-support inputs to exact 0.0 (and one band to exact
1.0) — 20 of 356 live-cache `value_win_prob` values were literal zeros, and on
the walk-forward validation folds the zero-claims cost it race log-loss 1.7432
vs 1.7052 for sigmoid bands (equal Brier, zero extreme emissions). The band
method was selected on VALIDATION FOLDS ONLY — see
``data/audit/stage4/selection.json::fl_band_method_selection``.

Fit set: the frozen v3nf model's own out-of-sample span — panel rows strictly
after its training cutoff (2025-12-02, the 80% chronological split of the June
matrix) and strictly before the Stage-4 final test window (2026-06-13). The
model never trained on these rows, and the final window never enters any fit.

The previous artifact is backed up (with SHA-256 recorded) before the swap; a
meta sidecar makes the new artifact reproducible.

Usage:
    python -m scripts.refit_fl_recalibrator            # fit + backup + swap
    python -m scripts.refit_fl_recalibrator --dry-run  # fit + report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE))

from audit.walkforward import _matrix  # noqa: E402
from models.calibration import OddsBandCalibrator  # noqa: E402
from utils.logger import get_logger  # noqa: E402

logger = get_logger("scripts.refit_fl_recalibrator")

ARTIFACT = _BASE / "models" / "fl_oddsband_v3nf_calib.pkl"
META_OUT = _BASE / "models" / "fl_oddsband_v3nf_calib_meta.json"
BACKUP_DIR = _BASE / "data" / "backups"
PANEL = _BASE / "data" / "audit" / "stage4" / "panel.parquet"

FIT_START = "2025-12-03"     # strictly after the frozen v3nf train cutoff
FIT_END = "2026-06-12"       # strictly before the Stage-4 final test window
V3NF_TRAIN_CUTOFF = "2025-12-02"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="scripts.refit_fl_recalibrator")
    ap.add_argument("--dry-run", action="store_true",
                    help="fit and report, but do not touch the artifact")
    args = ap.parse_args(argv)

    from catboost import CatBoostClassifier

    if not PANEL.exists():
        raise SystemExit(f"audit panel not found at {PANEL}; run "
                         "`python -m scripts.calibration_audit --run stage4 "
                         "--phase panel` first")

    panel = pd.read_parquet(PANEL)
    rd = pd.to_datetime(panel["race_date"], utc=True, errors="coerce")
    fit = panel.loc[(rd >= pd.to_datetime(FIT_START, utc=True))
                    & (rd <= pd.to_datetime(FIT_END, utc=True))].copy()
    logger.info("refit_fl: %d OOS rows in fit window %s..%s",
                len(fit), FIT_START, FIT_END)

    meta = json.loads((_BASE / "models" / "catboost_v3nf_meta.json")
                      .read_text(encoding="utf-8"))
    cols = list(meta.get("feature_cols") or [])

    model = CatBoostClassifier()
    model.load_model(str(_BASE / "models" / "catboost_won_v3nf.bin"))
    with open(_BASE / "models" / "catboost_won_v3nf_calib.pkl", "rb") as fh:
        base_cal = pickle.load(fh)

    raw = np.asarray(model.predict_proba(_matrix(fit, cols))[:, 1], dtype=float)
    ind = np.asarray(base_cal.predict(raw), dtype=float)
    odds = pd.to_numeric(fit["bet_price"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(fit["won"], errors="coerce").fillna(0).astype(int).to_numpy()

    fl = OddsBandCalibrator.fit(ind, odds, y, method="sigmoid", min_rows=400)

    # Sanity: sigmoid bands can never emit exact 0/1.
    grid_p = np.linspace(0.001, 0.999, 200)
    for d in (1.5, 2.0, 3.0, 5.0, 10.0, 30.0, 80.0):
        out = np.asarray(fl.predict(grid_p, np.full_like(grid_p, d)), float)
        assert (out > 0.0).all() and (out < 1.0).all(), f"extreme at odds {d}"

    n_bands = len(fl.centers)
    print(f"fitted sigmoid-band OBC: {n_bands} bands on {len(fit)} rows "
          f"({FIT_START}..{FIT_END}); no exact 0/1 on the sanity grid")

    if args.dry_run:
        print("dry run — artifact untouched")
        return 0

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_DIR / "fl_oddsband_v3nf_calib_isotonic_20260617.pkl"
    old_sha = None
    if ARTIFACT.exists():
        if not backup.exists():
            shutil.copy2(ARTIFACT, backup)
        old_sha = _sha256(ARTIFACT)

    with open(ARTIFACT, "wb") as fh:
        pickle.dump(fl, fh)
    assert b"numpy" not in ARTIFACT.read_bytes(), "artifact must stay portable"

    META_OUT.write_text(json.dumps({
        "artifact": ARTIFACT.name,
        "refit_date": str(date.today()),
        "method": "sigmoid",
        "band_count": n_bands,
        "fit_window": {"start": FIT_START, "end": FIT_END},
        "fit_rows": int(len(fit)),
        "model_train_cutoff": V3NF_TRAIN_CUTOFF,
        "fit_is_oos_for_model": True,
        "panel_source": str(PANEL),
        "selection_evidence": "data/audit/stage4/selection.json"
                              "::fl_band_method_selection",
        "previous_artifact": {"backup": str(backup), "sha256": old_sha,
                              "method": "isotonic",
                              "defect": "exact-0.0 floor in every band; "
                                        "exact-1.0 ceiling in odds-on bands"},
        "new_sha256": _sha256(ARTIFACT),
        "command": "python -m scripts.refit_fl_recalibrator",
    }, indent=2), encoding="utf-8")
    print(f"swapped {ARTIFACT.name} (backup: {backup.name}); meta: {META_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
