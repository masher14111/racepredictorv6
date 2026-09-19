"""Step 10: freeze the run manifest BEFORE any training happens.

Records data/config/code hashes, the eligible racing population, the intended
decision cutoff, the development folds, the calibration window and the
untouched final-window policy. Writes reports/improvement/10/run_manifest.json.

Must be run AFTER 01_build_matrix.py (needs data/audit/10/training_rebuilt.parquet)
and BEFORE 03_run_baselines.py (which reads this manifest's frozen windows and
must not re-derive them from anything past-the-cutoff).

Run: .venv/Scripts/python.exe reports/improvement/10/02_freeze_manifest.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
UNIFIED_PATH = "data/unified_races.parquet"
MANIFEST_PATH = "reports/improvement/10/run_manifest.json"

# Code that defines the population/feature/split/evaluation contract this
# manifest freezes. Hashing these pins the exact logic this run's numbers were
# produced with; a later stage that changes any of them must not silently
# claim to reproduce step 10's scorecard.
CODE_FILES = [
    "features/builder.py", "features/derive.py", "features/engine.py",
    "features/_trailing_fast.py", "features/_trainer_form.py",
    "features/_jockey_form.py", "features/fuse.py", "features/labels.py",
    "features/lgbm_adapter.py",
    "models/features.py", "models/split_utils.py", "models/train.py",
    "models/train_lgbm.py", "models/tuner.py", "models/lgbm_softmax.py",
    "models/conditional_logit.py", "models/head_to_head.py", "models/devig.py",
    "models/calibration.py", "backtest/data.py", "backtest/holdout.py",
]

# Final-holdout length matches the existing train_lgbm convention
# (models.train_lgbm._HOLDOUT_DAYS = 20, "the most recent ~3 weeks") so the
# frozen policy here is consistent with infrastructure already in the repo,
# not a bespoke number invented for this stage.
FINAL_HOLDOUT_DAYS = 20
DEV_OOS_DAYS = 20
# The step-04 stage-4 final evaluation already observed 2026-06-13..2026-07-25
# (data/audit/stage4/final_evaluation.json) — AGENTS.md forbids reusing it.
ALREADY_OBSERVED_WINDOW = ("2026-06-13", "2026-07-25")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_path(path: str) -> str:
    """Sha256 of a file, or a deterministic combined hash of a partitioned
    parquet DIRECTORY (data/unified_races.parquet is Hive-partitioned by
    year=YYYY, not a single file) — every member file's relative path + content
    hash, sorted, then hashed together."""
    if os.path.isfile(path):
        return sha256_file(path)
    combined = hashlib.sha256()
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            rel = os.path.relpath(full, path).replace(os.sep, "/")
            combined.update(rel.encode("utf-8"))
            combined.update(sha256_file(full).encode("utf-8"))
    return combined.hexdigest()


def main() -> None:
    if not os.path.exists(MATRIX_PATH):
        raise SystemExit(
            f"{MATRIX_PATH} not found — run reports/improvement/10/01_build_matrix.py first.")

    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    win = win[win["_day"].notna()]

    data_max = win["_day"].max()
    final_start = data_max - pd.Timedelta(days=FINAL_HOLDOUT_DAYS)
    dev = win[win["_day"] < final_start]
    dev_oos_start = final_start - pd.Timedelta(days=DEV_OOS_DAYS)
    dev_core = dev[dev["_day"] < dev_oos_start]
    dev_oos = dev[(dev["_day"] >= dev_oos_start) & (dev["_day"] < final_start)]
    final_holdout = win[win["_day"] >= final_start]

    already_obs_start, already_obs_end = (pd.Timestamp(x) for x in ALREADY_OBSERVED_WINDOW)
    overlap = final_holdout[
        (final_holdout["_day"] >= already_obs_start) & (final_holdout["_day"] <= already_obs_end)]
    if len(overlap) or dev_oos_start <= already_obs_end:
        raise SystemExit(
            "REFUSING to freeze: the computed final-holdout/dev-OOS window overlaps "
            "the already-observed July 2026 window (AGENTS.md: must not reuse it).")

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Step 10 comparability protocol: market-only, conditional-logit, "
            "CatBoost and grouped-softmax LightGBM baselines, independent vs "
            "market-assisted, on one eligible-race/cutoff protocol. Frozen "
            "BEFORE any candidate is trained or tuned."
        ),
        "data": {
            "unified_races_path": UNIFIED_PATH,
            "unified_races_sha256": sha256_path(UNIFIED_PATH),
            "rebuilt_matrix_path": MATRIX_PATH,
            "rebuilt_matrix_sha256": sha256_path(MATRIX_PATH),
            "rebuilt_matrix_rows": int(len(df)),
            "rebuilt_matrix_win_rows": int(len(win)),
            "pipeline_fixes_included": ["D26", "D27", "D33", "D38", "D39", "D40", "D41"],
        },
        "code_hashes": {p: sha256_file(p) for p in CODE_FILES if os.path.exists(p)},
        "eligible_population": {
            "market_type": "WIN",
            "rule": "market_type == 'WIN' and position.notna() (labelled, priced win-market rows); "
                    "PLACE rows are a different price book (D24) and are out of this scorecard's scope.",
            "race_grouping_key": "race_uid",
            "head_to_head_extra_filter": (
                "models.head_to_head.head_to_head additionally requires a COMPLETE "
                "odds book (every runner in the race priced on the pre-off price "
                "column) before a race counts toward the log-loss scorecard — "
                "identical rule applied to every candidate and the market line."
            ),
        },
        "decision_cutoff": {
            "assumption": "10-minutes-before-off feature freeze (CONTRACTS.md provisional "
                          "research assumption, not a deployed cutoff constant)",
            "price_used_for_decision_and_market_benchmark": "pre-off (morningwap -> ppwap fallback), "
                          "never odds_finish (closing/settlement price, CLV reference only)",
        },
        "windows": {
            "data_max_date": str(data_max.date()),
            "dev_core": {
                "start": str(dev_core["_day"].min().date()) if len(dev_core) else None,
                "end_exclusive": str(dev_oos_start.date()),
                "n_rows": int(len(dev_core)),
                "n_races": int(dev_core["race_uid"].nunique()) if len(dev_core) else 0,
            },
            "dev_oos": {
                "start": str(dev_oos_start.date()),
                "end_exclusive": str(final_start.date()),
                "n_rows": int(len(dev_oos)),
                "n_races": int(dev_oos["race_uid"].nunique()) if len(dev_oos) else 0,
                "role": "Primary development scorecard slice for ALL candidates "
                        "(race-level log loss vs the de-vigged market). NOT the "
                        "reserved final holdout.",
            },
            "final_holdout_RESERVED_UNTOUCHED": {
                "start": str(final_start.date()),
                "end": str(data_max.date()),
                "n_rows": int(len(final_holdout)),
                "n_races": int(final_holdout["race_uid"].nunique()) if len(final_holdout) else 0,
                "policy": "Reserved for step17's single frozen-selection evaluation. "
                          "Step 10 must not load these rows into any fit, tuning CV, "
                          "calibration carve or head-to-head scoring call. Its coverage "
                          "(row/null counts only, no performance metric) was inspected "
                          "in step06's candidate build; no model has ever been scored "
                          "against it. If step17 finds this window has since been "
                          "inspected for anything beyond coverage, it must say so rather "
                          "than silently treating it as still-untouched.",
            },
            "already_observed_excluded": {
                "window": list(ALREADY_OBSERVED_WINDOW),
                "source": "data/audit/stage4/final_evaluation.json",
                "note": "Verified non-overlapping with the windows above.",
            },
        },
        "calibration": {
            "carve": "most-recent whole-race-group 15% of dev_core (models.split_utils."
                     "chronological_group_split), matching models/train.py's existing "
                     "calibration_size=0.15 default -- not re-tuned for this stage.",
        },
        "tuning_budget": {
            "protocol": "Optuna TPE, whole-race walk-forward CV (models.split_utils."
                        "group_time_series_split) on dev_core, race-level log loss "
                        "objective for the conditional logit; CatBoost's own existing "
                        "calibration-aware Optuna study (models/tuner.py) uses the "
                        "SAME n_trials/cv_folds via models.train.train(trials=...).",
            "n_trials_matched": 6,
            "cv_folds_matched": 5,
            "note": "n_trials bounded down from config.yaml's default optuna_trials=50 "
                    "to fit this stage's CPU-only, up-to-2-worker performance budget; "
                    "cv_folds kept at config.yaml's existing default (5) and matched "
                    "for the conditional logit's tune_l2. LightGBM (models.train_lgbm) "
                    "has no Optuna search in this codebase -- it early-stops a single "
                    "chronological train/validation split at its existing default "
                    "hyperparameters (models/lgbm_softmax._DEFAULT_PARAMS). This is a "
                    "real, disclosed asymmetry in tuning effort across model families, "
                    "not a matched budget for LightGBM specifically.",
        },
        "weighting_schemes_compared": [
            "existing odds-inverse sample weight (models.train._sample_weights, capped)",
            "unweighted (uniform sample weight)",
            "race-weighted (uniform weight but rescaled so every race contributes equal "
            "total weight regardless of field size)",
        ],
        "candidate_output_paths": {
            "conditional_logit": "data/audit/10/models_condlogit/",
            "catboost": "data/audit/10/models_catboost/",
            "lgbm_market": "data/audit/10/models_lgbm_market/",
            "lgbm_independent": "data/audit/10/models_lgbm_indep/",
            "dev_oos_predictions": "data/audit/10/dev_oos_predictions.parquet",
            "scorecard": "reports/improvement/10/scorecard.json",
        },
        "champion_artifacts_untouched": [
            "models/catboost_*_v3*.bin", "models/catboost_*_v3*_calib.pkl",
            "models/lgbm_won_v3.txt", "models/lgbm_v3_meta.json",
        ],
    }

    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"wrote {MANIFEST_PATH}")
    print("dev_core   :", manifest["windows"]["dev_core"])
    print("dev_oos    :", manifest["windows"]["dev_oos"])
    print("final_holdout (RESERVED):", manifest["windows"]["final_holdout_RESERVED_UNTOUCHED"])
    print("FREEZE_MANIFEST_10_COMPLETE")


if __name__ == "__main__":
    main()
