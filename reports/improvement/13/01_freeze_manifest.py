"""Step 13: freeze the XGBoost/ensemble experiment protocol BEFORE any XGBoost
model is trained or any blend weight is selected.

Inherits step 10's eligible-population/decision-cutoff/windows verbatim (same
dev_core / dev_oos / final_holdout_RESERVED_UNTOUCHED as steps 10-12) and step
12's blend_dev/calib sub-windows (already frozen inside dev_core) so XGBoost's
own weight-selection and calibration-fitting periods are the SAME periods
already used for CatBoost/LightGBM/conditional-logit combination work, not a
newly-chosen boundary.

Run: .venv/Scripts/python.exe reports/improvement/13/01_freeze_manifest.py
"""
import hashlib
import json
import os
from datetime import datetime, timezone

MANIFEST_10 = "reports/improvement/10/run_manifest.json"
MANIFEST_12 = "reports/improvement/12/blend_manifest.json"
OUT = "reports/improvement/13/manifest.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} already exists — refusing to overwrite a frozen manifest")

    m10 = json.load(open(MANIFEST_10, "r", encoding="utf-8"))
    m12 = json.load(open(MANIFEST_12, "r", encoding="utf-8"))

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Step 13: add XGBoost as a third boosted-tree candidate alongside CatBoost "
            "and grouped-softmax LightGBM; measure the incremental gain (or lack of it) "
            "from blending it into a CatBoost+LightGBM ensemble. Frozen BEFORE any "
            "XGBoost model is fit or any blend weight is selected."
        ),
        "inherits": {
            "run_manifest_10": MANIFEST_10,
            "run_manifest_10_sha256": _sha256(MANIFEST_10),
            "blend_manifest_12": MANIFEST_12,
            "blend_manifest_12_sha256": _sha256(MANIFEST_12),
            "eligible_population": m10["eligible_population"],
            "decision_cutoff": m10["decision_cutoff"],
        },
        "data": {
            "rebuilt_matrix_path": m10["data"]["rebuilt_matrix_path"],
            "rebuilt_matrix_sha256": m10["data"]["rebuilt_matrix_sha256"],
            "step10_dev_oos_predictions": "data/audit/10/dev_oos_predictions.parquet",
            "step12_eval_panel_scored": "data/audit/12/eval_panel_scored.parquet",
            "step12_eval_panel_scored_sha256": _sha256("data/audit/12/eval_panel_scored.parquet"),
        },
        "windows": {
            # Verbatim from step 10 — NEVER recomputed here.
            "dev_core": m10["windows"]["dev_core"],
            "dev_oos": m10["windows"]["dev_oos"],
            "final_holdout_RESERVED_UNTOUCHED": m10["windows"]["final_holdout_RESERVED_UNTOUCHED"],
            # Verbatim from step 12 — reused, not re-chosen, for XGBoost's own
            # blend-weight selection (blend_dev) and calibration fit (calib).
            "blend_dev": m12["windows"]["blend_dev"],
            "calib": m12["windows"]["calib"],
            "eval": m12["windows"]["eval"],
        },
        "xgboost_candidate": {
            "model": "xgboost.XGBClassifier, tree_method=hist (CPU)",
            "reason_cpu_not_gpu": (
                "CatBoost already owns the GPU in this experiment session "
                "(models/tuner.py _hardware_params); XGBoost's histogram method is "
                "fast enough on this CPU for this row/column count. Avoids any "
                "GPU-driver interaction between two boosted-tree libraries sharing "
                "one session (AGENTS.md: avoid concurrent GPU training)."
            ),
            "target": "won (scope-matched to step 10's CatBoost scorecard, D24 — "
                      "WIN-market rows only, no PLACE/place-market model trained here)",
            "branches": ["indep (PRICE_FREE_FEATURE_COLS)", "market (FEATURE_COLS)"],
            "tuning_budget": {
                "protocol": "Optuna TPE, whole-race walk-forward CV (models.split_utils."
                            "group_time_series_split) on dev_core, row-level log-loss "
                            "objective on the unweighted validation fold — same shape as "
                            "models/tuner.py's CatBoost study, matched n_trials/cv_folds.",
                "n_trials_matched": 6,
                "cv_folds_matched": 5,
            },
            "output_dir": "data/audit/13/models_xgboost/",
        },
        "oof_blend_protocol": {
            "purpose": "chronological out-of-fold predictions for CatBoost, LightGBM "
                       "and XGBoost over dev_core, used ONLY to select simple blend "
                       "weights (no stacker fit here — scope allows a simple blend "
                       "and the result below did not warrant a stacking meta-learner).",
            "method": "models.split_utils.group_time_series_split(dev_core, n_splits) "
                      "expanding-window walk-forward; FIXED hyperparameters per model "
                      "(no re-tuning per fold — same economisation step 12 used for its "
                      "conditional-logit walk-forward, extended here to 3 GBDT families "
                      "instead of a matched-cost full re-tune, which is outside this "
                      "stage's CPU budget, same disclosed limitation as step 12).",
            "n_splits": 6,
            "hyperparam_sources": {
                "catboost_indep": "data/audit/10/models_catboost/catboost_s10idp_meta.json",
                "catboost_market": "data/audit/10/models_catboost/catboost_s10mkt_meta.json",
                "lgbm_indep": "models.lgbm_softmax._DEFAULT_PARAMS (no Optuna tuning exists "
                              "for LightGBM in this codebase — same disclosed asymmetry as "
                              "step 10's run_manifest.json tuning_budget note)",
                "lgbm_market": "models.lgbm_softmax._DEFAULT_PARAMS",
                "xgboost_indep": "this stage's own freshly-tuned best_params (xgboost_candidate above)",
                "xgboost_market": "this stage's own freshly-tuned best_params",
            },
            "weight_selection_window": "blend_dev (from windows.blend_dev)",
            "calibration_fit_window": "calib (from windows.calib), OOF rows only "
                                      "(genuinely out-of-fold, never rows the scored "
                                      "model itself trained on)",
            "weight_grid": "each model's weight in {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0} "
                           "(0 allowed — XGBoost/any member may be excluded, never forced "
                           "to an equal share), exhaustive over the simplex points, scored "
                           "by models.blend.race_level_log_loss on OOF rows only.",
            "candidates": ["cb_lgbm (2-model)", "cb_lgbm_xgb (3-model)"],
            "blend_form": "weighted geometric mean in log-space: p ∝ prod(p_i ** w_i), "
                          "renormalised within race (same normalisation rule as "
                          "models.blend.power_blend, generalised from 2 to N inputs).",
        },
        "evaluation": {
            "panel": "eval (= step 10's dev_oos, 618 races / 5,561 runners), reused from "
                    "data/audit/12/eval_panel_scored.parquet with XGBoost's own probability "
                    "columns merged in — NEVER used to select a hyperparameter, a blend "
                    "weight or a calibrator.",
            "metrics": "race log loss, Brier, ECE, sample counts (models.head_to_head."
                      "head_to_head); whole-race-clustered bootstrap CI95 for each "
                      "candidate's gap vs the market and for the 3-model-minus-2-model "
                      "blend delta (race-level resampling, never a per-runner resample).",
            "diversity": "pairwise Pearson correlation of OOF probabilities/residuals "
                        "between CatBoost, LightGBM and XGBoost on the blend_dev OOF rows.",
        },
        "champion_artifacts_untouched": [
            "models/catboost_*_v3*.bin", "models/catboost_*_v3*_calib.pkl",
            "models/lgbm_won_v3.txt", "models/lgbm_v3_meta.json",
        ],
        "output_paths": {
            "manifest": OUT,
            "xgboost_models": "data/audit/13/models_xgboost/",
            "eval_panel_with_xgb": "data/audit/13/eval_panel_scored_with_xgb.parquet",
            "oof_blend_dev": "data/audit/13/oof_blend_dev.parquet",
            "blend_weights": "data/audit/13/blend_weights.json",
            "scorecard": "reports/improvement/13/scorecard_13.json",
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
