"""Step 12: freeze the market-combination / calibration protocol BEFORE any
blend exponent or calibrator is fitted.

Inherits step 10's frozen protocol (reports/improvement/10/run_manifest.json)
and its eligible population verbatim, then carves the DEVELOPMENT side of that
protocol into three chronological, whole-race-disjoint periods:

  blend_dev  -- walk-forward out-of-sample predictions used ONLY to select the
                power-blend exponents.
  calib      -- the LATER reserved calibration period; the calibrators are fit
                here, after the blend is already chosen.
  eval       -- step 10's frozen dev-OOS panel (2026-08-08..2026-08-27). Never
                touched by any selection or fit in this stage; every reported
                number comes from here.

  final_holdout_RESERVED_UNTOUCHED -- never loaded at all (step 17 only).

Run: .venv/Scripts/python.exe reports/improvement/12/01_freeze_manifest.py
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

from models.blend import ALPHA_GRID, BETA_GRID

STEP10_MANIFEST = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
OUT = "reports/improvement/12/blend_manifest.json"

CODE_FILES = [
    "models/blend.py",
    "models/calibration.py",
    "models/devig.py",
    "models/head_to_head.py",
    "models/conditional_logit.py",
    "models/split_utils.py",
    "models/features.py",
    "features/lgbm_adapter.py",
]

# Walk-forward block boundaries inside dev_core. Each block is scored by a model
# fitted ONLY on whole races strictly before the block start, so every row of the
# blend/calibration panel is a genuine out-of-sample prediction.
WF_BOUNDARIES = [
    "2026-02-01", "2026-03-01", "2026-04-01", "2026-05-01",
    "2026-06-01", "2026-06-13", "2026-07-01", "2026-08-01", "2026-08-08",
]
BLEND_DEV = ("2026-02-01", "2026-06-13")   # [start, end_exclusive)
CALIB = ("2026-06-13", "2026-08-08")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} already frozen — refusing to overwrite it.")

    with open(STEP10_MANIFEST, "r", encoding="utf-8") as fh:
        s10 = json.load(fh)

    matrix_sha = sha256_file(MATRIX_PATH)
    if matrix_sha != s10["data"]["rebuilt_matrix_sha256"]:
        raise SystemExit(
            "training_rebuilt.parquet no longer matches step 10's frozen hash — "
            "rebuild the baselines before freezing step 12."
        )

    w10 = s10["windows"]
    eval_start = w10["dev_oos"]["start"]
    eval_end = w10["dev_oos"]["end_exclusive"]
    final_start = w10["final_holdout_RESERVED_UNTOUCHED"]["start"]

    # Hard guards: the selection/fit periods must sit strictly inside dev_core and
    # must never reach the evaluation panel or the reserved final holdout.
    assert pd.Timestamp(CALIB[1]) == pd.Timestamp(eval_start) == pd.Timestamp(
        w10["dev_core"]["end_exclusive"])
    assert pd.Timestamp(BLEND_DEV[1]) == pd.Timestamp(CALIB[0])
    assert pd.Timestamp(eval_end) == pd.Timestamp(final_start)
    assert pd.Timestamp(WF_BOUNDARIES[0]) == pd.Timestamp(BLEND_DEV[0])
    assert pd.Timestamp(WF_BOUNDARIES[-1]) == pd.Timestamp(CALIB[1])

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Step 12 market-combination protocol: select a normalized power blend of the "
            "INDEPENDENT (price-free) probability and the REFERENCE de-vigged market "
            "probability on chronological walk-forward development predictions, fit a small "
            "predetermined set of calibrators on a LATER reserved calibration period, and "
            "report only on step 10's untouched dev-OOS panel. Frozen BEFORE any exponent "
            "or calibrator is fitted."
        ),
        "inherits": {
            "manifest": STEP10_MANIFEST,
            "manifest_sha256": sha256_file(STEP10_MANIFEST),
            "eligible_population": s10["eligible_population"],
            "decision_cutoff": s10["decision_cutoff"],
        },
        "data": {
            "rebuilt_matrix_path": MATRIX_PATH,
            "rebuilt_matrix_sha256": matrix_sha,
            "step10_dev_oos_predictions": "data/audit/10/dev_oos_predictions.parquet",
            "step10_dev_oos_predictions_sha256": sha256_file(
                "data/audit/10/dev_oos_predictions.parquet"),
            "step10_scorecard": "reports/improvement/10/scorecard.json",
            "step10_scorecard_sha256": sha256_file("reports/improvement/10/scorecard.json"),
        },
        "code_hashes": {p: sha256_file(p) for p in CODE_FILES},
        "price_roles": {
            "independent_probability": (
                "price-free model output; no price column of any kind enters it "
                "(models.features.PRICE_FREE_FEATURE_COLS minus MARKET_DERIVED_FEATURE_COLS, D40)"
            ),
            "reference_market_probability": (
                "models.devig.devig(bet_price) over a COMPLETE book only. bet_price is the "
                "pre-off reference price (ppwap -> morningwap, backtest.data.PanelConfig); "
                "never odds_finish/sp (settlement) and never a best-of-N executable overlay."
            ),
            "executable_quote": (
                "the bookmaker price actually backable (odds_decimal / odds_by_book / "
                "execution.snapshots). Used ONLY for EV, stake and settlement. It is not an "
                "argument of models.blend.power_blend and must not move any probability."
            ),
        },
        "windows": {
            "blend_dev": {
                "start": BLEND_DEV[0], "end_exclusive": BLEND_DEV[1],
                "role": "select the power-blend exponents (alpha, beta). Nothing else.",
            },
            "calib": {
                "start": CALIB[0], "end_exclusive": CALIB[1],
                "role": (
                    "LATER reserved calibration period: fit the calibrator candidates on the "
                    "already-selected blend. Overlaps the already-observed 2026-06-13..2026-07-25 "
                    "window on purpose — that window is barred from being a NEW TEST (AGENTS.md), "
                    "not from being development FITTING data, and no performance claim is made "
                    "from it."
                ),
            },
            "eval": {
                "start": eval_start, "end_exclusive": eval_end,
                "role": (
                    "step 10's frozen dev-OOS panel (618 races / 5,561 runners). Reporting only; "
                    "never used to choose an exponent, a calibrator or a threshold."
                ),
            },
            "final_holdout_RESERVED_UNTOUCHED": {
                "start": final_start,
                "end": w10["final_holdout_RESERVED_UNTOUCHED"]["end"],
                "policy": "Not loaded by any step-12 script. Step 17 only.",
            },
        },
        "walk_forward": {
            "boundaries": WF_BOUNDARIES,
            "scheme": (
                "expanding window: block [b_i, b_i+1) is scored by a model fitted on every "
                "labelled WIN row with day < b_i (from 2024-01-01), whole races only. No "
                "hyperparameter is re-tuned per block."
            ),
            "model_family": "models.conditional_logit.ConditionalLogitModel",
            "l2_source": (
                "reports/improvement/10/scorecard.json conditional_logit_tuning "
                "(independent + market branches) — reused verbatim, NOT re-tuned here, so the "
                "walk-forward panel introduces no new selection."
            ),
            "disclosed_limitation": (
                "Only the conditional-logit family gets its own walk-forward panel; a CatBoost "
                "or LightGBM walk-forward over the same 8 blocks is outside this stage's CPU "
                "budget. Their combined arms on the eval panel therefore reuse the exponents "
                "selected on the conditional-logit panel (a TRANSFER, labelled as such in the "
                "scorecard), not exponents fitted to their own out-of-sample predictions."
            ),
        },
        "blend_grid": {
            "form": "p ∝ p_independent**alpha * p_reference**beta, renormalised within race",
            "alpha": list(ALPHA_GRID),
            "beta": list(BETA_GRID),
            "n_points": len(ALPHA_GRID) * len(BETA_GRID),
            "boundary_controls": {
                "independent_only": [1.0, 0.0],
                "market_only": [0.0, 1.0],
            },
            "selection_metric": "race-level log loss (models.blend.race_level_log_loss)",
        },
        "calibrator_candidates": [
            {"name": "none", "role": "UNCALIBRATED CONTROL — the blend, renormalised only."},
            {"name": "sigmoid", "impl": "models.calibration.SigmoidCalibrator (Platt on the logit)"},
            {"name": "isotonic", "impl": "models.calibration.IsotonicCalibrator"},
            {"name": "odds_band",
             "impl": "models.calibration.OddsBandCalibrator fit on the REFERENCE price",
             "note": (
                 "price-conditional by construction, therefore MARKET-ADJUSTED, not an ability "
                 "estimate. Fitted and applied on the reference price, never on an executable quote."
             )},
        ],
        "post_calibration_rule": (
            "every calibrated probability is renormalised within its complete race "
            "(models.calibration.normalize_within_race) before scoring, so all candidates and the "
            "market line are compared as probability vectors that sum to 1."
        ),
        "reported_breakdowns": [
            "odds band (models.head_to_head.ODDS_BANDS, reference pre-off price)",
            "field size band",
            "racing regime (surface x distance category)",
            "region (UK / IRE)",
            "selected-bet subset (EV = p*bet_price - 1 >= config.yaml value.min_expected_value = 0.05)",
        ],
        "output_paths": {
            "walkforward_panel": "data/audit/12/walkforward_oos.parquet",
            "blend_artifacts": "data/audit/12/blend/",
            "calibrator_artifacts": "data/audit/12/calibrators/",
            "scorecard": "reports/improvement/12/scorecard_12.json",
            "breakdowns": "reports/improvement/12/breakdowns/",
            "invariance": "reports/improvement/12/price_invariance.json",
        },
        "promoted_artifacts_untouched": [
            "models/catboost_*_v3*.bin", "models/catboost_*_v3*_calib.pkl",
            "models/fl_oddsband_v3nf_calib.pkl", "models/lgbm_won_v3.txt",
            "models/lgbm_v3_meta.json",
        ],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"froze {OUT} at {manifest['frozen_at_utc']}")


if __name__ == "__main__":
    main()
