"""Step 17 — freeze the candidate-SELECTION rule before any champion score exists.

Written and frozen BEFORE ``03_champion_replay.py`` is run, so the rule cannot be
bent toward a result. Refuses to overwrite itself.

Run:  .venv/Scripts/python.exe reports/improvement/17/02_freeze_selection_protocol.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).with_name("02_selection_protocol.json")

PROTOCOL = {
    "purpose": "Choose ONE frozen candidate for paper operation, using development evidence only.",
    "why_a_replay_is_needed": (
        "Steps 10-13 scored only NEWLY FITTED models. No stage ever scored the SERVED champion bundle "
        "(models/catboost_won_v3nf.bin + fl_oddsband_v3nf_calib.pkl, models/catboost_won_v3.bin, "
        "models/lgbm_won_v3.txt) on the same panel, and the champion was trained BEFORE the D38/D39/D41 "
        "feature repairs while live serving now computes features through the REPAIRED pipeline - a "
        "train/serve feature skew nobody has measured. 'Simplest supported candidate' cannot be chosen "
        "without that comparison."
    ),
    "panel": {
        "matrix": "data/audit/10/training_rebuilt.parquet (sha256 verified against step 10's manifest)",
        "window": "dev_oos 2026-08-08..2026-08-27 (618 races / 5,561 runners) - step 10's DEVELOPMENT panel",
        "out_of_sample_for_champion": "yes - catboost v3nf train cutoff 2025-12-02, FL calibrator fit window "
                                      "ends 2026-06-12, lgbm v3 data_max_date 2026-06-12",
        "final_holdout": "NOT loaded by the replay. The holdout (2026-08-28..2026-09-17) is used exactly once, "
                         "afterwards, by 05_final_holdout_once.py, for the single candidate this rule selects.",
    },
    "metric": "race-level log loss via models.head_to_head.head_to_head on the identical complete-book race set "
              "(same scorer, same filter, same bet_price column as steps 10/12/13)",
    "uncertainty": "paired race-clustered bootstrap, 2,000 resamples, seed 17, CI95 of (challenger - champion) "
                   "per-race log loss; negative = challenger better",
    "supported_means": (
        "A candidate is SUPPORTED for paper operation only if the live serving path can load it today without "
        "new serving code: i.e. a CatBoost bundle in models.predictor's <model_dir>/catboost_<target>_<tag>.bin "
        "layout. Step 10's conditional logit and LightGBM rebuilds, step 12's blend and step 13's XGBoost have "
        "no serving loader, so they are REPORTED but are not eligible for selection in this stage."
    ),
    "eligible_candidates": {
        "champion_as_served": "models/ v3nf (price-free) + v3 (market-assisted) + fl_oddsband_v3nf - no change",
        "step10_catboost_rebuild": "data/audit/10/models_catboost/ (same family, repaired features, same loader)",
    },
    "decision_rule": [
        "PRIMARY line = the PRICE-FREE win probability, because execution.gates prefers "
        "value_win_prob_independent and it is the only line whose edge claim is not circular with the price.",
        "DEFAULT = champion_as_served (simplest: zero serving change, existing rollback metadata).",
        "Select step10_catboost_rebuild ONLY IF its price-free line beats the champion's price-free line on the "
        "panel with a paired bootstrap CI95 that EXCLUDES zero. Otherwise the default stands.",
        "A selection of the rebuild is a RECOMMENDATION recorded in the manifest with a reversible promotion "
        "procedure; this stage does not overwrite any file under models/.",
        "Whatever is selected, the MODEL VERDICT is decided separately by execution.model_gate against the "
        "market and is not relaxed: no candidate that fails to beat the market becomes GO.",
    ],
    "not_permitted": [
        "re-running with a different metric, panel, seed or rule after seeing the result",
        "loading any final-holdout row in the replay",
        "lowering any model/forward gate threshold",
    ],
}


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite frozen protocol {OUT.name} "
              f"(frozen_at_utc={json.loads(OUT.read_text(encoding='utf-8'))['frozen_at_utc']})")
        return 0
    OUT.write_text(json.dumps({"frozen_at_utc": datetime.now(timezone.utc).isoformat(), **PROTOCOL}, indent=1),
                   encoding="utf-8")
    print(f"froze {OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
