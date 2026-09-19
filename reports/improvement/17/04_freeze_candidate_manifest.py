"""Step 17 — freeze ONE candidate manifest for paper operation.

Frozen AFTER the selection rule was applied on development data
(``03_champion_replay.json``) and BEFORE the final holdout is opened
(``05_final_holdout_once.py``). Thresholds/terms are READ from config.yaml, never
retyped. Refuses to overwrite itself. Writes nothing under models/.

Run:  .venv/Scripts/python.exe reports/improvement/17/04_freeze_candidate_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent
OUT = HERE / "candidate_manifest.json"


def sha(rel: str) -> dict:
    p = ROOT / rel
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"sha256": h.hexdigest(), "bytes": p.stat().st_size}


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite frozen manifest (frozen_at_utc="
              f"{json.loads(OUT.read_text(encoding='utf-8'))['frozen_at_utc']})")
        return 0
    replay = json.loads((HERE / "03_champion_replay.json").read_text(encoding="utf-8"))
    m10 = json.loads((ROOT / "reports/improvement/10/run_manifest.json").read_text(encoding="utf-8"))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ex = cfg["execution"]
    cand_dir = "data/audit/10/models_catboost"
    meta = {t: json.loads((ROOT / f"{cand_dir}/catboost_{t}_meta.json").read_text(encoding="utf-8"))
            for t in ("s10idp", "s10mkt")}
    assert replay["selected_by_frozen_rule"] == "step10_catboost_rebuild", replay["selected_by_frozen_rule"]

    served = ["models/catboost_won_v3nf.bin", "models/catboost_won_v3nf_calib.pkl", "models/catboost_v3nf_meta.json",
              "models/fl_oddsband_v3nf_calib.pkl", "models/catboost_won_v3.bin", "models/catboost_won_v3_calib.pkl",
              "models/catboost_placed_2_v3.bin", "models/catboost_showed_v3.bin", "models/catboost_v3_meta.json",
              "models/lgbm_won_v3.txt", "models/lgbm_v3_meta.json"]
    code = ["features/builder.py", "features/derive.py", "features/fuse.py", "features/_trailing_fast.py",
            "models/features.py", "models/predictor.py", "models/calibration.py", "models/devig.py",
            "backtest/holdout.py", "backtest/data.py", "models/head_to_head.py", "execution/gates.py",
            "execution/config.py", "execution/model_gate.py", "execution/forward_gate.py", "execution/tickets.py",
            "execution/snapshots.py", "execution/settlement.py", "scripts/daily_paper_loop.py", "scripts/refresh.py",
            "config.yaml"]

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "17", "mode": "PAPER-ONLY", "model_verdict_at_freeze": "NO-GO (execution.model_gate)",
        "selection": {
            "rule": "reports/improvement/17/02_selection_protocol.json (frozen before any champion score)",
            "evidence": "reports/improvement/17/03_champion_replay.json",
            "selected": "step10_catboost_rebuild",
            "primary_paired_delta": replay["paired_bootstrap"]["PRIMARY step10_catboost_indep minus champion_v3nf_indep"],
            "why_not_more_complex": "step 12 blend put ZERO weight on the model; step 13 XGBoost got weight 0.0; "
                                    "step 11 measured-speed family has no live ingestion path; step 16 text is "
                                    "DEFERRED_DATA. None is both supported and better.",
        },
        "candidate": {
            "family": "CatBoost binary `won`, WIN-market rows only (D24), sigmoid base calibrator, within-race "
                      "normalisation at scoring; trained by step 10 on the REPAIRED matrix (D26/D27/D33/D38-D41)",
            "primary_line_price_free": {
                "tag": "s10idp", "n_features": len(meta["s10idp"]["feature_cols"]),
                "feature_cols": meta["s10idp"]["feature_cols"],
                "calibration": meta["s10idp"]["targets"]["won"]["calibration_method"],
                "files": {f: sha(f"{cand_dir}/{f}") for f in
                          ("catboost_won_s10idp.bin", "catboost_won_s10idp_calib.pkl", "catboost_s10idp_meta.json")},
            },
            "disclosure_line_market_assisted": {
                "tag": "s10mkt", "n_features": len(meta["s10mkt"]["feature_cols"]),
                "feature_cols": meta["s10mkt"]["feature_cols"],
                "calibration": meta["s10mkt"]["targets"]["won"]["calibration_method"],
                "files": {f: sha(f"{cand_dir}/{f}") for f in
                          ("catboost_won_s10mkt.bin", "catboost_won_s10mkt_calib.pkl", "catboost_s10mkt_meta.json")},
                "note": "market-assisted: never qualifies a bet (execution.gates accepts only the price-free key)",
            },
            "training_window": m10["windows"]["dev_core"],
            "targets_available": ["won"],
            "targets_MISSING_vs_served_bundle": ["placed_2", "showed", "favourite-longshot recalibrator",
                                                 "grouped-softmax LightGBM line"],
        },
        "data": {"matrix": m10["data"]["rebuilt_matrix_path"], "matrix_sha256": m10["data"]["rebuilt_matrix_sha256"],
                 "unified_races_sha256": m10["data"]["unified_races_sha256"],
                 "eligible_population": m10["eligible_population"]},
        "operating_cutoff": {
            "research_assumption": m10["decision_cutoff"],
            "deployed": "NONE. No minutes-before-off feature freeze exists in serving; the only live time guards are "
                        "the 900 s quote/source TTLs. execution.safeguards.block_started_races is configured but "
                        "not called by scripts/daily_paper_loop.py.",
        },
        "selection_threshold": {k: ex["gates"][k] for k in
                                ("min_edge", "min_expected_value", "min_field_size", "max_overround")},
        "gate_requirements": {k: v for k, v in ex["gates"].items() if k.startswith("require_")},
        "staking_rule": {**ex["staking"], "max_stake_per_bet_eur": ex["frictions"]["max_stake_per_bet"],
                         "LIVE_STATUS": "NOT APPLIED on the live loop: scripts/daily_paper_loop.py passes no "
                                        "stake_decision to build_ticket, so every live ticket has stake=None."},
        "safeguards_configured": {**ex["safeguards"], "LIVE_STATUS": "execution.safeguards is used only by "
                                                                   "execution/simulator.py, not by the live loop."},
        "supported_market_terms": {
            "supported_live": ["WIN market only", "fixed-odds quote at a NAMED bookmaker (consensus/SP/Timeform "
                               "sources are non-executable)", "quote age <= %ss" % ex["snapshots"]["max_age_seconds"],
                               "complete reference book", "field >= %s" % ex["gates"]["min_field_size"]],
            "commission": {"exchange_commission": ex["frictions"]["exchange_commission"],
                           "exchange_sources": ex["frictions"]["exchange_sources"],
                           "live_effect": "0.0 - no live odds source is an exchange"},
            "rule_4_deductions": "implemented execution/settlement.py; OFFLINE SIMULATOR ONLY - UNSUPPORTED live",
            "dead_heats": "implemented execution/settlement.py; OFFLINE SIMULATOR ONLY - UNSUPPORTED live",
            "non_runners": "serving removes declared non-runners and renormalises; live SETTLEMENT maps a null "
                           "position to `void` (also catches non-finishers) - UNSUPPORTED as evidence",
            "each_way_place": "UNSUPPORTED live (terms recorded on the ticket only); PLACE odds are not captured",
            "best_odds_guaranteed": "never assumed", "accumulators": "never permitted",
            "fail_closed_rule": "Any term marked UNSUPPORTED must not produce settled CANDIDATE evidence. While the "
                                "verdict is NO-GO no CANDIDATE exists, so nothing can be mis-settled; before any GO, "
                                "blockers B2-B4 in READINESS.md must be closed.",
        },
        "forward_gate_retained": ex["forward_gate"],
        "real_money": {"execution.paper_only": ex["paper_only"], "placement_code_in_repo": False},
        "serving_status": {
            "currently_served": {f: sha(f) for f in served},
            "candidate_is_served": False,
            "note": "This stage promotes NOTHING. The served bundle above is also the ROLLBACK target.",
        },
        "promotion_procedure_reversible_NOT_EXECUTED": [
            "0. Precondition: a human approves; the final-holdout result in 05_final_holdout_once.json is read first.",
            "1. COPY (never move/overwrite) catboost_won_s10idp.bin, catboost_won_s10idp_calib.pkl and "
            "catboost_s10idp_meta.json from data/audit/10/models_catboost/ into models/. The tag is new, so no "
            "existing file is replaced. Verify the three sha256 values against this manifest.",
            "2. In config.yaml set value.model_tag: s10idp (was v3nf). Leave model.version_tag: v3 unchanged - the "
            "rebuild has no placed_2/showed targets, so the headline/place bundle cannot be swapped without a retrain.",
            "3. No fl_oddsband_s10idp_calib.pkl exists: the market-adjusted value_win_prob loses its F-L remap. "
            "Refit it on development data (scripts.refit_fl_recalibrator) or accept the unadjusted line; decide "
            "BEFORE switching and record which.",
            "4. Run: pytest -q tests/models/test_predictor.py tests/execution tests/scripts/test_daily_paper_loop.py, "
            "then the isolated replay reports/improvement/17/06_paper_replay.py.",
            "5. ROLLBACK = set value.model_tag back to v3nf. Nothing was deleted, so rollback is one config line.",
            "6. A promotion changes the evaluated model: any forward window must (re)start after it, never before.",
        ],
        "final_holdout_plan": {
            "window": m10["windows"]["final_holdout_RESERVED_UNTOUCHED"],
            "eligibility": "reserved by step 10 (D42) for this stage; 01_dispositions.json verifies no stage artifact "
                           "holds a model score on these rows; prior looks were label-free (feature fill, race counts)",
            "use": "EXACTLY ONCE, by 05_final_holdout_once.py, guarded by a lock file",
            "primary_endpoint": "s10idp price-free race log loss vs the de-vigged market (models.head_to_head)",
            "pre_declared_descriptive_lines": ["s10mkt", "champion_v3nf_indep", "champion_v3nf_fl_market_adjusted",
                                               "champion_v3_market", "champion_lgbm_v3_market"],
            "no_decision_from_holdout": "Selection is already frozen. The holdout can confirm or contradict; it "
                                        "cannot re-select, re-tune or relax a gate.",
        },
        "code_hashes": {f: sha(f)["sha256"] for f in code},
    }
    OUT.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"froze {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
