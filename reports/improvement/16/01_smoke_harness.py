"""Step 16 / 1: SMOKE TEST of the pre-registered harness — NOT the acceptance
ablation (see frozen_protocol.json's status_rule and coverage_report.json).

00_coverage_report.py already proved zero archived-text races in dev_oos or
final_holdout: only the 21 races on the single archived day (2026-06-17,
inside dev_core) have any text at all. This script exists only to prove the
join + feature-attachment + scoring code path in features/text_features_v1.py
runs end-to-end on real data without crashing, and to make the consequence of
that coverage gap concrete rather than asserted: a conditional-logit fit on
dev_core WITHOUT the 21 archived races sees a text-feature column that is
100% missing (constant/NaN) for every training row, so its fitted weight is
provably ~0 regardless of whether text "would" help — there is no training
variance for the model to learn from. Scoring on the one held-out archived
day is therefore expected to show no material difference between arms; that
is not evidence text is useless, it is a structural artefact of having only
one archived day, exactly why this stage is DEFERRED_DATA.

Reuses step 10's own frozen conditional-logit l2 (reports/improvement/10/scorecard.json
conditional_logit_tuning) rather than re-tuning on this tiny slice, which
would be meaningless. LightGBM/hosted-network arms are deliberately excluded
from this smoke run (cost/time disproportionate to a plumbing check); the
hosted arm is exercised directly by tests/features/test_text_features_v1.py
and by the cache-reuse probe below (cache hits only, zero spend).
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from features.lgbm_adapter import INDEPENDENT_FEATURE_COLS, build_lgbm_matrix  # noqa: E402
from features.text_features_v1 import (  # noqa: E402
    CANDIDATE_TEXT_FEATURE_COLS, TextJoinConfig, _archive_key,
    attach_text_features, build_cache_only_hosted_backend)
from llm.text_archive import get_text_archive  # noqa: E402
from llm.text_features import TextFeatureExtractor  # noqa: E402
from models.conditional_logit import ConditionalLogitModel  # noqa: E402

MANIFEST_PATH = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
STEP10_SCORECARD = "reports/improvement/10/scorecard.json"
OUT_PATH = "reports/improvement/16/smoke_scorecard.json"

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def race_log_loss_by_race(panel: pd.DataFrame, prob_col: str) -> dict:
    sub = panel.dropna(subset=[prob_col])
    out = {}
    for rid, grp in sub.groupby("race_uid", sort=False):
        p = grp[prob_col].to_numpy(dtype=float)
        total = p.sum()
        if total > 0:
            p = p / total
        y = grp["won"].to_numpy(dtype=float)
        wp = p[y.astype(bool)]
        if wp.size:
            out[rid] = -float(np.log(np.clip(wp, 1e-12, 1.0)).sum())
    return out


def paired_bootstrap(panel, base_col, test_col, *, n=2000, seed=16):
    base = race_log_loss_by_race(panel, base_col)
    test = race_log_loss_by_race(panel, test_col)
    shared = sorted(set(base) & set(test))
    if not shared:
        return {"n_races": 0}
    deltas = np.array([base[r] - test[r] for r in shared], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n_races": len(shared),
        "base_race_log_loss": float(np.mean([base[r] for r in shared])),
        "test_race_log_loss": float(np.mean([test[r] for r in shared])),
        "mean_delta_base_minus_test": float(deltas.mean()),
        "ci95_low": float(lo), "ci95_high": float(hi),
        "significant_improvement": bool(lo > 0),
    }


def main() -> int:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    with open(STEP10_SCORECARD, "r", encoding="utf-8") as fh:
        step10_scorecard = json.load(fh)
    l2_indep = step10_scorecard["conditional_logit_tuning"]["independent"]["l2"]

    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    w = manifest["windows"]
    dev_core = win[win["_day"] < pd.Timestamp(w["dev_core"]["end_exclusive"])].reset_index(drop=True)

    archive = get_text_archive()
    known = archive.known_race_uids()
    has_text = dev_core["race_uid"].astype(str).map(_archive_key).isin(known)
    smoke_eval = dev_core[has_text].reset_index(drop=True)
    fit_core = dev_core[~has_text].reset_index(drop=True)
    log(f"dev_core {len(dev_core)} rows / {dev_core['race_uid'].nunique()} races; "
        f"fit_core {len(fit_core)} rows / {fit_core['race_uid'].nunique()} races "
        f"(archived day excluded); smoke_eval {len(smoke_eval)} rows / "
        f"{smoke_eval['race_uid'].nunique()} races (the one archived day)")

    regex_extractor = TextFeatureExtractor(enabled=True, backend="regex")
    hosted_backend = build_cache_only_hosted_backend(
        model="deepseek/deepseek-v4.1-flash", reasoning_effort="low", max_output_tokens=1600)

    def build_arm(frame, extractor=None, hosted=None):
        joined = attach_text_features(frame, extractor=extractor, hosted_backend=hosted,
                                      archive=archive, config=TextJoinConfig())
        return joined

    fit_no_text = fit_core
    fit_regex = build_arm(fit_core, extractor=regex_extractor)
    fit_hosted = build_arm(fit_core, hosted=hosted_backend)
    eval_no_text = smoke_eval
    eval_regex = build_arm(smoke_eval, extractor=regex_extractor)
    eval_hosted = build_arm(smoke_eval, hosted=hosted_backend)

    coverage_of = {
        "fit_core_regex_rows_with_text": int(fit_regex["text_v1_available"].sum()),
        "fit_core_hosted_rows_with_text": int(fit_hosted["text_v1_available"].sum()),
        "smoke_eval_regex_rows_with_text": int(eval_regex["text_v1_available"].sum()),
        "smoke_eval_hosted_rows_with_text": int(eval_hosted["text_v1_available"].sum()),
        "smoke_eval_hosted_cache_hit_rows": int(
            (eval_hosted["text_v1_source"].str.startswith("hosted_deepseek")).sum()),
    }
    log(f"coverage: {coverage_of}")

    arms = {
        "no_text": (fit_no_text, eval_no_text, list(INDEPENDENT_FEATURE_COLS)),
        "regex_text": (fit_regex, eval_regex, list(INDEPENDENT_FEATURE_COLS) + CANDIDATE_TEXT_FEATURE_COLS),
        "hosted_text": (fit_hosted, eval_hosted, list(INDEPENDENT_FEATURE_COLS) + CANDIDATE_TEXT_FEATURE_COLS),
    }

    panel = smoke_eval[["race_uid", "horse_id", "won"]].copy()
    candidates = {}
    for name, (fit_frame, eval_frame, cols) in arms.items():
        X_fit, y_fit, rid_fit = build_lgbm_matrix(fit_frame, inference=False, feature_cols=cols)
        model = ConditionalLogitModel(l2=l2_indep).fit(X_fit, y_fit, rid_fit)
        X_eval, _, rid_eval = build_lgbm_matrix(eval_frame, inference=True, feature_cols=cols)
        prob = model.predict_proba(X_eval, rid_eval)
        col = f"{name}_prob"
        scored = pd.DataFrame({"race_uid": np.asarray(rid_eval),
                               "horse_id": eval_frame["horse_id"].to_numpy(), col: prob})
        panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
        race_ll = race_log_loss_by_race(panel, col)
        mean_ll = float(np.mean(list(race_ll.values()))) if race_ll else None
        candidates[name] = {
            "n_train_rows": int(len(fit_frame)), "n_train_races": int(fit_frame["race_uid"].nunique()),
            "n_eval_rows": int(len(eval_frame)), "n_eval_races": int(eval_frame["race_uid"].nunique()),
            "converged": model.metadata.get("converged"),
            "mean_race_log_loss": mean_ll,
            "n_scored_races": len(race_ll),
        }
        log(f"arm [{name}]: converged={model.metadata.get('converged')} "
            f"mean_race_log_loss={mean_ll}")

    bootstrap = {
        "regex_vs_no_text": paired_bootstrap(panel, "no_text_prob", "regex_text_prob"),
        "hosted_vs_no_text": paired_bootstrap(panel, "no_text_prob", "hosted_text_prob"),
    }

    report = {
        "status": "SMOKE_TEST_NOT_ACCEPTANCE_ABLATION",
        "reason": (
            "coverage_report.json shows 0 archived-text races in dev_oos/final_holdout; "
            "this run scores only the single archived day, held out of a same-window "
            "training fit -- not a chronological future-window comparison. See "
            "frozen_protocol.json status_rule."
        ),
        "manifest": MANIFEST_PATH, "matrix": MATRIX_PATH,
        "l2_reused_from_step10": l2_indep,
        "coverage": coverage_of,
        "candidates": candidates,
        "paired_bootstrap": bootstrap,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    log(f"wrote {OUT_PATH}")
    print(json.dumps(report, indent=2, default=str))
    print("SMOKE_HARNESS_16_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
