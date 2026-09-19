"""Step 10: train the comparable baselines and score them on the frozen dev-OOS
slice. Reads reports/improvement/10/run_manifest.json (must already be frozen
by 02_freeze_manifest.py) and NEVER loads any row from the manifest's
``final_holdout_RESERVED_UNTOUCHED`` window into any fit, tuning CV, calibration
carve or scoring call.

Candidates (independent + market-assisted branches, where applicable):
  * market-only  -- the de-vigged pre-off market line (models.devig), free by
                    construction of models.head_to_head's "market" side.
  * conditional-logit -- models.conditional_logit.ConditionalLogitModel
  * catboost     -- models.train.train(), target="won" only (scorecard scope)
  * lgbm_softmax -- models.train_lgbm.run_training()

Writes:
  reports/improvement/10/scorecard.json   -- the comparability table
  data/audit/10/dev_oos_predictions.parquet -- every candidate's scored dev-OOS
                                               probability, for later reuse
  data/audit/10/models_condlogit/*.json
  data/audit/10/models_catboost/*.bin/.pkl/.json  (candidate artifacts, NOT champion)
  data/audit/10/models_lgbm_market/, models_lgbm_indep/

Run: .venv/Scripts/python.exe reports/improvement/10/03_run_baselines.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from backtest.holdout import load_frozen_model, score as score_catboost
from features.lgbm_adapter import INDEPENDENT_FEATURE_COLS, FINAL_FEATURE_COLS, build_lgbm_matrix
from models.conditional_logit import ConditionalLogitModel, tune_l2
from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.head_to_head import head_to_head
from models.train import _sample_weights, train as train_catboost
from models.train_lgbm import run_training as run_lgbm_training

MANIFEST_PATH = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
OUT_SCORECARD = "reports/improvement/10/scorecard.json"
OUT_PREDICTIONS = "data/audit/10/dev_oos_predictions.parquet"

N_TRIALS = 6
CV_FOLDS = 5

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def load_manifest_and_data():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    win = win[win["_day"].notna()].reset_index(drop=True)

    w = manifest["windows"]
    dev_core_end = pd.Timestamp(w["dev_core"]["end_exclusive"])
    dev_oos_start = pd.Timestamp(w["dev_oos"]["start"])
    dev_oos_end = pd.Timestamp(w["dev_oos"]["end_exclusive"])
    final_start = pd.Timestamp(w["final_holdout_RESERVED_UNTOUCHED"]["start"])
    assert dev_core_end == dev_oos_start
    assert dev_oos_end == final_start

    dev_core = win[win["_day"] < dev_core_end].reset_index(drop=True)
    dev_oos = win[(win["_day"] >= dev_oos_start) & (win["_day"] < dev_oos_end)].reset_index(drop=True)
    # Final holdout rows are loaded into `win` above (unavoidable — it's one
    # parquet) but are NEVER selected into dev_core/dev_oos and never passed to
    # any function below. Only dev_core/dev_oos are used from here on.
    log(f"dev_core: {len(dev_core)} rows / {dev_core['race_uid'].nunique()} races "
        f"({dev_core['_day'].min().date()}..{dev_core['_day'].max().date()})")
    log(f"dev_oos : {len(dev_oos)} rows / {dev_oos['race_uid'].nunique()} races "
        f"({dev_oos['_day'].min().date() if len(dev_oos) else None}.."
        f"{dev_oos['_day'].max().date() if len(dev_oos) else None})")
    del win, df
    return manifest, dev_core, dev_oos


def build_dev_oos_panel(dev_oos: pd.DataFrame) -> pd.DataFrame:
    panel = load_panel(df=dev_oos, config=PanelConfig(feature_cols=()))
    log(f"dev_oos panel (priced+outcome rows): {len(panel)} rows / "
        f"{panel['race_uid'].nunique()} races")
    return panel


def _merge_prob(panel: pd.DataFrame, race_uid, horse_id, prob, colname: str) -> pd.DataFrame:
    scored = pd.DataFrame({"race_uid": np.asarray(race_uid), "horse_id": np.asarray(horse_id),
                           colname: np.asarray(prob, dtype=float)})
    scored = scored.drop_duplicates(subset=["race_uid", "horse_id"])
    before = len(panel)
    merged = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
    n_missing = int(merged[colname].isna().sum())
    if n_missing:
        log(f"  WARNING: {colname}: {n_missing}/{before} panel rows had no score")
    return merged


# ── conditional logit ──────────────────────────────────────────────────────


def run_conditional_logit(branch: str, feature_cols, dev_core, dev_oos, panel):
    log(f"conditional-logit [{branch}]: building matrices")
    X_core, y_core, rid_core = build_lgbm_matrix(dev_core, inference=False, feature_cols=feature_cols)
    keep = pd.to_numeric(dev_core["won"], errors="coerce").notna().to_numpy()
    order_core = dev_core.loc[keep, "race_date"].to_numpy()
    assert len(order_core) == len(X_core)

    log(f"conditional-logit [{branch}]: tuning L2 ({N_TRIALS} trials, {CV_FOLDS}-fold walk-forward CV)")
    tuned = tune_l2(X_core, y_core, rid_core, order_core, n_trials=N_TRIALS, n_splits=CV_FOLDS)
    log(f"conditional-logit [{branch}]: best l2={tuned['l2']:.5g} cv_log_loss={tuned['cv_log_loss']:.5f}")

    model = ConditionalLogitModel(l2=tuned["l2"]).fit(X_core, y_core, rid_core)
    out_path = f"data/audit/10/models_condlogit/{branch}.json"
    model.save(out_path)
    log(f"conditional-logit [{branch}]: saved {out_path}")

    X_oos, _, rid_oos = build_lgbm_matrix(dev_oos, inference=True, feature_cols=feature_cols)
    prob = model.predict_proba(X_oos, rid_oos)
    panel = _merge_prob(panel, dev_oos["race_uid"], dev_oos["horse_id"], prob,
                        f"condlogit_{branch}_prob")
    return panel, tuned


def run_weighting_ablation(dev_core, l2: float) -> dict:
    """Compare existing odds-inverse, unweighted and race-weighted sample
    weights on the market-assisted conditional logit, scored by whole-race
    walk-forward CV log loss on dev_core (never touches dev_oos or the final
    holdout — this ablation stays inside the development fold)."""
    from models.split_utils import group_time_series_split

    log("weighting ablation: building market-assisted dev_core matrix")
    X_core, y_core, rid_core = build_lgbm_matrix(dev_core, inference=False, feature_cols=FINAL_FEATURE_COLS)
    keep = pd.to_numeric(dev_core["won"], errors="coerce").notna().to_numpy()
    core_kept = dev_core.loc[keep].reset_index(drop=True)
    order_core = core_kept["race_date"].to_numpy()

    odds_inverse = _sample_weights(core_kept, max_weight=20.0)
    unweighted = np.ones(len(core_kept))
    field_size = core_kept.groupby("race_uid")["race_uid"].transform("size").to_numpy()
    race_weighted = 1.0 / np.maximum(field_size, 1)

    schemes = {"existing_odds_inverse": odds_inverse, "unweighted": unweighted,
              "race_weighted": race_weighted}
    results = {}
    for name, w in schemes.items():
        scores = []
        for train_idx, val_idx in group_time_series_split(rid_core, order_core, CV_FOLDS):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            y_tr = y_core[train_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            m = ConditionalLogitModel(l2=l2).fit(
                X_core.iloc[train_idx], y_tr, rid_core[train_idx], sample_weight=w[train_idx])
            p = m.predict_proba(X_core.iloc[val_idx], rid_core[val_idx])
            yv, rv = y_core[val_idx], rid_core[val_idx]
            per_race = []
            for rid in pd.unique(rv):
                mask = rv == rid
                wp = p[mask][yv[mask].astype(bool)]
                if wp.size:
                    per_race.append(-float(np.log(np.clip(wp, 1e-12, 1.0)).sum()))
            if per_race:
                scores.append(float(np.mean(per_race)))
        results[name] = float(np.mean(scores)) if scores else float("nan")
        log(f"weighting ablation [{name}]: dev_core CV race log loss = {results[name]:.5f}")
    return results


# ── CatBoost ──────────────────────────────────────────────────────────────


def run_catboost(branch: str, feature_cols, tag: str, dev_core, dev_oos, panel):
    log(f"catboost [{branch}]: training (target=won only, trials={N_TRIALS})")
    model_dir = "data/audit/10/models_catboost"
    meta = train_catboost(
        df=dev_core.copy(), targets=["won"], trials=N_TRIALS, model_dir=model_dir,
        feature_cols=feature_cols, version_tag=tag,
    )
    if meta is None:
        raise RuntimeError(f"catboost [{branch}]: train() returned None (empty matrix)")
    log(f"catboost [{branch}]: internal test AUC={meta['targets']['won']['test_auc']:.4f} "
        f"brier={meta['targets']['won']['test_brier']:.4f}")

    model_path = os.path.join(model_dir, f"catboost_won_{tag}.bin")
    frozen = load_frozen_model(model_path)
    cal_prob = score_catboost(frozen, dev_oos, price_col="bet_price")
    panel = _merge_prob(panel, dev_oos["race_uid"], dev_oos["horse_id"], cal_prob,
                        f"catboost_{branch}_prob")
    return panel, meta


# ── LightGBM ────────────────────────────────────────────────────────────────


def run_lgbm(branch: str, feature_cols, dev_core, dev_oos, manifest):
    w = manifest["windows"]
    dev_all = pd.concat([dev_core, dev_oos], ignore_index=True)
    out_dir = f"data/audit/10/models_lgbm_{branch}"
    log(f"lgbm [{branch}]: training via models.train_lgbm.run_training()")
    result = run_lgbm_training(
        df=dev_all,
        output_path=f"{out_dir}/lgbm_won.txt",
        meta_path=f"{out_dir}/lgbm_v3_meta.json",
        holdout_out=f"{out_dir}/holdout_dev_oos",
        max_date=(pd.Timestamp(w["dev_core"]["end_exclusive"]) - pd.Timedelta(days=1)).date().isoformat(),
        holdout_start=w["dev_oos"]["start"],
        holdout_end=(pd.Timestamp(w["dev_oos"]["end_exclusive"]) - pd.Timedelta(days=1)).date().isoformat(),
        model_version=f"s10-{branch}",
        model_params={"seed": 42},
        early_stopping_rounds=80,
        feature_cols=feature_cols,
    )
    h2h = result["summary"]["head_to_head"]
    log(f"lgbm [{branch}]: dev_oos race log loss model={h2h['model']['log_loss']:.5f} "
        f"market={h2h['market']['log_loss']:.5f}")
    return result


def main() -> None:
    manifest, dev_core, dev_oos = load_manifest_and_data()
    panel = build_dev_oos_panel(dev_oos)

    scorecard = {"manifest": MANIFEST_PATH, "dev_oos_panel_rows": int(len(panel)),
                "dev_oos_panel_races": int(panel["race_uid"].nunique()), "candidates": {}}

    # ── conditional logit (independent + market-assisted) ────────────────────
    # Same feature sets as the LightGBM branches below (features.lgbm_adapter's
    # rebuilt-from-pre-off-price market block), so a measured gap between the
    # conditional logit and LightGBM is attributable to model capacity, not to
    # a different feature definition.
    panel, tuned_indep = run_conditional_logit("indep", INDEPENDENT_FEATURE_COLS,
                                               dev_core, dev_oos, panel)
    panel, tuned_mkt = run_conditional_logit("market", FINAL_FEATURE_COLS,
                                             dev_core, dev_oos, panel)
    scorecard["conditional_logit_tuning"] = {"independent": tuned_indep, "market": tuned_mkt}

    # ── weighting-scheme ablation (development folds only, market-assisted) ──
    scorecard["weighting_ablation_dev_core_cv_log_loss"] = run_weighting_ablation(
        dev_core, l2=tuned_mkt["l2"])

    # ── CatBoost (independent + market-assisted), target=won only ────────────
    panel, cb_indep_meta = run_catboost("indep", PRICE_FREE_FEATURE_COLS, "s10idp",
                                        dev_core, dev_oos, panel)
    panel, cb_mkt_meta = run_catboost("market", FEATURE_COLS, "s10mkt",
                                      dev_core, dev_oos, panel)

    # ── LightGBM grouped-softmax (independent + market-assisted) ─────────────
    lgbm_indep = run_lgbm("indep", INDEPENDENT_FEATURE_COLS, dev_core, dev_oos, manifest)
    lgbm_mkt = run_lgbm("market", FINAL_FEATURE_COLS, dev_core, dev_oos, manifest)

    # ── unified dev-OOS head-to-head scorecard ────────────────────────────────
    candidates = {
        "condlogit_indep": "condlogit_indep_prob",
        "condlogit_market": "condlogit_market_prob",
        "catboost_indep": "catboost_indep_prob",
        "catboost_market": "catboost_market_prob",
    }
    market_metrics = None
    for name, col in candidates.items():
        if col not in panel.columns or panel[col].isna().all():
            log(f"scorecard: skipping {name} ({col} unavailable)")
            continue
        h2h = head_to_head(panel.dropna(subset=[col]), prob_col=col, odds_col="bet_price",
                           race_id_col="race_uid", label_col="won")
        h2h.pop("odds_band_table")
        scorecard["candidates"][name] = h2h
        market_metrics = h2h["market"]
        log(f"scorecard [{name}]: model_log_loss={h2h['model']['log_loss']:.5f} "
            f"market_log_loss={h2h['market']['log_loss']:.5f} "
            f"beats_market={h2h['model_beats_market_logloss']} n_races={h2h['n_races_kept']}")

    # LightGBM's own run already computed its dev_oos head-to-head via the
    # shared harness (same panel construction, same odds column) — reuse it
    # directly rather than re-deriving.
    for name, result in (("lgbm_indep", lgbm_indep), ("lgbm_market", lgbm_mkt)):
        h2h = dict(result["summary"]["head_to_head"])
        h2h.pop("odds_band_table", None)
        scorecard["candidates"][name] = h2h
        if market_metrics is None:
            market_metrics = h2h["market"]
        log(f"scorecard [{name}]: model_log_loss={h2h['model']['log_loss']:.5f} "
            f"market_log_loss={h2h['market']['log_loss']:.5f} "
            f"beats_market={h2h['model_beats_market_logloss']} n_races={h2h['n_races_kept']}")

    scorecard["market_only_baseline"] = market_metrics

    os.makedirs(os.path.dirname(OUT_SCORECARD), exist_ok=True)
    with open(OUT_SCORECARD, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    log(f"wrote {OUT_SCORECARD}")

    os.makedirs(os.path.dirname(OUT_PREDICTIONS), exist_ok=True)
    panel.to_parquet(OUT_PREDICTIONS, engine="pyarrow", index=False)
    log(f"wrote {OUT_PREDICTIONS} ({len(panel)} rows, {panel.shape[1]} cols)")

    print("RUN_BASELINES_10_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
