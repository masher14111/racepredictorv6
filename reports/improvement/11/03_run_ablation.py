"""Step 11 / 3: does the measured-performance family actually add anything?

Baseline vs baseline + ``MEASURED_SPEED_FEATURE_COLS``, on step 10's FROZEN
windows and the SAME dev-OOS panel, so the only thing that differs between the
two arms of each pair is the feature set. Reuses step 10's run_manifest.json
rather than choosing new windows, and never loads a
``final_holdout_RESERVED_UNTOUCHED`` row into any fit, tuning fold or score.

Arms (paired; each pair differs only by the added family):
  * condlogit indep   / indep + measured
  * condlogit market  / market + measured
  * lgbm indep        / indep + measured

Metrics: race-level log loss, runner-level Brier and ECE (models.head_to_head),
plus a **race-clustered bootstrap** on the paired log-loss delta, which is the
whole-race uncertainty the acceptance criteria require -- a mean improvement
whose interval straddles zero is not evidence.

Writes reports/improvement/11/ablation_scorecard.json and
data/audit/11/dev_oos_predictions_measured.parquet. Champion artifacts and
step 10's outputs are never touched.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from backtest.data import PanelConfig, load_panel  # noqa: E402
from features.lgbm_adapter import (  # noqa: E402
    FINAL_FEATURE_COLS, INDEPENDENT_FEATURE_COLS, build_lgbm_matrix)
from models.conditional_logit import ConditionalLogitModel, tune_l2  # noqa: E402
from models.features import MEASURED_SPEED_FEATURE_COLS  # noqa: E402
from models.head_to_head import head_to_head  # noqa: E402
from models.train_lgbm import run_training as run_lgbm_training  # noqa: E402

MANIFEST_PATH = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/11/training_measured.parquet"
OUT_SCORECARD = "reports/improvement/11/ablation_scorecard.json"
OUT_PREDICTIONS = "data/audit/11/dev_oos_predictions_measured.parquet"

N_TRIALS = 6
CV_FOLDS = 5
N_BOOTSTRAP = 2000

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def load_data():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    win = win[win["_day"].notna()].reset_index(drop=True)

    w = manifest["windows"]
    dev_core_end = pd.Timestamp(w["dev_core"]["end_exclusive"])
    dev_oos_start = pd.Timestamp(w["dev_oos"]["start"])
    dev_oos_end = pd.Timestamp(w["dev_oos"]["end_exclusive"])
    assert dev_core_end == dev_oos_start
    assert dev_oos_end == pd.Timestamp(w["final_holdout_RESERVED_UNTOUCHED"]["start"])

    dev_core = win[win["_day"] < dev_core_end].reset_index(drop=True)
    dev_oos = win[(win["_day"] >= dev_oos_start)
                  & (win["_day"] < dev_oos_end)].reset_index(drop=True)
    # Holdout rows exist in `win` (one parquet read) but are never selected here
    # and never reach a fit, a CV fold or a score below.
    assert dev_core["_day"].max() < dev_oos_start
    assert dev_oos["_day"].max() < dev_oos_end
    log(f"dev_core {len(dev_core)} rows / {dev_core['race_uid'].nunique()} races; "
        f"dev_oos {len(dev_oos)} rows / {dev_oos['race_uid'].nunique()} races")
    del win, df
    return manifest, dev_core, dev_oos


def _merge_prob(panel, race_uid, horse_id, prob, colname):
    scored = pd.DataFrame({"race_uid": np.asarray(race_uid),
                           "horse_id": np.asarray(horse_id),
                           colname: np.asarray(prob, dtype=float)})
    scored = scored.drop_duplicates(subset=["race_uid", "horse_id"])
    merged = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
    missing = int(merged[colname].isna().sum())
    if missing:
        log(f"  WARNING {colname}: {missing}/{len(panel)} panel rows unscored")
    return merged


def run_condlogit(name, feature_cols, dev_core, dev_oos, panel):
    log(f"condlogit [{name}]: {len(feature_cols)} features")
    X_core, y_core, rid_core = build_lgbm_matrix(dev_core, inference=False,
                                                 feature_cols=feature_cols)
    keep = pd.to_numeric(dev_core["won"], errors="coerce").notna().to_numpy()
    order_core = dev_core.loc[keep, "race_date"].to_numpy()
    tuned = tune_l2(X_core, y_core, rid_core, order_core,
                    n_trials=N_TRIALS, n_splits=CV_FOLDS)
    log(f"condlogit [{name}]: l2={tuned['l2']:.5g} cv_log_loss={tuned['cv_log_loss']:.5f}")
    model = ConditionalLogitModel(l2=tuned["l2"]).fit(X_core, y_core, rid_core)
    out_path = f"data/audit/11/models_condlogit/{name}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    X_oos, _, rid_oos = build_lgbm_matrix(dev_oos, inference=True, feature_cols=feature_cols)
    prob = model.predict_proba(X_oos, rid_oos)
    panel = _merge_prob(panel, dev_oos["race_uid"], dev_oos["horse_id"], prob,
                        f"condlogit_{name}_prob")
    return panel, tuned


def run_lgbm(name, feature_cols, dev_core, dev_oos, manifest, panel):
    w = manifest["windows"]
    dev_all = pd.concat([dev_core, dev_oos], ignore_index=True)
    out_dir = f"data/audit/11/models_lgbm_{name}"
    log(f"lgbm [{name}]: {len(feature_cols)} features")
    result = run_lgbm_training(
        df=dev_all,
        output_path=f"{out_dir}/lgbm_won.txt",
        meta_path=f"{out_dir}/lgbm_v3_meta.json",
        holdout_out=f"{out_dir}/holdout_dev_oos",
        max_date=(pd.Timestamp(w["dev_core"]["end_exclusive"])
                  - pd.Timedelta(days=1)).date().isoformat(),
        holdout_start=w["dev_oos"]["start"],
        holdout_end=(pd.Timestamp(w["dev_oos"]["end_exclusive"])
                     - pd.Timedelta(days=1)).date().isoformat(),
        model_version=f"s11-{name}",
        model_params={"seed": 42},
        early_stopping_rounds=80,
        feature_cols=feature_cols,
    )
    h2h = result["summary"]["head_to_head"]
    log(f"lgbm [{name}]: dev_oos race log loss model={h2h['model']['log_loss']:.5f}")
    # Pull LightGBM's own dev-OOS probabilities onto the shared panel so the
    # bootstrap can pair them race-for-race (step 10 left this gap). The ledger
    # keys on horse_name, not horse_id, so join on (race_uid, horse_name).
    ledger = os.path.join(out_dir, "holdout_dev_oos", "ledger.csv")
    colname = f"lgbm_{name}_prob"
    if os.path.exists(ledger):
        led = pd.read_csv(ledger)
        pcol = next((c for c in ("norm_prob", "model_prob", "prob", "won_prob")
                     if c in led.columns), None)
        if pcol and {"race_uid", "horse_name"} <= set(led.columns):
            scored = (led[["race_uid", "horse_name", pcol]]
                      .rename(columns={pcol: colname})
                      .drop_duplicates(subset=["race_uid", "horse_name"]))
            panel = panel.merge(scored, on=["race_uid", "horse_name"], how="left")
            log(f"  lgbm [{name}]: merged {int(panel[colname].notna().sum())}"
                f"/{len(panel)} panel rows from ledger")
        else:
            log(f"  NOTE: lgbm [{name}] ledger lacks a usable prob/name column "
                f"({list(led.columns)[:12]}) -- bootstrap will skip this pair")
    return panel, result


def race_log_loss_by_race(panel, prob_col):
    """Per-race negative log probability of the actual winner(s)."""
    sub = panel.dropna(subset=[prob_col])
    out = {}
    for rid, grp in sub.groupby("race_uid", sort=False):
        p = grp[prob_col].to_numpy(dtype=float)
        y = grp["won"].to_numpy(dtype=float)
        total = p.sum()
        if total > 0:
            p = p / total  # score the normalized full-race book
        wp = p[y.astype(bool)]
        if wp.size:
            out[rid] = -float(np.log(np.clip(wp, 1e-12, 1.0)).sum())
    return out


def paired_bootstrap(panel, base_col, test_col, *, n=N_BOOTSTRAP, seed=11):
    """Race-clustered bootstrap of the paired log-loss delta (base - test).

    Positive mean = the added family improves the race log loss. Resampling
    whole races, not runners, is what keeps the interval honest: runners inside
    one race are not independent observations.
    """
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
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "prob_improvement": float((means > 0).mean()),
        "significant_improvement": bool(lo > 0),
    }


def main() -> int:
    for col in ("measured_speed_z", "measured_speed_pct", "beaten_lengths",
                "est_speed_yps", "winner_speed_yps"):
        assert col not in MEASURED_SPEED_FEATURE_COLS, \
            f"{col} describes the race being predicted and must never be a feature"

    manifest, dev_core, dev_oos = load_data()
    panel = load_panel(df=dev_oos, config=PanelConfig(feature_cols=()))
    log(f"dev_oos panel: {len(panel)} rows / {panel['race_uid'].nunique()} races")

    indep_measured = list(INDEPENDENT_FEATURE_COLS) + list(MEASURED_SPEED_FEATURE_COLS)
    market_measured = list(FINAL_FEATURE_COLS) + list(MEASURED_SPEED_FEATURE_COLS)

    scorecard = {
        "matrix": MATRIX_PATH, "manifest": MANIFEST_PATH,
        "measured_family": list(MEASURED_SPEED_FEATURE_COLS),
        "dev_oos_panel_rows": int(len(panel)),
        "dev_oos_panel_races": int(panel["race_uid"].nunique()),
        "tuning": {}, "candidates": {}, "paired_bootstrap": {},
    }

    for name, cols in (("indep_baseline", INDEPENDENT_FEATURE_COLS),
                       ("indep_measured", indep_measured),
                       ("market_baseline", FINAL_FEATURE_COLS),
                       ("market_measured", market_measured)):
        panel, tuned = run_condlogit(name, cols, dev_core, dev_oos, panel)
        scorecard["tuning"][f"condlogit_{name}"] = tuned

    lgbm_results = {}
    for name, cols in (("indep_baseline", INDEPENDENT_FEATURE_COLS),
                       ("indep_measured", indep_measured)):
        panel, result = run_lgbm(name, cols, dev_core, dev_oos, manifest, panel)
        lgbm_results[name] = result

    market_metrics = None
    for col in [c for c in panel.columns if c.endswith("_prob")]:
        if panel[col].isna().all():
            continue
        h2h = head_to_head(panel.dropna(subset=[col]), prob_col=col, odds_col="bet_price",
                           race_id_col="race_uid", label_col="won")
        h2h.pop("odds_band_table", None)
        scorecard["candidates"][col[:-5]] = h2h
        market_metrics = h2h["market"]
        log(f"score [{col[:-5]}]: log_loss={h2h['model']['log_loss']:.5f} "
            f"brier={h2h['model']['brier_runner_level']:.6f} "
            f"beats_market={h2h['model_beats_market_logloss']}")
    scorecard["market_only_baseline"] = market_metrics

    for label, base, test in (
        ("condlogit_indep", "condlogit_indep_baseline_prob", "condlogit_indep_measured_prob"),
        ("condlogit_market", "condlogit_market_baseline_prob", "condlogit_market_measured_prob"),
        ("lgbm_indep", "lgbm_indep_baseline_prob", "lgbm_indep_measured_prob"),
    ):
        if base in panel.columns and test in panel.columns:
            result = paired_bootstrap(panel, base, test)
            scorecard["paired_bootstrap"][label] = result
            if result.get("n_races"):
                log(f"bootstrap [{label}]: delta={result['mean_delta_base_minus_test']:+.5f} "
                    f"CI95=[{result['ci95_low']:+.5f},{result['ci95_high']:+.5f}] "
                    f"significant={result['significant_improvement']}")
        else:
            scorecard["paired_bootstrap"][label] = {"n_races": 0, "reason": "column missing"}

    os.makedirs(os.path.dirname(OUT_SCORECARD), exist_ok=True)
    with open(OUT_SCORECARD, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    panel.to_parquet(OUT_PREDICTIONS, engine="pyarrow", index=False)
    log(f"wrote {OUT_SCORECARD} and {OUT_PREDICTIONS}")
    print("ABLATION_11_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
