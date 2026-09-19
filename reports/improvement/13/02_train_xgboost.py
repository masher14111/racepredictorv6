"""Step 13: train XGBoost (indep + market) on dev_core, score it on the
reserved dev_oos eval panel, and merge its probabilities into a copy of step
12's eval_panel_scored.parquet (which already carries CatBoost/LightGBM/
conditional-logit/market columns on the identical 618-race panel).

Mirrors reports/improvement/10/03_run_baselines.py::run_catboost, swapping
models.train_xgboost for models.train (models.train_xgboost.train() is the
XGBoost analogue of models.train.train()). Never loads final_holdout_RESERVED_
UNTOUCHED.

Run: .venv/Scripts/python.exe reports/improvement/13/02_train_xgboost.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.head_to_head import head_to_head
from models.train_xgboost import load_frozen_model, score, train as train_xgboost

MANIFEST = "reports/improvement/13/manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
EVAL_PANEL_IN = "data/audit/12/eval_panel_scored.parquet"
EVAL_PANEL_OUT = "data/audit/13/eval_panel_scored_with_xgb.parquet"
MODEL_DIR = "data/audit/13/models_xgboost"
OUT_PARTIAL_SCORECARD = "reports/improvement/13/scorecard_13_xgboost_only.json"

N_TRIALS = 6

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def load_dev_core(manifest: dict) -> pd.DataFrame:
    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    win = win[win["_day"].notna()].reset_index(drop=True)

    w = manifest["windows"]["dev_core"]
    dev_core_end = pd.Timestamp(w["end_exclusive"])
    dev_core = win[win["_day"] < dev_core_end].reset_index(drop=True)
    log(f"dev_core: {len(dev_core)} rows / {dev_core['race_uid'].nunique()} races")
    del win, df
    return dev_core


def run_branch(branch: str, feature_cols, tag: str, dev_core: pd.DataFrame,
               eval_panel: pd.DataFrame) -> tuple[pd.Series, dict]:
    log(f"xgboost [{branch}]: training (target=won only, trials={N_TRIALS})")
    meta = train_xgboost(
        df=dev_core.copy(), targets=["won"], trials=N_TRIALS, model_dir=MODEL_DIR,
        feature_cols=feature_cols, version_tag=tag,
    )
    if meta is None:
        raise RuntimeError(f"xgboost [{branch}]: train() returned None (empty matrix)")
    log(f"xgboost [{branch}]: internal test AUC={meta['targets']['won']['test_auc']:.4f} "
        f"brier={meta['targets']['won']['test_brier']:.4f} "
        f"best_params={meta['targets']['won']['best_params']}")

    model_path = os.path.join(MODEL_DIR, f"xgboost_won_{tag}.json")
    frozen = load_frozen_model(model_path)

    # Re-derive the 618-race eval-panel rows straight from the rebuilt matrix
    # (same rows load_panel resolved in step 10/12), so scoring uses the exact
    # feature columns the model trained on.
    df = pd.read_parquet(MATRIX_PATH)
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    del df
    keyed = eval_panel[["race_uid", "horse_id"]].drop_duplicates()
    rows = win.merge(keyed, on=["race_uid", "horse_id"], how="inner")
    prob = score(frozen, rows)
    scored = pd.DataFrame({
        "race_uid": rows["race_uid"].to_numpy(), "horse_id": rows["horse_id"].to_numpy(),
        f"xgboost_{branch}_prob": np.asarray(prob, dtype=float),
    }).drop_duplicates(subset=["race_uid", "horse_id"])
    n_missing = keyed.merge(scored, on=["race_uid", "horse_id"], how="left")[f"xgboost_{branch}_prob"].isna().sum()
    if n_missing:
        log(f"  WARNING: xgboost_{branch}_prob: {n_missing}/{len(keyed)} eval rows had no score")
    return scored, meta


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    dev_core = load_dev_core(manifest)
    eval_panel = pd.read_parquet(EVAL_PANEL_IN)
    log(f"eval panel (from step 12): {len(eval_panel)} rows / {eval_panel['race_uid'].nunique()} races")

    scorecard = {"manifest": MANIFEST, "candidates": {}}
    panel = eval_panel.copy()
    for branch, cols, tag in (("indep", PRICE_FREE_FEATURE_COLS, "s13idp"),
                              ("market", FEATURE_COLS, "s13mkt")):
        scored, meta = run_branch(branch, cols, tag, dev_core, eval_panel)
        panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
        col = f"xgboost_{branch}_prob"
        h2h = head_to_head(panel.dropna(subset=[col]), prob_col=col, odds_col="bet_price",
                           race_id_col="race_uid", label_col="won")
        h2h.pop("odds_band_table")
        scorecard["candidates"][f"xgboost_{branch}"] = {"training_meta": meta["targets"]["won"], "head_to_head": h2h}
        log(f"scorecard [xgboost_{branch}]: model_log_loss={h2h['model']['log_loss']:.5f} "
            f"market_log_loss={h2h['market']['log_loss']:.5f} "
            f"beats_market={h2h['model_beats_market_logloss']} n_races={h2h['n_races_kept']}")

    os.makedirs(os.path.dirname(EVAL_PANEL_OUT), exist_ok=True)
    panel.to_parquet(EVAL_PANEL_OUT, engine="pyarrow", index=False)
    log(f"wrote {EVAL_PANEL_OUT} ({len(panel)} rows, {panel.shape[1]} cols)")

    with open(OUT_PARTIAL_SCORECARD, "w", encoding="utf-8") as fh:
        json.dump(scorecard, fh, indent=2, default=str)
    log(f"wrote {OUT_PARTIAL_SCORECARD}")
    print("TRAIN_XGBOOST_13_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
