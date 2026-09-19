"""Step 12: build the chronological walk-forward out-of-sample panel that the
power blend is SELECTED on and the calibrators are FIT on.

For each frozen block [b_i, b_i+1) in blend_manifest.json's walk_forward
boundaries, a conditional logit is fitted on every labelled WIN row with day
< b_i (whole races only, expanding window) and used to score the block. Both
branches are produced: the price-free INDEPENDENT line and the market-assisted
line. L2 is reused verbatim from step 10's tuning — nothing is re-tuned here, so
this panel introduces no new selection.

The reserved evaluation panel (2026-08-08..) and the final holdout are never
loaded into any fit or score in this script.

Writes data/audit/12/walkforward_oos.parquet.
Run: .venv/Scripts/python.exe reports/improvement/12/02_walkforward_oos.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from features.lgbm_adapter import FINAL_FEATURE_COLS, INDEPENDENT_FEATURE_COLS, build_lgbm_matrix
from models.conditional_logit import ConditionalLogitModel
from models.devig import devig

MANIFEST = "reports/improvement/12/blend_manifest.json"
OUT = "data/audit/12/walkforward_oos.parquet"
MODEL_DIR = "data/audit/12/models_condlogit_wf"

# Regime / breakdown columns carried through the panel alongside the ids.
CARRY_COLS = ("region", "going_band", "going_is_all_weather", "distance_furlongs",
              "field_size")

t0 = time.monotonic()


def log(msg: str) -> None:
    print(f"[{time.monotonic() - t0:7.1f}s] {msg}", flush=True)


def main() -> None:
    manifest = json.load(open(MANIFEST, "r", encoding="utf-8"))
    bounds = [pd.Timestamp(b) for b in manifest["walk_forward"]["boundaries"]]
    eval_start = pd.Timestamp(manifest["windows"]["eval"]["start"])
    assert bounds[-1] == eval_start, "walk-forward must stop at the evaluation panel"

    s10 = json.load(open("reports/improvement/10/scorecard.json", "r", encoding="utf-8"))
    l2 = {"indep": s10["conditional_logit_tuning"]["independent"]["l2"],
          "market": s10["conditional_logit_tuning"]["market"]["l2"]}
    log(f"reusing step-10 tuned l2: {l2}")

    df = pd.read_parquet(manifest["data"]["rebuilt_matrix_path"])
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    # Hard cut: nothing at or after the evaluation panel ever enters this script.
    win = win[win["_day"].notna() & (win["_day"] < eval_start)].reset_index(drop=True)
    win = win[pd.to_numeric(win["won"], errors="coerce").notna()].reset_index(drop=True)
    del df
    log(f"development pool: {len(win)} rows / {win['race_uid'].nunique()} races "
        f"({win['_day'].min().date()}..{win['_day'].max().date()})")

    branches = {"indep": INDEPENDENT_FEATURE_COLS, "market": FINAL_FEATURE_COLS}
    parts = []
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        train = win[win["_day"] < lo]
        block = win[(win["_day"] >= lo) & (win["_day"] < hi)]
        if block.empty:
            log(f"block {i} {lo.date()}..{hi.date()}: EMPTY, skipped")
            continue
        panel = load_panel(df=block, config=PanelConfig(feature_cols=CARRY_COLS))
        log(f"block {i} {lo.date()}..{hi.date()}: train {len(train)} rows / "
            f"{train['race_uid'].nunique()} races -> panel {len(panel)} rows / "
            f"{panel['race_uid'].nunique()} races")

        for branch, cols in branches.items():
            X_tr, y_tr, rid_tr = build_lgbm_matrix(train, inference=False, feature_cols=cols)
            model = ConditionalLogitModel(l2=l2[branch]).fit(X_tr, y_tr, rid_tr)
            model.save(os.path.join(MODEL_DIR, f"block{i:02d}_{branch}.json"))
            X_te, _, rid_te = build_lgbm_matrix(block, inference=True, feature_cols=cols)
            prob = model.predict_proba(X_te, rid_te)
            scored = pd.DataFrame({
                "race_uid": block["race_uid"].to_numpy(),
                "horse_id": block["horse_id"].to_numpy(),
                f"condlogit_{branch}_prob": np.asarray(prob, dtype=float),
            }).drop_duplicates(subset=["race_uid", "horse_id"])
            panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
            n_missing = int(panel[f"condlogit_{branch}_prob"].isna().sum())
            log(f"  [{branch}] fitted on {model.metadata['n_races']} races, "
                f"converged={model.metadata['converged']}, unscored panel rows={n_missing}")

        panel["wf_block"] = i
        panel["wf_block_start"] = lo
        parts.append(panel)

    out = pd.concat(parts, ignore_index=True).sort_values("race_date").reset_index(drop=True)
    # REFERENCE market probability: de-vigged pre-off reference book, complete
    # books only (partial books come back NaN by construction).
    out["market_ref_prob"] = devig(out["bet_price"], out["race_uid"], method="proportional")

    blend_dev_end = pd.Timestamp(manifest["windows"]["blend_dev"]["end_exclusive"])
    day = out["race_date"].dt.tz_localize(None).dt.normalize()
    out["period"] = np.where(day < blend_dev_end, "blend_dev", "calib")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_parquet(OUT, engine="pyarrow", index=False)
    log(f"wrote {OUT}: {len(out)} rows / {out['race_uid'].nunique()} races, "
        f"{out.shape[1]} cols")
    for period, g in out.groupby("period"):
        complete = g["market_ref_prob"].notna()
        log(f"  {period}: {len(g)} rows / {g['race_uid'].nunique()} races; "
            f"complete reference book on {int(complete.sum())} rows / "
            f"{g.loc[complete, 'race_uid'].nunique()} races")
    print("WALKFORWARD_12_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
