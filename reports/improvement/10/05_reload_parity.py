"""Step 10 acceptance: verify reload/inference parity for every candidate saved
by 03_run_baselines.py. CatBoost already reloads from disk for its dev_oos score
(backtest.holdout.load_frozen_model); this checks the two families that scored
dev_oos from the in-memory object: conditional logit and LightGBM softmax.

Read-only against already-trained artifacts + the frozen dev_oos slice.
Run: .venv/Scripts/python.exe reports/improvement/10/05_reload_parity.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import numpy as np
import pandas as pd

from backtest.data import PanelConfig, load_panel
from features.lgbm_adapter import INDEPENDENT_FEATURE_COLS, FINAL_FEATURE_COLS, build_lgbm_matrix
from models.conditional_logit import ConditionalLogitModel
from models.head_to_head import head_to_head
from models.lgbm_softmax import LGBMSoftmaxModel

MANIFEST_PATH = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"

with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
    manifest = json.load(fh)
w = manifest["windows"]

df = pd.read_parquet(MATRIX_PATH)
win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
win = win[win["_day"].notna()]
dev_oos_start = pd.Timestamp(w["dev_oos"]["start"])
dev_oos_end = pd.Timestamp(w["dev_oos"]["end_exclusive"])
dev_oos = win[(win["_day"] >= dev_oos_start) & (win["_day"] < dev_oos_end)].reset_index(drop=True)

fails = []


def check(name, fresh, reloaded, atol=1e-9):
    fresh = np.asarray(fresh, dtype=float)
    reloaded = np.asarray(reloaded, dtype=float)
    max_abs = float(np.max(np.abs(fresh - reloaded))) if len(fresh) else float("nan")
    ok = len(fresh) == len(reloaded) and np.allclose(fresh, reloaded, atol=atol, equal_nan=True)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: n={len(fresh)} max|delta|={max_abs:.3g}")
    if not ok:
        fails.append(name)


# ── conditional logit ────────────────────────────────────────────────────────
for branch, feature_cols in (("indep", INDEPENDENT_FEATURE_COLS), ("market", FINAL_FEATURE_COLS)):
    path = f"data/audit/10/models_condlogit/{branch}.json"
    X_oos, _, rid_oos = build_lgbm_matrix(dev_oos, inference=True, feature_cols=feature_cols)
    with open(path, "r", encoding="utf-8") as fh:
        saved = json.load(fh)
    l2 = saved.get("l2", saved.get("params", {}).get("l2"))
    # Re-fit is NOT reload; use the model's own load() to test true reload parity.
    reloaded = ConditionalLogitModel.load(path)
    prob_reload = reloaded.predict_proba(X_oos, rid_oos)
    # cross-check the saved coefficients directly reproduce the scorecard's own
    # dev_oos_predictions.parquet column for this branch.
    preds = pd.read_parquet("data/audit/10/dev_oos_predictions.parquet")
    col = f"condlogit_{branch}_prob"
    scored = pd.DataFrame({"race_uid": np.asarray(dev_oos["race_uid"]),
                            "horse_id": np.asarray(dev_oos["horse_id"]),
                            "prob_reload": prob_reload}).drop_duplicates(["race_uid", "horse_id"])
    merged = preds[["race_uid", "horse_id", col]].merge(scored, on=["race_uid", "horse_id"], how="inner")
    check(f"condlogit[{branch}] reload vs scorecard", merged[col], merged["prob_reload"])

# ── lgbm softmax ─────────────────────────────────────────────────────────────
# The training run scored dev_oos with the in-memory model right after fit();
# reload from disk and recompute the SAME head-to-head log loss recorded in
# scorecard.json — an exact match proves the saved artifact, not the in-memory
# object, is what live inference would actually load.
with open("reports/improvement/10/scorecard.json", "r", encoding="utf-8") as fh:
    scorecard = json.load(fh)
panel = load_panel(df=dev_oos, config=PanelConfig(feature_cols=()))

for branch, feature_cols in (("indep", INDEPENDENT_FEATURE_COLS), ("market", FINAL_FEATURE_COLS)):
    model_path = f"data/audit/10/models_lgbm_{branch}/lgbm_won.txt"
    reloaded = LGBMSoftmaxModel.load(model_path)
    X_oos, _, rid_oos = build_lgbm_matrix(dev_oos, inference=True, feature_cols=feature_cols)
    prob_reload = reloaded.predict_proba(X_oos, rid_oos)
    scored = pd.DataFrame({"race_uid": np.asarray(dev_oos["race_uid"]),
                            "horse_id": np.asarray(dev_oos["horse_id"]),
                            "prob_reload": prob_reload}).drop_duplicates(["race_uid", "horse_id"])
    merged = panel.merge(scored, on=["race_uid", "horse_id"], how="left")
    h2h = head_to_head(merged.dropna(subset=["prob_reload"]), prob_col="prob_reload",
                        odds_col="bet_price", race_id_col="race_uid", label_col="won")
    reload_ll = h2h["model"]["log_loss"]
    scorecard_ll = scorecard["candidates"][f"lgbm_{branch}"]["model"]["log_loss"]
    ok = abs(reload_ll - scorecard_ll) < 1e-6
    print(f"[{'PASS' if ok else 'FAIL'}] lgbm[{branch}] reload vs scorecard: "
          f"reload_log_loss={reload_ll:.8f} scorecard_log_loss={scorecard_ll:.8f} "
          f"|delta|={abs(reload_ll - scorecard_ll):.3g}")
    if not ok:
        fails.append(f"lgbm[{branch}]")

print()
if fails:
    print(f"RELOAD_PARITY: FAIL ({len(fails)}): {fails}")
    raise SystemExit(1)
print("RELOAD_PARITY: PASS — every saved artifact reproduces its own scored dev_oos probabilities on reload")
