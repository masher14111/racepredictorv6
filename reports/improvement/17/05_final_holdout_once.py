"""Step 17 — the SINGLE use of step 10's reserved final holdout.

Scores the frozen candidate (``candidate_manifest.json``) on 2026-08-28..2026-09-17
exactly once. Inference only. No selection, tuning or gate change may follow from it.

  --rehearse   run the IDENTICAL code path on the dev_oos panel instead; must
               reproduce 03_champion_replay.json to 1e-9. Opens no holdout row.
  (no flag)    the real, once-only evaluation. Refuses if a result already exists.

Run:  .venv/Scripts/python.exe reports/improvement/17/05_final_holdout_once.py [--rehearse]
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).parent

from backtest.data import PanelConfig, load_panel  # noqa: E402
from backtest.holdout import load_frozen_model, score as score_catboost  # noqa: E402
from features.lgbm_adapter import build_lgbm_matrix  # noqa: E402
from models.head_to_head import head_to_head  # noqa: E402
from models.lgbm_softmax import LGBMSoftmaxModel  # noqa: E402

_spec = importlib.util.spec_from_file_location("replay03", HERE / "03_champion_replay.py")
replay03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(replay03)

MANIFEST = HERE / "candidate_manifest.json"
RESULT = HERE / "05_final_holdout_once.json"
LOCK = HERE / "05_final_holdout_once.lock.json"
PANEL_OUT = ROOT / "data/audit/17/final_holdout_scored.parquet"


def window_frame(which: str) -> pd.DataFrame:
    m10 = json.loads((ROOT / "reports/improvement/10/run_manifest.json").read_text(encoding="utf-8"))
    w = m10["windows"][which]
    start = pd.Timestamp(w["start"])
    end = pd.Timestamp(w["end_exclusive"]) if "end_exclusive" in w else pd.Timestamp(w["end"]) + pd.Timedelta(days=1)
    df = pd.read_parquet(ROOT / m10["data"]["rebuilt_matrix_path"])
    win = df[df["market_type"].astype(str).str.upper() == "WIN"]
    day = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    ev = win[(day >= start) & (day < end)].reset_index(drop=True)
    assert ev["race_uid"].nunique() == w["n_races"] and len(ev) == w["n_rows"], (len(ev), w)
    return ev


def score_all(ev: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    panel = load_panel(df=ev, config=PanelConfig(feature_cols=()))
    priced = ev.merge(panel[["race_uid", "horse_id", "bet_price"]].drop_duplicates(["race_uid", "horse_id"]),
                      on=["race_uid", "horse_id"], how="left")
    cand = ROOT / "data/audit/10/models_catboost"
    lines = {
        "s10idp": score_catboost(load_frozen_model(str(cand / "catboost_won_s10idp.bin")), priced, "bet_price"),
        "s10mkt": score_catboost(load_frozen_model(str(cand / "catboost_won_s10mkt.bin")), priced, "bet_price"),
    }
    champ_nf = load_frozen_model(str(ROOT / "models/catboost_won_v3nf.bin"))
    no_fl = copy.copy(champ_nf)
    no_fl.fl_calibrator = None
    lines["champion_v3nf_indep"] = score_catboost(no_fl, priced, "bet_price")
    lines["champion_v3nf_fl_market_adjusted"] = score_catboost(champ_nf, priced, "bet_price")
    lines["champion_v3_market"] = score_catboost(
        load_frozen_model(str(ROOT / "models/catboost_won_v3.bin")), priced, "bet_price")
    lgbm = LGBMSoftmaxModel.load(str(ROOT / "models/lgbm_won_v3.txt"))
    X, _, rid = build_lgbm_matrix(ev, inference=True, feature_cols=list(lgbm.feature_name))
    lines["champion_lgbm_v3_market"] = lgbm.predict_proba(X, rid)

    for name, prob in lines.items():
        scored = pd.DataFrame({"race_uid": ev["race_uid"].to_numpy(), "horse_id": ev["horse_id"].to_numpy(),
                               name: np.asarray(prob, dtype=float)}).drop_duplicates(["race_uid", "horse_id"])
        panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")

    card = {}
    for name in lines:
        h = head_to_head(panel.dropna(subset=[name]), prob_col=name, odds_col="bet_price", race_id_col="race_uid")
        card[name] = {"n_races_total": h["n_races_total"], "n_races": h["n_races_kept"], "n_runners": h["n_runners"],
                      "model_log_loss": h["model"]["log_loss"], "market_log_loss": h["market"]["log_loss"],
                      "gap_market_minus_model": h["gaps"]["log_loss"], "brier": h["model"]["brier_runner_level"],
                      "market_brier": h["market"]["brier_runner_level"], "ece": h["model"]["ece"],
                      "beats_market": bool(h["model_beats_market_logloss"])}
    complete = panel["bet_price"].notna().groupby(panel["race_uid"]).transform("all").to_numpy()
    kept = panel[complete]
    loss = {n: replay03.per_race_loss(kept, n) for n in lines}
    from models.devig import devig
    kept = kept.assign(_mkt=devig(kept["bet_price"], kept["race_uid"], method="proportional"))
    loss["market"] = replay03.per_race_loss(kept, "_mkt")
    boot = {
        "PRIMARY s10idp minus market (negative = candidate beats the market)":
            replay03.paired_bootstrap(loss["s10idp"], loss["market"]),
        "s10mkt minus market": replay03.paired_bootstrap(loss["s10mkt"], loss["market"]),
        "s10idp minus champion_v3nf_indep (negative = rebuild better)":
            replay03.paired_bootstrap(loss["s10idp"], loss["champion_v3nf_indep"]),
        "s10mkt minus champion_v3_market": replay03.paired_bootstrap(loss["s10mkt"], loss["champion_v3_market"]),
        "s10mkt minus champion_v3nf_fl_market_adjusted":
            replay03.paired_bootstrap(loss["s10mkt"], loss["champion_v3nf_fl_market_adjusted"]),
    }
    return panel, card, boot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearse", action="store_true")
    args = ap.parse_args()
    assert MANIFEST.exists(), "freeze candidate_manifest.json first"
    manifest_sha = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    if args.rehearse:
        _, card, _ = score_all(window_frame("dev_oos"))
        ref = json.loads((HERE / "03_champion_replay.json").read_text(encoding="utf-8"))["scorecard"]
        pairs = {"s10idp": "step10_catboost_indep", "s10mkt": "step10_catboost_market",
                 "champion_v3nf_indep": "champion_v3nf_indep", "champion_v3_market": "champion_v3_market",
                 "champion_v3nf_fl_market_adjusted": "champion_v3nf_fl_market_adjusted",
                 "champion_lgbm_v3_market": "champion_lgbm_v3_market"}
        worst = max(abs(card[a]["model_log_loss"] - ref[b]["model_log_loss"]) for a, b in pairs.items())
        print(f"REHEARSAL on dev_oos: max |delta| vs 03_champion_replay = {worst:.3e}")
        (HERE / "05_rehearsal.json").write_text(json.dumps({"max_abs_delta": worst, "scorecard": card}, indent=1),
                                                 encoding="utf-8")
        return 0 if worst < 1e-9 else 1

    if RESULT.exists():
        print(f"REFUSING: the final holdout was already used once ({RESULT.name} exists). It cannot be re-run.")
        return 2
    attempts = json.loads(LOCK.read_text(encoding="utf-8"))["attempts"] if LOCK.exists() else []
    attempts.append({"started_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_sha})
    LOCK.write_text(json.dumps({"attempts": attempts}, indent=1), encoding="utf-8")

    panel, card, boot = score_all(window_frame("final_holdout_RESERVED_UNTOUCHED"))
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUT, index=False)
    primary = boot["PRIMARY s10idp minus market (negative = candidate beats the market)"]
    RESULT.write_text(json.dumps({
        "used_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest_sha,
        "window": "final_holdout_RESERVED_UNTOUCHED 2026-08-28..2026-09-17",
        "panel": {"rows": int(len(panel)), "races": int(panel["race_uid"].nunique())},
        "scorecard": card, "paired_bootstrap": boot,
        "primary_endpoint_candidate_beats_market": bool(primary["ci_excludes_zero"] and primary["mean_delta"] < 0),
        "holdout_status_after_this_run": "CONSUMED - no untouched holdout remains; further evidence must be prospective",
        "panel_parquet": PANEL_OUT.relative_to(ROOT).as_posix(),
    }, indent=1), encoding="utf-8")
    for name, c in card.items():
        print(f"{name:36s} LL {c['model_log_loss']:.5f}  market {c['market_log_loss']:.5f}  "
              f"gap {c['gap_market_minus_model']:+.5f}  races {c['n_races']}/{c['n_races_total']}")
    for k, v in boot.items():
        print(f"{k}: {v['mean_delta']:+.5f} CI95 [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] n={v['n_races']}")
    print(f"wrote {RESULT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
