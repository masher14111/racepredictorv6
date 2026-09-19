"""Step 17 — score the SERVED champion bundle on step 10's development panel.

Inference only: nothing is fitted, nothing under models/ is written. Runs the
frozen rule in ``02_selection_protocol.json``; the final holdout is asserted
absent from every frame that reaches a model.

Run:  .venv/Scripts/python.exe reports/improvement/17/03_champion_replay.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from backtest.data import PanelConfig, load_panel  # noqa: E402
from backtest.holdout import load_frozen_model, score as score_catboost  # noqa: E402
from features.lgbm_adapter import build_lgbm_matrix  # noqa: E402
from models.calibration import normalize_within_race  # noqa: E402
from models.head_to_head import head_to_head  # noqa: E402
from models.lgbm_softmax import LGBMSoftmaxModel  # noqa: E402

HERE = Path(__file__).parent
PROTOCOL = HERE / "02_selection_protocol.json"
OUT = HERE / "03_champion_replay.json"
PANEL_OUT = ROOT / "data/audit/17/dev_oos_champion_replay.parquet"
SEED, N_BOOT = 17, 2000


def dev_oos_frame() -> pd.DataFrame:
    manifest = json.loads((ROOT / "reports/improvement/10/run_manifest.json").read_text(encoding="utf-8"))
    w = manifest["windows"]
    start, end = pd.Timestamp(w["dev_oos"]["start"]), pd.Timestamp(w["dev_oos"]["end_exclusive"])
    assert end == pd.Timestamp(w["final_holdout_RESERVED_UNTOUCHED"]["start"])
    df = pd.read_parquet(ROOT / manifest["data"]["rebuilt_matrix_path"])
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    day = pd.to_datetime(win["race_date"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()
    ev = win[(day >= start) & (day < end)].reset_index(drop=True)
    del df, win
    ev_day = pd.to_datetime(ev["race_date"], utc=True).dt.tz_localize(None)
    assert ev_day.max() < end, "holdout row leaked into the replay frame"
    assert ev["race_uid"].nunique() == w["dev_oos"]["n_races"] and len(ev) == w["dev_oos"]["n_rows"]
    return ev


def per_race_loss(panel: pd.DataFrame, col: str) -> pd.Series:
    """−log(normalised winner probability) per race — head_to_head's own criterion."""
    p = normalize_within_race(panel[col].astype(float).to_numpy(), panel["race_uid"].to_numpy())
    won = panel["won"].astype(float).to_numpy().astype(bool)
    loss = pd.Series(-np.log(np.clip(p[won], 1e-15, 1.0)), index=panel.loc[won, "race_uid"].to_numpy())
    return loss.groupby(level=0).sum()


def paired_bootstrap(a: pd.Series, b: pd.Series) -> dict:
    """CI95 of mean(a − b) over whole races; negative => `a` has the lower (better) loss."""
    races = a.index.intersection(b.index)
    d = (a.loc[races] - b.loc[races]).to_numpy()
    rng = np.random.default_rng(SEED)
    means = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"n_races": int(len(d)), "mean_delta": float(d.mean()), "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "p_a_better": float((means < 0).mean())}


def main() -> int:
    assert PROTOCOL.exists(), "freeze 02_selection_protocol.json first"
    ev = dev_oos_frame()
    panel = load_panel(df=ev, config=PanelConfig(feature_cols=()))
    priced = ev.merge(panel[["race_uid", "horse_id", "bet_price"]].drop_duplicates(["race_uid", "horse_id"]),
                      on=["race_uid", "horse_id"], how="left")

    lines: dict[str, np.ndarray] = {}
    notes: dict[str, str] = {}

    # champion, price-free: base-calibrated CatBoost v3nf WITHOUT the favourite-longshot remap
    champ_nf = load_frozen_model(str(ROOT / "models/catboost_won_v3nf.bin"))
    nf_no_fl = copy.copy(champ_nf)
    nf_no_fl.fl_calibrator = None
    lines["champion_v3nf_indep"] = score_catboost(nf_no_fl, priced, price_col="bet_price")
    # champion, market-adjusted value line: same model THROUGH the FL remap on the reference price
    lines["champion_v3nf_fl_market_adjusted"] = score_catboost(champ_nf, priced, price_col="bet_price")
    # champion, market-assisted CatBoost v3
    lines["champion_v3_market"] = score_catboost(
        load_frozen_model(str(ROOT / "models/catboost_won_v3.bin")), priced, price_col="bet_price")
    # step 10 rebuilds (same family, same loader) — must reproduce step 10's 1.93620 / 1.71924
    for tag, name in (("s10idp", "step10_catboost_indep"), ("s10mkt", "step10_catboost_market")):
        frozen = load_frozen_model(str(ROOT / f"data/audit/10/models_catboost/catboost_won_{tag}.bin"))
        lines[name] = score_catboost(frozen, priced, price_col="bet_price")
    # champion LightGBM v3 (market-assisted grouped softmax), via its own saved feature list
    try:
        lgbm = LGBMSoftmaxModel.load(str(ROOT / "models/lgbm_won_v3.txt"))
        X, _, rid = build_lgbm_matrix(ev, inference=True, feature_cols=list(lgbm.feature_name))
        lines["champion_lgbm_v3_market"] = lgbm.predict_proba(X, rid)
    except Exception as exc:  # reported, never hidden
        notes["champion_lgbm_v3_market"] = f"UNSCORABLE: {type(exc).__name__}: {exc}"

    # train/serve feature skew: which champion inputs are absent or all-null in the repaired matrix?
    skew = {}
    for tag, fm in (("v3nf", champ_nf),):
        cols = fm.feature_cols
        skew[tag] = {"n_features": len(cols),
                     "absent_from_matrix": [c for c in cols if c not in ev.columns],
                     "all_null_on_panel": [c for c in cols if c in ev.columns and ev[c].isna().all()]}

    for name, prob in lines.items():
        scored = pd.DataFrame({"race_uid": ev["race_uid"].to_numpy(), "horse_id": ev["horse_id"].to_numpy(),
                               name: np.asarray(prob, dtype=float)}).drop_duplicates(["race_uid", "horse_id"])
        panel = panel.merge(scored, on=["race_uid", "horse_id"], how="left")

    complete = panel["bet_price"].notna().groupby(panel["race_uid"]).transform("all").to_numpy()
    scorecard = {}
    for name in lines:
        h = head_to_head(panel.dropna(subset=[name]), prob_col=name, odds_col="bet_price", race_id_col="race_uid")
        scorecard[name] = {"n_races": h["n_races_kept"], "n_runners": h["n_runners"],
                           "model_log_loss": h["model"]["log_loss"], "market_log_loss": h["market"]["log_loss"],
                           "gap_market_minus_model": h["gaps"]["log_loss"], "brier": h["model"]["brier_runner_level"],
                           "ece": h["model"]["ece"], "beats_market": bool(h["model_beats_market_logloss"])}
        print(f"{name:38s} LL {h['model']['log_loss']:.5f}  market {h['market']['log_loss']:.5f}  "
              f"gap {h['gaps']['log_loss']:+.5f}  races {h['n_races_kept']}")

    kept = panel[complete]
    loss = {name: per_race_loss(kept, name) for name in lines}
    comparisons = {
        "PRIMARY step10_catboost_indep minus champion_v3nf_indep":
            paired_bootstrap(loss["step10_catboost_indep"], loss["champion_v3nf_indep"]),
        "step10_catboost_market minus champion_v3_market":
            paired_bootstrap(loss["step10_catboost_market"], loss["champion_v3_market"]),
        "step10_catboost_market minus champion_v3nf_fl_market_adjusted":
            paired_bootstrap(loss["step10_catboost_market"], loss["champion_v3nf_fl_market_adjusted"]),
    }
    primary = comparisons["PRIMARY step10_catboost_indep minus champion_v3nf_indep"]
    rebuild_wins = primary["ci_excludes_zero"] and primary["mean_delta"] < 0
    selected = "step10_catboost_rebuild" if rebuild_wins else "champion_as_served"
    for k, v in comparisons.items():
        print(f"{k}: delta {v['mean_delta']:+.5f} CI95 [{v['ci95'][0]:+.5f},{v['ci95'][1]:+.5f}] n={v['n_races']}")
    print(f"SELECTED (frozen rule): {selected}")

    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUT, index=False)
    OUT.write_text(json.dumps({
        "protocol_frozen_at_utc": json.loads(PROTOCOL.read_text(encoding="utf-8"))["frozen_at_utc"],
        "panel": {"rows": int(len(panel)), "races": int(panel["race_uid"].nunique()),
                  "max_race_date": str(pd.to_datetime(panel["race_date"], utc=True).max().date())},
        "scorecard": scorecard, "unscorable": notes, "paired_bootstrap": comparisons,
        "champion_feature_skew": skew, "selected_by_frozen_rule": selected,
        "seed": SEED, "n_boot": N_BOOT, "panel_parquet": str(PANEL_OUT.relative_to(ROOT).as_posix()),
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
