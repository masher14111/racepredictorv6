"""calib-fl-04: re-validate the F-L recalibration on CLV + A/E and re-tune value gates.

Leak-free design
----------------
* `prob` in the saved scored.parquet is already walk-forward OOS (model fit on
  train-only per fold).
* The recalibrated prob = OddsBandCalibrator.predict(prob, bet_price). `bet_price`
  is the pre-off execution price (ppwap) — exactly the price live scoring feeds the
  recalibrator via `_effective_decimal`, and the same price EV uses. It is NOT the
  close (`close_price` = BSP), which we reserve for CLV only.
* The recalibrator artifact was fit on folds 0-4 (race_date < 2026-03-26). So the
  genuinely-held-out evaluation of the *recalibration* is fold 5
  (race_date >= 2026-03-26, n~12.9k). Full-frame numbers are reported too but are
  partly recalibrator-in-sample.
"""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd
from dataclasses import replace

from models.value import evaluate_filter, ValueConfig

RUN = "data/backtests/20260616_195957"
TAIL_CUT = pd.Timestamp("2026-03-26", tz="UTC")

scored = pd.read_parquet(f"{RUN}/scored.parquet")
scored["race_date"] = pd.to_datetime(scored["race_date"])
with open("models/fl_oddsband_v3nf_calib.pkl", "rb") as f:
    obc = pickle.load(f)

# Recalibrated prob, applied exactly as live scoring does (price = bet_price).
recal = scored.copy()
recal["prob"] = obc.predict(
    scored["prob"].to_numpy(float), scored["bet_price"].to_numpy(float)
)

frames = {
    "FULL": (scored, recal),
    "TAIL": (scored[scored.race_date >= TAIL_CUT].copy(),
             recal[recal.race_date >= TAIL_CUT].copy()),
}

BASE = ValueConfig.from_config()  # validated defaults


def fmt_ae(ae):
    return " ".join(
        f"{b['bucket']}:{(b['ae'] if b['ae'] is not None else float('nan')):.2f}(n{b['n']})"
        for b in ae
    )


def row(tag, cfg, frame):
    r = evaluate_filter(frame, cfg)
    return {
        "tag": tag, "n": r["n_bets"], "yield%": r["yield_pct"],
        "CLV%": r["clv_pct_mean"], "beat_close": r["beat_close_rate"],
        "hit": r["hit_rate"], "avg_odds": r["avg_odds"],
        "flat_bank": r["flat_final_bankroll"], "kelly_bank": r["kelly_final_bankroll"],
        "ae": r["ae_by_odds"],
    }


_OUT = []


def show(title, rows):
    _OUT.append(f"\n{'='*100}\n{title}\n{'='*100}")
    _OUT.append(f"{'config':<34}{'n':>6}{'yld%':>8}{'CLV%':>8}{'beat':>7}{'hit':>7}{'odds':>7}{'flat':>8}{'kelly':>9}")
    for r in rows:
        _OUT.append(f"{r['tag']:<34}{r['n']:>6}{_s(r['yield%']):>8}{_s(r['CLV%']):>8}"
                    f"{_s(r['beat_close']):>7}{_s(r['hit']):>7}{_s(r['avg_odds']):>7}"
                    f"{_s(r['flat_bank']):>8}{_s(r['kelly_bank']):>9}")
        _OUT.append(f"    A/E: {fmt_ae(r['ae'])}")


def _s(v):
    return "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))


# ── 1) Baseline reproduction (raw prob) vs recalibrated, validated default gate ──
for scope in ("FULL", "TAIL"):
    raw_f, rec_f = frames[scope]
    rows = [
        row(f"[{scope}] RAW  ev>=.05 band[2,6]", BASE, raw_f),
        row(f"[{scope}] RECAL ev>=.05 band[2,6]", BASE, rec_f),
    ]
    show(f"DEFAULT GATE — raw vs recalibrated ({scope})", rows)

# ── 2) Grid over min_ev × odds band on RECALIBRATED prob (edge gates OFF) ──
bands = [(1.01, 999.0), (2.0, 4.0), (2.0, 6.0), (2.0, 8.0), (2.0, 13.0),
         (3.0, 8.0), (4.0, 13.0), (1.5, 6.0)]
evs = [0.0, 0.05, 0.10, 0.15, 0.20]
for scope in ("TAIL", "FULL"):
    _, rec_f = frames[scope]
    rows = []
    for lo, hi in bands:
        for ev in evs:
            cfg = replace(BASE, min_odds=lo, max_odds=hi, min_ev=ev,
                          min_edge_pct=0.0, min_abs_edge=0.0)
            rows.append(row(f"band[{lo},{hi}] ev>={ev}", cfg, rec_f))
    show(f"GRID min_ev x band — RECALIBRATED ({scope}), edge gates OFF", rows)

# ── 3) Re-confirm edge gates harmful (recalibrated, best band) ──
for scope in ("TAIL", "FULL"):
    _, rec_f = frames[scope]
    rows = []
    for mae in (0.0, 0.02, 0.05):
        for mep in (0.0, 0.10, 0.25):
            cfg = replace(BASE, min_odds=2.0, max_odds=6.0, min_ev=0.05,
                          min_abs_edge=mae, min_edge_pct=mep)
            rows.append(row(f"band[2,6] ev.05 absedge{mae} pctedge{mep}", cfg, rec_f))
    show(f"EDGE-GATE re-confirm — RECALIBRATED ({scope})", rows)

with open("scripts/_fl04_results.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(_OUT))
print("\n".join(_OUT))
