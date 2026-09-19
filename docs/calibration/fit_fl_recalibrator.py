"""Fit the favourite-longshot recalibrator and write the OOS before/after report.

Offline only — fits OddsBandCalibrator on a time-ordered TRAIN slice of the saved
walk-forward OOS frame, validates on the held-out tail (no leakage), saves the
portable artifact to models/, and writes docs/calibration/fl_recalibration_report.md.

Run:  python -m docs.calibration.fit_fl_recalibrator
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backtest import metrics
from models.calibration import OddsBandCalibrator

RUN = "data/backtests/20260616_195957/scored.parquet"
CUT = pd.Timestamp("2026-03-26", tz="UTC")  # fold-5 test_start
ARTIFACT = "models/fl_oddsband_v3nf_calib.pkl"
REPORT = "docs/calibration/fl_recalibration_report.md"
_EPS = 1e-6


def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece10(y, p):
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 0, 1)
    b = np.clip((p * 10).astype(int), 0, 9)
    return float(sum(((b == k).mean()) * abs(p[b == k].mean() - y[b == k].mean())
                     for k in range(10) if (b == k).any()))


def ae_rows(p, y, d):
    return metrics.ae_table(p, y, group_values=d, bands=metrics._ODDS_BANDS)


def md_ae_table(before, after):
    lines = ["| odds band | n | A/E before | A/E after |",
             "| --------- | - | ---------- | --------- |"]
    bmap = {r["bucket"]: r for r in before}
    amap = {r["bucket"]: r for r in after}
    for b in [x[0] for x in metrics._ODDS_BANDS]:
        rb, ra = bmap.get(b), amap.get(b)
        if rb is None:
            continue
        lines.append(f"| {b} | {rb['n']} | {rb['ae']} | "
                     f"{ra['ae'] if ra else '—'} |")
    return "\n".join(lines)


def main():
    df = pd.read_parquet(RUN)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    p = df["prob"].to_numpy(float)
    d = df["bet_price"].to_numpy(float)
    y = df["won"].to_numpy(float)

    tr = (df["race_date"] < CUT).to_numpy()
    te = ~tr
    cal = OddsBandCalibrator.fit(p[tr], d[tr], y[tr], method="isotonic")

    with open(ARTIFACT, "wb") as fh:
        pickle.dump(cal, fh)
    blob = open(ARTIFACT, "rb").read()
    assert b"numpy" not in blob, "artifact leaked a numpy object"

    # held-out tail metrics
    pte, dte, yte = p[te], d[te], y[te]
    cte = cal.predict(pte, dte)
    full_cal = cal.predict(p, d)  # full-frame for the headline A/E table

    def block(tag, yy, pp, dd):
        return (f"- **{tag}** — Brier {brier(yy, pp):.5f}, ECE10 {ece10(yy, pp):.5f}, "
                f"AUC {roc_auc_score(yy, pp):.4f}")

    report = f"""# Favourite-longshot recalibration — OOS report

Source frame: `{RUN}` (walk-forward, leak-free; bets at pre-off `ppwap`).
Train slice: `race_date < {CUT.date()}` (n={int(tr.sum())}). Held-out tail:
`race_date >= {CUT.date()}` (n={int(te.sum())}, base rate {yte.mean():.4f}).
Recalibrator: `OddsBandCalibrator` (9 fixed odds bands, isotonic per band, log-odds
blended). Artifact: `{ARTIFACT}`.

## Held-out tail — overall quality (before → after)

{block('RAW price-free prob', yte, pte, dte)}
{block('F-L recalibrated   ', yte, cte, dte)}

(AUC rises because the pre-off price — legitimately available before the off,
unlike the `odds_finish` close used only for CLV — is a strong predictor the
price-free model cannot see; the recalibrator folds it in *per band* without
discarding the model's within-price ranking.)

## A/E by odds band — full OOS frame (n={len(df)})

{md_ae_table(ae_rows(p, y, d), ae_rows(full_cal, y, d))}

## A/E by odds band — held-out tail only (n={int(te.sum())})

{md_ae_table(ae_rows(pte, yte, dte), ae_rows(cte, yte, dte))}

## Validation command

```
python -m docs.calibration.fit_fl_recalibrator   # re-fits + regenerates this report
python -m docs.calibration.fl_prototype          # full method comparison (logistic2 vs OBC)
```
"""
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print(f"\nSaved artifact -> {ARTIFACT}\nSaved report   -> {REPORT}")


if __name__ == "__main__":
    main()
