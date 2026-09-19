"""Prototype + evaluation harness for the favourite-longshot (F-L) recalibration.

Offline only. Loads the saved walk-forward OOS frame, quantifies the F-L bias,
then fits and compares candidate recalibrators on a time-ordered held-out tail
(NO leakage). Picks by held-out Brier/ECE AND A/E flattening toward 1.0.

Run:  python -m docs.calibration.fl_prototype
(or)  python docs/calibration/fl_prototype.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from backtest import metrics

RUN = "data/backtests/20260616_195957/scored.parquet"
CUT = pd.Timestamp("2026-03-26", tz="UTC")  # fold-5 test_start: train < CUT, tail >= CUT
_EPS = 1e-6


# ── metrics ────────────────────────────────────────────────────────────────--
def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def ece_eqwidth(y, p, n_bins=10):
    """Equal-width 10-bin ECE — matches backtest.engine / ensemble_experiment."""
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(out)


def ece_eqmass(y, p, n_bins=15):
    """Equal-mass ECE — sharper where the mass is (probs all in [0,0.5])."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = 0.0
    n = len(p)
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            out += len(chunk) / n * abs(p[chunk].mean() - y[chunk].mean())
    return float(out)


def ae_band_dict(p, y, d):
    return {r["bucket"]: r["ae"]
            for r in metrics.ae_table(p, y, group_values=d, bands=metrics._ODDS_BANDS)}


def ae_spread(p, y, d):
    """Max |log A/E| across odds bands — 0 == perfectly flat at 1.0."""
    rows = metrics.ae_table(p, y, group_values=d, bands=metrics._ODDS_BANDS)
    vals = [r["ae"] for r in rows if r["ae"] and r["ae"] > 0]
    return float(max(abs(np.log(v)) for v in vals)) if vals else float("nan")


def report(tag, y, p, d):
    return dict(tag=tag, brier=brier(y, p), ece=ece_eqwidth(y, p),
               ece_mass=ece_eqmass(y, p), auc=roc_auc_score(y, p),
               ae_spread=ae_spread(p, y, d), ae=ae_band_dict(p, y, d))


# ── candidate recalibrators ──────────────────────────────────────────────────
def logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def fit_logistic2(p, d, y):
    """2-feature logistic: sigmoid(a*logit(p) + b*log(d) + c)."""
    X = np.column_stack([logit(p), np.log(np.clip(d, 1.0 + _EPS, None))])
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(X, np.asarray(y, int))
    return lr


def pred_logistic2(lr, p, d):
    X = np.column_stack([logit(p), np.log(np.clip(d, 1.0 + _EPS, None))])
    return lr.predict_proba(X)[:, 1]


def main():
    df = pd.read_parquet(RUN)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    p = df["prob"].to_numpy(float)
    d = df["bet_price"].to_numpy(float)
    y = df["won"].to_numpy(float)

    print(f"FULL FRAME  n={len(df)}  base={y.mean():.4f}")
    print(f"  AUC={roc_auc_score(y,p):.4f}  Brier={brier(y,p):.4f}  "
          f"ECE10={ece_eqwidth(y,p):.4f}  ECEmass={ece_eqmass(y,p):.4f}")
    print("  A/E by odds band:", ae_band_dict(p, y, d))

    tr = df["race_date"] < CUT
    te = ~tr
    ptr, dtr, ytr = p[tr.values], d[tr.values], y[tr.values]
    pte, dte, yte = p[te.values], d[te.values], y[te.values]
    print(f"\nSPLIT  train n={tr.sum()} (<{CUT.date()})  tail n={te.sum()} (>={CUT.date()})")
    print(f"  tail base={yte.mean():.4f}")

    rows = [report("RAW (uncalibrated)", yte, pte, dte)]

    # --- Approach B: 2-feature logistic ---
    lr = fit_logistic2(ptr, dtr, ytr)
    print(f"\n[logistic2] coef(logit_p)={lr.coef_[0,0]:.4f}  "
          f"coef(log_d)={lr.coef_[0,1]:.4f}  intercept={lr.intercept_[0]:.4f}")
    rows.append(report("logistic2 (prob+log-odds)", yte, pred_logistic2(lr, pte, dte), dte))

    # --- Approach A: per-odds-band isotonic with log-odds blending ---
    from models.calibration import OddsBandCalibrator
    EDGES_6 = [1.0, 2.0, 4.0, 8.0, 16.0, 34.0, float("inf")]
    EDGES_9 = OddsBandCalibrator.DEFAULT_EDGES
    BANDS_FINE = [("", lo, hi) for lo, hi in zip(EDGES_9[:-1], EDGES_9[1:])]
    obc_models = {}
    for edges, method, label in [
        (EDGES_6, "isotonic", "OBC 6-band isotonic"),
        (EDGES_9, "isotonic", "OBC 9-band(fixed) isotonic"),
        (EDGES_9, "auto", "OBC 9-band(fixed) auto"),
    ]:
        obc = OddsBandCalibrator.fit(ptr, dtr, ytr, edges=edges, method=method)
        obc_models[label] = obc
        rows.append(report(label, yte, obc.predict(pte, dte), dte))

    # --- table ---
    print(f"\n{'method':<30}{'Brier':>9}{'ECE10':>9}{'ECEmass':>9}{'AUC':>8}{'AEspread':>10}")
    for r in rows:
        print(f"{r['tag']:<30}{r['brier']:>9.5f}{r['ece']:>9.5f}"
              f"{r['ece_mass']:>9.5f}{r['auc']:>8.4f}{r['ae_spread']:>10.4f}")
    print(f"\n{'method':<30}" + "".join(f"{b.split()[0]:>10}" for b in rows[0]['ae']))
    for r in rows:
        print(f"{r['tag']:<30}" + "".join(
            f"{(r['ae'].get(b) if r['ae'].get(b) is not None else float('nan')):>10.3f}"
            for b in rows[0]['ae']))

    # --- value-signal retention: AUC within a NARROW odds slice [3,5] on tail ---
    # In a narrow price slice, odds barely varies, so any ranking power comes from
    # the MODEL, not the market. logistic2 (which discards the model) collapses to
    # ~0.5 here; a good value recalibrator keeps the model's within-price AUC.
    sl = (dte >= 3.8) & (dte < 4.2)  # TIGHT slice: odds ~const, so AUC = model signal
    rawp = pte[sl]
    print(f"\nValue-signal check — TIGHT odds slice [3.8,4.2) on tail (n={sl.sum()}):")
    print(f"  metric                  within-slice-AUC   corr(calib_p, raw_model_p)")
    print(f"  RAW model prob          {roc_auc_score(yte[sl], rawp):.4f}             1.0000")
    lp = pred_logistic2(lr, rawp, dte[sl])
    print(f"  logistic2               {roc_auc_score(yte[sl], lp):.4f}             {np.corrcoef(lp, rawp)[0,1]:.4f}")
    for label, obc in obc_models.items():
        op = obc.predict(rawp, dte[sl])
        print(f"  {label:<24}{roc_auc_score(yte[sl], op):.4f}             {np.corrcoef(op, rawp)[0,1]:.4f}")

    # per-band train row/win counts for the winning design
    print("\nPer-band train counts (9-band fixed):")
    for _, lo, hi in BANDS_FINE:
        m = (dtr >= lo) & (dtr < hi)
        print(f"  [{lo:>4},{hi if np.isfinite(hi) else 'inf':>4}) n={int(m.sum()):>6} wins={int(ytr[m].sum()):>5} rate={ytr[m].mean():.4f}")


if __name__ == "__main__":
    main()
