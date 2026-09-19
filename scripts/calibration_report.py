"""Before/after calibration report on the held-out tail.

Reproduces train.py's whole-race chronological split (test = last ~20% of rows,
rounded to whole race_uid groups — models.split_utils, step 02), then
for each model (v3 priced, v3nf price-free) and target compares three probability
variants on the SAME held-out test set:

    raw           — model.predict_proba (no calibrator)
    calibrated    — raw mapped through the persisted calibrator
    cal+norm      — calibrated, then within-race normalized (win field sums to 1)

Reports Brier, ECE, AUC and writes reliability-curve PNGs to docs/calibration/.

Within-race normalization needs a per-race key, but historical race_time is
100% null (audit C3). We recover an approximate key from (race_date, venue,
distance): median field ~12 vs ~62 for venue-day. Normalization numbers are
therefore directional, not exact; the serving path uses the real (venue,
race_time) key.

    python -m scripts.calibration_report
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from models.calibration import brier_score, normalize_within_race
from models.targets import add_targets
from models.train import _time_split

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_TRAINING = os.path.join(_BASE, "data", "features", "training.parquet")
_MODELS = os.path.join(_BASE, "models")
_OUT = os.path.join(_BASE, "docs", "calibration")
_TARGETS = ["won", "placed_2", "showed"]
_TEST_SIZE = 0.20
_N_BINS = 10


def _ece(y, p, n_bins=_N_BINS):
    """Expected Calibration Error: count-weighted mean |confidence - accuracy|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(p[m].mean() - y[m].mean())
    return float(ece)


def _reliability(y, p, n_bins=_N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def _load(tag):
    meta = json.load(open(os.path.join(_MODELS, f"catboost_{tag}_meta.json")))
    return meta["feature_cols"]


def _calibrator(tag, target):
    path = os.path.join(_MODELS, f"catboost_{target}_{tag}_calib.pkl")
    return pickle.load(open(path, "rb")) if os.path.exists(path) else None


def main():
    os.makedirs(_OUT, exist_ok=True)
    # Whole-race chronological split (models.train._time_split, step 02) — the
    # SAME split train.py itself uses, so this report scores the actual
    # held-out test set, not an approximation of it.
    _train_df, df = _time_split(add_targets(pd.read_parquet(_TRAINING)), _TEST_SIZE)
    # Approximate per-race key (audit C3: race_time null historically).
    race_key = list(zip(
        df["race_date"].astype(str).str[:10], df["venue"].astype(str),
        df["distance"].astype(str),
    ))
    print(f"held-out test rows: {len(df):,}  approx races: {len(set(race_key)):,}")

    rows = []
    for tag in ("v3", "v3nf"):
        feat = _load(tag)
        avail = [c for c in feat if c in df.columns]
        X = df[avail].fillna(np.nan).astype(float).values

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        fig.suptitle(f"{tag} reliability — before (raw) vs after (calibrated + within-race norm)",
                     fontsize=12)

        for ax, target in zip(axes, _TARGETS):
            model = CatBoostClassifier()
            model.load_model(os.path.join(_MODELS, f"catboost_{target}_{tag}.bin"))
            calib = _calibrator(tag, target)

            mask = df[target].notna().values
            y = df.loc[mask, target].astype(int).values
            raw = model.predict_proba(X[mask])[:, 1]
            cal = np.asarray(calib.predict(raw), dtype=float) if calib else raw
            rk = [race_key[i] for i in np.where(mask)[0]]
            norm = normalize_within_race(cal, rk) if target == "won" else cal

            variants = {"raw": raw, "calibrated": cal}
            if target == "won":
                variants["cal+norm"] = norm
            for name, p in variants.items():
                rows.append({
                    "model": tag, "target": target, "variant": name,
                    "brier": round(brier_score(y, p), 4),
                    "ece": round(_ece(y, p), 4),
                    "auc": round(roc_auc_score(y, p), 4),
                })

            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            for name, p, style in [("raw", raw, "o-"), ("calibrated", cal, "s-")]:
                xs, ys = _reliability(y, p)
                ax.plot(xs, ys, style, ms=4, label=name)
            if target == "won":
                xs, ys = _reliability(y, norm)
                ax.plot(xs, ys, "^-", ms=4, label="cal+norm")
            ax.set_title(target)
            ax.set_xlabel("mean predicted prob")
            ax.set_ylabel("observed frequency")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.legend(fontsize=8)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = os.path.join(_OUT, f"{tag}_reliability.png")
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("wrote", out)

    # per-race win-prob field sum (sanity for normalization)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print("\n", res.to_string(index=False))
    res.to_csv(os.path.join(_OUT, "metrics.csv"), index=False)

    # Field-sum check on won target (v3) before/after normalization.
    model = CatBoostClassifier()
    model.load_model(os.path.join(_MODELS, "catboost_won_v3.bin"))
    feat = _load("v3")
    avail = [c for c in feat if c in df.columns]
    raw = model.predict_proba(df[avail].fillna(np.nan).astype(float).values)[:, 1]
    cal = np.asarray(_calibrator("v3", "won").predict(raw), dtype=float)
    norm = normalize_within_race(cal, race_key)
    s_cal = pd.Series(cal).groupby(pd.factorize(pd.Index([str(k) for k in race_key]))[0]).sum()
    s_norm = pd.Series(norm).groupby(pd.factorize(pd.Index([str(k) for k in race_key]))[0]).sum()
    print(f"\nv3 won per-race field sum — calibrated: median={s_cal.median():.2f} "
          f"mean={s_cal.mean():.2f} | normalized: median={s_norm.median():.3f} "
          f"mean={s_norm.mean():.3f}")


if __name__ == "__main__":
    main()
