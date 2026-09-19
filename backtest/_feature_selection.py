"""Feature-selection analysis for the price-free win/place/show models.

Loads the regenerated full (model-10) matrix, computes global feature importance
two ways (CatBoost PredictionValuesChange + exact tree SHAP), flags dead /
low-signal / redundant features, then retrains on a pruned whitelist and compares
AUC / log-loss / Brier / ECE on the held-out tail for all three targets.

Writes docs/calibration/feature_selection.json for the report + memory note.
"""
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import log_loss, roc_auc_score

from models.calibration import brier_score, fit_calibrator
from models.features import PRICE_FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.targets import add_targets

MATRIX = "data/features/training_full.parquet"
META = "models/catboost_v3nf_meta.json"
OUT = "docs/calibration/feature_selection.json"
TARGETS = ["won", "placed_2", "showed"]
SHOW_POSITIONS = 3
MAX_W = 20.0
TEST_FRAC = 0.18
CALIB_FRAC = 0.12   # of the whole, taken from the tail of train


def ece(y, p, n_bins=10):
    """Expected Calibration Error (equal-width bins, weighted |conf-acc|)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def sample_weights(df, max_w=MAX_W):
    ip = pd.to_numeric(df["implied_prob"], errors="coerce").fillna(1.0)
    ip = ip.clip(lower=1.0 / max_w)
    return (1.0 / ip).clip(upper=max_w).to_numpy(float)


def best_params():
    with open(META, encoding="utf-8") as fh:
        meta = json.load(fh)
    return {t: d.get("best_params", {}) or {} for t, d in meta["targets"].items()}


def fit_one(X_fit, y_fit, w_fit, X_es, y_es, params):
    fp = {
        "iterations": 800, "loss_function": "Logloss", "eval_metric": "AUC",
        "early_stopping_rounds": 50, "verbose": False, "allow_writing_files": False,
        "random_seed": 42, "thread_count": -1, **params,
    }
    if "auto_class_weights" not in fp and "scale_pos_weight" not in fp:
        fp["auto_class_weights"] = "Balanced"
    m = CatBoostClassifier(**fp)
    m.fit(X_fit, y_fit, sample_weight=w_fit, eval_set=(X_es, y_es), verbose=False)
    return m


def evaluate(cols, df_tr, df_cal, df_te, params_by_t, want_importance=False):
    """Train+calibrate each target on `cols`; return metrics dict (+ importance)."""
    res = {"metrics": {}, "importance": {}}
    for t in TARGETS:
        trm = df_tr[t].notna().values
        tem = df_te[t].notna().values
        calm = df_cal[t].notna().values

        X_tr = df_tr.loc[trm, cols].astype(float).values
        y_tr = df_tr.loc[trm, t].astype(int).values
        w_tr = df_tr.loc[trm, "_w"].to_numpy(float)
        X_cal = df_cal.loc[calm, cols].astype(float).values
        y_cal = df_cal.loc[calm, t].astype(int).values
        X_te = df_te.loc[tem, cols].astype(float).values
        y_te = df_te.loc[tem, t].astype(int).values

        n_val = max(1, int(len(X_tr) * 0.10))
        m = fit_one(X_tr[:-n_val], y_tr[:-n_val], w_tr[:-n_val],
                    X_tr[-n_val:], y_tr[-n_val:], params_by_t.get(t, {}))

        raw_cal = m.predict_proba(X_cal)[:, 1]
        calib, _ = fit_calibrator("isotonic", raw_cal, y_cal)

        raw_te = m.predict_proba(X_te)[:, 1]
        p_te = np.asarray(calib.predict(raw_te), float)

        res["metrics"][t] = {
            "auc": float(roc_auc_score(y_te, p_te)),
            "logloss": float(log_loss(y_te, np.clip(p_te, 1e-6, 1 - 1e-6))),
            "brier": brier_score(y_te, p_te),
            "ece": ece(y_te, p_te),
            "n_test": int(tem.sum()),
        }

        if want_importance and t == "won":
            pvc = m.get_feature_importance(type="PredictionValuesChange")
            # exact tree SHAP on a test subsample for speed
            n = min(20000, len(X_te))
            pool = Pool(X_te[:n], y_te[:n])
            sv = m.get_feature_importance(type="ShapValues", data=pool)
            mean_abs = np.abs(sv[:, :-1]).mean(axis=0)
            res["importance"] = {
                "pvc": {c: float(v) for c, v in zip(cols, pvc)},
                "shap_mean_abs": {c: float(v) for c, v in zip(cols, mean_abs)},
            }
    return res


def main():
    t0 = time.time()
    df = pd.read_parquet(MATRIX)
    df = add_targets(df, SHOW_POSITIONS)
    df = df.sort_values("race_date").reset_index(drop=True)
    df["_w"] = sample_weights(df)

    n = len(df)
    i_te = int(n * (1 - TEST_FRAC))
    i_cal = int(n * (1 - TEST_FRAC - CALIB_FRAC))
    df_tr, df_cal, df_te = df.iloc[:i_cal].copy(), df.iloc[i_cal:i_te].copy(), df.iloc[i_te:].copy()
    print(f"split train={len(df_tr)} calib={len(df_cal)} test={len(df_te)}  "
          f"dates {df_te['race_date'].min()} -> {df_te['race_date'].max()}")

    full_cols = list(PRICE_FREE_FEATURE_COLS)
    print(f"price-free full feature set: {len(full_cols)} cols")

    # null fractions over the whole matrix
    nullf = {c: float(df[c].isna().mean()) for c in full_cols}

    pby = best_params()
    print("training FULL feature set...")
    full = evaluate(full_cols, df_tr, df_cal, df_te, pby, want_importance=True)

    imp = full["importance"]
    pvc = imp["pvc"]; shp = imp["shap_mean_abs"]

    # ── classify features ────────────────────────────────────────────────
    dead = [c for c in full_cols if nullf[c] >= 0.999]
    # low-signal: tiny on BOTH metrics and not dead
    pvc_max = max(pvc.values()) or 1.0
    shp_max = max(shp.values()) or 1.0
    low = [c for c in full_cols if c not in dead
           and pvc[c] / pvc_max < 0.01 and shp[c] / shp_max < 0.01]

    # redundancy: |corr| >= 0.95 pairs (numeric, pairwise complete)
    corr = df[full_cols].corr().abs()
    redundant_pairs = []
    for i, a in enumerate(full_cols):
        for b in full_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= 0.95:
                # drop the weaker (lower combined importance)
                ia = pvc[a] / pvc_max + shp[a] / shp_max
                ib = pvc[b] / pvc_max + shp[b] / shp_max
                redundant_pairs.append({"a": a, "b": b, "corr": float(r),
                                        "drop": b if ib < ia else a})

    drop = set(dead) | set(low) | {p["drop"] for p in redundant_pairs}
    pruned_cols = [c for c in full_cols if c not in drop]
    print(f"dead={dead}")
    print(f"low-signal={low}")
    print(f"redundant_pairs={redundant_pairs}")
    print(f"PRUNED set: {len(pruned_cols)} cols (dropped {len(drop)})")

    print("training PRUNED feature set...")
    pruned = evaluate(pruned_cols, df_tr, df_cal, df_te, pby)

    out = {
        "n_rows": n,
        "split": {"train": len(df_tr), "calib": len(df_cal), "test": len(df_te),
                  "test_start": str(df_te["race_date"].min()),
                  "test_end": str(df_te["race_date"].max())},
        "full_cols": full_cols,
        "pruned_cols": pruned_cols,
        "dropped": sorted(drop),
        "dead": dead, "low_signal": low, "redundant_pairs": redundant_pairs,
        "null_fraction": nullf,
        "importance": imp,
        "metrics_full": full["metrics"],
        "metrics_pruned": pruned["metrics"],
    }
    import os
    os.makedirs("docs/calibration", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {OUT}")

    # ── console summary ──────────────────────────────────────────────────
    print("\n=== GLOBAL IMPORTANCE (won, price-free) — top 15 by SHAP ===")
    for c, v in sorted(shp.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        print(f"  {c:<24} shap={v:.4f}  pvc={pvc[c]:6.2f}  null={nullf[c]:.2%}")

    print("\n=== FULL vs PRUNED (held-out tail) ===")
    hdr = f"{'target':<10} {'metric':<8} {'full':>9} {'pruned':>9} {'delta':>9}"
    print(hdr)
    for t in TARGETS:
        f, p = full["metrics"][t], pruned["metrics"][t]
        for k in ("auc", "logloss", "brier", "ece"):
            print(f"{t:<10} {k:<8} {f[k]:9.4f} {p[k]:9.4f} {p[k]-f[k]:+9.4f}")
    print(f"\nfeatures: {len(full_cols)} -> {len(pruned_cols)}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
