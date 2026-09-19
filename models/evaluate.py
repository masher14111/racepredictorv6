"""Evaluation metrics for a trained CatBoost binary classifier.

report() logs to the rotating logger and returns a metrics dict.
No plots are written — the Streamlit UI handles visualisation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from models.calibration import brier_score
from utils.logger import get_logger

logger = get_logger(__name__)

_BANDS = [
    ("favourite", lambda p: p >= 0.25),
    ("mid",       lambda p: (p >= 0.10) & (p < 0.25)),
    ("underdog",  lambda p: p < 0.10),
]

# A/E outside this range flags a mis-calibrated band (actual far from expected).
_AE_TOLERANCE = 0.20
# Number of equal-width buckets in [0, 1] for the reliability curve.
_RELIABILITY_BINS = 10


def _ae_by_band(y, y_prob, ip):
    """Actual/Expected ratio per odds band.

    Expected wins = sum of model-predicted probabilities in the band.
    Actual wins   = sum of observed outcomes (y) in the band.
    A well-calibrated model has A/E ≈ 1.0; `flagged` marks |A/E - 1| > tolerance.
    """
    valid = ~np.isnan(ip)
    out = {}
    for band_name, mask_fn in _BANDS:
        mask = mask_fn(ip) & valid
        n = int(mask.sum())
        if n == 0:
            out[band_name] = {"n": 0, "actual": 0.0, "expected": 0.0,
                              "ae": None, "flagged": False}
            continue
        actual = float(y[mask].sum())
        expected = float(y_prob[mask].sum())
        ae = float(actual / expected) if expected > 0 else None
        flagged = ae is not None and abs(ae - 1.0) > _AE_TOLERANCE
        out[band_name] = {"n": n, "actual": actual, "expected": expected,
                          "ae": ae, "flagged": flagged}
    return out


def _reliability_table(y, y_prob, n_bins=_RELIABILITY_BINS):
    """Reliability curve: predicted-probability bucket vs observed frequency.

    Returns a list of per-bucket dicts (empty buckets omitted), each with the
    bucket edges, count, mean predicted prob, and observed positive frequency.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # clip so prob==1.0 lands in the last bucket, not a phantom (n_bins)th one
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)
    table = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        table.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_pred": float(y_prob[mask].mean()),
            "observed_freq": float(y[mask].mean()),
        })
    return table


def report(model, X_test, y_test, implied_prob, feature_cols, target_name,
           calibrator=None):
    """Evaluate model on hold-out test set and log all metrics.

    Parameters
    ----------
    model        : trained CatBoostClassifier
    X_test       : np.ndarray or pd.DataFrame of shape (n, n_features)
    y_test       : 1-D array of int labels (0/1)
    implied_prob : 1-D array of float, same length as y_test
    feature_cols : list[str], same order as X_test columns
    target_name  : str, used in log prefixes
    calibrator   : optional fitted calibrator with a .predict(p)->p method
                   (e.g. sklearn IsotonicRegression). When given, the raw model
                   probabilities are mapped through it before all metrics are
                   computed, so the logged A/E and reliability reflect the
                   probabilities the predictor will actually serve. Isotonic is
                   monotonic, so AUC is unchanged.

    Returns
    -------
    dict with keys: auc, band_aucs, classification_report, confusion_matrix,
                    feature_importance
    """
    X = np.asarray(X_test, dtype=float)
    y = np.asarray(y_test, dtype=int)
    ip = pd.to_numeric(pd.Series(implied_prob), errors="coerce").to_numpy(dtype=float)

    y_prob = model.predict_proba(X)[:, 1]
    if calibrator is not None:
        y_prob = np.asarray(calibrator.predict(y_prob), dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    # --- overall AUC --------------------------------------------------------
    auc = roc_auc_score(y, y_prob)
    metrics["auc"] = float(auc)
    logger.info("[%s] AUC=%.4f  n=%d  pos=%.1f%%",
                target_name, auc, len(y), 100 * y.mean())

    # --- Brier score (calibration quality) ----------------------------------
    # Mean squared error of the served probabilities. Reference point: a
    # sharpness-free model that always predicts the base rate scores
    # base_rate*(1-base_rate), so we log both to show resolution beyond it.
    brier = brier_score(y, y_prob)
    base_rate = float(y.mean())
    metrics["brier"] = brier
    metrics["brier_baseline"] = base_rate * (1.0 - base_rate)
    logger.info("[%s] Brier=%.4f (base-rate Brier=%.4f)",
                target_name, brier, metrics["brier_baseline"])

    # --- per-odds-band AUC --------------------------------------------------
    band_aucs = {}
    for band_name, mask_fn in _BANDS:
        valid = ~np.isnan(ip)
        mask = mask_fn(ip) & valid
        if mask.sum() >= 2 and len(np.unique(y[mask])) > 1:
            b_auc = float(roc_auc_score(y[mask], y_prob[mask]))
            band_aucs[band_name] = b_auc
            logger.info("[%s] %s-band AUC=%.4f (n=%d)",
                        target_name, band_name, b_auc, int(mask.sum()))
        else:
            band_aucs[band_name] = None
            logger.info("[%s] %s-band: insufficient data (n=%d, classes=%s)",
                        target_name, band_name, int(mask.sum()),
                        list(np.unique(y[mask])) if mask.sum() else [])
    metrics["band_aucs"] = band_aucs

    # --- calibration: A/E ratio per odds band -------------------------------
    ae_by_band = _ae_by_band(y, y_prob, ip)
    for band_name, d in ae_by_band.items():
        if d["ae"] is None:
            logger.info("[%s] %s-band A/E: n/a (n=%d)",
                        target_name, band_name, d["n"])
        else:
            logger.info("[%s] %s-band A/E=%.3f (actual=%.1f exp=%.1f n=%d)%s",
                        target_name, band_name, d["ae"], d["actual"],
                        d["expected"], d["n"],
                        "  <-- MISCALIBRATED" if d["flagged"] else "")
    metrics["ae_by_band"] = ae_by_band

    # --- calibration: reliability curve -------------------------------------
    reliability = _reliability_table(y, y_prob)
    logger.info("[%s] reliability curve (pred bucket -> observed freq):",
                target_name)
    for row in reliability:
        logger.info("  [%.2f-%.2f) n=%d pred=%.3f obs=%.3f",
                    row["bin_lo"], row["bin_hi"], row["n"],
                    row["mean_pred"], row["observed_freq"])
    metrics["reliability"] = reliability

    # --- classification report ----------------------------------------------
    cr = classification_report(y, y_pred, zero_division=0)
    logger.info("[%s] classification report:\n%s", target_name, cr)
    metrics["classification_report"] = cr

    # --- confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y, y_pred)
    cm_str = f"[[{cm[0,0]} {cm[0,1]}]\n [{cm[1,0]} {cm[1,1]}]]"
    logger.info("[%s] confusion matrix:\n%s", target_name, cm_str)
    metrics["confusion_matrix"] = cm.tolist()

    # --- feature importance (top-20) ----------------------------------------
    importances = model.get_feature_importance(type="PredictionValuesChange")
    top_n = min(20, len(feature_cols))
    pairs = sorted(zip(feature_cols, importances.tolist()),
                   key=lambda x: x[1], reverse=True)[:top_n]
    logger.info("[%s] top features: %s", target_name, pairs)
    metrics["feature_importance"] = pairs

    return metrics
