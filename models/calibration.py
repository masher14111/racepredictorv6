"""Probability calibration + within-race normalization.

The training pipeline boosts recall with ``auto_class_weights`` + odds-inverse
sample weights, which inflates raw CatBoost probabilities (A/E far below 1). A
calibrator fit on a time-ordered held-out slice maps them back to true
frequencies. Two methods are supported, sharing a ``.predict(p) -> p``
interface so :func:`models.predictor._load_calibrator` stays method-agnostic
and the pickled object loads regardless of method:

* **isotonic** — non-parametric monotone fit (fit via ``sklearn.IsotonicRegression``,
  then stored as portable threshold lists; see :class:`IsotonicCalibrator`).
  Flexible; needs enough rows to avoid a step-function overfit.
* **sigmoid** — Platt scaling: a 1-parameter logistic fit on the model's logit.
  Robust on small slices; assumes a sigmoidal distortion.

Both persisted calibrators store only plain Python floats, so a pickle written
under one numpy major version loads under another. (``sklearn``'s raw
``IsotonicRegression`` pickles numpy arrays whose internal module path changed
in numpy 2.0, ``numpy.core`` → ``numpy._core``, breaking cross-version loads —
the reason this wrapper exists.)

``fit_calibrator(method="auto", ...)`` fits both on the earlier part of the
calibration slice and keeps whichever scores a lower Brier on a held-out tail
of that same slice (no leakage into the model OR the test set), then refits the
winner on the full slice.

Isotonic and sigmoid are both monotone, so neither changes per-race ranking;
they only correct the probability value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# Below this many rows a calibration-validation fold is too small to choose a
# method reliably, so "auto" falls back to isotonic without a comparison.
_MIN_SELECT_ROWS = 200
_EPS = 1e-12


class SigmoidCalibrator:
    """Platt scaling: ``p_calibrated = sigmoid(a * logit(p_raw) + b)``.

    Fitting a logistic regression on the model's logit (rather than the raw
    probability) keeps the map well-behaved at the extremes. ``a``/``b`` are the
    only state, so the object pickles trivially and is independent of the
    scikit-learn version used to fit it.
    """

    def __init__(self, a: float, b: float) -> None:
        self.a = float(a)
        self.b = float(b)

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
        return np.log(p / (1.0 - p))

    @classmethod
    def fit(cls, p_raw, y) -> "SigmoidCalibrator":
        from sklearn.linear_model import LogisticRegression

        f = cls._logit(p_raw).reshape(-1, 1)
        y = np.asarray(y, dtype=int)
        # Near-zero regularization → genuine Platt scaling, not a shrunk fit.
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(f, y)
        return cls(a=float(lr.coef_[0, 0]), b=float(lr.intercept_[0]))

    def predict(self, p) -> np.ndarray:
        z = self.a * self._logit(p) + self.b
        z = np.clip(z, -700.0, 700.0)  # guard np.exp overflow at the extremes
        return 1.0 / (1.0 + np.exp(-z))


class IsotonicCalibrator:
    """Portable isotonic calibrator: piecewise-linear interpolation over the
    fitted step thresholds, stored as plain Python floats.

    ``sklearn.IsotonicRegression`` pickles its fitted numpy arrays, whose
    internal module path changed in numpy 2.0 (``numpy.core`` → ``numpy._core``),
    so a pickle written under one numpy major version fails to unpickle under the
    other. We fit with ``IsotonicRegression`` but persist only the threshold
    lists and predict with :func:`numpy.interp` — which interpolates between the
    same thresholds and clips out-of-range inputs to the endpoint values,
    exactly matching ``IsotonicRegression(out_of_bounds="clip")``. The object is
    therefore independent of the numpy/scikit-learn version that fitted it, like
    :class:`SigmoidCalibrator`.
    """

    def __init__(self, x, y) -> None:
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]

    @classmethod
    def from_isotonic(cls, iso: IsotonicRegression) -> "IsotonicCalibrator":
        return cls(iso.X_thresholds_, iso.y_thresholds_)

    @classmethod
    def fit(cls, p_raw, y) -> "IsotonicCalibrator":
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.asarray(p_raw, dtype=float), np.asarray(y, dtype=int)
        )
        return cls.from_isotonic(iso)

    def predict(self, p) -> np.ndarray:
        # np.interp clamps inputs outside [x[0], x[-1]] to the endpoint y values,
        # matching IsotonicRegression(out_of_bounds="clip").
        return np.interp(np.asarray(p, dtype=float), self.x, self.y)


class OddsBandCalibrator:
    """Favourite-longshot recalibrator: ``.predict(prob, odds) -> prob``.

    The price-free win model is calibrated *marginally* (A/E ≈ 1 within its own
    probability buckets) yet badly biased *conditional on the market price*: on
    the saved 77k-runner OOS frame its A/E runs 2.80 at odds-on and 1.88 at
    [2,4] (favourites win more than predicted) down to 0.74 at [8,16], 0.44 at
    [16,34] and 0.17 beyond 34 (longshots hugely over-predicted). A calibrator
    that sees only ``prob`` cannot fix this — by prob bucket the model is already
    honest, so a prob-only map is ~identity. The correction must use the price.

    We therefore fit an *independent* probability calibrator within each decimal-
    odds band and, at predict time, **blend the two band calibrators that bracket
    the runner's price in log-odds space**, so the correction is continuous
    across band edges (no jump between a 3.99 and a 4.01 shot). Each per-band map
    is monotone in ``prob`` (isotonic/sigmoid) and the blend weights depend only
    on ``odds``; a convex combination of monotone maps is monotone, so the output
    is monotone in ``prob`` at any fixed price — within-race ranking among
    similarly-priced runners is preserved (it only removes the band-level bias,
    keeping the model's within-price discrimination, which is the value signal).

    Like :class:`IsotonicCalibrator`/:class:`SigmoidCalibrator` the stored state
    is plain Python floats plus those (also plain-float) per-band calibrators, so
    the pickle is independent of the numpy/scikit-learn version that fitted it.
    """

    # Default decimal-odds edges (favourite … longshot). A dedicated odds-on band
    # [1,2) resolves the strongest correction; finer mid/longshot bands flatten
    # the tail. Tuned on the OOS frame (see docs/calibration/).
    DEFAULT_EDGES = [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 60.0, float("inf")]
    _MIN_ODDS = 1.0 + 1e-9

    def __init__(self, centers, calibrators) -> None:
        if len(centers) != len(calibrators) or not calibrators:
            raise ValueError("centers and calibrators must be non-empty and aligned")
        # centers are band log-odds centres, ascending; calibrators run parallel.
        order = np.argsort(np.asarray(centers, dtype=float))
        self.centers = [float(centers[i]) for i in order]
        self.calibrators = [calibrators[i] for i in order]

    @classmethod
    def fit(cls, prob, odds, y, *, edges=None, method: str = "isotonic",
            min_rows: int = 400) -> "OddsBandCalibrator":
        """Fit one probability calibrator per odds band.

        ``edges`` is the list of decimal-odds cut points (default
        :attr:`DEFAULT_EDGES`); ``method`` is passed to :func:`fit_calibrator`
        (``isotonic`` / ``sigmoid`` / ``auto``). A band with fewer than
        ``min_rows`` rows or only one outcome class is skipped (its price range is
        then covered by blending its populated neighbours). At least one band must
        survive.
        """
        prob = np.asarray(prob, dtype=float)
        odds = np.asarray(odds, dtype=float)
        y = np.asarray(y, dtype=int)
        log_d = np.log(np.clip(odds, cls._MIN_ODDS, None))
        e = list(edges) if edges is not None else cls.DEFAULT_EDGES

        centers, calibrators = [], []
        for lo, hi in zip(e[:-1], e[1:]):
            m = (odds >= lo) & (odds < hi) & np.isfinite(prob) & np.isfinite(log_d)
            if int(m.sum()) < min_rows or len(np.unique(y[m])) < 2:
                continue
            cal, _ = fit_calibrator(method, prob[m], y[m])
            centers.append(float(log_d[m].mean()))
            calibrators.append(cal)
        if not calibrators:
            raise ValueError("no odds band had enough rows to fit a calibrator")
        return cls(centers, calibrators)

    def predict(self, prob, odds) -> np.ndarray:
        """Recalibrated win probability for ``(prob, odds)`` (vectorised).

        Below the lowest / above the highest band centre the nearest band's map
        is used unblended (clamp); in between, the two bracketing bands' outputs
        are linearly interpolated by the runner's log-odds position. Returns an
        ndarray (a 0-d array for scalar input, matching the other calibrators).
        """
        p = np.atleast_1d(np.asarray(prob, dtype=float))
        d = np.atleast_1d(np.asarray(odds, dtype=float))
        log_d = np.log(np.clip(d, self._MIN_ODDS, None))
        c = np.asarray(self.centers, dtype=float)

        # Per-band predictions for every row, then a log-odds blend across centres.
        band_preds = np.column_stack([cal.predict(p) for cal in self.calibrators])
        if c.size == 1:
            out = band_preds[:, 0]
        else:
            x = np.clip(log_d, c[0], c[-1])
            j = np.clip(np.searchsorted(c, x, side="right") - 1, 0, c.size - 2)
            span = c[j + 1] - c[j]
            w = np.where(span > 0, (x - c[j]) / np.where(span > 0, span, 1.0), 0.0)
            idx = np.arange(len(p))
            out = (1.0 - w) * band_preds[idx, j] + w * band_preds[idx, j + 1]
        out = np.clip(out, 0.0, 1.0)
        return out if np.ndim(prob) else out.reshape(())


def brier_score(y, p) -> float:
    """Mean squared error between predicted probability and 0/1 outcome.

    Lower is better; the value a perfectly-calibrated, sharpness-free model
    reaches equals ``base_rate * (1 - base_rate)``.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.mean((p - y) ** 2))


def _fit_one(method: str, p_raw, y):
    if method == "isotonic":
        return IsotonicCalibrator.fit(p_raw, y)
    if method == "sigmoid":
        return SigmoidCalibrator.fit(p_raw, y)
    raise ValueError(f"unknown calibration method: {method!r}")


def fit_calibrator(method: str, p_raw, y, *, val_frac: float = 0.30):
    """Fit a calibrator on ``(p_raw, y)``; return ``(calibrator, method_used)``.

    ``method`` is one of ``isotonic``, ``sigmoid``, ``auto``. For ``auto`` the
    slice is split chronologically (it arrives in date order): both methods are
    fit on the earlier ``1 - val_frac`` and compared by Brier on the held-out
    tail, then the winner is refit on the full slice so it uses every row.
    """
    method = str(method).strip().lower()
    p_raw = np.asarray(p_raw, dtype=float)
    y = np.asarray(y, dtype=int)

    if method in ("isotonic", "sigmoid"):
        return _fit_one(method, p_raw, y), method
    if method != "auto":
        raise ValueError(f"unknown calibration method: {method!r}")

    n = len(p_raw)
    n_val = int(n * val_frac)
    if n_val < _MIN_SELECT_ROWS or (n - n_val) < _MIN_SELECT_ROWS:
        # Too small to choose reliably — isotonic is the project default.
        return _fit_one("isotonic", p_raw, y), "isotonic"

    p_fit, y_fit = p_raw[:-n_val], y[:-n_val]
    p_val, y_val = p_raw[-n_val:], y[-n_val:]
    scores = {}
    for m in ("isotonic", "sigmoid"):
        # A degenerate single-class fold can't train a calibrator; skip it.
        if len(np.unique(y_fit)) < 2:
            continue
        cal = _fit_one(m, p_fit, y_fit)
        scores[m] = brier_score(y_val, cal.predict(p_val))
    if not scores:
        return _fit_one("isotonic", p_raw, y), "isotonic"
    best = min(scores, key=scores.get)
    return _fit_one(best, p_raw, y), best


def normalize_within_race(probs, group_ids) -> np.ndarray:
    """Rescale ``probs`` so each race group sums to 1.0.

    Exactly one runner wins a race, so the field's win probabilities must sum to
    1. Per-runner calibration enforces no such constraint, leaving the field sum
    drifting above/below 1; this divides each runner's probability by its race
    total. Groups whose total is 0 or NaN are left unchanged (nothing to anchor
    the rescale to).
    """
    p = pd.to_numeric(pd.Series(np.asarray(probs, dtype=float)), errors="coerce")
    # Flatten each group id to a single hashable code. A list of (venue, time)
    # tuples would otherwise coerce to a 2-D array and break groupby.
    keys = [
        "\x1f".join(map(str, k)) if isinstance(k, (tuple, list)) else str(k)
        for k in group_ids
    ]
    codes = pd.factorize(pd.Index(keys))[0]
    totals = p.groupby(codes).transform("sum")
    norm = p / totals.where(totals > 0)
    return norm.fillna(p).to_numpy()
