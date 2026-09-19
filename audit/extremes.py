"""Probability-extreme / support / portability diagnostics (requirement 4).

Answers, for the persisted calibration stack (base calibrator + favourite-
longshot OddsBandCalibrator):

* where the maps can emit **exact 0.0 / exact 1.0** (isotonic endpoint clamps);
* the **support** each per-band isotonic was fit on (inputs outside it are
  clamped to the endpoint value — extrapolation by clamping);
* how often real inputs fall **outside support** (missing support);
* **portability**: the F-L artifact was fit against pre-off exchange prices
  (ppwap); live scoring feeds it best-board bookmaker prices. The band-occupancy
  shift between the two price distributions measures how far out of its fitted
  distribution the artifact operates live.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from models.calibration import IsotonicCalibrator, OddsBandCalibrator
from utils.logger import get_logger

logger = get_logger(__name__)


def load_pickle(path) -> object:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def describe_base_calibrator(cal) -> dict:
    """Type + zero/one reachability of a base (per-prob) calibrator."""
    out = {"type": type(cal).__name__}
    if isinstance(cal, IsotonicCalibrator) or hasattr(cal, "x"):
        out.update(n_thresholds=len(cal.x),
                   x_support=(float(cal.x[0]), float(cal.x[-1])),
                   y_range=(float(cal.y[0]), float(cal.y[-1])),
                   can_emit_zero=bool(cal.y[0] <= 0.0),
                   can_emit_one=bool(cal.y[-1] >= 1.0))
    elif hasattr(cal, "a"):
        out.update(a=float(cal.a), b=float(cal.b),
                   can_emit_zero=False, can_emit_one=False)
    return out


def describe_fl_bands(fl: OddsBandCalibrator) -> pd.DataFrame:
    rows = []
    for centre, cal in zip(fl.centers, fl.calibrators):
        row = {"odds_centre": round(float(np.exp(centre)), 2),
               "type": type(cal).__name__}
        if hasattr(cal, "x"):
            row.update(n_thresholds=len(cal.x),
                       x_min=float(cal.x[0]), x_max=float(cal.x[-1]),
                       y_min=float(cal.y[0]), y_max=float(cal.y[-1]),
                       emits_zero_below_x_min=bool(cal.y[0] <= 0.0),
                       emits_top_above_x_max=float(cal.y[-1]))
        elif hasattr(cal, "a"):
            row.update(a=float(cal.a), b=float(cal.b),
                       emits_zero_below_x_min=False)
        rows.append(row)
    return pd.DataFrame(rows)


def grid_extremes(fl: OddsBandCalibrator, *,
                  prob_grid=None, odds_grid=None) -> dict:
    """Scan (prob, odds) grid; report where output hits exact 0/1."""
    prob_grid = np.asarray(prob_grid if prob_grid is not None
                           else np.concatenate([[0.001, 0.0025], np.linspace(0.005, 0.995, 199)]))
    odds_grid = np.asarray(odds_grid if odds_grid is not None
                           else [1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0,
                                 10.0, 13.0, 17.0, 21.0, 26.0, 34.0, 51.0, 81.0])
    zero_cells, one_cells, total = [], [], 0
    for d in odds_grid:
        out = np.asarray(fl.predict(prob_grid, np.full_like(prob_grid, d)),
                         dtype=float)
        total += out.size
        z = prob_grid[out <= 0.0]
        o = prob_grid[out >= 1.0]
        if z.size:
            zero_cells.append({"odds": float(d), "n_zero": int(z.size),
                               "max_prob_mapped_to_zero": float(z.max())})
        if o.size:
            one_cells.append({"odds": float(d), "n_one": int(o.size),
                              "min_prob_mapped_to_one": float(o.min())})
    return {"n_grid": int(total),
            "zero_cells": zero_cells, "one_cells": one_cells,
            "any_zero": bool(zero_cells), "any_one": bool(one_cells)}


def support_hit_rate(fl: OddsBandCalibrator, prob: np.ndarray,
                     odds: np.ndarray) -> dict:
    """Fraction of real (prob, odds) inputs falling outside each bracketing
    band's fitted prob support (i.e. answered by endpoint clamping)."""
    prob = np.asarray(prob, dtype=float)
    odds = np.asarray(odds, dtype=float)
    m = np.isfinite(prob) & np.isfinite(odds) & (odds > 1.0)
    prob, odds = prob[m], odds[m]
    if prob.size == 0:
        return {"n": 0}
    log_d = np.log(np.clip(odds, 1.0 + 1e-9, None))
    c = np.asarray(fl.centers)
    j = np.clip(np.searchsorted(c, np.clip(log_d, c[0], c[-1]),
                                side="right") - 1, 0, max(c.size - 2, 0))
    below = np.zeros(prob.size, dtype=bool)
    above = np.zeros(prob.size, dtype=bool)
    for band in np.unique(j):
        cal = fl.calibrators[int(band)]
        if not hasattr(cal, "x"):
            continue
        mm = j == band
        below[mm] = prob[mm] < cal.x[0]
        above[mm] = prob[mm] > cal.x[-1]
    return {"n": int(prob.size),
            "below_support_frac": float(below.mean()),
            "above_support_frac": float(above.mean()),
            "clamped_frac": float((below | above).mean())}


def band_occupancy(odds: pd.Series, edges=None) -> pd.Series:
    """Fraction of rows per F-L odds band (fit-domain vs live-domain shift)."""
    e = list(edges or OddsBandCalibrator.DEFAULT_EDGES)
    o = pd.to_numeric(odds, errors="coerce")
    labels = [f"[{lo},{hi})" for lo, hi in zip(e[:-1], e[1:])]
    cut = pd.cut(o, bins=e, right=False, labels=labels)
    return cut.value_counts(normalize=True, dropna=True).reindex(labels).fillna(0.0)


def portability_table(fit_odds: pd.Series, live_odds: pd.Series) -> pd.DataFrame:
    """Side-by-side band occupancy of the fit-domain and live-domain prices."""
    fit_occ = band_occupancy(fit_odds)
    live_occ = band_occupancy(live_odds)
    return pd.DataFrame({
        "band": fit_occ.index,
        "fit_frac": fit_occ.to_numpy().round(4),
        "live_frac": live_occ.to_numpy().round(4),
        "shift": (live_occ.to_numpy() - fit_occ.to_numpy()).round(4),
    })


def cache_extremes(cache: dict) -> dict:
    """Count exact-0/1 probabilities in a predictions.json payload, per field."""
    fields = ("value_win_prob", "value_win_prob_independent", "won_prob",
              "won_prob_normalized", "catboost_win_prob", "lgbm_win_prob",
              "market_prob")
    counts = {f: {"n": 0, "zeros": 0, "ones": 0} for f in fields}
    for race in cache.get("races", []):
        runners = race.get("runners") or []
        for r in runners:
            for f in fields:
                v = r.get(f)
                if v is None:
                    continue
                counts[f]["n"] += 1
                if v == 0.0:
                    counts[f]["zeros"] += 1
                if v == 1.0:
                    counts[f]["ones"] += 1
    return counts
