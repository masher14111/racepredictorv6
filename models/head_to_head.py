"""Probabilistic head-to-head: the v4 model vs the de-vigged market line.

This is the metric that exposed the truth in the 3-week test. Every other number
v4 reports — backtest ROI, A/E, CLV — can look healthy while the model adds no
information the morning market did not already contain. The head-to-head removes
that ambiguity: it scores the model's win probabilities against the de-vigged
market over the **same races**, on the same proper scoring rules (race-level
log-loss, runner-level Brier, ECE), and collapses the comparison to a single
GO / NO-GO boolean — ``model_beats_market_logloss``. If the model cannot beat the
fair market line on log-loss, it has no edge to bet, regardless of what the P&L
simulation says.

What makes the comparison honest:

* **De-vigged market, not raw odds.** The benchmark is the margin-free market
  probability vector (:func:`models.devig.devig`), so the model is measured
  against the market's *opinion*, not against the bookmaker's overround.
* **Complete books only.** A de-vig over a partial book is not the market's
  probability vector — its booksum is wrong. Races where any runner is unpriced
  are dropped entirely (mirroring :func:`models.devig.devig`'s own rule), so both
  sides are scored over an identical, fully-priced race set.
* **Pre-off price, no look-ahead.** ``odds_col`` must be a pre-off price
  (morningwap / ppwap), the price available at decision time — NEVER the finishing
  SP. Passing a closing price silently turns the benchmark into a look-ahead
  oracle and the GO gate becomes meaningless.
* **Within-race-normalised model probs.** Exactly one runner wins, so the field's
  win probabilities must sum to 1 before they are a comparable probability vector;
  the model side is run through :func:`models.calibration.normalize_within_race`
  first, matching how the de-vigged market is constructed.

Ported from racing_ingestion's evaluation suite (``ml.evaluation.metrics``,
``ml.evaluation.calibration``, ``ml.evaluation.odds_bands``) and the
``head_to_head_logloss`` gate in ``backtest/phase4_holdout_backtest.py``, kept as
pure functions with no I/O and no look-ahead.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from models.calibration import normalize_within_race
from models.devig import devig

logger = logging.getLogger(__name__)

_LOGLOSS_EPS = 1e-7  # clip floor matching ml.evaluation.metrics.log_loss_grouped


# ── Odds bands (PRE-RACE decimal odds; lo inclusive, hi exclusive) ────────────
# 11.0 decimal = 10/1 fractional — everything from "10/1+" up is outsider land,
# where a value model must prove itself. Mirrors ml.evaluation.odds_bands.
ODDS_BANDS: list[tuple[str, float, float]] = [
    ("<2.0 (odds-on)", 1.0, 2.0),
    ("2.0-3.0", 2.0, 3.0),
    ("3.0-5.0", 3.0, 5.0),
    ("5.0-8.0", 5.0, 8.0),
    ("8.0-11.0", 8.0, 11.0),
    ("11.0-21.0", 11.0, 21.0),
    ("21.0-51.0", 21.0, 51.0),
    ("51.0+", 51.0, float("inf")),
]

# Aggregate rows appended after the individual bands.
_AGGREGATE_BANDS: list[tuple[str, float, float]] = [
    ("ALL", 1.0, float("inf")),
    ("10/1+ (>=11.0)", 11.0, float("inf")),
]


# ── Scoring primitives ────────────────────────────────────────────────────────


def _race_level_log_loss(
    y_true: np.ndarray, y_pred: np.ndarray, race_ids: np.ndarray, eps: float = _LOGLOSS_EPS
) -> float:
    """Mean per-race negative log-likelihood: −log(p_winner), averaged over races.

    A race with k runners contributes ONE outcome, not k, so log-loss is computed
    per race (on the winner's normalised probability) and then averaged — never as
    a runner-level Bernoulli sum, which would overcount the field size. This is the
    head-to-head's primary criterion and matches the model's training objective.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1.0)
    race_ids = np.asarray(race_ids)

    per_race: list[float] = []
    for rid in pd.unique(race_ids):
        mask = race_ids == rid
        winner_probs = y_pred[mask][y_true[mask].astype(bool)]
        if winner_probs.size > 0:
            per_race.append(-float(np.log(winner_probs).sum()))

    return float(np.mean(per_race)) if per_race else float("nan")


def _runner_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Runner-level Brier score: mean squared error of probability vs 0/1 outcome."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean((y_pred - y_true) ** 2))


def _expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
) -> float:
    """ECE over equal-frequency bins: Σ_b (n_b/N)·|actual_rate_b − pred_mean_b|.

    Equal-frequency (rather than equal-width) binning keeps the buckets populated
    at the extremes, where racing probabilities are sparse. Mirrors
    ml.evaluation.calibration.expected_calibration_error. Returns NaN when there
    are too few rows, or every prediction is identical, to define bins.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")

    bin_edges = np.unique(np.percentile(y_pred, np.linspace(0, 100, n_bins + 1)))
    if len(bin_edges) < 2:
        # All predictions identical → calibration is a single point, gap is well
        # defined as |actual_rate − that prediction|.
        return float(abs(y_true.mean() - y_pred.mean()))

    bin_idx = np.digitize(y_pred, bin_edges[1:-1], right=True)
    n_total = y_true.size
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / n_total) * abs(y_true[mask].mean() - y_pred[mask].mean())
    return float(ece)


def _score(y_true: np.ndarray, y_pred: np.ndarray, race_ids: np.ndarray, n_bins: int) -> dict:
    """Full proper-scoring summary for one probability vector over the race set."""
    return {
        "log_loss": _race_level_log_loss(y_true, y_pred, race_ids),
        "brier_runner_level": _runner_brier(y_true, y_pred),
        "ece": _expected_calibration_error(y_true, y_pred, n_bins),
    }


def odds_band_table(
    y_true: np.ndarray,
    model_probs: np.ndarray,
    market_probs: np.ndarray,
    decimal_odds: np.ndarray,
) -> pd.DataFrame:
    """Per-odds-band calibration: actual win rate vs model vs de-vigged market.

    A single aggregate log-loss can hide a model that is well calibrated on
    favourites and badly mis-set on longshots. This slices the comparison by
    PRE-RACE decimal odds band so the favourite-end (efficient) and the
    outsider-end (sparse, mispriced) are visible separately. Mirrors
    ml.evaluation.odds_bands.model_vs_market_by_odds_band.

    All four inputs are aligned per-runner arrays. One row per band (plus ALL and
    10/1+ aggregates) with the band's runner/winner counts, mean model & market
    probabilities, calibration gaps (actual − mean), A/E ratios, and Briers.
    """
    y = np.asarray(y_true, dtype=float)
    mp = np.asarray(model_probs, dtype=float)
    kp = np.asarray(market_probs, dtype=float)
    odds = np.asarray(decimal_odds, dtype=float)

    if not (len(y) == len(mp) == len(kp) == len(odds)):
        raise ValueError(
            "odds_band_table: input lengths differ "
            f"({len(y)}, {len(mp)}, {len(kp)}, {len(odds)})"
        )

    rows = []
    for label, lo, hi in ODDS_BANDS + _AGGREGATE_BANDS:
        mask = (odds >= lo) & (odds < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "band": label, "n_runners": 0, "n_wins": 0,
                "actual_rate": np.nan, "model_mean": np.nan, "market_mean": np.nan,
                "model_gap": np.nan, "market_gap": np.nan,
                "ae_model": np.nan, "ae_market": np.nan,
                "model_brier": np.nan, "market_brier": np.nan,
            })
            continue
        n_wins = int(y[mask].sum())
        actual = n_wins / n
        mm = float(mp[mask].mean())
        km = float(kp[mask].mean())
        rows.append({
            "band": label,
            "n_runners": n,
            "n_wins": n_wins,
            "actual_rate": actual,
            "model_mean": mm,
            "market_mean": km,
            "model_gap": actual - mm,
            "market_gap": actual - km,
            "ae_model": actual / mm if mm > 0 else np.nan,
            "ae_market": actual / km if km > 0 else np.nan,
            "model_brier": float(np.mean((mp[mask] - y[mask]) ** 2)),
            "market_brier": float(np.mean((kp[mask] - y[mask]) ** 2)),
        })

    return pd.DataFrame(rows)


# ── Public head-to-head ───────────────────────────────────────────────────────


def head_to_head(
    df: pd.DataFrame,
    prob_col: str,
    odds_col: str,
    race_id_col: str,
    label_col: str = "won",
    devig_method: str = "proportional",
    n_bins: int = 10,
) -> dict:
    """Score the v4 model against the de-vigged market over complete-odds races.

    The single question this answers: does the model's win probability beat the
    fair (margin-free) market line on log-loss? Everything else in the returned
    dict is supporting detail for that GO / NO-GO verdict.

    Args:
        df:          One row per runner, with at least ``label_col`` (win 0/1),
                     ``prob_col`` (model win probability), ``odds_col``
                     (PRE-OFF decimal odds — morningwap/ppwap, NEVER odds_finish),
                     and ``race_id_col``.
        prob_col:    Column of model win probabilities. Re-normalised within race
                     (sum-to-1) before scoring via
                     :func:`models.calibration.normalize_within_race`.
        odds_col:    Column of PRE-OFF decimal odds. Using a closing/finishing
                     price here is look-ahead and invalidates the gate.
        race_id_col: Race identifier column.
        label_col:   Binary win-label column (default ``"won"``).
        devig_method: ``"proportional"`` (default), ``"power"``, or ``"shin"`` —
                     passed to :func:`models.devig.devig`.
        n_bins:      Equal-frequency bins for ECE.

    Returns:
        dict with:
            n_races_total, n_races_kept, n_runners, devig_method
            model:  {log_loss, brier_runner_level, ece}
            market: {log_loss, brier_runner_level, ece}
            gaps:   {log_loss, brier_runner_level, ece}  (market − model; a
                    POSITIVE value means the model scores the lower / better loss)
            model_beats_market_logloss: bool  (model log-loss < market log-loss)
            odds_band_table: per-band actual-vs-model-vs-market DataFrame

    The model is restricted to COMPLETE-ODDS races (every runner priced); a de-vig
    over a partial book is not the market's probability vector, so such races are
    dropped from BOTH sides.
    """
    for col in (prob_col, odds_col, race_id_col, label_col):
        if col not in df.columns:
            raise KeyError(f"head_to_head: column {col!r} not in df")

    n_total = int(df[race_id_col].nunique())

    # Complete-odds races only: every runner in the race must be priced. A single
    # unpriced runner makes the booksum wrong, so the whole race is dropped — the
    # same rule models.devig.devig applies, kept consistent here so both sides are
    # scored over an identical, fully-priced race set.
    complete = (
        df[odds_col].notna().groupby(df[race_id_col]).transform("all").to_numpy()
    )
    h = df.loc[complete]
    n_kept = int(h[race_id_col].nunique())

    if n_kept == 0:
        logger.warning("head_to_head: no complete-odds races to score")
        empty = {"log_loss": float("nan"), "brier_runner_level": float("nan"), "ece": float("nan")}
        return {
            "n_races_total": n_total,
            "n_races_kept": 0,
            "n_runners": 0,
            "devig_method": devig_method,
            "model": dict(empty),
            "market": dict(empty),
            "gaps": dict(empty),
            "model_beats_market_logloss": False,
            "odds_band_table": odds_band_table(
                np.array([]), np.array([]), np.array([]), np.array([])
            ),
        }

    y = h[label_col].astype(float).to_numpy()
    rid = h[race_id_col].to_numpy()

    # Model side: re-normalise within race so the field sums to 1 (a comparable
    # probability vector), exactly as the de-vigged market is constructed.
    model_p = normalize_within_race(h[prob_col].astype(float).to_numpy(), rid)
    # Market side: the de-vigged pre-off line. Complete books → no NaN here.
    market_p = devig(h[odds_col], h[race_id_col], method=devig_method)

    model_m = _score(y, model_p, rid, n_bins)
    market_m = _score(y, market_p, rid, n_bins)

    # gap = market − model: positive means the model has the LOWER (better) value
    # on every metric (log-loss, Brier, ECE are all lower-is-better).
    gaps = {k: market_m[k] - model_m[k] for k in ("log_loss", "brier_runner_level", "ece")}

    return {
        "n_races_total": n_total,
        "n_races_kept": n_kept,
        "n_runners": int(len(h)),
        "devig_method": devig_method,
        "model": model_m,
        "market": market_m,
        "gaps": gaps,
        "model_beats_market_logloss": bool(model_m["log_loss"] < market_m["log_loss"]),
        "odds_band_table": odds_band_table(y, model_p, market_p, h[odds_col].to_numpy()),
    }
