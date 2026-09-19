"""Normalized power blend of an INDEPENDENT probability and a REFERENCE market
probability, plus the race-level scoring helper the blend is selected on.

Three price-shaped quantities exist in this codebase and must never be conflated
(DESIGN.md / audit req 5-6; ``models.predictor._reference_decimals`` is the
production expression of the same rule):

* **independent probability** — the price-free model's estimate of horse
  ability. No price of any kind enters it.
* **reference market probability** — the de-vigged fair line
  (:func:`models.devig.devig`) built from ONE complete reference book, i.e. the
  market's opinion.
* **executable quote** — the bookmaker price we would actually back at. It
  drives EV, stake and settlement, and is NOT an input to this module at all.

:func:`power_blend` therefore takes exactly two probability vectors and the race
grouping. There is no parameter through which an executable quote could reach
it, which is what makes "changing only an executable quote cannot move the
blended probability" a structural property rather than a test result.

The blend itself is::

    p_i  ∝  p_model_i ** alpha  *  p_reference_i ** beta        (within race)

with the result renormalised so each complete race sums to 1.0. ``alpha=1,
beta=0`` reproduces the within-race-normalised model line; ``alpha=0, beta=1``
reproduces the reference market line; both are kept in the selection grid as
explicit boundary controls.

Partial books are never blended. A race in which ANY runner's reference
probability (or model probability) is missing/non-finite is emitted as NaN for
every one of its runners — the same all-or-nothing rule
:func:`models.devig.devig` applies, because a blend over an incomplete field is
not a probability vector. Non-runners carry no reference probability and no
outcome, so they are excluded upstream (the panel is built from priced runners);
whatever field remains is renormalised over itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

_EPS = 1e-12
_LOGLOSS_EPS = 1e-7  # matches models.head_to_head._race_level_log_loss

# Predetermined selection grid (frozen in the stage-12 manifest before any fit).
# alpha weights the independent model, beta the reference market line. beta may
# exceed 1 because a power blend's exponents are not a convex combination — the
# normalisation absorbs the scale, so only the RATIO and the overall sharpness
# matter.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
BETA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)


# ── race grouping ────────────────────────────────────────────────────────────


def _race_codes(race_ids) -> np.ndarray:
    """Integer code per race, flattening tuple/list keys like
    :func:`models.calibration.normalize_within_race` does."""
    keys = [
        "\x1f".join(map(str, k)) if isinstance(k, (tuple, list)) else str(k)
        for k in np.asarray(race_ids, dtype=object).ravel()
    ]
    return pd.factorize(pd.Index(keys))[0]


# ── scoring ──────────────────────────────────────────────────────────────────


def race_level_log_loss(y_true, y_pred, race_ids, eps: float = _LOGLOSS_EPS) -> float:
    """Mean per-race ``-log(p_winner)``.

    Same definition as :func:`models.head_to_head._race_level_log_loss` (a race
    contributes ONE outcome, never one per runner) — ``tests/models/test_blend.py``
    pins the two against each other so the blend can never be selected on a
    different scoring rule than the scorecard reports.

    Races with no winning row, or whose winner's probability is NaN, contribute
    nothing (they are not scoreable), matching the head-to-head's behaviour of
    only collecting races where a winner probability exists.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    codes = _race_codes(race_ids)
    win = y.astype(bool) & np.isfinite(p)
    if not win.any():
        return float("nan")
    # Sum -log p over the winner rows of each race, then average over races.
    pc = np.clip(p[win], eps, 1.0)
    per_race = pd.Series(-np.log(pc)).groupby(codes[win]).sum()
    return float(per_race.mean())


# ── the blend ────────────────────────────────────────────────────────────────


def power_blend(p_model, p_reference, race_ids, alpha: float, beta: float) -> np.ndarray:
    """``p ∝ p_model**alpha * p_reference**beta``, renormalised within race.

    Args:
        p_model:     independent (price-free) win probability per runner.
        p_reference: de-vigged REFERENCE market probability per runner. Never an
                     executable quote, and never a raw (vigged) implied prob.
        race_ids:    race key per runner.
        alpha/beta:  the two exponents.

    Returns:
        ndarray aligned to the inputs. Every runner of a race that has any
        missing/non-finite model or reference probability is NaN.
    """
    pm = np.asarray(p_model, dtype=float).ravel()
    pr = np.asarray(p_reference, dtype=float).ravel()
    if not (len(pm) == len(pr) == len(np.asarray(race_ids).ravel())):
        raise ValueError(
            f"power_blend: length mismatch ({len(pm)}, {len(pr)}, "
            f"{len(np.asarray(race_ids).ravel())})"
        )
    codes = _race_codes(race_ids)
    out = np.full(pm.shape, np.nan, dtype=float)
    if pm.size == 0:
        return out

    usable = np.isfinite(pm) & np.isfinite(pr) & (pm > 0) & (pr > 0)
    # All-or-nothing per race: an incomplete reference book is not a book.
    complete = pd.Series(usable).groupby(codes).transform("all").to_numpy()
    if not complete.any():
        return out

    log_p = alpha * np.log(np.clip(pm[complete], _EPS, None)) + beta * np.log(
        np.clip(pr[complete], _EPS, None)
    )
    s = pd.Series(log_p).groupby(codes[complete])
    # Subtract the per-race max before exponentiating (softmax stability).
    z = np.exp(log_p - s.transform("max").to_numpy())
    totals = pd.Series(z).groupby(codes[complete]).transform("sum").to_numpy()
    out[complete] = np.where(totals > 0, z / totals, np.nan)
    return out


class PowerBlend:
    """Fitted ``(alpha, beta)`` pair with a JSON round-trip.

    Deliberately carries no price column name and no calibrator: the blend maps
    two probability vectors to one, and any price-conditional correction is a
    SEPARATE, explicitly market-adjusted stage (see
    :class:`models.calibration.OddsBandCalibrator`).
    """

    def __init__(self, alpha: float, beta: float, metadata: Optional[dict] = None) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.metadata: dict = dict(metadata or {})

    def transform(self, p_model, p_reference, race_ids) -> np.ndarray:
        return power_blend(p_model, p_reference, race_ids, self.alpha, self.beta)

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"model_type": "power_blend", "alpha": self.alpha, "beta": self.beta,
                 "metadata": self.metadata},
                indent=2, default=str,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "PowerBlend":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(state["alpha"], state["beta"], state.get("metadata", {}))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"PowerBlend(alpha={self.alpha:.4g}, beta={self.beta:.4g})"


def select_power_blend(
    p_model,
    p_reference,
    race_ids,
    y_true,
    *,
    alpha_grid: Iterable[float] = ALPHA_GRID,
    beta_grid: Iterable[float] = BETA_GRID,
) -> tuple[PowerBlend, pd.DataFrame]:
    """Exhaustive search over the predetermined grid, by race-level log loss.

    Returns ``(best_blend, full_grid_table)``. The table is returned in full so
    the selection surface — including the ``(1,0)`` independent-only and
    ``(0,1)`` market-only boundary controls — is reportable rather than a single
    unaudited winner.
    """
    rows = []
    for a in alpha_grid:
        for b in beta_grid:
            p = power_blend(p_model, p_reference, race_ids, a, b)
            rows.append({
                "alpha": float(a),
                "beta": float(b),
                "race_log_loss": race_level_log_loss(y_true, p, race_ids),
            })
    table = pd.DataFrame(rows)
    scored = table.dropna(subset=["race_log_loss"])
    if scored.empty:
        raise ValueError("select_power_blend: no grid point produced a scoreable blend")
    best = scored.loc[scored["race_log_loss"].idxmin()]
    blend = PowerBlend(
        best["alpha"], best["beta"],
        metadata={
            "selected_by": "race_level_log_loss",
            "dev_race_log_loss": float(best["race_log_loss"]),
            "n_grid_points": int(len(table)),
        },
    )
    return blend, table
