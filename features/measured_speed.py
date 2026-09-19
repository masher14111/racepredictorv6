"""Normalized measured-performance figures and their pre-race features (step 11).

``features/engine.py::add_speed_figures`` builds ``horse_speed``: a *finishing
position percentile* proxy. Its meaning is preserved exactly. This module is a
separately named, additive family built from the **measured** timings mined by
:mod:`features.measured_timing`.

Two stages, with different leak properties:

1. :func:`add_speed_figures` turns each runner's measured speed into a figure
   normalized against a **par** for the race's course, distance, surface, going
   and race type. A figure describes a race that has already been run, so it is
   *history*, never a feature of the race it comes from.
2. :func:`add_measured_pre_race_features` turns a horse's **prior** figures into
   pre-race features, through ``features/_trailing_fast.py`` so the family
   inherits D26's market-duplicate collapse and D39's real off-time ordering.

The par is fitted on **earlier races only**
---------------------------------------------
For every par key, the mean/std of ``log(winner speed)`` is accumulated over
races at a *strictly earlier off time* than the race being normalized. It is
computed by aggregating whole same-instant blocks and taking the cumulative
total up to (not including) each block, which gives two properties the
acceptance criteria require and :mod:`tests.features.test_measured_speed`
asserts directly:

* **append invariance** - adding later races never changes an earlier race's par
* **row-order invariance** - the result does not depend on frame order, because
  no same-instant race contributes to its own block

Backoff. The most specific key with at least :data:`MIN_PRIOR_RACES` prior races
wins; otherwise the next one down. ``par_level`` records which was used, so a
figure resting on a thin cell can always be identified rather than silently
trusted.

Units. Figures are built from ``log`` speed in yards/second.
``measured_speed_pct`` is 100x the log-ratio to par (positive = faster than
par); ``measured_speed_z`` divides by the par's prior standard deviation, which
is what makes figures from different courses and distances comparable. A
non-finisher has no time and therefore a **null** figure - never a slow one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features import _trailing_fast
from features.derive import _going_band

#: A par cell needs this many strictly-prior races before it is trusted.
MIN_PRIOR_RACES = 15

#: Par key hierarchy, most specific first. ``()`` is the global fallback.
PAR_LEVELS: tuple[tuple[str, ...], ...] = (
    ("venue", "surface", "race_type", "distance_yards", "going_band"),
    ("venue", "surface", "race_type", "distance_yards"),
    ("venue", "surface", "race_type", "furlong_band", "going_band"),
    ("venue", "surface", "race_type", "furlong_band"),
    ("surface", "race_type", "furlong_band", "going_band"),
    ("surface", "race_type", "furlong_band"),
    ("race_type", "furlong_band"),
    (),
)

#: The additive pre-race feature family this module contributes.
MEASURED_FEATURE_COLS = [
    "msf_last",
    "msf_mean3",
    "msf_mean6",
    "msf_trend",
    "msf_distance",
    "msf_going",
    "msf_rank",
]


# ─────────────────────────────── par fitting ────────────────────────────────
def _prior_stats(races: pd.DataFrame, keys: tuple[str, ...]) -> pd.DataFrame:
    """Strictly-prior mean/std/count of ``y`` for one par key.

    Aggregates each (key, off_time) block first and takes the cumulative total
    *before* that block, so a race never contributes to its own par and frame
    order cannot matter.
    """
    frame = races.loc[races["y"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=list(keys) + ["off_time", "par_mean", "par_std", "par_n"])
    frame["_y2"] = frame["y"] ** 2
    group_cols = list(keys) + ["off_time"]
    blocks = (frame.groupby(group_cols, dropna=False, sort=False)
              .agg(_sum=("y", "sum"), _sumsq=("_y2", "sum"), _n=("y", "size"))
              .reset_index()
              .sort_values(group_cols, kind="stable"))

    if keys:
        grouped = blocks.groupby(list(keys), dropna=False, sort=False)
        prior_sum = grouped["_sum"].cumsum() - blocks["_sum"]
        prior_sumsq = grouped["_sumsq"].cumsum() - blocks["_sumsq"]
        prior_n = grouped["_n"].cumsum() - blocks["_n"]
    else:
        prior_sum = blocks["_sum"].cumsum() - blocks["_sum"]
        prior_sumsq = blocks["_sumsq"].cumsum() - blocks["_sumsq"]
        prior_n = blocks["_n"].cumsum() - blocks["_n"]

    n = prior_n.to_numpy(dtype="float64")
    mean = np.divide(prior_sum.to_numpy(dtype="float64"), n,
                     out=np.full(len(n), np.nan), where=n > 0)
    var = np.divide(prior_sumsq.to_numpy(dtype="float64"), n,
                    out=np.full(len(n), np.nan), where=n > 0) - mean ** 2
    out = blocks[group_cols].copy()
    out["par_mean"] = mean
    out["par_std"] = np.sqrt(np.clip(var, 0.0, None))
    out["par_n"] = n
    return out


def add_race_pars(races: pd.DataFrame, *, min_prior: int = MIN_PRIOR_RACES) -> pd.DataFrame:
    """Attach ``par_mean``/``par_std``/``par_n``/``par_level`` to a race frame.

    ``races`` needs ``off_time``, ``venue``, ``surface``, ``going``,
    ``race_type``, ``distance_yards`` and ``winner_speed_yps``.
    """
    out = races.copy().reset_index(drop=True)
    out["going_band"] = [_going_band(g) for g in out.get("going", pd.Series(index=out.index))]
    out["going_band"] = out["going_band"].fillna("(unknown)")
    out["furlong_band"] = (pd.to_numeric(out["distance_yards"], errors="coerce")
                           / 220.0).round().astype("Float64")
    if "winner_speed_yps" in out.columns:
        speed = pd.to_numeric(out["winner_speed_yps"], errors="coerce")
    else:
        # Race-level extracts carry the two published figures, not the ratio.
        yards = pd.to_numeric(out["distance_yards"], errors="coerce")
        secs = pd.to_numeric(out["winning_time_seconds"], errors="coerce")
        speed = yards / secs.where(secs > 0)
        out["winner_speed_yps"] = speed
    out["y"] = np.log(speed.where(speed > 0))

    out["par_mean"] = np.nan
    out["par_std"] = np.nan
    out["par_n"] = 0.0
    out["par_level"] = pd.NA

    unresolved = pd.Series(True, index=out.index)
    for level, keys in enumerate(PAR_LEVELS):
        if not unresolved.any():
            break
        stats = _prior_stats(out, keys)
        if stats.empty:
            continue
        merged = out[list(keys) + ["off_time"]].merge(
            stats, on=list(keys) + ["off_time"], how="left")
        ok = unresolved & (merged["par_n"].to_numpy() >= min_prior)
        out.loc[ok, "par_mean"] = merged.loc[ok, "par_mean"].to_numpy()
        out.loc[ok, "par_std"] = merged.loc[ok, "par_std"].to_numpy()
        out.loc[ok, "par_n"] = merged.loc[ok, "par_n"].to_numpy()
        out.loc[ok, "par_level"] = level
        unresolved &= ~ok
    return out


def add_speed_figures(runners: pd.DataFrame, races: pd.DataFrame, *,
                      min_prior: int = MIN_PRIOR_RACES) -> pd.DataFrame:
    """Per-runner normalized measured figures, joined onto a par-fitted race set.

    Adds ``measured_speed_pct`` (100x log-ratio to par) and
    ``measured_speed_z`` (standardized by the par's prior spread). Both are
    **post-race** quantities describing the run they come from.
    """
    pars = add_race_pars(races, min_prior=min_prior)
    cols = ["race_key", "par_mean", "par_std", "par_n", "par_level",
            "going_band", "furlong_band"]
    out = runners.merge(pars[cols], on="race_key", how="left")

    speed = pd.to_numeric(out.get("est_speed_yps"), errors="coerce")
    log_speed = np.log(speed.where(speed > 0))
    # A runner that did not complete has no time, so it gets no figure at all.
    finished = out["finished"].astype("boolean").fillna(False).to_numpy()
    delta = (log_speed - out["par_mean"]).where(pd.Series(finished, index=out.index))
    out["measured_speed_pct"] = 100.0 * delta
    std = pd.to_numeric(out["par_std"], errors="coerce")
    out["measured_speed_z"] = delta / std.where(std > 0)
    return out


# ───────────────────────── pre-race feature family ──────────────────────────
def add_measured_pre_race_features(
    df: pd.DataFrame, *, figure_col: str = "measured_speed_z",
    lookback_months: int = 24, distance_months: int = 36,
) -> pd.DataFrame:
    """Turn a horse's **prior** measured figures into pre-race features.

    ``df`` must already carry ``figure_col`` as the value *that row's own race*
    produced, plus ``race_uid``/``race_date``/``horse_id``. Every output is a
    strictly-prior trailing aggregate via ``features/_trailing_fast.py``, so the
    race being predicted never contributes to its own feature.
    """
    out = df.copy().reset_index(drop=True)
    if figure_col not in out.columns:
        for col in MEASURED_FEATURE_COLS:
            out[col] = np.nan
        return out

    def trailing(keys, runs, months):
        return _trailing_fast.windowed_mean(out, keys, figure_col, runs=runs,
                                            months=months, strict=False, dropna=True)

    out["msf_last"] = trailing(["horse_id"], 1, lookback_months)
    out["msf_mean3"] = trailing(["horse_id"], 3, lookback_months)
    out["msf_mean6"] = trailing(["horse_id"], 6, lookback_months)
    out["msf_trend"] = out["msf_mean3"] - out["msf_mean6"]

    if "_msf_dist_band" not in out.columns:
        out["_msf_dist_band"] = (pd.to_numeric(out.get("distance_furlongs"), errors="coerce")
                                 .round().astype("Float64").astype("string").fillna("(na)"))
    out["msf_distance"] = trailing(["horse_id", "_msf_dist_band"], 6, distance_months)

    if "_msf_going_band" not in out.columns:
        band = out["going_band"] if "going_band" in out.columns else pd.Series(
            [_going_band(g) for g in out.get("going", pd.Series(index=out.index))],
            index=out.index)
        out["_msf_going_band"] = pd.Series(band, index=out.index).astype("string").fillna("(na)")
    out["msf_going"] = trailing(["horse_id", "_msf_going_band"], 6, distance_months)

    # Within-race rank of the pre-race form figure: pure ordering of values that
    # are themselves already strictly prior, so it adds no new information about
    # the race being predicted.
    group = _race_cols(out)
    out["msf_rank"] = out.groupby(group, sort=False)["msf_mean3"].rank(
        ascending=False, method="min")
    return out.drop(columns=["_msf_dist_band", "_msf_going_band"], errors="ignore")


def _race_cols(df: pd.DataFrame) -> list[str]:
    """Race grouping that keeps WIN and PLACE books apart (D23)."""
    cols = ["race_uid"] if "race_uid" in df.columns else ["race_date", "venue"]
    if "market_type" in df.columns:
        cols = cols + ["market_type"]
    return cols
