"""Class-par performance figures: speed adjusted for race class.

Adds per-runner columns:
  - class_par_speed: horse_speed adjusted upward/downward based on the class of
    the race the figure was set in vs a baseline class level.
  - class_par_rank:  within-race descending rank of class_par_speed.

Rationale: a speed figure posted in a Class-2 handicap is more impressive than
the same figure in a Class-6 seller. By shifting each horse's best recent speed
by a class-par adjustment, runners from different class backgrounds become
comparable on raw ability.

All adjustments are leak-free: class_par_speed is computed from horse_speed and
race_class, both of which are known pre-race. No future runs are consulted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.derive import _race_cols

# Speed-point adjustment per class level above/below the baseline (class 5).
# A higher class (lower number) is faster, so moving from class 2 → 5 means
# the horse faced tougher opposition → its figure should be adjusted UP when
# compared against runners from lower classes.
_DEFAULT_CLASS_POINTS_PER_LEVEL = 2.0
_DEFAULT_BASE_CLASS = 5


def _class_adjustment(
    race_class: float | None,
    base_class: int = _DEFAULT_BASE_CLASS,
    points_per_level: float = _DEFAULT_CLASS_POINTS_PER_LEVEL,
) -> float:
    """Speed adjustment for a horse whose `horse_speed` was set at ``race_class``.

    Classes in UK/IRE racing are numbered so lower = higher quality (class 1 is
    Group/Listed, class 6 is lowest). Adjusting UP means adding points per class
    level below base (tougher company → more credit).

    Returns 0.0 when race_class is unknown (no adjustment).
    """
    if race_class is None or pd.isna(race_class) or race_class <= 0:
        return 0.0
    return (base_class - float(race_class)) * points_per_level


def add_class_par(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add class_par_speed and class_par_rank.

    Parameters
    ----------
    df : pd.DataFrame
        Must carry ``horse_speed`` and ``race_class`` columns.
    cfg : dict or None
        Optional override: ``base_class``, ``class_points_per_level``.
    """
    cfg = cfg or {}
    base_class = int(cfg.get("class_par_base_class", _DEFAULT_BASE_CLASS))
    pts = float(cfg.get("class_par_points_per_level", _DEFAULT_CLASS_POINTS_PER_LEVEL))

    out = df.copy()

    speed = pd.to_numeric(out.get("horse_speed"), errors="coerce") \
        if "horse_speed" in out.columns else pd.Series(pd.NA, index=out.index)
    rclass = pd.to_numeric(out.get("race_class"), errors="coerce") \
        if "race_class" in out.columns else pd.Series(pd.NA, index=out.index)

    adj = np.array([
        _class_adjustment(rc, base_class, pts) for rc in rclass
    ], dtype=float)
    out["class_par_speed"] = speed.to_numpy(dtype=float) + adj
    # Clamp negative adjustments don't produce physically-impossible negative speeds
    out["class_par_speed"] = out["class_par_speed"].clip(lower=0.0)

    # Within-race rank: higher class-par speed = better
    rkey = _race_cols(out)
    out["class_par_rank"] = out.groupby(rkey, sort=False)["class_par_speed"].rank(
        ascending=False, method="min"
    )
    return out