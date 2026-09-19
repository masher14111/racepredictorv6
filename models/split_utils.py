"""Shared whole-race-group chronological splitting for training and tuning.

Every boundary that separates model-fitting rows from rows used to pick
hyperparameters, calibrate probabilities, or score a held-out test must keep
every row of a given race (``race_uid``) on the same side. Splitting by
absolute row count (the old ``models/train.py::_time_split`` behaviour) does
not guarantee this: a race whose rows straddle the cut index leaks across the
boundary. Everything here cuts on whole race-group boundaries instead.

Two primitives cover every caller in this codebase:

* :func:`chronological_group_split` — a single train/test-style cut into two
  disjoint, chronologically-ordered blocks (used for the outer test carve, the
  calibration carve, and the early-stopping carve in ``models/train.py`` and
  ``models/ensemble_experiment.py``).
* :func:`group_time_series_split` — a walk-forward (expanding-window) K-fold
  generator over whole groups (used by ``models/tuner.py`` in place of a plain
  row-indexed ``sklearn.model_selection.TimeSeriesSplit``).

Both are deterministic given the same ``(group_ids, order_keys)`` regardless
of input row order (rows are never assumed pre-sorted), and neither requires
or assumes a horse's races stay in one partition — grouping is by race only,
so a horse's earlier races may fall in an earlier block and its later races in
a later block, which is the realistic prior-history relationship.
"""
from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _order_keys_and_groups(
    group_ids: Sequence, order_keys: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs shared by both split functions."""
    groups = np.asarray(group_ids)
    keys = pd.to_datetime(pd.Series(np.asarray(order_keys)), utc=True, errors="coerce")
    if len(groups) != len(keys):
        raise ValueError("group_ids and order_keys must be the same length")
    return groups, keys.to_numpy()


def _group_chronology(groups: np.ndarray, keys: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per unique group: its earliest order_key and a stable tiebreak.

    Rows with a null/unknown group id are each treated as their own singleton
    group (keyed by row position) rather than silently merged together — an
    unknown race key must never let two otherwise-unrelated rows "share" an
    identity that would make them look like the same race for leakage checks.
    Groups whose order_key is entirely unresolved (NaT) sort deterministically
    last, rather than relying on pandas' NaT ordering.
    """
    is_na = pd.isna(groups)
    keyed = groups.astype(object).copy()
    if is_na.any():
        for pos in np.where(is_na)[0]:
            keyed[pos] = ("__unknown_race_key__", int(pos))

    frame = pd.DataFrame({"group": keyed, "order_key": keys})
    chrono = frame.groupby("group", sort=False)["order_key"].min().reset_index()
    chrono["_group_str"] = chrono["group"].astype(str)
    chrono["_nat"] = chrono["order_key"].isna()
    chrono = chrono.sort_values(["_nat", "order_key", "_group_str"], kind="stable")
    return chrono.reset_index(drop=True), keyed


def chronological_group_split(
    group_ids: Sequence,
    order_keys: Sequence,
    frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split row positions into (left, right) blocks on whole-group boundaries.

    ``right`` holds the most-recent groups, sized to be the smallest suffix of
    chronologically-sorted groups whose row count is >= ``frac`` of all rows
    (i.e. rounds up to the nearest whole group rather than truncating mid-race).
    ``left`` holds everything else, strictly earlier in ``order_keys`` than any
    row in ``right`` at the group-boundary granularity.

    Parameters
    ----------
    group_ids  : per-row group identity (e.g. race_uid). May contain nulls;
                 each null row is treated as its own singleton group.
    order_keys : per-row chronological ordering key (e.g. race_date). Rows of
                 the same group should share one value; only the group's
                 minimum is used, so a mismatched stray value cannot move part
                 of a group across the boundary.
    frac       : desired right-side row fraction, in [0, 1].

    Returns
    -------
    (left_idx, right_idx) : arrays of integer row positions into the original
    ``group_ids``/``order_keys`` arrays (not a copy of the data). Deterministic
    regardless of input row order; safe to reuse to index a DataFrame or a
    feature matrix built in the same row order.
    """
    if not 0.0 <= frac <= 1.0:
        raise ValueError(f"frac must be in [0, 1], got {frac}")
    groups, keys = _order_keys_and_groups(group_ids, order_keys)
    n = len(groups)
    if n == 0 or frac <= 0.0:
        return np.arange(n), np.array([], dtype=int)
    if frac >= 1.0:
        return np.array([], dtype=int), np.arange(n)

    chrono, keyed = _group_chronology(groups, keys)
    sizes = pd.Series(keyed).value_counts()
    chrono["size"] = chrono["group"].map(sizes).to_numpy()

    total = int(chrono["size"].sum())
    target_right = frac * total

    # Walk from the most-recent group backwards, accumulating rows, until the
    # accumulated count first reaches the target — that group and everything
    # after it forms the right block. A single group can never itself be split.
    sizes_desc = chrono["size"].to_numpy()[::-1]
    cum_from_end = np.cumsum(sizes_desc)
    n_right_groups = int(np.searchsorted(cum_from_end, target_right, side="left")) + 1
    n_right_groups = min(n_right_groups, len(chrono))
    if len(chrono) == 1:
        # Only one group exists overall — it cannot straddle a boundary, so it
        # goes to whichever side its own presence made non-empty already
        # decided above; a single group is indivisible, put it all in left.
        return np.arange(n), np.array([], dtype=int)

    right_groups = set(chrono["group"].to_numpy()[-n_right_groups:])
    right_mask = np.array([g in right_groups for g in keyed])
    right_idx = np.where(right_mask)[0]
    left_idx = np.where(~right_mask)[0]
    return left_idx, right_idx


def group_time_series_split(
    group_ids: Sequence,
    order_keys: Sequence,
    n_splits: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward (expanding-window) K-fold over whole groups, not rows.

    Behaves like ``sklearn.model_selection.TimeSeriesSplit`` conceptually
    (fold *i* trains on everything chronologically before its validation
    slice) except the unit of division is a whole group (race), so no group's
    rows are ever split between a fold's train and validation sides. Groups
    are chronologically ordered by their minimum ``order_keys`` value (with a
    deterministic string tiebreak on the group id for equal dates), so the
    result does not depend on the input row order.

    Yields ``n_splits`` ``(train_idx, val_idx)`` pairs of row positions. Raises
    ``ValueError`` if there are fewer than ``n_splits + 1`` unique groups (not
    enough groups to form the requested number of expanding folds).
    """
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")
    groups, keys = _order_keys_and_groups(group_ids, order_keys)
    chrono, keyed = _group_chronology(groups, keys)
    n_groups = len(chrono)
    if n_groups < n_splits + 1:
        raise ValueError(
            f"group_time_series_split: need at least {n_splits + 1} unique groups "
            f"for {n_splits} folds, found {n_groups}"
        )

    ordered_group_ids = chrono["group"].to_numpy()
    fold_chunks = np.array_split(ordered_group_ids, n_splits + 1)

    group_to_positions: dict = {}
    for pos, g in enumerate(keyed):
        group_to_positions.setdefault(g, []).append(pos)

    train_groups: list = []
    for i in range(n_splits):
        train_groups.extend(fold_chunks[i].tolist())
        val_groups = fold_chunks[i + 1].tolist()
        train_idx = np.array(
            sorted(pos for g in train_groups for pos in group_to_positions[g])
        )
        val_idx = np.array(
            sorted(pos for g in val_groups for pos in group_to_positions[g])
        )
        yield train_idx, val_idx


def race_group_overlap(*index_and_group_pairs: Tuple[np.ndarray, np.ndarray]) -> int:
    """Count race_uid values shared across two or more (idx, group_ids) pairs.

    Utility for regression tests / audits: pass ``(idx, group_ids)`` pairs
    where ``group_ids`` is the full per-row group array and ``idx`` selects
    the rows belonging to one partition. Returns the total number of distinct
    group ids that appear in more than one of the supplied partitions.
    """
    if len(index_and_group_pairs) < 2:
        raise ValueError("need at least two partitions to check overlap")
    sets = []
    for idx, group_ids in index_and_group_pairs:
        arr = np.asarray(group_ids)[np.asarray(idx)]
        sets.append(set(arr.tolist()))
    shared: set = set()
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            shared |= sets[i] & sets[j]
    return len(shared)
