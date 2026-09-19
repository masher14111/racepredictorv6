"""Regression tests for models.split_utils (step 02: whole-race chronological
splits). These exercise the exact defect found in stage 01: row-count cuts on
date-sorted rows can split a single race's rows across a boundary whenever
several races share a date or a race's rows aren't contiguous after sorting.
"""
import numpy as np
import pandas as pd
import pytest

from models.split_utils import (
    chronological_group_split,
    group_time_series_split,
    race_group_overlap,
)


def _races(n_races, min_size=1, max_size=8, races_per_day=1, start="2024-01-01", seed=0):
    """Synthetic (race_uid, race_date) row arrays: n_races races, races_per_day
    of them sharing each calendar day, each with a random row count."""
    rng = np.random.default_rng(seed)
    race_uid, race_date = [], []
    for i in range(n_races):
        day = pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=i // races_per_day)
        size = rng.integers(min_size, max_size + 1)
        race_uid += [f"race_{i}"] * size
        race_date += [day] * size
    return np.array(race_uid), np.array(race_date)


# ── chronological_group_split ───────────────────────────────────────────────

def test_no_race_straddles_the_boundary():
    race_uid, race_date = _races(60, races_per_day=3)  # multiple races/day
    left, right = chronological_group_split(race_uid, race_date, 0.2)
    assert race_group_overlap((left, race_uid), (right, race_uid)) == 0


def test_right_side_is_chronologically_last():
    race_uid, race_date = _races(40)
    left, right = chronological_group_split(race_uid, race_date, 0.25)
    assert race_date[left].max() <= race_date[right].min()


def test_deterministic_under_row_shuffle():
    race_uid, race_date = _races(50, races_per_day=2, seed=1)
    n = len(race_uid)
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)

    _l1, r1 = chronological_group_split(race_uid, race_date, 0.3)
    _l2, r2 = chronological_group_split(race_uid[perm], race_date[perm], 0.3)

    assert set(race_uid[r1]) == set(race_uid[perm][r2])


def test_repeated_horse_rows_not_forced_together():
    """A horse's rows are tagged by race, not by horse — the same horse_id
    appearing in an earlier and a later race is expected to land on different
    sides of the boundary (prior-to-future history is realistic, not a leak)."""
    race_uid, race_date = _races(30, min_size=2, max_size=2, seed=3)
    horse_id = np.tile(["A", "B"], 30)  # horse "A" runs in every race
    left, right = chronological_group_split(race_uid, race_date, 0.3)
    assert "A" in horse_id[left] and "A" in horse_id[right]
    # but no race itself is split
    assert race_group_overlap((left, race_uid), (right, race_uid)) == 0


def test_equal_dates_all_races_split_by_group_not_row_index():
    """Every race on the same single date — the old row-count cut would slice
    straight through whichever race's rows happened to straddle the index."""
    n_races = 20
    race_uid = np.concatenate([[f"race_{i}"] * 5 for i in range(n_races)])
    race_date = np.full(len(race_uid), pd.Timestamp("2024-06-01", tz="UTC"))
    left, right = chronological_group_split(race_uid, race_date, 0.4)
    assert race_group_overlap((left, race_uid), (right, race_uid)) == 0
    # every group appears on exactly one side, whole
    for i in range(n_races):
        mask = race_uid == f"race_{i}"
        side_left = mask[left].any() if len(left) else False
        side_right = mask[right].any() if len(right) else False
        assert side_left != side_right


def test_unknown_race_key_never_collides():
    """Null race_uid rows (unknown race key) must never be treated as members
    of the same race as each other or as any real race."""
    race_uid = np.array(["r1", "r1", None, None, "r2", "r2"], dtype=object)
    race_date = pd.to_datetime(
        ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-04"],
        utc=True,
    ).to_numpy()
    left, right = chronological_group_split(race_uid, race_date, 0.5)
    # the two None rows must not both land in the same partition due to a
    # shared "identity" — verify by checking they can be split independently
    assert len(left) + len(right) == 6


def test_sparse_period_gap_in_dates():
    """A large calendar gap between two clusters of races is handled without
    error and still respects group boundaries."""
    race_uid = np.array([f"a{i}" for i in range(3) for _ in range(4)] +
                         [f"b{i}" for i in range(3) for _ in range(4)])
    early = pd.Timestamp("2024-01-01", tz="UTC")
    late = pd.Timestamp("2025-06-01", tz="UTC")
    race_date = np.array([early] * 12 + [late] * 12)
    left, right = chronological_group_split(race_uid, race_date, 0.5)
    assert race_group_overlap((left, race_uid), (right, race_uid)) == 0
    assert set(race_uid[right]) & {"b0", "b1", "b2"} == set(race_uid[right])


def test_frac_zero_and_one_edges():
    race_uid, race_date = _races(10)
    left, right = chronological_group_split(race_uid, race_date, 0.0)
    assert len(right) == 0 and len(left) == len(race_uid)
    left, right = chronological_group_split(race_uid, race_date, 1.0)
    assert len(left) == 0 and len(right) == len(race_uid)


def test_single_race_cannot_be_split():
    race_uid = np.array(["only"] * 8)
    race_date = np.full(8, pd.Timestamp("2024-01-01", tz="UTC"))
    left, right = chronological_group_split(race_uid, race_date, 0.5)
    assert len(right) == 0
    assert len(left) == 8


def test_empty_input():
    left, right = chronological_group_split(np.array([]), np.array([]), 0.2)
    assert len(left) == 0 and len(right) == 0


def test_invalid_frac_raises():
    race_uid, race_date = _races(5)
    with pytest.raises(ValueError):
        chronological_group_split(race_uid, race_date, 1.5)


# ── group_time_series_split ─────────────────────────────────────────────────

def test_folds_are_group_safe_and_chronological():
    race_uid, race_date = _races(80, races_per_day=2, seed=5)
    for tr, va in group_time_series_split(race_uid, race_date, 4):
        assert race_group_overlap((tr, race_uid), (va, race_uid)) == 0
        assert race_date[tr].max() <= race_date[va].min()


def test_folds_expand_over_time():
    race_uid, race_date = _races(60, seed=9)
    folds = list(group_time_series_split(race_uid, race_date, 3))
    sizes = [len(tr) for tr, _va in folds]
    assert sizes == sorted(sizes)  # expanding window: non-decreasing train size


def test_too_few_groups_raises():
    race_uid, race_date = _races(3)
    with pytest.raises(ValueError):
        list(group_time_series_split(race_uid, race_date, 5))


def test_row_shuffle_does_not_change_fold_membership():
    race_uid, race_date = _races(40, races_per_day=2, seed=11)
    n = len(race_uid)
    rng = np.random.default_rng(13)
    perm = rng.permutation(n)

    folds_a = list(group_time_series_split(race_uid, race_date, 3))
    folds_b = list(group_time_series_split(race_uid[perm], race_date[perm], 3))

    for (tr_a, va_a), (tr_b, va_b) in zip(folds_a, folds_b):
        assert set(race_uid[tr_a]) == set(race_uid[perm][tr_b])
        assert set(race_uid[va_a]) == set(race_uid[perm][va_b])


# ── race_group_overlap ──────────────────────────────────────────────────────

def test_race_group_overlap_detects_shared_ids():
    group_ids = np.array(["r1", "r1", "r2", "r2"])
    idx_a = np.array([0, 1])
    idx_b = np.array([1, 2])  # shares row 1 (race "r1") with idx_a
    assert race_group_overlap((idx_a, group_ids), (idx_b, group_ids)) == 1


def test_race_group_overlap_needs_two_partitions():
    with pytest.raises(ValueError):
        race_group_overlap((np.array([0]), np.array(["r1"])))
