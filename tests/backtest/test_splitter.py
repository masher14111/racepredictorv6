"""Tests for backtest.splitter — walk-forward fold construction.

The splitter is the single source of truth for no-look-ahead, so the key tests
assert the train/test boundary is strict (no test race is ever <= a train race)
and that folds tile the timeline forward by the configured step.
"""
import pandas as pd
import pytest

from backtest.splitter import Fold, WalkForwardConfig, walk_forward_folds


def _dates(start="2020-01-01", days=800):
    return pd.date_range(start, periods=days, freq="D", tz="UTC").to_series()


# ── fold construction / tiling ────────────────────────────────────────────────

class TestWalkForwardFolds:
    def test_empty_dates_returns_no_folds(self):
        assert walk_forward_folds(pd.Series([], dtype="datetime64[ns]"),
                                  WalkForwardConfig()) == []

    def test_span_shorter_than_min_train_returns_no_folds(self):
        rd = _dates(days=100)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365))
        assert folds == []

    def test_first_origin_is_min_plus_min_train(self):
        rd = _dates(start="2020-01-01", days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365,
                                                         test_window_days=30))
        first = folds[0]
        assert first.train_end == pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=365)
        assert first.test_start == first.train_end

    def test_non_overlapping_windows_tile_by_window(self):
        # step defaults to test_window_days -> back-to-back test windows.
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365,
                                                         test_window_days=30))
        for a, b in zip(folds, folds[1:]):
            assert b.train_end == a.train_end + pd.Timedelta(days=30)
            # previous test_end meets next test_start exactly (no gap/overlap).
            assert a.test_end == b.test_start

    def test_custom_step_smaller_than_window_overlaps(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365,
                                                         test_window_days=30,
                                                         step_days=15))
        assert folds[1].train_end - folds[0].train_end == pd.Timedelta(days=15)

    def test_indices_are_sequential(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365))
        assert [f.index for f in folds] == list(range(len(folds)))

    def test_expanding_window_has_no_train_start(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365, rolling=False))
        assert all(f.train_start is None for f in folds)

    def test_rolling_window_sets_train_start(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365, rolling=True,
                                                         train_window_days=200))
        f = folds[0]
        assert f.train_start == f.train_end - pd.Timedelta(days=200)

    def test_zero_step_raises(self):
        rd = _dates(days=800)
        with pytest.raises(ValueError):
            walk_forward_folds(rd, WalkForwardConfig(min_train_days=365, test_window_days=0,
                                                     step_days=0))


# ── the no-look-ahead guarantee (most important) ──────────────────────────────

class TestNoLookAhead:
    def test_train_and_test_masks_are_disjoint(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365,
                                                         test_window_days=30))
        for f in folds:
            tr = f.train_mask(rd)
            te = f.test_mask(rd)
            assert not (tr & te).any(), f"fold {f.index} has overlapping train/test"

    def test_every_test_race_strictly_after_every_train_race(self):
        rd = _dates(days=800)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365,
                                                         test_window_days=30))
        for f in folds:
            train_dates = rd[f.train_mask(rd)]
            test_dates = rd[f.test_mask(rd)]
            if train_dates.empty or test_dates.empty:
                continue
            # The whole point: max train date < min test date.
            assert train_dates.max() < test_dates.min(), \
                f"fold {f.index} leaks: train max {train_dates.max()} >= test min {test_dates.min()}"

    def test_rolling_train_does_not_reach_before_window(self):
        rd = _dates(days=900)
        folds = walk_forward_folds(rd, WalkForwardConfig(min_train_days=365, rolling=True,
                                                         train_window_days=180))
        for f in folds:
            train_dates = rd[f.train_mask(rd)]
            if train_dates.empty:
                continue
            assert train_dates.min() >= f.train_start

    def test_test_window_upper_bound_inclusive(self):
        # A race exactly on test_end is included; one a day later is not.
        f = Fold(index=0, train_start=None,
                 train_end=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_start=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_end=pd.Timestamp("2021-01-31", tz="UTC"))
        rd = pd.Series(pd.to_datetime(
            ["2021-01-31", "2021-02-01"], utc=True))
        mask = f.test_mask(rd)
        assert mask.tolist() == [True, False]

    def test_train_end_inclusive_test_start_exclusive(self):
        # A race on the origin T belongs to TRAIN, never TEST.
        f = Fold(index=0, train_start=None,
                 train_end=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_start=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_end=pd.Timestamp("2021-01-31", tz="UTC"))
        rd = pd.Series(pd.to_datetime(["2021-01-01"], utc=True))
        assert f.train_mask(rd).tolist() == [True]
        assert f.test_mask(rd).tolist() == [False]

    def test_multiple_races_sharing_a_date_never_split_within_the_race(self):
        """Step 02 audit: race_date is date-only (docs/improvement/CONTRACTS.md),
        so every row of one race shares exactly one race_date value. Because
        Fold masks cut on race_date (not row count or row index), a race can
        never have some rows land in train and others in test — verify this
        holds even with several distinct races sharing the boundary date."""
        f = Fold(index=0, train_start=None,
                 train_end=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_start=pd.Timestamp("2021-01-01", tz="UTC"),
                 test_end=pd.Timestamp("2021-01-31", tz="UTC"))
        race_uid = pd.Series(
            ["A|2021-01-01T13:00", "A|2021-01-01T13:00", "B|2021-01-01T14:30",
             "B|2021-01-01T14:30", "C|2021-01-02T13:00", "C|2021-01-02T13:00"])
        race_date = pd.Series(pd.to_datetime(
            ["2021-01-01", "2021-01-01", "2021-01-01",
             "2021-01-01", "2021-01-02", "2021-01-02"], utc=True))
        tr = f.train_mask(race_date)
        te = f.test_mask(race_date)
        for uid in race_uid.unique():
            rows = race_uid == uid
            assert not (tr[rows].any() and te[rows].any()), \
                f"race {uid} split across train/test"
