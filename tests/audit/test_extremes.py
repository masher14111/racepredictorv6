"""Probability-extreme diagnostics: the audit must detect exact-0/1 emission,
support clamping, and band-occupancy shift on synthetic calibrators."""
import numpy as np
import pandas as pd
import pytest

from audit.extremes import (
    band_occupancy,
    cache_extremes,
    describe_base_calibrator,
    describe_fl_bands,
    grid_extremes,
    support_hit_rate,
)
from models.calibration import (
    IsotonicCalibrator,
    OddsBandCalibrator,
    SigmoidCalibrator,
)


def _zero_floor_obc() -> OddsBandCalibrator:
    """Two-band OBC whose isotonic maps clamp to 0 below support and to a
    sub-1.0 ceiling above — the production failure shape."""
    lo = IsotonicCalibrator(x=[0.05, 0.2, 0.4], y=[0.0, 0.3, 0.8])
    hi = IsotonicCalibrator(x=[0.02, 0.1, 0.3], y=[0.0, 0.05, 0.2])
    return OddsBandCalibrator(centers=[np.log(2.5), np.log(10.0)],
                              calibrators=[lo, hi])


def _sigmoid_obc() -> OddsBandCalibrator:
    lo = SigmoidCalibrator(a=1.0, b=0.5)
    hi = SigmoidCalibrator(a=1.0, b=-1.0)
    return OddsBandCalibrator(centers=[np.log(2.5), np.log(10.0)],
                              calibrators=[lo, hi])


class TestGridExtremes:
    def test_isotonic_zero_floor_detected(self):
        res = grid_extremes(_zero_floor_obc())
        assert res["any_zero"]
        zero_odds = {c["odds"] for c in res["zero_cells"]}
        assert 2.5 in zero_odds or 2.0 in zero_odds

    def test_sigmoid_bands_emit_no_extremes(self):
        res = grid_extremes(_sigmoid_obc())
        assert not res["any_zero"]
        assert not res["any_one"]


class TestSupportHitRate:
    def test_below_support_counted(self):
        obc = _zero_floor_obc()
        prob = np.array([0.01, 0.01, 0.3, 0.5])   # two below band-lo support
        odds = np.array([2.5, 2.5, 2.5, 2.5])
        res = support_hit_rate(obc, prob, odds)
        assert res["n"] == 4
        assert res["below_support_frac"] == 0.5
        assert res["above_support_frac"] == 0.25   # 0.5 > x_max 0.4

    def test_empty_input(self):
        assert support_hit_rate(_zero_floor_obc(), np.array([]),
                                np.array([]))["n"] == 0


class TestDescribe:
    def test_base_sigmoid_cannot_emit_extremes(self):
        d = describe_base_calibrator(SigmoidCalibrator(a=1.1, b=0.5))
        assert d["can_emit_zero"] is False and d["can_emit_one"] is False

    def test_base_isotonic_zero_floor_flagged(self):
        d = describe_base_calibrator(IsotonicCalibrator(x=[0.1, 0.5],
                                                        y=[0.0, 1.0]))
        assert d["can_emit_zero"] and d["can_emit_one"]

    def test_fl_band_table_shape(self):
        t = describe_fl_bands(_zero_floor_obc())
        assert len(t) == 2
        assert t["emits_zero_below_x_min"].all()


class TestOccupancyAndCache:
    def test_band_occupancy_sums_to_one(self):
        occ = band_occupancy(pd.Series([1.5, 2.5, 4.0, 9.0, 40.0, 100.0]))
        assert occ.sum() == pytest.approx(1.0)

    def test_cache_extremes_counts_zeros(self):
        cache = {"races": [{"runners": [
            {"value_win_prob": 0.0, "won_prob": 0.2},
            {"value_win_prob": 0.5, "won_prob": 0.3},
            {"value_win_prob": 1.0, "won_prob": 0.4},
        ]}]}
        c = cache_extremes(cache)
        assert c["value_win_prob"] == {"n": 3, "zeros": 1, "ones": 1}
        assert c["won_prob"]["zeros"] == 0
