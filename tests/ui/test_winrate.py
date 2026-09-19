"""UI-label regressions: realized win rates always carry their denominator and
interval (Stage-4 requirement 9)."""
import math

from ui._winrate import realized_rate_sub, realized_rate_value, wilson_interval


class TestWilson:
    def test_contains_point_estimate(self):
        lo, hi = wilson_interval(37, 100)
        assert lo < 0.37 < hi

    def test_zero_n_is_nan(self):
        lo, hi = wilson_interval(0, 0)
        assert math.isnan(lo) and math.isnan(hi)

    def test_extreme_rates_stay_in_unit_interval(self):
        lo, hi = wilson_interval(3, 3)
        assert 0.0 <= lo <= hi <= 1.0
        # 3/3 must NOT read as a certain 100% — the interval is wide.
        assert lo < 0.5


class TestFormatting:
    def test_value_always_shows_denominator(self):
        assert realized_rate_value(74, 200) == "37% (74/200)"

    def test_zero_bets_never_shows_a_rate(self):
        assert realized_rate_value(0, 0) == "— (0/0)"
        assert realized_rate_sub(0, 0) == "no settled bets yet"

    def test_sub_carries_ci_and_window(self):
        sub = realized_rate_sub(74, 200, window_label="last 30 days")
        assert "95% CI" in sub
        assert "last 30 days" in sub

    def test_small_sample_interval_is_wide(self):
        sub = realized_rate_sub(3, 3)
        assert "95% CI" in sub
        lo, hi = wilson_interval(3, 3)
        assert (hi - lo) > 0.5  # 3-bet "100%" reads as deeply uncertain
