"""Local-time day/month bucketing.

Timestamps are parsed with utc=True everywhere, so bucketing them straight off
the UTC clock puts a late-night Irish bet in the wrong day — and at a month
boundary, the wrong month. Through IST (UTC+1) anything from 00:00 to 00:59
local belongs to the previous UTC day, which is what these pin down.
"""

import pandas as pd
import pytest

from utils.timezone import TZ, local_day, local_month


def _utc(*stamps):
    """A UTC-parsed Series, the shape these helpers see in production."""
    return pd.Series(pd.to_datetime(list(stamps), utc=True))


def test_configured_timezone_is_dublin():
    assert str(TZ) == "Europe/Dublin"


# ---------------------------------------------------------------------------
# The bug: midnight-local during IST rolls back a UTC day
# ---------------------------------------------------------------------------
def test_just_after_local_midnight_keeps_the_local_day():
    """00:30 IST on 2 July is 23:30 UTC on 1 July. The bet happened on the 2nd."""
    s = _utc("2026-07-01T23:30:00Z")
    assert str(local_day(s)[0]) == "2026-07-02"
    assert str(s.dt.date[0]) == "2026-07-01"       # the old, wrong answer


def test_month_boundary_is_not_rolled_back():
    """00:30 IST on 1 August is 23:30 UTC on 31 July — an August bet."""
    s = _utc("2026-07-31T23:30:00Z")
    assert str(local_month(s)[0]) == "2026-08"


def test_evening_card_stays_put():
    """A 20:30 Dundalk race is 19:30 UTC — same day either way."""
    s = _utc("2026-09-18T19:30:00Z")
    assert str(local_day(s)[0]) == "2026-09-18"
    assert str(local_month(s)[0]) == "2026-09"


def test_winter_utc_offset_is_zero_so_nothing_shifts():
    """In GMT (winter) local == UTC, so the fix must be a no-op there."""
    s = _utc("2026-01-15T23:30:00Z")
    assert str(local_day(s)[0]) == "2026-01-15"
    assert str(local_month(s)[0]) == "2026-01"


def test_dst_transition_day_is_handled():
    """IST begins 29 March 2026. 01:30 UTC that day is 02:30 local."""
    s = _utc("2026-03-29T01:30:00Z")
    assert str(local_day(s)[0]) == "2026-03-29"


# ---------------------------------------------------------------------------
# Shape and robustness
# ---------------------------------------------------------------------------
def test_accepts_naive_strings_by_treating_them_as_utc():
    s = pd.Series(["2026-07-01 23:30:00"])
    assert str(local_day(s)[0]) == "2026-07-02"


def test_unparseable_values_become_nat_not_an_exception():
    s = pd.Series(["not a date", "2026-07-01T23:30:00Z"])
    days = local_day(s)
    assert pd.isna(days[0])
    assert str(days[1]) == "2026-07-02"


def test_empty_series_round_trips():
    s = pd.Series([], dtype="datetime64[ns, UTC]")
    assert len(local_day(s)) == 0
    assert len(local_month(s)) == 0


def test_local_month_accepts_other_frequencies():
    s = _utc("2026-07-31T23:30:00Z")
    assert str(local_month(s, freq="Q")[0]) == "2026Q3"


def test_no_timezone_warning_is_emitted():
    """The pandas "will drop timezone information" warning was the symptom.
    Converting to local before dropping the tz removes it honestly, rather than
    silencing a message that was pointing at a real bucketing bug."""
    import warnings

    s = _utc("2026-07-31T23:30:00Z")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        local_month(s)   # must not raise
        local_day(s)


def test_accepts_an_index_not_just_a_series():
    idx = pd.to_datetime(["2026-07-01T23:30:00Z"], utc=True)
    assert str(local_day(idx)[0]) == "2026-07-02"
