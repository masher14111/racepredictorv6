from datetime import datetime
import pytz

from utils.config_loader import get_config

TZ = pytz.timezone(get_config().get("timezone", "Europe/Dublin"))


def now() -> datetime:
    """Return current time in the configured timezone."""
    return datetime.now(tz=TZ)


def to_local(dt: datetime) -> datetime:
    """Convert a timezone-aware datetime to the configured local timezone."""
    if dt.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    return dt.astimezone(TZ)


def to_utc(dt: datetime) -> datetime:
    """Convert a local datetime (naive or aware) to UTC."""
    if dt.tzinfo is None:
        dt = TZ.localize(dt)
    return dt.astimezone(pytz.utc)


# ---------------------------------------------------------------------------
# Pandas helpers — bucketing timestamps into local days and months
#
# Timestamps are parsed with utc=True throughout, so `.dt.date` and
# `.dt.to_period("M")` bucket by UTC, not by Dublin. That is wrong for a P&L
# view: through BST/IST (late March to late October) anything settled between
# 00:00 and 00:59 local falls into the previous UTC day, and on the 1st of a
# month into the previous month. Irish evening cards run to 21:00+, so bets near
# midnight are not hypothetical. These helpers convert to local first.
#
# They also silence pandas' "Converting to PeriodArray/Index representation will
# drop timezone information" warning honestly — by making the wall clock local
# before the timezone is dropped, rather than suppressing the message.
# ---------------------------------------------------------------------------
def _as_local(series):
    """A tz-aware datetime Series converted to the configured timezone.

    Accepts anything pandas can parse — Series, Index or plain list. Callers pass
    DataFrame columns, but an Index has no ``.dt`` accessor, so normalise to a
    Series first rather than failing on the caller's shape.
    """
    import pandas as pd

    parsed = pd.to_datetime(series, utc=True, errors="coerce")
    if not isinstance(parsed, pd.Series):
        parsed = pd.Series(parsed)
    return parsed.dt.tz_convert(TZ)


def local_day(series):
    """Calendar dates in the configured timezone (not UTC)."""
    return _as_local(series).dt.date


def local_month(series, freq: str = "M"):
    """Period labels (default 'YYYY-MM') in the configured timezone (not UTC)."""
    # tz is dropped only AFTER the wall clock is local, so the period is right.
    return _as_local(series).dt.tz_localize(None).dt.to_period(freq)
