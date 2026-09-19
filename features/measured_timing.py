"""Measured race timing mined from the raw Sporting Life archive (step 11).

The existing ``horse_speed`` (``features/engine.py::add_speed_figures``) is a
**finishing-position percentile proxy**, not a measured speed. Its meaning is
preserved untouched. This module adds a separately named *measured* family,
sourced only from figures the archive genuinely publishes.

What the archive actually carries (measured, not assumed -- see
``reports/improvement/11/00_probe_archive.txt``):

===========================  =================================================
``race_summary.winning_time``  ``"5m 14.84s"`` / ``"58.31s"``. Present on
                               98.50% of sampled race documents.
``race_summary.distance``      ``"2m 4f 56y"``. Present on 100%.
``ride.finish_distance``       Margin behind the horse **in front**, i.e.
                               INCREMENTAL. Established empirically, not
                               assumed: in a 40-day sample 373 races contain a
                               decrease somewhere down the finishing order,
                               which is impossible under a cumulative reading,
                               against 34 merely cumulative-compatible.
                               Vocabulary is numeric with ``¼ ½ ¾`` fractions
                               plus ``nse``/``sh``/``hd``/``nk``/``dh``.
``course_surface.surface``     ``TURF`` / ``ALLWEATHER`` / ``POLYTRACK``.
===========================  =================================================

**Sectionals do not exist here.** A recursive key scan over the sampled
documents found no ``sectional``/``split_time``/``furlong_time`` field of any
kind. Sectional features are therefore not buildable from this archive and are
recorded as a data gap rather than synthesised.

Contract
--------
units
    Distance in **yards**, time in **seconds**, margins in **lengths**, speed
    in **yards per second**. Every emitted column carries its unit in its name.
non-finishers
    ``ride_status != "RUNNER"`` is excluded (a non-runner has no performance).
    A runner with no finishing position -- Sporting Life stamps 0, and every
    such RUNNER carries a ``casualty`` reason -- ran but did not complete: it
    gets ``finished=False`` and a **null** time/speed. It is never imputed and
    never treated as a slow finisher.
missingness
    A race with no parseable ``winning_time`` or ``distance`` emits no runner
    timing rows at all. Null is propagated, never filled.
publication timestamp
    ``available_from`` is the race's own local off time: a result is observable
    only once the race has run. Downstream point-in-time machinery must cut on
    this instant (D39), exactly as it does for ``position``. Documents for a
    future-dated card carry no ``winning_time`` and no finishing positions and
    so contribute nothing -- verified by :func:`extract_timing_window`.
per-runner time
    Only the **winner's** time is published. A beaten runner's time is
    ``winning_time + cumulative_lengths / lengths_per_second``, where
    ``lengths_per_second`` is a *declared parameter*
    (:data:`DEFAULT_LENGTHS_PER_SECOND`), not a measured quantity. Columns
    derived through it are named ``*_est_*`` so no caller can mistake an
    estimate for a published figure. ``beaten_lengths`` itself is exact.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from utils.logger import get_logger
from utils.text_norm import minute_key, norm_horse, norm_venue

logger = get_logger(__name__)

DEFAULT_RAW_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "historical", "raw")
)
DEFAULT_SOURCE = "sporting_life"

_RACING_TZ = ZoneInfo("Europe/Dublin")
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)

YARDS_PER_MILE = 1760.0
YARDS_PER_FURLONG = 220.0

#: Lengths-equivalent of the archive's named short margins. These are the
#: standard industry readings of the tokens; they are a *convention*, not a
#: measurement, and they only ever affect sub-length gaps.
NAMED_MARGINS = {
    "dh": 0.0,     # dead heat: no separation
    "dht": 0.0,
    "nse": 0.05,   # nose
    "sh": 0.1,     # short head
    "shd": 0.1,
    "hd": 0.2,     # head
    "nk": 0.3,     # neck
    "snk": 0.2,    # short neck
}

#: Declared lengths-per-second used to turn an exact beaten-lengths margin into
#: an *estimated* finishing time. Not fitted -- the archive publishes only the
#: winner's time, so there is no within-race time spread to fit against. Any
#: feature built through it must be sensitivity-checked against plausible
#: alternatives (see ``reports/improvement/11/``).
DEFAULT_LENGTHS_PER_SECOND = 5.0

_FRACTIONS = {"¼": 0.25, "½": 0.5, "¾": 0.75,
              "⅓": 1 / 3, "⅔": 2 / 3}

#: Race type, read off the race title. A 2m hurdle and a 2m flat race are run at
#: completely different speeds, so a speed par that ignores this compares horses
#: against the wrong standard. Ordered most specific first.
_RACE_TYPE_PATTERNS = (
    ("chase", re.compile(r"\bchase\b|\bsteeple", re.I)),
    ("hurdle", re.compile(r"\bhurdle", re.I)),
    ("nhf", re.compile(r"national hunt flat|\bnhf\b|\bbumper\b|\bstandard open\b", re.I)),
)


def classify_race_type(name: Any) -> str:
    """Race title -> ``chase`` / ``hurdle`` / ``nhf`` / ``flat``."""
    text = str(name or "")
    for label, pattern in _RACE_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "flat"

# Sanity envelope for a published winning time, as yards/second. A thoroughbred
# racing pace is ~15-20 yd/s; anything outside this is a parse or data fault and
# is dropped rather than fed forward.
MIN_SPEED_YPS = 8.0
MAX_SPEED_YPS = 25.0


# ──────────────────────────────── unit parsers ───────────────────────────────
def parse_distance_yards(text: Any) -> Optional[float]:
    """``"2m 4f 56y"`` -> yards. Returns ``None`` if nothing parses."""
    s = str(text or "").strip().lower()
    if not s:
        return None
    total = 0.0
    seen = False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*(m|f|y)", s):
        v = float(value)
        if unit == "m":
            total += v * YARDS_PER_MILE
        elif unit == "f":
            total += v * YARDS_PER_FURLONG
        else:
            total += v
        seen = True
    return total if seen and total > 0 else None


def parse_time_seconds(text: Any) -> Optional[float]:
    """``"5m 14.84s"`` / ``"58.31s"`` / ``"1m 0.00s"`` -> seconds."""
    s = str(text or "").strip().lower()
    if not s:
        return None
    mins = re.search(r"(\d+(?:\.\d+)?)\s*m", s)
    secs = re.search(r"(\d+(?:\.\d+)?)\s*s", s)
    if mins is None and secs is None:
        return None
    total = 0.0
    if mins:
        total += float(mins.group(1)) * 60.0
    if secs:
        total += float(secs.group(1))
    return total if total > 0 else None


def parse_margin_lengths(text: Any) -> Optional[float]:
    """A ``finish_distance`` token -> lengths behind the horse in front.

    Handles ``"4 ¾"``, ``"1/2"``, ``"nk"``, ``"dh"``. Unknown tokens return
    ``None`` so the runner's margin stays missing instead of becoming zero.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    if s in NAMED_MARGINS:
        return NAMED_MARGINS[s]
    total = 0.0
    seen = False
    for ch, value in _FRACTIONS.items():
        if ch in s:
            total += value
            seen = True
            s = s.replace(ch, " ")
    frac = re.search(r"(\d+)\s*/\s*(\d+)", s)
    if frac:
        total += float(frac.group(1)) / float(frac.group(2))
        seen = True
        s = s[: frac.start()] + " " + s[frac.end():]
    whole = re.search(r"\d+(?:\.\d+)?", s)
    if whole:
        total += float(whole.group(0))
        seen = True
    return total if seen else None


# ───────────────────────────────── extraction ────────────────────────────────
@dataclass(frozen=True)
class RaceTiming:
    """One archived race's measured timing, plus its per-runner rows."""

    race_key: str
    race_date: str
    off_time: str
    venue: str
    distance_yards: Optional[float]
    winning_time_seconds: Optional[float]
    surface: str
    going: str
    race_class: str
    handicap: bool
    source_path: str
    race_name: str = ""
    race_type: str = "flat"
    runners: tuple[dict, ...] = ()

    @property
    def usable(self) -> bool:
        return bool(
            self.distance_yards
            and self.winning_time_seconds
            and MIN_SPEED_YPS <= self.distance_yards / self.winning_time_seconds <= MAX_SPEED_YPS
        )


def _off_datetime(summary: dict) -> str:
    """``race_summary`` -> local ``YYYY-MM-DDTHH:MM``.

    Sporting Life stores a naive clock in UTC while every race key in this repo
    is local racing time; this mirrors ``execution.race_facts._off_datetime``
    exactly so the two archives produce the same join key.
    """
    day = str(summary.get("date") or "").strip()
    clock = str(summary.get("time") or "").strip()
    if not clock:
        clock = str(summary.get("off_time") or "").strip()[:5]
    if not day:
        return ""
    if not clock:
        return day
    try:
        utc = datetime.strptime(f"{day}T{clock[:5]}", "%Y-%m-%dT%H:%M")
    except ValueError:
        return f"{day}T{clock[:5]}"
    return utc.replace(tzinfo=timezone.utc).astimezone(_RACING_TZ).strftime("%Y-%m-%dT%H:%M")


def parse_timing_document(payload: Any, *, source_path: str = "") -> Optional[RaceTiming]:
    """Decoded ``__NEXT_DATA__`` -> :class:`RaceTiming`, or ``None``.

    Returns ``None`` for index/racecard pages. A *future* card parses to a race
    with no winning time, which :attr:`RaceTiming.usable` rejects.
    """
    if not isinstance(payload, dict):
        return None
    race = (((payload.get("props") or {}).get("pageProps") or {}).get("race")) or {}
    if not isinstance(race, dict) or not race:
        return None
    summary = race.get("race_summary") or {}
    if not isinstance(summary, dict) or not summary:
        return None

    venue = str(summary.get("course_name") or "").strip()
    off = _off_datetime(summary)
    if not venue or not off:
        return None

    rides = race.get("rides") or []
    if not isinstance(rides, list):
        rides = []

    # Finishing order, then a running sum of the INCREMENTAL margins.
    ordered: list[tuple[int, dict]] = []
    others: list[dict] = []
    for ride in rides:
        if not isinstance(ride, dict):
            continue
        pos = ride.get("finish_position")
        pos = int(pos) if isinstance(pos, (int, float)) and int(pos) > 0 else None
        (ordered.append((pos, ride)) if pos is not None else others.append(ride))
    ordered.sort(key=lambda p: p[0])

    runners: list[dict] = []
    cumulative = 0.0
    cumulative_known = True
    for pos, ride in ordered:
        status = str(ride.get("ride_status") or "").strip().upper()
        if status and status != "RUNNER":
            continue
        margin = parse_margin_lengths(ride.get("finish_distance"))
        if pos == 1:
            margin = 0.0
        if margin is None:
            cumulative_known = False
        elif cumulative_known:
            cumulative += margin
        runners.append(_runner_row(ride, pos, True,
                                   cumulative if cumulative_known else None, margin))
    for ride in others:
        status = str(ride.get("ride_status") or "").strip().upper()
        if status and status != "RUNNER":
            continue  # non-runner: no performance at all
        # Ran but did not complete: null time, never imputed.
        runners.append(_runner_row(ride, None, False, None, None))

    surface = str(((summary.get("course_surface") or {}) or {}).get("surface") or "").strip()
    return RaceTiming(
        race_key=f"{norm_venue(venue)}|{minute_key(off)}",
        race_date=off[:10],
        off_time=off,
        venue=venue,
        distance_yards=parse_distance_yards(summary.get("distance")),
        winning_time_seconds=parse_time_seconds(summary.get("winning_time")),
        surface=surface,
        going=str(summary.get("going") or "").strip(),
        race_class=str(summary.get("race_class") or "").strip(),
        handicap=bool(summary.get("has_handicap")),
        source_path=source_path,
        race_name=str(summary.get("name") or "").strip(),
        race_type=classify_race_type(summary.get("name")),
        runners=tuple(runners),
    )


def _runner_row(ride: dict, position, finished: bool, cumulative, margin) -> dict:
    horse = ride.get("horse") if isinstance(ride.get("horse"), dict) else {}
    name = str((horse or {}).get("name") or ride.get("horse_name") or "").strip()
    casualty = ride.get("casualty") if isinstance(ride.get("casualty"), dict) else {}
    return {
        "horse_key": norm_horse(name),
        "horse_name": name,
        "finish_position": position,
        "finished": finished,
        "beaten_lengths": cumulative,
        "margin_lengths": margin,
        "casualty_reason": str((casualty or {}).get("reason") or "").strip() or None,
    }


def parse_timing_file(path: str) -> Optional[RaceTiming]:
    """Decompress, extract ``__NEXT_DATA__``, parse. Never raises."""
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            html = fh.read()
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        logger.debug("measured_timing: unreadable %s (%s)", path, exc)
        return None
    match = _NEXT_DATA_RE.search(html)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except ValueError:
        return None
    try:
        return parse_timing_document(payload, source_path=path)
    except (TypeError, ValueError, AttributeError) as exc:  # pragma: no cover
        logger.debug("measured_timing: unparseable %s (%s)", path, exc)
        return None


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def iter_day_paths(day: Any, *, root: str = DEFAULT_RAW_ROOT,
                   source: str = DEFAULT_SOURCE) -> list[str]:
    d = _as_date(day)
    folder = os.path.join(root, source, f"{d:%Y}", f"{d:%Y-%m-%d}")
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if n.endswith(".html") or n.endswith(".html.gz"))


def iter_window_days(start: Any, end: Any) -> Iterator[date]:
    day, last = _as_date(start), _as_date(end)
    while day <= last:
        yield day
        day += timedelta(days=1)


def extract_timing_window(start: Any, end: Any, *, root: str = DEFAULT_RAW_ROOT,
                          source: str = DEFAULT_SOURCE) -> list[RaceTiming]:
    """Parse an inclusive day window, keeping the most complete duplicate."""
    best: dict[str, RaceTiming] = {}
    for day in iter_window_days(start, end):
        for path in iter_day_paths(day, root=root, source=source):
            timing = parse_timing_file(path)
            if timing is None:
                continue
            prior = best.get(timing.race_key)
            if prior is None or _completeness(timing) > _completeness(prior):
                best[timing.race_key] = timing
    return list(best.values())


def _completeness(timing: RaceTiming) -> tuple:
    return (bool(timing.usable), sum(1 for r in timing.runners if r["finished"]),
            len(timing.runners))


def to_runner_frame(timings: Iterable[RaceTiming], *,
                    lengths_per_second: float = DEFAULT_LENGTHS_PER_SECOND) -> pd.DataFrame:
    """Flatten to one row per (race, runner) with measured-performance columns.

    Only races passing :attr:`RaceTiming.usable` contribute. ``*_est_*`` columns
    are derived through ``lengths_per_second`` and are estimates, not published
    figures; ``beaten_lengths`` and ``winning_time_seconds`` are exact.
    """
    rows: list[dict] = []
    for t in timings:
        if not t.usable:
            continue
        for r in t.runners:
            beaten = r["beaten_lengths"] if r["finished"] else None
            est_time = (t.winning_time_seconds + beaten / lengths_per_second
                        if beaten is not None else None)
            rows.append({
                "race_key": t.race_key,
                "race_date": t.race_date,
                "off_time": t.off_time,
                "available_from": t.off_time,
                "venue": t.venue,
                "surface": t.surface,
                "going": t.going,
                "race_class": t.race_class,
                "race_type": t.race_type,
                "handicap": t.handicap,
                "distance_yards": t.distance_yards,
                "winning_time_seconds": t.winning_time_seconds,
                "winner_speed_yps": t.distance_yards / t.winning_time_seconds,
                "horse_key": r["horse_key"],
                "horse_name": r["horse_name"],
                "finish_position": r["finish_position"],
                "finished": r["finished"],
                "beaten_lengths": beaten,
                "est_time_seconds": est_time,
                "est_speed_yps": (t.distance_yards / est_time) if est_time else None,
                "casualty_reason": r["casualty_reason"],
                "lengths_per_second": lengths_per_second,
                "source": DEFAULT_SOURCE,
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.astype({"distance_yards": "float64", "winning_time_seconds": "float64",
                         "winner_speed_yps": "float64", "beaten_lengths": "float64",
                         "est_time_seconds": "float64", "est_speed_yps": "float64"})
