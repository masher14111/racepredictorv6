"""Sporting Life results source — full-field finishing positions from clean JSON.

Sporting Life is a Next.js app: every page embeds a ``__NEXT_DATA__`` script tag
holding the typed page state as JSON. The daily *index* lists meetings/races but
only the top finishers per race; the *per-race* detail page carries the full field
(every runner's finish_position, jockey, trainer) plus race-level course/time/going.

So enrichment is two-stage:
  1. fetch the index for the day -> enumerate UK/IRE race ids,
  2. fetch each race-detail page -> parse race.rides[] into ResultRow per runner.

Parsing is from typed JSON fields, not HTML class names, so it does not break when
the site reskins its markup.
"""
import json
import re
from datetime import date, datetime
from typing import Optional

import pytz

from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from utils.logger import get_logger
from utils.timezone import to_local

logger = get_logger(__name__)

_RESULTS_INDEX = "https://www.sportinglife.com/racing/results/{ymd}"
# Detail URL: the date segment is authoritative; the course-slug and trailing
# name-slug segments are ignored by the server (any non-empty token works), so we
# pass a readable course slug for the first and a constant "r" for the last.
_RESULTS_DETAIL = "https://www.sportinglife.com/racing/results/{ymd}/{course}/{rid}/r"

_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)
# Sporting Life country codes for the UK & Ireland (everything else — FR/USA/UAE/
# SAF/IND — is dropped so we don't spend requests/bandwidth on races the Betfair
# SP backbone (uk+ire only) can never join against). NB: Ireland is "Eire" here.
_UK_IRE = {"ENG", "SCO", "WAL", "EIRE", "IRE", "NIR"}


def _next_data(html: str) -> Optional[dict]:
    m = _NEXT_DATA_RE.search(html or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except (ValueError, TypeError):
        return None


def _course_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name or "").lower()).strip("-") or "r"


def _ref_id(node: Optional[dict], *keys) -> Optional[str]:
    """Pull a stable id from a *_reference sub-object, returned as a string."""
    if not isinstance(node, dict):
        return None
    for k in keys:
        ref = node.get(k)
        if isinstance(ref, dict) and ref.get("id") is not None:
            return str(ref["id"])
    return None


def _to_position(value) -> Optional[int]:
    """finish_position -> positive int, or None for non-finishers (PU/F/non-runner)."""
    try:
        pos = int(value)
    except (ValueError, TypeError):
        return None
    return pos if pos > 0 else None


def _local_minute(rdate: str, rtime: Optional[str]) -> str:
    """Build the join key 'YYYY-MM-DDTHH:MM' in the project's local timezone.

    Sporting Life's race time/off_time are in UTC, but the Betfair SP backbone
    (the join spine) carries true LOCAL wall-clock times. They coincide only in
    winter (GMT); under BST the SL time trails local by an hour, which silently
    zeroed the join for every Apr-Sep race. Convert UTC -> local so the
    minute-key matches year-round.
    """
    if not rtime:
        return rdate
    try:
        naive = datetime.strptime(f"{rdate} {rtime}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return f"{rdate}T{rtime}"
    local = to_local(pytz.utc.localize(naive))
    return local.strftime("%Y-%m-%dT%H:%M")


class SportingLife(ResultsSource):
    name = "sporting_life"

    def fetch_raw(self, day: date, get_html) -> list:
        ymd = day.strftime("%Y-%m-%d")
        index_url = _RESULTS_INDEX.format(ymd=ymd)
        try:
            index_html = get_html(index_url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("SportingLife index fetch failed %s: %s", index_url, exc)
            return []

        data = _next_data(index_html)
        if not data:
            logger.warning("SportingLife: no __NEXT_DATA__ on index %s", index_url)
            return []

        meetings = (data.get("props", {}).get("pageProps", {}).get("meetings") or [])
        raws = []
        for meeting in meetings:
            for race in meeting.get("races", []):
                if str(race.get("country_short_name", "")).upper() not in _UK_IRE:
                    continue
                rid = (race.get("race_summary_reference") or {}).get("id")
                if rid is None:
                    continue
                course = race.get("course_name") or ""
                url = _RESULTS_DETAIL.format(
                    ymd=ymd, course=_course_slug(course), rid=rid
                )
                try:
                    detail_html = get_html(url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("SportingLife detail fetch failed %s: %s", url, exc)
                    continue
                raws.append(RawResult(
                    source=self.name, race_date=day, url=url,
                    html=detail_html, venue=course,
                ))
        return raws

    def parse(self, raw: RawResult) -> list:
        data = _next_data(raw.html)
        if not data:
            return []
        race = data.get("props", {}).get("pageProps", {}).get("race")
        if not isinstance(race, dict):
            return []

        summary = race.get("race_summary") or {}
        venue = summary.get("course_name") or raw.venue or ""
        going = summary.get("going") or None
        # Join key needs an exact-minute ISO timestamp in LOCAL time (the backbone
        # is local wall-clock; SL's time is UTC) — see _local_minute.
        rdate = summary.get("date") or raw.race_date.isoformat()
        when = _local_minute(rdate, summary.get("time"))  # "16:30" UTC -> local

        rows = []
        for ride in race.get("rides", []):
            horse = ride.get("horse") or {}
            horse_name = horse.get("name")
            if not horse_name:
                continue
            rows.append(ResultRow(
                race_date=when,
                venue=venue,
                horse_name=horse_name,
                source=self.name,
                position=_to_position(ride.get("finish_position")),
                going=going,
                jockey_id=_ref_id(ride.get("jockey"), "person_reference"),
                jockey_name=(ride.get("jockey") or {}).get("name") or None,
                trainer_id=_ref_id(ride.get("trainer"),
                                   "business_reference", "person_reference"),
                trainer_name=(ride.get("trainer") or {}).get("name") or None,
            ))
        return rows
