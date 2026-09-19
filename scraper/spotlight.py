"""Sporting Life racecard *spotlight commentary* scraper.

Provides the free-text feed that ``llm/text_features.py`` needs: each runner on a
Sporting Life racecard carries a prose "spotlight" comment (the ``commentary``
field in the page's ``__NEXT_DATA__``), e.g.

    "240,000 gns breezer who justified market support with ease on her 5f
     Lingfield debut (good to firm) 18 days ago. Looks sure to progress..."

This is exactly the angle/excuse-bearing text the LLM (or regex) backend turns
into boolean features (ground_excuse, trip_trouble, headgear_first_time, ...).

Two-stage fetch, mirroring ``scraper.betsp.results.sporting_life`` (same site,
same ``__NEXT_DATA__`` shape, same proxy/curl_cffi/rate-limiter path):
  1. index  ``/racing/racecards/{YYYY-MM-DD}``  -> UK/IRE meetings + race ids,
  2. detail ``/racing/racecards/{ymd}/{course}/racecard/{rid}/{slug}`` -> rides[].

Output is persisted to ``data/spotlight.parquet`` keyed so it can later be joined
onto the unified runners by (race_date, venue, horse_name).

    python -m scraper.spotlight              # scrape today's UK/IRE cards
    python -m scraper.spotlight --date 2026-06-17 --max-races 5
"""
from __future__ import annotations

import argparse
import html as _html
import os
import re
from datetime import date, datetime
from typing import Callable, Optional

import pandas as pd

from scraper.betsp.results.sporting_life import (
    _next_data, _local_minute, _ref_id, _course_slug, _UK_IRE,
)
from utils.logger import get_logger
from utils.timezone import now as _now

_ARCHIVE_SOURCE = "sporting_life_spotlight"

logger = get_logger(__name__)

_INDEX = "https://www.sportinglife.com/racing/racecards/{ymd}"
_BASE = "https://www.sportinglife.com"
_DEFAULT_PARQUET = "data/spotlight.parquet"
# rid -> detail href, pulled straight from the index HTML (robust to slug rules).
_HREF_RE = re.compile(r'href="(/racing/racecards/[^"]*?/racecard/(\d+)/[^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")

_COLUMNS = ["race_date", "race_time", "venue", "horse_name", "horse_id",
            "cloth_number", "commentary", "race_verdict", "source", "fetched_at"]


def _clean(text: Optional[str]) -> str:
    """Strip HTML tags + unescape entities + collapse whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", _html.unescape(_TAG_RE.sub(" ", str(text)))).strip()


def _default_get_html(url: str) -> str:
    # Reuse the results fetcher: curl_cffi Chrome impersonation + proxy rotation
    # + the central per-domain rate limiter (www.sportinglife.com is configured).
    from scraper.betsp_historical import _results_get_html
    return _results_get_html(url)


def _uk_ire_race_ids(index_data: dict) -> dict[str, dict]:
    """{race_id: {venue, time, going, name}} for UK/IRE meetings only."""
    out: dict[str, dict] = {}
    meetings = (index_data.get("props", {}).get("pageProps", {}).get("meetings") or [])
    for meeting in meetings:
        ms = meeting.get("meeting_summary") or {}
        country = (((ms.get("course") or {}).get("country") or {}).get("short_name") or "")
        if str(country).upper() not in _UK_IRE:
            continue
        for race in meeting.get("races") or []:
            rid = (race.get("race_summary_reference") or {}).get("id")
            if rid is None:
                continue
            out[str(rid)] = {
                "venue": race.get("course_name") or "",
                "time": race.get("time"),
                "going": race.get("going"),
                "name": race.get("name") or "",
                "verdict": _clean(race.get("verdict")),
            }
    return out


def _parse_detail(html: str, day: date, fallback: dict) -> list[dict]:
    data = _next_data(html)
    if not data:
        return []
    race = data.get("props", {}).get("pageProps", {}).get("race")
    if not isinstance(race, dict):
        return []
    summary = race.get("race_summary") or {}
    venue = summary.get("course_name") or fallback.get("venue") or ""
    rtime = summary.get("time") or fallback.get("time")
    when = _local_minute(summary.get("date") or day.isoformat(), rtime)
    verdict = fallback.get("verdict", "")
    rows = []
    for ride in race.get("rides") or []:
        horse = ride.get("horse") or {}
        name = horse.get("name")
        commentary = _clean(ride.get("commentary"))
        if not name or not commentary:
            continue
        rows.append({
            "race_date": when,
            "race_time": rtime,
            "venue": venue,
            "horse_name": name,
            "horse_id": _ref_id(horse, "horse_reference") or None,
            "cloth_number": ride.get("cloth_number"),
            "commentary": commentary,
            "race_verdict": verdict,
            "source": "sporting_life",
        })
    return rows


def _archive_rows(df: pd.DataFrame, archive_store=None) -> int:
    """Best-effort append of every scraped comment into the immutable text archive.

    Separate from the ``data/spotlight.parquet`` mutable "latest view" cache:
    ``_persist`` below still dedupes to the newest comment per runner for quick
    reads, but the archive keeps every observation, including a later edit of
    the same runner's comment, retrievable by its own content hash.
    A single bad row never aborts the batch.
    """
    from execution.race_facts import race_facts_key
    from llm.text_archive import get_text_archive

    archive = archive_store if archive_store is not None else get_text_archive()
    n = 0
    for row in df.itertuples(index=False):
        try:
            if not str(row.venue or "").strip() or not row.commentary:
                continue
            race_uid = race_facts_key(row.venue, row.race_date)
            if not race_uid:
                continue
            archive.record(
                text=row.commentary,
                source=_ARCHIVE_SOURCE,
                race_uid=race_uid,
                horse_name=row.horse_name,
                horse_id=row.horse_id,
                text_kind="spotlight",
                fetched_at=row.fetched_at,
                published_at=None,  # Sporting Life supplies no comment publish time
                metadata={"cloth_number": row.cloth_number, "race_verdict": row.race_verdict},
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("spotlight: archive failed for %s/%s: %s", row.venue, row.horse_name, exc)
    return n


def scrape(day: Optional[date] = None, *, max_races: int = 0,
           parquet_path: str = _DEFAULT_PARQUET,
           get_html: Optional[Callable[[str], str]] = None,
           write: bool = True, archive: bool = True, archive_store=None) -> pd.DataFrame:
    """Scrape UK/IRE spotlight commentary for ``day`` (default today); persist parquet.

    Each row is one runner's spotlight comment, keyed (race_date, venue, horse_name)
    so it can be joined onto the unified runners later. Best-effort: a failed race
    is logged and skipped, never fatal. When ``archive`` is true (default), every
    comment is also appended to the immutable ``llm.text_archive`` store, which
    keeps every observed version rather than only the newest.
    """
    day = day or _now().date()
    get_html = get_html or _default_get_html
    ymd = day.strftime("%Y-%m-%d")

    try:
        index_html = get_html(_INDEX.format(ymd=ymd))
    except Exception as exc:  # noqa: BLE001
        logger.warning("spotlight: index fetch failed for %s: %s", ymd, exc)
        return pd.DataFrame(columns=_COLUMNS)

    index_data = _next_data(index_html)
    if not index_data:
        logger.warning("spotlight: no __NEXT_DATA__ on index %s", ymd)
        return pd.DataFrame(columns=_COLUMNS)

    races = _uk_ire_race_ids(index_data)
    href_by_rid = {rid: href for href, rid in _HREF_RE.findall(index_html)}

    rows: list[dict] = []
    n_done = 0
    for rid, meta in races.items():
        href = href_by_rid.get(rid)
        if not href:  # fall back to the documented detail pattern
            href = (f"/racing/racecards/{ymd}/{_course_slug(meta['venue'])}"
                    f"/racecard/{rid}/{_course_slug(meta['name'])}")
        try:
            detail_html = get_html(_BASE + href)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spotlight: detail fetch failed %s: %s", href, exc)
            continue
        race_rows = _parse_detail(detail_html, day, meta)
        rows.extend(race_rows)
        n_done += 1
        if max_races and n_done >= max_races:
            break

    df = pd.DataFrame(rows, columns=_COLUMNS)
    if not df.empty:
        df["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
    logger.info("spotlight: %d comments across %d UK/IRE races (%s)",
                len(df), n_done, ymd)

    if archive and not df.empty:
        n_archived = _archive_rows(df, archive_store=archive_store)
        logger.info("spotlight: archived %d/%d comments", n_archived, len(df))
    if write and not df.empty:
        _persist(df, parquet_path)
    return df


def _persist(df: pd.DataFrame, parquet_path: str) -> None:
    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    if os.path.exists(parquet_path):
        try:
            old = pd.read_parquet(parquet_path)
            df = pd.concat([old, df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spotlight: could not merge existing parquet (%s)", exc)
    df = df.drop_duplicates(subset=["race_date", "venue", "horse_name"], keep="last")
    df.to_parquet(parquet_path, index=False)
    logger.info("spotlight: wrote %d rows -> %s", len(df), parquet_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", help="YYYY-MM-DD (default: today)")
    ap.add_argument("--max-races", type=int, default=0, help="cap races (0 = all UK/IRE)")
    ap.add_argument("--parquet", default=_DEFAULT_PARQUET)
    args = ap.parse_args()
    day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else None
    df = scrape(day, max_races=args.max_races, parquet_path=args.parquet)
    print(f"spotlight: scraped {len(df)} runner comments")
    if not df.empty:
        ex = df.iloc[0]
        print(f"  e.g. {ex['venue']} — {ex['horse_name']}: {ex['commentary'][:160]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
