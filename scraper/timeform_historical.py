"""Timeform historical fetcher — PUBLIC orchestrator.

Scrapes racecards (ratings/pace/class/going/form) and writes a year-partitioned
parquet that fuses with betsp.parquet. Positions/results are sourced separately;
trailing win/place rates compute when a `position` column is present, else stay null.
"""
import re
from typing import Callable, Optional

import pandas as pd

from scraper.timeform import features, pages, writer
from scraper.timeform.client import TimeformClient
from scraper.timeform.extractor import StubExtractor
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.text_norm import norm_venue
from utils.timezone import now as _now

logger = get_logger(__name__)

# /racecards/{course}/{YYYY-MM-DD}/{HHMM}/...
_URL_RE = re.compile(r"/racecards/([^/]+)/(\d{4}-\d{2}-\d{2})/(\d{3,4})/")


def _load_cfg() -> dict:
    return get_config().get("timeform", {})


def _resolve_days(cfg: dict, days: Optional[list]) -> list:
    # Timeform serves racecards for TODAY only at the index path; dated index URLs 500.
    # Historical backfill (past races) comes from the separate results/archive effort,
    # so the default scope here is today. Callers may still pass explicit days.
    if days:
        return days
    return [_now().date()]


def _parse_url(url: str) -> tuple:
    """(venue, race_date_str, race_time_iso) from a racecard URL."""
    m = _URL_RE.search(url)
    if not m:
        return url.split("/racecards/")[-1].split("/")[0], None, None
    venue, day, hhmm = m.group(1), m.group(2), m.group(3)
    hhmm = hhmm.zfill(4)
    return venue, day, f"{day}T{hhmm[:2]}:{hhmm[2:]}"


def fetch(days: Optional[list] = None, get_html: Optional[Callable[[str], str]] = None,
          parquet_path: Optional[str] = None, max_events: Optional[int] = None) -> pd.DataFrame:
    cfg = _load_cfg()
    parquet_path = parquet_path or cfg.get("parquet_path", "data/historical/timeform.parquet")
    max_events = cfg.get("max_events_per_run", 0) if max_events is None else max_events
    if get_html is None:
        get_html = TimeformClient(cfg).get_html

    days = _resolve_days(cfg, days)
    today = _now().date()
    extractor = StubExtractor()
    records = []
    event_count = 0
    for d in days:
        # Today's cards live at the bare index; dated index URLs 500 on Timeform.
        index_target = None if d == today else d
        try:
            index_html = get_html(pages.index_url(index_target))
        except Exception as exc:  # noqa: BLE001
            logger.warning("timeform index fetch failed for %s: %s", d, exc)
            continue
        goings = pages.meeting_goings(index_html)
        for url in pages.event_links(index_html):
            if max_events and event_count >= max_events:
                break
            event_count += 1
            venue, day_str, race_time = _parse_url(url)
            going = goings.get(norm_venue(venue))
            try:
                html = get_html(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("timeform event failed %s: %s", url, exc)
                continue
            for row in extractor.extract(html, venue=venue, going=going,
                                         race_time=race_time):
                rec = row.__dict__.copy()
                rec["race_date"] = pd.Timestamp(day_str or d, tz="UTC")
                records.append(rec)

    if not records:
        logger.warning("timeform: no rows scraped")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["source"] = "timeform"
    df["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")

    df = features.add_going_speed(df, cfg.get("going_speed_map", {}))
    df = features.add_class_change(df)
    if "position" in df.columns and df["position"].notna().any():
        df = features.add_trailing_rates(
            df, entity="horse_name", win_col="historical_win_rate",
            place_col="historical_place_rate",
            lookback_months=int(cfg.get("lookback_months", 12)),
            lookback_runs=int(cfg.get("lookback_runs", 20)),
            place_positions=int(cfg.get("place_positions", 3)))
    else:
        for col in ("historical_win_rate", "historical_place_rate", "jockey_win_rate"):
            df[col] = pd.NA
        df["runs_in_window"] = 0

    writer.write(df, path=parquet_path)
    return df
