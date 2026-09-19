"""Betfair SP daily price file fetcher and parser (the betSP spine)."""
import io
import re
import time as _time
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from utils.logger import get_logger
from utils.proxy_manager import get_proxy_manager
from utils.rate_limiter import get_rate_limiter
from utils.timezone import TZ

logger = get_logger(__name__)

_BASE = "https://promo.betfair.com/betfairsp/prices"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    )
}
# 1m, 1m2f, 5f, 2m4f etc. — first distance token in the event name.
_DISTANCE_RE = re.compile(r"\b(\d+m(?:\d+f)?|\d+f)\b", re.IGNORECASE)

# Backbone columns produced by this module (subset of the final schema).
BACKBONE_COLUMNS = [
    "race_date", "venue", "horse_id", "horse_name", "odds_finish",
    "win_lose", "distance", "market_type", "region", "morningwap", "ppwap",
]


def file_url(region: str, market: str, year: int, month: int, day: int) -> str:
    """Build a daily SP file URL. region in {uk,ire}, market in {win,place}."""
    ddmmyyyy = f"{day:02d}{month:02d}{year:04d}"
    return f"{_BASE}/dwbfprices{region.lower()}{market.lower()}{ddmmyyyy}.csv"


def _parse_distance(event_name: str) -> Optional[str]:
    m = _DISTANCE_RE.search(event_name or "")
    return m.group(1).lower() if m else None


def _parse_dt(raw: str) -> Optional[datetime]:
    """'31-05-2026 16:55' -> Dublin-aware datetime."""
    try:
        naive = datetime.strptime(str(raw).strip(), "%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        return None
    return TZ.localize(naive)  # pytz: localize naive, never tzinfo=TZ


def _venue_from_menu_hint(menu_hint: str) -> str:
    # "Nottingham 31st May" -> "Nottingham"; strip trailing date words.
    text = str(menu_hint or "").strip()
    text = re.sub(r"\s+\d+(st|nd|rd|th)\s+\w+.*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def parse_csv(text: str, region: str, market: str) -> pd.DataFrame:
    """Parse one SP CSV body into backbone rows."""
    raw = pd.read_csv(io.StringIO(text))
    raw.columns = [c.strip().lower() for c in raw.columns]
    if raw.empty:
        return pd.DataFrame(columns=BACKBONE_COLUMNS)
    out = pd.DataFrame()
    out["race_date"] = raw["event_dt"].map(_parse_dt)
    out["venue"] = raw["menu_hint"].map(_venue_from_menu_hint)
    out["horse_id"] = pd.to_numeric(raw["selection_id"], errors="coerce").astype("Int64")
    out["horse_name"] = raw["selection_name"].astype(str).str.strip()
    bsp = pd.to_numeric(raw["bsp"], errors="coerce")
    out["odds_finish"] = bsp.where(bsp < 1001.0)  # 1001 = no SP backers -> null
    out["win_lose"] = pd.to_numeric(raw["win_lose"], errors="coerce").astype("Int8")
    out["distance"] = raw["event_name"].map(_parse_distance)
    out["market_type"] = market.upper()
    out["region"] = region.upper()
    out["morningwap"] = pd.to_numeric(raw.get("morningwap"), errors="coerce")
    out["ppwap"] = pd.to_numeric(raw.get("ppwap"), errors="coerce")
    return out[BACKBONE_COLUMNS]


def fetch_csv(url: str, max_retries: int = 5) -> Optional[str]:
    """GET a daily SP file. 404 -> None (no racing/not published)."""
    delay = 1.0
    rotator = get_proxy_manager()
    for attempt in range(max_retries):
        proxy = rotator.next()
        kwargs: dict = {"headers": _HEADERS, "timeout": 20, "follow_redirects": True}
        if proxy:
            kwargs["proxy"] = proxy
        try:
            with get_rate_limiter().acquire_for_url(url):
                with httpx.Client(**kwargs) as client:
                    resp = client.get(url)
            if resp.status_code == 404:
                logger.debug("SP file 404 (skipping): %s", url)
                rotator.report_success(proxy)
                return None
            resp.raise_for_status()
            rotator.report_success(proxy)
            return resp.text
        except httpx.HTTPError as exc:
            logger.warning("SP fetch error %s (attempt %d): %s", url, attempt + 1, exc)
            rotator.report_failure(proxy)
            _time.sleep(delay)
            delay *= 2
    logger.warning("SP fetch giving up: %s", url)
    return None
