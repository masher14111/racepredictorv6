import os
import re
import ssl
import time as _time
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx
import pandas as pd
from lxml import html as lxml_html
from playwright.sync_api import sync_playwright

from utils import source_health
from utils.config_loader import get_config
from utils.currency import currency_for_venue
from utils.logger import get_logger
from utils.proxy_manager import ProxyUnavailableError, get_proxy_manager, redact_proxy_url
from utils.rate_limiter import get_rate_limiter
from utils.timezone import TZ
from utils.market_validation import INVALID, annotate_primary_win_rows, validate_live_dataframe

logger = get_logger(__name__)

# BoyleSports is confirmed Cloudflare-fronted (see module docstring below) —
# both HTTP fetch tiers below require the proxy gateway; a gateway failure
# must never silently degrade to a direct connection (req 2).
_SOURCE = "boylesports"

# ---------------------------------------------------------------------------
# Config
#
# Confirmed via DevTools 2026-06-12: BoyleSports is a server-rendered ASP.NET
# MVC site behind Cloudflare. There is NO JSON odds API. The race-card index is
# an HTML partial; per-event odds live in the event page HTML. Each-way place
# terms are NOT exposed in the cards — they are cross-sourced from other bookies.
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "boylesports.json"
)

_cfg = get_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
LOW_ODDS_THRESHOLD: float = float(_cfg.get("low_odds_threshold", 2.0))
_BS_CFG: dict = _cfg.get("boylesports", {})

_BASE = "https://www.boylesports.com"
_HORSE_RACING_URL: str = _BS_CFG.get(
    "horse_racing_url", f"{_BASE}/sports/horse-racing"
)
_RACE_CARD_URL: str = _BS_CFG.get(
    "race_card_url",
    f"{_BASE}/sports/horse-racing/race-card?partial=true&widget=true",
)
_PARQUET_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "live_odds.parquet")
)

# Confirmed primary race-winner labels. The parser must find one labelled market
# container and then read only buttons inside it; a document-wide odds-button
# query is forbidden because event pages also contain specials.
_PRIMARY_WIN_MARKET_LABELS = frozenset({
    "win or each way",
    "win or e/w",
    "race winner",
    "to win",
})

# Cap on per-event page fetches per scrape (each is a separate HTTP/DOM request).
_MAX_EVENTS: int = int(_BS_CFG.get("max_events", 60))

# Concurrency for the stateless HTTP tiers (curl_cffi). Event pages are
# independent fetches, so a small thread pool overlaps the network wait. Kept
# modest: Cloudflare 403s adaptively under bursty load, so a smaller pool +
# per-fetch proxy-rotation retry is more reliable than a large pool. The central
# rate limiter still bounds per-domain request rate.
_SCRAPE_WORKERS: int = int(_BS_CFG.get("scrape_workers", 4))

# Per-fetch retry budget for the curl_cffi tier: a 403 / challenge usually means
# the current proxy IP is flagged, so each retry rotates to a fresh IP.
_CURL_ATTEMPTS: int = int(_BS_CFG.get("curl_attempts", 3))

# Sticky-session tier. Workers default to 1: the whole point is to look like one
# person, and a shared curl_cffi Session is not documented thread-safe anyway.
_STICKY_ENABLED: bool = bool(_BS_CFG.get("sticky_session", True))
_STICKY_WORKERS: int = max(1, int(_BS_CFG.get("sticky_workers", 1)))
_STICKY_TTL: int = int(_BS_CFG.get("sticky_ttl_minutes", 10))

# Botasaurus anti-detect browser tier (optional dependency).
_BOTA_ENABLED: bool = bool(_BS_CFG.get("botasaurus", True))
_BOTA_HEADLESS: bool = bool(_BS_CFG.get("botasaurus_headless", False))
_BOTA_MAX_EVENTS: int = int(_BS_CFG.get("botasaurus_max_events", 20))
_BOTA_WAIT: int = int(_BS_CFG.get("botasaurus_wait_seconds", 2))

# Races a tier must cover before its card is accepted and the chain stops. Below
# this, the result is kept as a fallback but the next tier still runs — see
# _run_live_tiers for why a 1-race "success" was the real failure mode.
_MIN_RACES: int = int(_BS_CFG.get("min_races_accept", 3))

# Regions to drop from the index (e.g. simulated "virtuals" races). Matched
# case-insensitively against the region slug parsed from the event href.
_EXCLUDED_REGIONS: set = {
    str(r).strip().lower()
    for r in _BS_CFG.get("exclude_regions", ["virtuals"])
}

# Regions to KEEP. When set, this allowlist wins over the blacklist above, so a
# region BoyleSports adds later is skipped by default instead of being fetched
# and then discarded at normalize (uk_ire_only). That matters here: every foreign
# event is a full proxy request against a Cloudflare-fronted site, so scraping
# Churchill Downs to throw it away both costs bandwidth and raises the 403 rate
# on the UK/IRE events we actually want.
#
# It is applied fail-safe — see _parse_index. If the allowlist ever matches
# nothing (BoyleSports renames a slug), the blacklist is used instead and the
# real slugs are logged, so a stale allowlist can never silently empty the card.
_INCLUDED_REGIONS: set = {
    str(r).strip().lower()
    for r in (_BS_CFG.get("include_regions") or [])
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IE,en;q=0.9",
    "Referer": _HORSE_RACING_URL,
}

# Cloudflare "Just a moment" interstitial markers.
_CF_CHALLENGE_RE = re.compile(
    r"just a moment|cf-challenge|challenge-platform|cf_chl_", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BotDetectedError(Exception):
    """Raised on 403 or a Cloudflare challenge interstitial."""

    def __init__(self, message: str = "", reason: str = "bot_detected"):
        super().__init__(message)
        self.reason = reason


class ScraperError(Exception):
    """Raised when all fetch tiers (including stale cache) are exhausted."""

    def __init__(self, message: str = "", reason: str = "other"):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------
class Throttler:
    def __init__(self, rate: float = 1.0):
        self._min_interval = 1.0 / rate
        self._last_call: float = 0.0

    def acquire(self) -> None:
        elapsed = _time.monotonic() - self._last_call
        wait = self._min_interval - elapsed
        if wait > 0:
            _time.sleep(wait)
        self._last_call = _time.monotonic()


_BS_DOMAIN = "www.boylesports.com"


class BoyleSportsClient:
    """Fetches HTML (race-card index + event pages). No JSON involved."""

    def __init__(self, throttler: "Throttler" = None, proxy_rotator=None, rate_limiter=None):
        # throttler injected by tests (Throttler(rate=100.0)); production uses rate_limiter
        self._throttler = throttler
        self._rate_limiter = rate_limiter
        self._rotator = proxy_rotator or get_proxy_manager()

    def _acquire_slot(self):
        """Return a context manager that enforces rate + concurrency limits."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            if self._throttler is not None:
                self._throttler.acquire()
                yield
            else:
                rl = self._rate_limiter or get_rate_limiter()
                with rl.acquire(_BS_DOMAIN):
                    yield

        return _cm()

    def get_html(self, url: str) -> str:
        with self._acquire_slot():
            try:
                proxy = self._rotator.next(required=True)
            except ProxyUnavailableError as exc:
                source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
                raise ScraperError(
                    f"BoyleSports: proxy gateway unavailable ({url})",
                    reason="proxy_unavailable",
                ) from exc
            mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
            logger.debug("GET %s proxy=%s", url, redact_proxy_url(proxy))
            try:
                with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
                    resp = client.get(url, follow_redirects=True)
                if resp.status_code == 403:
                    self._rotator.report_failure(proxy)
                    raise BotDetectedError(f"403 from BoyleSports: {url}", reason="bot_detected")
                text = resp.text
                if _CF_CHALLENGE_RE.search(text[:4000]):
                    self._rotator.report_failure(proxy)
                    raise BotDetectedError(f"Cloudflare challenge on {url}", reason="challenge")
                resp.raise_for_status()
                self._rotator.report_success(proxy)
                return text
            except BotDetectedError:
                raise
            except httpx.TimeoutException as exc:
                self._rotator.report_failure(proxy)
                raise ScraperError(f"Timeout fetching {url}", reason="timeout") from exc
            except ssl.SSLError as exc:
                self._rotator.report_failure(proxy)
                raise ScraperError(f"TLS error fetching {url}", reason="tls_error") from exc
            except Exception:
                self._rotator.report_failure(proxy)
                raise


class CacheManager:
    """Thin adapter over utils.cache.Cache preserving the original is_fresh/read/write API."""

    def __init__(self, path: str = _CACHE_PATH, ttl: int = SCRAPE_INTERVAL):
        from pathlib import Path as _Path
        from utils.cache import Cache
        self.path = path
        self.ttl = ttl
        _p = _Path(path)
        self._cache = Cache(cache_dir=_p.parent, default_ttl=ttl)
        self._key = _p.stem

    def is_fresh(self) -> bool:
        return self._cache.is_fresh(self._key, ttl=self.ttl)

    def read(self) -> Optional[dict]:
        return self._cache.get_stale(self._key)

    def write(self, data: dict) -> None:
        self._cache.set(self._key, data, ttl=self.ttl)


# ---------------------------------------------------------------------------
# Price parsing (low-odds robust)
# ---------------------------------------------------------------------------
def _to_decimal(raw) -> float:
    """Parse a BoyleSports price into decimal odds.

    Handles decimal floats/strings, fractional ("5/2"), and "EVS"/"Evens".
    Returns NaN for missing, "NR", or malformed input.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    if not text or text in ("nr", "sp", "-"):
        return float("nan")
    if text in ("evs", "even", "evens"):
        return 2.0
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            return round(1.0 + float(num) / float(den), 6)
        except (ValueError, ZeroDivisionError):
            return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


import math as _math


def _ew_margin(win_odds_list, places, reduction) -> float:
    """Place-market overround: sum(1/place_odds) - places.

    place_odds = 1 + (win_odds - 1) * reduction. Returns NaN if terms missing
    or no valid runners.
    """
    if not places or reduction is None:
        return float("nan")
    total = 0.0
    count = 0
    for win_odds in win_odds_list:
        if win_odds is None or _math.isnan(win_odds) or win_odds <= 1.0:
            continue
        place_odds = 1.0 + (win_odds - 1.0) * reduction
        if place_odds <= 0:
            continue
        total += 1.0 / place_odds
        count += 1
    if count == 0:
        return float("nan")
    return total - places


def _is_low(odds) -> bool:
    return bool(odds == odds and odds < LOW_ODDS_THRESHOLD)


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------
# Event href: /sports/horse-racing/{region}/{ddmmyy}/{course}/{HH:MM}
_EVENT_HREF_RE = re.compile(
    r"^/sports/horse-racing/(?P<region>[a-z-]+)/(?P<date>\d{6})/"
    r"(?P<course>[a-z0-9-]+)/(?P<time>\d{1,2}:\d{2})$",
    re.IGNORECASE,
)


def _race_time_iso(ddmmyy: str, hhmm: str) -> str:
    """Build a Europe/Dublin ISO timestamp from ddmmyy + HH:MM."""
    day, month, year = int(ddmmyy[0:2]), int(ddmmyy[2:4]), 2000 + int(ddmmyy[4:6])
    hour, minute = (int(p) for p in hhmm.split(":"))
    # TZ is a pytz timezone — must localize a naive datetime, not pass tzinfo=.
    dt = TZ.localize(datetime(year, month, day, hour, minute))
    return dt.isoformat()


def _course_to_venue(course: str) -> str:
    return course.replace("-", " ").title()


def _parse_index(html: str) -> list:
    """Parse the race-card index partial into a list of event meta dicts.

    Each: {race_id, event_url, venue, race_time, region, resulted}.
    """
    tree = lxml_html.fromstring(html)
    events = []
    seen = set()
    for grp in tree.xpath('//*[contains(@class,"race-card-group")]'):
        cls = grp.get("class", "")
        resulted = "race-resulted" in cls
        # The href + data-eventid live on the descendant race-card anchor,
        # not on the group <li> itself.
        href = None
        anchor = None
        for a in grp.xpath('.//a[@href]'):
            m = _EVENT_HREF_RE.match(a.get("href", ""))
            if m:
                href, anchor, groups = a.get("href"), a, m.groupdict()
                break
        if not href:
            continue
        region = groups["region"]
        race_id = anchor.get("data-eventid", "") or href
        if href in seen:
            continue
        seen.add(href)
        course = groups["course"]
        try:
            race_time = _race_time_iso(groups["date"], groups["time"])
        except (ValueError, IndexError):
            race_time = ""
        events.append({
            "race_id": race_id or href,
            "event_url": _BASE + href,
            "venue": _course_to_venue(course),
            "race_time": race_time,
            "region": region,
            "resulted": resulted,
        })
    return _filter_regions(events)


def _filter_regions(events: list) -> list:
    """Apply the region allowlist, falling back to the blacklist if it misses.

    The fallback is the point: an allowlist that stops matching (a renamed slug)
    would otherwise turn a full card into an empty one silently, which looks
    exactly like a total scrape failure. Instead we log the slugs actually seen
    and degrade to the previous behaviour.
    """
    if not events:
        return events

    if _INCLUDED_REGIONS:
        kept = [e for e in events if str(e["region"]).lower() in _INCLUDED_REGIONS]
        if kept:
            skipped = len(events) - len(kept)
            if skipped:
                logger.info(
                    "BoyleSports: skipping %d of %d events outside %s "
                    "(they would be dropped at normalize anyway)",
                    skipped, len(events), sorted(_INCLUDED_REGIONS),
                )
            return kept
        logger.warning(
            "BoyleSports: include_regions %s matched none of the %d events "
            "(slugs seen: %s) — falling back to exclude_regions",
            sorted(_INCLUDED_REGIONS), len(events),
            sorted({str(e["region"]).lower() for e in events}),
        )

    return [e for e in events if str(e["region"]).lower() not in _EXCLUDED_REGIONS]


def _parse_event(html: str, meta: dict) -> list:
    """Parse an event page's WIN-market runner buttons into standardized rows.

    Each-way fields are left null here; they are cross-sourced later.
    """
    tree = lxml_html.fromstring(html)
    venue = meta.get("venue", "")
    currency = currency_for_venue(venue)
    race_id = meta.get("race_id", "")
    race_time = meta.get("race_time", "")

    market_groups = _market_button_groups(tree)
    primary = [
        group for group in market_groups
        if group["market_name"] in _PRIMARY_WIN_MARKET_LABELS
    ]
    if not primary:
        logger.warning(
            "BoyleSports race %s: no confirmed primary race-winner container",
            race_id,
        )
        return []

    rows = []
    for group in primary:
        seen_sel = set()
        for a in group["buttons"]:
            sel_id = str(a.get("data-selectionid") or "").strip()
            if sel_id in seen_sel:
                # Retain duplicate rows so race-level validation can fail closed.
                pass
            name = (a.get("data-name") or "").strip()
            if not name or (a.get("data-isnr") or "").lower() == "true":
                continue
            seen_sel.add(sel_id)
            odds = _to_decimal(a.get("data-price"))
            rows.append({
                "race_id": race_id,
                "race_time": race_time,
                "venue": venue,
                "market_type": "WIN",
                "market_id": group["market_id"],
                "market_name": group["market_name_raw"],
                "selection_id": sel_id,
                "ew_places": None,
                "ew_reduction": None,
                "ew_margin": float("nan"),
                "horse_name": name,
                "odds_decimal": odds,
                "sp": float("nan"),
                "is_low_odds": _is_low(odds),
                "currency": currency,
            })

    annotated = annotate_primary_win_rows(
        rows, primary_market_count=len(primary)
    )
    if annotated and annotated[0]["validation_status"] == INVALID:
        logger.error(
            "BoyleSports race %s failed closed: %s",
            race_id, annotated[0]["validation_reasons"],
        )
    return annotated


def _normalise_market_label(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _market_container_label(container) -> tuple[str, str]:
    """Return normalized/raw label from market metadata or heading text."""
    for attr in (
        "data-market-name", "data-marketname", "data-name", "aria-label",
    ):
        raw = container.get(attr)
        if _normalise_market_label(raw) in _PRIMARY_WIN_MARKET_LABELS:
            return _normalise_market_label(raw), str(raw).strip()

    labels = container.xpath(
        ".//*[self::h1 or self::h2 or self::h3 or self::h4 or self::h5 "
        "or self::h6 or self::legend or "
        "contains(concat(' ',normalize-space(@class),' '),' market-title ') or "
        "contains(concat(' ',normalize-space(@class),' '),' market-header ') "
        "or contains(concat(' ',normalize-space(@class),' '),' market-name ')]"
    )
    for node in labels:
        raw = " ".join(t.strip() for t in node.xpath(".//text()") if t.strip())
        normalized = _normalise_market_label(raw)
        if normalized in _PRIMARY_WIN_MARKET_LABELS:
            return normalized, raw
    return "", ""


def _market_button_groups(tree) -> list[dict]:
    """Return confirmed primary-market groups, scoped to their runner container.

    The live race card declares the active primary market in
    ``#racingNavMiniMenu`` and renders its runners in
    ``#RacingMarketSelections``. Price boosts and other specials are siblings
    under ``#RacingMarketRefresh`` and must never participate in discovery.
    """
    nav_markets = []
    for nav in tree.xpath('//*[@id="racingNavMiniMenu"]//li[@data-marketid]'):
        raw_label = str(
            nav.get("data-marketname")
            or nav.get("data-markethorsename")
            or ""
        ).strip()
        label = _normalise_market_label(raw_label)
        classes = set(str(nav.get("class") or "").split())
        if label in _PRIMARY_WIN_MARKET_LABELS and "active" in classes:
            nav_markets.append({
                "market_id": str(nav.get("data-marketid") or "").strip(),
                "market_name": label,
                "market_name_raw": raw_label,
            })

    selections = tree.xpath('//*[@id="RacingMarketSelections"]')
    if nav_markets or selections:
        if not nav_markets or len(selections) != 1:
            return []
        runner_container = selections[0]
        groups = []
        for market in nav_markets:
            market_id = market["market_id"]
            buttons = runner_container.xpath(
                './/a[contains(concat(" ",normalize-space(@class)," ")," odds ") '
                "and @data-selectionid and @data-name and @data-marketid=$mid "
                "and not(ancestor::*[contains(concat(' ',normalize-space(@class),' '),"
                "' featuredSelectionPopup ')]) "
                "and not(ancestor::*[contains(concat(' ',normalize-space(@class),' '),"
                "' default-fave ')])]",
                mid=market_id,
            )
            groups.append({
                **market,
                "container": runner_container,
                "buttons": buttons,
            })
        return groups

    # Strict compatibility fallback for archived fixtures/markup: only a
    # self-labelled market container can qualify. There is no largest-market or
    # runner-name inference here.
    buttons = tree.xpath(
        '//a[contains(concat(" ",normalize-space(@class)," ")," odds ") '
        "and @data-selectionid and @data-name and @data-marketid]"
    )
    by_market: dict[str, list] = {}
    for button in buttons:
        market_id = str(button.get("data-marketid") or "").strip()
        if market_id:
            by_market.setdefault(market_id, []).append(button)

    groups = []
    for market_id, market_buttons in by_market.items():
        container = None
        first = market_buttons[0]
        # Find the nearest common ancestor containing the complete button group
        # and no buttons belonging to another market.
        for ancestor in reversed(first.xpath("ancestor::*")):
            descendant = ancestor.xpath(
                './/a[contains(concat(" ",normalize-space(@class)," ")," odds ") '
                "and @data-selectionid and @data-name and @data-marketid]"
            )
            ids = {str(a.get("data-marketid") or "").strip() for a in descendant}
            same = [a for a in descendant if str(a.get("data-marketid") or "").strip() == market_id]
            if ids == {market_id} and len(same) == len(market_buttons):
                container = ancestor
                break
        if container is None:
            continue
        label, raw_label = _market_container_label(container)
        groups.append({
            "market_id": market_id,
            "market_name": label,
            "market_name_raw": raw_label,
            "container": container,
            "buttons": market_buttons,
        })
    return groups


# ---------------------------------------------------------------------------
# Cross-source each-way terms (BoyleSports cards don't expose them)
# ---------------------------------------------------------------------------
def _ew_key(venue: str, race_time_iso: str) -> Optional[tuple]:
    """Key a race by (venue_lower, UTC epoch-minute) for cross-source matching."""
    if not race_time_iso:
        return None
    try:
        dt = datetime.fromisoformat(race_time_iso).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None
    return ((venue or "").strip().lower(), int(dt.timestamp() // 60))


def _cross_source_ew(rows: list, parquet_path: str = _PARQUET_PATH) -> list:
    """Fill ew_places/ew_reduction from other bookies' rows (matched venue+time),
    then recompute ew_margin per race from BoyleSports win odds."""
    if not rows or not os.path.exists(parquet_path):
        return rows
    try:
        other = pd.read_parquet(parquet_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Cross-source EW: could not read parquet (%s)", exc)
        return rows
    other = other[(other["source"] != "boylesports") & other["ew_places"].notna()]
    if other.empty:
        return rows

    lookup = {}
    for _, r in other.iterrows():
        rt = r["race_time"]
        rt_iso = rt.isoformat() if hasattr(rt, "isoformat") else str(rt)
        key = _ew_key(r["venue"], rt_iso)
        if key and key not in lookup:
            lookup[key] = (r["ew_places"], r["ew_reduction"])

    # Apply terms.
    for row in rows:
        key = _ew_key(row["venue"], row["race_time"])
        if key and key in lookup:
            places, reduction = lookup[key]
            row["ew_places"] = None if pd.isna(places) else int(places)
            row["ew_reduction"] = None if pd.isna(reduction) else float(reduction)

    # Recompute margin per race_id from the matched terms + win odds.
    by_race = {}
    for row in rows:
        by_race.setdefault(row["race_id"], []).append(row)
    for race_rows in by_race.values():
        places = race_rows[0]["ew_places"]
        reduction = race_rows[0]["ew_reduction"]
        margin = _ew_margin([r["odds_decimal"] for r in race_rows], places, reduction)
        for r in race_rows:
            r["ew_margin"] = margin
    return rows


# ---------------------------------------------------------------------------
# DataFrame / parquet
# ---------------------------------------------------------------------------
_PARQUET_COLUMNS = [
    "fetched_at", "source", "race_id", "race_time", "venue", "market_type",
    "market_id", "market_name", "selection_id",
    "ew_places", "ew_reduction", "ew_margin", "horse_name", "odds_decimal",
    "sp", "is_low_odds", "currency", "stale", "validation_status",
    "validation_reasons", "field_size", "booksum",
]


def _rows_to_df(rows: list, fetched_at: datetime, stale: bool = False) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "fetched_at", fetched_at.astimezone(timezone.utc))
    df.insert(1, "source", "boylesports")
    df["stale"] = stale
    df = validate_live_dataframe(df)
    if df.empty:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["race_time"] = pd.to_datetime(df["race_time"], utc=True, errors="coerce")
    df["ew_places"] = pd.array(df["ew_places"].tolist(), dtype="Int64")
    df["ew_reduction"] = pd.array(df["ew_reduction"].tolist(), dtype="Float64")
    df["ew_margin"] = pd.array(df["ew_margin"].tolist(), dtype="Float64")
    df["is_low_odds"] = df["is_low_odds"].astype(bool)
    df["stale"] = df["stale"].astype(bool)
    return df[_PARQUET_COLUMNS]


def _merge_parquet(df: pd.DataFrame, path: str = _PARQUET_PATH) -> None:
    """Read-merge-write: drop this source's old rows, append fresh, write back."""
    from utils.storage import parquet_store
    existing = parquet_store.read_parquet(path)
    if not existing.empty and "source" in existing.columns:
        existing = existing[existing["source"] != "boylesports"]
        df = pd.concat([existing, df], ignore_index=True)
    parquet_store.write_parquet(df, path)


# ---------------------------------------------------------------------------
# Row collection (index → per-event), parameterised by an HTML fetcher
# ---------------------------------------------------------------------------
def _collect_rows(
    get_html: Callable[[str], str],
    max_events: int = _MAX_EVENTS,
    max_workers: int = 1,
) -> list:
    """Fetch the index, then each open event, parsing rows. `get_html(url)->str`.

    With ``max_workers > 1`` the per-event fetches run concurrently in a thread
    pool — the events are independent network calls, so this overlaps the I/O
    wait and is the main scraper speedup. Only safe for stateless fetchers
    (curl_cffi / httpx); the Playwright/Selenium tiers share one page/driver and
    must stay serial (max_workers=1). The central rate limiter still throttles
    per-domain, so concurrency can't hammer the site.
    """
    index_html = get_html(_RACE_CARD_URL)
    events = _parse_index(index_html)
    open_events = [e for e in events if not e["resulted"]]
    if max_events:
        open_events = open_events[:max_events]

    def _one(ev: dict) -> list:
        try:
            return _parse_event(get_html(ev["event_url"]), ev)
        except ScraperError as exc:
            if getattr(exc, "reason", None) == "proxy_unavailable":
                # The gateway itself is gone — retrying per-event just repeats
                # the same failure; propagate so the tier aborts immediately
                # instead of looping through every remaining event.
                raise
            logger.warning("BoyleSports: event %s failed: %s", ev["event_url"], exc)
            return []
        except Exception as exc:  # noqa: BLE001
            logger.warning("BoyleSports: event %s failed: %s", ev["event_url"], exc)
            return []

    if max_workers > 1 and len(open_events) > 1:
        from concurrent.futures import ThreadPoolExecutor
        rows: list = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(open_events))) as pool:
            for ev_rows in pool.map(_one, open_events):
                rows.extend(ev_rows)
        return rows

    rows = []
    for ev in open_events:
        rows.extend(_one(ev))
    return rows


# ---------------------------------------------------------------------------
# Sticky-session tier — one IP, one cookie jar, serial. Tried FIRST.
# ---------------------------------------------------------------------------
# Every other tier here is stateless: a fresh rotating exit IP per request and no
# cookie jar anywhere, so each event page arrives as a brand-new visitor with no
# cf_clearance. Cloudflare reads forty such requests as a bot farm, which is why
# the index would fetch cleanly and then the event burst would 403 — the tuning
# note about "8 workers is the reliable ceiling" was describing that symptom.
#
# This tier does the opposite: pin ONE Irish exit IP for the pass, keep ONE
# session so the clearance cookie earned on the index is replayed on every event
# page, and fetch serially. One plausible human instead of forty strangers.
def _sticky_collect() -> list:
    """Tier-0 collector: single sticky IP + persistent cookies, serial by default."""
    from curl_cffi import requests as creq

    rotator = get_proxy_manager()
    session_id = f"bs{int(_time.time())}"  # fresh per run, so a burned IP is not reused
    try:
        proxy = rotator.sticky(session_id, ttl_minutes=_STICKY_TTL, required=True)
    except ProxyUnavailableError as exc:
        source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
        raise ScraperError(
            "BoyleSports: proxy gateway unavailable (sticky)", reason="proxy_unavailable"
        ) from exc

    proxies = {"http": proxy, "https": proxy} if proxy else None
    logger.info(
        "BoyleSports: sticky-session tier active (one IP, cookies kept, %d worker(s))",
        _STICKY_WORKERS,
    )

    session = creq.Session()
    try:
        def get_html(url: str) -> str:
            with get_rate_limiter().acquire_for_url(url):
                resp = session.get(
                    url, headers=_CURL_HEADERS, impersonate="chrome",
                    proxies=proxies, timeout=30, allow_redirects=True,
                )
            text = resp.text
            if _is_cf_block(resp.status_code, text):
                raise BotDetectedError(
                    f"Cloudflare block ({resp.status_code}) on {url}", reason="bot_detected"
                )
            if resp.status_code >= 400:
                resp.raise_for_status()
            return text

        # _collect_rows fetches the index first, so the clearance cookie the
        # index hands back is already in the jar for every event page after it.
        rows = _collect_rows(get_html, max_workers=_STICKY_WORKERS)
    finally:
        try:
            session.close()
        except Exception:  # noqa: BLE001 — closing must never mask a fetch error
            pass

    if not rows:
        raise ScraperError("sticky session: no rows parsed", reason="empty_card")
    rotator.report_success(proxy)
    return rows


# ---------------------------------------------------------------------------
# curl_cffi tier (Chrome TLS impersonation) — primary Cloudflare bypass
# ---------------------------------------------------------------------------
# Headers minus User-Agent: impersonate="chrome" sets the UA + sec-ch-ua client
# hints to match the spoofed TLS fingerprint, so a hand-set UA would only
# desync them. Keep the contextual headers (Referer / Accept / Accept-Language).
_CURL_HEADERS = {k: v for k, v in _HEADERS.items() if k.lower() != "user-agent"}


def _is_cf_block(status: int, text: str) -> bool:
    """True only for a genuine Cloudflare block/interstitial.

    The "Just a moment…" challenge page always carries that literal text. The
    broad _CF_CHALLENGE_RE also matches the benign `challenge-platform` telemetry
    script Cloudflare injects into EVERY fronted page, so it's only trusted on a
    tiny body (real challenge pages are small; real race cards are 100s of KB).
    """
    if status == 403:
        return True
    low = text.lower()
    if "just a moment" in low:
        return True
    return len(text) < 4096 and bool(_CF_CHALLENGE_RE.search(text))


def _curl_cffi_get_html(url: str, attempts: int = _CURL_ATTEMPTS) -> str:
    """Fetch a page with a real Chrome TLS handshake via curl_cffi.

    BoyleSports sits behind Cloudflare, which TLS-fingerprints plain httpx and
    serves a 403 / "Just a moment" challenge. curl_cffi's `impersonate="chrome"`
    reproduces Chrome's JA3 fingerprint and gets through without a browser.
    Cloudflare 403s adaptively per source IP, so a block triggers a retry on a
    fresh proxy IP (up to `attempts`). Routed through the proxy rotator + central
    rate limiter, mirroring the proven results-site fetcher in betsp_historical.
    """
    from curl_cffi import requests as creq
    from curl_cffi.requests.exceptions import SSLError as CurlSSLError, Timeout as CurlTimeout

    rotator = get_proxy_manager()
    last_exc: Optional[Exception] = None
    for _ in range(max(1, attempts)):
        try:
            proxy = rotator.next(required=True)
        except ProxyUnavailableError as exc:
            source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
            raise ScraperError(
                f"BoyleSports: proxy gateway unavailable ({url})",
                reason="proxy_unavailable",
            ) from exc
        proxies = {"http": proxy, "https": proxy} if proxy else None
        with get_rate_limiter().acquire_for_url(url):
            try:
                resp = creq.get(
                    url, headers=_CURL_HEADERS, impersonate="chrome",
                    proxies=proxies, timeout=30, allow_redirects=True,
                )
            except CurlTimeout as exc:
                rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="timeout")
                last_exc = exc
                continue
            except CurlSSLError as exc:
                rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="tls_error")
                last_exc = exc
                continue
            except Exception as exc:  # noqa: BLE001 — network failure: rotate IP and retry
                rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="other")
                last_exc = exc
                continue
        text = resp.text
        if _is_cf_block(resp.status_code, text):
            # This proxy IP is flagged — penalise it and try a fresh one.
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="bot_detected")
            last_exc = BotDetectedError(
                f"Cloudflare block ({resp.status_code}) on {url}", reason="bot_detected"
            )
            continue
        if resp.status_code >= 400:
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="other")
            resp.raise_for_status()
        rotator.report_success(proxy)
        return text
    raise last_exc or BotDetectedError(f"curl_cffi exhausted retries on {url}", reason="bot_detected")


def _curl_collect() -> list:
    """Tier-1 collector: drive _collect_rows through the curl_cffi fetcher,
    fetching event pages concurrently (the per-domain rate limiter still applies)."""
    logger.info("BoyleSports: curl_cffi (Chrome impersonation) tier active")
    rows = _collect_rows(_curl_cffi_get_html, max_workers=_SCRAPE_WORKERS)
    if not rows:
        raise ScraperError("curl_cffi: no rows parsed", reason="empty_card")
    return rows


# ---------------------------------------------------------------------------
# Botasaurus tier — anti-detect browser, tried before Playwright/Selenium
# ---------------------------------------------------------------------------
# The existing browser tiers launch vanilla headless Chrome, which announces
# itself: navigator.webdriver is set, CDP artifacts are visible, and the headless
# UA is distinctive. Cloudflare rejects that on sight, so tiers 3-4 were never
# really a second chance — they were the same request wearing a thin disguise.
#
# botasaurus-driver patches those tells and adds two things that matter here:
# google_get(bypass_cloudflare=True), which arrives with a Google referer and
# sits through the interstitial, and is_bot_detected_by_cloudflare(), which lets
# this tier report an honest failure instead of parsing a challenge page as if
# it were a race card.
#
# It reuses the sticky proxy, so the browser also keeps one exit IP for the pass.
def _resolve_chrome_path() -> Optional[str]:
    """Which Chromium binary Botasaurus should drive.

    Botasaurus looks for a real Google Chrome install and raises if it finds
    none. This machine has no Chrome, but Playwright already ships a Chromium
    for the tier below — reusing it means the anti-detect tier works out of the
    box with nothing extra to install. Returning None lets Botasaurus auto-detect
    (correct when real Chrome IS present, which fingerprints better).
    """
    configured = _BS_CFG.get("botasaurus_chrome_path")
    if configured and os.path.exists(configured):
        return configured

    for candidate in (
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ):
        if os.path.exists(candidate):
            return None  # real Chrome present — let Botasaurus find it itself

    # Fall back to Playwright's bundled Chromium.
    pw_root = os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "ms-playwright"
    )
    if os.path.isdir(pw_root):
        for entry in sorted(os.listdir(pw_root), reverse=True):
            if not entry.startswith("chromium-"):
                continue
            for sub in ("chrome-win64", "chrome-win"):
                exe = os.path.join(pw_root, entry, sub, "chrome.exe")
                if os.path.exists(exe):
                    return exe
    return None


def _botasaurus_collect() -> list:
    """Anti-detect browser tier. Optional dependency — skipped cleanly if absent."""
    try:
        from botasaurus_driver import Driver
    except ImportError as exc:
        raise ScraperError(
            "botasaurus-driver not installed (pip install botasaurus-driver)",
            reason="not_configured",
        ) from exc

    rotator = get_proxy_manager()
    session_id = f"bsbota{int(_time.time())}"
    try:
        proxy = rotator.sticky(session_id, ttl_minutes=_STICKY_TTL, required=True)
    except ProxyUnavailableError as exc:
        source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
        raise ScraperError(
            "BoyleSports: proxy gateway unavailable (botasaurus)",
            reason="proxy_unavailable",
        ) from exc

    chrome = _resolve_chrome_path()
    logger.warning(
        "BoyleSports: Botasaurus anti-detect browser tier active "
        "(headless=%s, chrome=%s)", _BOTA_HEADLESS, chrome or "auto-detected",
    )
    driver = Driver(
        headless=_BOTA_HEADLESS,
        proxy=proxy or None,
        # The card is text; skipping images/CSS cuts proxy bandwidth a lot and
        # does not change how the page fingerprints.
        block_images_and_css=True,
        lang="en-GB",
        chrome_executable_path=chrome,
        # BoyleSports event pages keep long-lived connections open for live price
        # updates, so they never reach "complete" and the default ready-state wait
        # times out after 60s on a page that had in fact already rendered.
        wait_for_complete_page_load=False,
    )
    first = True
    try:
        def get_html(url: str) -> str:
            nonlocal first
            with get_rate_limiter().acquire_for_url(url):
                if first:
                    # Arrive with a Google referer and sit through the
                    # interstitial. Once cleared, the browser holds cf_clearance,
                    # so every later page is an ordinary navigation in the same
                    # session — no need to route them all through Google.
                    driver.google_get(url, bypass_cloudflare=True)
                    first = False
                else:
                    driver.get(url, wait=_BOTA_WAIT)
            if driver.is_bot_detected_by_cloudflare():
                raise BotDetectedError(
                    f"Cloudflare still challenging {url} after bypass",
                    reason="bot_detected",
                )
            html = driver.page_html or ""
            if _is_cf_block(200, html):
                raise BotDetectedError(f"Cloudflare interstitial on {url}",
                                       reason="bot_detected")
            return html

        rows = _collect_rows(get_html, max_events=_BOTA_MAX_EVENTS, max_workers=1)
    finally:
        try:
            driver.close()
        except Exception:  # noqa: BLE001 — a close failure must not mask the result
            pass

    if not rows:
        raise ScraperError("Botasaurus: no rows parsed", reason="empty_card")
    rotator.report_success(proxy)
    return rows


# ---------------------------------------------------------------------------
# Browser fetchers (Playwright / Selenium tiers)
# ---------------------------------------------------------------------------
def _playwright_collect() -> list:
    logger.warning("BoyleSports: Playwright DOM fallback active")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(extra_http_headers={"Accept-Language": "en-IE,en;q=0.9"})

        def get_html(url: str) -> str:
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_timeout(2500)
            return page.content()

        try:
            rows = _collect_rows(get_html)
        finally:
            browser.close()
    if not rows:
        raise ScraperError("PlaywrightFallback: no rows parsed", reason="empty_card")
    return rows


def _selenium_collect() -> list:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    logger.warning("BoyleSports: Selenium DOM fallback active")
    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=opts)
    try:
        def get_html(url: str) -> str:
            driver.get(url)
            _time.sleep(2.5)
            return driver.page_source

        rows = _collect_rows(get_html)
    finally:
        driver.quit()
    if not rows:
        raise ScraperError("SeleniumFallback: no rows parsed", reason="empty_card")
    return rows


# ---------------------------------------------------------------------------
# Firecrawl tier (server-side fetch) — last resort, and the only PAID tier
# ---------------------------------------------------------------------------
def _firecrawl_collect() -> list:
    """Tier-5 collector: fetch server-side via Firecrawl.

    Tiers 1-4 all egress through our residential proxy pool, so when Cloudflare
    has flagged the pool they fail identically. Firecrawl fetches from its own
    infrastructure, which is why it is worth having — but every page is a billed
    credit, so the event list is capped separately from the free tiers and the
    module-level budget in firecrawl_fetch hard-stops a runaway.
    """
    from scraper import firecrawl_fetch

    ok, why = firecrawl_fetch.is_available()
    if not ok:
        raise ScraperError(f"Firecrawl tier skipped: {why}", reason="not_configured")

    fc_cfg = _cfg.get("firecrawl", {}) or {}
    max_events = int(fc_cfg.get("boylesports_max_events", 40))
    # Never fetch more than the shared cap would have (_MAX_EVENTS == 0 = unlimited).
    if _MAX_EVENTS:
        max_events = min(max_events, _MAX_EVENTS)
    workers = int(fc_cfg.get("workers", 4))

    logger.warning(
        "BoyleSports: Firecrawl tier active (PAID — up to %d event pages)", max_events
    )
    rows = _collect_rows(
        firecrawl_fetch.get_html, max_events=max_events, max_workers=workers
    )
    logger.info("BoyleSports: Firecrawl used %d page(s)", firecrawl_fetch.pages_used())
    if not rows:
        raise ScraperError("Firecrawl: no rows parsed", reason="empty_card")
    return rows


def _run_live_tiers() -> Optional[list]:
    """sticky → curl_cffi → httpx → Botasaurus → Playwright → Selenium → Firecrawl.

    The sticky-session tier goes first because it is the only one that presents a
    coherent session: one pinned exit IP with a cookie jar, so the clearance
    earned on the index is replayed on the event pages. The rotating tiers below
    remain as the fallback — if the sticky IP is already flagged, rotating gives
    a fresh one to try. Firecrawl is last because it is the only paid tier (and
    is disabled by default, having measurably failed to reach this site at all).
    """
    # A tier "succeeding" with a handful of rows is the common failure mode here:
    # Cloudflare lets the index through and then 403s most event pages, so a tier
    # returns a card with one race on it. Returning that stopped the chain before
    # the anti-detect browser ever ran — on 2026-09-18 curl_cffi returned 14 rows
    # from 1 race while Botasaurus could deliver 80 from 9. So each tier's result
    # is only accepted once it covers `min_races`; otherwise the best partial card
    # is remembered and the next tier is tried.
    best: list = []

    def _accept(rows: Optional[list], tier: str) -> bool:
        nonlocal best
        if rows and len(rows) > len(best):
            best = rows
        n_races = len({r.get("race_id") for r in (rows or []) if r.get("race_id")})
        if rows and n_races >= _MIN_RACES:
            return True
        if rows:
            logger.warning(
                "BoyleSports: %s returned only %d row(s) across %d race(s) "
                "(min_races=%d) — keeping it as a fallback and trying the next tier",
                tier, len(rows), n_races, _MIN_RACES,
            )
        return False

    # Tier 0: one sticky IP + persistent cookies, serial.
    if _STICKY_ENABLED:
        try:
            if _accept(_sticky_collect(), "sticky session"):
                return best
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier 0 (sticky session) failed (%s)", exc)
            source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 1: curl_cffi with Chrome TLS impersonation (concurrent event fetches).
    try:
        if _accept(_curl_collect(), "curl_cffi"):
            return best
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 1 (curl_cffi) failed (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 2: plain httpx (fast; usually fails on Cloudflare challenge).
    try:
        rows = _collect_rows(BoyleSportsClient().get_html)
        if _accept(rows, "httpx"):
            return best
        if not rows:
            raise BotDetectedError("httpx returned no rows", reason="empty_card")
    except BotDetectedError as exc:
        logger.warning("Tier 2 (httpx) failed (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 2 (httpx) error (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 3: Botasaurus anti-detect browser (before the vanilla browser tiers,
    # which Cloudflare rejects on sight).
    if _BOTA_ENABLED:
        try:
            if _accept(_botasaurus_collect(), "Botasaurus"):
                return best
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier 3 (Botasaurus) failed (%s)", exc)
            source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 4: Playwright DOM.
    try:
        if _accept(_playwright_collect(), "Playwright"):
            return best
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 4 (Playwright) failed (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 5: Selenium DOM.
    try:
        if _accept(_selenium_collect(), "Selenium"):
            return best
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 5 (Selenium) failed (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Tier 6: Firecrawl (server-side, PAID). Only reached when every free tier
    # has been challenged — typically means our whole proxy pool is flagged.
    try:
        if _accept(_firecrawl_collect(), "Firecrawl"):
            return best
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 6 (Firecrawl) failed (%s)", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

    # Nothing reached min_races. A partial card still beats falling back to a
    # stale cache, so hand back the richest one any tier managed.
    if best:
        logger.warning(
            "BoyleSports: no tier reached min_races=%d — returning the best "
            "partial card (%d rows)", _MIN_RACES, len(best),
        )
    return best or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scrape(force: bool = False) -> pd.DataFrame:
    """Fetch live BoyleSports horse-racing odds as a standardized DataFrame.

    BoyleSports serves HTML (no JSON API). WIN odds + low-odds flag + race status
    are scraped from the cards; each-way terms are cross-sourced from other
    bookies in the shared parquet. Falls back to stale cache before raising.
    """
    cache = CacheManager(path=_CACHE_PATH)

    if not force and cache.is_fresh():
        logger.debug("BoyleSports cache hit")
        cached = cache.read()
        return _rows_to_df(cached.get("rows", []),
                           datetime.fromisoformat(cached["fetched_at"]))

    rows = _run_live_tiers()

    if rows is None:
        cached = cache.read()
        if cached and cached.get("rows"):
            logger.warning("All live tiers failed — returning STALE cache")
            df = _rows_to_df(cached["rows"],
                             datetime.fromisoformat(cached["fetched_at"]),
                             stale=True)
            # Revalidation intentionally rejects legacy unscoped cache rows; merge
            # even an empty frame so contaminated on-disk rows are purged.
            _merge_parquet(df, _PARQUET_PATH)
            return df
        raise ScraperError("BoyleSports: all fetch tiers failed and no cache")

    rows = _cross_source_ew(rows, _PARQUET_PATH)
    fetched_at = datetime.now(tz=TZ)
    df = _rows_to_df(rows, fetched_at)
    _merge_parquet(df, _PARQUET_PATH)
    cache.write({"fetched_at": fetched_at.isoformat(), "rows": rows})
    race_count = df["race_id"].nunique() if "race_id" in df.columns and not df.empty else 0
    source_health.record_success(_SOURCE, rows=len(df), races=int(race_count))
    logger.debug("BoyleSports scraped %d rows", len(df))
    return df
