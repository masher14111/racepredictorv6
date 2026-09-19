import json
import os
import time as _time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
from lxml import html as lxml_html
from playwright.sync_api import sync_playwright

from utils import source_health
from utils.config_loader import get_config
from utils.currency import currency_for_venue
from utils.logger import get_logger
from utils.proxy_manager import (
    ProxyUnavailableError, get_proxy_manager, redact_proxy_url,
)
from utils.rate_limiter import get_rate_limiter
from utils.storage import parquet_store
from utils.timezone import TZ, to_local
from utils.market_validation import INVALID, annotate_primary_win_rows, validate_live_dataframe

_SOURCE = "livescorebet"

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "livescorebet.json"
)

_cfg = get_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
LOW_ODDS_THRESHOLD: float = float(_cfg.get("low_odds_threshold", 2.0))
_LSB_CFG: dict = _cfg.get("livescorebet", {})

_HORSE_RACING_URL: str = _LSB_CFG.get(
    "horse_racing_url", "https://www.livescorebet.com/horse-racing"
)

# Restrict the racinghome tree to UK & Irish meetings only. The feed groups
# meetings under `type: "Country"` nodes ("UK & Ireland", "United States",
# "France", ...); we keep only the country nodes whose name is in this set and
# drop every foreign subtree before fetching per-event detail. Set
# ``livescorebet.countries: []`` (or null) in config to disable the filter and
# scrape every country again.
_DEFAULT_LSB_COUNTRIES = [
    "uk & ireland", "uk and ireland", "united kingdom", "great britain",
    "ireland", "uk", "ire",
]
_raw_countries = _LSB_CFG.get("countries", _DEFAULT_LSB_COUNTRIES)
_LSB_COUNTRIES: set = {
    str(c).strip().lower() for c in (_raw_countries or []) if str(c).strip()
}
_PARQUET_PATH: str = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        _LSB_CFG.get("parquet_path", "data/live_odds.parquet"),
    )
)

# Browser headers for HTML page fetches (Tier 3 lxml fallback)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IE,en;q=0.9",
}

# API headers for direct gateway calls (Tier 1)
_API_BASE = "https://gateway-ie.livescorebet.com/sportsbook/gateway"
_RACING_HOME_URL = f"{_API_BASE}/v2/view/horses/racinghome?lang=en-ie"
_EVENT_URL_TPL = _API_BASE + "/v1/view/horses/event?eventid={event_id}&lang=en-ie"

# Concurrency for Tier-1 per-event detail fetches. Each get_event() rotates to a
# fresh proxy IP and is bounded by the central per-domain rate limiter
# (gateway-ie.livescorebet.com), so a thread pool overlaps network latency
# without exceeding the configured rps/max_concurrent. The semaphore is the real
# ceiling — keep this ≥ the domain's max_concurrent in config.yaml.
_EVENT_WORKERS: int = int(_LSB_CFG.get("event_workers", 10))

# Per-fetch retry budget for the curl_cffi tier: a proxy can MITM the gateway's
# TLS or stall on a dead exit IP, so rotate to a fresh IP (eventually direct).
_CURL_ATTEMPTS: int = int(_LSB_CFG.get("curl_attempts", 3))

_API_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "client-id": "web",
    "client-device-type": "desktop",
    "client-app-version": "2.66.4985",
    "client-language": "en",
    "content-type": "application/json",
}

# curl_cffi header variants: impersonate="chrome" sets the User-Agent + sec-ch-ua
# client hints to match the spoofed TLS fingerprint, so a hand-set UA would only
# desync them — drop it and keep the contextual/app headers.
_CURL_API_HEADERS = {k: v for k, v in _API_HEADERS.items() if k.lower() != "user-agent"}
_CURL_HTML_HEADERS = {k: v for k, v in _HEADERS.items() if k.lower() != "user-agent"}

# Strict primary WIN identity, reconfirmed against the live gateway on
# 2026-07-25. A market must satisfy all three fields; unknown group IDs/types are
# ignored and can never inherit a default WIN label.
_PRIMARY_WIN_GROUP_ID = 758
_PRIMARY_WIN_MARKET_TYPES = frozenset({"1001558122"})
_PRIMARY_WIN_MARKET_NAMES = frozenset({"to win"})

_RACE_DATA_KEYS = ("events", "races", "meetings", "data")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BotDetectedError(Exception):
    """Raised when LivescoreBet returns 403, HTML challenge, or no race data."""

    def __init__(self, message: str, reason: str = "bot_detected"):
        super().__init__(message)
        self.reason = reason


class ScraperError(Exception):
    """Raised when all fetch tiers are exhausted."""

    def __init__(self, message: str, reason: str = "other"):
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


_LSB_API_DOMAIN = "gateway-ie.livescorebet.com"
_LSB_HTML_DOMAIN = "www.livescorebet.com"


class LivescoreBetClient:
    def __init__(
        self,
        throttler: "Throttler" = None,
        proxy_rotator=None,
        rate_limiter=None,
    ):
        # throttler injected by tests (Throttler(rate=100.0)); production uses rate_limiter
        self._throttler = throttler
        self._rate_limiter = rate_limiter
        self._rotator = proxy_rotator or get_proxy_manager()

    def _acquire_slot(self, domain: str):
        """Return a context manager that enforces rate + concurrency limits."""
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            if self._throttler is not None:
                self._throttler.acquire()
                yield
            else:
                rl = self._rate_limiter or get_rate_limiter()
                with rl.acquire(domain):
                    yield

        return _cm()

    def _curl_get(self, url: str, headers: dict, domain: str, label: str,
                  attempts: int = _CURL_ATTEMPTS):
        """Primary fetch tier: a real Chrome TLS handshake via curl_cffi.

        Mirrors the boylesports/paddy_power tier-1 path. ``impersonate="chrome"``
        reproduces Chrome's JA3 fingerprint, which both clears Cloudflare's TLS
        gate and — unlike plain httpx — completes the handshake against the
        gateway without tripping a self-signed-certificate error (Windows
        enterprise TLS inspection injects a root certifi does not carry).

        The gateway is a plain JSON API (no Cloudflare), so it works cleanly
        direct; a rotating residential proxy can MITM its TLS (cert subject
        mismatch) or stall on a dead exit IP. So each attempt rotates to a fresh
        proxy, penalising the bad one — once the pool blacklists, the rotator
        yields None (direct), which is the reliable path here (this domain does
        NOT require the proxy, unlike the Cloudflare-protected HTML page used by
        LxmlFallback below — required=True there, not here). A 403 becomes
        BotDetectedError only after the retry budget is spent, so scrape() falls
        through to the Playwright/Selenium tiers."""
        from curl_cffi import requests as creq
        from curl_cffi.requests.exceptions import SSLError as CurlSSLError, Timeout as CurlTimeout

        last_exc: Optional[Exception] = None
        for _ in range(max(1, attempts)):
            with self._acquire_slot(domain):
                proxy = self._rotator.next()
                proxies = {"http": proxy, "https": proxy} if proxy else None
                logger.debug("GET %s proxy=%s (curl_cffi)", url, redact_proxy_url(proxy))
                try:
                    resp = creq.get(
                        url, headers=headers, impersonate="chrome",
                        proxies=proxies, timeout=20, allow_redirects=True,
                    )
                except CurlTimeout as exc:
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="timeout")
                    last_exc = exc
                    continue
                except CurlSSLError as exc:
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="tls_error")
                    last_exc = exc
                    continue
                except Exception as exc:  # noqa: BLE001 — rotate IP and retry
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="other")
                    last_exc = exc
                    continue
            if resp.status_code == 403:
                self._rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="bot_detected")
                last_exc = BotDetectedError(f"403 from LivescoreBet ({label})", reason="bot_detected")
                continue
            if resp.status_code >= 400:
                self._rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="other")
                last_exc = BotDetectedError(f"{resp.status_code} from LivescoreBet ({label})")
                continue
            self._rotator.report_success(proxy)
            return resp
        raise last_exc or BotDetectedError(f"curl_cffi exhausted retries ({label})")

    def get_racing_home(self) -> dict:
        """Fetch racinghome JSON — returns dict with meetingsToday[]."""
        return self._curl_get(
            _RACING_HOME_URL, _CURL_API_HEADERS, _LSB_API_DOMAIN, "racing home").json()

    def get_event(self, event_id: str) -> dict:
        """Fetch a single race event — returns dict with currentEvent[]."""
        url = _EVENT_URL_TPL.format(event_id=event_id)
        return self._curl_get(
            url, _CURL_API_HEADERS, _LSB_API_DOMAIN, f"event {event_id}").json()

    def get_page(self) -> str:
        """Fetch the horse-racing HTML page (used by lxml fallback)."""
        return self._curl_get(
            _HORSE_RACING_URL, _CURL_HTML_HEADERS, _LSB_HTML_DOMAIN, "page").text


class PlaywrightFallback:
    def fetch(self) -> list:
        """
        Launch headless Chromium, intercept gateway API responses, parse into rows.
        Returns list of row dicts. Raises ScraperError if no race data captured.
        """
        logger.warning("Bot detected — triggering Playwright XHR fallback")
        captured_rows: list = []
        home_data: Optional[dict] = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def _handle_response(response) -> None:
                nonlocal home_data
                if "gateway-ie.livescorebet.com" not in response.url:
                    return
                try:
                    body = response.json()
                except Exception:
                    return
                if "currentEvent" in body:
                    rows = _parse_event(body)
                    captured_rows.extend(rows)
                    logger.debug(
                        "Playwright captured event from %s (%d rows)",
                        response.url, len(rows),
                    )
                elif "meetingsToday" in body and home_data is None:
                    home_data = body
                    logger.debug("Playwright captured racinghome from %s", response.url)

            page.on("response", _handle_response)
            page.goto(_HORSE_RACING_URL)
            page.wait_for_timeout(8000)

            # If racinghome was captured but no event details yet, fetch them
            if not captured_rows and home_data is not None:
                event_ids = _extract_event_ids(home_data)
                for eid in event_ids[:15]:
                    try:
                        event_data = page.context.request.get(
                            _EVENT_URL_TPL.format(event_id=eid),
                            headers=_API_HEADERS,
                        ).json()
                        captured_rows.extend(_parse_event(event_data))
                    except Exception as exc:
                        logger.debug("Playwright: failed to fetch event %s: %s", eid, exc)

            browser.close()

        if not captured_rows:
            raise ScraperError(
                "PlaywrightFallback: no race data captured in XHR responses",
                reason="empty_card",
            )
        return captured_rows


class LxmlFallback:
    def __init__(
        self,
        throttler: "Throttler" = None,
        proxy_rotator=None,
    ):
        self._throttler = throttler or Throttler()
        self._rotator = proxy_rotator or get_proxy_manager()

    def fetch(self) -> list:
        import ssl

        logger.warning(
            "JSON unavailable — lxml HTML fallback active (partial data expected)"
        )
        self._throttler.acquire()
        try:
            proxy = self._rotator.next(required=True)
        except ProxyUnavailableError as exc:
            source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
            raise ScraperError(
                "LxmlFallback: proxy gateway unavailable", reason="proxy_unavailable"
            ) from exc
        mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
        try:
            with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
                resp = client.get(_HORSE_RACING_URL)
            resp.raise_for_status()
            self._rotator.report_success(proxy)
        except httpx.TimeoutException:
            self._rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="timeout")
            raise
        except ssl.SSLError:
            self._rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="tls_error")
            raise
        except httpx.HTTPStatusError as exc:
            self._rotator.report_failure(proxy)
            reason = "bot_detected" if exc.response.status_code == 403 else "other"
            source_health.record_failure(_SOURCE, reason=reason)
            raise
        except Exception:
            self._rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="other")
            raise
        tree = lxml_html.fromstring(resp.content)

        rows = []
        race_containers = tree.xpath(
            '//*[contains(@class,"race-card") or contains(@class,"race-event")]'
            '[@data-primary-market-id and '
            'translate(@data-market-type,"win","WIN")="WIN"]'
        )
        for container in race_containers:
            venue_els = container.xpath(
                './/*[contains(@class,"race-title") or contains(@class,"venue")]/text()'
            )
            venue = venue_els[0].strip() if venue_els else ""
            time_els = container.xpath(
                './/*[contains(@class,"race-time") or contains(@class,"start-time")]/text()'
            )
            race_time = time_els[0].strip() if time_els else ""
            race_id = container.get(
                "data-race-id", container.get("data-event-id", "")
            )
            market_id = container.get("data-primary-market-id", "")
            market_name = container.get("data-market-name", "To win")

            runner_els = container.xpath(
                './/*[contains(@class,"runner") or contains(@class,"selection")]'
            )
            for runner in runner_els:
                name_els = runner.xpath(
                    './/*[contains(@class,"horse-name") or contains(@class,"runner-name")]/text()'
                )
                horse_name = name_els[0].strip() if name_els else ""
                odds_els = runner.xpath(
                    './/*[contains(@class,"odds") or contains(@class,"price")]/text()'
                )
                raw_odds = odds_els[0].strip() if odds_els else ""
                try:
                    odds_decimal = float(raw_odds)
                except (ValueError, TypeError):
                    odds_decimal = float("nan")

                if not horse_name:
                    continue
                rows.append(
                    {
                        "race_id": race_id,
                        "race_time": race_time,
                        "venue": venue,
                        "market_type": "WIN",
                        "market_id": market_id,
                        "market_name": market_name,
                        "selection_id": runner.get("data-selection-id", ""),
                        "ew_places": None,
                        "ew_reduction": None,
                        "ew_margin": float("nan"),
                        "horse_name": horse_name,
                        "odds_decimal": odds_decimal,
                        "sp": float("nan"),
                        "is_low_odds": _is_low(odds_decimal),
                        "currency": currency_for_venue(venue),
                    }
                )

        if not rows:
            raise ScraperError("LxmlFallback: DOM yielded zero races", reason="empty_card")
        return rows


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
# Helpers
# ---------------------------------------------------------------------------


def _find_race_data(obj, key: str):
    """Recursively search obj for a non-empty list under key. Returns list or None."""
    if isinstance(obj, dict):
        if key in obj and isinstance(obj[key], list) and obj[key]:
            return obj[key]
        for v in obj.values():
            result = _find_race_data(v, key)
            if result is not None:
                return result
    return None


def _extract_next_data(html_text: str) -> dict:
    """Extract and validate JSON from <script id="__NEXT_DATA__"> tag."""
    tree = lxml_html.fromstring(html_text)
    scripts = tree.xpath('//script[@id="__NEXT_DATA__"]/text()')
    if not scripts:
        raise BotDetectedError("__NEXT_DATA__ script tag not found", reason="challenge")
    try:
        data = json.loads(scripts[0])
    except json.JSONDecodeError as exc:
        raise BotDetectedError(f"__NEXT_DATA__ JSON malformed: {exc}", reason="invalid_json") from exc
    for key in _RACE_DATA_KEYS:
        if _find_race_data(data, key) is not None:
            return data
    raise BotDetectedError(
        "__NEXT_DATA__ contains no recognisable race data", reason="parser_rejected"
    )


def _extract_event_ids(home_data: dict) -> list:
    """Walk racinghome meetingsToday tree and collect event IDs for the
    configured countries (UK & Ireland by default). Foreign country subtrees are
    skipped; an empty ``_LSB_COUNTRIES`` keeps every country."""
    ids: list = []
    for meeting in home_data.get("meetingsToday") or []:
        _walk_node_for_events(meeting, ids)
    return ids


def _country_allowed(node: dict) -> bool:
    """A ``type: "Country"`` node is kept iff its name is in the configured set."""
    if not _LSB_COUNTRIES:
        return True
    return str(node.get("name", "")).strip().lower() in _LSB_COUNTRIES


def _walk_node_for_events(node: dict, acc: list) -> None:
    # Prune an entire foreign country subtree before descending into its venues.
    if str(node.get("type", "")).lower() == "country" and not _country_allowed(node):
        return
    for ev in node.get("events") or []:
        eid = ev.get("id", "")
        if eid and eid not in acc:
            acc.append(eid)
    childs = node.get("childs") or {}
    if isinstance(childs, dict):
        for child in childs.values():
            _walk_node_for_events(child, acc)
    elif isinstance(childs, list):
        for child in childs:
            _walk_node_for_events(child, acc)


def _parse_race_time(raw: str) -> str:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return to_local(dt).isoformat()


def _ew_margin(win_odds_list, places, reduction) -> float:
    """Place-market overround: sum(1/place_odds) - places. NaN if no terms."""
    import math as _m
    if not places or reduction is None:
        return float("nan")
    total, count = 0.0, 0
    for o in win_odds_list:
        if o is None or _m.isnan(o) or o <= 1.0:
            continue
        po = 1.0 + (o - 1.0) * reduction
        if po > 0:
            total += 1.0 / po
            count += 1
    return (total - places) if count else float("nan")


def _is_low(odds) -> bool:
    return bool(odds == odds and odds < LOW_ODDS_THRESHOLD)


def _parse_event(event_data: dict) -> list:
    """
    Parse a single event API response (currentEvent[]) into flat row dicts.
    Field names confirmed via DevTools 2026-06-12.
    """
    events = event_data.get("currentEvent") or []
    if not events:
        return []
    event = events[0]

    race_id = event.get("id", "")
    raw_time = event.get("startTime", "")  # "YYYY-MM-DD HH:MM:SS" UTC
    try:
        race_time = _parse_race_time(raw_time.replace(" ", "T") + "Z") if raw_time else ""
    except (ValueError, TypeError):
        race_time = raw_time or ""

    racing = event.get("racingData") or {}
    venue = racing.get("racetrackName") or event.get("name", "").split(":")[0].strip()

    primary_markets = []
    for market in event.get("markets") or []:
        group_ids = {
            gr.get("marketGroupId") for gr in (market.get("groupRanks") or [])
        }
        market_name = str(market.get("name") or "").strip().lower()
        market_type_id = str(market.get("type") or "").strip()
        if (
            _PRIMARY_WIN_GROUP_ID in group_ids
            and market_type_id in _PRIMARY_WIN_MARKET_TYPES
            and market_name in _PRIMARY_WIN_MARKET_NAMES
        ):
            primary_markets.append(market)
        elif _PRIMARY_WIN_GROUP_ID in group_ids:
            logger.warning(
                "LivescoreBet race %s: quarantined group %s with unknown "
                "primary market type/name (%r/%r)",
                race_id, _PRIMARY_WIN_GROUP_ID, market_type_id, market.get("name"),
            )

    if not primary_markets:
        logger.warning("LivescoreBet race %s: no confirmed primary WIN market", race_id)
        return []

    rows = []
    for market in primary_markets:

        ew = market.get("eachWay") or {}
        ew_place_str = ew.get("place")
        ew_odds_str = ew.get("odds")
        try:
            ew_places = int(ew_place_str) if ew_place_str else None
        except (ValueError, TypeError):
            ew_places = None
        try:
            ew_reduction = round(1.0 / int(ew_odds_str), 6) if ew_odds_str else None
        except (ValueError, TypeError, ZeroDivisionError):
            ew_reduction = None

        live_sel = [
            s for s in (market.get("selections") or [])
            if s.get("kind") != "NR" and (s.get("odds") or 0) >= 0
        ]
        win_odds = [
            float(s.get("odds")) if s.get("odds") is not None else float("nan")
            for s in live_sel
        ]
        ew_margin = _ew_margin(win_odds, ew_places, ew_reduction)
        venue_currency = currency_for_venue(venue)

        for sel, odds_decimal in zip(live_sel, win_odds):
            horse_name = sel.get("name", "")
            if not horse_name:
                continue
            rows.append(
                {
                    "race_id": race_id,
                    "race_time": race_time,
                    "venue": venue,
                    "market_type": "WIN",
                    "market_id": str(market.get("id") or ""),
                    "market_name": str(market.get("name") or ""),
                    "selection_id": str(sel.get("id") or ""),
                    "ew_places": ew_places,
                    "ew_reduction": ew_reduction,
                    "ew_margin": ew_margin,
                    "horse_name": horse_name,
                    "odds_decimal": odds_decimal,
                    "sp": float("nan"),
                    "is_low_odds": _is_low(odds_decimal),
                    "currency": venue_currency,
                }
            )
    annotated = annotate_primary_win_rows(
        rows, primary_market_count=len(primary_markets)
    )
    if annotated and annotated[0]["validation_status"] == INVALID:
        logger.error(
            "LivescoreBet race %s failed closed: %s",
            race_id, annotated[0]["validation_reasons"],
        )
    return annotated


def _parse_races(raw: dict) -> list:
    """
    Normalise raw JSON (legacy __NEXT_DATA__ format) -> flat list of row dicts.
    Kept for Playwright fallback and backwards-compat. For gateway API responses
    use _parse_event() instead.
    """
    rows = []
    races_data = None
    for key in _RACE_DATA_KEYS:
        races_data = _find_race_data(raw, key)
        if races_data is not None:
            break
    if not races_data:
        return rows

    for race in races_data:
        race_id = str(race.get("id", race.get("raceId", "")))
        raw_time = race.get("startTime", race.get("raceTime", race.get("time", "")))
        try:
            race_time = _parse_race_time(raw_time) if raw_time else ""
        except (ValueError, TypeError):
            race_time = raw_time or ""
        venue = race.get("venue", race.get("course", race.get("name", "")))

        markets = race.get("markets", race.get("outcomes", []))

        primary_markets = [
            market for market in markets
            if str(market.get("type", market.get("marketType", ""))).upper() == "WIN"
        ]
        race_rows = []
        for market in primary_markets:
            ew = market.get("eachWayTerms", market.get("eachWay"))
            ew_places = ew.get("places", ew.get("numberOfPlaces")) if ew else None
            ew_reduction = ew.get("reduction", ew.get("reducedOdds")) if ew else None

            runners = market.get("runners", market.get("selections", []))
            parsed = []
            for runner in runners:
                horse_name = runner.get(
                    "name", runner.get("horseName", runner.get("selectionName", ""))
                )
                odds_raw = runner.get(
                    "price", runner.get("decimal", runner.get("oddsDecimal"))
                )
                sp_raw = runner.get("sp", runner.get("startingPrice"))
                parsed.append((
                    horse_name,
                    float(odds_raw) if odds_raw is not None else float("nan"),
                    float(sp_raw) if sp_raw is not None else float("nan"),
                ))

            ew_margin = _ew_margin([o for _, o, _ in parsed], ew_places, ew_reduction)
            venue_currency = currency_for_venue(venue)
            for runner, (horse_name, odds_decimal, sp) in zip(runners, parsed):
                race_rows.append(
                    {
                        "race_id": race_id,
                        "race_time": race_time,
                        "venue": venue,
                        "market_type": "WIN",
                        "market_id": str(market.get("id", market.get("marketId", ""))),
                        "market_name": str(market.get("name", market.get("marketName", "WIN"))),
                        "selection_id": str(
                            runner.get("id", runner.get("selectionId", ""))
                        ),
                        "ew_places": ew_places,
                        "ew_reduction": ew_reduction,
                        "ew_margin": ew_margin,
                        "horse_name": horse_name,
                        "odds_decimal": odds_decimal,
                        "sp": sp,
                        "is_low_odds": _is_low(odds_decimal),
                        "currency": venue_currency,
                    }
                )
        rows.extend(annotate_primary_win_rows(
            race_rows, primary_market_count=len(primary_markets)
        ))
    return rows


_PARQUET_COLUMNS = [
    "fetched_at", "source", "race_id", "race_time", "venue", "market_type",
    "market_id", "market_name", "selection_id",
    "ew_places", "ew_reduction", "ew_margin", "horse_name", "odds_decimal",
    "sp", "is_low_odds", "currency", "validation_status",
    "validation_reasons", "field_size", "booksum",
]


def _rows_to_df(rows: list, fetched_at: datetime) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "fetched_at", fetched_at.astimezone(timezone.utc))
    df.insert(1, "source", "livescorebet")
    # Tolerate older/partial rows (cache, mocks) lacking the newer columns.
    for col, default in (
        ("ew_margin", float("nan")), ("is_low_odds", False), ("currency", "EUR"),
    ):
        if col not in df.columns:
            df[col] = default
    df = validate_live_dataframe(df)
    if df.empty:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["race_time"] = pd.to_datetime(df["race_time"], utc=True, errors="coerce")
    df["ew_places"] = pd.array(df["ew_places"].tolist(), dtype="Int64")
    df["ew_reduction"] = pd.array(df["ew_reduction"].tolist(), dtype="Float64")
    df["ew_margin"] = pd.array(df["ew_margin"].tolist(), dtype="Float64")
    df["is_low_odds"] = df["is_low_odds"].astype(bool)
    return df[_PARQUET_COLUMNS]


def _merge_parquet(df: pd.DataFrame, path: str = _PARQUET_PATH) -> None:
    """Read-merge-write: drop this source's old rows, append fresh, write back atomically."""
    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            existing = existing[existing["source"] != "livescorebet"]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not merge parquet (%s) — overwriting", exc)
    parquet_store.write_parquet(df, path)


def _fetch_events_concurrent(client: "LivescoreBetClient", event_ids: list) -> list:
    """Fetch per-event detail concurrently and flatten to rows.

    Each ``client.get_event()`` rotates to a fresh proxy IP and passes through
    the central per-domain rate limiter, so the thread pool only overlaps network
    latency — it never exceeds the configured rps/max_concurrent for the gateway.
    A single event's failure is logged and skipped (mirrors the old serial loop).
    Row order is irrelevant (rows are regrouped by race downstream).
    """
    def _one(eid) -> list:
        try:
            return _parse_event(client.get_event(eid))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tier 1: failed to fetch event %s: %s", eid, exc)
            return []

    workers = min(_EVENT_WORKERS, len(event_ids))
    if workers <= 1:
        rows: list = []
        for eid in event_ids:
            rows.extend(_one(eid))
        return rows

    from concurrent.futures import ThreadPoolExecutor
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ev_rows in pool.map(_one, event_ids):
            rows.extend(ev_rows)
    return rows


def scrape(force: bool = False) -> pd.DataFrame:
    """
    Fetch live LivescoreBet horse racing odds.

    Returns a pd.DataFrame with the live_odds schema. Raises ScraperError if
    all three fetch tiers are exhausted.
    """
    cache = CacheManager(path=_CACHE_PATH)

    if not force and cache.is_fresh():
        logger.debug("Cache hit — returning cached data")
        cached = cache.read()
        rows = cached.get("rows", [])
        return _rows_to_df(rows, datetime.fromisoformat(cached["fetched_at"]))

    # No throttler injection in production: the client falls through to the
    # central rate limiter (get_rate_limiter), which enforces the per-domain
    # rps configured in config.yaml. Tests inject Throttler(rate=100.0) directly.
    client = LivescoreBetClient()
    playwright_fb = PlaywrightFallback()
    lxml_fb = LxmlFallback()

    rows: Optional[list] = None

    # Tier 1 — direct gateway API (racinghome + per-event detail)
    try:
        logger.debug("Tier 1: gateway API")
        home_data = client.get_racing_home()
        event_ids = _extract_event_ids(home_data)
        logger.debug("Tier 1: found %d events", len(event_ids))
        all_rows = _fetch_events_concurrent(client, event_ids)
        if not all_rows:
            raise BotDetectedError("Tier 1 API returned no event rows", reason="empty_card")
        rows = all_rows
        logger.debug("Tier 1 succeeded: %d rows", len(rows))
    except BotDetectedError as exc:
        logger.warning("Tier 1 failed (%s) — trying Playwright", exc)
        source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))

        # Tier 2 — Playwright XHR interception
        try:
            logger.debug("Tier 2: Playwright XHR interception")
            rows = playwright_fb.fetch()
            logger.debug("Tier 2 succeeded: %d rows", len(rows))
        except ScraperError as exc2:
            logger.warning("Tier 2 failed (%s) — trying Selenium", exc2)
            source_health.record_failure(_SOURCE, reason=getattr(exc2, "reason", "other"))

            # Tier 2.5 — Selenium XHR interception
            try:
                from scraper._selenium_fallback import SeleniumFallback
                logger.debug("Tier 2.5: Selenium XHR interception")
                rows = SeleniumFallback(
                    url=_HORSE_RACING_URL,
                    xhr_url_predicate=lambda u: "gateway-ie.livescorebet.com" in u,
                    parse_xhr=lambda b: _parse_event(b) if "currentEvent" in b else [],
                ).fetch()
                logger.debug("Tier 2.5 succeeded: %d rows", len(rows))
            except Exception as exc25:  # noqa: BLE001
                logger.warning("Tier 2.5 (Selenium) failed (%s) — trying lxml", exc25)
                source_health.record_failure(_SOURCE, reason="other")

                # Tier 3 — lxml HTML (Cloudflare-protected page — proxy required)
                try:
                    logger.debug("Tier 3: lxml HTML fallback")
                    rows = lxml_fb.fetch()
                    logger.debug("Tier 3 succeeded: %d rows", len(rows))
                except ScraperError as exc3:
                    final_reason = getattr(exc3, "reason", "other")
                    raise ScraperError(
                        "All three fetch tiers failed", reason=final_reason
                    ) from exc3

    fetched_at = datetime.now(tz=TZ)
    df = _rows_to_df(rows, fetched_at)

    _merge_parquet(df, _PARQUET_PATH)
    logger.debug("Merged %d livescorebet rows into %s", len(df), _PARQUET_PATH)

    # Write cache
    cache.write({"fetched_at": fetched_at.isoformat(), "rows": rows})

    race_count = df["race_id"].nunique() if "race_id" in df.columns and not df.empty else 0
    source_health.record_success(_SOURCE, rows=len(df), races=int(race_count))

    return df
