import json
import os
import time
from datetime import datetime
from typing import Optional

import httpx
from playwright.sync_api import sync_playwright

from utils import source_health
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.proxy_manager import ProxyUnavailableError, get_proxy_manager, redact_proxy_url
from utils.rate_limiter import get_rate_limiter
from utils.timezone import TZ, to_local
from utils.market_validation import annotate_primary_win_rows

logger = get_logger(__name__)

# Paddy Power's content-managed-page API is confirmed Cloudflare-fronted (see
# BotDetectedError below) — both fetch tiers require the proxy gateway; a
# gateway failure must never silently degrade to a direct connection (req 2).
_SOURCE = "paddy_power"

# Confirmed non-runner placeholders in Paddy Power's WIN feed. These stable,
# global selection IDs represent unnamed favourite bets rather than declared
# horses and are present in many otherwise-valid primary markets.
_NON_RUNNER_SELECTION_IDS = frozenset({"10518227", "10518230"})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "paddy_power.json"
)

_cfg = get_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
_PP_CFG: dict = _cfg.get("paddypower", {})

# Per-scrape retry budget for the curl_cffi tier. Paddy is a SINGLE JSON API
# call, so the only failure mode is a transient 403 / network blip on the
# current proxy IP — each retry rotates to a fresh IP before falling through to
# the slow browser tiers.
_CURL_ATTEMPTS: int = int(_PP_CFG.get("curl_attempts", 3))

# ---------------------------------------------------------------------------
# Endpoints & Headers
# ---------------------------------------------------------------------------
_HORSE_RACING_PAGE = "https://www.paddypower.com/horse-racing"
_PP_URL = (
    "https://apisms.paddypower.com/smspp/content-managed-page/v7"
    "?_ak=vsd0Rm5ph2sS2uaK"
    "&betexRegion=IRL&capiJurisdiction=intl&cardsToFetch=21149"
    "&countryCode=IE&currencyCode=EUR&eventTypeId=7&exchangeLocale=en_GB"
    "&includeEuromillionsWithoutLogin=false&includeMarketBlurbs=true"
    "&includePrices=true&includeRaceCards=true&language=en"
    "&layoutFetchedCardsOnly=true&loggedIn=false&nextRacesMarketsLimit=1"
    "&page=SPORT&priceHistory=3&regionCode=IRE&requestCountryCode=IE"
    "&staticCardsIncluded=SEO_CONTENT_SUMMARY&timezone=Europe%2FDublin"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IE,en;q=0.9",
    "Referer": _HORSE_RACING_PAGE,
}

# Headers minus User-Agent: curl_cffi's impersonate="chrome" sets the UA +
# sec-ch-ua client hints to match the spoofed TLS fingerprint; a hand-set UA
# would only desync them. Keep the contextual headers.
_CURL_HEADERS = {k: v for k, v in _HEADERS.items() if k.lower() != "user-agent"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BotDetectedError(Exception):
    """Raised when Paddy Power returns a 403 or Cloudflare HTML challenge."""

    def __init__(self, message: str = "", reason: str = "bot_detected"):
        super().__init__(message)
        self.reason = reason


class ScraperError(Exception):
    """Raised when all fetch paths are exhausted."""

    def __init__(self, message: str = "", reason: str = "other"):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------
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
# PaddyPowerClient
# ---------------------------------------------------------------------------
class PaddyPowerClient:
    def __init__(
        self,
        proxy_rotator=None,
        max_retries: int = 3,
    ):
        self._rotator = proxy_rotator or get_proxy_manager()
        self._max_retries = max_retries

    def get(self, url: str, params: dict = None) -> dict:
        delay = 1
        last_exc: Optional[Exception] = None
        last_reason = "other"

        for attempt in range(self._max_retries):
            try:
                proxy = self._rotator.next(required=True)
            except ProxyUnavailableError as exc:
                source_health.record_unavailable(_SOURCE, reason="proxy_unavailable")
                raise ScraperError(
                    f"Paddy Power: proxy gateway unavailable ({url})",
                    reason="proxy_unavailable",
                ) from exc
            mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
            logger.debug("GET %s attempt=%d proxy=%s", url, attempt + 1, redact_proxy_url(proxy))
            try:
                with get_rate_limiter().acquire_for_url(url):
                    with httpx.Client(
                        headers=_HEADERS, mounts=mounts, timeout=15
                    ) as client:
                        resp = client.get(url, params=params)

                if resp.status_code == 403:
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="bot_detected")
                    raise BotDetectedError(f"403 on {url}", reason="bot_detected")
                if "text/html" in resp.headers.get("content-type", ""):
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="challenge")
                    raise BotDetectedError(f"HTML response on {url}", reason="challenge")

                resp.raise_for_status()
                try:
                    data = resp.json()
                    self._rotator.report_success(proxy)
                    return data
                except json.JSONDecodeError as exc:
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="invalid_json")
                    raise BotDetectedError(
                        f"Non-JSON response on {url}: {exc}", reason="invalid_json"
                    ) from exc

            except BotDetectedError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                self._rotator.report_failure(proxy)
                last_reason = "timeout" if isinstance(exc, httpx.TimeoutException) else "other"
                source_health.record_failure(_SOURCE, reason=last_reason)
                last_exc = exc
                logger.debug("Transient error, retrying in %ds: %s", delay, exc)
                time.sleep(delay)
                delay *= 2
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:
                    self._rotator.report_failure(proxy)
                    last_reason = "other"
                    source_health.record_failure(_SOURCE, reason=last_reason)
                    last_exc = exc
                    logger.debug("5xx, retrying in %ds", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    self._rotator.report_failure(proxy)
                    source_health.record_failure(_SOURCE, reason="other")
                    raise ScraperError(
                        f"HTTP {exc.response.status_code} on {url}", reason="other"
                    ) from exc

        raise ScraperError(
            f"Failed after {self._max_retries} attempts on {url}", reason=last_reason
        ) from last_exc


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _parse_race_time(raw: str) -> str:
    """Parse ISO datetime string and return Europe/Dublin ISO string.

    Returns "" for empty/malformed input so a single market missing
    ``marketTime`` (e.g. in-play / SP-only) can't crash the whole parse.
    """
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return ""
    return to_local(dt).isoformat()


def _parse_pp_response(data: dict) -> list:
    """
    Build races list from content-managed-page/v7 cardsToFetch=21149 response.

    Structure: data.attachments.races[raceId].venue
               data.attachments.markets[marketId].{runners, marketType, ...}
    """
    att = data.get("attachments", {})
    races_meta = att.get("races", {})
    markets_dict = att.get("markets", {})

    by_race: dict[str, list[tuple[str, dict]]] = {}
    for market_key, market in markets_dict.items():
        race_id = str(market.get("raceId") or "")
        # The cardsToFetch=21149/nextRacesMarketsLimit=1 feed currently exposes
        # the primary race market as marketType=WIN. Unknown/other types are
        # skipped, never coerced to WIN.
        if race_id and str(market.get("marketType") or "").upper() == "WIN":
            by_race.setdefault(race_id, []).append((str(market_key), market))

    races = []
    for race_id, primary_markets in by_race.items():
        venue = races_meta.get(race_id, {}).get("venue", "")
        race_time = _parse_race_time(primary_markets[0][1].get("marketTime", ""))
        parsed_markets = []
        validation_rows = []

        for market_key, mkt in primary_markets:
            market_id = str(mkt.get("marketId") or market_key)
            market_name = str(mkt.get("marketName") or "")
            ew_available = mkt.get("eachwayAvailable", False)
            if ew_available:
                pf = mkt.get("placeFraction") or {}
                num = pf.get("numerator", 1)
                den = pf.get("denominator", 1)
                each_way_terms = {
                    "places": mkt.get("numberOfPlaces"),
                    "reduction": num / den if den else None,
                }
            else:
                each_way_terms = None

            selections = []
            for runner in mkt.get("runners", []):
                if runner.get("runnerStatus") == "REMOVED":
                    continue
                selection_id = str(
                    runner.get("selectionId", runner.get("runnerId", ""))
                )
                if selection_id in _NON_RUNNER_SELECTION_IDS:
                    continue
                odds = (
                    runner.get("winRunnerOdds", {})
                    .get("trueOdds", {})
                    .get("decimalOdds", {})
                    .get("decimalOdds")
                )
                selection = {
                    "selection_id": selection_id,
                    "horse_name": runner.get("runnerName", ""),
                    "odds_decimal": odds,
                    "sp": None,
                }
                selections.append(selection)
                validation_rows.append({
                    "race_id": race_id,
                    "market_type": "WIN",
                    "market_id": market_id,
                    "market_name": market_name,
                    **selection,
                })

            parsed_markets.append({
                "market_id": market_id,
                "market_name": market_name,
                "market_type": "WIN",
                "each_way_terms": each_way_terms,
                "selections": selections,
            })

        annotated = annotate_primary_win_rows(
            validation_rows, primary_market_count=len(primary_markets)
        )
        status = annotated[0]["validation_status"] if annotated else "INVALID"
        reasons = (
            annotated[0]["validation_reasons"]
            if annotated else '["implausible_field_size:0","extreme_booksum:none"]'
        )
        booksum = annotated[0]["booksum"] if annotated else None
        field_size = annotated[0]["field_size"] if annotated else 0
        for market in parsed_markets:
            market["validation_status"] = status
            market["validation_reasons"] = reasons
            for selection in market["selections"]:
                selection["validation_status"] = status
                selection["validation_reasons"] = reasons

        races.append({
            "race_id": race_id,
            "race_time": race_time,
            "venue": venue,
            "validation_status": status,
            "validation_reasons": reasons,
            "field_size": field_size,
            "booksum": booksum,
            "markets": parsed_markets,
        })
    return races


# ---------------------------------------------------------------------------
# ChromeFallback
# ---------------------------------------------------------------------------
class ChromeFallback:
    def fetch(self) -> dict:
        """
        Drive a real Chromium instance, intercept the content-managed-page
        cardsToFetch=21149 JSON response, and return it for parsing.
        """
        logger.warning("Bot detected — triggering Playwright Chrome fallback")
        captured: list = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def _handle_response(response) -> None:
                if (
                    "content-managed-page" not in response.url
                    or "cardsToFetch=21149" not in response.url
                ):
                    return
                try:
                    body = response.json()
                except Exception:
                    return
                if "attachments" in body:
                    captured.append(body)
                    logger.debug("Chrome captured PP response from %s", response.url)

            page.on("response", _handle_response)
            page.goto(_HORSE_RACING_PAGE)
            page.wait_for_timeout(8000)
            browser.close()

        if not captured:
            raise ScraperError("ChromeFallback: no API responses captured", reason="empty_card")

        return captured[0]


# ---------------------------------------------------------------------------
# Selenium fallback
# ---------------------------------------------------------------------------
def _selenium_fetch() -> dict:
    """Selenium tier: capture the content-managed-page JSON, return raw dict."""
    from scraper._selenium_fallback import SeleniumFallback

    fb = SeleniumFallback(
        url=_HORSE_RACING_PAGE,
        xhr_url_predicate=lambda u: "content-managed-page" in u
        and "cardsToFetch=21149" in u,
        parse_xhr=lambda b: [b] if "attachments" in b else [],
    )
    bodies = fb.fetch()
    if not bodies:
        raise ScraperError("Selenium PP: no content-managed-page captured", reason="empty_card")
    return bodies[0]


# ---------------------------------------------------------------------------
# curl_cffi tier (Chrome TLS impersonation) — primary Cloudflare bypass
# ---------------------------------------------------------------------------
def _curl_cffi_fetch(url: str = _PP_URL, attempts: int = _CURL_ATTEMPTS) -> dict:
    """Fetch the Paddy Power JSON API with a real Chrome TLS handshake.

    Plain httpx is TLS-fingerprinted by Cloudflare and returns a 403 / HTML
    challenge. curl_cffi's `impersonate="chrome"` reproduces Chrome's JA3
    fingerprint and pulls the same JSON the browser would — no Playwright/Selenium
    needed. Routed through the proxy rotator + central rate limiter. A 403 /
    transient error usually just means the current proxy IP is flagged, so each
    retry rotates to a fresh IP (up to `attempts`); only after the budget is
    exhausted does it raise so scrape() falls through to the legacy tiers.
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
                f"Paddy Power: proxy gateway unavailable ({url})",
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
            except Exception as exc:  # noqa: BLE001 — network blip: rotate IP and retry
                rotator.report_failure(proxy)
                source_health.record_failure(_SOURCE, reason="other")
                last_exc = exc
                continue
        if resp.status_code == 403:
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="bot_detected")
            last_exc = BotDetectedError(f"403 on {url}", reason="bot_detected")
            continue
        if "text/html" in resp.headers.get("content-type", ""):
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="challenge")
            last_exc = BotDetectedError(f"HTML response on {url}", reason="challenge")
            continue
        try:
            resp.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — 5xx etc.: rotate IP and retry
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="other")
            last_exc = exc
            continue
        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — any JSON decode failure = blocked/garbage
            rotator.report_failure(proxy)
            source_health.record_failure(_SOURCE, reason="invalid_json")
            last_exc = BotDetectedError(f"Non-JSON response on {url}: {exc}", reason="invalid_json")
            continue
        rotator.report_success(proxy)
        return data
    raise last_exc or BotDetectedError(f"curl_cffi exhausted retries on {url}", reason="bot_detected")


def _firecrawl_fetch(url: str = _PP_URL) -> dict:
    """Tier-5 fetch: pull the Paddy Power JSON API server-side via Firecrawl.

    Every tier above egresses through our residential proxy pool, so once
    Cloudflare flags the pool they all fail the same way. Firecrawl fetches from
    its own infrastructure. This is the only PAID tier — one page per call — so
    it sits last and the budget in firecrawl_fetch caps a runaway.

    Paddy Power's card is a single JSON endpoint, so this costs exactly 1 page.
    """
    from scraper import firecrawl_fetch

    ok, why = firecrawl_fetch.is_available()
    if not ok:
        raise ScraperError(f"Firecrawl tier skipped: {why}", reason="not_configured")

    logger.warning("Paddy Power: Firecrawl tier active (PAID — 1 page)")
    return firecrawl_fetch.get_json(url)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scrape(force: bool = False) -> dict:
    """
    Fetch live Paddy Power horse racing cards and odds.

    Returns a dict matching the cache schema. Raises ScraperError if all
    fetch paths are exhausted.
    """
    cache = CacheManager(path=_CACHE_PATH, ttl=SCRAPE_INTERVAL)

    if not force and cache.is_fresh():
        logger.debug("Cache hit — returning cached data")
        return cache.read()

    client = PaddyPowerClient()
    fallback = ChromeFallback()

    # Tier 1: curl_cffi (Chrome TLS impersonation) — primary Cloudflare bypass.
    try:
        raw = _curl_cffi_fetch(_PP_URL)
    except Exception as curl_exc:  # noqa: BLE001
        logger.warning("curl_cffi tier failed (%s) — trying httpx", curl_exc)
        source_health.record_failure(_SOURCE, reason=getattr(curl_exc, "reason", "other"))
        # Tier 2: plain httpx (usually 403s on Cloudflare).
        try:
            raw = client.get(_PP_URL)
        except BotDetectedError:
            logger.warning("Bot detected on httpx path — triggering ChromeFallback")
            try:
                raw = fallback.fetch()
            except Exception as exc:
                logger.warning("ChromeFallback failed (%s) — trying Selenium", exc)
                source_health.record_failure(_SOURCE, reason=getattr(exc, "reason", "other"))
                try:
                    raw = _selenium_fetch()
                except Exception as exc2:
                    logger.warning("Selenium failed (%s) — trying Firecrawl", exc2)
                    source_health.record_failure(_SOURCE, reason=getattr(exc2, "reason", "other"))
                    # Tier 5: Firecrawl (server-side, PAID) — last resort.
                    try:
                        raw = _firecrawl_fetch(_PP_URL)
                    except Exception as exc3:
                        source_health.record_failure(
                            _SOURCE, reason=getattr(exc3, "reason", "other")
                        )
                        raise ScraperError(
                            "curl_cffi, httpx, ChromeFallback, Selenium and Firecrawl all failed",
                            reason=getattr(exc3, "reason", "other"),
                        ) from exc3

    result = {
        "fetched_at": datetime.now(tz=TZ).isoformat(),
        "races": _parse_pp_response(raw),
    }
    cache.write(result)
    # rows was hard-coded to 0, so Paddy Power always reported "0 rows" in the
    # health line however much it had scraped. Count the priced selections.
    n_sel = sum(
        len(market.get("selections") or [])
        for race in result["races"]
        for market in (race.get("markets") or [])
    )
    source_health.record_success(_SOURCE, rows=n_sel, races=len(result["races"]))
    logger.debug("Scraped %d races, cache written to %s", len(result["races"]), cache.path)
    return result
