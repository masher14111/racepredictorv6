"""Firecrawl REST backend — the last-resort fetch tier for Cloudflare-fronted books.

Tiers 1-4 (curl_cffi TLS impersonation -> httpx -> Playwright -> Selenium) all
run from this machine through the residential proxy pool. When Cloudflare has
flagged the whole pool, every one of them fails the same way and the scraper
falls back to a stale cache. Firecrawl fetches server-side from its own
infrastructure instead, so it is the one tier that is not affected by our exit
IPs being burned.

It is deliberately LAST: each page is a paid API credit, whereas tiers 1-4 are
free. Two guards keep a bad day from becoming an expensive one:

  * ``max_pages_per_run`` — a hard per-process page budget. Once spent, every
    further call raises ``FirecrawlError(reason="budget_exhausted")`` rather
    than silently billing on.
  * the tier only runs at all once the free tiers have been exhausted.

Two correctness details that are easy to get wrong:

  * ``maxAge=0`` is mandatory. Firecrawl's default is to reuse indexed content
    for up to two days; for odds that would mean silently scraping *stale
    prices*, which is worse than scraping nothing.
  * ``location.country`` must be IE/GB. Both books geo-tailor their cards, and a
    default US exit would return a different (or empty) market set.

The API key is read from the FIRECRAWL_API_KEY environment variable, falling
back to ``firecrawl.api_key`` in the git-ignored config.local.yaml. No key means
the tier reports itself unavailable and is skipped — never a hard failure.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import threading
from typing import Any

import httpx

from utils.config_loader import get_config
from utils.logger import get_logger
from utils.rate_limiter import get_rate_limiter

logger = get_logger(__name__)

_API_URL = "https://api.firecrawl.dev/v2/scrape"

# Browsers render a bare JSON document inside <pre>; Paddy Power's endpoint is a
# JSON API, so unwrap that before parsing.
_PRE_RE = re.compile(r"<pre[^>]*>(.*?)</pre>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


class FirecrawlError(RuntimeError):
    """Firecrawl tier failure. `reason` matches the scrapers' source_health vocab."""

    def __init__(self, message: str = "", reason: str = "other"):
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Config / credentials
# ---------------------------------------------------------------------------
def _cfg() -> dict:
    return get_config().get("firecrawl", {}) or {}


def api_key() -> str:
    """FIRECRAWL_API_KEY env var, else firecrawl.api_key from config.local.yaml."""
    return (os.environ.get("FIRECRAWL_API_KEY") or str(_cfg().get("api_key", "") or "")).strip()


def is_available() -> tuple[bool, str]:
    """(usable, human-readable reason). Never raises — callers use it to skip."""
    if not bool(_cfg().get("enabled", True)):
        return False, "disabled in config (firecrawl.enabled: false)"
    if not api_key():
        return False, "no FIRECRAWL_API_KEY set"
    if _budget_left() <= 0:
        return False, "page budget for this run is spent"
    return True, "ready"


# ---------------------------------------------------------------------------
# Per-process credit budget
# ---------------------------------------------------------------------------
_budget_lock = threading.Lock()
_pages_used = 0


def _budget_left() -> int:
    cap = int(_cfg().get("max_pages_per_run", 50))
    if cap <= 0:
        return 0
    with _budget_lock:
        return max(0, cap - _pages_used)


def _spend_page() -> int:
    """Claim one page from the budget. Returns the running total, or raises."""
    global _pages_used
    cap = int(_cfg().get("max_pages_per_run", 50))
    with _budget_lock:
        if cap <= 0 or _pages_used >= cap:
            raise FirecrawlError(
                f"Firecrawl page budget exhausted ({_pages_used}/{cap} this run)",
                reason="budget_exhausted",
            )
        _pages_used += 1
        return _pages_used


def pages_used() -> int:
    """Pages billed so far in this process — logged by the scrapers after a run."""
    with _budget_lock:
        return _pages_used


def reset_budget() -> None:
    """Test hook / long-lived-process hook to start a fresh run budget."""
    global _pages_used
    with _budget_lock:
        _pages_used = 0


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _post(url: str, want_raw: bool) -> dict:
    """One Firecrawl scrape call. Returns the `data` object."""
    key = api_key()
    if not key:
        raise FirecrawlError("no FIRECRAWL_API_KEY set", reason="not_configured")

    cfg = _cfg()
    fmt = "rawHtml" if want_raw else "html"
    timeout_ms = int(cfg.get("timeout_ms", 60000))
    payload: dict[str, Any] = {
        "url": url,
        "formats": [fmt],
        # Never reuse Firecrawl's index for live odds — always refetch.
        "maxAge": 0,
        "onlyMainContent": False,
        "blockAds": False,
        "proxy": str(cfg.get("proxy", "auto")),
        "timeout": timeout_ms,
        "location": {
            "country": str(cfg.get("country", "IE")),
            "languages": [str(cfg.get("language", "en-GB"))],
        },
    }
    wait_for = int(cfg.get("wait_for_ms", 0))
    if wait_for > 0:
        payload["waitFor"] = wait_for

    n = _spend_page()
    logger.info("Firecrawl: fetching %s (page %d of this run's budget)", url, n)

    # Rate-limit against Firecrawl's own domain, not the bookmaker's — the
    # bookmaker never sees this request.
    with get_rate_limiter().acquire_for_url(_API_URL):
        try:
            resp = httpx.post(
                _API_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                timeout=timeout_ms / 1000 + 30,
            )
        except httpx.TimeoutException as exc:
            raise FirecrawlError(f"Firecrawl timed out on {url}", reason="timeout") from exc
        except Exception as exc:  # noqa: BLE001 — network failure reaching the API
            raise FirecrawlError(
                f"Firecrawl request failed on {url}: {exc}", reason="other"
            ) from exc

    if resp.status_code == 402:
        raise FirecrawlError("Firecrawl credits exhausted (HTTP 402)", reason="payment_required")
    if resp.status_code == 429:
        raise FirecrawlError("Firecrawl rate limit hit (HTTP 429)", reason="rate_limited")
    if resp.status_code in (401, 403):
        raise FirecrawlError(
            f"Firecrawl rejected the API key (HTTP {resp.status_code})", reason="not_configured"
        )
    if resp.status_code >= 400:
        raise FirecrawlError(f"Firecrawl HTTP {resp.status_code} on {url}", reason="other")

    try:
        body = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise FirecrawlError(
            f"Firecrawl returned non-JSON for {url}", reason="invalid_json"
        ) from exc

    if not body.get("success"):
        raise FirecrawlError(
            f"Firecrawl reported failure on {url}: {body.get('error', 'unknown')}",
            reason="other",
        )

    data = body.get("data") or {}
    upstream = (data.get("metadata") or {}).get("statusCode")
    if isinstance(upstream, int) and upstream >= 400:
        # Firecrawl succeeded in *fetching*, but the book returned 403/challenge.
        raise FirecrawlError(f"Upstream returned HTTP {upstream} for {url}", reason="bot_detected")
    return data


def get_html(url: str) -> str:
    """Fetch `url` and return its HTML. Signature matches the other tiers' fetchers."""
    data = _post(url, want_raw=False)
    text = data.get("html") or data.get("rawHtml") or ""
    if not text.strip():
        raise FirecrawlError(f"Firecrawl returned empty HTML for {url}", reason="empty_card")
    return text


def get_json(url: str) -> dict:
    """Fetch a JSON endpoint through Firecrawl and decode it.

    Firecrawl renders in a browser, so a bare JSON document arrives wrapped in
    the browser's <pre> viewer. Try a straight decode first, then unwrap.
    """
    data = _post(url, want_raw=True)
    text = (data.get("rawHtml") or data.get("html") or "").strip()
    if not text:
        raise FirecrawlError(f"Firecrawl returned empty body for {url}", reason="empty_card")

    try:
        return json.loads(text)
    except ValueError:
        pass

    match = _PRE_RE.search(text)
    candidate = match.group(1) if match else _TAG_RE.sub("", text)
    candidate = _html.unescape(candidate).strip()
    try:
        return json.loads(candidate)
    except ValueError as exc:
        raise FirecrawlError(
            f"Firecrawl body for {url} was not JSON (likely a challenge page)",
            reason="invalid_json",
        ) from exc
