"""Timeform fetch ladder: httpx(+proxy) -> browser -> selenium -> raise.
WAF/paywall detection routes to the next tier; results section stays out of scope."""
import re
import time
from typing import Callable, Optional

import httpx

from utils.logger import get_logger
from utils.proxy_manager import get_proxy_manager
from utils.rate_limiter import get_rate_limiter

logger = get_logger(__name__)


class TimeformError(RuntimeError):
    pass


_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36")
_WAF_RE = re.compile(r"Azure WAF|/\.azwaf/|cf-browser-verification", re.I)


def _looks_blocked(html: str) -> bool:
    return bool(_WAF_RE.search(html or "")) or "<html" not in (html or "").lower()


class TimeformClient:
    def __init__(self, cfg: dict, proxy_rotator=None,
                 browser_fetch: Optional[Callable[[str], str]] = None,
                 selenium_fetch: Optional[Callable[[str], str]] = None,
                 max_retries: int = 3):
        self._cfg = cfg or {}
        self._rotator = proxy_rotator or get_proxy_manager()
        self._browser_fetch = browser_fetch
        self._selenium_fetch = selenium_fetch
        self._max_retries = max_retries
        self._delay = float(self._cfg.get("request_delay", 1.5) or 0)

    def _headers(self) -> dict:
        h = {"User-Agent": _UA,
             "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
             "Accept-Language": "en-GB,en;q=0.9"}
        cookie = self._cfg.get("session_cookie")
        if cookie:
            h["Cookie"] = cookie
        return h

    def _httpx_get(self, url: str) -> Optional[str]:
        delay = 1
        for attempt in range(self._max_retries):
            proxy = self._rotator.next()
            kwargs = {"headers": self._headers(), "timeout": 25, "follow_redirects": True}
            if proxy:
                kwargs["proxy"] = proxy
            try:
                with httpx.Client(**kwargs) as client:
                    resp = client.get(url)
                if resp.status_code == 200 and not _looks_blocked(resp.text):
                    self._rotator.report_success(proxy)
                    return resp.text
                logger.info("timeform httpx tier blocked (%s) on %s", resp.status_code, url)
                self._rotator.report_failure(proxy)
                return None
            except httpx.HTTPError as exc:
                self._rotator.report_failure(proxy)
                logger.warning("timeform httpx error %s (attempt %d)", exc, attempt + 1)
                time.sleep(delay)
                delay *= 2
        return None

    def get_html(self, url: str) -> str:
        # Central rate limiter enforces the configured rps for this domain.
        # _delay is kept as a fallback for domains not covered by rate_limits config.
        with get_rate_limiter().acquire_for_url(url):
            if self._delay:
                time.sleep(self._delay)
            body = self._httpx_get(url)
            if body:
                return body
            for tier in (self._browser_fetch, self._selenium_fetch):
                if tier is None:
                    continue
                try:
                    body = tier(url)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("timeform fallback tier failed: %s", exc)
                    continue
                if body and not _looks_blocked(body):
                    return body
            raise TimeformError(f"all tiers failed/blocked for {url}")
