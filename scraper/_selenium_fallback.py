"""Reusable headless-Chrome (Selenium) fallback for scrapers.

Captures XHR/fetch JSON via Chrome performance logging + CDP, applies a
caller-supplied predicate + parser, and optionally falls back to DOM parsing.
Holds no site-specific logic.
"""
import json
from typing import Callable, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


class ScraperError(Exception):
    """Raised when the Selenium tier captures no usable rows."""


def _rows_from_bodies(bodies, xhr_url_predicate, parse_xhr) -> list:
    """Turn (url, json_body) pairs into rows using the injected parser.

    Skips non-matching URLs and swallows per-body parser errors.
    """
    rows: list = []
    for url, body in bodies:
        if not xhr_url_predicate(url):
            continue
        try:
            rows.extend(parse_xhr(body))
        except Exception as exc:  # noqa: BLE001 - one bad body shouldn't abort
            logger.debug("Selenium parse_xhr failed for %s: %s", url, exc)
    return rows


class SeleniumFallback:
    def __init__(
        self,
        url: str,
        xhr_url_predicate: Callable[[str], bool],
        parse_xhr: Callable[[dict], list],
        dom_parse: Optional[Callable[[str], list]] = None,
        wait_ms: int = 8000,
        proxy_rotator=None,
    ):
        self._url = url
        self._predicate = xhr_url_predicate
        self._parse_xhr = parse_xhr
        self._dom_parse = dom_parse
        self._wait_ms = wait_ms
        self._rotator = proxy_rotator

    def _build_driver(self):
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        opts = Options()
        opts.add_argument("--headless=new")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--no-sandbox")
        opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
        if self._rotator is not None:
            proxy = self._rotator.next()
            if proxy:
                opts.add_argument(f"--proxy-server={proxy}")
        return webdriver.Chrome(options=opts)

    def _capture_bodies(self) -> list:
        """Drive Chrome, return list of (url, json_body) for response bodies."""
        driver = self._build_driver()
        bodies: list = []
        try:
            driver.get(self._url)
            import time as _t
            _t.sleep(self._wait_ms / 1000.0)
            for entry in driver.get_log("performance"):
                try:
                    msg = json.loads(entry["message"])["message"]
                except (KeyError, json.JSONDecodeError):
                    continue
                if msg.get("method") != "Network.responseReceived":
                    continue
                params = msg.get("params", {})
                url = params.get("response", {}).get("url", "")
                request_id = params.get("requestId")
                if not request_id or not self._predicate(url):
                    continue
                try:
                    body = driver.execute_cdp_cmd(
                        "Network.getResponseBody", {"requestId": request_id}
                    )
                    bodies.append((url, json.loads(body.get("body", "") or "null")))
                except Exception as exc:  # noqa: BLE001
                    logger.debug("CDP getResponseBody failed for %s: %s", url, exc)
        finally:
            driver.quit()
        return bodies

    def _dom_rows(self) -> list:
        if self._dom_parse is None:
            return []
        driver = self._build_driver()
        try:
            driver.get(self._url)
            import time as _t
            _t.sleep(self._wait_ms / 1000.0)
            return self._dom_parse(driver.page_source)
        finally:
            driver.quit()

    def fetch(self) -> list:
        logger.warning("Selenium fallback active for %s", self._url)
        bodies = self._capture_bodies()
        rows = _rows_from_bodies(bodies, self._predicate, self._parse_xhr)
        if not rows:
            rows = self._dom_rows()
        if not rows:
            raise ScraperError(f"SeleniumFallback: no rows captured for {self._url}")
        return rows
