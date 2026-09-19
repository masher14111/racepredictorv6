# Paddy Power Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scraper/paddy_power.py` that fetches live horse racing cards and odds from Paddy Power's JSON API via `httpx`, with exponential backoff retry, proxy rotation, TTL-based caching, and a Playwright Chrome fallback when bot detection fires.

**Architecture:** Four focused components (`CacheManager`, `ProxyRotator`, `PaddyPowerClient`, `ChromeFallback`) live in a single module. A public `scrape()` function orchestrates them: check cache → httpx fetch → on bot-detection trigger Playwright fallback → parse → write cache → return. All config is read from `config.yaml` at import time following the same pattern as `utils/timezone.py`.

**Tech Stack:** Python, `httpx` (HTTP client), `playwright` (Chrome CDP fallback), `pyyaml`, `pytz`, `pytest`, `respx` (httpx mock), `pytest-mock`

---

## File Map

| Action | Path                                | Responsibility                                     |
| ------ | ----------------------------------- | -------------------------------------------------- |
| Create | `scraper/__init__.py`               | Package marker                                     |
| Create | `scraper/paddy_power.py`            | All scraper logic                                  |
| Create | `tests/__init__.py`                 | Test package marker                                |
| Create | `tests/scraper/__init__.py`         | Test subpackage marker                             |
| Create | `tests/scraper/test_paddy_power.py` | Full test suite                                    |
| Modify | `requirements.txt`                  | Add `playwright`, `pytest`, `respx`, `pytest-mock` |

---

## Task 1: Scaffolding & Dependencies

**Files:**

- Create: `scraper/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/scraper/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Create package markers**

Create `scraper/__init__.py` (empty file):

```python

```

Create `tests/__init__.py` (empty file):

```python

```

Create `tests/scraper/__init__.py` (empty file):

```python

```

- [ ] **Step 2: Add new dependencies to requirements.txt**

Append these lines to `requirements.txt`:

```
playwright
pytest
respx
pytest-mock
```

- [ ] **Step 3: Install dependencies**

Run:

```
pip install playwright respx pytest pytest-mock
playwright install chromium
```

Expected: all installs succeed, `playwright install chromium` downloads the Chromium binary.

- [ ] **Step 4: Commit**

```bash
git add scraper/__init__.py tests/__init__.py tests/scraper/__init__.py requirements.txt
git commit -m "chore: scaffold scraper package and add test/playwright deps"
```

---

## Task 2: Exceptions & Config Loading

**Files:**

- Create: `scraper/paddy_power.py` (initial skeleton)
- Create: `tests/scraper/test_paddy_power.py` (initial)

- [ ] **Step 1: Write failing tests for exceptions and config**

Create `tests/scraper/test_paddy_power.py`:

```python
import pytest
from scraper.paddy_power import BotDetectedError, ScraperError, SCRAPE_INTERVAL


def test_bot_detected_error_is_exception():
    with pytest.raises(BotDetectedError):
        raise BotDetectedError("test")


def test_scraper_error_is_exception():
    with pytest.raises(ScraperError):
        raise ScraperError("test")


def test_scrape_interval_is_int():
    assert isinstance(SCRAPE_INTERVAL, int)
    assert SCRAPE_INTERVAL > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -v
```

Expected: `ImportError` — `scraper.paddy_power` does not exist yet.

- [ ] **Step 3: Create scraper/paddy_power.py with exceptions and config**

Create `scraper/paddy_power.py`:

```python
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import yaml

from utils.logger import get_logger
from utils.timezone import TZ, to_local

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "paddy_power.json"
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
_PROXY_CFG: dict = _cfg.get("proxy_pool", {})

# ---------------------------------------------------------------------------
# Endpoints & Headers
# ---------------------------------------------------------------------------
_EVENTS_URL = "https://api.paddypower.com/api/v1/events"
_MARKETS_URL = "https://api.paddypower.com/api/v1/competitions/{race_id}/markets"
_HORSE_RACING_PAGE = "https://www.paddypower.com/horse-racing"

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


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BotDetectedError(Exception):
    """Raised when Paddy Power returns a 403 or Cloudflare HTML challenge."""


class ScraperError(Exception):
    """Raised when all fetch paths are exhausted."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```
pytest tests/scraper/test_paddy_power.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: add scraper exceptions and config loading"
```

---

## Task 3: CacheManager

**Files:**

- Modify: `scraper/paddy_power.py` (append `CacheManager`)
- Modify: `tests/scraper/test_paddy_power.py` (append cache tests)

- [ ] **Step 1: Write failing tests for CacheManager**

Append to `tests/scraper/test_paddy_power.py`:

```python
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone

from scraper.paddy_power import CacheManager


def _write_cache(path: str, age_seconds: int, data: dict = None) -> None:
    """Helper: write a cache file with fetched_at set to `age_seconds` ago."""
    fetched_at = (
        datetime.now(tz=timezone.utc) - timedelta(seconds=age_seconds)
    ).isoformat()
    payload = {"fetched_at": fetched_at, "races": [], **(data or {})}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def test_cache_miss_when_file_absent(tmp_path):
    cm = CacheManager(path=str(tmp_path / "pp.json"), ttl=3600)
    assert cm.is_fresh() is False
    assert cm.read() is None


def test_cache_fresh_within_ttl(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=100)
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is True


def test_cache_stale_beyond_ttl(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=4000)
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is False


def test_cache_read_returns_data(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=10, data={"races": [{"race_id": "1"}]})
    cm = CacheManager(path=path, ttl=3600)
    result = cm.read()
    assert result["races"][0]["race_id"] == "1"


def test_cache_write_creates_directories_and_file(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "pp.json")
    cm = CacheManager(path=path, ttl=3600)
    data = {"fetched_at": "2026-06-12T12:00:00+01:00", "races": []}
    cm.write(data)
    assert os.path.exists(path)
    with open(path) as f:
        saved = json.load(f)
    assert saved["races"] == []


def test_cache_stale_on_corrupt_json(tmp_path):
    path = str(tmp_path / "pp.json")
    with open(path, "w") as f:
        f.write("not json")
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is False
    assert cm.read() is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py::test_cache_miss_when_file_absent -v
```

Expected: `ImportError` — `CacheManager` not imported yet.

- [ ] **Step 3: Implement CacheManager**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# CacheManager
# ---------------------------------------------------------------------------
class CacheManager:
    def __init__(self, path: str = _CACHE_PATH, ttl: int = SCRAPE_INTERVAL):
        self.path = path
        self.ttl = ttl

    def is_fresh(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path) as f:
                data = json.load(f)
            fetched_at_str = data.get("fetched_at")
            if not fetched_at_str:
                return False
            fetched_at = datetime.fromisoformat(fetched_at_str)
            age = (
                datetime.now(tz=timezone.utc)
                - fetched_at.astimezone(timezone.utc)
            ).total_seconds()
            return age < self.ttl
        except (json.JSONDecodeError, ValueError, KeyError):
            return False

    def read(self) -> Optional[dict]:
        if not os.path.exists(self.path):
            return None
        try:
            with open(self.path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None

    def write(self, data: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
```

Also add `CacheManager` to the import in the test file — update the import line:

```python
from scraper.paddy_power import BotDetectedError, ScraperError, SCRAPE_INTERVAL, CacheManager
```

- [ ] **Step 4: Run all cache tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "cache" -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement CacheManager with TTL check"
```

---

## Task 4: ProxyRotator

**Files:**

- Modify: `scraper/paddy_power.py` (append `ProxyRotator`)
- Modify: `tests/scraper/test_paddy_power.py` (append proxy tests)

- [ ] **Step 1: Write failing tests for ProxyRotator**

Append to `tests/scraper/test_paddy_power.py`:

```python
from scraper.paddy_power import ProxyRotator


def test_proxy_rotator_disabled_returns_none():
    r = ProxyRotator(cfg={"enabled": False, "proxies": ["http://p1:8080"]})
    assert r.next() is None


def test_proxy_rotator_empty_list_returns_none():
    r = ProxyRotator(cfg={"enabled": True, "proxies": []})
    assert r.next() is None


def test_proxy_rotator_single_proxy():
    r = ProxyRotator(cfg={"enabled": True, "proxies": ["http://p1:8080"]})
    assert r.next() == "http://p1:8080"
    assert r.next() == "http://p1:8080"


def test_proxy_rotator_cycles():
    proxies = ["http://p1:8080", "http://p2:8080", "http://p3:8080"]
    r = ProxyRotator(cfg={"enabled": True, "proxies": proxies})
    results = [r.next() for _ in range(5)]
    assert results == [
        "http://p1:8080",
        "http://p2:8080",
        "http://p3:8080",
        "http://p1:8080",
        "http://p2:8080",
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "proxy" -v
```

Expected: `ImportError` — `ProxyRotator` not imported yet.

- [ ] **Step 3: Implement ProxyRotator**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# ProxyRotator
# ---------------------------------------------------------------------------
class ProxyRotator:
    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = _PROXY_CFG
        self._enabled: bool = cfg.get("enabled", False)
        self._proxies: list = cfg.get("proxies", [])
        self._index: int = 0

    def next(self) -> Optional[str]:
        if not self._enabled or not self._proxies:
            return None
        proxy = self._proxies[self._index % len(self._proxies)]
        self._index += 1
        return proxy
```

Update the import line in the test file:

```python
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, ProxyRotator,
)
```

- [ ] **Step 4: Run proxy tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "proxy" -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement ProxyRotator with round-robin cycling"
```

---

## Task 5: PaddyPowerClient

**Files:**

- Modify: `scraper/paddy_power.py` (append `PaddyPowerClient`)
- Modify: `tests/scraper/test_paddy_power.py` (append client tests)

- [ ] **Step 1: Write failing tests for PaddyPowerClient**

Append to `tests/scraper/test_paddy_power.py`:

```python
import respx
import httpx as _httpx

from scraper.paddy_power import PaddyPowerClient, BotDetectedError, ScraperError


@respx.mock
def test_client_successful_get():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(200, json={"events": []})
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    result = client.get("https://api.example.com/data")
    assert result == {"events": []}


@respx.mock
def test_client_raises_bot_detected_on_403():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(403)
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    with pytest.raises(BotDetectedError):
        client.get("https://api.example.com/data")


@respx.mock
def test_client_raises_bot_detected_on_html_body():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(
            200,
            content=b"<html>Cloudflare</html>",
            headers={"content-type": "text/html"},
        )
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    with pytest.raises(BotDetectedError):
        client.get("https://api.example.com/data")


@respx.mock
def test_client_retries_on_500_then_succeeds():
    route = respx.get("https://api.example.com/data")
    route.side_effect = [
        _httpx.Response(500),
        _httpx.Response(200, json={"events": [{"id": "1"}]}),
    ]
    client = PaddyPowerClient(
        proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}),
        max_retries=3,
    )
    # patch sleep so test doesn't actually wait
    import scraper.paddy_power as pp_module
    original_sleep = pp_module.time.sleep
    pp_module.time.sleep = lambda x: None
    try:
        result = client.get("https://api.example.com/data")
        assert result == {"events": [{"id": "1"}]}
    finally:
        pp_module.time.sleep = original_sleep


@respx.mock
def test_client_raises_scraper_error_after_max_retries():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(500)
    )
    client = PaddyPowerClient(
        proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}),
        max_retries=2,
    )
    import scraper.paddy_power as pp_module
    pp_module.time.sleep = lambda x: None
    with pytest.raises(ScraperError):
        client.get("https://api.example.com/data")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "client" -v
```

Expected: `ImportError` — `PaddyPowerClient` not imported yet.

- [ ] **Step 3: Implement PaddyPowerClient**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# PaddyPowerClient
# ---------------------------------------------------------------------------
class PaddyPowerClient:
    def __init__(
        self,
        proxy_rotator: "ProxyRotator" = None,
        max_retries: int = 3,
    ):
        self._rotator = proxy_rotator or ProxyRotator()
        self._max_retries = max_retries

    def get(self, url: str, params: dict = None) -> dict:
        delay = 1
        last_exc: Optional[Exception] = None

        for attempt in range(self._max_retries):
            proxy = self._rotator.next()
            proxies = {"all://": proxy} if proxy else None
            logger.debug("GET %s attempt=%d proxy=%s", url, attempt + 1, proxy)
            try:
                with httpx.Client(
                    headers=_HEADERS, proxies=proxies, timeout=15
                ) as client:
                    resp = client.get(url, params=params)

                if resp.status_code == 403:
                    raise BotDetectedError(f"403 on {url}")
                if "text/html" in resp.headers.get("content-type", ""):
                    raise BotDetectedError(f"HTML response on {url}")

                resp.raise_for_status()
                return resp.json()

            except BotDetectedError:
                raise
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                last_exc = exc
                logger.debug("Transient error, retrying in %ds: %s", delay, exc)
                time.sleep(delay)
                delay *= 2
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code >= 500:
                    last_exc = exc
                    logger.debug("5xx, retrying in %ds", delay)
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise ScraperError(
                        f"HTTP {exc.response.status_code} on {url}"
                    ) from exc

        raise ScraperError(
            f"Failed after {self._max_retries} attempts on {url}"
        ) from last_exc
```

Update the import line in the test file:

```python
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, ProxyRotator, PaddyPowerClient,
)
```

- [ ] **Step 4: Run client tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "client" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement PaddyPowerClient with retry and bot detection"
```

---

## Task 6: Response Parser

**Files:**

- Modify: `scraper/paddy_power.py` (append parser functions)
- Modify: `tests/scraper/test_paddy_power.py` (append parser tests)

- [ ] **Step 1: Write failing tests for the parser**

Append to `tests/scraper/test_paddy_power.py`:

```python
from scraper.paddy_power import _parse_events, _parse_race_time


_SAMPLE_EVENTS = {
    "events": [
        {
            "id": 12345,
            "startTime": "2026-06-12T14:00:00Z",
            "venue": "Ascot",
        }
    ]
}

_SAMPLE_MARKETS = {
    "12345": {
        "markets": [
            {
                "id": "m_1",
                "type": "WIN",
                "eachWayTerms": {"places": 3, "reduction": 0.25},
                "runners": [
                    {
                        "id": "r_1",
                        "name": "Mighty Oak",
                        "sp": 6.0,
                        "prices": [{"decimal": 6.5}],
                    }
                ],
            }
        ]
    }
}


def test_parse_race_time_converts_to_dublin():
    # 14:00 UTC = 15:00 Europe/Dublin (BST, UTC+1 in June)
    result = _parse_race_time("2026-06-12T14:00:00Z")
    assert "15:00:00" in result
    assert "+01:00" in result


def test_parse_events_structure():
    races = _parse_events(_SAMPLE_EVENTS, _SAMPLE_MARKETS)
    assert len(races) == 1
    race = races[0]
    assert race["race_id"] == "12345"
    assert race["venue"] == "Ascot"
    assert "15:00:00" in race["race_time"]


def test_parse_events_market_fields():
    races = _parse_events(_SAMPLE_EVENTS, _SAMPLE_MARKETS)
    market = races[0]["markets"][0]
    assert market["market_type"] == "WIN"
    assert market["each_way_terms"] == {"places": 3, "reduction": 0.25}


def test_parse_events_selection_fields():
    races = _parse_events(_SAMPLE_EVENTS, _SAMPLE_MARKETS)
    sel = races[0]["markets"][0]["selections"][0]
    assert sel["selection_id"] == "r_1"
    assert sel["horse_name"] == "Mighty Oak"
    assert sel["sp"] == 6.0
    assert sel["odds_decimal"] == 6.5


def test_parse_events_null_each_way_when_absent():
    events = {"events": [{"id": 99, "startTime": "2026-06-12T14:00:00Z", "venue": "York"}]}
    markets = {
        "99": {
            "markets": [
                {
                    "id": "m_2",
                    "type": "PLACE",
                    "runners": [],
                }
            ]
        }
    }
    races = _parse_events(events, markets)
    assert races[0]["markets"][0]["each_way_terms"] is None


def test_parse_events_empty_markets_when_race_id_missing():
    events = {"events": [{"id": 77, "startTime": "2026-06-12T14:00:00Z", "venue": "York"}]}
    races = _parse_events(events, {})
    assert races[0]["markets"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "parse" -v
```

Expected: `ImportError` — `_parse_events`, `_parse_race_time` not defined yet.

- [ ] **Step 3: Implement parser functions**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def _parse_race_time(raw: str) -> str:
    """Parse ISO datetime string and return Europe/Dublin ISO string."""
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return to_local(dt).isoformat()


def _parse_events(events_data: dict, markets_by_id: dict) -> list:
    """
    Build races list from events API response and a dict of markets keyed by
    race_id string. markets_by_id[race_id] is the raw /markets API response.
    """
    races = []
    for event in events_data.get("events", []):
        race_id = str(event.get("id", ""))
        raw_markets = markets_by_id.get(race_id, {}).get("markets", [])

        markets_list = []
        for market in raw_markets:
            ew = market.get("eachWayTerms")
            each_way_terms = (
                {"places": ew.get("places"), "reduction": ew.get("reduction")}
                if ew
                else None
            )
            selections = [
                {
                    "selection_id": str(runner.get("id", "")),
                    "horse_name": runner.get("name", ""),
                    "sp": runner.get("sp"),
                    "odds_decimal": (
                        runner["prices"][0].get("decimal")
                        if runner.get("prices")
                        else None
                    ),
                }
                for runner in market.get("runners", [])
            ]
            markets_list.append(
                {
                    "market_type": market.get("type", ""),
                    "each_way_terms": each_way_terms,
                    "selections": selections,
                }
            )

        races.append(
            {
                "race_id": race_id,
                "race_time": _parse_race_time(event.get("startTime", "")),
                "venue": event.get("venue", ""),
                "markets": markets_list,
            }
        )
    return races
```

Update the import line in the test file:

```python
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, ProxyRotator, PaddyPowerClient,
    _parse_events, _parse_race_time,
)
```

- [ ] **Step 4: Run parser tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "parse" -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement response parser with timezone conversion"
```

---

## Task 7: ChromeFallback

**Files:**

- Modify: `scraper/paddy_power.py` (append `ChromeFallback`)
- Modify: `tests/scraper/test_paddy_power.py` (append fallback tests)

- [ ] **Step 1: Write failing tests for ChromeFallback**

Append to `tests/scraper/test_paddy_power.py`:

```python
from unittest.mock import MagicMock, patch

from scraper.paddy_power import ChromeFallback, ScraperError


def _make_mock_response(url: str, body: dict):
    resp = MagicMock()
    resp.url = url
    resp.json.return_value = body
    return resp


def test_chrome_fallback_captures_events_and_markets(mocker):
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser

    captured_handlers = []

    def fake_on(event, handler):
        if event == "response":
            captured_handlers.append(handler)

    mock_page.on.side_effect = fake_on

    def fake_goto(url):
        # simulate API responses firing after navigation
        events_resp = _make_mock_response(
            "https://api.paddypower.com/api/v1/events",
            {"events": [{"id": 1, "startTime": "2026-06-12T14:00:00Z", "venue": "Ascot"}]},
        )
        markets_resp = _make_mock_response(
            "https://api.paddypower.com/api/v1/competitions/1/markets",
            {"markets": [], "race_id": "1"},
        )
        for handler in captured_handlers:
            handler(events_resp)
            handler(markets_resp)

    mock_page.goto.side_effect = fake_goto

    mocker.patch("scraper.paddy_power.sync_playwright").__enter__ = lambda s: mock_p
    # patch as context manager
    cm = MagicMock()
    cm.__enter__.return_value = mock_p
    mocker.patch("scraper.paddy_power.sync_playwright", return_value=cm)

    fallback = ChromeFallback()
    result = fallback.fetch()
    assert "events_data" in result
    assert result["events_data"]["events"][0]["id"] == 1


def test_chrome_fallback_raises_if_no_events_captured(mocker):
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser
    mock_page.on.return_value = None

    cm = MagicMock()
    cm.__enter__.return_value = mock_p
    mocker.patch("scraper.paddy_power.sync_playwright", return_value=cm)

    fallback = ChromeFallback()
    with pytest.raises(ScraperError, match="no API responses"):
        fallback.fetch()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "chrome" -v
```

Expected: `ImportError` — `ChromeFallback` not defined yet.

- [ ] **Step 3: Implement ChromeFallback**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# ChromeFallback
# ---------------------------------------------------------------------------
from playwright.sync_api import sync_playwright  # noqa: E402  (top-level import moved here for mockability)


class ChromeFallback:
    def fetch(self) -> dict:
        """
        Drive a real Chromium instance, intercept JSON responses from /api/
        paths, and return the raw events and markets data for parsing.
        """
        logger.warning("Bot detected — triggering Playwright Chrome fallback")
        captured_events: Optional[dict] = None
        captured_markets: dict = {}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def _handle_response(response) -> None:
                if "/api/" not in response.url:
                    return
                try:
                    body = response.json()
                except Exception:
                    return

                nonlocal captured_events
                if "events" in body and captured_events is None:
                    captured_events = body
                    logger.debug("Chrome captured events from %s", response.url)
                elif "markets" in body:
                    # Extract race_id from URL: .../competitions/{race_id}/markets
                    parts = response.url.rstrip("/").split("/")
                    try:
                        race_id = parts[parts.index("competitions") + 1]
                    except (ValueError, IndexError):
                        race_id = "_unknown"
                    captured_markets[race_id] = body
                    logger.debug("Chrome captured markets for race_id=%s", race_id)

            page.on("response", _handle_response)
            page.goto(_HORSE_RACING_PAGE)
            page.wait_for_timeout(5000)
            browser.close()

        if captured_events is None:
            raise ScraperError("ChromeFallback: no API responses captured")

        return {"events_data": captured_events, "markets_data": captured_markets}
```

Also move the `from playwright.sync_api import sync_playwright` to the top of the file, right after the other imports. Remove the inline import added above and place it with the other imports at the top of `scraper/paddy_power.py`:

Add after `import yaml`:

```python
from playwright.sync_api import sync_playwright
```

Update the test import:

```python
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, ProxyRotator, PaddyPowerClient,
    ChromeFallback, _parse_events, _parse_race_time,
)
```

- [ ] **Step 4: Run fallback tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "chrome" -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement ChromeFallback with Playwright XHR interception"
```

---

## Task 8: scrape() Public Function

**Files:**

- Modify: `scraper/paddy_power.py` (append `scrape()`)
- Modify: `tests/scraper/test_paddy_power.py` (append integration tests)

- [ ] **Step 1: Write failing tests for scrape()**

Append to `tests/scraper/test_paddy_power.py`:

```python
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

from scraper.paddy_power import scrape, ScraperError, BotDetectedError


def _fresh_cache_data():
    return {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "races": [{"race_id": "cached"}],
    }


def test_scrape_returns_cached_data_when_fresh(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    data = _fresh_cache_data()
    with open(cache_path, "w") as f:
        json.dump(data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    result = scrape()
    assert result["races"][0]["race_id"] == "cached"


def test_scrape_fetches_when_cache_stale(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    stale_data = {
        "fetched_at": (
            datetime.now(tz=timezone.utc) - timedelta(seconds=5000)
        ).isoformat(),
        "races": [],
    }
    with open(cache_path, "w") as f:
        json.dump(stale_data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = [
        {"events": [{"id": 42, "startTime": "2026-06-12T14:00:00Z", "venue": "York"}]},
        {"markets": []},
    ]
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)
    mocker.patch("scraper.paddy_power.ChromeFallback")

    result = scrape()
    assert result["races"][0]["race_id"] == "42"


def test_scrape_triggers_chrome_fallback_on_bot_detection(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = BotDetectedError("403")
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)

    mock_fallback = MagicMock()
    mock_fallback.fetch.return_value = {
        "events_data": {
            "events": [{"id": 99, "startTime": "2026-06-12T14:00:00Z", "venue": "Epsom"}]
        },
        "markets_data": {},
    }
    mocker.patch("scraper.paddy_power.ChromeFallback", return_value=mock_fallback)

    result = scrape()
    assert result["races"][0]["race_id"] == "99"
    assert os.path.exists(cache_path)


def test_scrape_raises_scraper_error_when_both_paths_fail(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = BotDetectedError("403")
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)

    mock_fallback = MagicMock()
    mock_fallback.fetch.side_effect = ScraperError("Chrome also failed")
    mocker.patch("scraper.paddy_power.ChromeFallback", return_value=mock_fallback)

    with pytest.raises(ScraperError):
        scrape()


def test_scrape_force_bypasses_fresh_cache(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    data = _fresh_cache_data()
    with open(cache_path, "w") as f:
        json.dump(data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = [
        {"events": [{"id": 7, "startTime": "2026-06-12T14:00:00Z", "venue": "Naas"}]},
        {"markets": []},
    ]
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)
    mocker.patch("scraper.paddy_power.ChromeFallback")

    result = scrape(force=True)
    assert result["races"][0]["race_id"] == "7"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "scrape" -v
```

Expected: `ImportError` — `scrape` not defined yet.

- [ ] **Step 3: Implement scrape()**

Append to `scraper/paddy_power.py`:

```python
# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scrape(force: bool = False) -> dict:
    """
    Fetch live Paddy Power horse racing cards and odds.

    Returns a dict matching the cache schema. Raises ScraperError if all
    fetch paths are exhausted.
    """
    cache = CacheManager()

    if not force and cache.is_fresh():
        logger.debug("Cache hit — returning cached data")
        return cache.read()

    client = PaddyPowerClient()
    fallback = ChromeFallback()

    try:
        events_data = client.get(_EVENTS_URL, params={"categoryId": "21"})
        markets_by_id: dict = {}
        for event in events_data.get("events", []):
            race_id = str(event.get("id", ""))
            url = _MARKETS_URL.format(race_id=race_id)
            try:
                markets_by_id[race_id] = client.get(url)
            except ScraperError as exc:
                logger.warning("Failed to fetch markets for race_id=%s: %s", race_id, exc)

    except BotDetectedError:
        logger.warning("Bot detected on httpx path — triggering ChromeFallback")
        try:
            raw = fallback.fetch()
            events_data = raw["events_data"]
            markets_by_id = raw["markets_data"]
        except Exception as exc:
            raise ScraperError("Both httpx and ChromeFallback failed") from exc

    result = {
        "fetched_at": datetime.now(tz=TZ).isoformat(),
        "races": _parse_events(events_data, markets_by_id),
    }
    cache.write(result)
    logger.debug("Scraped %d races, cache written to %s", len(result["races"]), cache.path)
    return result
```

Update the final import in test file:

```python
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, ProxyRotator, PaddyPowerClient,
    ChromeFallback, _parse_events, _parse_race_time, scrape,
)
```

- [ ] **Step 4: Run all scrape() tests**

Run:

```
pytest tests/scraper/test_paddy_power.py -k "scrape" -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Run full test suite**

Run:

```
pytest tests/scraper/test_paddy_power.py -v
```

Expected: all tests PASSED. If any fail, investigate before committing.

- [ ] **Step 6: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py
git commit -m "feat: implement scrape() public function with cache and fallback"
```

---

## Task 9: Verify Data Directory & Final Smoke Test

**Files:**

- No code changes — runtime verification only

- [ ] **Step 1: Ensure data/cache directory exists**

Run:

```
python -c "import os; os.makedirs('data/cache', exist_ok=True); print('OK')"
```

Expected: `OK`

- [ ] **Step 2: Run a live smoke test (optional, requires network)**

Run:

```
python -c "
from scraper.paddy_power import scrape, ScraperError
try:
    data = scrape(force=True)
    print(f'Fetched {len(data[\"races\"])} races')
    if data['races']:
        print('First race:', data['races'][0]['race_id'], data['races'][0]['venue'])
except ScraperError as e:
    print(f'ScraperError (expected if endpoints need tuning): {e}')
"
```

Expected: either prints race data, or prints `ScraperError` with a descriptive message indicating where the API shape differs from the assumed spec. If the latter, the endpoint URLs or JSON field names in `_parse_events` will need adjustment based on the actual API response.

- [ ] **Step 3: Commit if data/cache/.gitkeep needed**

```bash
echo "" > data/cache/.gitkeep
git add data/cache/.gitkeep
git commit -m "chore: add data/cache directory to version control"
```

---

## Notes

- **Endpoint shapes are assumed:** The field names in `_parse_events` (`events`, `id`, `startTime`, `venue`, `markets`, `eachWayTerms`, `runners`, `prices`, `decimal`) are based on typical Flutter/Paddy Power API patterns. Run Task 9 Step 2 and inspect the actual response to adjust field names if needed.
- **Playwright install:** `playwright install chromium` must be run once after `pip install playwright` to download the browser binary.
- **Proxy config:** To enable proxies, set `proxy_pool.enabled: true` and add proxy URLs to `proxy_pool.proxies` in `config.yaml`.
