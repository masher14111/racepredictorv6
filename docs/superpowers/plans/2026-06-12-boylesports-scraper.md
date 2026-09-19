# BoyleSports Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `scraper/boylesports.py` producing the standardized flat-row odds DataFrame, with robust low-odds parsing, computed each-way margins, GBP→EUR currency tagging, and a shared Selenium fallback retrofitted into all three scrapers.

**Architecture:** BoyleSports mirrors `livescorebet.py`'s component structure (Throttler, ProxyRotator, Client, fallback tiers, CacheManager, `_parse_*`, `scrape()`). A new `scraper/_selenium_fallback.py` holds reusable headless-Chrome XHR/DOM capture used by all three bookies. The shared `live_odds.parquet` switches from clobber to read-merge-write keyed on `source`.

**Tech Stack:** Python 3.14, httpx, respx (tests), pandas/pyarrow, lxml, Playwright, Selenium, PyYAML.

---

## File Structure

- Create: `scraper/boylesports.py` — BoyleSports scraper (all components + `scrape()`).
- Create: `scraper/_selenium_fallback.py` — reusable `SeleniumFallback`.
- Create: `utils/currency.py` — `to_eur()` and UK-course detection.
- Create: `tests/scraper/test_boylesports.py`, `tests/scraper/test_selenium_fallback.py`, `tests/utils/__init__.py`, `tests/utils/test_currency.py`.
- Modify: `config.yaml` — `low_odds_threshold`, `gbp_eur_rate`, `boylesports:` block.
- Modify: `requirements.txt` — add `selenium`.
- Modify: `scraper/livescorebet.py` — read-merge-write parquet, new columns, Selenium tier.
- Modify: `scraper/paddy_power.py` — Selenium tier.

---

## Task 0: Project setup

**Files:**

- Modify: `requirements.txt`
- Modify: `config.yaml`

- [ ] **Step 1: Init git (repo does not exist yet)**

Run:

```bash
cd "C:/Users/mshr/Desktop/Race Predictor v3"
git init && git add -A && git commit -m "chore: baseline before BoyleSports scraper"
```

- [ ] **Step 2: Add selenium to requirements.txt**

Append line:

```
selenium
```

- [ ] **Step 3: Add config keys to config.yaml**

Add after `each_way_threshold: 8.0`:

```yaml
low_odds_threshold: 2.0 # odds_decimal below this flags is_low_odds (odds-on)
gbp_eur_rate: 1.18 # static GBP→EUR rate for GUI monetary display
```

Add at end of file:

```yaml
boylesports:
  horse_racing_url: https://www.boylesports.com/horse-racing
  api_base: https://www.boylesports.com/cms/api # ⚠️ placeholder — confirm via DevTools
```

- [ ] **Step 4: Install selenium**

Run: `pip install selenium`
Expected: installs without error.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.yaml && git commit -m "chore: add selenium dep and boylesports/currency config"
```

---

## Task 1: Currency utility

**Files:**

- Create: `utils/currency.py`
- Create: `tests/utils/__init__.py`
- Test: `tests/utils/test_currency.py`

- [ ] **Step 1: Write the failing tests**

`tests/utils/test_currency.py`:

```python
import math


def test_eur_passthrough():
    from utils.currency import to_eur
    assert to_eur(10.0, "EUR") == 10.0


def test_gbp_converts_with_config_rate():
    from utils.currency import to_eur
    # default config gbp_eur_rate = 1.18
    assert math.isclose(to_eur(10.0, "GBP"), 11.8, rel_tol=1e-9)


def test_unknown_currency_passthrough():
    from utils.currency import to_eur
    assert to_eur(5.0, "USD") == 5.0


def test_uk_course_detected_as_gbp():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Ascot") == "GBP"
    assert currency_for_venue("ascot") == "GBP"


def test_irish_course_detected_as_eur():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Leopardstown") == "EUR"


def test_unknown_venue_defaults_eur():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Nowhere Park") == "EUR"


def test_country_code_overrides_venue():
    from utils.currency import currency_for_venue
    assert currency_for_venue("Nowhere Park", country_code="GB") == "GBP"
    assert currency_for_venue("Ascot", country_code="IE") == "EUR"
```

`tests/utils/__init__.py`: empty file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/utils/test_currency.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement utils/currency.py**

```python
import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()
GBP_EUR_RATE: float = float(_cfg.get("gbp_eur_rate", 1.18))

# Known UK (GBP) racecourses. Anything not listed defaults to EUR (Irish/other).
_UK_COURSES = {
    "ascot", "aintree", "ayr", "bangor", "bath", "beverley", "brighton",
    "carlisle", "cartmel", "catterick", "chelmsford", "cheltenham", "chepstow",
    "chester", "doncaster", "epsom", "exeter", "fakenham", "ffos las",
    "fontwell", "goodwood", "hamilton", "haydock", "hereford", "hexham",
    "huntingdon", "kelso", "kempton", "leicester", "lingfield", "ludlow",
    "market rasen", "musselburgh", "newbury", "newcastle", "newmarket",
    "newton abbot", "nottingham", "perth", "plumpton", "pontefract", "redcar",
    "ripon", "salisbury", "sandown", "sedgefield", "southwell", "stratford",
    "taunton", "thirsk", "uttoxeter", "warwick", "wetherby", "wincanton",
    "windsor", "wolverhampton", "worcester", "yarmouth", "york",
}


def currency_for_venue(venue: str, country_code: str = None) -> str:
    """Return 'GBP' for UK races, else 'EUR'. country_code (ISO-2) overrides venue."""
    if country_code:
        return "GBP" if country_code.upper() in ("GB", "UK") else "EUR"
    name = (venue or "").strip().lower()
    return "GBP" if name in _UK_COURSES else "EUR"


def to_eur(amount: float, currency: str) -> float:
    """Convert a monetary amount to EUR. EUR/unknown pass through; GBP uses rate."""
    if currency == "GBP":
        return amount * GBP_EUR_RATE
    return amount
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/utils/test_currency.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add utils/currency.py tests/utils/ && git commit -m "feat: add currency utility with GBP->EUR conversion and UK course detection"
```

---

## Task 2: BoyleSports module scaffold (config, exceptions, Throttler, ProxyRotator, CacheManager)

**Files:**

- Create: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests**

`tests/scraper/test_boylesports.py`:

```python
import time as _time

import pytest


def test_imports():
    from scraper.boylesports import (
        BotDetectedError,
        ScraperError,
        Throttler,
        ProxyRotator,
        BoyleSportsClient,
        CacheManager,
        scrape,
    )


def test_throttler_first_call_immediate():
    from scraper.boylesports import Throttler
    t = Throttler(rate=1.0)
    start = _time.monotonic()
    t.acquire()
    assert _time.monotonic() - start < 0.1


def test_proxy_rotator_disabled_returns_none():
    from scraper.boylesports import ProxyRotator
    r = ProxyRotator(cfg={"enabled": False, "proxies": ["http://p1"]})
    assert r.next() is None


def test_proxy_rotator_cycles():
    from scraper.boylesports import ProxyRotator
    r = ProxyRotator(cfg={"enabled": True, "proxies": ["http://p1", "http://p2"]})
    assert r.next() == "http://p1"
    assert r.next() == "http://p2"
    assert r.next() == "http://p1"


def test_cache_write_read_roundtrip(tmp_path):
    from scraper.boylesports import CacheManager
    cache = CacheManager(path=str(tmp_path / "bs.json"), ttl=3600)
    cache.write({"fetched_at": "2026-06-12T10:00:00+01:00", "rows": []})
    data = cache.read()
    assert data["rows"] == []


def test_cache_missing_is_not_fresh(tmp_path):
    from scraper.boylesports import CacheManager
    cache = CacheManager(path=str(tmp_path / "missing.json"), ttl=3600)
    assert cache.is_fresh() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the scaffold in scraper/boylesports.py**

```python
import json
import os
import time as _time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
import yaml
from lxml import html as lxml_html
from playwright.sync_api import sync_playwright

from utils.currency import currency_for_venue
from utils.logger import get_logger
from utils.timezone import TZ, to_local

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "boylesports_cache.json"
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
LOW_ODDS_THRESHOLD: float = float(_cfg.get("low_odds_threshold", 2.0))
_PROXY_CFG: dict = _cfg.get("proxy_pool", {})
_BS_CFG: dict = _cfg.get("boylesports", {})

_HORSE_RACING_URL: str = _BS_CFG.get(
    "horse_racing_url", "https://www.boylesports.com/horse-racing"
)
_API_BASE: str = _BS_CFG.get("api_base", "https://www.boylesports.com/cms/api")
# ⚠️ placeholder endpoints — confirm via DevTools (Network tab)
_RACING_INDEX_URL = f"{_API_BASE}/horse-racing/meetings"
_EVENT_URL_TPL = _API_BASE + "/horse-racing/event/{event_id}"
_PARQUET_PATH: str = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "live_odds.parquet")
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IE,en;q=0.9",
    "Referer": _HORSE_RACING_URL,
}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class BotDetectedError(Exception):
    """Raised when BoyleSports returns 403, an HTML challenge, or non-JSON."""


class ScraperError(Exception):
    """Raised when all fetch tiers (including stale cache) are exhausted."""


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
                datetime.now(tz=timezone.utc) - fetched_at.astimezone(timezone.utc)
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
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: scaffold boylesports scraper (config, exceptions, throttler, proxy, cache)"
```

---

## Task 3: Low-odds robust price parsing (`_to_decimal`)

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests** (append to test file)

```python
import math as _math


def test_to_decimal_float_passthrough():
    from scraper.boylesports import _to_decimal
    assert _to_decimal(6.5) == 6.5


def test_to_decimal_numeric_string():
    from scraper.boylesports import _to_decimal
    assert _to_decimal("6.5") == 6.5


def test_to_decimal_fractional():
    from scraper.boylesports import _to_decimal
    assert _to_decimal("5/2") == 3.5
    assert _to_decimal("1/5") == 1.2
    assert _to_decimal("2/7") == round(1 + 2 / 7, 6)


def test_to_decimal_evens():
    from scraper.boylesports import _to_decimal
    assert _to_decimal("EVS") == 2.0
    assert _to_decimal("Evens") == 2.0
    assert _to_decimal("evs") == 2.0


def test_to_decimal_garbage_is_nan():
    from scraper.boylesports import _to_decimal
    assert _math.isnan(_to_decimal(""))
    assert _math.isnan(_to_decimal(None))
    assert _math.isnan(_to_decimal("not-a-price"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k to_decimal -v`
Expected: FAIL (`_to_decimal` not defined).

- [ ] **Step 3: Implement `_to_decimal`** (append to `scraper/boylesports.py`)

```python
# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------
def _to_decimal(raw) -> float:
    """Parse a BoyleSports price into decimal odds.

    Handles decimal floats/strings, fractional ("5/2"), and "EVS"/"Evens".
    Returns NaN for missing or malformed input.
    """
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().lower()
    if not text:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -k to_decimal -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: robust low-odds price parsing for boylesports"
```

---

## Task 4: Each-way margin computation (`_ew_margin`)

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_ew_margin_basic():
    from scraper.boylesports import _ew_margin
    # 2 runners, 2 places, reduction 0.25.
    # win odds 5.0 -> place 1+(4*0.25)=2.0 ; win odds 3.0 -> place 1+(2*0.25)=1.5
    # sum(1/2.0 + 1/1.5) = 0.5 + 0.6667 = 1.1667 ; minus places(2) = -0.8333
    margin = _ew_margin([5.0, 3.0], places=2, reduction=0.25)
    assert round(margin, 4) == round(0.5 + 1 / 1.5 - 2, 4)


def test_ew_margin_missing_terms_is_nan():
    from scraper.boylesports import _ew_margin
    import math
    assert math.isnan(_ew_margin([5.0], places=None, reduction=0.25))
    assert math.isnan(_ew_margin([5.0], places=2, reduction=None))


def test_ew_margin_ignores_nan_odds():
    from scraper.boylesports import _ew_margin
    import math
    margin = _ew_margin([5.0, float("nan")], places=1, reduction=0.25)
    # only the 5.0 runner counts: place 2.0 -> 0.5 - 1 = -0.5
    assert round(margin, 4) == -0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k ew_margin -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_ew_margin`** (append)

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -k ew_margin -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: compute each-way place-market margin for boylesports"
```

---

## Task 5: BoyleSportsClient HTTP fetch

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
import respx
import httpx as _httpx


@respx.mock
def test_client_get_returns_json():
    from scraper.boylesports import BoyleSportsClient, Throttler, _RACING_INDEX_URL
    respx.get(_RACING_INDEX_URL).mock(
        return_value=_httpx.Response(200, json={"meetings": []})
    )
    client = BoyleSportsClient(throttler=Throttler(rate=100.0))
    data = client.get_index()
    assert data == {"meetings": []}


@respx.mock
def test_client_raises_bot_detected_on_403():
    from scraper.boylesports import (
        BoyleSportsClient, Throttler, BotDetectedError, _RACING_INDEX_URL,
    )
    respx.get(_RACING_INDEX_URL).mock(return_value=_httpx.Response(403, text="no"))
    client = BoyleSportsClient(throttler=Throttler(rate=100.0))
    with pytest.raises(BotDetectedError):
        client.get_index()


@respx.mock
def test_client_raises_bot_detected_on_html():
    from scraper.boylesports import (
        BoyleSportsClient, Throttler, BotDetectedError, _RACING_INDEX_URL,
    )
    respx.get(_RACING_INDEX_URL).mock(
        return_value=_httpx.Response(
            200, text="<html>challenge</html>",
            headers={"content-type": "text/html"},
        )
    )
    client = BoyleSportsClient(throttler=Throttler(rate=100.0))
    with pytest.raises(BotDetectedError):
        client.get_index()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k client -v`
Expected: FAIL.

- [ ] **Step 3: Implement `BoyleSportsClient`** (append)

```python
# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class BoyleSportsClient:
    def __init__(self, throttler: "Throttler" = None, proxy_rotator: "ProxyRotator" = None):
        self._throttler = throttler or Throttler()
        self._rotator = proxy_rotator or ProxyRotator()

    def _get(self, url: str) -> dict:
        self._throttler.acquire()
        proxy = self._rotator.next()
        mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
        logger.debug("GET %s proxy=%s", url, proxy)
        with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
            resp = client.get(url)
        if resp.status_code == 403:
            raise BotDetectedError(f"403 from BoyleSports: {url}")
        if "text/html" in resp.headers.get("content-type", ""):
            raise BotDetectedError(f"HTML response from BoyleSports: {url}")
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError as exc:
            raise BotDetectedError(f"Non-JSON from {url}: {exc}") from exc

    def get_index(self) -> dict:
        """Fetch the meetings index JSON."""
        return self._get(_RACING_INDEX_URL)

    def get_event(self, event_id: str) -> dict:
        """Fetch a single event's detail JSON."""
        return self._get(_EVENT_URL_TPL.format(event_id=event_id))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -k client -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: add BoyleSportsClient with bot-detection guards"
```

---

## Task 6: Event parsing into standardized rows (`_parse_event`, `_extract_event_ids`)

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

**Note on shape:** these placeholder field names (`meetings`, `events`, `id`,
`startTime`, `venue`, `countryCode`, `markets`, `marketType`, `eachWay.places`,
`eachWay.reduction`, `selections`, `name`, `price`, `nonRunner`) must be confirmed
via DevTools and adjusted. Tests use this assumed shape.

- [ ] **Step 1: Write the failing tests** (append)

```python
def _sample_event():
    return {
        "events": [{
            "id": "evt1",
            "startTime": "2026-06-12T15:30:00Z",
            "venue": "Ascot",
            "countryCode": "GB",
            "markets": [{
                "marketType": "EACH_WAY",
                "eachWay": {"places": 3, "reduction": 0.25},
                "selections": [
                    {"name": "Fast Horse", "price": "5/2"},
                    {"name": "Odds On Fav", "price": "1/5"},
                    {"name": "Scratched", "price": "10.0", "nonRunner": True},
                ],
            }],
        }]
    }


def test_extract_event_ids():
    from scraper.boylesports import _extract_event_ids
    index = {"meetings": [{"events": [{"id": "a"}, {"id": "b"}]},
                          {"events": [{"id": "c"}]}]}
    assert _extract_event_ids(index) == ["a", "b", "c"]


def test_parse_event_rows_and_flags():
    from scraper.boylesports import _parse_event
    rows = _parse_event(_sample_event())
    # non-runner dropped -> 2 rows
    assert len(rows) == 2
    fav = next(r for r in rows if r["horse_name"] == "Odds On Fav")
    assert fav["odds_decimal"] == 1.2
    assert fav["is_low_odds"] is True          # 1.2 < 2.0
    assert fav["currency"] == "GBP"            # Ascot / GB
    assert fav["ew_places"] == 3
    assert fav["ew_reduction"] == 0.25


def test_parse_event_high_odds_not_low():
    from scraper.boylesports import _parse_event
    rows = _parse_event(_sample_event())
    fast = next(r for r in rows if r["horse_name"] == "Fast Horse")
    assert fast["odds_decimal"] == 3.5
    assert fast["is_low_odds"] is False


def test_parse_event_margin_present():
    from scraper.boylesports import _parse_event
    import math
    rows = _parse_event(_sample_event())
    # ew_margin identical across rows of the market, and not NaN
    margins = {r["ew_margin"] for r in rows}
    assert len(margins) == 1
    assert not math.isnan(rows[0]["ew_margin"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k "parse_event or extract_event" -v`
Expected: FAIL.

- [ ] **Step 3: Implement parsing** (append)

```python
# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_race_time(raw: str) -> str:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return to_local(dt).isoformat()


def _extract_event_ids(index: dict) -> list:
    """Collect event ids from the meetings index."""
    ids: list = []
    for meeting in index.get("meetings") or []:
        for ev in meeting.get("events") or []:
            eid = ev.get("id", "")
            if eid and eid not in ids:
                ids.append(eid)
    return ids


def _parse_event(event_data: dict) -> list:
    """Parse one event JSON into standardized flat row dicts."""
    events = event_data.get("events") or []
    if not events:
        return []
    event = events[0]

    race_id = str(event.get("id", ""))
    raw_time = event.get("startTime", "")
    try:
        race_time = _parse_race_time(raw_time) if raw_time else ""
    except (ValueError, TypeError):
        race_time = raw_time or ""
    venue = event.get("venue", "")
    currency = currency_for_venue(venue, event.get("countryCode"))

    rows = []
    for market in event.get("markets") or []:
        market_type = (market.get("marketType") or "WIN").upper()
        ew = market.get("eachWay") or {}
        ew_places = ew.get("places")
        ew_reduction = ew.get("reduction")

        live = [s for s in (market.get("selections") or []) if not s.get("nonRunner")]
        win_odds = [_to_decimal(s.get("price")) for s in live]
        margin = _ew_margin(win_odds, ew_places, ew_reduction)

        for sel, odds in zip(live, win_odds):
            name = sel.get("name", "")
            if not name:
                continue
            rows.append({
                "race_id": race_id,
                "race_time": race_time,
                "venue": venue,
                "market_type": market_type,
                "ew_places": ew_places,
                "ew_reduction": ew_reduction,
                "ew_margin": margin,
                "horse_name": name,
                "odds_decimal": odds,
                "sp": float("nan"),
                "is_low_odds": bool(odds == odds and odds < LOW_ODDS_THRESHOLD),
                "currency": currency,
            })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -k "parse_event or extract_event" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: parse boylesports events into standardized rows with low-odds flag, margin, currency"
```

---

## Task 7: Shared Selenium fallback helper

**Files:**

- Create: `scraper/_selenium_fallback.py`
- Test: `tests/scraper/test_selenium_fallback.py`

**Approach:** the helper is constructed with callbacks so it stays site-agnostic.
Selenium itself is not driven in unit tests (no browser in CI); we test the pure
logic (`_rows_from_logs`) that turns captured response bodies into rows via the
injected parser, and that `fetch()` raises `ScraperError` when nothing is captured.

- [ ] **Step 1: Write the failing tests**

`tests/scraper/test_selenium_fallback.py`:

```python
import pytest


def test_rows_from_logs_uses_parser_on_matching_url():
    from scraper._selenium_fallback import _rows_from_bodies

    bodies = [
        ("https://x/api/event/1", {"events": [{"id": 1}]}),
        ("https://x/static/app.js", {"junk": True}),
    ]
    rows = _rows_from_bodies(
        bodies,
        xhr_url_predicate=lambda u: "/api/event" in u,
        parse_xhr=lambda body: [{"id": body["events"][0]["id"]}],
    )
    assert rows == [{"id": 1}]


def test_rows_from_logs_skips_parser_errors():
    from scraper._selenium_fallback import _rows_from_bodies

    def boom(body):
        raise KeyError("missing")

    bodies = [("https://x/api/event/1", {})]
    rows = _rows_from_bodies(
        bodies, xhr_url_predicate=lambda u: True, parse_xhr=boom
    )
    assert rows == []


def test_fetch_raises_when_no_rows(monkeypatch):
    from scraper._selenium_fallback import SeleniumFallback, ScraperError

    fb = SeleniumFallback(
        url="https://x/horse-racing",
        xhr_url_predicate=lambda u: True,
        parse_xhr=lambda b: [],
    )
    # Stub the browser capture to yield nothing.
    monkeypatch.setattr(fb, "_capture_bodies", lambda: [])
    monkeypatch.setattr(fb, "_dom_rows", lambda: [])
    with pytest.raises(ScraperError):
        fb.fetch()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_selenium_fallback.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement `scraper/_selenium_fallback.py`**

```python
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
            driver.implicitly_wait(self._wait_ms / 1000.0)
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_selenium_fallback.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scraper/_selenium_fallback.py tests/scraper/test_selenium_fallback.py && git commit -m "feat: add reusable Selenium fallback helper"
```

---

## Task 8: BoyleSports DataFrame conversion + parquet read-merge-write

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_rows_to_df_columns():
    from scraper.boylesports import _rows_to_df, _PARQUET_COLUMNS
    from datetime import datetime, timezone
    rows = [{
        "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
        "venue": "Ascot", "market_type": "EACH_WAY", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": -0.5, "horse_name": "H",
        "odds_decimal": 1.2, "sp": float("nan"), "is_low_odds": True,
        "currency": "GBP",
    }]
    df = _rows_to_df(rows, datetime(2026, 6, 12, tzinfo=timezone.utc))
    assert list(df.columns) == _PARQUET_COLUMNS
    assert df.iloc[0]["source"] == "boylesports"
    assert df.iloc[0]["is_low_odds"] == True  # noqa: E712


def test_merge_parquet_preserves_other_sources(tmp_path):
    from scraper.boylesports import _merge_parquet
    import pandas as pd
    path = str(tmp_path / "live_odds.parquet")
    existing = pd.DataFrame([{"source": "livescorebet", "race_id": "x"}])
    existing.to_parquet(path, index=False)
    new = pd.DataFrame([{"source": "boylesports", "race_id": "y"}])
    _merge_parquet(new, path)
    out = pd.read_parquet(path)
    sources = set(out["source"])
    assert sources == {"livescorebet", "boylesports"}


def test_merge_parquet_replaces_own_source(tmp_path):
    from scraper.boylesports import _merge_parquet
    import pandas as pd
    path = str(tmp_path / "live_odds.parquet")
    pd.DataFrame([{"source": "boylesports", "race_id": "old"}]).to_parquet(path, index=False)
    _merge_parquet(pd.DataFrame([{"source": "boylesports", "race_id": "new"}]), path)
    out = pd.read_parquet(path)
    assert list(out["race_id"]) == ["new"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k "rows_to_df or merge_parquet" -v`
Expected: FAIL.

- [ ] **Step 3: Implement DataFrame + merge** (append)

```python
# ---------------------------------------------------------------------------
# DataFrame / parquet
# ---------------------------------------------------------------------------
_PARQUET_COLUMNS = [
    "fetched_at", "source", "race_id", "race_time", "venue", "market_type",
    "ew_places", "ew_reduction", "ew_margin", "horse_name", "odds_decimal",
    "sp", "is_low_odds", "currency",
]


def _rows_to_df(rows: list, fetched_at: datetime) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "fetched_at", fetched_at.astimezone(timezone.utc))
    df.insert(1, "source", "boylesports")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["race_time"] = pd.to_datetime(df["race_time"], utc=True, errors="coerce")
    df["ew_places"] = pd.array(df["ew_places"].tolist(), dtype="Int64")
    df["ew_reduction"] = pd.array(df["ew_reduction"].tolist(), dtype="Float64")
    df["ew_margin"] = pd.array(df["ew_margin"].tolist(), dtype="Float64")
    df["is_low_odds"] = df["is_low_odds"].astype(bool)
    return df[_PARQUET_COLUMNS]


def _merge_parquet(df: pd.DataFrame, path: str = _PARQUET_PATH) -> None:
    """Read-merge-write: drop this source's old rows, append fresh, write back."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            existing = existing[existing["source"] != "boylesports"]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not merge existing parquet (%s) — overwriting", exc)
    df.to_parquet(path, engine="pyarrow", index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/scraper/test_boylesports.py -k "rows_to_df or merge_parquet" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: boylesports DataFrame conversion and parquet read-merge-write"
```

---

## Task 9: `scrape()` orchestration with tiered fallback

**Files:**

- Modify: `scraper/boylesports.py`
- Test: `tests/scraper/test_boylesports.py`

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_scrape_returns_cached_when_fresh(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    cache_path = str(tmp_path / "bs.json")
    monkeypatch.setattr(bs, "_CACHE_PATH", cache_path)
    bs.CacheManager(path=cache_path).write({
        "fetched_at": bs.datetime.now(tz=bs.TZ).isoformat(),
        "rows": [{
            "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
            "venue": "Ascot", "market_type": "WIN", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "H",
            "odds_decimal": 1.2, "sp": float("nan"), "is_low_odds": True,
            "currency": "GBP",
        }],
    })
    df = bs.scrape()
    assert len(df) == 1
    assert df.iloc[0]["horse_name"] == "H"


def test_scrape_tier1_success(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    monkeypatch.setattr(bs, "_CACHE_PATH", str(tmp_path / "bs.json"))
    monkeypatch.setattr(bs, "_PARQUET_PATH", str(tmp_path / "live.parquet"))

    monkeypatch.setattr(bs.BoyleSportsClient, "get_index",
                        lambda self: {"meetings": [{"events": [{"id": "e1"}]}]})
    monkeypatch.setattr(bs.BoyleSportsClient, "get_event",
                        lambda self, eid: _sample_event())
    df = bs.scrape(force=True)
    assert len(df) == 2
    assert set(df["currency"]) == {"GBP"}


def test_scrape_falls_back_to_stale_cache(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    cache_path = str(tmp_path / "bs.json")
    monkeypatch.setattr(bs, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(bs, "_PARQUET_PATH", str(tmp_path / "live.parquet"))
    # stale cache (old timestamp)
    bs.CacheManager(path=cache_path).write({
        "fetched_at": "2000-01-01T00:00:00+00:00",
        "rows": [{
            "race_id": "old", "race_time": "2000-01-01T00:00:00+00:00",
            "venue": "Cork", "market_type": "WIN", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "Z",
            "odds_decimal": 3.0, "sp": float("nan"), "is_low_odds": False,
            "currency": "EUR",
        }],
    })
    def boom(self):
        raise bs.BotDetectedError("blocked")
    monkeypatch.setattr(bs.BoyleSportsClient, "get_index", boom)
    monkeypatch.setattr(bs, "_PlaywrightFallback_fetch", None, raising=False)
    # force all live tiers to fail
    monkeypatch.setattr(bs, "_run_live_tiers", lambda *a, **k: None)
    df = bs.scrape(force=True)
    assert df.iloc[0]["horse_name"] == "Z"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_boylesports.py -k scrape -v`
Expected: FAIL.

- [ ] **Step 3: Implement Playwright tier + `_run_live_tiers` + `scrape()`** (append)

```python
# ---------------------------------------------------------------------------
# Browser fallbacks
# ---------------------------------------------------------------------------
class PlaywrightFallback:
    def fetch(self) -> list:
        logger.warning("BoyleSports: Playwright XHR fallback active")
        rows: list = []
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def _handle(response):
                if "/horse-racing/event" not in response.url:
                    return
                try:
                    body = response.json()
                except Exception:
                    return
                rows.extend(_parse_event(body))

            page.on("response", _handle)
            page.goto(_HORSE_RACING_URL)
            page.wait_for_timeout(8000)
            browser.close()
        if not rows:
            raise ScraperError("PlaywrightFallback: no event rows captured")
        return rows


def _lxml_fetch() -> list:
    """Static-HTML DOM fallback. Best-effort; returns [] structure on failure."""
    rotator = ProxyRotator()
    proxy = rotator.next()
    mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
    with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
        resp = client.get(_HORSE_RACING_URL)
    resp.raise_for_status()
    rows = _dom_parse(resp.text)
    if not rows:
        raise ScraperError("lxml fallback: DOM yielded zero rows")
    return rows


def _dom_parse(page_source: str) -> list:
    """Parse rendered HTML into rows (shared by lxml + Selenium DOM modes)."""
    tree = lxml_html.fromstring(page_source)
    rows: list = []
    for container in tree.xpath(
        '//*[contains(@class,"race-card") or contains(@class,"race-meeting")]'
    ):
        venue_els = container.xpath('.//*[contains(@class,"venue")]/text()')
        venue = venue_els[0].strip() if venue_els else ""
        currency = currency_for_venue(venue)
        for runner in container.xpath('.//*[contains(@class,"selection")]'):
            name_els = runner.xpath('.//*[contains(@class,"runner-name")]/text()')
            odds_els = runner.xpath('.//*[contains(@class,"odds")]/text()')
            name = name_els[0].strip() if name_els else ""
            if not name:
                continue
            odds = _to_decimal(odds_els[0].strip() if odds_els else "")
            rows.append({
                "race_id": container.get("data-event-id", ""),
                "race_time": "",
                "venue": venue,
                "market_type": "WIN",
                "ew_places": None, "ew_reduction": None, "ew_margin": float("nan"),
                "horse_name": name, "odds_decimal": odds, "sp": float("nan"),
                "is_low_odds": bool(odds == odds and odds < LOW_ODDS_THRESHOLD),
                "currency": currency,
            })
    return rows


def _run_live_tiers() -> Optional[list]:
    """Try API -> Playwright -> Selenium -> lxml. Return rows or None."""
    client = BoyleSportsClient()
    # Tier 1: JSON API
    try:
        index = client.get_index()
        ids = _extract_event_ids(index)
        rows: list = []
        for eid in ids:
            try:
                rows.extend(_parse_event(client.get_event(eid)))
            except Exception as exc:
                logger.warning("BoyleSports: event %s failed: %s", eid, exc)
        if rows:
            return rows
        raise BotDetectedError("API returned no rows")
    except BotDetectedError as exc:
        logger.warning("Tier 1 failed (%s)", exc)

    # Tier 2: Playwright
    try:
        return PlaywrightFallback().fetch()
    except (ScraperError, Exception) as exc:
        logger.warning("Tier 2 (Playwright) failed (%s)", exc)

    # Tier 3: Selenium
    try:
        from scraper._selenium_fallback import SeleniumFallback
        return SeleniumFallback(
            url=_HORSE_RACING_URL,
            xhr_url_predicate=lambda u: "/horse-racing/event" in u,
            parse_xhr=_parse_event,
            dom_parse=_dom_parse,
        ).fetch()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 3 (Selenium) failed (%s)", exc)

    # Tier 4: lxml
    try:
        return _lxml_fetch()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tier 4 (lxml) failed (%s)", exc)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scrape(force: bool = False) -> pd.DataFrame:
    """Fetch live BoyleSports horse-racing odds as a standardized DataFrame."""
    cache = CacheManager(path=_CACHE_PATH)

    if not force and cache.is_fresh():
        logger.debug("BoyleSports cache hit")
        cached = cache.read()
        return _rows_to_df(cached.get("rows", []),
                           datetime.fromisoformat(cached["fetched_at"]))

    rows = _run_live_tiers()

    if rows is None:
        # Tier 5: stale cache rather than failing hard
        cached = cache.read()
        if cached and cached.get("rows"):
            logger.warning("All live tiers failed — returning STALE cache")
            return _rows_to_df(cached["rows"],
                               datetime.fromisoformat(cached["fetched_at"]))
        raise ScraperError("BoyleSports: all fetch tiers failed and no cache")

    fetched_at = datetime.now(tz=TZ)
    df = _rows_to_df(rows, fetched_at)
    _merge_parquet(df)
    cache.write({"fetched_at": fetched_at.isoformat(), "rows": rows})
    logger.debug("BoyleSports scraped %d rows", len(df))
    return df
```

- [ ] **Step 4: Run the full BoyleSports suite**

Run: `pytest tests/scraper/test_boylesports.py -v`
Expected: PASS (all). If `test_scrape_falls_back_to_stale_cache` references
`_run_live_tiers`, the monkeypatch replaces it — confirm it passes.

- [ ] **Step 5: Commit**

```bash
git add scraper/boylesports.py tests/scraper/test_boylesports.py && git commit -m "feat: boylesports scrape() with API->Playwright->Selenium->lxml->stale-cache tiers"
```

---

## Task 10: Retrofit LivescoreBet (read-merge-write, new columns, Selenium tier)

**Files:**

- Modify: `scraper/livescorebet.py`
- Test: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests** (append to test file)

```python
def test_lsb_parquet_columns_include_new_fields():
    from scraper.livescorebet import _PARQUET_COLUMNS
    assert "ew_margin" in _PARQUET_COLUMNS
    assert "is_low_odds" in _PARQUET_COLUMNS
    assert "currency" in _PARQUET_COLUMNS


def test_lsb_merge_preserves_other_sources(tmp_path):
    from scraper.livescorebet import _merge_parquet
    import pandas as pd
    path = str(tmp_path / "live.parquet")
    pd.DataFrame([{"source": "boylesports", "race_id": "x"}]).to_parquet(path, index=False)
    _merge_parquet(pd.DataFrame([{"source": "livescorebet", "race_id": "y"}]), path)
    out = pd.read_parquet(path)
    assert set(out["source"]) == {"boylesports", "livescorebet"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/scraper/test_livescorebet.py -k "parquet_columns or merge_preserves" -v`
Expected: FAIL.

- [ ] **Step 3: Update `livescorebet.py`**

3a. Add import near the other utils import:

```python
from utils.currency import currency_for_venue
```

3b. Update `LOW_ODDS_THRESHOLD` constant after `SCRAPE_INTERVAL`:

```python
LOW_ODDS_THRESHOLD: float = float(_cfg.get("low_odds_threshold", 2.0))
```

3c. In `_parse_event`, compute margin per market and extend each appended row.
Replace the row-append dict's tail so it includes the new fields. After the
existing `ew_reduction` computation in `_parse_event`, add before the selection
loop:

```python
        live_sel = [
            s for s in (market.get("selections") or [])
            if s.get("kind") != "NR" and (s.get("odds") or 0) >= 0
        ]
        win_odds = [float(s.get("odds")) if s.get("odds") is not None else float("nan")
                    for s in live_sel]
        # place_odds = 1 + (win-1)*reduction ; margin = sum(1/place)-places
        if ew_places and ew_reduction:
            _m = 0.0
            _n = 0
            for _o in win_odds:
                if _o == _o and _o > 1.0:
                    _po = 1.0 + (_o - 1.0) * ew_reduction
                    if _po > 0:
                        _m += 1.0 / _po
                        _n += 1
            ew_margin = (_m - ew_places) if _n else float("nan")
        else:
            ew_margin = float("nan")
        venue_currency = currency_for_venue(venue)
```

Then change the selection loop to iterate `live_sel` and append rows with the
new keys:

```python
        for sel in live_sel:
            horse_name = sel.get("name", "")
            if not horse_name:
                continue
            odds_val = sel.get("odds")
            odds_decimal = float(odds_val) if odds_val is not None else float("nan")
            rows.append({
                "race_id": race_id, "race_time": race_time, "venue": venue,
                "market_type": market_type, "ew_places": ew_places,
                "ew_reduction": ew_reduction, "ew_margin": ew_margin,
                "horse_name": horse_name, "odds_decimal": odds_decimal,
                "sp": float("nan"),
                "is_low_odds": bool(odds_decimal == odds_decimal
                                    and odds_decimal < LOW_ODDS_THRESHOLD),
                "currency": venue_currency,
            })
```

Apply the same new keys to the rows built in `_parse_races` and `LxmlFallback`
(add `ew_margin: float("nan")`, `is_low_odds` computed from `odds_decimal`,
`currency: currency_for_venue(venue)`).

3d. Update `_PARQUET_COLUMNS`:

```python
_PARQUET_COLUMNS = [
    "fetched_at", "source", "race_id", "race_time", "venue", "market_type",
    "ew_places", "ew_reduction", "ew_margin", "horse_name", "odds_decimal",
    "sp", "is_low_odds", "currency",
]
```

3e. In `_rows_to_df`, after the `ew_reduction` cast add:

```python
    df["ew_margin"] = pd.array(df["ew_margin"].tolist(), dtype="Float64")
    df["is_low_odds"] = df["is_low_odds"].astype(bool)
```

3f. Add `_merge_parquet` (same as BoyleSports but `source != "livescorebet"`):

```python
def _merge_parquet(df, path=_PARQUET_PATH):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        try:
            existing = pd.read_parquet(path)
            existing = existing[existing["source"] != "livescorebet"]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception as exc:
            logger.warning("Could not merge parquet (%s) — overwriting", exc)
    df.to_parquet(path, engine="pyarrow", index=False)
```

3g. In `scrape()`, replace the direct `df.to_parquet(...)` write block with:

```python
    _merge_parquet(df)
    logger.debug("Merged %d livescorebet rows into %s", len(df), _PARQUET_PATH)
```

3h. Add a Selenium tier between Tier 2 (Playwright) and Tier 3 (lxml) in
`scrape()`'s except chain:

```python
            # Tier 2.5 — Selenium
            try:
                from scraper._selenium_fallback import SeleniumFallback
                rows = SeleniumFallback(
                    url=_HORSE_RACING_URL,
                    xhr_url_predicate=lambda u: "gateway-ie.livescorebet.com" in u,
                    parse_xhr=lambda b: _parse_event(b) if "currentEvent" in b else [],
                ).fetch()
            except Exception as exc25:
                logger.warning("Tier 2.5 (Selenium) failed (%s) — trying lxml", exc25)
                # existing Tier 3 lxml block follows
```

- [ ] **Step 4: Run the LivescoreBet suite**

Run: `pytest tests/scraper/test_livescorebet.py -v`
Expected: PASS (existing + 2 new). Fix any row-shape assertions in older tests
that now expect the new columns.

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py && git commit -m "refactor: livescorebet read-merge-write parquet, ew_margin/is_low_odds/currency, selenium tier"
```

---

## Task 11: Retrofit Paddy Power Selenium tier

**Files:**

- Modify: `scraper/paddy_power.py`
- Test: `tests/scraper/test_paddy_power.py`

**Note:** Paddy Power keeps its nested-dict schema and is not part of the parquet
pipeline. It only gains a Selenium tier after `ChromeFallback`.

- [ ] **Step 1: Write the failing test** (append)

```python
def test_pp_has_selenium_fallback_symbol():
    import scraper.paddy_power as pp
    assert hasattr(pp, "_selenium_fetch")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/scraper/test_paddy_power.py -k selenium -v`
Expected: FAIL.

- [ ] **Step 3: Add Selenium tier to `paddy_power.py`**

3a. Add a module-level helper:

```python
def _selenium_fetch() -> dict:
    """Selenium tier: capture the content-managed-page JSON, return raw dict."""
    from scraper._selenium_fallback import SeleniumFallback

    captured: list = []
    SeleniumFallback(
        url=_HORSE_RACING_PAGE,
        xhr_url_predicate=lambda u: "content-managed-page" in u
        and "cardsToFetch=21149" in u,
        parse_xhr=lambda b: [b] if "attachments" in b else [],
    )
    # SeleniumFallback returns rows; here each "row" is the raw body dict.
    fb = SeleniumFallback(
        url=_HORSE_RACING_PAGE,
        xhr_url_predicate=lambda u: "content-managed-page" in u
        and "cardsToFetch=21149" in u,
        parse_xhr=lambda b: [b] if "attachments" in b else [],
    )
    bodies = fb.fetch()
    if not bodies:
        raise ScraperError("Selenium PP: no content-managed-page captured")
    return bodies[0]
```

3b. In `scrape()`, extend the fallback chain so Selenium runs if `ChromeFallback`
fails:

```python
    try:
        raw = client.get(_PP_URL)
    except BotDetectedError:
        logger.warning("Bot detected on httpx path — trying ChromeFallback")
        try:
            raw = fallback.fetch()
        except Exception as exc:
            logger.warning("ChromeFallback failed (%s) — trying Selenium", exc)
            try:
                raw = _selenium_fetch()
            except Exception as exc2:
                raise ScraperError(
                    "httpx, ChromeFallback, and Selenium all failed"
                ) from exc2
```

- [ ] **Step 4: Run the Paddy Power suite**

Run: `pytest tests/scraper/test_paddy_power.py -v`
Expected: PASS (existing + new).

- [ ] **Step 5: Commit**

```bash
git add scraper/paddy_power.py tests/scraper/test_paddy_power.py && git commit -m "feat: add Selenium fallback tier to paddy power scraper"
```

---

## Task 12: Full suite + memory update

**Files:**

- Modify: `C:\Users\mshr\.claude\projects\C--Users-mshr-Desktop-Race-Predictor-v3\memory\project-overview.md` and `MEMORY.md`

- [ ] **Step 1: Run the entire test suite**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 2: Update project memory**

Add a `scraper/boylesports.py — COMPLETE` section to `project-overview.md`
(mirroring the existing PP/LSB sections): components, tier order, new columns
(`ew_margin`, `is_low_odds`, `currency`), shared `_selenium_fallback.py`, the
parquet read-merge-write change, and the `⚠️ confirm endpoints via DevTools`
note. Update the structure block to list `boylesports.py`, `_selenium_fallback.py`,
`utils/currency.py`.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs: record BoyleSports scraper in project memory"
```

---

## Self-Review Notes

- **Spec coverage:** low-odds parse (T3) + flag (T6); raw EW terms + margin (T4/T6);
  proxy pool (T2); response validation (T5 bot guards); fallback to
  `boylesports_cache.json` (T9 stale-cache tier); standardized schema matching
  LSB (T8); Selenium for all three (T7/T9/T10/T11); currency EUR/GBP (T1/T6).
- **Endpoint placeholders** are intentional and flagged `⚠️ confirm via DevTools`
  — field names in `_parse_event` and `api_base` must be verified on first live run.
- **Type consistency:** `_PARQUET_COLUMNS` identical in BoyleSports and LivescoreBet;
  `_merge_parquet` differs only by the `source` literal; `_parse_event` returns the
  same 14-key row dict everywhere.
