# LivescoreBet Scraper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scraper/livescorebet.py` — a three-tier horse-racing scraper (httpx + `__NEXT_DATA__` → Playwright XHR → lxml HTML) that writes flat odds data to `data/live_odds.parquet`.

**Architecture:** Module-level classes (`Throttler`, `ProxyRotator`, `LivescoreBetClient`, `PlaywrightFallback`, `LxmlFallback`, `CacheManager`) plus `_extract_next_data`, `_parse_races`, `_rows_to_df` helpers and a single public `scrape()` entry point. Mirrors `scraper/paddy_power.py` structure exactly. Cache sidecar stores flat rows as JSON; parquet is the canonical output.

**Tech Stack:** `httpx`, `playwright`, `lxml`, `pandas`, `pyarrow`, `pyyaml`, `respx`, `pytest`, `pytest-mock`

---

## File Map

| Action | Path                                 | Responsibility            |
| ------ | ------------------------------------ | ------------------------- |
| Create | `scraper/livescorebet.py`            | Full scraper module       |
| Create | `tests/scraper/test_livescorebet.py` | All unit tests            |
| Modify | `config.yaml`                        | Add `livescorebet:` block |

---

## Task 1: Add livescorebet config block

**Files:**

- Modify: `config.yaml`

- [ ] **Step 1: Add the config block**

Append to `config.yaml`:

```yaml
livescorebet:
  horse_racing_url: https://www.livescorebet.com/horse-racing
  parquet_path: data/live_odds.parquet
```

- [ ] **Step 2: Commit**

```bash
git add config.yaml
git commit -m "config: add livescorebet block"
```

---

## Task 2: Module skeleton — exceptions, constants, imports

**Files:**

- Create: `scraper/livescorebet.py`

- [ ] **Step 1: Write the failing test**

Create `tests/scraper/test_livescorebet.py`:

```python
import pytest


def test_imports():
    from scraper.livescorebet import (
        BotDetectedError,
        ScraperError,
        Throttler,
        ProxyRotator,
        LivescoreBetClient,
        PlaywrightFallback,
        LxmlFallback,
        CacheManager,
        scrape,
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/scraper/test_livescorebet.py::test_imports -v
```

Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Create the module skeleton**

Create `scraper/livescorebet.py`:

```python
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
import pandas as pd
import yaml
from lxml import html as lxml_html
from playwright.sync_api import sync_playwright

from utils.logger import get_logger
from utils.timezone import TZ, to_local

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "cache", "livescorebet.json"
)


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


_cfg = _load_config()
SCRAPE_INTERVAL: int = int(_cfg.get("scrape_interval", 3600))
_PROXY_CFG: dict = _cfg.get("proxy_pool", {})
_LSB_CFG: dict = _cfg.get("livescorebet", {})

_HORSE_RACING_URL: str = _LSB_CFG.get(
    "horse_racing_url", "https://www.livescorebet.com/horse-racing"
)
_PARQUET_PATH: str = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        _LSB_CFG.get("parquet_path", "data/live_odds.parquet"),
    )
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IE,en;q=0.9",
}

_RACE_DATA_KEYS = ("events", "races", "meetings", "data")

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BotDetectedError(Exception):
    """Raised when LivescoreBet returns 403, HTML challenge, or no __NEXT_DATA__."""


class ScraperError(Exception):
    """Raised when all fetch tiers are exhausted."""


# ---------------------------------------------------------------------------
# Stubs (filled in subsequent tasks)
# ---------------------------------------------------------------------------


class Throttler:
    pass


class ProxyRotator:
    pass


class LivescoreBetClient:
    pass


class PlaywrightFallback:
    pass


class LxmlFallback:
    pass


class CacheManager:
    pass


def scrape(force: bool = False) -> pd.DataFrame:
    raise NotImplementedError
```

- [ ] **Step 4: Run test to confirm it passes**

```
pytest tests/scraper/test_livescorebet.py::test_imports -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: livescorebet scraper skeleton"
```

---

## Task 3: Throttler

**Files:**

- Modify: `scraper/livescorebet.py` (replace `Throttler` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
import time as _time


def test_throttler_first_call_is_immediate():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=1.0)
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed < 0.1  # first call never waits


def test_throttler_second_call_waits():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=1.0)
    t.acquire()
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed >= 0.9  # allows 100ms tolerance


def test_throttler_high_rate_does_not_wait():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=100.0)
    t.acquire()
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed < 0.1
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py::test_throttler_first_call_is_immediate tests/scraper/test_livescorebet.py::test_throttler_second_call_waits tests/scraper/test_livescorebet.py::test_throttler_high_rate_does_not_wait -v
```

Expected: FAIL — `Throttler` has no `acquire()` method.

- [ ] **Step 3: Replace the Throttler stub**

In `scraper/livescorebet.py`, replace:

```python
class Throttler:
    pass
```

With:

```python
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
```

Also add `import time as _time` at the top of the file (the stub already imports `time` — rename it):

Replace `import time` with `import time as _time` in the imports block.

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py::test_throttler_first_call_is_immediate tests/scraper/test_livescorebet.py::test_throttler_second_call_waits tests/scraper/test_livescorebet.py::test_throttler_high_rate_does_not_wait -v
```

Expected: all PASS (test_throttler_second_call_waits will take ~1 second)

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: Throttler at 1 req/sec"
```

---

## Task 4: ProxyRotator

**Files:**

- Modify: `scraper/livescorebet.py` (replace `ProxyRotator` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
def test_proxy_rotator_disabled_returns_none():
    from scraper.livescorebet import ProxyRotator
    r = ProxyRotator(cfg={"enabled": False, "proxies": ["http://p1", "http://p2"]})
    assert r.next() is None


def test_proxy_rotator_empty_returns_none():
    from scraper.livescorebet import ProxyRotator
    r = ProxyRotator(cfg={"enabled": True, "proxies": []})
    assert r.next() is None


def test_proxy_rotator_cycles():
    from scraper.livescorebet import ProxyRotator
    r = ProxyRotator(cfg={"enabled": True, "proxies": ["http://p1", "http://p2"]})
    assert r.next() == "http://p1"
    assert r.next() == "http://p2"
    assert r.next() == "http://p1"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "proxy_rotator" -v
```

Expected: FAIL

- [ ] **Step 3: Replace the ProxyRotator stub**

In `scraper/livescorebet.py`, replace:

```python
class ProxyRotator:
    pass
```

With:

```python
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

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "proxy_rotator" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: ProxyRotator round-robin"
```

---

## Task 5: LivescoreBetClient

**Files:**

- Modify: `scraper/livescorebet.py` (replace `LivescoreBetClient` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
import respx
import httpx as _httpx
import pytest


@respx.mock
def test_client_returns_html_on_200():
    from scraper.livescorebet import LivescoreBetClient, Throttler, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(200, text="<html>ok</html>")
    )
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    html = client.get_page()
    assert "<html>" in html


@respx.mock
def test_client_raises_bot_detected_on_403():
    from scraper.livescorebet import LivescoreBetClient, Throttler, BotDetectedError, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(403, text="Forbidden")
    )
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    with pytest.raises(BotDetectedError):
        client.get_page()


@respx.mock
def test_client_raises_on_500():
    from scraper.livescorebet import LivescoreBetClient, Throttler, ScraperError, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(500, text="error")
    )
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    with pytest.raises(Exception):
        client.get_page()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "client" -v
```

Expected: FAIL

- [ ] **Step 3: Replace the LivescoreBetClient stub**

In `scraper/livescorebet.py`, replace:

```python
class LivescoreBetClient:
    pass
```

With:

```python
class LivescoreBetClient:
    def __init__(
        self,
        throttler: "Throttler" = None,
        proxy_rotator: "ProxyRotator" = None,
    ):
        self._throttler = throttler or Throttler()
        self._rotator = proxy_rotator or ProxyRotator()

    def get_page(self) -> str:
        self._throttler.acquire()
        proxy = self._rotator.next()
        mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
        logger.debug("GET %s proxy=%s", _HORSE_RACING_URL, proxy)
        with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
            resp = client.get(_HORSE_RACING_URL)
        if resp.status_code == 403:
            raise BotDetectedError("403 from LivescoreBet")
        resp.raise_for_status()
        return resp.text
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "client" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: LivescoreBetClient with throttle and proxy"
```

---

## Task 6: `_extract_next_data` and `_find_race_data`

**Files:**

- Modify: `scraper/livescorebet.py` (add functions after class definitions)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
def test_extract_next_data_success():
    from scraper.livescorebet import _extract_next_data
    import json
    payload = {"props": {"pageProps": {"races": [{"id": "1", "name": "Ascot"}]}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    result = _extract_next_data(html)
    assert result == payload


def test_extract_next_data_missing_tag_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    html = "<html><body>no script here</body></html>"
    with pytest.raises(BotDetectedError, match="not found"):
        _extract_next_data(html)


def test_extract_next_data_malformed_json_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    html = '<html><script id="__NEXT_DATA__" type="application/json">{bad json}</script></html>'
    with pytest.raises(BotDetectedError, match="malformed"):
        _extract_next_data(html)


def test_extract_next_data_no_race_data_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    import json
    payload = {"props": {"pageProps": {"football": []}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    with pytest.raises(BotDetectedError, match="no recognisable"):
        _extract_next_data(html)


def test_find_race_data_nested():
    from scraper.livescorebet import _find_race_data
    obj = {"a": {"b": {"events": [{"id": 1}]}}}
    result = _find_race_data(obj, "events")
    assert result == [{"id": 1}]


def test_find_race_data_empty_list_not_matched():
    from scraper.livescorebet import _find_race_data
    obj = {"events": []}
    result = _find_race_data(obj, "events")
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "next_data or find_race" -v
```

Expected: FAIL — functions not defined.

- [ ] **Step 3: Add the functions to the module**

After the `CacheManager` stub in `scraper/livescorebet.py`, add:

```python
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
        raise BotDetectedError("__NEXT_DATA__ script tag not found")
    try:
        data = json.loads(scripts[0])
    except json.JSONDecodeError as exc:
        raise BotDetectedError(f"__NEXT_DATA__ JSON malformed: {exc}") from exc
    for key in _RACE_DATA_KEYS:
        if _find_race_data(data, key) is not None:
            return data
    raise BotDetectedError("__NEXT_DATA__ contains no recognisable race data")
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "next_data or find_race" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: _extract_next_data and _find_race_data"
```

---

## Task 7: `_parse_races` and `_rows_to_df`

**Files:**

- Modify: `scraper/livescorebet.py`
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
from datetime import datetime, timezone as _tz

_SAMPLE_RAW = {
    "races": [
        {
            "id": "r1",
            "startTime": "2026-06-12T14:00:00Z",
            "venue": "Ascot",
            "markets": [
                {
                    "type": "WIN",
                    "eachWayTerms": {"places": 3, "reduction": 0.25},
                    "runners": [
                        {"name": "Thunder Bay", "price": 6.5, "sp": None},
                        {"name": "Night Owl", "price": 4.0, "sp": 4.1},
                    ],
                },
                {
                    "type": "EACH_WAY",
                    "eachWayTerms": None,
                    "runners": [
                        {"name": "Thunder Bay", "price": 6.5, "sp": None},
                    ],
                },
            ],
        }
    ]
}


def test_parse_races_row_count():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    assert len(rows) == 3  # 2 runners in WIN + 1 runner in EACH_WAY


def test_parse_races_field_names():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    expected_keys = {"race_id", "race_time", "venue", "market_type", "ew_places", "ew_reduction", "horse_name", "odds_decimal", "sp"}
    assert expected_keys == set(rows[0].keys())


def test_parse_races_each_way_terms():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    win_rows = [r for r in rows if r["market_type"] == "WIN"]
    assert win_rows[0]["ew_places"] == 3
    assert win_rows[0]["ew_reduction"] == 0.25


def test_parse_races_null_each_way_terms():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    ew_rows = [r for r in rows if r["market_type"] == "EACH_WAY"]
    assert ew_rows[0]["ew_places"] is None
    assert ew_rows[0]["ew_reduction"] is None


def test_parse_races_sp_none_becomes_nan():
    from scraper.livescorebet import _parse_races
    import math
    rows = _parse_races(_SAMPLE_RAW)
    thunder_win = next(r for r in rows if r["horse_name"] == "Thunder Bay" and r["market_type"] == "WIN")
    assert math.isnan(thunder_win["sp"])


def test_parse_races_empty_payload():
    from scraper.livescorebet import _parse_races
    rows = _parse_races({"other": "data"})
    assert rows == []


def test_rows_to_df_schema():
    from scraper.livescorebet import _parse_races, _rows_to_df
    import pandas as pd
    rows = _parse_races(_SAMPLE_RAW)
    fetched_at = datetime(2026, 6, 12, 14, 0, 0, tzinfo=_tz.utc)
    df = _rows_to_df(rows, fetched_at)
    assert list(df.columns) == [
        "fetched_at", "source", "race_id", "race_time", "venue",
        "market_type", "ew_places", "ew_reduction", "horse_name",
        "odds_decimal", "sp",
    ]
    assert df["source"].iloc[0] == "livescorebet"
    assert str(df["ew_places"].dtype) == "Int64"
    assert str(df["ew_reduction"].dtype) == "Float64"


def test_rows_to_df_empty_returns_correct_columns():
    from scraper.livescorebet import _rows_to_df
    import pandas as pd
    from datetime import datetime, timezone as _tz
    df = _rows_to_df([], datetime(2026, 6, 12, tzinfo=_tz.utc))
    assert len(df) == 0
    assert "horse_name" in df.columns
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "parse_races or rows_to_df" -v
```

Expected: FAIL

- [ ] **Step 3: Add `_parse_races` and `_rows_to_df` to the module**

After `_extract_next_data` in `scraper/livescorebet.py`, add:

```python
def _parse_race_time(raw: str) -> str:
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return to_local(dt).isoformat()


def _parse_races(raw: dict) -> list:
    """
    Normalise raw JSON → flat list of row dicts.
    Field names cover common bookmaker API patterns; may need tuning after
    inspecting the live LivescoreBet API via DevTools.
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
        if not markets:
            markets = [
                {
                    "type": "WIN",
                    "runners": race.get("runners", race.get("selections", [])),
                }
            ]

        for market in markets:
            market_type = market.get("type", market.get("marketType", "WIN")).upper()
            ew = market.get("eachWayTerms", market.get("eachWay"))
            ew_places = ew.get("places", ew.get("numberOfPlaces")) if ew else None
            ew_reduction = ew.get("reduction", ew.get("reducedOdds")) if ew else None

            runners = market.get("runners", market.get("selections", []))
            for runner in runners:
                horse_name = runner.get(
                    "name", runner.get("horseName", runner.get("selectionName", ""))
                )
                odds_raw = runner.get(
                    "price", runner.get("decimal", runner.get("oddsDecimal"))
                )
                sp_raw = runner.get("sp", runner.get("startingPrice"))
                rows.append(
                    {
                        "race_id": race_id,
                        "race_time": race_time,
                        "venue": venue,
                        "market_type": market_type,
                        "ew_places": ew_places,
                        "ew_reduction": ew_reduction,
                        "horse_name": horse_name,
                        "odds_decimal": float(odds_raw) if odds_raw is not None else float("nan"),
                        "sp": float(sp_raw) if sp_raw is not None else float("nan"),
                    }
                )
    return rows


_PARQUET_COLUMNS = [
    "fetched_at", "source", "race_id", "race_time", "venue",
    "market_type", "ew_places", "ew_reduction", "horse_name",
    "odds_decimal", "sp",
]


def _rows_to_df(rows: list, fetched_at: datetime) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_PARQUET_COLUMNS)
    df = pd.DataFrame(rows)
    df.insert(0, "fetched_at", fetched_at.astimezone(timezone.utc))
    df.insert(1, "source", "livescorebet")
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True)
    df["race_time"] = pd.to_datetime(df["race_time"], utc=True, errors="coerce")
    df["ew_places"] = pd.array(df["ew_places"].tolist(), dtype="Int64")
    df["ew_reduction"] = pd.array(df["ew_reduction"].tolist(), dtype="Float64")
    return df[_PARQUET_COLUMNS]
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "parse_races or rows_to_df" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: _parse_races and _rows_to_df"
```

---

## Task 8: CacheManager

**Files:**

- Modify: `scraper/livescorebet.py` (replace `CacheManager` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
import os
import tempfile


def test_cache_manager_is_fresh_false_when_missing():
    from scraper.livescorebet import CacheManager
    with tempfile.TemporaryDirectory() as d:
        cm = CacheManager(path=os.path.join(d, "lsb.json"), ttl=3600)
        assert cm.is_fresh() is False


def test_cache_manager_is_fresh_true_within_ttl():
    from scraper.livescorebet import CacheManager
    import json
    from datetime import datetime, timezone as _tz
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        data = {"fetched_at": datetime.now(_tz.utc).isoformat(), "rows": []}
        with open(path, "w") as f:
            json.dump(data, f)
        cm = CacheManager(path=path, ttl=3600)
        assert cm.is_fresh() is True


def test_cache_manager_is_fresh_false_when_stale():
    from scraper.livescorebet import CacheManager
    import json
    from datetime import datetime, timezone as _tz, timedelta
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        old_time = (datetime.now(_tz.utc) - timedelta(hours=2)).isoformat()
        data = {"fetched_at": old_time, "rows": []}
        with open(path, "w") as f:
            json.dump(data, f)
        cm = CacheManager(path=path, ttl=3600)
        assert cm.is_fresh() is False


def test_cache_manager_write_and_read():
    from scraper.livescorebet import CacheManager
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        cm = CacheManager(path=path, ttl=3600)
        cm.write({"fetched_at": "2026-06-12T14:00:00+00:00", "rows": [{"horse_name": "X"}]})
        data = cm.read()
        assert data["rows"][0]["horse_name"] == "X"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "cache_manager" -v
```

Expected: FAIL

- [ ] **Step 3: Replace the CacheManager stub**

In `scraper/livescorebet.py`, replace:

```python
class CacheManager:
    pass
```

With:

```python
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

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "cache_manager" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: CacheManager"
```

---

## Task 9: PlaywrightFallback

**Files:**

- Modify: `scraper/livescorebet.py` (replace `PlaywrightFallback` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
def test_playwright_fallback_returns_payload(mocker):
    from scraper.livescorebet import PlaywrightFallback

    mock_response = mocker.MagicMock()
    mock_response.url = "https://www.livescorebet.com/api/horse-racing"
    mock_response.json.return_value = {"races": [{"id": "1", "venue": "Ascot"}]}

    mock_page = mocker.MagicMock()
    mock_browser = mocker.MagicMock()
    mock_browser.new_page.return_value = mock_page

    captured_handler = {}

    def fake_on(event, handler):
        if event == "response":
            captured_handler["fn"] = handler

    mock_page.on.side_effect = fake_on

    def fake_goto(url):
        captured_handler["fn"](mock_response)

    mock_page.goto.side_effect = fake_goto

    mock_pw = mocker.MagicMock()
    mock_pw.__enter__ = mocker.MagicMock(return_value=mock_pw)
    mock_pw.__exit__ = mocker.MagicMock(return_value=False)
    mock_pw.chromium.launch.return_value = mock_browser

    mocker.patch("scraper.livescorebet.sync_playwright", return_value=mock_pw)

    fb = PlaywrightFallback()
    result = fb.fetch()
    assert result == {"races": [{"id": "1", "venue": "Ascot"}]}


def test_playwright_fallback_raises_when_no_data(mocker):
    from scraper.livescorebet import PlaywrightFallback, ScraperError

    mock_page = mocker.MagicMock()
    mock_browser = mocker.MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_page.on.return_value = None

    mock_pw = mocker.MagicMock()
    mock_pw.__enter__ = mocker.MagicMock(return_value=mock_pw)
    mock_pw.__exit__ = mocker.MagicMock(return_value=False)
    mock_pw.chromium.launch.return_value = mock_browser

    mocker.patch("scraper.livescorebet.sync_playwright", return_value=mock_pw)

    fb = PlaywrightFallback()
    with pytest.raises(ScraperError, match="no race data"):
        fb.fetch()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "playwright" -v
```

Expected: FAIL

- [ ] **Step 3: Replace the PlaywrightFallback stub**

In `scraper/livescorebet.py`, replace:

```python
class PlaywrightFallback:
    pass
```

With:

```python
class PlaywrightFallback:
    def fetch(self) -> dict:
        logger.warning("Bot detected — triggering Playwright XHR fallback")
        captured: Optional[dict] = None

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            def _handle_response(response) -> None:
                nonlocal captured
                if captured is not None:
                    return
                if not any(t in response.url for t in ("/api/", "/horse-racing/")):
                    return
                try:
                    body = response.json()
                except Exception:
                    return
                for key in _RACE_DATA_KEYS:
                    if _find_race_data(body, key) is not None:
                        captured = body
                        logger.debug(
                            "Playwright captured race data from %s", response.url
                        )
                        return

            page.on("response", _handle_response)
            page.goto(_HORSE_RACING_URL)
            page.wait_for_timeout(8000)
            browser.close()

        if captured is None:
            raise ScraperError(
                "PlaywrightFallback: no race data captured in XHR responses"
            )
        return captured
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "playwright" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: PlaywrightFallback XHR interception"
```

---

## Task 10: LxmlFallback

**Files:**

- Modify: `scraper/livescorebet.py` (replace `LxmlFallback` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
_SAMPLE_HTML = b"""
<html><body>
  <div class="race-card" data-race-id="r99">
    <span class="race-title">Cheltenham</span>
    <span class="race-time">15:30</span>
    <div class="runner">
      <span class="horse-name">Fast Buck</span>
      <span class="odds">5.5</span>
    </div>
    <div class="runner">
      <span class="horse-name">Silver Star</span>
      <span class="odds">3.0</span>
    </div>
  </div>
</body></html>
"""


@respx.mock
def test_lxml_fallback_extracts_runners(mocker):
    from scraper.livescorebet import LxmlFallback, Throttler, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(200, content=_SAMPLE_HTML)
    )
    fb = LxmlFallback(throttler=Throttler(rate=100.0))
    rows = fb.fetch()
    horse_names = [r["horse_name"] for r in rows]
    assert "Fast Buck" in horse_names
    assert "Silver Star" in horse_names


@respx.mock
def test_lxml_fallback_raises_when_no_races(mocker):
    from scraper.livescorebet import LxmlFallback, Throttler, ScraperError, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(200, content=b"<html><body>nothing</body></html>")
    )
    fb = LxmlFallback(throttler=Throttler(rate=100.0))
    with pytest.raises(ScraperError, match="zero races"):
        fb.fetch()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "lxml" -v
```

Expected: FAIL

- [ ] **Step 3: Replace the LxmlFallback stub**

In `scraper/livescorebet.py`, replace:

```python
class LxmlFallback:
    pass
```

With:

```python
class LxmlFallback:
    def __init__(
        self,
        throttler: "Throttler" = None,
        proxy_rotator: "ProxyRotator" = None,
    ):
        self._throttler = throttler or Throttler()
        self._rotator = proxy_rotator or ProxyRotator()

    def fetch(self) -> list:
        logger.warning(
            "JSON unavailable — lxml HTML fallback active (partial data expected)"
        )
        self._throttler.acquire()
        proxy = self._rotator.next()
        mounts = {"all://": httpx.HTTPTransport(proxy=proxy)} if proxy else {}
        with httpx.Client(headers=_HEADERS, mounts=mounts, timeout=15) as client:
            resp = client.get(_HORSE_RACING_URL)
        resp.raise_for_status()
        tree = lxml_html.fromstring(resp.content)

        rows = []
        # Selectors refined against live HTML during implementation.
        race_containers = tree.xpath(
            '//*[contains(@class,"race-card") or contains(@class,"race-event")]'
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
                        "ew_places": None,
                        "ew_reduction": None,
                        "horse_name": horse_name,
                        "odds_decimal": odds_decimal,
                        "sp": float("nan"),
                    }
                )

        if not rows:
            raise ScraperError("LxmlFallback: DOM yielded zero races")
        return rows
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "lxml" -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: LxmlFallback HTML parser"
```

---

## Task 11: `scrape()` — integration, cache, and parquet write

**Files:**

- Modify: `scraper/livescorebet.py` (replace `scrape` stub)
- Modify: `tests/scraper/test_livescorebet.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/scraper/test_livescorebet.py`:

```python
def test_scrape_returns_dataframe_from_cache(tmp_path, mocker):
    from scraper.livescorebet import scrape, _PARQUET_COLUMNS
    import json
    from datetime import datetime, timezone as _tz

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    cache_data = {
        "fetched_at": datetime.now(_tz.utc).isoformat(),
        "rows": [
            {
                "race_id": "r1", "race_time": "2026-06-12T14:00:00+00:00",
                "venue": "Ascot", "market_type": "WIN",
                "ew_places": None, "ew_reduction": None,
                "horse_name": "Speedy", "odds_decimal": 5.0, "sp": float("nan"),
            }
        ],
    }
    cache_path.write_text(json.dumps(cache_data))
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    df = scrape()
    assert len(df) == 1
    assert df["horse_name"].iloc[0] == "Speedy"
    assert list(df.columns) == _PARQUET_COLUMNS
    # Cache hit should NOT write parquet
    assert not parquet_path.exists()


def test_scrape_tier1_success_writes_parquet(tmp_path, mocker):
    from scraper.livescorebet import scrape, _HORSE_RACING_URL, _PARQUET_COLUMNS
    import json

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    html_payload = {"races": [{"id": "r1", "startTime": "2026-06-12T14:00:00Z", "venue": "York", "markets": [{"type": "WIN", "eachWayTerms": None, "runners": [{"name": "Blaze", "price": 3.0, "sp": None}]}]}]}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(html_payload)}</script></html>'

    mock_client = mocker.MagicMock()
    mock_client.get_page.return_value = html
    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback")
    mocker.patch("scraper.livescorebet.LxmlFallback")

    df = scrape(force=True)
    assert len(df) == 1
    assert df["venue"].iloc[0] == "York"
    assert parquet_path.exists()
    import pandas as pd
    written = pd.read_parquet(str(parquet_path))
    assert list(written.columns) == _PARQUET_COLUMNS


def test_scrape_falls_through_to_playwright(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_page.side_effect = BotDetectedError("403")

    playwright_payload = {"races": [{"id": "r2", "startTime": "2026-06-12T15:00:00Z", "venue": "Doncaster", "markets": [{"type": "WIN", "eachWayTerms": None, "runners": [{"name": "Night Run", "price": 7.0, "sp": None}]}]}]}
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.return_value = playwright_payload

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback")

    df = scrape(force=True)
    assert df["venue"].iloc[0] == "Doncaster"


def test_scrape_falls_through_to_lxml(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError, ScraperError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_page.side_effect = BotDetectedError("403")
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.side_effect = ScraperError("no XHR data")
    mock_lxml = mocker.MagicMock()
    mock_lxml.fetch.return_value = [
        {"race_id": "r3", "race_time": "", "venue": "Kempton", "market_type": "WIN",
         "ew_places": None, "ew_reduction": None, "horse_name": "Starfire",
         "odds_decimal": 2.5, "sp": float("nan")}
    ]

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback", return_value=mock_lxml)

    df = scrape(force=True)
    assert df["horse_name"].iloc[0] == "Starfire"


def test_scrape_raises_when_all_tiers_fail(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError, ScraperError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_page.side_effect = BotDetectedError("403")
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.side_effect = ScraperError("no XHR")
    mock_lxml = mocker.MagicMock()
    mock_lxml.fetch.side_effect = ScraperError("zero races")

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback", return_value=mock_lxml)

    with pytest.raises(ScraperError, match="All three fetch tiers failed"):
        scrape(force=True)
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/scraper/test_livescorebet.py -k "scrape" -v
```

Expected: FAIL — `scrape` raises `NotImplementedError`.

- [ ] **Step 3: Replace the `scrape` stub**

In `scraper/livescorebet.py`, replace:

```python
def scrape(force: bool = False) -> pd.DataFrame:
    raise NotImplementedError
```

With:

```python
def scrape(force: bool = False) -> pd.DataFrame:
    """
    Fetch live LivescoreBet horse racing odds.

    Returns a pd.DataFrame with the live_odds schema. Raises ScraperError if
    all three fetch tiers are exhausted.
    """
    cache = CacheManager()

    if not force and cache.is_fresh():
        logger.debug("Cache hit — returning cached data")
        cached = cache.read()
        rows = cached.get("rows", [])
        return _rows_to_df(rows, datetime.fromisoformat(cached["fetched_at"]))

    throttler = Throttler()
    rotator = ProxyRotator()
    client = LivescoreBetClient(throttler=throttler, proxy_rotator=rotator)
    playwright_fb = PlaywrightFallback()
    lxml_fb = LxmlFallback(throttler=throttler, proxy_rotator=rotator)

    rows: Optional[list] = None

    # Tier 1 — httpx + __NEXT_DATA__
    try:
        logger.debug("Tier 1: httpx + __NEXT_DATA__")
        html = client.get_page()
        raw = _extract_next_data(html)
        rows = _parse_races(raw)
        logger.debug("Tier 1 succeeded: %d rows", len(rows))
    except BotDetectedError as exc:
        logger.warning("Tier 1 failed (%s) — trying Playwright", exc)

        # Tier 2 — Playwright XHR
        try:
            logger.debug("Tier 2: Playwright XHR interception")
            raw = playwright_fb.fetch()
            rows = _parse_races(raw)
            logger.debug("Tier 2 succeeded: %d rows", len(rows))
        except ScraperError as exc2:
            logger.warning("Tier 2 failed (%s) — trying lxml", exc2)

            # Tier 3 — lxml HTML
            try:
                logger.debug("Tier 3: lxml HTML fallback")
                rows = lxml_fb.fetch()
                logger.debug("Tier 3 succeeded: %d rows", len(rows))
            except ScraperError as exc3:
                raise ScraperError("All three fetch tiers failed") from exc3

    fetched_at = datetime.now(tz=TZ)
    df = _rows_to_df(rows, fetched_at)

    # Write parquet
    os.makedirs(os.path.dirname(_PARQUET_PATH), exist_ok=True)
    df.to_parquet(_PARQUET_PATH, engine="pyarrow", index=False)
    logger.debug("Wrote %d rows to %s", len(df), _PARQUET_PATH)

    # Write cache
    cache.write({"fetched_at": fetched_at.isoformat(), "rows": rows})

    return df
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/scraper/test_livescorebet.py -k "scrape" -v
```

Expected: all PASS

- [ ] **Step 5: Run the full test suite**

```
pytest tests/scraper/test_livescorebet.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add scraper/livescorebet.py tests/scraper/test_livescorebet.py
git commit -m "feat: scrape() with three-tier fallback and parquet write"
```

---

## Task 12: Full suite smoke check

**Files:** none (read-only verification)

- [ ] **Step 1: Run all project tests**

```
pytest tests/ -v
```

Expected: all PASS (paddy_power tests unaffected)

- [ ] **Step 2: Verify parquet schema from Python REPL**

```
python -c "
import pandas as pd, json
df = pd.read_parquet('data/live_odds.parquet') if __import__('os').path.exists('data/live_odds.parquet') else None
print('parquet exists:', df is not None)
if df is not None: print(df.dtypes)
"
```

Expected: either `parquet exists: False` (no live run yet, which is fine) or correct dtypes printed.

- [ ] **Step 3: Final commit if any loose files**

```bash
git status
# commit anything unstaged
```
