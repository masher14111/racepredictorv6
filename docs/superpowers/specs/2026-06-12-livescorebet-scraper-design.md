# LivescoreBet Scraper — Design Spec

**Date:** 2026-06-12
**Module:** `scraper/livescorebet.py`
**Status:** Approved

---

## Overview

A single-module scraper that fetches live horse racing cards and odds from LivescoreBet. Follows the same structural pattern as `scraper/paddy_power.py`. Three fetch tiers are attempted in order:

1. `httpx` GET → extract JSON from `<script id="__NEXT_DATA__">` in the page HTML
2. Playwright (CDP) → intercept XHR/fetch responses from a real Chromium session
3. `lxml` HTML → parse race containers and odds directly from the DOM

Results are normalised to a flat `pd.DataFrame` and written to `data/live_odds.parquet`. A JSON sidecar cache (same `CacheManager` pattern as Paddy Power) avoids redundant fetches within the TTL window.

---

## Architecture & File Layout

```
scraper/
  __init__.py
  paddy_power.py   (existing)
  livescorebet.py  (new)

data/
  cache/
    livescorebet.json   (raw JSON sidecar cache)
  live_odds.parquet     (flat normalised output)
```

`livescorebet.py` contains these internal components and one public function:

| Component                  | Responsibility                                                                                     |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `Throttler`                | Token-bucket at 1 req/sec; `acquire()` blocks until slot is free                                   |
| `ProxyRotator`             | Round-robin over `config.proxy_pool.proxies`; returns `None` when disabled/empty                   |
| `LivescoreBetClient`       | `httpx` GET with Chrome headers + throttle + proxy; raises `BotDetectedError` on 403/HTML/non-JSON |
| `PlaywrightFallback`       | Headless Chromium via Playwright; intercepts XHR/fetch responses containing race data              |
| `LxmlFallback`             | Re-GETs the page via `httpx`; extracts race/odds data from DOM with `lxml.html` XPath              |
| `_extract_next_data(html)` | Pulls JSON from `<script id="__NEXT_DATA__" type="application/json">`                              |
| `_parse_races(raw)`        | Normalises raw JSON → flat list of row dicts matching the parquet schema                           |
| `scrape(force)`            | Public entry point: cache → tier 1 → tier 2 → tier 3 → parquet write → return DataFrame            |

---

## Target URL & Headers

```python
_HORSE_RACING_URL = "https://www.livescorebet.com/horse-racing"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-IE,en;q=0.9",
}
```

Both URL and headers are overridable via `config.yaml` under the `livescorebet` block (see Config section).

---

## Three-Tier Fetch Strategy

### Tier 1 — httpx + `__NEXT_DATA__`

`LivescoreBetClient.get_page()` fetches the horse-racing page HTML. `_extract_next_data()` finds `<script id="__NEXT_DATA__" type="application/json">` and parses its contents. If the tag is absent, malformed, or the parsed tree contains no race data, `BotDetectedError` is raised to trigger tier 2.

### Tier 2 — Playwright XHR interception

`PlaywrightFallback.fetch()` launches headless Chromium, navigates to `_HORSE_RACING_URL`, and registers a `response` event listener. Any response whose URL contains `/api/` or `/horse-racing/` and whose body parses as JSON is inspected for race data. A payload is considered valid if its top-level keys include any of `events`, `races`, `meetings`, or `data`, and the value is a non-empty list. Waits up to 8 seconds after page load. Returns the first valid payload. Raises `ScraperError` if no valid payload is captured within the timeout.

### Tier 3 — lxml HTML fallback

`LxmlFallback.fetch()` re-GETs the page via `httpx` (throttled) and parses with `lxml.html`. XPath selectors target race card containers, runner names, and odds elements. Exact selectors are determined during implementation by inspecting the live HTML structure; implementation will start with common patterns (e.g. elements with class tokens containing `race`, `runner`, `odds`) and harden them against the actual DOM. Returns whatever it can extract; logs a `WARNING` that field coverage may be partial (e.g. `sp` will be `NaN`, `ew_places` will be `pd.NA`). Raises `ScraperError` if the DOM yields zero races.

**Throttle scope:** `Throttler.acquire()` is called before every `httpx` request made by `LivescoreBetClient` and `LxmlFallback`. Playwright tier is self-throttled by browser page load time.

---

## Data Model — Parquet Schema

File: `data/live_odds.parquet` (overwritten on every successful fresh fetch).
Written via `pandas.DataFrame.to_parquet(engine="pyarrow")`.

| Column         | dtype                 | Notes                                                        |
| -------------- | --------------------- | ------------------------------------------------------------ |
| `fetched_at`   | `datetime64[ns, UTC]` | Scrape timestamp, Dublin-local → UTC                         |
| `source`       | `str`                 | `"livescorebet"` — allows future merge with Paddy Power rows |
| `race_id`      | `str`                 | Unique race identifier from JSON payload                     |
| `race_time`    | `datetime64[ns, UTC]` | Race start time, Dublin-local → UTC                          |
| `venue`        | `str`                 | Course name                                                  |
| `market_type`  | `str`                 | `"WIN"`, `"EACH_WAY"`, or `"PLACE"`                          |
| `ew_places`    | `Int64` (nullable)    | Each-way place terms; `pd.NA` when not offered               |
| `ew_reduction` | `Float64` (nullable)  | Each-way reduction factor; `pd.NA` when not offered          |
| `horse_name`   | `str`                 | Runner name                                                  |
| `odds_decimal` | `float`               | Current decimal odds                                         |
| `sp`           | `float`               | Starting price; `NaN` if not yet available                   |

**Write behaviour:** `scrape()` always overwrites `live_odds.parquet` with the latest fetch. The `fetched_at` column timestamps the snapshot. Callers wanting history should read-and-concat before calling `scrape()`.

---

## Cache Layer

Sidecar cache at `data/cache/livescorebet.json` stores the raw parsed race list. Uses `CacheManager` (same class pattern as `paddy_power.py`) with TTL from `scrape_interval` in `config.yaml`. Cache hit skips all three fetch tiers and reconstructs the DataFrame from the cached JSON. The parquet file is only written on a fresh fetch — cache hits do not touch it.

Cache schema:

```json
{
  "fetched_at": "2026-06-12T14:30:00+01:00",
  "races": [
    {
      "race_id": "...",
      "race_time": "...",
      "venue": "...",
      "markets": [
        {
          "market_type": "WIN",
          "each_way_terms": { "places": 3, "reduction": 0.25 },
          "selections": [
            { "horse_name": "...", "odds_decimal": 6.5, "sp": null }
          ]
        }
      ]
    }
  ]
}
```

---

## Throttling

`Throttler` uses a simple time-delta approach: records `_last_call` timestamp; `acquire()` sleeps for `max(0, 1.0 - elapsed)` seconds before returning. Not thread-safe by design — this scraper runs single-threaded.

---

## Error Handling

| Exception          | When raised                                               | Handling                 |
| ------------------ | --------------------------------------------------------- | ------------------------ |
| `BotDetectedError` | 403, HTML body, `__NEXT_DATA__` absent/empty              | Drops to Playwright tier |
| `ScraperError`     | All three tiers exhausted, or each tier yields zero races | Propagated to caller     |

Logging via `get_logger(__name__)`:

| Level     | When                                                                     |
| --------- | ------------------------------------------------------------------------ |
| `DEBUG`   | Each request attempt, proxy used, tier entered, cache hit/miss           |
| `WARNING` | Bot detected (tier 1→2 switch), lxml fallback active (partial data risk) |
| `ERROR`   | All tiers exhausted                                                      |

---

## Config

New block to add to `config.yaml`:

```yaml
livescorebet:
  horse_racing_url: https://www.livescorebet.com/horse-racing
  parquet_path: data/live_odds.parquet
```

Both keys have hardcoded defaults in the module so existing `config.yaml` files without the block still work.

`proxy_pool` and `scrape_interval` are shared with `paddy_power.py` — no duplication.

---

## Public API

```python
from scraper.livescorebet import scrape

df = scrape()            # returns pd.DataFrame; raises ScraperError on total failure
df = scrape(force=True)  # bypass TTL, always fetch fresh
```

Returns an empty DataFrame (correct columns, zero rows) if cache is fresh but contained no races. The parquet file is written as a side-effect of every successful fresh fetch.

---

## Dependencies

Already in `requirements.txt`: `httpx`, `pyyaml`, `lxml`, `pandas`, `playwright`.
No new dependencies required.
`playwright install chromium` must be run once after install (already noted in Paddy Power spec).
