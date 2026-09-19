# Paddy Power Scraper — Design Spec

**Date:** 2026-06-12  
**Module:** `scraper/paddy_power.py`  
**Status:** Approved

---

## Overview

A single-module scraper that fetches live horse racing cards and odds from Paddy Power's internal JSON API. Primary path uses `httpx` with browser-mimicking headers. When bot detection triggers, falls back to Playwright (Chrome DevTools Protocol) to intercept XHR traffic from a real Chrome session. Results are cached to `data/cache/paddy_power.json` with a TTL equal to `scrape_interval` from `config.yaml`.

---

## Architecture & File Layout

```
scraper/
  __init__.py
  paddy_power.py

data/
  cache/
    paddy_power.json
```

`paddy_power.py` contains four internal components and one public function:

| Component          | Responsibility                                                                     |
| ------------------ | ---------------------------------------------------------------------------------- |
| `CacheManager`     | Read/write `paddy_power.json`; TTL check via `fetched_at` vs `scrape_interval`     |
| `ProxyRotator`     | Cycle through `config.proxy_pool.proxies`; return `None` when disabled/empty       |
| `PaddyPowerClient` | `httpx` requests with browser headers, exponential backoff retry, proxy support    |
| `ChromeFallback`   | Playwright browser session (CDP) to intercept JSON from network traffic            |
| `scrape()`         | Public entry point: cache → httpx → chrome fallback → parse → cache write → return |

---

## Endpoint Discovery

Endpoints are reverse-engineered from Paddy Power's live site frontend traffic:

- **Events list:** `https://api.paddypower.com/api/v1/events?categoryId=21` (horse racing)
- **Race markets:** `https://api.paddypower.com/api/v1/competitions/{race_id}/markets`

The `PaddyPowerClient` uses headers that mimic Chrome on Windows:

```python
{
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-IE,en;q=0.9",
    "Referer": "https://www.paddypower.com/horse-racing",
}
```

---

## Data Model

Both the cache file and the `scrape()` return value use this structure:

```json
{
  "fetched_at": "2026-06-12T14:30:00+01:00",
  "races": [
    {
      "race_id": "12345",
      "race_time": "2026-06-12T15:00:00+01:00",
      "venue": "Ascot",
      "markets": [
        {
          "market_type": "WIN",
          "each_way_terms": { "places": 3, "reduction": 0.25 },
          "selections": [
            {
              "selection_id": "98765",
              "horse_name": "Mighty Oak",
              "sp": 6.0,
              "odds_decimal": 6.0
            }
          ]
        }
      ]
    }
  ]
}
```

**Field notes:**

- `race_time` — converted to Europe/Dublin timezone via `utils/timezone.py`
- `odds_decimal` — live odds as decimal; `sp` (starting price) stored separately as it may differ
- `each_way_terms` — `null` when not offered for a market
- `market_type` — stored as-is from API; expected values: `"WIN"`, `"EACH_WAY"`, `"PLACE"`

---

## Request & Retry Strategy

- **Retries:** 3 attempts, exponential backoff: `1s → 2s → 4s`
- **Retry on:** `httpx.TimeoutException`, `httpx.ConnectError`, HTTP 5xx
- **Immediate fail (no retry):** HTTP 403 or HTML response body → raise `BotDetectedError`
- **Proxy:** injected per-request from `ProxyRotator`; falls back to direct connection when pool is disabled or empty

---

## Fallback: Playwright (Chrome DevTools Protocol)

Triggered only on `BotDetectedError`. Implemented using **Playwright** (Python), which drives Chrome via the Chrome DevTools Protocol — the same protocol used by the `chrome-devtools-mcp` plugin. `playwright` will be added to `requirements.txt`.

Steps:

1. Launch a Chromium browser instance via `playwright`
2. Navigate to the Paddy Power horse racing page
3. Intercept all `fetch`/`XHR` responses whose URL contains `/api/`
4. Extract JSON response bodies directly from captured network traffic
5. Parse using the same parser as the `httpx` path

Note: `chrome-devtools-mcp` tools are only accessible within Claude Code sessions and cannot be called from standalone Python code at runtime. Playwright provides equivalent capability as a Python library.

---

## Error Handling

| Exception          | When raised                                       | Handling                  |
| ------------------ | ------------------------------------------------- | ------------------------- |
| `BotDetectedError` | 403 or Cloudflare HTML from `PaddyPowerClient`    | Triggers `ChromeFallback` |
| `ScraperError`     | Both `PaddyPowerClient` and `ChromeFallback` fail | Propagated to caller      |

Logging via `utils/logger.py`:

- `DEBUG` — each attempt, proxy used, cache hit/miss
- `WARNING` — bot detection triggered, falling back to Chrome
- `ERROR` — total failure, both paths exhausted

---

## Public API

```python
from scraper.paddy_power import scrape

data = scrape()           # returns parsed dict; raises ScraperError on total failure
data = scrape(force=True) # bypass TTL, always fetch fresh
```

**Cache TTL behaviour:**

- `CacheManager` reads `fetched_at` from existing cache on every call
- If age < `scrape_interval` (from `config.yaml`): return cached data, no HTTP calls
- If stale or missing: fetch → parse → overwrite cache → return
- `force=True` skips the TTL check entirely

`scrape_interval` is read once from `config.yaml` at module import time, consistent with the pattern in `utils/timezone.py`.

---

## Dependencies

Already in `requirements.txt`: `httpx`, `pyyaml`, `pytz`.  
New dependency to add: `playwright` (for `ChromeFallback`; also requires `playwright install chromium` post-install).
