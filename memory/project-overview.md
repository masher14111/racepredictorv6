---
name: project-overview
description: "Race Predictor v3 — project goals, environment, structure, and current implementation status"
metadata:
  node_type: memory
  type: project
  originSessionId: 71d442e9-10c5-49d2-89ee-48e1e6e30ef3
---

Race Predictor v3 — Python horse racing prediction tool. `C:\Users\mshr\Desktop\Race Predictor v3`.

**Why:** Third iteration; fresh start with proper structure, LightGBM model, Streamlit UI.

**How to apply:** Use when scaffolding, choosing deps, or understanding what's built vs. what's next.

## Environment

- Python 3.14, Windows 11, PowerShell (Bash also available)
- Timezone: Europe/Dublin (`utils/timezone.py` — exports `TZ`, `to_local`, `to_utc`, `now`)
- Logging: rotating file + console, `utils/logger.py` → `get_logger(name)` → `logs/race_predictor.log`
- No git repo yet

## Project Structure

```
/data/cache/        — paddy_power.json (scrape cache)
/data/races.db      — SQLite (not yet used)
/features           — (not yet built)
/models             — (not yet built)
/scraper/           — paddy_power.py ✅, livescorebet.py ✅, boylesports.py ✅, _selenium_fallback.py ✅
/ui                 — (not yet built)
/utils/             — logger.py, timezone.py, currency.py
/tests/scraper/     — test_paddy_power.py, test_livescorebet.py, test_boylesports.py, test_selenium_fallback.py
/tests/utils/       — test_currency.py (full suite 109 tests, all passing)
config.yaml
requirements.txt
```

## Key Config (config.yaml)

- `scrape_interval`: 3600s — also used as cache TTL
- `proxy_pool.enabled`: false, `proxies`: [] — proxy rotation ready but off
- `each_way_threshold`: 8.0
- `model_weights`: lightgbm=0.6, baseline=0.4

## scraper/paddy_power.py — COMPLETE (2026-06-12)

Public API: `scrape(force=False) -> dict`

**Components:**

- `CacheManager(path, ttl)` — TTL from `scrape_interval`; `is_fresh()`, `read()`, `write()`
- `ProxyRotator(cfg)` — round-robin; returns `None` when disabled/empty (silent fallback)
- `PaddyPowerClient(proxy_rotator, max_retries=3)` — httpx + Chrome headers; backoff 1→2→4s; raises `BotDetectedError` on 403/HTML/non-JSON; raises `ScraperError` after retries exhausted
- `ChromeFallback()` — Playwright CDP; intercepts `/api/` XHR from `paddypower.com/horse-racing`
- `_parse_race_time(raw)` — UTC ISO → Europe/Dublin ISO
- `_parse_events(events_data, markets_by_id)` — builds races list with race_id, race_time, venue, markets, selections

**Scrape flow:** cache hit → httpx (events + markets per race) → BotDetectedError → ChromeFallback → parse → write cache → return

**Cache schema:**

```json
{"fetched_at": "<Dublin ISO>", "races": [{"race_id": "...", "race_time": "...", "venue": "...", "markets": [{"market_type": "WIN|EACH_WAY|PLACE", "each_way_terms": {"places": 3, "reduction": 0.25} | null, "selections": [{"selection_id": "...", "horse_name": "...", "sp": 6.0, "odds_decimal": 6.5}]}]}]}
```

**⚠️ Endpoint URLs need tuning:** `_EVENTS_URL = "https://api.paddypower.com/api/v1/events"` returned connection errors on first live run. Actual endpoints must be confirmed via browser DevTools (Network tab on paddypower.com/horse-racing). Field names in `_parse_events` (`events`, `id`, `startTime`, `venue`, `eachWayTerms`, `runners`, `prices`, `decimal`) are assumed and may also need adjustment.

**Exceptions:** `BotDetectedError`, `ScraperError`

**Deps added:** `playwright`, `pytest`, `respx`, `pytest-mock` (run `playwright install chromium` once after install)

## scraper/boylesports.py — COMPLETE (2026-06-12)

Public API: `scrape(force=False) -> pd.DataFrame` (flat-row standardized schema, `source="boylesports"`).

**REWRITTEN to confirmed HTML reality (2026-06-12) and validated against live data.** It is a DOM scraper, not a JSON client.

**Focus areas delivered:**

- **Low-odds:** `_to_decimal()` parses fractional (`10/11`,`5/2`)/`EVS`/`NR` → decimal; `is_low_odds` = `odds_decimal < low_odds_threshold` (config 2.0).
- **Each-way:** terms are NOT in BoyleSports cards, so they are **cross-sourced** — `_cross_source_ew()` reads other bookies' rows from the shared parquet, matches by (venue, UTC-minute), fills `ew_places`/`ew_reduction`, and recomputes `ew_margin` from BoyleSports win odds.
- **Race status:** `resulted` from `.race-resulted`; only non-resulted (open) events are fetched for odds.
- **Region filter:** `_EXCLUDED_REGIONS` (config `boylesports.exclude_regions`, default `[virtuals]`) drops regions in `_parse_index` by href slug, case-insensitive — keeps simulated `virtuals` races (262 of 576) out of the predictor.
- **Currency:** `currency` col EUR/GBP via `utils/currency.py`. `to_eur()` ready for GUI (`gbp_eur_rate`).

**Fetch flow:** `_collect_rows(get_html)` fetches the race-card index partial → `_parse_index()` (576 races, regions, status, Dublin times) → for each open event `get_html(event_url)` → `_parse_event()` (runner anchors `a.odds[data-selectionid][data-name][data-price]`, dedupe by selectionid, drop `data-isNr=True`). Capped at `boylesports.max_events` (60).

**Tiers:** httpx (fast, fails on Cloudflare challenge) → Playwright DOM → Selenium DOM → **stale cache** → raise. `_race_time_iso` uses `TZ.localize()` (pytz — do NOT pass `tzinfo=TZ`, gives LMT −00:25).

**Schema (shared `_PARQUET_COLUMNS`, written to `data/live_odds.parquet` via read-merge-write keyed on `source`):** fetched_at, source, race_id, race_time, venue, market_type, ew_places, ew_reduction, ew_margin, horse_name, odds_decimal, sp, is_low_odds, currency. Cache: `data/boylesports_cache.json` (`{fetched_at, rows}`).

**⚠️ DevTools confirmation (2026-06-12) — original JSON-API design was WRONG. Reality:**

- Site is server-rendered ASP.NET MVC behind Cloudflare. **No JSON odds API.** Odds are HTML.
- Racing landing path: `/sports/horse-racing` (the `/horse-racing` placeholder 404s). Config updated.
- Index data endpoint (HTML partial): `GET /sports/horse-racing/race-card?partial=true&widget=true` → all meetings. Region groups class `{region}_content` (ukirefeatured/international/australia/usa/france/newzealand/virtuals); each race `div.race-card-group.race-time`, finished races carry **`.race-resulted`** (race status); `data-eventid`, `data-date`, time in `.time`; event link `/sports/horse-racing/{region}/{ddmmyy}/{course}/{HH:MM}`.
- Per-event odds: in the **event page HTML document** (no separate odds XHR). WIN market = one `data-marketid`; runner buttons `a.odds[data-selectionid][data-name][data-price][data-marketid][data-isnr][data-tradable]`. Prices are **fractional** (`10/11`,`EVS`,`7/2`,`NR`) — `_to_decimal` already handles.
- **Each-way place terms (1/4|1/5 + places) are NOT exposed in the race-card DOM** — only betslip-level/acca promo text. Resolved by cross-sourcing EW terms from other bookies (see `_cross_source_ew` above).
- Runner-anchor real attrs: `data-name`, `data-price` (fractional), `data-selectionid`, `data-marketid`, `data-isNr` (capital N — lxml lowercases), `data-tradable`. Index race anchor (`a.ui-raceCards`) carries `href` + `data-eventid` — NOT the group `<li>`. Server HTML is clean; only browser-serialized outerHTML double-encodes quotes (a parsing red herring).

## scraper/\_selenium_fallback.py — shared (2026-06-12)

Reusable `SeleniumFallback(url, xhr_url_predicate, parse_xhr, dom_parse=None)` — headless Chrome, CDP perf-log XHR capture → caller parser, optional DOM fallback. Used by all three scrapers. LivescoreBet retrofit also switched parquet to read-merge-write + gained ew_margin/is_low_odds/currency cols. Paddy Power gained a Selenium tier after ChromeFallback (keeps its nested-dict schema, not in parquet pipeline).

Spec: `docs/superpowers/specs/2026-06-12-boylesports-scraper-design.md`. Plan: `docs/superpowers/plans/2026-06-12-boylesports-scraper.md`.

## Next Steps

- Confirm real Paddy Power API endpoints via DevTools, update `_EVENTS_URL`/`_MARKETS_URL` and field names in `_parse_events`
- Build `/features` — feature engineering from scraped data
- Build `/models` — LightGBM + baseline
- Build `/ui` — Streamlit interface
