# Race Predictor v3 — Scrapers Review

_Reviewed: 2026-06-13 | All scraper files read in full, cross-checked against utils layer_

---

## Paddy Power

**Verdict: FUNCTIONAL but rate limiter not wired; endpoint still unconfirmed.**

**What works:**

- `_PP_URL` now targets `apisms.paddypower.com/smspp/content-managed-page/v7` with real query params (`_ak=vsd0Rm5ph2sS2uaK`, `betexRegion=IRL`, `cardsToFetch=21149`). Much more specific than the original stub.
- `CacheManager` is a correct thin adapter over `utils.cache.Cache`. `is_fresh()`, `read()`, `write()` all route through the central cache. Cache key = `paddy_power`, lives in `data/cache/`.
- Fallback chain: httpx → `ChromeFallback` (Playwright, validates `"attachments" in body`) → `SeleniumFallback` (CDP XHR capture). All three tiers pass the same `_parse_pp_response` parser.
- `BotDetectedError` raised on 403, HTML content-type, and JSON decode failure — all correctly re-routed to fallback before retry exhaustion.
- `_parse_race_time` uses `to_local()` (Dublin TZ) — correct.
- `_parse_pp_response` uses `.get()` with defaults throughout — will silently yield empty/None data on wrong field names rather than crashing.

**Known gaps / bugs:**

1. **⚠️ ENDPOINT STILL UNCONFIRMED** — field names in `_parse_pp_response` (`attachments.races[id].venue`, `mkt.marketTime`, `runner.winRunnerOdds.trueOdds.decimalOdds.decimalOdds`, etc.) are assumed. If the real API uses different keys, the parser silently returns empty races. Must be confirmed via DevTools before trusting the output.

2. **No rate limiter acquisition** (`paddy_power.py` entire file): `PaddyPowerClient.get()` does exponential sleep only on _error_ paths (backoff). On a successful request sequence there is zero throttling. The config has `apisms.paddypower.com: rps: 0.5` but `get_rate_limiter().acquire_for_url()` is never called. Fix: wrap the `httpx.Client.get()` call in `with get_rate_limiter().acquire_for_url(url):`.

3. **`_parse_race_time` unguarded** (`paddy_power.py:165`): `datetime.fromisoformat(raw.replace("Z", "+00:00"))` raises `ValueError` if `raw` is `None`, `""`, or non-ISO. Called inside `_parse_pp_response` loop — one bad `marketTime` aborts the whole parse. Add try/except with fallback `""`.

4. **`_parse_pp_response` iterates `markets_dict` and looks up `races_meta[race_id]`** but does not guard against a race_id present in markets but absent from races_meta — `.get(race_id, {}).get("venue", "")` handles this, so venue would silently be `""`. Low risk.

**Integration status:** cache ✓ | rate_limiter ✗ | proxy_manager ✓

---

## LivescoreBet

**Verdict: SOLID parse logic; production rate limiter bypassed by Throttler injection.**

**What works:**

- API field names confirmed via DevTools 2026-06-12. `_MARKET_GROUP_TYPE` mapping and `currentEvent` / `meetingsToday` / `childs` walk are correct.
- `CacheManager` correct adapter. Cache key = `livescorebet`, lives in `data/cache/`.
- `_parse_event` EW reduction: `round(1.0 / int(ew_odds_str))` where `ew_odds_str` is the denominator (e.g. `"4"` → 0.25) — correct UK/Ireland EW terms.
- `_ew_margin` computes `sum(1/place_odds) - places` — correct overround formula.
- `_rows_to_df` uses nullable `Int64`/`Float64` arrays — correct for NaN-safe parquet.
- Merge uses read-merge-write (drop livescorebet rows, concat, write) — atomic-ish.
- Four-tier fallback chain (gateway API → Playwright XHR → Selenium XHR → lxml HTML) is the most robust of all scrapers.

**Bugs:**

1. **Production rate limiter bypassed** (`livescorebet.py:667-668`): `scrape()` constructs `throttler = Throttler()` (1 req/s local token) and injects it into `LivescoreBetClient(throttler=throttler)`. When `throttler is not None`, `_acquire_slot()` uses the local Throttler and skips `get_rate_limiter()`. Tests inject `Throttler(rate=100.0)` to bypass limiting — fine. But production should NOT inject a throttler; it should pass `throttler=None` (or omit it) so `_acquire_slot()` falls through to the central rate limiter. Same applies to `LxmlFallback(throttler=throttler)` on line 671. **Fix:** remove `throttler = Throttler()` from `scrape()` and delete the `throttler=` argument at lines 668 and 671.

2. **`cache.write()` with NaN rows** (`livescorebet.py:730`): `rows` dicts contain `float("nan")` for `sp`, `ew_margin`, `ew_reduction`. Python's `json` module writes `NaN` as the literal `NaN` (not valid JSON). While Python reads it back, external tools (JS, jq, strict JSON parsers) will reject the cache file. Consider replacing `float("nan")` with `None` before caching.

3. **`_parse_race_time` unguarded** (same issue as PP): `datetime.fromisoformat(raw.replace(" ", "T") + "Z")` is tried in `_parse_event` with a try/except fallback — this one is fine. The standalone `_parse_race_time(raw)` used in `_parse_races()` is not guarded.

4. **`_walk_node_for_events` dedup is O(n) list check** (`livescorebet.py:421`): `if eid not in acc` where `acc` is a list. For typical ~30 events this is negligible, but for very large meetings should use a set.

5. **Parquet merge is manual** (`livescorebet.py:637-649`): read→filter→concat→write without advisory lock. BoyleSports uses `parquet_store` (which has advisory locking). If two scrapes run concurrently (unlikely, but possible if the scraper is called from multiple threads), the merge could corrupt. Low risk.

**Integration status:** cache ✓ | rate_limiter ✗ (bypassed in prod) | proxy_manager ✓

---

## BoyleSports

**Verdict: BEST-STRUCTURED live scraper; one cache directory bug.**

**What works:**

- DevTools-confirmed selectors: `a.odds[data-selectionid][data-name][data-price]`, `data-isnr` for NR filter, `_EVENT_HREF_RE` for URL parsing. All match the confirmed 2026-06-12 DevTools note.
- `_to_decimal()` correctly handles fractional (`5/2`), decimal, `EVS`/`Evens`, `NR`, `SP`, `None`.
- `_race_time_iso` uses `TZ.localize()` (pytz-safe DST handling) on naive datetimes — correct.
- `_cross_source_ew()` is well-designed: reads `live_odds.parquet`, matches by `(venue_lower, epoch_minute)`, fills EW terms from other bookies, recomputes margin from BoyleSports win odds. Clean separation of concerns.
- `_merge_parquet` delegates to `parquet_store.write_parquet` — uses advisory locking from storage layer. Best practice.
- Stale cache fallback in `scrape()` is explicit and correct: if all live tiers fail and stale cache exists, returns it with a WARNING log.
- Region filter correctly normalises `_EXCLUDED_REGIONS` to lowercase and compares with `region.lower()`.

**Bugs:**

1. **Wrong cache directory** (`boylesports.py:31`): `_CACHE_PATH = ...data/boylesports_cache.json"`. Parent dir is `data/`, not `data/cache/`. So `CacheManager` constructs `Cache(cache_dir="data/")` and the cache file is `data/boylesports_cache.json`. All other scrapers use `data/cache/`. This is inconsistent — any tooling that scans `data/cache/` for scraper caches (e.g., future monitoring code) will miss BoyleSports. **Fix:** change to `data/cache/boylesports.json`.

2. **Playwright/Selenium tiers bypass rate limiter**: `_playwright_collect()` and `_selenium_collect()` drive a browser directly and sleep 2.5s between page requests, with no call to `get_rate_limiter()`. Acceptable since they're fallback paths, but worth documenting.

3. **`_parse_index` group class detection** (`boylesports.py:276`): `resulted = "race-resulted" in cls` where `cls = grp.get("class", "")`. HTML class attributes can be whitespace-separated tokens; checking `"in"` on the raw string is fragile if the class string is `"race-resulted-card"` or `"notrace-resulted"`. Should use `set(cls.split()) & {"race-resulted"}`. Low risk for now since BoyleSports class names are stable.

4. **`_collect_rows` silently drops empty-parse events** (`boylesports.py:455-458`): `_parse_event` returns `[]` if no `a.odds[data-selectionid]` found, and the result just isn't appended. If the DOM changes this gives zero rows with no warning beyond the `len(df)` log line. Low risk.

**Integration status:** cache partial (wrong dir) | rate_limiter ✓ (httpx tier) | proxy_manager ✓

---

## Selenium Fallback

**Verdict: CLEAN and reusable; two minor correctness gaps.**

**What works:**

- Zero site-specific logic — all caller-specific behaviour injected as callables.
- CDP pattern (`Network.responseReceived` → `getResponseBody`) is correct for Selenium 4 + ChromeDriver.
- DOM fallback only if XHR capture yields nothing — correct ordering.
- Per-body parse errors caught and logged; one bad body doesn't abort. ✓
- `proxy_rotator.next()` used to set `--proxy-server` if injected. ✓

**Bugs:**

1. **Predicate applied twice** (`_selenium_fallback.py:85` and `28`): `_capture_bodies()` already filters entries by `self._predicate(url)` (line 85). `_rows_from_bodies()` then checks `xhr_url_predicate(url)` again (line 28). The second check is redundant. Minor.

2. **Proxy manager `report_success/failure` never called** (`_selenium_fallback.py`): The proxy is set via `--proxy-server` Chrome arg but the rotator never learns whether the scrape succeeded or failed. The proxy manager's blacklist/health logic never fires for Selenium scrapes.

3. **DOM fallback spawns a second browser** (`_selenium_fallback.py:98-107`): `_dom_rows()` calls `_build_driver()` again — a second full Chrome launch after `_capture_bodies()` already quit one. If the XHR tier fails and DOM fallback is needed, two browsers are launched sequentially. Could be merged into one session by keeping the driver alive.

**Integration status:** cache N/A | rate_limiter ✗ | proxy_manager partial (no success/failure reporting)

---

## Timeform Historical

**Verdict: CORRECT orchestration; one writer bug on missing `position` column.**

**What works:**

- `TimeformClient.get_html()` wraps all requests in `with get_rate_limiter().acquire_for_url(url):` — correct integration.
- Proxy manager integrated in `TimeformClient.__init__` with `report_success/failure` in `_httpx_get()`. ✓
- `_looks_blocked()` detects Azure WAF and Cloudflare challenges — routes to browser/Selenium tiers. ✓
- `pages.meeting_goings()` uses a safe XPath `normalize-space` class check (not naive `contains`). ✓
- `pages.event_links()` uses `_EVENT_RE` to exclude meeting-summary links (fewer segments). ✓
- `_strip_name_noise()` correctly drops `(20H)`, `(39H) D BF` while preserving country codes like `(IRE)`. ✓
- Features deferred gracefully: `historical_win_rate`, `historical_place_rate`, `jockey_win_rate` set to `pd.NA` and `runs_in_window=0` when no `position` column is available. ✓
- `writer.py` delegates to `parquet_store.append_parquet` — uses storage layer advisory locking. ✓

**Bugs:**

1. **Writer may KeyError on missing `position`** (`timeform/writer.py:9-16`): `FINAL_COLUMNS` includes `"position"`. `fetch()` never adds a `position` column (results are not yet scraped). If `parquet_store.append_parquet(..., columns=FINAL_COLUMNS)` selects `df[FINAL_COLUMNS]` strictly, this raises `KeyError: 'position'` on every run. **Fix:** add `if "position" not in df.columns: df["position"] = pd.NA` in `fetch()` before calling `writer.write()`.

2. **`race_time` from `_parse_url` is timezone-naive** (`timeform_historical.py:48`): Returns `f"{day}T{hhmm[:2]}:{hhmm[2:]}"` — e.g. `"2026-06-13T14:30"` without timezone offset. When written to parquet and later parsed by the normalizer with `pd.to_datetime(utc=True)`, a naive string is ambiguous. Should be appended with `+01:00` (Irish summer) or left as-is and converted at write time. Low risk while results are absent.

3. **Double throttling in `TimeformClient.get_html()`** (`timeform/client.py:77-79`): Rate limiter holds the slot AND `time.sleep(self._delay)` is called inside. `request_delay=1.5s` (config default) is longer than `1/0.67 ≈ 1.49s` from the rate limiter, so the effective delay is dominated by the sleep. Harmless but wastes a little time. Could drop `_delay` now that rate limiter is wired.

4. **`StubExtractor` discards extractor context between calls** (timeform/extractor.py): The interface is stateless per call, which is correct. No issue.

5. **Azure WAF blocks Timeform results section** (documented, not a code bug): `recent_form` figures (free) are extractable; finishing positions are paywalled/WAF-blocked. Training matrix remains empty until an alternative results source is confirmed.

**Integration status:** cache N/A (batch) | rate_limiter ✓ | proxy_manager ✓

---

## Betfair SP Historical

**Verdict: BACKBONE logic correct; results enrichment is stub-only; no rate limiting anywhere.**

**What works:**

- `betfair_sp.file_url()` constructs correct promo.betfair.com SP CSV URLs (`dwbfpricesukwin{ddmmyyyy}.csv` etc.). ✓
- `parse_csv()` handles column normalisation (`c.strip().lower()`), BSP 1001 sentinel clipping, `_parse_dt` using `TZ.localize()` — all correct.
- `_venue_from_menu_hint()` strips trailing date words correctly. ✓
- `_fetch_backbone()` per-day/region/market loop with individual exception handling — one bad day doesn't abort. ✓
- `raw_store.write()` SHA-1 path dedup (`data/historical/raw/{source}/{year}/{date}/{hash}.html`) — correct. ✓
- `joiner.join()` priority-ranked source selection (racing_post first by default) — correct. Position conflict logged as `data_quality` warning. ✓
- `writer.py` dedupe key `["race_date","venue","horse_id","market_type"]` — correct (Betfair selection_id is stable). ✓

**Bugs:**

1. **No rate limiter on `fetch_csv()`** (`betsp/betfair_sp.py:82`): Hits `promo.betfair.com` with up to `3 * len(days) * len(regions) * len(markets)` requests in a tight loop (e.g. 3 years × 365 × 2 × 2 = 4380 requests) with no throttle. `fetch_csv()` does backoff on error only. The config has `promo.betfair.com: rps: 2.0` but it's never called. **Fix:** wrap the `httpx.Client.get()` call in `with get_rate_limiter().acquire_for_url(url):`.

2. **No rate limiter or proxy on `_results_get_html()`** (`betsp_historical.py:53-61`): Plain httpx with no throttle, no proxy, no retry beyond `raise_for_status()`. Racing Post and Sporting Life will rate-limit/block after a few rapid requests for full-year backfills. **Fix:** use `_results_get_html` only as a default; integrate rate limiter and proxy manager (or pass the same from a configured client).

3. **⚠️ Results source selectors are ILLUSTRATIVE** (all three sources): Sporting Life (`//section[contains(@class,"race")]` + `data-going`, `data-course`, `data-time`), Racing Post (`//*[contains(@class,"rp-result")]`), At The Races (`//*[contains(@class,"result")]`) — none confirmed via DevTools. Until confirmed, enrichment yields zero `ResultRow` objects and `position` stays `pd.NA` in the parquet (blocking the training matrix).

4. **`joiner.join()` is O(n×m)** (`betsp/joiner.py:39-50`): The match loop iterates all backbone rows with `out.at[idx, ...]` writes inside a Python for-loop. For 3-year datasets (~100k+ rows) this is slow. Should be a vectorized merge: build an enrichment DataFrame keyed on `(_norm_venue, _norm_horse, _minute_key)` and `DataFrame.merge(how="left")`.

5. **`_results_get_html` `raise_for_status()` propagates unhandled** (`betsp_historical.py:62`): If a results site returns 429 or 503, the exception propagates up through `_fetch_enrichment` and crashes the entire enrichment step (though `_fetch_backbone` result is preserved). Should wrap in try/except per-URL as BoyleSports does.

6. **`betfair_sp.fetch_csv()` on 404 returns `None` — correct**; `_fetch_backbone()` skips it via `if not text: continue` — correct. ✓

7. **Writer `FINAL_COLUMNS` includes `jockey_id`, `trainer_id`, `result_source`** which come from enrichment. When enrichment yields nothing, these are `pd.NA` (set in `joiner.join()`). `parquet_store.append_parquet` must handle NA for string columns — this depends on parquet_store's implementation but should be fine with pyarrow's nullable string type.

**Integration status:** cache N/A (batch) | rate_limiter ✗ | proxy_manager ✗

---

## Integration Matrix

| Scraper                  | `utils/cache.py`                                | `utils/rate_limiter.py`         | `utils/proxy_manager.py`              |
| ------------------------ | ----------------------------------------------- | ------------------------------- | ------------------------------------- |
| `paddy_power.py`         | ✓ `data/cache/paddy_power.json`                 | ✗ never acquired                | ✓ `report_success/failure` ✓          |
| `livescorebet.py`        | ✓ `data/cache/livescorebet.json`                | ✗ bypassed by Throttler in prod | ✓ `report_success/failure` ✓          |
| `boylesports.py`         | partial — wrong dir (`data/` not `data/cache/`) | ✓ httpx tier                    | ✓ `report_success/failure` ✓          |
| `_selenium_fallback.py`  | N/A                                             | ✗                               | partial — `next()` only, no reporting |
| `timeform_historical.py` | N/A (batch)                                     | ✓ via `TimeformClient`          | ✓ via `TimeformClient`                |
| `betsp_historical.py`    | N/A (batch)                                     | ✗ neither CSV nor HTML fetcher  | ✗                                     |

---

## Critical Fixes Needed

| #   | Severity   | File                                | Description                                                                                                                                                                                  |
| --- | ---------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **HIGH**   | `scraper/livescorebet.py:667-671`   | `Throttler` injected in production, bypassing central `rate_limiter`. Delete `throttler = Throttler()` and the `throttler=` kwargs to `LivescoreBetClient` and `LxmlFallback` in `scrape()`. |
| 2   | **HIGH**   | `scraper/betsp/betfair_sp.py:87`    | `fetch_csv()` hits promo.betfair.com with no rate limiting. Add `with get_rate_limiter().acquire_for_url(url):` around the GET.                                                              |
| 3   | **HIGH**   | `scraper/betsp_historical.py:53-62` | `_results_get_html()` — no rate limiter, no proxy, no retry. Needs `get_rate_limiter()` + `get_proxy_manager()` integration or it will be blocked on any full-year backfill.                 |
| 4   | **MEDIUM** | `scraper/boylesports.py:31`         | Cache path `data/boylesports_cache.json` — should be `data/cache/boylesports.json`.                                                                                                          |
| 5   | **MEDIUM** | `scraper/paddy_power.py` (client)   | No rate limiter acquisition. Wrap `httpx` GET in `with get_rate_limiter().acquire_for_url(url):`.                                                                                            |
| 6   | **MEDIUM** | `scraper/timeform/writer.py`        | `FINAL_COLUMNS` includes `position` but `fetch()` never adds it. Add `if "position" not in df.columns: df["position"] = pd.NA` before `writer.write()`.                                      |
| 7   | **LOW**    | `scraper/paddy_power.py:185`        | `_parse_race_time(raw)` called inside `_parse_pp_response` loop with no try/except. One malformed `marketTime` aborts the full parse. Wrap in try/except.                                    |
| 8   | **LOW**    | `scraper/_selenium_fallback.py`     | Proxy `report_success/failure` never called; DOM fallback spawns a second browser. Wire reporting; keep driver alive across XHR + DOM tiers.                                                 |
| 9   | **LOW**    | `scraper/betsp/joiner.py:39-50`     | Row-by-row `out.at[idx, ...]` loop — O(n). Replace with vectorized `DataFrame.merge`.                                                                                                        |

---

## Next

**Prompt 4:** Apply the five highest-severity fixes (items 1-5 above) in order — rate limiter wiring in livescorebet and betsp, BoyleSports cache path, Paddy Power rate limiter, timeform writer position column — then confirm via the existing test suite that nothing regresses.
