# BoyleSports Odds Scraper — Design

**Date:** 2026-06-12
**Status:** Approved (pending spec review)
**Author:** Race Predictor v3

## Purpose

Add `scraper/boylesports.py` to scrape live BoyleSports horse-racing odds, with
particular care for two areas that commonly break:

1. **Low-odds detection** — robust parsing of short/odds-on prices (fractional,
   `EVS`, odds-on) _and_ a boolean flag marking unusually short-priced runners.
2. **Each-way margins** — capture the raw each-way terms (1/4 or 1/5) _and_
   compute the place-market overround.

Output must match the standardized flat-row schema used by `livescorebet.py` so
BoyleSports rows feed the shared `data/live_odds.parquet` aggregation table.

This task also introduces a **shared Selenium fallback helper** and retrofits it
into all three scrapers (BoyleSports, Paddy Power, LivescoreBet).

## Non-Goals

- Confirming real BoyleSports API endpoints (done later via DevTools; placeholders
  are clearly flagged `⚠️ confirm via DevTools`).
- Building the GUI. We only ship the currency metadata + conversion utility it
  will consume.
- Live FX rates (static configured rate now; live fetch is a future swap).

## Architecture

`boylesports.py` mirrors `livescorebet.py`'s component structure:

- `Throttler(rate)` — token-bucket-style request spacing (copied pattern).
- `ProxyRotator(cfg)` — round-robin proxy pool from `config.yaml`, `None` when
  disabled/empty.
- `BoyleSportsClient(throttler, proxy_rotator)` — httpx + Chrome headers; raises
  `BotDetectedError` on 403 / HTML / non-JSON.
- Fallback tiers (see Fetch Flow).
- `CacheManager(path, ttl)` — identical to existing implementations; path is
  `data/boylesports_cache.json`.
- `_parse_*` helpers + `scrape(force=False) -> pd.DataFrame`.

### New shared module: `scraper/_selenium_fallback.py`

A reusable `SeleniumFallback` usable by every scraper:

```python
class SeleniumFallback:
    def __init__(self, url, xhr_url_predicate, parse_xhr, dom_parse=None,
                 wait_ms=8000, proxy_rotator=None): ...
    def fetch(self) -> list:   # returns list[row dict]; raises ScraperError if empty
```

- Launches headless Chrome via Selenium WebDriver.
- Enables CDP / performance logging to intercept network responses; for each
  response whose URL matches `xhr_url_predicate`, calls `parse_xhr(body)` and
  collects rows.
- If no XHR rows captured and `dom_parse` is provided, falls back to parsing the
  rendered DOM via `dom_parse(page_source)`.
- Proxy applied through Chrome options when the rotator yields one.

Each scraper supplies its own predicate + parser, so the helper holds zero
site-specific logic.

**Retrofit:** `paddy_power.py` and `livescorebet.py` each gain a Selenium tier
using this helper, slotted after their existing Playwright tier.

## Fetch Flow (BoyleSports)

1. **Cache check** — if `boylesports_cache.json` is fresh (age < `scrape_interval`)
   and `force=False`, return it rebuilt into a DataFrame.
2. **Tier 1 — JSON API** (`BoyleSportsClient`): fetch racing index + per-event
   detail. `BotDetectedError` → next tier.
3. **Tier 2 — Playwright XHR** interception.
4. **Tier 3 — Selenium** (`SeleniumFallback`): XHR capture, then DOM.
5. **Tier 4 — lxml DOM** parse of the static HTML page.
6. **Tier 5 — stale cache**: if all live tiers fail but a (stale) cache exists,
   log a warning and return it rather than raising. Raise `ScraperError` only when
   no data is obtainable at all.

## Output Schema

Flat rows → `pd.DataFrame`. Extends the shared `_PARQUET_COLUMNS` standard with
three new columns. The full standardized schema becomes:

| column         | type           | meaning                                                   |
| -------------- | -------------- | --------------------------------------------------------- |
| `fetched_at`   | datetime (UTC) | scrape timestamp                                          |
| `source`       | str            | `"boylesports"`                                           |
| `race_id`      | str            | event id                                                  |
| `race_time`    | datetime (UTC) | race start, parsed to Europe/Dublin then stored UTC       |
| `venue`        | str            | course name                                               |
| `market_type`  | str            | `WIN` / `PLACE` / `EACH_WAY`                              |
| `ew_places`    | Int64          | raw each-way places (e.g. 3)                              |
| `ew_reduction` | Float64        | raw each-way fraction as decimal (1/4 → 0.25, 1/5 → 0.20) |
| `ew_margin`    | Float64        | **new** — place-market overround (see below)              |
| `horse_name`   | str            | runner name                                               |
| `odds_decimal` | Float64        | win odds, decimal                                         |
| `sp`           | Float64        | starting price (NaN until available)                      |
| `is_low_odds`  | bool           | **new** — `odds_decimal < low_odds_threshold`             |
| `currency`     | str            | **new** — `"EUR"` or `"GBP"` (UK races)                   |

### Parquet write (read-merge-write)

`live_odds.parquet` is now updated, not clobbered:

1. Read existing parquet if present.
2. Drop rows where `source == "boylesports"`.
3. Concat the fresh BoyleSports rows.
4. Write back.

This is added to `boylesports.py`. During the Selenium retrofit, `livescorebet.py`
is switched to the same read-merge-write keyed on `source="livescorebet"` (it
currently overwrites the whole file) and gains the `ew_margin`, `is_low_odds`,
`currency` columns so the shared schema stays consistent. `paddy_power.py` keeps
its independent nested-dict schema and is **not** part of the parquet pipeline; it
only receives the Selenium tier.

## Low-Odds Robust Parsing

`_to_decimal(raw) -> float` handles BoyleSports price formats:

- decimal floats / numeric strings → as-is
- fractional strings `"5/2"`, `"1/5"`, `"2/7"` → `1 + num/den`
- `"EVS"` / `"Evens"` / `"evs"` → `2.0`
- malformed / missing → `float("nan")`

Result rounded to 6 dp to guard fractional rounding error. `is_low_odds` is set
when the parsed decimal is below `low_odds_threshold` (config, default `2.0` =
odds-on).

## Each-Way Margin

Per market with each-way terms, for each runner:

```
place_odds_i = 1 + (win_odds_i - 1) * ew_reduction
```

Place-market overround (the bookmaker's edge on the place part):

```
ew_margin = sum(1 / place_odds_i for valid runners) - ew_places
```

Computed once per market and repeated on every row of that market. `NaN` when
each-way terms or odds are missing.

## Currency Handling

Decimal odds are dimensionless and are **never** converted. Currency applies only
to monetary amounts the future GUI will display (stake, returns).

- Each row carries a `currency` field: `"GBP"` for UK races, else `"EUR"`.
- UK detection: meeting/event country code from the API when present; otherwise a
  known-UK-courses lookup. Default `"EUR"` for Irish meetings.
- New `utils/currency.py`:

```python
def to_eur(amount: float, currency: str) -> float:
    """Convert a monetary amount to EUR. EUR passes through; GBP uses gbp_eur_rate."""
```

Uses `gbp_eur_rate` from `config.yaml` (static). The GUI calls `to_eur()` on any
GBP monetary value at display time.

## Config Additions (`config.yaml`)

```yaml
low_odds_threshold: 2.0 # odds_decimal below this flags is_low_odds (odds-on)
gbp_eur_rate: 1.18 # static GBP→EUR rate for GUI monetary display

boylesports:
  horse_racing_url: https://www.boylesports.com/horse-racing
  api_base: <TBD — confirm via DevTools> # ⚠️ placeholder
```

## Error Handling

- `BotDetectedError` — 403 / HTML challenge / non-JSON; triggers next tier.
- `ScraperError` — raised only when every tier (including stale cache) fails.
- Per-event failures inside a tier are logged and skipped, not fatal (matches LSB).

## Testing

`tests/scraper/test_boylesports.py`, mirroring the existing `respx`-based suites:

- `_to_decimal`: decimal, fractional (`1/5`, `2/7`, `5/2`), `EVS`/`Evens`, garbage.
- `ew_margin`: known inputs → expected overround; missing terms → `NaN`.
- `is_low_odds`: boundary around `low_odds_threshold`.
- `currency`: UK course → `GBP`, Irish course → `EUR`.
- `CacheManager` freshness/read/write.
- Tier fallthrough: API `BotDetectedError` → Playwright → Selenium → lxml.
- Parquet read-merge-write preserves other sources' rows.

Plus `tests/utils/test_currency.py` for `to_eur`.

## Endpoint Confirmation (follow-up)

Real BoyleSports endpoints and JSON field names are placeholders. Confirm via
browser DevTools (Network tab on boylesports.com/horse-racing) and update
`api_base`, URL templates, and field names in `_parse_event`.
