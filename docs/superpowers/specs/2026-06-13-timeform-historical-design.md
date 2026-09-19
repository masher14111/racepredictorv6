# Timeform Historical Fetcher — Design

**Date:** 2026-06-13
**Module:** `scraper/timeform_historical.py` (+ `scraper/timeform/` package)
**Status:** Design approved, pending spec review

## Goal

Add Timeform as an extra historical data source for the Race Predictor, supplying
pre-race **Timeform ratings, pace figures, race class, and going** plus leak-safe
**point-in-time win/place rate** features. Output a standardized year-partitioned
parquet at `data/historical/timeform.parquet` that fuses with `betsp.parquet` on the
same `(venue, horse, minute)` key the betSP joiner already uses.

## Live access reality (confirmed via probe, 2026-06-13)

Probed through the DataImpulse residential proxy (Ireland targeting):

- ✅ **Racecards index** (`/horse-racing/racecards`) — httpx 200. Grid of meetings/races
  with course, going (`w-racecard-grid-info-going`), resulted flag
  (`w-racecard-grid-race-result`), and date-nav pagination buttons.
- ✅ **Individual racecard page** (`/horse-racing/racecards/{course}/{date}/{HHMM}/{courseId}/{raceNo}/{slug}`)
  — httpx 200, ~645 KB. Runner grid `rp-horse-row`, cells `rp-entry-number`,
  `rp-td-horse-name`, `rp-td-horse-jockey`, `rp-td-horse-form` (recent finishing
  figures, e.g. `50318-586`). Race header `rp-header` carries class (the `(N)` suffix
  in the race name), `Distance : 1m 3f 188y`, `Rated : (0-90)`, `Surface : Turf`.
- ⚠️ **Paywalled (empty without subscriber session):** `rp-td-horse-tfr` (TFR rating)
  and the Pace Map are blank in free mode — header shows "Unlock this Race". Free fields:
  name, **form figures**, jockey, class, distance, going. `timeform_rating`/`pace_rating`
  are session-gated (null until `timeform.session_cookie` supplied).
- 💡 **Free `rp-td-horse-form` figures encode recent finishing positions** — a free data
  source the feature layer can later use to derive win/place rates without the WAF'd
  results section.
- ❌ **Results section** (`/horse-racing/results...`) — Azure WAF JS challenge
  (`<meta content="Azure WAF JS Challenge">`, `/.azwaf/jsc/challenge.*.js`). httpx 403;
  headless Playwright returns the 12 KB challenge interstitial, NOT solved.

**Consequence:** finishing positions are WAF-blocked. Scraping them is handled by a
**separate future Timeform-results effort** (user-directed). This module scrapes only
the accessible pre-race data and exposes a plug-in slot for positions.

## Scope

**This build scrapes:** racecard index + per-race pages → per-runner `timeform_rating`,
`pace_rating`, `race_class`, `going`.

**This build computes now:**

- `going_speed` — config going→numeric map.
- `class_change` — this race's class vs the horse's previous race class, within the
  scraped set.

**Deferred (interface built + tested, values null until positions arrive):**

- `historical_win_rate`, `historical_place_rate`, `jockey_win_rate` — computed by
  `features.py` from a positions/runs table keyed `(norm_venue, norm_horse, date)`.
  The separate results scraper supplies positions later; the leak-safe rolling logic is
  built and unit-tested now against synthetic positions.

## Architecture (Approach A — mirrors `betsp_historical`)

```
scraper/timeform_historical.py   PUBLIC orchestrator: fetch(years, force, ...) -> pd.DataFrame
scraper/timeform/
  __init__.py
  client.py      anti-bot fetch tiers + ProxyRotator + optional auth/session
  pages.py       URL builders + pagination (index by date -> event pages)
  parser.py      HTML -> list[RunRow] (TFR, pace, class, going)
  extractor.py   PerfExtractor ABC + StubExtractor (delegates to parser; LLM deferred)
  features.py    runs table -> leak-safe trailing win/place rates + class_change + going_speed
  writer.py      year-partitioned parquet writer (read-merge-write dedupe)
scraper/betsp/raw_store.py     REUSED (raw HTML under raw/timeform/...)
scraper/_selenium_fallback.py  REUSED (Selenium tier)
tests/scraper/timeform/        unit tests per module
```

Each unit has one purpose, a defined interface, and is independently testable.

## Data flow

```
resolve years (config rolling_years) -> iter days
  -> client.get_html(index_url(day)) -> parser.parse_index -> event URLs (+ pages.paginate)
  -> for each event: client.get_html(event_url)
       -> raw_store.write(html, root=raw/timeform/...)
       -> extractor.extract(html) -> [RunRow, ...]
  -> concat -> RUNS TABLE (one row per runner per race)
  -> features.compute(runs) -> adds going_speed, class_change, (deferred) trailing rates
  -> writer.write(enriched, path=data/historical/timeform.parquet)
  -> return DataFrame
```

## Output schema — `data/historical/timeform.parquet` (Hive `year=YYYY`)

| column                        | source                     | notes                                                    |
| ----------------------------- | -------------------------- | -------------------------------------------------------- |
| `race_date`                   | scraped                    | Dublin-localized via pytz `TZ.localize`                  |
| `venue`                       | scraped                    |                                                          |
| `race_time`                   | scraped                    | ISO, for minute-key join                                 |
| `horse_name` / `horse_id`     | scraped                    | id null if not exposed                                   |
| `jockey_name` / `jockey_id`   | scraped                    |                                                          |
| `trainer_name` / `trainer_id` | scraped                    |                                                          |
| `position`                    | deferred (results scraper) | drives rate math; null until available                   |
| `timeform_rating`             | scraped                    | TFR (`rp-td-horse-tfr`)                                  |
| `pace_rating`                 | scraped                    | requested feature                                        |
| `race_class`                  | scraped                    | requested feature                                        |
| `going`                       | scraped                    | official going string                                    |
| `going_speed`                 | derived                    | requested feature — config going→numeric map             |
| `distance`                    | scraped                    |                                                          |
| `class_change`                | derived                    | this race class vs horse's previous race class (±levels) |
| `historical_win_rate`         | derived (deferred)         | point-in-time trailing (horse)                           |
| `historical_place_rate`       | derived (deferred)         | point-in-time trailing (horse)                           |
| `jockey_win_rate`             | derived (deferred)         | point-in-time trailing (jockey)                          |
| `runs_in_window`              | derived (deferred)         | sample size behind the rate                              |
| `region`                      | config                     | uk/ire                                                   |
| `source`                      | const                      | `"timeform"`                                             |
| `fetched_at`                  | runtime                    | UTC                                                      |
| `year`                        | partition                  | from `race_date`                                         |

**Join contract:** reuse the betSP joiner normalizers (`_norm_venue`, `_norm_horse`,
`_minute_key`) — extract to a shared helper so Timeform and betSP align by construction.
**Writer dedupe key:** `(race_date, venue, race_time, horse_name)` — uniquely identifies a runner in a race (no `market_type` split for racecards).

## Leak-safe features (`features.py`)

- Input: runs with `(horse_id|horse_name, race_date, position)`.
- Sort each entity's runs ascending by `race_date`.
- Trailing window = runs **strictly before** `race_date`, bounded by config
  `lookback_months` AND `lookback_runs` (tighter wins).
- `historical_win_rate = wins_in_window / runs_in_window`; place uses config places
  (default top-3). Emit `runs_in_window`.
- **Leak guard:** current row's own result never included (strict `<`). Explicit unit
  test with a fixture where same-day inclusion would change the number.

## Anti-bot / error handling

- Fetch ladder: httpx + Chrome headers via `ProxyRotator` (DataImpulse pool) → 403/WAF/
  non-HTML → Playwright DOM → Selenium (`_selenium_fallback.py`) → stale cache → raise
  `TimeformError`.
- Backoff 1→2→4s; rotate proxy per retry; config `request_delay` between pages.
- `raw_store.write` before parsing (no re-crawl on parser change; feeds future LLM extractor).
- Each page fetch independently try/logged — one bad race never aborts the run.
- Empty result → empty DataFrame with correct columns, never crash.
- Results section (WAF) is out of scope — client raises a clear logged error if pointed
  at it rather than returning the challenge page silently.

## Auth ("build for both")

Default runs on free racecard pages (confirmed working). To unlock any paywalled fields:
add a Timeform session under config (`timeform.session_cookie` or `username`/`password`),
which `client.py` injects into requests. **What's needed to unlock:** a logged-in
Timeform subscriber session cookie (or credentials) — documented in config comments.

## Config block (`config.yaml`)

```yaml
timeform:
  racecards_url: https://www.timeform.com/horse-racing/racecards
  rolling_years: 3
  regions: [uk, ire]
  request_delay: 1.5 # seconds between page fetches
  lookback_months: 12
  lookback_runs: 20
  place_positions: 3 # top-N counts as a place
  parquet_path: data/historical/timeform.parquet
  raw_path: data/historical/raw
  session_cookie: "" # optional — paste subscriber session to unlock paywalled fields
  going_speed_map: # going string -> numeric speed index
    firm: 5
    good-to-firm: 4
    good: 3
    good-to-soft: 2
    soft: 1
    heavy: 0
```

## Testing

- `parser` — against saved `_probe_out` HTML fixtures (real DOM): index, runner rows
  (TFR/pace/class/going), pagination.
- `features` — leak-safety (same-day exclusion), window bounds (months vs runs),
  cold-start (null + `runs_in_window=0`).
- `writer` — year partitioning, dedupe.
- `client` — tier fallthrough (respx-mocked), proxy rotation, WAF→raise.
- `pages` — URL builders, date formatting (pytz gotcha).
- Orchestrator — end-to-end, network mocked.

## Notes

- `probe_timeform.py` + `_probe_out/` are throwaway — delete once parser selectors are
  locked from the fixtures.
- Selectors confirmed from live DOM (racecard `rp-td-horse-tfr`, index `w-racecard-grid-*`),
  not assumed — but full runner-row field mapping to be finalized against fixtures during build.

```

```
