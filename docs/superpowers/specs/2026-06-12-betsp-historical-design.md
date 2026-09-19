# betSP Historical Data Fetcher — Design Spec

**Date:** 2026-06-12
**Status:** Approved (design), pending implementation plan
**Module (public entrypoint):** `scraper/betsp_historical.py`

## 1. Purpose

Build a historical data fetcher that produces a per-horse-run dataset for the
Race Predictor v3 model: race results, Betfair Starting Prices, market-confidence
signals, and (via enrichment) jockey strike rates, trainer stats, and horse form
lines. Output is a year-partitioned parquet at `data/historical/betsp.parquet`.

The dataset feeds downstream feature engineering (`/features`) and the LightGBM
model (`/models`).

## 2. Recon findings (ground truth, 2026-06-12)

"betSP" = the Betfair Starting Price daily price dumps at
`https://promo.betfair.com/betfairsp/prices/`.

Confirmed by fetching a live file (`dwbfpricesukwin01062026.csv`). The **only**
columns in the feed are:

```
event_id, menu_hint, event_name, event_dt, selection_id, selection_name,
win_lose, bsp, ppwap, morningwap, ppmax, ppmin, ipmax, ipmin,
morningtradedvol, pptradedvol, iptradedvol
```

Sample row:
`258715928, Nottingham 31st May, 1m2f Hcap, 31-05-2026 16:55, 96505154, Sharp Romance, 1, 2.51387501, ...`

Mapping the requested 8 columns against reality:

| Requested column | Betfair SP source                      | Status                          |
| ---------------- | -------------------------------------- | ------------------------------- |
| `race_date`      | `event_dt` (`DD-MM-YYYY HH:MM`)        | direct                          |
| `horse_id`       | `selection_id` (+ `selection_name`)    | direct                          |
| `odds_finish`    | `bsp`                                  | direct                          |
| `distance`       | parsed from `event_name` (`1m2f Hcap`) | derived                         |
| `position`       | only `win_lose` (1/0 binary)           | partial — no finishing position |
| `jockey_id`      | —                                      | **absent in feed**              |
| `trainer_id`     | —                                      | **absent in feed**              |
| `going`          | —                                      | **absent in feed**              |

**Conclusion:** Betfair SP gives a strong SP + result + distance + market-confidence
backbone, but jockey, trainer, true finishing position, and going do **not** exist
in the feed. Those require a results source. Decision: enrich inline by scraping
results sites and joining on (race_date, course, horse).

## 3. Architecture

The request names a single file `scraper/betsp_historical.py`. That file is kept
as the **public orchestrator**; internals are split into a focused package so each
unit has one job and is independently testable.

```
scraper/betsp_historical.py        ← PUBLIC orchestrator: fetch(years, force) -> pd.DataFrame
scraper/betsp/
  betfair_sp.py     — downloads daily SP CSVs (UK+IRE, win+place), parses → backbone rows
  results/
    base.py         — ResultsSource ABC: fetch_raw(date) -> list[RawResult]; parse(raw) -> rows
    sporting_life.py
    racing_post.py
    at_the_races.py
  raw_store.py      — persists raw HTML/JSON under data/historical/raw/{source}/{year}/{date}/
  extractor.py      — PerfExtractor ABC + StubExtractor (deterministic, no LLM); LLM impl later
  joiner.py         — joins Betfair backbone ⨝ extracted results on (race_date, course, horse)
  writer.py         — writes data/historical/betsp.parquet partitioned by year
```

**Data flow:** orchestrator picks the rolling year window → `betfair_sp` builds the
SP/result/distance backbone → each `ResultsSource` fetches + stores raw, then the
`StubExtractor` parses raw into jockey/trainer/position/going records → `joiner`
matches them to the backbone by (date, course, horse) → `writer` outputs
year-partitioned parquet.

**Reuses existing infrastructure:** `scraper/_selenium_fallback.py`
(httpx → Playwright → Selenium tiering), `utils/timezone.py`, `utils/logger.py`,
and the read-merge-write parquet pattern already used by `boylesports.py`.

## 4. Output schema

`data/historical/betsp.parquet/year=YYYY/…` — Hive-partitioned via pandas/pyarrow
`partition_cols=["year"]`.

| column          | type                | source                           | notes                                              |
| --------------- | ------------------- | -------------------------------- | -------------------------------------------------- |
| `race_date`     | datetime64 (Dublin) | Betfair `event_dt`               | full timestamp; `year` partition derived from it   |
| `venue`         | str                 | Betfair `menu_hint`              | course; also a join key                            |
| `horse_id`      | int64               | Betfair `selection_id`           | stable Betfair id                                  |
| `horse_name`    | str                 | Betfair `selection_name`         | normalized for joining                             |
| `jockey_id`     | str (nullable)      | results scrape                   | site id if present, else slug of name              |
| `trainer_id`    | str (nullable)      | results scrape                   | same                                               |
| `odds_finish`   | float64             | Betfair `bsp`                    | `1001`/blank → null (no SP backers)                |
| `position`      | int (nullable)      | results scrape                   | true finishing position                            |
| `win_lose`      | int8                | Betfair `win_lose`               | 1/0 — always present, unlike `position`            |
| `going`         | str (nullable)      | results scrape                   |                                                    |
| `distance`      | str                 | parsed from Betfair `event_name` | e.g. `1m2f`; null if unparseable                   |
| `market_type`   | str                 | filename                         | `WIN`/`PLACE`                                      |
| `region`        | str                 | filename                         | `UK`/`IRE`                                         |
| `morningwap`    | float64             | Betfair                          | SP-vs-result edge feature                          |
| `ppwap`         | float64             | Betfair                          | SP-vs-result edge feature                          |
| `result_source` | str (nullable)      | joiner                           | which site filled enrichment, or null if unmatched |
| `fetched_at`    | datetime64          | runtime                          | provenance                                         |

**Nullable-by-design:** `jockey_id`, `trainer_id`, `position`, `going`,
`result_source` start null on any row where no results-site match is found. Betfair
rows are never dropped for lack of enrichment — the parquet is complete from day one.

**Idempotency / incremental:** writer dedupes on
`(race_date, venue, horse_id, market_type)`; re-running a year overwrites that
year's partition (read-merge-write per partition), so re-scrapes and late
enrichment update cleanly. The orchestrator skips dates already present unless
`force=True`.

## 5. Betfair fetcher (`betfair_sp.py`)

- File URL pattern (confirmed):
  `https://promo.betfair.com/betfairsp/prices/dwbfprices{REGION}{MARKET}{DDMMYYYY}.csv`
  where REGION ∈ `{uk, ire}`, MARKET ∈ `{win, place}`. Four files per day.
- Iterate every date in the requested year window; httpx GET each of the 4 files.
- **404 = no racing / not published** → skip silently (expected). Other errors →
  existing backoff retry, then log + skip the date so one bad day can't abort a year.
- Parse with pandas; map columns per the schema.
- `bsp` of `1001` or blank → null.
- Parse `distance` from `event_name` via regex (`\d+m(\d+f)?|\d+f`); unparseable → null.
- `event_dt` (`DD-MM-YYYY HH:MM`) → Dublin-aware datetime via `utils/timezone`
  using `TZ.localize()` (pytz — do NOT pass `tzinfo=TZ`; see memory note).

## 6. Results sources & extractor

**`ResultsSource` ABC (`results/base.py`):**
`fetch_raw(date) -> list[RawResult]` and `parse(raw) -> list[ResultRow]`.
Three implementations: `sporting_life.py`, `racing_post.py`, `at_the_races.py`.
Each uses the tiered fetch (httpx → Playwright → Selenium via `_selenium_fallback`).

**Raw store (`raw_store.py`):** persists captured raw HTML/JSON under
`data/historical/raw/{source}/{year}/{date}/` so the future LLM extractor can
re-run over already-fetched history without re-scraping.

**Extractor (`extractor.py`)** — the seam where the LLM lands later:

```python
class PerfExtractor(ABC):
    @abstractmethod
    def extract(self, raw: RawResult) -> list[ResultRow]: ...
    # ResultRow: race_date, venue, horse_name, jockey_id, trainer_id,
    #            position, going, source
```

- `StubExtractor` — deterministic, no LLM. Calls each source's `parse()` and
  validates/coerces into `ResultRow`s. The pipeline produces real enrichment today
  with no model dependency.
- `LLMExtractor` (future, not built now) — same interface, consumes the raw blobs
  from `raw_store` and uses Gemini/OpenAI/local to reconcile messy/partial HTML
  across the three sources. Selected via `betsp_historical.extractor` config key
  (defaults to `stub`).

## 7. Join logic (`joiner.py`)

- Key: `(race_date_to_minute, normalized_venue, normalized_horse_name)`.
  `menu_hint` ("Nottingham 31st May") and results-site course names normalized via
  `_normalize_course()`; horse names lowercased, `(IRE)`/`(GB)` country suffixes
  and punctuation stripped.
- Left join onto the Betfair backbone (backbone is the spine of truth for SP/result).
- Multiple sources agreeing → prefer by configured priority
  (`racing_post > sporting_life > at_the_races`). Disagreement on `position` logged
  as a `data_quality` warning; winner-by-priority taken.
- Unmatched Betfair rows keep null enrichment + null `result_source`.
- Unmatched results rows (horse in results but not Betfair) dropped with a debug log.

## 8. Config additions (`config.yaml`)

```yaml
betsp_historical:
  rolling_years: 3 # window depth for incremental backfill
  regions: [uk, ire]
  markets: [win, place]
  results_sources: [sporting_life, racing_post, at_the_races]
  source_priority: [racing_post, sporting_life, at_the_races]
  extractor: stub # stub | llm (llm not built in this spec)
  max_days_per_run: 0 # 0 = unlimited; >0 caps a run for testing
  parquet_path: data/historical/betsp.parquet
  raw_path: data/historical/raw
```

## 9. Error handling (degrade, never abort a backfill)

- Betfair file 404 → skip date (normal). Network error → backoff retry → log + skip.
- A `ResultsSource` failing for a date (Cloudflare, layout change) → log, store
  whatever raw was captured, continue with other sources; that source's rows stay
  unmatched (null enrichment).
- Per-tier fallback httpx → Playwright → Selenium via `_selenium_fallback`; if all
  tiers fail for a source, skip that source for the run, not fatal.
- Parse failure on a single race → log to `data_quality`, skip that race, keep the rest.
- All logging via `utils/logger`.

## 10. Testing (TDD, matching `tests/scraper/` layout)

- `tests/scraper/betsp/test_betfair_sp.py` — fixture CSVs (real header from recon):
  column mapping, `bsp=1001`→null, distance regex (`1m`, `1m2f`, `5f`,
  unparseable→null), Dublin tz correctness, 404-skip.
- `test_results_sporting_life.py` / `test_results_racing_post.py` /
  `test_results_at_the_races.py` — fixture HTML → `ResultRow`s.
- `test_extractor.py` — StubExtractor coercion/validation; interface contract.
- `test_joiner.py` — normalization, match/no-match, source-priority conflict
  resolution, Betfair-spine retention of unmatched rows.
- `test_writer.py` — year partitioning, dedupe key, read-merge-write idempotency.
- `test_betsp_historical.py` — orchestrator end-to-end with all upstreams mocked
  (fixture CSVs + fixture HTML) → asserts final parquet schema and joined rows.

## 11. Public API

```python
# scraper/betsp_historical.py
def fetch(years: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """Build/refresh the betSP historical dataset.

    years: explicit years to (re)fetch; default = rolling window from config.
    force: re-fetch dates already present in the parquet.
    Returns the full dataset as a DataFrame; also writes year-partitioned parquet.
    """
```

## 12. Out of scope (future specs)

- `LLMExtractor` implementation (Gemini/OpenAI/local) — interface seam built now.
- Feature engineering / rolling strike-rate aggregation (`/features`).
- Model integration (`/models`).
