# Unified Race Normalizer — Design

**Date:** 2026-06-13
**Module:** `utils/normalizer.py`
**Output:** `data/unified_races.parquet` (year-partitioned, Hive `year=YYYY`)

## Purpose

Merge every scraped and historical source into one canonical race-runner table so
the `/features` and `/models` layers read a single, validated schema instead of
four source-specific ones. Standardize horse/jockey/trainer IDs, coerce odds to
decimals, align timestamps to Europe/Dublin, handle missing values, validate with
pydantic, drop duplicates, and write a year-partitioned parquet dataset.

## Sources (and how they reach us)

| Logical source                                                             | Input                              | Notes                                                                                             |
| -------------------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| Live odds (BoyleSports, LivescoreBet)                                      | `data/live_odds.parquet`           | Already merged by `source` column                                                                 |
| Paddy Power                                                                | `data/cache/paddy_power.json`      | Nested dict cache (races → markets → selections); flattened by the adapter                        |
| betSP (Betfair SP + Racing Post / Sporting Life / At The Races enrichment) | `data/historical/betsp.parquet`    | racingpost/racingform arrive folded in via the `result_source` column — **not** separate adapters |
| Timeform                                                                   | `data/historical/timeform.parquet` | Year-partitioned                                                                                  |

Missing inputs are skipped gracefully (a source file that does not exist contributes
no rows; `normalize()` does not fail).

## Canonical Schema

Grain: **one runner, one race, one source** —
key `(race_date, venue, horse_id, market_type, source)`.

The schema is the **union** of all source columns. A source leaves any column it
cannot supply as null (live odds have no `position`; historical has no live
`odds_decimal`).

- **Keys / identity:** `race_date, race_time, venue, region, horse_name, horse_id,
jockey_name, jockey_id, trainer_name, trainer_id`
- **Source-native IDs (traceability):** `src_horse_id, src_jockey_id, src_trainer_id`
- **Market / odds:** `market_type, odds_decimal, sp, ew_places, ew_reduction,
ew_margin, is_low_odds, currency, morningwap, ppwap, odds_finish`
- **Outcome:** `position, win_lose`
- **Ratings / form:** `timeform_rating, pace_rating, race_class, going, going_speed,
distance, class_change, recent_form, historical_win_rate, historical_place_rate,
jockey_win_rate, runs_in_window`
- **Provenance:** `source, fetched_at, year`

## Identity Standardization

Canonical IDs are **deterministic name hashes** — `horse_id = _canonical_id(norm_horse(name))`,
jockey/trainer likewise on their normalized names. Same normalized name → same id
across every source, with no persisted state. Source-native ids are preserved
separately in `src_horse_id` / `src_jockey_id` / `src_trainer_id` for traceability.

`_canonical_id` uses a stable hash (e.g. blake2b of the normalized string, hex-truncated)
— **not** Python's salted `hash()`, so ids are reproducible across processes and runs.
Empty/missing names hash to null, not to a shared sentinel id.

## Components (all in `utils/normalizer.py`)

### Shared transforms

- `_canonical_id(name) -> str | None` — stable hash of the normalized name; null on empty.
- `_to_decimal(x) -> float | None` — coerce odds to decimal; guards stray fractional /
  `EVS` / `NR` values (live + paddy are already decimal).
- `_to_dublin(ts) -> str` — align `race_time` / `fetched_at` to Europe/Dublin via
  `utils/timezone`; `race_date` derived from the Dublin race time.
- `_reindex_canonical(df) -> df` — add any missing canonical columns as NA, order columns.

### Per-source adapters (each returns canonical-column DataFrame)

- `_from_live_odds(df)`
- `_from_paddy_power(cache_dict)` — flattens nested races → markets → selections.
- `_from_betsp(df)`
- `_from_timeform(df)`

Each adapter maps native column names to canonical, applies `_canonical_id` to names,
coerces odds via `_to_decimal`, aligns timestamps via `_to_dublin`, then
`_reindex_canonical`.

### Validation

pydantic v2 `UnifiedRaceRow` model — typed and optional fields plus validators:
`odds_decimal > 1.0` or null, `position >= 1` or null, `market_type` in the allowed
enum (`WIN`, `EACH_WAY`, `PLACE`). Validation runs via `TypeAdapter(list[UnifiedRaceRow])`.
Invalid rows are **dropped and logged** (count + reason), never fatal.

### Public API

```python
normalize(sources=None, write=True) -> pd.DataFrame
```

Reads each available source (skipping missing files) → runs its adapter → concats →
validates → dedupes (`drop_duplicates` on the grain key, `keep="last"` so the latest
`fetched_at` wins) → writes. `sources` optionally restricts which inputs to read
(useful for tests); `write=False` returns the frame without persisting.

## Output

`data/unified_races.parquet`, **year-partitioned (Hive `year=YYYY`) via read-merge-write**,
matching the `betsp` / `timeform` convention. `year` is derived from `race_date`.
The existing pyarrow gotcha applies: `shutil.rmtree` the dataset dir before
`to_parquet(partition_cols=["year"])` so stale partition files are overwritten, not appended.

## Testing — `tests/utils/test_normalizer.py`

- `_canonical_id` determinism and cross-process stability; null on empty name.
- Each adapter maps a tiny native fixture to the correct canonical columns.
- `_to_decimal` coercion (decimal passthrough, fractional/`EVS`/`NR` guards).
- `_to_dublin` timestamp alignment and `race_date` derivation.
- pydantic rejects bad rows (`odds_decimal <= 1`, `position < 1`, bad `market_type`).
- Dedupe keeps the latest `fetched_at` on the grain key.
- End-to-end `normalize()` over small fixtures of all four inputs → one validated frame.

## Dependencies

Add `pydantic` (v2) to `requirements.txt`.

## Out of Scope

- Feature engineering beyond what sources already provide (lives in `/features`).
- Building separate Racing Post / Sporting Life / At The Races adapters (they arrive
  via betSP enrichment).
- Cross-source row _fusion_ (merging a Timeform row and a betSP row for the same runner
  into one row) — the unified table keeps one row per source; downstream `/features`
  decides how to combine.
