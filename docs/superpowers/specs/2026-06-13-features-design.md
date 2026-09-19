# /features — Model-Ready Matrices — Design

**Date:** 2026-06-13
**Package:** `features/`
**Inputs:** `data/unified_races.parquet` (from `utils/normalizer.py`)
**Output:** `data/features/training.parquet` (training); in-memory frame (inference)

## Purpose

Turn the unified race dataset (one row per runner **per source**) into model-ready
matrices for the LightGBM predictor. Two builders share the **same** feature code to
guarantee train/serve parity:

- `build_training_matrix()` — historical, **labelled** (`won`, `placed`).
- `build_inference_matrix()` — today's live races, **unlabelled**, identical feature columns.

## Grain & Fusion

Prediction unit = **one runner in one race**, keyed `(race_date, venue, horse_id)`.

`fuse_sources(df)` collapses the per-source unified rows to this grain: group by the
key, and for each column take the **first non-null** across the group, breaking ties by
source priority **timeform > betsp > live** (boylesports/livescorebet/paddy_power).
The fused row carries the union of all sources' signals — Timeform ratings/going/class,
betSP SP/position/odds_finish, and live bookie odds.

## Components (a `features/` package)

### `features/fuse.py`

- `fuse_sources(df) -> pd.DataFrame` — coalesce to runner-race grain (above).

### `features/derive.py` — pure functions, each adds columns to a runner-race frame

Relocated from `scraper/timeform/features.py` (its canonical home moves here;
`timeform/features.py` re-imports them so its existing pipeline and 24 tests are unchanged):

- `add_going_speed(df, going_map)`
- `add_class_change(df)`
- `add_trailing_rates(df, entity, win_col, place_col, lookback_months, lookback_runs, place_positions)`
  — leak-safe: counts only runs strictly **before** each row's `race_date`, bounded by
  `lookback_months` AND `lookback_runs`.

New derive functions:

- `add_odds_features(df)` — `implied_prob = 1/odds_decimal`; `overround_norm_prob`
  (implied_prob normalized to sum 1 within each race); `log_odds = ln(odds_decimal)`;
  `market_rank` (ascending odds rank within race); `field_size` (runners per race).
  Uses `odds_decimal`, falling back to `sp` then `odds_finish` when `odds_decimal` is null.
- `add_distance_furlongs(df)` — parse `distance` (`1m2f188y`) to numeric furlongs.
- `add_recent_form(df)` — parse the `recent_form` figure string (e.g. `50318-586`):
  `recent_form_runs` (count of figure chars), `recent_form_wins` (count of `1`s),
  `recent_form_avg` (mean of numeric figures; non-numeric like `-`/`P`/`F` ignored).
- `add_rating_rank(df)` — descending `timeform_rating` rank within each race.

The trailing-rate convenience wrapper `add_all_trailing_rates(df, cfg)` applies
`add_trailing_rates` for horse (`historical_win_rate`/`historical_place_rate`),
jockey (`jockey_win_rate`), and trainer (`trainer_win_rate`) entities.

### `features/labels.py`

- `add_labels(df, place_positions) -> df` — `won = (position == 1)`,
  `placed = (position <= place_positions)`. Rows with null `position` get null labels.

### `features/builder.py` — public API

- `build_training_matrix(unified=None, write=True, output_path=None) -> pd.DataFrame`
  Load unified (default `data/unified_races.parquet`) → `fuse_sources` → derive (all
  functions) → `add_labels` → **drop rows with null `position`** → optionally write
  `data/features/training.parquet`. Not partitioned (single file; the model reads it whole).
- `build_inference_matrix(unified=None) -> pd.DataFrame`
  Load unified → `fuse_sources` → derive trailing rates over **live + full history
  together** (so today's runners get their prior rates under the strict `<` cutoff) →
  return only the **live** runner-races (null `position`), **no labels**. Feature columns
  identical to training.

## Data Flow & Parity

Both builders run the same `fuse → derive` pipeline; they differ only in (a) labelling
and (b) the final row filter (training keeps known-position rows; inference keeps
null-position live rows). Leak-safety comes from the strict `<` `race_date` cutoff in
`add_trailing_rates`. A parity test asserts the two matrices share an identical
feature-column set.

## Config

Reads existing `config.yaml` keys under `timeform`: `lookback_months`, `lookback_runs`,
`place_positions`, `going_speed_map`. No new config. Model hyperparameters belong to the
later `/models` step, not here.

## Error Handling

- Missing `data/unified_races.parquet` → raise a clear error (the pipeline requires it).
- Null/zero/invalid `odds_decimal` → odds features fall back to `sp`/`odds_finish`, else
  null (never divide-by-zero); `implied_prob` null when no usable odds.
- Empty input → builders return an empty frame with the full feature-column set.

## Testing — `tests/features/`

- `fuse_sources`: coalesce takes first non-null; source-priority tie-break; one row per
  runner-race.
- `add_odds_features`: implied prob, overround normalization sums to 1 per race,
  market_rank ordering, sp/odds_finish fallback, null-odds safety.
- `add_distance_furlongs`, `add_recent_form`: parser correctness incl. malformed input.
- `add_rating_rank`, `add_going_speed`, `add_class_change`: ranking/mapping correctness.
- `add_all_trailing_rates`: **leak-safety** — a horse's rate excludes its same-day and
  future runs; respects lookback bounds.
- `add_labels`: won/placed correctness; null position → null label.
- `build_training_matrix`: labelled, no null-position rows, writes parquet.
- `build_inference_matrix`: live rows only, no labels, trailing rates reflect history.
- **Parity:** training and inference feature-column sets are identical.
- Regression: `scraper/timeform/features.py` re-export keeps Timeform's existing tests green.

## Out of Scope

- LightGBM/baseline model training and weighting (`/models`).
- Feature selection / importance analysis.
- Backfilling historical odds for live-only bookies.
