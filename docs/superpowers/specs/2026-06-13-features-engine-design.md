# features/engine.py — Predictive Feature Engine Design

**Date:** 2026-06-13
**Status:** Approved (design)
**Extends:** `features/` layer (`derive.py`, `fuse.py`, `builder.py`, `labels.py`)

## Goal

Add a set of higher-order predictive features on top of the existing leak-safe
feature pipeline: horse speed figures, jockey/trainer combo win%, each-way value
index, going preference, pace bias, odds delta vs. market, and a race complexity
score. Surface the full derived matrix to `data/features.parquet`.

The single-entity rolling trends (horse/jockey/trainer win/place rate) and class
differential the request also mentions **already exist** in `derive.py`
(`add_all_trailing_rates`, `add_class_change`); engine builds on them rather than
re-implementing.

## Architecture

A new module `features/engine.py` holds the new pure feature functions. It is
invoked by `builder.py`'s `_derive_all` **after** the existing `derive.*` steps,
so engine functions can build on already-derived columns (`implied_prob`,
`overround_norm_prob`, `market_rank`, `field_size`, `class_change`,
`going_speed`, `race_class`).

Design principles inherited from the existing layer:

- **Pure functions** — each `add_*(df, ...) -> df` takes and returns a copy; no I/O.
- **Train/serve parity** — engine steps live inside the shared `_derive_all`, so
  `build_training_matrix` and `build_inference_matrix` get identical columns. The
  existing parity test guards this and is extended to the new columns.
- **Leak-safe** — every trend feature uses runs **strictly before** each row's
  `race_date`, bounded by config `lookback_months` AND `lookback_runs` (tighter
  wins), reusing the established `add_trailing_rates` pattern via one shared
  generalized helper `_trailing_rate(df, key_cols, predicate=None)`.
- **Null-graceful** — features whose inputs are absent on current on-disk data
  (finishing `position`, `timeform_rating`, `pace_rating`, `morningwap`) compute
  to null and activate automatically when that data lands. No errors, no dropped
  rows.

## Config

Reuses existing `timeform.{lookback_months, lookback_runs, place_positions,
going_speed_map}`. No new config keys required.

## Functions

### `add_speed_figures(df, cfg) -> df`

Cols: `horse_speed`, `horse_speed_rank`.

- `horse_speed = timeform_rating` when present, **else** a derived proxy:
  leak-safe trailing mean of a per-run speed proxy. The per-run proxy is a
  self-contained field-relative finishing percentile,
  `(field_size - position) / (field_size - 1) * 100` (0 = last, 100 = won), so a
  trailing mean over strictly-prior runs is inherently leak-free. (A
  distance/going adjustment is a possible future refinement but is deliberately
  omitted to keep the per-run proxy self-contained and leakage-proof.)
- `horse_speed_rank` = within-race rank of `horse_speed` (method="min", desc).
- Null until finishing positions OR a TFR session land.

### `add_combo_win_rate(df, cfg) -> df`

Cols: `jt_combo_win_rate`, `jt_combo_runs`.

- Leak-safe trailing win% keyed on the `(jockey_id, trainer_id)` pair via
  `_trailing_rate`. `jt_combo_runs` = runs in window (sample-size guard).
- Null until finishing positions land.

### `add_going_preference(df, cfg) -> df`

Cols: `going_pref_win_rate`, `going_pref_place_rate`.

- Horse's leak-safe trailing win/place rate computed **only over prior runs whose
  `going_speed` band equals today's** (predicate filter on `_trailing_rate`).
- Distinct from track `going_speed` (which is the surface), this is the horse's
  affinity for that surface band.
- Null until finishing positions land.

### `add_pace_bias(df) -> df`

Col: `pace_bias`.

- Thin pass-through of `pace_rating` (paywalled-null today), coerced to numeric.
  No derived pace pressure (no run-style data exists).
- Null until a Timeform subscriber session supplies pace ratings.

### `add_ew_value_index(df) -> df`

Col: `ew_value_index`.

- Expected each-way return per 2-unit stake (1 win + 1 place) from `ew_places`,
  `ew_reduction`, win odds (`odds_decimal`/`sp`), and the market-implied place
  probability derived from `market_rank`/`field_size`.
- Definition: `ew_value_index = win_leg_ev + place_leg_ev - 2`, where the place
  leg pays `1 + (odds-1) * ew_reduction` on a top-`ew_places` finish. Positive =
  positive-expectation each-way bet at market odds.
- Works on live rows now (uses market odds + EW terms, not finishing position).

### `add_odds_delta(df) -> df`

Cols: `odds_drift`, `odds_value_delta`.

- `odds_drift = (morningwap - bsp) / morningwap` — morning-to-SP steamer/drifter
  signal from betSP history; null for live single-snapshot rows.
- `odds_value_delta = implied_prob - overround_norm_prob` — within-race relative
  mispricing; works on any snapshot including live.

### `add_race_complexity(df) -> df`

Col: `race_complexity` (per-race scalar broadcast to every runner).

- z-blend of available components, each skipped gracefully when its input is
  all-null in the race:
  - field size
  - implied-prob entropy / favourite gap (market competitiveness)
  - stdev(`race_class`) and recent-form spread
  - stdev(`timeform_rating`) (null until ratings available)
- Each component standardized across races, then averaged over the
  non-null components. Higher = harder/more competitive.

## Shared helper

`_trailing_rate(df, key_cols, win_col, place_col, runs_col, cfg, predicate=None)`
— generalizes `derive.add_trailing_rates` to arbitrary group keys and an optional
per-prior-run boolean predicate (used by going preference). `derive.add_trailing_rates`
remains the canonical single-entity implementation; engine's helper is for the
composite/predicate cases. Both share the strict-`<` cutoff + dual-bound logic.

## builder.py changes

`_derive_all` gains, after the existing `derive.*` calls:

```python
out = engine.add_speed_figures(out, cfg)
out = engine.add_combo_win_rate(out, cfg)
out = engine.add_going_preference(out, cfg)
out = engine.add_pace_bias(out)
out = engine.add_ew_value_index(out)
out = engine.add_odds_delta(out)
out = engine.add_race_complexity(out)
```

`build_training_matrix` additionally writes the **full derived matrix** (labelled
history + live runners, BEFORE the null-position filter) to `data/features.parquet`.
The existing `data/features/training.parquet` (labelled subset) is unchanged and
remains the model training file.

## Output files

- `data/features.parquet` — NEW: full derived matrix (every runner-race, labelled
  or not). Single file (not partitioned), matching the request.
- `data/features/training.parquet` — unchanged: labelled-only training subset.

## Testing

`tests/features/test_engine.py`:

- Per-function unit tests on small hand-built frames.
- **Leak-safety assertions**: a future-dated run never influences a past row's
  trailing feature (speed proxy, combo rate, going preference).
- **Null-graceful**: functions return the right null columns when `position`,
  `timeform_rating`, `pace_rating`, or `morningwap` are absent.
- `ew_value_index` numeric check against a hand-computed example.
- `race_complexity` broadcast test (same value for all runners in a race) and
  graceful component-skipping when a component input is all-null.

Extend the existing builder parity test so the new columns are asserted identical
between train and serve paths. Full suite must stay green (191 → ~191+N).

## Out of scope

- Sourcing the blocked inputs (finishing positions, TFR/pace ratings) — separate
  efforts already tracked in project Next Steps.
- Real pace-pressure modelling (needs run-style data we don't collect).
- Model training on these features — `/models` is the next layer.
