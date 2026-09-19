# Implementation Plan

[Overview]
Overhaul the UK/Ireland horse racing predictor to fix known data-leakage, calibration, and feature-completeness bugs that make the system unprofitable, then add several new predictive signals to improve discrimination power and close the persistent negative Closing-Line Value gap.

The codebase is well-structured with CatBoost models predicting won/placed_2/showed targets, a walk-forward backtester, an isotonic/sigmoid calibration layer, a value-betting layer (with de-vigging and EV gates), and a recent favourite-longshot odds-band recalibrator. The project's own audit documents (memory/model-09-baseline-audit.md, memory/model-14-backtest.md, memory/calib-fl-04-backtest.md) identify several P0/P1 issues that collectively explain why the system is unprofitable:

1. **C1: The displayed v3 model serves uncooked (uncalibrated) probabilities** — ECE_won=0.42, A/E≈0.2–0.3, ~3–4× inflated. The UI shows grossly overconfident numbers. The v3 model on disk has no calibrator `.pkl` files (only v3nf has them).

2. **C2: Outcome-adjacent leakage via odds_finish** — 99.9% of historical rows build EVERY market feature (implied_prob, overround_norm_prob, log_odds, market_rank, ew_value_index, odds_value_delta) from `odds_finish` (the finishing/returned Betfair SP), which correlates with the outcome. v3's AUC 0.789 is optimistic and will not hold for live predictions where only pre-off `odds_decimal` is available.

3. **C3: Per-race key partially fixed** — `add_race_key()` now produces `race_uid` (venue + off-time minute) using recovered `race_time` from the normalizer. However, the backtest panel's `race_uid` constructor in `backtest/data.py` checks `race_time.notna().any()` before using it, and if the normalizer recovery failed for historical rows, it still degrades to venue-day. Field-size ranks and within-race normalization then operate on the wrong grouping.

4. **Missing features in the backtest panel** — 9 of 30 price-free feature columns (course_win_rate, course_place_rate, course_runs, distance_win_rate, distance_place_rate, distance_runs, days_since_last_run, horse_career_runs, speed_trend) are absent from the training parquet because the matrix was never rebuilt after model-10 added them.

5. **CLV stays negative even after the favourite-longshot recalibration** — best CLV is −4% on the tightest favourite band [2.0, 4.0]. The model systematically bets at worse prices than the BSP close. The recalibration flattened A/E (2.80→0.94 on odds-on) but did not fix CLV, suggesting the underlying discrimination is insufficient or the price-capture timing is wrong.

6. **No new features added since model-10** — the feature set has been static. The model has modest sharpness (Brier 0.100 vs base-rate Brier 0.104 means most of the prediction is just the base rate). More discriminating features are needed to beat the market.

The implementation will fix the P0 bugs first (data integrity), complete the missing features (regenerate the matrix), then add several new predictive signals that have strong theoretical backing in racing literature and betting practice: draw/bias features, weight-adjusted speed ratings, trainer form cycle, market-movement signals, and class-par figures.

[Types]
New data structures and modified type definitions for the improved feature pipeline.

- **`DrawBiasConfig`** — configuration for draw/bias features:
  - `band_edges: list[int]` — draw-number cut points (default [1, 5, 10, 15, float("inf")])
  - `min_runners_per_band: int` — minimum historical runners needed in a band to trust the bias (default 50)
  - `lookback_months: int` — how far back to look for draw records (default 24)

- **`WeightConfig`** — configuration for weight-adjusted speed:
  - `lbs_per_length: float` — pounds per length conversion factor (default 3.0 for Flat, 2.0 for NH)
  - `use_official_rating: bool` — whether to use official BHA ratings when available (default True, falls back to horse_speed proxy)

- **`TrainerFormCycle`** — enum/config for trainer form detection:
  - `short_window_days: int` — recent window for "hot" detection (default 14)
  - `long_window_days: int` — baseline window for comparison (default 90)
  - `min_runners_in_window: int` — minimum runners to compute rate (default 5)

- **`MarketMovementFeatures`** — additional columns added to the fused DataFrame during feature derivation:
  - `price_steam_pct: float` — percentage the early price has shortened (morning_wap → current odds), positive = backed in
  - `price_drift_pct: float` — same but for drift (negative = drifting out)
  - `relative_market_share: float` — runner's implied probability share of the total book (replaces/improves overround_norm_prob which was computed per venue-day)
  - `market_book_pct: float` — the book overround as a percentage (field sum of 1/d − 1)

- **`ClassParFigure`** — class-par performance measure:
  - `class_par_speed: float` — best recent speed figure adjusted for class of race (higher class = faster expected time)
  - `class_par_rank: int` — within-race rank on class-par speed

[Files]
File-level changes required, with full paths and specific modifications.

**New files to be created:**

1. `features/_draw_bias.py` — compute per-course per-draw-band win/place rates from historical data; add `draw_band`, `course_draw_win_rate`, `course_draw_place_rate` columns.

2. `features/_weight_speed.py` — compute weight-adjusted finishing speed figures; add `weight_adjusted_speed`, `weight_speed_rank` columns using the cardinal `horse_speed` proxy + weight carried.

3. `features/_trainer_form.py` — compute short-term vs long-term trainer strike rates to detect "in-form" yards; add `trainer_hot_strike_rate`, `trainer_form_zscore` columns.

4. `features/_market_movement.py` — derive price steaming/drifting signals from `morningwap` and current `odds_decimal` (pre-off prices only, NOT `odds_finish`); add `price_steam_pct`, `relative_market_share`, `market_book_pct` columns.

5. `features/_class_par.py` — compute class-par adjusted speed figures; add `class_par_speed`, `class_par_rank` columns.

**Existing files to be modified:**

6. `features/builder.py` — `_derive_all()`: insert calls to new feature modules (draw_bias, weight_speed, trainer_form, market_movement, class_par) after the existing engine features. Add new config keys to `_load_cfg()`.

7. `features/engine.py` — no structural changes needed (new features are separate modules), but `add_race_complexity` may be enhanced with new inputs.

8. `models/features.py` — `FEATURE_COLS`: add the ~10 new feature column names. `PRICE_FEATURE_COLS`: add `price_steam_pct`, `relative_market_share` (these are market-derived and must stay out of `PRICE_FREE_FEATURE_COLS`). `EMPIRICALLY_DEAD_COLS`: remove `going_speed` from the dead list if the normalizer now populates it (check current null rate).

9. `config.yaml` — add draw_bias configuration block, weight_speed block, trainer_form block under the `model:` or a new `features:` section. Adjust `max_odds` from 4.0 to 6.0 to expand the value-bet pool once features improve (revisit in backtest validation).

10. `backtest/data.py` — `load_panel()`: update to use the new feature whitelist after the matrix is rebuilt. Add logging when `race_uid` degrades to venue-day so the operator can diagnose.

11. `backtest/model.py` — `CatBoostFactory`: set `fl_recalibrate=True` by default so the backtester's per-fold calibration matches the live predictor (currently off by default, producing different probabilities in backtest vs live — a consistency bug).

12. `models/predictor.py` — no structural changes needed (new features flow through `_feature_frame` automatically). Add diagnostic log line counting how many `race_uid`s map to unique races vs venue-day fallbacks.

13. `models/train.py` — no changes needed (new features appear transparently in the feature matrix). Optionally, bump `optuna_trials` from 50 to 75 since the feature space is growing.

14. `memory/MEMORY.md` — add entry for this improvement cycle with cross-references to the audit findings.

**Files to delete or archive:**
- None. All existing files are retained.

[Functions]
Function-level changes for all new and modified functions.

**New functions (by file):**

`features/_draw_bias.py`:
- `add_draw_features(df, cfg) -> pd.DataFrame` — main entry point. Assigns each runner to a draw band, then computes per-course per-band historical win/place rates using strictly-prior runs (leak-safe via `_trailing_fast`).
- `_draw_band(draw_number, edges) -> str` — maps a draw number (1-indexed stall) to a band label.

`features/_weight_speed.py`:
- `add_weight_speed(df, cfg) -> pd.DataFrame` — computes `lbs_per_length * (horse_speed / 100)` adjusted by the weight carried vs the field average, producing a weight-adjusted speed figure.
- `_weight_carried(row) -> float` — extracts weight carried from the unified schema (column `weight_lbs` or derived from `weight_st`).

`features/_trainer_form.py`:
- `add_trainer_form(df, cfg) -> pd.DataFrame` — for each trainer, computes recent (short_window) win rate and compares to long-term (long_window) win rate, producing a z-score of the difference.
- `_recent_strike_rate(df, trainer_id, cutoff_date, window_days) -> float` — leak-safe strike rate over a window.

`features/_market_movement.py`:
- `add_market_movement(df) -> pd.DataFrame` — adds `price_steam_pct`, `relative_market_share`, `market_book_pct`. Uses `morningwap` (pre-off early price) and `odds_decimal` (current board price) — NEVER `odds_finish`. Falls back gracefully when morning prices are unavailable.
- `_market_book_overround(df, group_col) -> pd.Series` — computes the field's book overround per race.

`features/_class_par.py`:
- `add_class_par(df) -> pd.DataFrame` — uses the horse's best recent speed figure and adjusts it by a class-rating difference to produce a class-par speed figure comparable across different class levels.
- `_class_adjustment(race_class, base_class=5) -> float` — speed adjustment per class level.

**Modified functions (by file):**

`features/builder.py`:
- `_load_cfg()` → add new config keys: `draw_bias` (lookback months, min runners, band edges), `weight` (lbs_per_length, use_official_rating), `trainer_form` (short_window_days, long_window_days, min_runners).
- `_derive_all()` → insert calls in order: after `add_career_runs`, before engine features: `add_draw_features`, `add_weight_speed`, `add_trainer_form`, `add_market_movement`, `add_class_par`.

`backtest/data.py`:
- `load_panel()` → add warning logging when `race_uid` degrades to venue-day (count and log). Use updated `PRICE_FREE_FEATURE_COLS` (will be auto-resolved by the factory).

`backtest/model.py`:
- `CatBoostFactory.__init__()` → change default `fl_recalibrate=False` to `fl_recalibrate=True` so backtest probabilities match the live predictor (consistency fix).

`models/features.py`:
- `FEATURE_COLS` → append new columns: `draw_band_encoded`, `course_draw_win_rate`, `course_draw_place_rate`, `weight_adjusted_speed`, `weight_speed_rank`, `trainer_hot_strike_rate`, `trainer_form_zscore`, `price_steam_pct`, `relative_market_share`, `market_book_pct`, `class_par_speed`, `class_par_rank`.
- `PRICE_FEATURE_COLS` → add `price_steam_pct`, `relative_market_share`, `market_book_pct` (market-derived, correctly excluded from price-free variant).
- `EMPIRICALLY_DEAD_COLS` → re-validate per-column null rates after matrix rebuild. `going_speed` may no longer be dead if the normalizer now populates `going`.

[Classes]
Class-level changes for modified classes and dataclasses.

**Modified classes:**

`CatBoostFactory` (backtest/model.py):
- Default `fl_recalibrate: bool` changes from `False` to `True`. This is a one-character change that ensures the walk-forward backtester applies the same favourite-longshot recalibration that the live `Predictor` applies, making backtest results directly comparable to live results.

`ValueConfig` (models/value.py):
- No structural changes needed. The `max_odds` in config.yaml is the source of truth and defaults to 4.0. After feature improvements and matrix rebuild, backtest validation may support widening back to 6.0 (revisit in testing phase).

No new classes are introduced (the new features are pure functions, not class-based).

[Dependencies]
Dependency changes — no new packages, no version bumps.

All new features use only existing dependencies: `numpy`, `pandas`, and the project's own `_trailing_fast` vectorized window module. The `_market_movement.py` module references `morningwap` from the existing unified schema; if `morningwap` is not populated, the columns gracefully degrade to NaN (CatBoost handles NaN natively).

No changes to `requirements.txt` or `requirements-dev.txt`.

[Testing]
Testing approach and validation strategy.

**Required new tests:**

1. `tests/features/test_draw_bias.py` — test draw-band assignment edge cases (draw=0, draw=None, draw > field_size). Test leak-safety (no future runs leak into historical rate). Test min-runners gate (bands with insufficient data return NaN).

2. `tests/features/test_weight_speed.py` — test weight adjustment maths: a horse carrying 5lbs more than field average at 3lbs/length should have speed reduced by ~1.67%. Test missing weight column gracefully produces NaN.

3. `tests/features/test_trainer_form.py` — test that trainer_form_zscore is computed from strictly-prior runs only. Test handling of trainers with fewer than min_runners.

4. `tests/features/test_market_movement.py` — verify that `odds_finish` is NEVER accessed (grep the file for "odds_finish" and assert it doesn't appear). Test that missing `morningwap` produces NaN steam/drift columns.

5. `tests/features/test_class_par.py` — test that class-par speed is monotonic in horse_speed and class_adjustment. Test edge case where race_class is null.

**Existing test modifications:**

6. `tests/models/test_explain.py` — if FEATURE_COLS grows, the explainer tests may need updated feature labels. Check and update `FEATURE_LABELS` in `models/explain.py` for any new features that need human-readable names.

7. `tests/backtest/` — the walk-forward engine tests should still pass since the `CatBoostFactory` uses `PRICE_FREE_FEATURE_COLS` and the new market-movement features are correctly added to `PRICE_FEATURE_COLS` (excluded from price-free).

**Validation strategy (after implementation):**

8. Run `python pipeline.py` end-to-end to regenerate the full feature matrix with new features.

9. Run `python -m models.train --price-free --version-tag v3nf` to train the new price-free model.

10. Run `python -m models.train --reuse-params` to refit v3 calibrators (fix C1).

11. Run `python -m backtest --iterations 400` to produce a new OOS panel.

12. Run `python -m scripts.calib_fl_04_eval` to re-evaluate favourite-longshot recalibration with improved features.

13. Verify: (a) v3 calibrator exists on disk, (b) Brier < 0.30 for v3 won (was 0.296 raw), (c) ECE < 0.10 for v3 won (was 0.42 raw), (d) price-free AUC > 0.68 (was 0.670, should improve with new features), (e) CLV on band[2,4] > −3% (was −4%, target modest improvement from better discrimination).

[Implementation Order]
Numbered sequence of implementation steps, ordered to minimize conflicts and enable incremental validation.

1. **Add new feature modules** — create `features/_draw_bias.py`, `features/_weight_speed.py`, `features/_trainer_form.py`, `features/_market_movement.py`, `features/_class_par.py` with complete implementations. These are standalone and don't affect existing code until wired in.

2. **Update feature whitelist** — modify `models/features.py` to add ~12 new column names to `FEATURE_COLS`, update `PRICE_FEATURE_COLS` to include market-movement features, and revalidate `EMPIRICALLY_DEAD_COLS` (check if `going_speed` is still mostly null after matrix rebuild).

3. **Wire features into the pipeline** — modify `features/builder.py` (`_load_cfg` and `_derive_all`) to call the new feature modules. Update `models/explain.py` `FEATURE_LABELS` if needed for new columns.

4. **Fix config** — update `config.yaml` to add configuration blocks for the new features (draw_bias, weight, trainer_form). Revisit `model.tuning.optuna_trials` (50→75) and `value.max_odds` (4.0→6.0 tentative, validate in step 11).

5. **Fix C1 (v3 calibration)** — run `python -m models.train --reuse-params` to fit and persist `catboost_won_v3_calib.pkl`, `catboost_placed_2_v3_calib.pkl`, `catboost_showed_v3_calib.pkl`. Verify files exist on disk and load correctly in the predictor.

6. **Fix C2 (odds_finish leakage) partially** — the new `_market_movement.py` module uses `morningwap` and `odds_decimal` (pre-off), never `odds_finish`. The historical `implied_prob` still comes from `odds_finish` for existing rows because the parquet was built that way. **Full fix requires the normalizer to build `implied_prob` from `morningwap` or `ppwap` for training rows.** Add a warning in `builder.py` when `implied_prob` is derived from `odds_finish` and the row is a training (labelled) row.

7. **Fix backtest consistency** — change `CatBoostFactory.fl_recalibrate` default from `False` to `True` in `backtest/model.py`. This ensures backtest probabilities go through the same F-L recalibration as live predictions.

8. **Regenerate full feature matrix** — run `python -m utils.normalizer` then `python -m features.builder` to produce a new `data/features/training.parquet` and `data/features.parquet` with all new feature columns present. Verify that all ~42 FEATURE_COLS are populated (check null rates per column).

9. **Retrain v3nf (price-free) model** — run `python -m models.train --price-free --version-tag v3nf` with the new full feature matrix. Verify improved AUC and Brier on the held-out test set.

10. **Retrain v3 (priced) model** — run `python -m models.train --version-tag v3` with the new full feature matrix (now with calibration). Verify `catboost_won_v3_calib.pkl` is saved. Verify Brier ≤ 0.10 and ECE ≤ 0.05 for won target.

11. **Run backtest validation** — run `python -m backtest --iterations 500` on the full matrix, then `python -m scripts.calib_fl_04_eval`. Compare CLV, A/E, and yield against the `calib-fl-04-backtest.md` baseline. If CLV on band[2,4] < −4%, revert `value.max_odds` to 4.0 and document.

12. **Run last-week paper-trade** — run `python -m scripts.last_week_backtest` to verify the latest settled week shows improved ROI and hit rate.

13. **Smoke test the full pipeline** — run `python pipeline.py` end-to-end. Verify the predictor loads all models and produces predictions with the new feature columns present.

14. **Update documentation** — add a new entry to `memory/MEMORY.md` summarizing the improvements, the before/after metrics, and any known limitations (especially noting that CLV may remain negative and picks are paper-trade only).