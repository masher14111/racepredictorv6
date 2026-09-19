# Data and evaluation contracts

Updated: 2026-09-19. Keep below 200 lines. This is the shared identity/timing/evaluation
contract for prompts 02..18. It records what is verified today, not aspirational design.

## Race/horse/market identity

- Canonical race identity is `race_uid` (`"<Venue>|<ISO-8601 local race datetime>"`),
  verified unique per race (29,512 unique values in `data/features/training.parquet`,
  no race_uid maps to more than one distinct value on the other identity axis checked).
- `race_id` in the same file is entirely null (`nunique() == 0`, only `None`). Treat it as a
  dead/unused column. Any future join or split MUST use `race_uid`, never `race_id`.
- Horse identity within a race is row-level (one row per horse per race per market_type);
  no horse-identity column audit was run in this stage — deferred to whichever later stage
  first needs cross-race horse history joins.
- `market_type` is a two-value categorical: `WIN` (148,860 rows) and `PLACE` (111,630 rows)
  in the current training file. Treat WIN/PLACE as separate label spaces sharing one race;
  do not silently merge them without checking a consuming stage's assumptions.

## Timezone and publication/fetch/as-of semantics

- `race_date` in the training matrix is a **date-only** value (midnight UTC-stamped), not a
  race post time. It is not sufficient on its own to prove no-lookahead at the row level.
- `race_uid` embeds a local race datetime string (e.g. `Doncaster|2026-03-28T13:20`); this is
  the closest available field to true off-time and should be preferred over `race_date` for
  any within-day ordering or cutoff logic.
- Declared operating timezone is Europe/Dublin per AGENTS.md; verify any new timestamp field's
  tz-awareness before using it in a cutoff comparison — do not assume naive timestamps are UTC.
- No independently captured "as-of" odds/timing snapshot corpus was found in this stage beyond
  `data/execution/race_facts.parquet` and the paper-ticket store; provenance of those feeds is
  unverified here and is explicitly in scope for step 07 (as-of odds capture), not step 01.

## Independent vs. market-assisted features

- `models/predictor.py` and prior dated review already separate an "independent" line
  (`audit_independent`) from market-adjusted lines (`audit_market_adjusted*`) in
  `reports/forward_validation_20260918.md`; that separation is real and reproducible today
  (independent LL 1.8740 vs market LL 1.6878 over the same 904-race window, reproduced here
  from the existing saved report, not re-run in this stage).
- `race_complexity` is globally normalized and market-derived. Verified structurally in step 09:
  perturbing every market price column and re-deriving a real 120-race slice moved it on 100% of
  rows, while `race_complexity_v2` and `going_speed_v2` did not move at all and
  `race_market_entropy` correctly did — yet it sat in `PRICE_FREE_FEATURE_COLS`.
  **FIXED in step 09 (D40):** `models/features.MARKET_DERIVED_FEATURE_COLS` now strips it from
  the price-free whitelist (it stays in `FEATURE_COLS`, and every existing bundle's own
  `meta["feature_cols"]` is untouched, so no trained artefact changes).
  `audit/leakage.py::check_provenance` enforces the strip. Re-run after the fix: **no price-free
  or candidate-independent feature reacts to a market-price perturbation**. Note this means the
  "independent" lines already SAVED in `reports/forward_validation_20260918.md` were trained with
  `race_complexity` in scope — they are not price-free and cannot be cited as an independent
  baseline; step 10's retrain produces the first genuinely price-free line.
  Evidence: `reports/improvement/09/audit_market_identity_AFTER_F3.txt`.
- `horse_speed` is a finishing-percentile proxy (partly outcome-derived); live inference uses
  a last-known fallback for connections. Treat it as market/outcome-adjacent, not a clean
  independent signal, until a stage re-derives it from raw sectionals.

## Missingness

Reproduced today on `data/features/training.parquet` (260,490 rows):

| column | null fraction |
|---|---|
| timeform_rating | 1.00 |
| rating_rank | 1.00 |
| pace_bias | 1.00 |
| race_class | 1.00 |
| class_change | 1.00 |
| recent_form_avg | 1.00 |
| going_speed | 0.79 (20.92% filled) |

These six fully-null columns must not be used as live features until a stage restores their
source. `going_speed` is real but sparse; consuming stages must handle the missing 79% rather
than imputing silently.

## Candidate artifact ownership

- Existing champion artifacts (CatBoost binary models, LightGBM grouped-softmax) are owned by
  `models/` and `data/audit/stage4/`; do not overwrite them in-place.
- New experiments from steps 02..18 must write to distinct `data/audit/<stage>/` or
  `reports/improvement/<NN>/` directories with their own run IDs; never reuse an existing
  candidate directory name.
- Paper-ticket store (`data/execution/*`) is the one ground-truth ledger for forward evidence.
  Corrected in step 08 and re-read from the live store in step 09: **988 rows, all `PASS`
  disclosure records, 0 CANDIDATE wagers, 0 settled**. The earlier "988 open tickets" phrasing
  was a PASS/wager conflation, not open exposure. No stage may fabricate or backdate settlement
  rows into it.

## Complete-race splitting and final-test policy

- **Fixed in step 02** (previously a verified defect, stage 01): `models/train.py::_time_split`
  used to sort the full frame by `race_date` (date-only) and cut by absolute row count, not by
  day or by race_uid boundary, reproducibly leaking 28 (train/test), 13 (fit-core/calibration)
  and 45 (fit/early-stop) shared race_uid on current data. `models/split_utils.py` now provides
  `chronological_group_split` (single cut) and `group_time_series_split` (walk-forward K-fold),
  both cutting on whole `race_uid` groups, deterministic regardless of input row order. `train.py`
  uses these for the outer test carve, the calibration carve and the early-stop carve; `tuner.py`
  uses the K-fold variant for Optuna CV (`run_study(..., race_ids=, order_keys=)`);
  `ensemble_experiment.py`'s calibration carve and stacker OOF folds use the same primitives.
  Re-verified 2026-09-19 on the live matrix (`reports/improvement/02/reproduce_overlap.py`,
  `reports/improvement/02/overlap_audit_20260919.txt`): **zero** shared race_uid across every
  fit/tune/early-stop/calibration/test boundary, for all three targets. Tuning also no longer
  runs on rows that later become the calibration slice (it was previously called on the full
  80% train block before the calibration carve; now it only sees the post-calibration-carve core).
  `backtest/splitter.py`'s `walk_forward_folds` was audited too and found already race-safe: it
  cuts on whole calendar days via boolean masks on `race_date`, and since every row of one race
  shares exactly one `race_date`, a day-level cut can never split a race (test added:
  `tests/backtest/test_splitter.py::test_multiple_races_sharing_a_date_never_split_within_the_race`).
- Final-test policy: the July 2026 window (2026-06-13..2026-07-25, 904 races) in
  `data/audit/stage4/final_evaluation.json` is already-observed and MUST NOT be reused as an
  untouched final test for any new candidate. New candidates need a freeze policy defined
  before their first look, per step 10.

## Trailing-window point-in-time status (step 09 audit)

- A race's true off-time is recoverable from `race_uid` on **100%** of rows of both the frozen
  and the step-06 candidate matrix. It — not the date-only `race_date` — is the only valid
  ordering key for a within-day point-in-time cut.
- **FIXED in step 09 attempt 2 (D39).** `features/_trailing_fast.py::_bounds` (`strict=False`)
  and `_windowed_win_rate_days_core` used to cut "priors" POSITIONALLY inside a `race_date`,
  and the frame's within-day row order is not the off-time order. Measured before the repair on
  the step-06 candidate matrix: `trainer_win_rate` counted 126,408 same-day priors that ran
  LATER (46.48% of its same-day priors), affecting 25.36% of runner-events; `jockey_win_rate`
  132,594 (49.95%), affecting 29.98%. The cut now runs on the real off-time recovered from
  `race_uid`, with a content-derived tiebreak so the `lookback_runs` tail cap is row-order
  invariant. After the repair: **0 later-running priors counted** (251,361 trainer / 265,323
  jockey genuinely-earlier same-day priors still count) and **0 rows move under a pure row
  shuffle** across every window path. `features/builder.py::build_inference_matrix` now admits
  today's already-run races (`position.notna()`) so the serving path sees the same history the
  training matrix does. Feature impact for step 10: `trainer_win_rate` moved on 20.59% of rows,
  `jockey_win_rate` on 20.91%, `historical_win_rate` on 0.00%.
  Evidence: `audit_lookahead_magnitude_AFTER_F1.txt`, `audit_row_order_dependence_AFTER_F1.txt`,
  `audit_f1_feature_delta.txt`.
- **FIXED in step 09 attempt 2 (D41).** The trainer/jockey day windows compared a hard-coded
  nanoseconds-per-day constant against `race_date.asi8`, which pandas 3 resolves at MICROSECOND
  precision here — so `window_days=14` actually meant 14,000 days. `trainer_hot_strike_rate` and
  its 90-day baseline were the same all-history rate on **100%** of comparable rows, making
  `{trainer,jockey}_form_zscore` structurally ~0. Repaired 14-day window now averages 18.11
  priors vs 79.80 for 90 days. Evidence: `audit_day_window_unit.txt`.
- **FIXED in step 09 attempt 1 (D38).** The market-duplicate collapse now keys on
  `(entity keys..., RUNNER, race event)`. Keyed on `(entity, race)` alone it also deleted every
  runner but one from a yard's multi-runner race — 6.50% of real trainer runner-events, moving
  `trainer_win_rate` on 20.65% of rows. `data/audit/04/training_pit_fixed.parquet` and
  `data/audit/06/training_refreshed.parquet` predate D38/D39/D41 and must be REBUILT, not reused.

## Prediction cutoff

- No explicit "minutes before off" prediction-time cutoff constant was found in config.yaml or
  the scraper/model code (only a date-level `train_cutoff` and an alerting
  `race_soon_minutes: 10`, which is a UI/alert concern, not a feature-freeze rule).
- Provisional assumption for step 02+ research splits only (does not change the live service):
  a **10-minutes-before-off** feature-freeze cutoff, matching the existing
  `race_soon_minutes` alert value for consistency. Any stage relying on this must re-confirm
  it is still just a research assumption, not a deployed behavior, before citing it as fact.

## Data gaps (not to be invented)

- No independently labelled text-extraction benchmark corpus exists yet; `data/spotlight.parquet`
  has 243 rows, all dated 2026-06-17 — a single-day sample, insufficient for any multi-period
  text-feature claim (reproduced today, matches prior dated finding).
- No paid timing/sectionals data access was found or purchased in this stage.
- Hosted DeepSeek trial: `OPENROUTER_API_KEY` is absent from this shell's environment as of
  2026-09-19; the hosted trial stays unavailable until a key is supplied locally. Not searched
  for or printed.
