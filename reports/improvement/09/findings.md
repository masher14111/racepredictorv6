# Step 09 — independent foundation audit: severity-ranked findings

Reviewer: Claude Opus 5. Attempt 1 2026-09-19 (verdict NEEDS_FIX); **attempt 2
2026-09-19 — verdict APPROVED**. Environment: `.venv` Python 3.14.7, pandas
3.0.3, catboost 1.2.10, lightgbm 4.7.0, scikit-learn 1.9.0.

Method: read the actual working-tree diffs and current code for steps 02..08,
then re-derived every load-bearing claim on real data with the scripts in this
directory. Stage summaries were not accepted as evidence. Attempt 2 re-ran every
check against the current code and repaired the two defects that blocked
approval, plus two further defects it found while repairing them.

All attempt-2 output files carry an `_AFTER_*` or `_S09A2` suffix; the unsuffixed
`.txt` files are attempt 1's pre-repair measurements, kept as the before-state.

---

## F1 — HIGH — FIXED — jockey/trainer trailing rates counted races that had not yet run

**Owning repair stage: 04** (point-in-time independent features). The positional
cut predates step 04, but step 04 reported these features as repaired and
"leak-free", so it is squarely in its remit. Attempt 1 routed it for supervised
repair; attempt 2 was re-launched on this stage, so the repair was made here and
re-verified by this stage's own acceptance scripts.

**Locations** — `features/_trailing_fast.py::_bounds` (`strict=False`:
`end = np.arange(len(d))`, a POSITIONAL cut) and
`_windowed_win_rate_days_core`; consumed by
`features/derive.py::add_all_trailing_rates` (`historical_win_rate`,
`jockey_win_rate`, `trainer_win_rate`) and
`features/_trainer_form.py` / `_jockey_form.py`
(`{trainer,jockey}_hot_strike_rate`, `_form_zscore`).

**Why it was wrong.** `race_date` is date-only, so a positional "prior" cut
counted every earlier-*positioned* row sharing the date — and within a day the
frame's row order is not the off-time order (reproduced directly: trainer
T D Easterby, 2024-07-06, 17 races, frame order not off-time order, 60 counted
"prior" pairs actually ran later). A runner in the 13:30 was scored on its yard's
20:20 result.

**Measured before the repair** (`audit_lookahead_magnitude.txt`, on
`data/audit/06/training_refreshed.parquet`, 551,815 rows):
`trainer_win_rate` 271,978 same-day priors counted, **126,408 (46.48%) ran
later**, **25.36%** of runner-events affected; `jockey_win_rate` 265,480 priors,
**132,594 (49.95%) later**, **29.98%** affected; `historical_win_rate`
(horse-keyed) unaffected in practice (2 rows).

**Repair.** `_order_instants` recovers each race's real off-time from the event
key (`race_uid` parses on 100% of rows) and the prior cut is made on that
instant. An event whose key carries no off-time is treated as unavailable in
both directions (it sees no same-day prior and is seen by none). `_chrono_rank`
adds a content-derived tiebreak (event, then runner) so the `lookback_runs` tail
cap picks the same priors regardless of row order.
`features/builder.py::build_inference_matrix` now admits today's already-run
races (still gated on `position.notna()`), closing the train/serve skew: the
serving path used to hard-exclude every same-day row while training counted them.

**Verified after the repair**
- `audit_lookahead_magnitude_AFTER_F1.txt`: **0** later-running priors counted
  (0.00%) for both trainer and jockey; 0 of 276,538 runner-events affected.
  Genuinely-earlier same-day priors still count (251,361 trainer / 265,323
  jockey), so the fix did not simply discard the day.
- `audit_row_order_dependence_AFTER_F1.txt`: **0 rows move** under a pure row
  shuffle, on all 10 measured paths (rate and n, month-window and day-window,
  trainer/jockey/horse). Before: 21.94% (strict=False), 6.78% (strict=True).
- `audit_f1_feature_delta.txt` (what step 10 inherits): `trainer_win_rate` moved
  on **20.59%** of rows (mean |Δ| 0.0116, p95 0.0500), `jockey_win_rate`
  **20.91%**, `historical_win_rate` **0.00%**.
- 10 regression tests added to `tests/features/test_trailing_fast.py` (off-time
  cut in both directions, parametrised over frame order; untimed event; tail-cap
  row-order invariance) and 2 to `tests/features/test_builder.py`.

---

## F7 — HIGH — FIXED — the trainer/jockey "form cycle" day windows were 1,000x too long

Found by attempt 2 while repairing F1; not previously reported by any stage.

`_windowed_win_rate_days_core` compared a hard-coded **nanoseconds**-per-day
constant (`86_400_000_000_000`) against `race_date.asi8`. pandas 3 parses this
project's date column at **microsecond** resolution, so `window_days=14`
subtracted 14,000 days. Measured (`audit_day_window_unit.txt`, 200-day slice):
the nominal 14-day window averaged **108.03** priors and was identical to the
90-day baseline on **100.00%** of comparable rows — so
`trainer_hot_strike_rate` == its own baseline and
`{trainer,jockey}_form_zscore`, which is `(short - long) / SE`, was structurally
~0. Four `FEATURE_COLS` members carried no form-cycle signal at all.

Repaired by deriving the day length from the frame's own datetime resolution
(`_ticks_per_day`). After: 14-day window averages **18.11** priors vs **79.80**
for 90 days; short == long on 9.48% of rows (real short-window ties);
`trainer_form_zscore` non-null 70.58%, std 0.91. Regression test parametrised
over `s`/`ms`/`us`/`ns` frames.

---

## F3 — MEDIUM — FIXED — `race_complexity` was market-derived but in `PRICE_FREE_FEATURE_COLS`

`audit_market_identity.py` perturbs every market price column on a real
120-race slice and re-derives: every `PRICE_FEATURE_COL` moves (expected),
`race_complexity_v2` and `going_speed_v2` do not (their independence is real),
`race_market_entropy` moves (correct, it is the declared market companion) — and
`race_complexity` moved on all 2,354 probe rows while listed as price-free.

Repaired at source: `models/features.MARKET_DERIVED_FEATURE_COLS = ["race_complexity"]`
is now stripped from `PRICE_FREE_FEATURE_COLS`. It stays in `FEATURE_COLS`, and
every existing bundle's own `meta["feature_cols"]` is unchanged (verified:
`catboost_v3nf_meta.json` carries its own 40-column list, so no trained artefact
or serving path changes). `audit/leakage.py::check_provenance` now fails if any
market-derived column re-enters the price-free list.
`audit_market_identity_AFTER_F3.txt`: **no price-free or candidate-independent
feature reacts to a market-price perturbation**.

Consequence to carry forward: the "independent" lines already saved in
`reports/forward_validation_20260918.md` were fitted with `race_complexity` in
scope. They are not price-free and must not be cited as an independent baseline;
step 10's retrain produces the first genuinely price-free line. Adopting
`race_complexity_v2` remains step 10's option, not a requirement of this fix.

---

## F6 — LOW — FIXED — a literal `"nan"` name became a real canonical entity id

`utils/normalizer.py::_canonical_id("nan")` -> `e3f0bf10b8cc4488`, because the
betsp store holds the *string* `"nan"` in some `jockey_name`/`trainer_name`
cells. 66,661 raw rows carry that id; after fusion **1,002 labelled rows
(0.18%)**, all 2026-07-25, pooled 501 distinct horses into one fake jockey and
one fake trainer that then shared a trailing form history (`audit_placeholder_ids
_S09A2.txt` still shows the pool in the pre-fix candidate matrix, which is why
step 10 must rebuild it). Repaired at the normalizer boundary: a narrow set of
rendered-null tokens (`nan`, `none`, `null`, `n/a`, `-`, `--`, matched on both
the raw and the normalized text) now returns `None`, exactly as the empty string
already did. `"unknown"` is deliberately NOT treated as null — that is a source's
judgement about a real value. 8 tests added.

---

## F5 — LOW — FIXED — `runner_status` had no consumer

Step 05 carried a free-tier `runner_status` ("RUNNER"/"NON_RUNNER") from
`scraper/timeform/parser.py` through the normalizer, but nothing read it:
`models/predictor.py::_is_non_runner` detected a withdrawal solely via
`jockey_name == "non runner"`, so a declared non-runner still reached the field
whenever the odds feed carried a normal jockey name. `_is_non_runner` now ORs
both signals; a null `runner_status` (every historical and odds-only row) is
never read as a withdrawal. 2 tests added. Full-field normalization itself was
already correct (`won_prob_normalized` is computed after non-runners are
stripped; the built-in invariant check at `models/predictor.py` enforces
sum-to-1).

---

## F4 — MEDIUM — DEFERRED (step 08's call, confirmed) — `execution.snapshots.race_uid` drops venue

Reproduced again on the current store: **4 of 35,107** distinct off-times carry
more than one venue (0.0114%) — Kilbeggan/Newmarket, Chester/Wexford,
Leopardstown/York. Step 08's assessment holds: settlement builds its own
venue-explicit key and is not exposed (verified adversarially,
`audit_execution_layer_S09A2.txt` D4), so the residue is snapshot-book
composition and `closing_quote`/CLV only, and a fix needs an on-disk schema
migration across `odds_snapshots` and `paper_tickets`. Not a blocker for step 10.

---

## F8 — LOW — NOT FIXED — `backtest/_feature_selection.py` splits by row count

Found by attempt 2. It cuts train/calibration/test with `df.iloc[:i_cal]` —
absolute row positions, the exact construct D22 outlaws — so its two boundaries
can straddle a race. It has no caller in the repo and does not touch training or
serving; it is the one-off study behind `EMPIRICALLY_DEAD_COLS` /
`docs/calibration/feature_selection.json`. The contamination is bounded by two
boundary races, so its conclusions are very unlikely to flip, but step 10 should
not re-run it as-is when revisiting the lean whitelists.

---

## Claims that reproduced cleanly (no defect)

- **Step 02 splits** (`audit_splits_S09A2.txt`): on BOTH the frozen matrix and
  the step-06 candidate, with `models/train.py`'s real WIN-market restriction
  applied first: zero shared `race_uid` across every
  fit/early-stop/calibration/test pair, for all three targets; chronologically
  admissible; the tuner's grouped CV folds race-disjoint and never touching
  calibration or test; test-group membership invariant under a real row shuffle;
  and **0** `race_uid` spanning more than one `race_date` or venue
  (`backtest/splitter.py`'s premise).
- **Step 03/06 market identity** (`audit_market_identity_AFTER_F3.txt`): 0
  duplicate `(race_uid, horse_id, market_type)`; WIN and PLACE prices genuinely
  distinct (only 2.00% coincide); finishing position agrees across a runner's two
  rows on 100% of 275,277 pairs; `field_size` == that race's own book size and
  `overround_norm_prob` sums to 1 within each book to 2.2e-16 on a freshly
  derived pre-label-filter frame; `market_rank` never escapes its own book.
- **Step 04 market-duplicate collapse** (`audit_trainer_collapse_S09A2.txt`):
  after D38, runner-events lost **0** for trainer, jockey and horse keys, and a
  runner-aware recomputation of `trainer_win_rate` differs on **0** rows.
- **Step 07 snapshots** (`audit_execution_layer_S09A2.txt` D3, real
  `data/races.db`): 3,547 rows, no future-stamped row, both append-only triggers
  present, UPDATE physically refused, and an as-of read one second before the
  newest fetch does not return it.
- **Step 08 reporting** (D2, real ledger): 988 rows, `{'PASS': 988}`, 0
  candidates, 0 settled — the corrected accounting is what the live store shows.
- **Step 08 settlement identity** (D4): two meetings at one off-time yield two
  distinct keys and resolve to their own results; the day/venue fallback is
  venue-scoped; two disagreeing results under one identity are refused, not
  guessed. Idempotency covered by
  `tests/execution/test_tickets.py::test_double_settlement_is_refused` and
  `tests/scripts/test_daily_paper_loop.py`.
- **Forward-validation honesty**: `reports/forward_validation_20260918.md` states
  NO-GO, forward gate 9/9 criteria failed, 0 qualified bets. No claim of
  completed forward validation exists anywhere in the memory chain.
- **Live fail-closed behaviour** (re-checked this session): today's board is
  1,236 rows across 3 odds sources, 22,179-22,349 s old against a 900 s TTL, so
  `build_inference_matrix` returns 0 rows. Step 05's declared-card provenance
  path therefore still has no live exercise — exactly the limitation stage 05
  recorded.
- **Unrelated dirty work preserved**: nothing staged, stashed or reverted; no
  stash entries. Modified tracked files went 94 -> 97, the three being exactly
  the files this attempt touched that were previously clean (`audit/leakage.py`
  +6, `audit/final_eval.py` +3, `tests/models/test_predictor.py` +30). Every
  other file this attempt edited was already dirty at session start, and its
  pre-existing changes are untouched. The 3 pre-existing deletions
  (`ui/_theme.py`, `ui/performance.py`, `ui/predictions.py`) and the untracked
  set are unchanged.
