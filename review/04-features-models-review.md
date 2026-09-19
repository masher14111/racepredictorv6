# Race Predictor v3 — Features + Models (ML Pipeline) Review

_Reviewed: 2026-06-13 | All 12 files read in full + cross-checked against `utils/normalizer.py` (canonical schema), `config.yaml`, and the existing test suite._

**Scope:** `features/{engine,builder,derive,fuse,labels}.py` → `models/{train,tuner,predictor,evaluate,retrain_trigger,targets,features}.py`.

---

## TL;DR — answers to the 5 review questions

1. **Valid DataFrame matching `train.py`?** ✅ Structurally yes — every `FEATURE_COLS` entry is produced, and `position` / `race_date` / `implied_prob` always exist (canonical schema guarantees them). But two correctness bugs live _inside_ the pipeline (sample-weight misalignment in `train.py`; trailing-rate denominator in `derive.py`).
2. **Column names consistent features → models?** ✅ for `FEATURE_COLS` (all 28 align exactly). ❌ for the **predictor output dict**, which reads `jockey`/`trainer` while the canonical columns are `jockey_name`/`trainer_name` — every selection ships blank jockey/trainer.
3. **`predictor` returns composite_score in [0,1]?** ✅ Yes. Weights `{won:.50, placed_2:.30, showed:.20}` sum to 1, each prob ∈ [0,1], and `wsum` re-normalises when a target is missing.
4. **`retrain_trigger` reads PSI threshold from config?** ✅ Yes — `retrain.psi_threshold` (0.2) is read correctly at `retrain_trigger.py:47`.
5. **Import cycles features/ ↔ models/?** ✅ None. `features/` imports only `features.*` + `utils.*`. The one real cycle (`train` ↔ `retrain_trigger`) is broken with a lazy import at `retrain_trigger.py:324`.

> ⚠️ **Premise mismatch:** the task brief says "LightGBM + CatBoost". The code is **CatBoost-only** (`grep lightgbm models/` → nothing). `config.yaml:22-24` `model_weights.lightgbm/baseline` is **dead config** — no ensembling, no LightGBM, no baseline model anywhere.

---

## Feature Pipeline

**Verdict: SOUND architecture, leak-safe by design, with one denominator bug that silently deflates the headline horse/jockey/trainer win-rates.**

**What works:**

- `builder._derive_all` is the single shared path for **both** train and serve (`build_training_matrix` and `build_inference_matrix` both call it), so train/serve feature parity is structurally guaranteed. Training = `position.notna()` rows; inference = `position.isna()` rows. Clean split.
- Derivation runs over the **full** matrix (history + live together) _before_ the position split, so a live runner's trailing rates are computed from its prior historical runs. Correct and necessary.
- Leak-safety is genuinely careful: `engine._trailing_rate`, `add_speed_figures`, and `derive.add_trailing_rates` all read **strictly-prior** runs bounded by `lookback_months` AND `lookback_runs`. `add_speed_figures` builds `_run_speed` from the current row but only ever consumes it for _future_ rows (drops it before return) — no self-leak.
- `race_date` is coerced to tz-aware UTC (`builder.py:45`) immediately after fuse, so all `cutoff - window` datetime math is well-defined.
- `fuse._first_valid` correctly takes the first non-null in source-priority order (`timeform > betsp > live`), so e.g. a real `position` from betsp wins even if a higher-priority timeform row has it null.

**Column schema produced** (model-facing — all consumed by `models/features.FEATURE_COLS`):

```
implied_prob, overround_norm_prob, log_odds, market_rank, field_size,     ← derive.add_odds_features
going_speed,                                                              ← derive.add_going_speed
class_change,                                                             ← derive.add_class_change
distance_furlongs,                                                        ← derive.add_distance_furlongs
recent_form_avg, recent_form_wins, recent_form_runs,                      ← derive.add_recent_form
rating_rank,                                                              ← derive.add_rating_rank
historical_win_rate, historical_place_rate,                              ← derive.add_all_trailing_rates
jockey_win_rate, trainer_win_rate,                                       ← derive.add_all_trailing_rates
horse_speed, horse_speed_rank,                                           ← engine.add_speed_figures
jt_combo_win_rate, jt_combo_runs,                                        ← engine.add_combo_win_rate
going_pref_win_rate, going_pref_place_rate,                              ← engine.add_going_preference
pace_bias,                                                               ← engine.add_pace_bias
ew_value_index,                                                          ← engine.add_ew_value_index
odds_drift, odds_value_delta,                                            ← engine.add_odds_delta
race_complexity                                                          ← engine.add_race_complexity
timeform_rating                                                          ← RAW passthrough (canonical col)
```

Plus passthrough canonical cols (`race_date, race_time, venue, horse_id, horse_name, jockey_name, trainer_name, position, sp, odds_decimal, ew_places, ew_reduction, morningwap, …`) and labels `won, placed` (training only).

**Bugs:**

1. **🔴 `derive.add_trailing_rates` counts null-position priors in the denominator** (`features/derive.py:56-64`). It does `n = len(prior)` over _all_ prior runs in the window, then `wins = (prior["position"] == 1).sum()`. Prior rows whose `position` is NA (which per the scraper review is **most** historical rows — Timeform/Betfair results are WAF-blocked / stub) count toward `n` but can never count as a win/place. Result: `historical_win_rate`, `historical_place_rate`, `jockey_win_rate`, `trainer_win_rate` are systematically **deflated toward 0**. `engine._trailing_rate` (`features/engine.py:54`) does it _correctly_ — `prior = prior[prior["position"].notna()]` before `n = len(prior)`. The two trailing-rate implementations disagree on denominator semantics, and the primary horse/jockey/trainer features use the wrong one. **Fix:** add `prior = prior[prior["position"].notna()]` before `n = len(prior)` in `derive.add_trailing_rates` (and keep `runs_in_window` = known-result runs, or split into a separate "total runs" column if you genuinely want both).

2. **🟠 `derive.add_trailing_rates` lacks the explicit `< cutoff` guard** (`features/derive.py:54-55`). It relies purely on positional slicing `g.loc[idx[:pos]]` for "strictly prior", whereas `engine._trailing_rate:49` adds `prior = prior[prior["race_date"] < cutoff]`. After `fuse` the runner key is unique per `(race_date, venue, horse_id)`, so same-date duplicates are essentially impossible and this is **low-risk today** — but the inconsistency should be closed so the two functions can't diverge under future data.

3. **🟠 Live runners have NULL `jockey_id`/`trainer_id`** (data contract, not a code defect in `features/`). `normalizer._from_live_odds` only derives `horse_id`; `jockey_id`/`trainer_id` are reindexed to NA. So for live rows, `jockey_win_rate`, `trainer_win_rate`, `jt_combo_win_rate`, `going_pref_*` are computed over the NA-keyed group (meaningless). `historical_win_rate` / `going_pref` _can_ still match history because `horse_id` is a stable name-hash shared across sources. **Net:** jockey/trainer/combo signal is effectively absent at inference time. Document this and consider dropping those features from the live composite, or capturing jockey/trainer in the live scrapers.

4. **🟡 `add_class_change` keys on `horse_name`, trailing rates key on `horse_id`** (`derive.py:32-33` vs `:73`). Mixed entity keys. `horse_id` = blake2b(`norm_horse(name)`), so they're 1:1 in practice, but using `horse_id` everywhere would be consistent and null-safe.

---

## Training Pipeline

**Verdict: RUNS and saves correct artefacts, but the odds-inverse sample weighting is silently applied to the WRONG rows in production. The test suite masks it.**

**What works:**

- `train()` is disk-free when given a `df` (good for tests), filters `FEATURE_COLS` to those actually present (`train.py:115`), logs the missing set, and persists `feature_cols` into `meta.json` so the predictor scores on exactly the trained columns.
- Time-based split (`_time_split` sorts by `race_date`, last 20% = test) is the right choice for racing (no future leakage into test). `test_time_split_train_before_test` confirms it.
- Per-target null-dropping (`tr_mask`/`te_mask`), early-stopping on the last 10% of (already time-sorted) train, `auto_class_weights="Balanced"`, `allow_writing_files=False`, and `save_reference_snapshot` wiring are all correct.
- `tuner.run_study` returns **only tuned params**; `train` merges fixed params (`loss_function`, `eval_metric`, …) before the final fit. No param leakage between tuning and final fit. `StratifiedKFold` is appropriate for the imbalanced binary targets.

**Bugs:**

1. **🔴 Sample weights are misaligned with training rows** (`train.py:101-105, 141`). Sequence:

   ```python
   w_full = _sample_weights(df, …)          # :101  weights in df's ORIGINAL order
   train_df, test_df = _time_split(df, …)   # :103  _time_split RE-SORTS df by race_date
   w_train_full = w_full[:n_train]          # :105  first n_train weights of the ORIGINAL order
   …
   w_tr = w_train_full[tr_mask]             # :141  positional mask over the SORTED train_df
   ```

   `build_training_matrix` returns rows ordered by **`(horse_name, race_date)`** (because `derive.add_class_change:32` sorts that way and nothing re-sorts afterward). `_time_split` then re-sorts to **`race_date`** order. So `w_full[:n_train]` (horse-name order) does not correspond to `train_df` (race-date order) — every row gets some _other_ horse's odds-inverse weight. The model trains, AUC looks plausible, nothing crashes → **silent quality degradation**.
   **Why the tests miss it:** `tests/models/test_train._synthetic_df` builds `race_date = date_range(freq="D")` — already strictly ascending — so `_time_split`'s sort is a no-op and `w_full[:n_train]` happens to align. Real data is not pre-sorted by date.
   **Fix:** attach the weight as a column before the split so it travels with the rows:

   ```python
   df["_sample_weight"] = _sample_weights(df, cfg["max_sample_weight"])
   train_df, test_df = _time_split(df, cfg["test_size"])
   …
   w_tr = train_df.loc[tr_mask, "_sample_weight"].to_numpy(dtype=float)
   ```

   (and drop `_sample_weight` from `avail_cols` — it's not in `FEATURE_COLS`, so it's already excluded).

2. **🟡 Double weighting: `auto_class_weights="Balanced"` AND odds-inverse `sample_weight`** (`train.py:169,176`). CatBoost multiplies class weights by sample weights. Balanced class weights already up-weight the rare positive class; the odds-inverse weights _also_ up-weight longshots (which correlate with the negative class). The combined effect on the effective objective is non-obvious and probably not what's intended. Pick one, or verify the interaction is desired.

3. **🟡 `placed` (top-3) computed twice, then shadowed.** `labels.add_labels` (called in `build_training_matrix`) adds `won` + `placed`; `train` then calls `targets.add_targets` which re-adds `won` and adds `placed_2`/`showed` where `showed` == `placed`. The `placed` column is dead weight (never a training target). Harmless, but `showed` and `placed` being identical is a foot-gun if someone later trains on `placed`.

4. **🟢 Note:** `X_tr = X_tr_raw.fillna(np.nan).astype(float)` — `.fillna(np.nan)` is a no-op; `going_speed` is object dtype but `astype(float)` coerces None→nan cleanly. Works, just slightly confusing.

---

## Prediction

**Verdict: COMPOSITE SCORING and EW/odds logic are correct; one column-name bug blanks jockey/trainer in every selection.**

**What works:**

- `composite_score` ∈ [0,1] guaranteed (weights sum to 1, probs ∈ [0,1], `wsum` re-normalises for missing targets) — answers Q3.
- `_score` scores on `[c for c in self._feature_cols if c in df.columns]` **in feature_cols order**, and `self._feature_cols` is overridden from `meta.json` → predictor uses exactly the columns (and order) the model trained on. Correct CatBoost contract.
- NaN-safe odds handling: `_decimal_odds` returns `np.nan` when price unknown; pandas `< / >=` against NaN → `False`, so unknown-priced runners are neither excluded nor EW-flagged (`predictor.py:165-173`). Nicely reasoned.
- Atomic cache write (`tmp` → `replace`), top-N selection, `excluded_low_odds` surfacing, and the alert snapshot/diff against the previous run are all coherent. `notifier.race_soon_minutes` / `odds_drop_threshold_pct` exist and match `utils/notifications.py:149-150`.

**Bugs:**

1. **🟠 Jockey/trainer always blank in output** (`predictor.py:283-284`). `_runner_dict` reads `row.get("jockey", "")` / `row.get("trainer", "")`, but the canonical columns (`normalizer.py:23-24`) are **`jockey_name` / `trainer_name`**. There is no `jockey`/`trainer` column, so every selection's `jockey` and `trainer` fields are `""`. **Fix:** `row.get("jockey_name", "")` / `row.get("trainer_name", "")`. (Also note live data has these null anyway per Feature bug #3, but historical-matched live horses won't surface their jockey/trainer until both this rename _and_ live jockey capture are fixed.)

2. **🟡 `predict_proba` failure silently zeroes a target** (`predictor.py:142-146,154`). If a model raises (e.g. live matrix has fewer feature columns than the model expects — possible if a derive column is entirely absent), the prob is set to NaN, `.fillna(0.0)` makes it contribute 0 to the numerator while its weight still counts in `wsum` → composite is deflated, only logged at WARNING. Since train/serve share `_derive_all` the column set should match, but the failure mode is invisible in the returned scores. Consider failing louder or excluding the weight when a target errors.

3. **🟡 `implied_prob` NaN leaks into JSON** (`predictor.py:287`). `round(float(row.get("implied_prob") or 0), 4)` — `float('nan') or 0` evaluates to **NaN** (NaN is truthy), so `implied_prob` can be the literal `NaN` in `predictions.json` (invalid JSON for strict external parsers — same class of issue flagged in the scrapers review for cache files). Use an explicit `pd.isna` guard.

4. **🟢 Note:** `_decimal_odds` falls back to a `decimal_odds` column (`predictor.py:259`) that doesn't exist in the canonical schema (it's `odds_decimal`). The primary path (`1/implied_prob`) works, so the fallback is just dead — but it confirms the column-name drift between `predictor.py` and the canonical schema (same root cause as bug #1).

---

## Drift Detection

**Verdict: CORRECT mechanics (PSI + KS, config-driven, archive/prune/decay-log all sound), but the trigger is mis-tuned and the reference/current populations aren't comparable.**

**What works:**

- `_load_cfg` reads `retrain.psi_threshold` / `ks_pvalue_threshold` / `min_drift_features` / `archive_keep` / `auc_decay_threshold` correctly (`retrain_trigger.py:42-53`) — answers Q4.
- `_psi` strips NaN, builds quantile edges from the reference, dedups edges, and adds an epsilon to avoid `log(0)` — numerically safe. `check_drift` skips features absent from either frame.
- Archive → prune (keep last N) → retrain → AUC-decay log is well-ordered, and `old_meta` is read _before_ `train()` overwrites it. `train` is lazily imported (`:324`) to break the `train ↔ retrain_trigger` cycle. `save_reference_snapshot` writes the baseline after every successful train so the reference always matches the deployed model.

**Concerns (no hard crash, but the detector will behave badly in production):**

1. **🟠 KS + `min_drift_features=1` makes the trigger fire almost always.** The KS two-sample p-value shrinks toward 0 as sample size grows; on a multi-thousand-row matrix, nearly any feature's distribution differs "significantly", so `ks_pvalue < 0.05` will be true for ≥1 feature on essentially every check → retrain triggers constantly. PSI>0.2 is the robust criterion; the KS-OR clause (`retrain_trigger.py:118-120`) largely defeats it. Recommend either dropping KS from the trigger (keep it as a reported diagnostic only) or raising `min_drift_features`.
2. **🟠 Reference vs current are different populations.** Reference = `train_df` (the time-split **training** subset, labelled, `position.notna()`). Current = `data/features.parquet`, which `build_training_matrix` writes as the **full** matrix _including_ unlabelled live + future rows (`builder.py:71-76`). So drift is measured between "past labelled runners" and "everything including today's live runners" — an apples-to-oranges comparison that will inflate PSI/KS regardless of true drift. Compare against a like-for-like current slice (e.g. recent labelled rows, or at least the same `position.notna()` filter).

---

## Data contract (features/ → models/ column alignment)

`FEATURE_COLS` (models/features.py) vs producer in features/:

| Column                             | Producer                      | Aligned                                              |
| ---------------------------------- | ----------------------------- | ---------------------------------------------------- |
| implied_prob                       | derive.add_odds_features      | ✓                                                    |
| overround_norm_prob                | derive.add_odds_features      | ✓                                                    |
| log_odds                           | derive.add_odds_features      | ✓                                                    |
| market_rank                        | derive.add_odds_features      | ✓                                                    |
| field_size                         | derive.add_odds_features      | ✓                                                    |
| going_speed                        | derive.add_going_speed        | ✓                                                    |
| class_change                       | derive.add_class_change       | ✓                                                    |
| distance_furlongs                  | derive.add_distance_furlongs  | ✓                                                    |
| recent_form_avg / \_wins / \_runs  | derive.add_recent_form        | ✓                                                    |
| timeform_rating                    | RAW (canonical)               | ✓ (all-null until a Timeform session cookie is set)  |
| rating_rank                        | derive.add_rating_rank        | ✓                                                    |
| pace_bias                          | engine.add_pace_bias          | ✓ (all-null until pace_rating available)             |
| historical_win_rate / \_place_rate | derive.add_all_trailing_rates | ⚠️ present but **deflated** (Feature bug #1)         |
| jockey_win_rate                    | derive.add_all_trailing_rates | ⚠️ deflated + null at inference (Feature bugs #1,#3) |
| trainer_win_rate                   | derive.add_all_trailing_rates | ⚠️ deflated + null at inference                      |
| jt_combo_win_rate / jt_combo_runs  | engine.add_combo_win_rate     | ⚠️ null/meaningless at inference (Feature bug #3)    |
| going_pref_win_rate / \_place_rate | engine.add_going_preference   | ✓ (horse-keyed; works at inference)                  |
| ew_value_index                     | engine.add_ew_value_index     | ✓                                                    |
| odds_drift                         | engine.add_odds_delta         | ✓ (needs `morningwap`+`sp`; betSP-only)              |
| odds_value_delta                   | engine.add_odds_delta         | ✓                                                    |
| horse_speed / horse_speed_rank     | engine.add_speed_figures      | ✓                                                    |
| race_complexity                    | engine.add_race_complexity    | ✓                                                    |

**Label/target columns:** `won` (labels.add_labels → re-derived by targets.add_targets), `placed` (labels), `placed_2` + `showed` (targets) — all ✓; `placed`≡`showed` (redundant).

**Predictor output-dict ↔ canonical raw columns:**

| predictor reads                         | canonical has                 | Aligned                                |
| --------------------------------------- | ----------------------------- | -------------------------------------- |
| horse_id, horse_name, venue, race_time  | same                          | ✓                                      |
| implied_prob, composite_score, \*\_prob | derived                       | ✓                                      |
| `jockey`, `trainer`                     | `jockey_name`, `trainer_name` | ✗ **always blank** (Prediction bug #1) |
| `decimal_odds` (fallback)               | `odds_decimal`                | ✗ dead fallback (primary path OK)      |

**Verdict: model feature contract is ✓; the predictor↔raw-column contract has a real ✗ (jockey/trainer).**

---

## Critical fixes needed

| #   | Severity    | File:line                                              | Fix                                                                                                                                                                                                                            |
| --- | ----------- | ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | **🔴 HIGH** | `models/train.py:101-105,141`                          | Sample weights misaligned after `_time_split` re-sorts. Carry weight as a df column through the split: `df["_sample_weight"]=_sample_weights(...)`, then `w_tr = train_df.loc[tr_mask,"_sample_weight"].to_numpy()`.           |
| 2   | **🔴 HIGH** | `features/derive.py:56-64`                             | `add_trailing_rates` counts null-position priors in the denominator → deflates `historical_/jockey_/trainer_win_rate`. Add `prior = prior[prior["position"].notna()]` before `n = len(prior)` (match `engine._trailing_rate`). |
| 3   | **🟠 MED**  | `models/predictor.py:283-284`                          | `_runner_dict` reads `jockey`/`trainer`; canonical cols are `jockey_name`/`trainer_name`. Rename.                                                                                                                              |
| 4   | **🟠 MED**  | `models/retrain_trigger.py:118-120` + `config.yaml:83` | KS-OR clause with `min_drift_features=1` makes drift fire on nearly every check at scale. Drop KS from the trigger (keep as diagnostic) or raise `min_drift_features`.                                                         |
| 5   | **🟠 MED**  | `models/retrain_trigger.py:248-264`                    | Reference (train split) vs current (`features.parquet`, includes unlabelled live rows) are not comparable populations → inflated drift. Filter current to `position.notna()` (or a recent-labelled slice).                     |
| 6   | **🟡 LOW**  | `models/train.py:169,176`                              | `auto_class_weights="Balanced"` + odds-inverse `sample_weight` double-weight the objective. Choose one or confirm intended.                                                                                                    |
| 7   | **🟡 LOW**  | `features/derive.py:54`                                | Add explicit `prior = prior[prior["race_date"] < cutoff]` to match `engine._trailing_rate` (defensive; low risk post-fuse).                                                                                                    |
| 8   | **🟡 LOW**  | `models/predictor.py:287`                              | `float(row.get("implied_prob") or 0)` yields NaN (NaN is truthy) → invalid JSON. Use explicit `pd.isna` guard.                                                                                                                 |
| 9   | **🟢 INFO** | `config.yaml:22-24`                                    | `model_weights.lightgbm/baseline` is dead config — pipeline is CatBoost-only, no ensemble. Remove or implement.                                                                                                                |

---

## Next

**Prompt 5:** Apply fixes #1 (sample-weight alignment) and #2 (trailing-rate denominator) first — they're silent model-quality killers — then add regression tests with **unsorted, NA-position-heavy** synthetic data (the current `_synthetic_df` is pre-sorted and fully-labelled, which is exactly why both bugs hide); afterward review the Streamlit `ui/` layer for how it consumes `predictions.json`.
