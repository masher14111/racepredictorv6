---
name: model-17-inference
description: Live-inference contract for models/predictor.py — imputation rules, upcoming guard, coherence/normalization, typed result object (2026-06-16)
metadata:
  type: project
---

Hardened `models/predictor.py` for messy live data (debut horses, unknown
jockey/trainer, no form). The serving contract, so future changes don't silently
regress it:

**Imputation = NaN passthrough, full-width + ordered.** `Predictor._feature_frame(df, cols)`
builds the model matrix with _exactly_ `cols`, in order, inserting an all-NaN
column for any feature absent from the live matrix. Two failure modes this fixes:
(1) the old `[c for c in cols if c in df.columns]` dropped missing columns →
narrower X → CatBoost shape mismatch → caught → **all-NaN probs silently emitted**;
(2) dropping a _middle_ column shifted every later feature into the wrong slot.
Missing values stay NaN (never filled 0/mean): CatBoost learns a dedicated
"missing" split at train time (features.py: "CatBoost handles NaN natively"), so
NaN is the imputation that _matches training_. Used by both the main models and
the value model (`_score_value`).

**Upcoming-only guard (historical-join-miss regression).** `_guard_upcoming(live)`
re-asserts `features.builder._live_mask` inside `predict()` (defence-in-depth on
top of the builder): drops any row with a finishing position OR race_date < today
(Europe/Dublin). Past null-position rows are the join-miss signature that once
scored 87k phantom "live" runners ([[review-08-inference-live-fix]],
[[review-09-audit-2026-06-14]]). Tested at builder level
(`test_inference_matrix_empty_when_only_past_null_position_rows`) AND predictor
level (`TestUpcomingGuard`). NOTE: this is why `_live_df` test fixture is now
**tomorrow-dated** — a past-dated synthetic live frame is dropped by the guard.

**Coherence + normalization.** Win ⊆ place(top-2) ⊆ show(top-3), so probs are
clipped monotone in `_score`: `placed_2_prob = max(placed_2, won)`,
`showed_prob = max(showed, placed_2)` (independent binary models can violate this).
`won_prob` stays the calibrated headline (ECE≈0.005); `won_prob_normalized` is a
SEPARATE column rescaling each field to sum→1 (calibration.normalize_within_race).
Don't overwrite `won_prob` with the normalized value — see model-09 note that
marginal calibration and per-race sum-to-1 pull against each other.

**Data-quality layer (`_add_data_quality`).** Per runner: `data_completeness`
(fraction of signal features populated; denominator = FEATURE_COLS minus
EMPIRICALLY_DEAD_COLS so always-null paywalled cols don't drag everyone down);
`first_time_runner` (no career runs OR null historical_place_rate = no form
signal); `confidence` ∈ {high ≥0.8, medium, low <0.4} — first-time runner is
ALWAYS low. Lets the UI distinguish a form-backed read from a debutant's base-rate
guess (the same trap the value layer's `value_supported` gate guards —
[[review-04-features-models-bugs]]).

**Typed result object.** `@dataclass RunnerPrediction` / `RacePrediction` (with
`to_dict`/`from_dict`). `_runner_dict` now BUILDS a RunnerPrediction and returns
`.to_dict()` — the JSON cache shape is unchanged (UI consumes dicts via `.get()`),
the dataclass is the single source of truth. `Predictor.predict_typed()` returns
typed objects; `from_dict` tolerates old caches missing the newer keys.

Suite: 910 pass / 3 skip. tests/models/test_predictor.py classes added:
TestMissingFeatures, TestTargetCoherence, TestUpcomingGuard, TestTypedResults.
