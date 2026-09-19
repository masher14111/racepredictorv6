---
name: cleanup-08-tests
description: Test suite status (695 pass / 3 skip / 84% cov) + prioritized testing backlog
metadata:
  type: project
---

# Test Suite Status & Backlog (2026-06-16)

## Current status

- `pytest -q`: **695 passed, 3 skipped, ~68s** (was 687/3 before this session).
- 3 skips are **environment-only** (benign): `tests/utils/test_reporter.py` skips when
  `reportlab` isn't installed (optional PDF export). Not a bug, not stale. To clear:
  `pip install reportlab` (it lives in requirements-dev — see [[cleanup-04-deps]]).
- **No failures.** No real bugs, no stale tests found this pass.
- `pytest-cov` was not installed; installed it locally this session. Consider adding it to
  `requirements-dev.txt` (not yet done).

## Coverage (models/ + features/ + utils/) — TOTAL 84%

Well-covered: `models/evaluate.py` 100%, `models/tuner.py` 100%, `features/*` 94-100%,
`models/predictor.py` 77% but **richly tested** (TestValueLayer etc. — remaining misses are
`_load_value_model` + CLI `_main`).

Biggest real gaps (silent-bug risk):

- `models/retrain_trigger.py` — was 62%, **now 89%** after this session (added
  `check_and_retrain` orchestration suite). Remaining misses: KS too-few-samples branch (115),
  position filter (270), old-meta read (323-327), CLI `main` (346-362).
- `models/train.py` 74% — misses are mostly `main()`/CLI (306-358) + a few guard branches.
- `utils/reporter.py` 57% — reportlab-gated PDF paths (skipped).
- `utils/normalizer.py` 80%, `utils/logger.py` 78%, `utils/timezone.py` 71%.

## Implemented this session (top 2)

1. **`tests/models/test_retrain_trigger.py::check_and_retrain` suite (7 tests)** — the entire
   retrain decision/orchestration was untested (biggest models/ gap). Covers: force+check_only
   skips train; no-reference guard; no-current-features guard; no-drift→no-retrain+report written;
   drift→train called; force→train; train()==None→False. Mocks `models.train.train` (lazy import)
   and patches `_load_cfg` + `_DEFAULT_FEATURES`.
2. **`tests/features/test_engine.py::test_going_preference_null_when_going_speed_column_absent`** —
   covers the missing-`going_speed`-column guard in `add_going_preference` (engine.py 119-121):
   returns null rates instead of raising KeyError.

## Remaining backlog (prioritized, NOT yet implemented)

3. **`models/train.py` happy-path with mocked CatBoost** — assert `train()` returns meta with
   per-target test*auc, writes `catboost*\*\_meta.json`, calls `save_reference_snapshot`. The
   sample-weight + price-free feature-drop logic here is silent-bug prone (see [[review-04-features-models-bugs]]).
4. **`models/predictor.py::_load_value_model` graceful degradation** — missing .bin disables value
   layer (no crash); meta.json feature_cols read; calibrator absent → raw passthrough warning.
   Lines 198-219 untested.
5. **`models/retrain_trigger` performance-decay integration** — drift→retrain path where new AUC
   < old AUC by >threshold logs the decay warning (wire `old_meta`/`new_meta` through real files).
6. **`utils/normalizer.py` edge cases (80%)** — venue/horse-name normalization corners (lines
   232-260, 375-391); normalization mismatches silently break cross-source joins (the recurring
   class of bug in [[review-09-audit-2026-06-14]]).
7. **`models/train.py` empty-matrix → returns None** — the guard `retrain_trigger` relies on
   (test #1 above mocks it; an end-to-end check that train() actually returns None on empty input).
8. **`utils/timezone.py` (71%)** — DST/BST boundary behavior of `now()`/conversion helpers; a
   DST join bug already bit once (see [[review-07-sl-extractor-fix]]).

## Notes

- No lint/coverage config added (consistent with [[cleanup-03-dead-code]] decision to keep config minimal).
