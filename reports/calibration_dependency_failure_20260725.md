# Calibration audit dependency failure — 2026-07-25

## Conclusion

**NO-GO. The requested calibration/retraining work was not run.**

The prerequisite market-contamination/freshness repair is not yet complete on
the executable probability-to-EV path. Per the task's stop condition, no model,
preprocessor, calibrator, value gate, or training artifact was fit or changed
using the affected data.

One narrow directly blocking regression was repaired: the raw per-bookmaker
best-price overlay now rejects missing/invalid validation metadata, missing or
expired `fetched_at`, explicit stale rows, and the entire source race containing
any such row. Previously this overlay bypassed `features.fuse` freshness checks,
so a stale raw quote could outrank the fresh fused price and become the
executable EV price.

## Dependency verification

| Dependency | Result | Evidence |
|---|---:|---|
| Primary `WIN` market identity and specials quarantine | Partial pass | Scraper/normalizer/market-validation tests pass. Current live audit: BoyleSports 46 VALID / 9 INVALID races; LivescoreBet 51 / 0; Paddy Power 38 / 1. |
| Contaminated source race fails closed | Pass in validated/fused paths | `utils.market_validation`, `utils.normalizer`, `features.fuse`, and predictor guard tests pass. |
| Proxy-fed odds carry source freshness metadata | Pass at scraper/raw-row level | Scrapers emit `fetched_at`; BoyleSports degraded cache rows also emit `stale`. |
| Stale executable best-price overlay cannot produce EV | Fixed in this run | `models.predictor._odds_by_book_map` now repeats validity/freshness gates and rejects an entire bad source race. |
| Partial or freshness-unknown snapshot cannot produce any EV | **Fail** | `models.predictor._score_value` and `models.predict_unified.enrich` calculate EV before a shared race-level complete/fresh snapshot contract. `_build_race` uses the core EV/odds gate without the completeness check implemented later in `models.value.find_value_bets`. |
| Live cache contains the complete field | **Fail for current cache artifact** | `data/predictions.json` declares 514 runners but serializes only 170 selection/excluded rows. `reports/predictions_reconciliation.json` serializes all 514, but it is an offline reconstructed artifact, not the live cache. |
| Existing reconciliation proves freshness | **Fail** | `scripts/reconcile_prob_to_ev.py` deliberately rewrites every `fetched_at` to now and clears `stale`, so it can prove arithmetic identities but not live-source freshness. |

The earlier live-market/freshness repair prompt must be revisited to establish
one race-level eligibility decision, carried into the cache, that gates
`expected_value`, `ev_catboost`, `ev_lgbm`, `value_bet`, and suggestions together.
It must fail closed when the declared field, coherent reference book, validation
status, or freshness provenance is missing.

## Why `value_win_prob` reaches exactly zero

This is not a missing-model or numeric-underflow failure. The independent v3nf
CatBoost model is calibrated first, then `OddsBandCalibrator.predict(prob, odds)`
overwrites that result in `value_win_prob`. The odds-band isotonic mappings have
an exact zero floor for sufficiently small inputs; for example, independent
probabilities 0.005, 0.01, and 0.02 map to 0.0 at offered odds 2.0–6.0.

Observed artifacts:

- `data/predictions.json`: 3 exact zeros among 170 serialized field/display rows.
- `reports/predictions_reconciliation.json`: 2 exact zeros among 514 full-field rows.
- No exact zeros in `won_prob`, `won_prob_normalized`, CatBoost, LightGBM, or
  market probabilities in those same artifacts.

Therefore the current name and comments are materially misleading:
`value_win_prob` is **market-adjusted**, not price-free, because it is a function
of the same offered odds used in EV. The independent probability is not persisted
separately. A cross-fitted same-odds circularity ablation is required, but was not
run because the data dependency failed.

## Model-validity evidence inspected, but not accepted as a new verdict

- The static v3nf feature list has 40 columns and no direct intersection with
  odds, SP, finishing odds, market rank, result, or target columns.
- The existing LightGBM audit passed its post-off denylist, pre-off-price
  reconstruction, closing-move partial-correlation, and outcome-correlation
  checks.
- The saved LightGBM holdout says log loss 1.71698 vs market 1.72846, but model
  Brier 0.08032 is worse than market 0.07981, CLV is -12.23%, and no bootstrap
  confidence interval supports the point log-loss advantage.
- The requested price-free perturbation proof, expanding-window fully nested
  preprocessing/calibration, calibration-method comparison, cross-fitted
  favourite-longshot ablation, subgroup tables, bootstrap confidence intervals,
  drift analysis, and untouched-final-period verdict were not run.

The old cache-level `GO` stamps are not sufficient for the requested standard.
Until the prerequisite repair and full audit are completed, the operational
conclusion remains **NO-GO / paper-only**.

## Code changed

- `models/predictor.py`: fail-closed validity/freshness filtering for the raw
  best-price overlay, with whole-source-race rejection.
- `tests/models/test_predictor.py`: fresh valid fixtures plus regression tests
  for whole-race stale rejection and missing-freshness rejection.

No training data, model binary, calibrator, prediction cache, config gate, or UI
probability semantics were changed by this run.

## Reproducible commands and results

```powershell
# Baseline dependency suite, before editing
.\.venv\Scripts\python.exe -m pytest -q `
  tests/utils/test_market_validation.py tests/utils/test_source_health.py `
  tests/utils/storage/test_parquet_store.py tests/utils/test_normalizer.py `
  tests/utils/test_proxy_manager.py tests/features/test_fuse.py `
  tests/scraper/test_boylesports.py tests/scraper/test_livescorebet.py `
  tests/scraper/test_paddy_power.py tests/models/test_predictor.py `
  tests/models/test_value.py tests/test_train_lgbm.py
# 424 passed

.\.venv\Scripts\python.exe scripts\audit_live_markets.py `
  --output-csv reports\live_market_dependency_check_20260725.csv

# Focused regression suite after the narrow fix
.\.venv\Scripts\python.exe -m pytest -q `
  tests/models/test_predictor.py tests/models/test_value.py `
  tests/features/test_fuse.py tests/utils/test_market_validation.py `
  tests/utils/test_source_health.py tests/utils/storage/test_parquet_store.py `
  tests/utils/test_normalizer.py tests/utils/test_proxy_manager.py `
  tests/scraper/test_boylesports.py tests/scraper/test_livescorebet.py `
  tests/scraper/test_paddy_power.py
# 418 passed

.\.venv\Scripts\python.exe tools\audit_lgbm_features.py `
  data\features\training.parquet
# PASS: zero selected post-off columns; pre-off reconstruction exact

.\.venv\Scripts\python.exe -m pytest -q
# 1351 passed in 141.45s
```

## Artifact hashes (SHA-256)

| Artifact | SHA-256 |
|---|---|
| `data/features/training.parquet` | `1C20FA97BDC982F5CC3DAEF5F8881D88E300B16A3C0891219575C27B031B9246` |
| `models/catboost_won_v3nf.bin` | `9D45EF36CAE6FEB26DDD8C5C3130E3423B736B24E1960E46033359523BCB518B` |
| `models/catboost_won_v3nf_calib.pkl` | `27C992ADCDDCC22E531E25DDC37899ADDD07728C4F8688C7F3C4D04EF7A91FD3` |
| `models/fl_oddsband_v3nf_calib.pkl` | `548BB776C71C33FD30C2E7E872AD39B967CFE6F166E2856D225F580E61D9C2C2` |
| `models/lgbm_won_v3.txt` | `2CB869453916C6B55DFA80F5C48B7C435CA1147C9181C7E2F10ECFA0EE848948` |
| `reports/prob_to_ev_reconciliation.md` | `7DFD727C61EBE1D5750B7A501D9C619ADDDB76A22B310B94A11F7368DADB8475` |
| `models/predictor.py` after fix | `A3E0AB0939E336349BDDB4C291FF7C49A24192AA4A5D12BB80B432139AC00F34` |
| `tests/models/test_predictor.py` after fix | `DB83D0A04FE2BA8F0BC76E48BCA0289A606D8FE750E59740DE29361FC6280798` |

Training cutoff and untouched final-test window: **not established by this run**
because fitting/evaluation was stopped at the failed prerequisite. The most
recent saved LightGBM metadata states training cutoff 2026-05-22 and holdout
2026-05-23 through 2026-06-12; those dates belong to the earlier artifact and
are not presented as this audit's final test.
