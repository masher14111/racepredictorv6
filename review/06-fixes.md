# 06 — Fixes Applied

Single session fixing all **critical** bugs collected across reviews 01–05.
Priority order followed: (1) startup crashes → (2) Telegram → (3) prediction
pipeline → (4) UI pages → (5) test failures. No startup ImportError and no
Telegram-breaking bug was found live, so work concentrated on the pipeline, the
scraper rate-limiting layer, and the UI pages.

## Fixes applied

### Prediction pipeline (features → models → predictor)

1. **`models/train.py` — sample-weight / row misalignment (HIGH).**
   Weights were computed on the pre-split frame as a separate array, then
   sliced positionally after `_time_split` re-sorts rows by `race_date`. Each
   row was being paired with _another_ row's weight, silently corrupting every
   trained model. Fix: attach weights as a `df["_sample_weight"]` column before
   the split so they travel with their rows, and pull the train-fold weights via
   `train_df.loc[tr_mask, "_sample_weight"]`. `_sample_weight` is not in
   `FEATURE_COLS`, so it is never fed to the model as a feature.

2. **`features/derive.py` — trailing-rate denominator (HIGH).**
   `add_trailing_rates` counted NA-position priors (every historical row before
   results are scraped) in the denominator, deflating every win/place rate
   toward 0. Added `prior = prior[prior["position"].notna()]` so only
   known-result runs count — matching `engine._trailing_rate`.

3. **`models/predictor.py` — jockey/trainer blank in output (MED).**
   `_runner_dict` read `row.get("jockey")` / `row.get("trainer")`, but the
   feature frame carries `jockey_name` / `trainer_name`. Output keys unchanged;
   only the source columns were corrected, so the UI now shows names.

4. **`models/retrain_trigger.py` — over-eager retrain trigger + wrong baseline
   (MED).** Two changes: (a) KS removed from the _trigger_ (still computed and
   reported as a diagnostic) — its two-sample p-value collapses toward 0 as
   sample size grows, so an OR clause would fire on nearly every check at scale
   regardless of true drift; PSI is now the sole trigger. (b)
   `check_and_retrain` now restricts `features.parquet` to labelled rows
   (`position.notna()`) before comparing against the reference snapshot (the
   labelled training split), so live/future unlabelled rows no longer skew the
   drift comparison.

### Scrapers / rate-limiting layer

5. **`scraper/livescorebet.py` — central rate limiter bypassed (HIGH).**
   `scrape()` injected a local `Throttler`, so production traffic skipped the
   per-domain rps configured in `config.yaml`. Removed the production throttler
   injection; the client now falls through to `get_rate_limiter()`. Tests still
   inject their own fast `Throttler` directly.

6. **`scraper/betsp/betfair_sp.py` — unthrottled CSV fetch (HIGH).**
   `fetch_csv` issued a raw `httpx.Client().get()` with no rate limiting. Wrapped
   the GET in `get_rate_limiter().acquire_for_url(url)`.

7. **`scraper/betsp_historical.py` — unthrottled, un-proxied results fetch
   (HIGH).** `_results_get_html` (the live fetcher for full-year backfills) had
   no rate limiter and no proxy rotation. Rewrote it to pull a proxy from
   `get_proxy_manager()`, gate the GET through `acquire_for_url`, and report
   success/failure to the rotator. Per-URL error handling is unchanged (callers'
   `fetch_raw()` still catch `httpx.HTTPError`).

8. **`scraper/paddy_power.py` — unthrottled client GET (MED).**
   `PaddyPowerClient.get` issued raw GETs inside its retry loop. Wrapped the GET
   in `get_rate_limiter().acquire_for_url(url)`. **NOTE:** the endpoint URL /
   `cardsToFetch` query was intentionally **left unchanged** (requires DevTools
   confirmation per instructions).

9. **`scraper/boylesports.py` — cache path (MED).**
   `_CACHE_PATH` pointed at `data/boylesports_cache.json`; standardised to
   `data/cache/boylesports.json` (the `data/cache/` convention used by the other
   scrapers, e.g. Paddy Power). Tests override the path via monkeypatch.

### UI pages

10. **`ui/predictions.py` — features path + importance columns (HIGH).**
    `_FEATURES_PATH` was `data/features/features.parquet` but `builder.py` writes
    `data/features.parquet`, so prediction breakdowns silently never loaded.
    Corrected the path. Also fixed `_load_importance` to label importances with
    the trained model's `feature_cols` from `meta` (falling back to `FEATURE_COLS`
    only when lengths match), so the importance chart aligns with the model.

11. **`ui/race_compare.py` — features path + error text (MED).**
    Same `data/features/features.parquet` → `data/features.parquet` path fix, and
    the user-facing "file not found" message updated to the correct path.

12. **`ui/settings.py` — crash on missing config + invalid page links (HIGH /
    MED).** (a) `_load_cfg` now wraps the YAML open in
    `try/except (FileNotFoundError, OSError): return {}`, so the Settings page no
    longer crashes when `config.yaml` is absent. (b) The six `st.page_link(...)`
    sidebar calls (which raise outside a `pages/` multipage app — these are
    standalone scripts) were replaced with an informational `st.caption` listing
    the `streamlit run ui/<page>.py` launch commands.

### Utils

13. **`utils/bet_tracker.py` — TOCTOU over-crediting (HIGH).**
    `bankroll_before = self.bankroll` was read _outside_ the write lock, so two
    concurrent bets could both deduct from the same pre-bet balance. Moved the
    read inside `with self._pool.write_lock():`. The `bankroll` property is a
    pure SELECT, so reading it under the write lock is safe (no nested-lock
    deadlock).

14. **`utils/config_loader.py` — `reporter` flagged as unknown key (MED).**
    `config.yaml` has a `reporter:` block but `reporter` was missing from
    `_KNOWN_TOP_LEVEL_KEYS`, producing a spurious "unknown config key" warning.
    Added it.

## Pytest result after fixes

Command: `python -m pytest tests/ -v --tb=short`

```
604 passed, 3 skipped in 38.74s
```

- **Passed:** 604
- **Skipped:** 3 (all `reporter` PDF tests — `reportlab` not installed; expected)
- **Failed:** 0

Identical to the pre-fix baseline in `01-test-health.md` (607 collected / 604
passed / 3 skipped / 0 failed). No regressions introduced. Each fix was checked
against its corresponding test module before the run.

## Unfixed items

These were deliberately skipped — either out of scope per instructions, or not
live bugs on inspection.

- **Paddy Power endpoint URL / `cardsToFetch` query** — per instructions, the
  API endpoint needs DevTools confirmation before changing. Only the _rate
  limiter_ was added to that scraper, not the URL.
- **Timeform `session_cookie`** — per instructions, requires a real user account;
  not actionable in this session.
- **Review 03 #6 — Timeform writer "KeyError" on missing column** — _not a live
  bug._ `utils/storage/parquet_store.write_parquet` (lines 32–36) defensively
  fills any missing expected columns with `pd.NA` before writing, so the
  described KeyError cannot occur. No change needed.
- **Review 03 #5 — `betsp_historical` unhandled `raise_for_status`** — _not a
  live bug._ Each results source's `fetch_raw()` already wraps the `get_html`
  call and catches exceptions per-URL (e.g.
  `scraper/betsp/results/sporting_life.py` returns `[]` on error), so a single
  failed page does not crash a backfill.
- **Review 05 #6 — UI auto-refresh scheduler wiring** — a _new feature_ with
  background-thread side-effects, not a bug fix. Out of scope for a
  minimal-change session.
- **All items marked "minor" / "nice to have" / LOW / INFO** across reviews
  02–05 — explicitly out of scope.

## Next

Prompt 7: wire and smoke-test the live scrapers end-to-end (confirm the Paddy
Power endpoint URL via DevTools, then run a real `scrape()` + feature-build +
predict pass against fresh data to validate the now-corrected pipeline).
