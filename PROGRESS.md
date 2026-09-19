# Race Predictor v3 — Working Memory / Progress Log

> **This file is the single source of truth for the repair effort.** It survives `/clear`.
> Every agent run must (1) read this file first, (2) do exactly one task, (3) tick the task
> and add a short note below, (4) append a line to `CHANGELOG.md`. Do not re-audit the repo —
> the findings are already recorded here.

Project: `C:\Users\mshr\Desktop\Race Predictor v3` · Python 3.14 · Windows · PowerShell + Bash
Timezone: Europe/Dublin · Launch UI: `streamlit run ui/app.py` · Tests: `python -m pytest -q`

---

## VERIFIED STATE (audit 2026-06-14) — read before trusting older notes

The `memory/` notes saying "training matrix empty" are **stale**. Ground truth from disk today:

| Fact                                                  | Value                                                       | Implication                                                                                                               |
| ----------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `training.parquet`                                    | **248,173 rows**, all labelled, dates 2024-01-01→2026-06-12 | model trained fine                                                                                                        |
| `models/catboost_{won,placed_2,showed}_v3.bin` + meta | present (2026-06-13)                                        | predictor can load                                                                                                        |
| `data/predictions.json`                               | present, **3 races, dated 2026-06-13**                      | stale; see GUI bug                                                                                                        |
| `unified_races.parquet`                               | **582,529 rows**, 495,228 with `position`                   | history OK                                                                                                                |
| **`trainer_name` populated**                          | **0 / 582,529**                                             | trainer blank everywhere; `trainer_win_rate`, `jt_combo_*` features are DEAD                                              |
| **`jockey_name` populated**                           | **28 / 582,529**                                            | jockey only on today's scraped runners                                                                                    |
| **`odds_decimal` populated**                          | **2 / 582,529**                                             | live/market odds never attach → `decimal_odds=null`, `implied_prob=NaN`, no EW flags, 5 market features dead at inference |
| **Upcoming races for today**                          | **0** (newest data 2026-06-13)                              | `build_inference_matrix()` returns empty today → predictor returns `[]`                                                   |
| `data/features.parquet`                               | **stale 5-row file** (training has 248k)                    | UI feature-breakdown join finds nothing                                                                                   |

### Root causes of the two reported symptoms

1. **"GUI doesn't load predictions":** (a) no upcoming races in `unified` today → predictor produces nothing on refresh; (b) the stale 06-13 cache is hidden because `ui/app.py` defaults the date filter to **"Today"** (06-14). Net = blank screen.
2. **"Builder is broken":** the builder _code_ (`features/builder.py`) is sound. The breakage is **data** (no trainer, ~no jockey, ~no odds) plus the **stale `data/features.parquet`** artifact. Also `models/predictor.py` does **not filter non-runners** ("Non Runner" was ranked #2 at Sandown in the live cache), and many composite scores tie because most features are NaN at inference.

### Already done (2026-06-14, before this loop)

- **LightGBM removed** (was already CatBoost-only in code): dropped from `requirements.txt`, removed dead `model_weights` block from `config.yaml`, removed from `_KNOWN_TOP_LEVEL_KEYS` in `utils/config_loader.py`. Config verified loading.

### Decisions still needed from the human (do NOT guess — ask in the relevant task)

- **D1 — Odds leakage:** `models/features.py FEATURE_COLS` feeds market price (`implied_prob`, `log_odds`, `market_rank`, `overround_norm_prob`, `odds_drift`, `odds_value_delta`, `ew_value_index`) INTO the model. For _value betting_ (model must disagree with market) the probability model should be **price-free**. For _pure winner prediction_ keeping odds is fine. Task 12 handles this; confirm the goal first.
- **D2 — Local LLM:** whether to add an optional Ollama text→feature extractor (Task 21). Hardware (RTX 5070 Ti, 32GB) supports it easily, but only worth it once a text source exists. CatBoost stays the probability engine regardless.

---

## TASK CHECKLIST (one task per `/clear` cycle; see PROMPTS.md)

- [x] 01 — Bootstrap: confirm PROGRESS.md/CHANGELOG.md, run full test suite, record baseline
- [x] 02 — Data-health script → `DATA_HEALTH.md`
- [x] 03 — GUI empty-state + stale-cache + date-default fix (`ui/app.py`)
- [x] 04 — Filter non-runners in `models/predictor.py`
- [x] 05 — Regenerate `data/features.parquet` + fix feature-breakdown join (`ui/predictions.py`)
- [x] 06 — Fix `trainer_name` (0% populated) end-to-end
- [x] 07 — Fix `jockey_name` (28 rows) end-to-end
- [x] 08 — Fix live odds pipeline (`odds_decimal` 2/582k)
- [x] 09 — Fix "no upcoming races" / live racecard ingestion (+ optional scheduler)
- [x] 10 — End-to-end live smoke: scrape→normalize→build→predict, assert >0 live races with odds
- [x] 11 — Feature/schema audit: `FEATURE_COLS` vs derive/engine; handle dead features
- [ ] 12 — Odds-leakage decision + price-free model variant (gated on D1)
- [x] 13 — Calibration: A/E by odds band + reliability curves in `models/evaluate.py`
- [x] 14 — Backtester honesty review: settle at SP/BSP, add CLV, per-band CI, longshot gate
- [x] 15 — Live scrapers review: paddy_power / boylesports / livescorebet
- [x] 16 — Historical scrapers review: betsp_historical / timeform_historical
- [x] 17 — Normalizer + storage review
- [x] 18 — Telegram notifications review (+ guarded live send test)
- [x] 19 — Bet tracker review (stake math, EW settlement, stop-loss)
- [x] 20 — Reporter + retrain-trigger review
- [x] 21 — Local LLM text-feature scaffold (gated on D2)
- [x] 22 — Secrets/config hygiene (.env, .gitignore, dead config)
- [x] 23 — Test gaps: add tests covering every bug fixed in tasks 03–10
- [x] 24 — Full regression pass over the whole pipeline
- [x] 25 — Generate `FINALSETUP.md`
- [x] 26 — Stop pytest spamming real Telegram (conftest + Notifier self-disable)
- [x] 27 — Finish Cloudflare-bookie scraper speedup (paddy retry robustness + boyle worker/rps tune)
- [x] 28 — Per-bookmaker odds end-to-end (odds_by_book/best_odds/best_book; value uses best price; UI per-book row)

---

## RUN LOG (newest first — append a short entry per completed task)

- 2026-06-17 — **calib-fl-05: landed the calibration programme + honest verdict.**
  Final prompt of the headline-calibration + favourite-longshot (F-L) recalibration
  programme (`memory/calib-fl-00..05`). **(1) Config:** applied the calib-fl-04
  recommended value gate set to `config.yaml` — `value.max_odds` **6.0 → 4.0**
  (CLV-first; the F-L recalibration flattened A/E so the cap is now closing-line-value
  driven, not A/E-collapse driven), rewrote the `max_odds` + edge-gate comments (the
  edge gates are no longer "harmful" post-recalibration, just CLV-neutral → kept off).
  Aligned the now-stale code-default fallbacks that mirror config: `ValueConfig.max_odds`
  6.0→4.0 + module docstring (`models/value.py`), `predictor._load_cfg` 26.0→4.0,
  `backtest/__main__` 26.0→4.0. `config.local.yaml` untouched. **(2) Predictions:**
  `python -m scripts.refresh --no-scrape` → 111 races / 1223 runners; F-L recalibrator
  loaded; headline `won_prob_normalized` mean 0.093, max 0.375, **zero >0.70 saturation**,
  full-field within-race sum = 1.000 across all 122 races (raw `won_prob` debug col still
  saturates at 0.734 as designed). **(3) Backtest** (`scripts.last_week_backtest`,
  2026-06-06..12): headline **ECE 0.222 → 0.029, AUC 0.773 → 0.778** (Bug-1 fix holds);
  value strategy still loses **at SP** (EW −60%, win −72%, 18 bets) and top-pick ~unchanged
  (EW −25.5%, win −32.7%) — SP settlement is conservative; the real test is CLV (calib-fl-04:
  A/E flattened 2.80..0.17 → 0.94..0.78, yield +5.3%→+7.4% at the price taken, but CLV still
  negative, best ~−4% on band [2,4]). **(4) Docs:** README known-limits rewritten (headline =
  normalised prob; value calibrated but provisional/paper until live CLV measured); this
  log. Tests updated for the new band ([2,4]) and stale defaults; **`pytest -q` green**.
  **Bottom line: ranks well (AUC 0.78), headline now calibrated, F-L bias fixed — but NOT
  yet bettable (CLV-negative at SP/ppwap); paper-only.** Memory: `calib-fl-05-final.md`.

- 2026-06-17 — **Yesterday's Bet Predictor** + proxy/docs follow-ups. **(1)** New
  `models/yesterday.py` engine: takes one _settled_ racing day, scores every runner blind to
  the result with the live models + price-free value model, places bets (live value picks, or
  top win pick/race), settles win/each-way at the starting price, reports profit/ROI. Loads
  the persisted `data/features/training.parquet` (instant) and **defaults to the most recent
  day with results** — finishing positions lag the calendar (latest settled = 2026-06-12 vs
  "today" 06-17), so literal "yesterday" has no results yet; surfaced honestly. **(2)** New
  Streamlit page `ui/pages/13_Yesterdays_Bet_Predictor.py` (strategy/bet-type/stake/date
  controls, KPI rail, settled-bets table, honesty note). **(3)** 13 unit tests
  (`tests/models/test_yesterday.py`, all green). **(4)** README: added the new page + the
  Today's Suggestions page to the UI table, corrected the stale "v3 uncalibrated" limit (v3 is
  now served calibrated per `model-11`; the `odds_finish` market-feature leak remains), added a
  results-lag/SP-settlement limit. Retrain NOT needed — models retrained 2026-06-16 (v3
  calibrated, v3nf price-free); retrain only on drift via `python -m models.retrain_trigger`.
  Memory: `wrap-29-yesterday-predictor.md`.

- 2026-06-17 — Docs refresh. Wrote a new top-level **README.md** (what the project is, the
  scrape→normalize→features→models→predict→UI pipeline diagram, setup + secrets, how to run
  each stage/the UI, how paper betting works, how to run tests, status & known limits). Kept
  claims accurate to current state: v3 served raw/uncalibrated + trained on `odds_finish`
  market features → use v3nf for trustworthy probs (per `memory/model-09-baseline-audit.md`);
  live feeds carry no jockey/trainer; boyle/paddy Cloudflare-gated, livescorebet reliable.
  Test baseline stated as 695 pass / 3 skip (reportlab) per `memory/cleanup-08-tests.md`.
  `FINALSETUP.md` left as the deeper ops guide; no code changed. Memory: `wrap-28-docs.md`.

- 2026-06-15 — Scraper + UI session. **(1)** LiveScore Bet → curl_cffi Chrome
  impersonation (was the only live scraper on plain httpx → the only one hitting a
  self-signed-cert TLS error); added `_curl_get()` + proxy-rotating retry. Verified
  live: 757 rows/38 races, zero SSL warnings; tests/scraper 141 passed. **(2)** Wired
  the 6 orphaned `ui/*.py` pages into the multipage app (Live Races, Place Bet,
  Performance, Compare, Settings via runpy wrappers; skipped perf/predictions clones),
  - new **Scrape & Refresh** page with per-book scrape buttons (`ui/refresh_ops.py`).
    Fixed perf_dashboard plotly dup-key + bad icon. All 7 pages render (AppTest); tests/ui
    12 passed. **(3)** Dark/light text fix round 2: `.streamlit/config.toml base="light"`
  - `ui/_theme.py::pin_sidebar_nav()`; Playwright contrast-scan clean for the reported
    bugs (sidebar nav + native titles). **(4) FIXED the venue-day collapse:** predictor
    `_RACE_KEY` was `["venue","race_date"]` → all races at a track collapsed into one;
    now groups by `(venue, race_time)` (race_date fallback for missing post-times).
    **12 → 106 races**, field sizes/ranks self-correct (`_build_race` uses `len(grp)`),
    **value bets 0 → 14**. `ui/app.py` now groups race cards under collapsible per-venue
    sections (open a venue, pick a race). Regression test added; test_predictor 81 passed;
    Playwright screenshot of the Kilbeggan 8-race drill-down. Remaining: derive/engine
    feature-grain `_RACE_KEY` still venue-day (separate train/serve concern, non-blocking)
    - paddy_power scrape returns only ~2 rows (Cloudflare) + foreign tracks (JP/AU/CL) in
      feed → consider a UK/IE venue filter.

- 2026-06-15 — Debug (Playwright): two symptoms reported — horse/jockey invisible +
  all EVs negative. **(1) Horse/odds invisible = dark-mode CSS bug, FIXED.** `.rp-card`
  forces a white surface but table text inherited Streamlit's theme colour (near-white
  under OS dark mode) → white-on-white. Pinned `color: var(--fg)` on `.rp-card`.
  Reproduced + verified with headless Playwright `color_scheme="dark"` (strong
  rgb(250,250,250)→rgb(23,32,51)); tests/ui 12 passed. **(2) Jockey "—" = no source
  data** (live odds feeds emit only horse+price; jockey/trainer blank for all 72 live
  runners) — not a rendering bug. **(3) All EVs negative ≠ bad training.** Measured
  `PRICE_FREE_FEATURE_COLS` fill on today's 1072 live runners: going_speed/ratings/
  jockey_win_rate/trainer_win_rate/jt_combo/going_pref/recent_form/distance all **0%**;
  historical_win_rate/horse_speed ~36%; only field_size/race_complexity/odds populated.
  Starved of features the value model returns ~base rate (value_win_prob 0.05–0.11,
  mean 0.048) for everyone → EV = vwp×best_odds−1 ≈ −0.85 for all 69 priced runners.
  The price-AWARE win% (87–91%) only looks confident because it echoes implied_prob.
  **Root cause = live racecards aren't joined to each horse's history/ratings + no
  jockey/trainer on live rows → this is exactly Task 29 (the value-feature fix).**

- 2026-06-15 — Task 28: per-bookmaker odds end-to-end (see each book's price, back
  the best). **predictor.py** — `_odds_by_book_map()` builds `{source: price}` per
  runner from the **un-fused** unified rows (keyed `(race_date, venue, horse_id)` —
  same grain as `fuse_sources`, captured _before_ fusion collapses cross-source prices
  to one), restricted to the live bookmakers (`livescorebet/paddy_power/boylesports`;
  betsp/timeform excluded — they're SP/ratings, not a price you can take). `predict()`
  loads raw unified once, builds the map, and `_attach_odds_by_book()` adds
  `odds_by_book`/`best_book`/`best_odds` columns before scoring; `_runner_dict` emits
  all three (legacy `decimal_odds` kept, default-safe when a runner has one/zero books).
  **Value uses BEST price:** new `_effective_decimal` (best_odds when known else fused
  odds) drives both `expected_value` (`_score_value`) and the `value_bet` odds-band gate
  (`_build_race`). **normalizer.py** — `_load_default` now unwraps the `utils.cache.Cache`
  envelope for `paddy_power.json` (adapter expected top-level `races`); paddy odds reach
  unified for the first time (0 → 580 rows / 430 priced). **ui/app.py** — Odds column
  shows the best price as headline + a compact per-book sub-row (`_books_row`), best
  marked with ★ + bold green (not colour-only; WCAG AA per PRODUCT.md). All UI bits
  conditional → old caches render unchanged. **Verified:** `python -m scripts.refresh`
  (real scrape; boyle partial under Cloudflare, lsb+paddy OK) → 16 races / 865 runners;
  `predictions.json` has 43 runners with ≥2 books, all 3 books present (paddy 56 / lsb 37
  / boyle 37). Sample — Flying Fletcher@Wetherby: LSB 2.88 / Boyle 4.33 / **Paddy 4.50**
  → `best_odds=4.50` (fused legacy was 2.88, +56% better price). Added `TestOddsByBook`
  (11 tests). `pytest tests/models tests/utils/test_normalizer.py tests/ui` → **182 passed**.

- 2026-06-15 — Task 27: finished the Cloudflare-bookie scraper speedup. **Paddy** —
  added a bounded retry to `_curl_cffi_fetch` (loop `attempts=_CURL_ATTEMPTS`, default 3
  via new `paddypower.curl_attempts` config): a 403 / transient / non-JSON now rotates
  to a fresh proxy IP and retries instead of immediately falling to the slow browser
  tiers. Bumped `apisms.paddypower.com` rps 2→3 for retry headroom. **Boyle** — empirically
  tuned over 3 consecutive runs each: baseline (workers 4 / rps 3) fell to the Selenium
  tier (173 partial rows, 124–351s); **8 workers / rps 10 / concurrent 10** clears CF
  cleanly every run (~572–757 rows, 6–49s); 16/20 regressed (collapsed to 71 rows on 2/3
  runs — adaptive 403s). Set `boylesports.scrape_workers: 8` + `www.boylesports.com`
  rps 10 / concurrent 10 (proxy gives a fresh IP/request, so per-IP 403s, like the SL
  rps-30 precedent). Added `paddypower` to config_loader known keys. **Verify (final, real
  config):** paddy 50 rows in 0.9/0.6/0.9s (< 5s ✓, 3/3); boyle 630/757/757 rows in
  49/6/6s (< 60s ✓, 3/3). `pytest tests/scraper -q` → **141 passed**.

- 2026-06-15 — Task 26: pytest no longer spams real Telegram. Belt-and-braces:
  (1) new `tests/conftest.py` sets `RP_DISABLE_NOTIFICATIONS` at collection time +
  autouse session fixture installs a channel-less singleton; (2) `utils/notifications.Notifier`
  self-disables (`_enabled=False`) when `PYTEST_CURRENT_TEST` or `RP_DISABLE_NOTIFICATIONS`
  is set — production unchanged. Adjusted `test_notifications` `_make()` to lift the flag for
  its respx-mocked channel-construction tests. Verified: full suite 674 passed; respx smoke
  on the bet-settle path shows `_active=False` and 0 POSTs to api.telegram.org.

- 2026-06-14 — Task 25: wrote `FINALSETUP.md` — clean end-to-end guide reflecting the
  post-01–24 state. Covers prerequisites (Py 3.14 / `pip install -r requirements.txt` /
  `playwright install`), one-time setup (`utils.storage migrate`, `pipeline.py`), secrets
  (config.local.yaml + `RP_` env precedence, Task 22), the daily flow (`scripts.refresh`
  → `streamlit run ui/app.py`), a full CLI table (refresh/pipeline/data_health/storage/
  train/predictor/retrain_trigger/backtester/reporter/backfill/compare_pricefree), all 8
  Streamlit pages, retrain, backtest, Telegram enable, and a Troubleshooting section
  mapping each symptom (blank GUI→Task03, blank trainer/jockey→06/07+backfill, null
  odds→08, non-runner→04, blank breakdown→05) to fix + verify cmd. **Verified every
  documented command:** `--help` on all 7 module entrypoints, `utils.storage version`
  (current=2 target=2), `data_health.py`, and the two predictions.json verify one-liners
  (cache: 8 races, 24/24 priced) all run. No code changed.

- 2026-06-14 — Task 24: full pipeline E2E on real data. **The Tasks 06/07 backfill is
  DONE.** Root cause confirmed: on-disk `betsp.parquet` predated the name fix (carried
  only `*_id`). Network re-scrape of 3yr is infeasible (Cloudflare), so wrote a reusable
  offline replayer `scripts/backfill_betsp_from_raw.py` that re-parses the 29,746 STORED
  raw Sporting Life pages (`data/historical/raw/...`, 908 days) through the current parser
  → 308,683 ResultRows → re-joins onto the SP backbone checkpoint → rewrites `betsp.parquet`
  (trainer_name/jockey_name now **527,433/584,471 = 90.2%**, position 496,762). Then
  re-normalized → `build_training_matrix` (292,306 full-matrix rows incl. live; 248,173
  labelled) → predicted with existing v3 models (8 races/350 runners → `predictions.json`,
  generated_at today, selections carry decimal_odds+implied_prob+won/placed_2/showed probs).
  **data_health.py vs audit baseline:** trainer_name **0% → 90.22%**, jockey_name
  **0.00%(28) → 90.22%**, odds_decimal **0.06% → 0.06%** (UNCHANGED, by design — it is a
  live-only board price; historical price is `odds_finish` = 99.9% in unified, which feeds
  `implied_prob` = 100% in training). **Dead trailing features REVIVED in the matrix:**
  trainer_win_rate 201,595 nonzero, jockey_win_rate 203,711, jt_combo_win_rate 144,605
  (all were 0% — see Task 11). features.parquet now 292,306 rows (was stale 5) so the UI
  breakdown join works. **Retrain deliberately SKIPPED** (task says optional): deployed v3
  models were trained when those features were dead, so a tuned retrain would now let the
  model actually use trainer/jockey signal — but that is gated on **D1** (price-free variant)
  and a full Optuna run, so left as the recommended next step rather than overwriting prod
  models with an untuned fit. Full suite **658 passed, 3 skipped** (reportlab PDF; unchanged
  from Task 23). **Still open:** D1/D2-gated Task 12; LSB global-card scoping (Brazilian/US
  cards still scored as "live" — Task 15 carry); live jockey/trainer empty (LSB source has
  no jockey/trainer — Task 17); a tuned retrain to exploit the revived features.

- 2026-06-14 — Task 23: test gaps for tasks 03–10. Extracted the GUI date-default +
  empty-state logic out of `ui/app.py` into a pure, Streamlit-free `ui/logic.py`
  (`parse_race_date`, `default_date_index`, `empty_state`) and refactored `app.py` to
  import it (behaviour identical, now headless-testable). Added 17 focused tests:
  `tests/ui/test_logic.py` (13 — stale-cache→"All upcoming" default, 4-way empty-state),
  +2 builder (live rows returned when upcoming exist / empty when only past null-position
  rows — the Task 08 phantom-live guard), +1 derive (odds_decimal→implied_prob), +1
  normalizer (`_validate` preserves trainer_name/jockey_name). Non-runner filter + the
  remaining odds/name paths were already covered (predictor/derive/normalizer suites).
  Verified: full suite **658 passed, 3 skipped** (Task 01 baseline 606/3; +17 from this
  task, rest cumulative from tasks 02–22). `ui/app.py`+`ui/logic.py` py_compile clean.

- 2026-06-14 — Task 22: secrets/config hygiene. Moved live Telegram `bot_token`/`chat_id`,
  the DataImpulse proxy URL (creds), and the Timeform `session_cookie` out of `config.yaml`
  (now blanked/`[]` with pointer comments) into a new git-ignored `config.local.yaml`.
  `utils/config_loader.py` now deep-merges `config.local.yaml` over `config.yaml` **before**
  env overrides (`_apply_local_overlay`/`_deep_merge`), so precedence = env > local > committed.
  Added `.gitignore` (config.local.yaml, .env, _.local.yaml, **pycache**, .pytest_cache,
  data/_.parquet + data/\*_/_.parquet, data/races.db[-wal/-shm], data/cache/, logs/, \*.log).
  Verified: `git check-ignore config.local.yaml` ✓, config validates with secrets resolved via
  overlay (no values printed), RP\_ env override still wins, `test_config_loader.py` 28 passed.

- 2026-06-14 — Task 21: **D2 opted in.** Scaffolded optional local-LLM text→feature
  extractor `llm/text_features.py` (+ `llm/__init__.py`): free-text race comments →
  6 boolean flags (`ground_excuse`, `distance_excuse`, `trip_trouble`,
  `headgear_first_time`, `returning_from_layoff`, `course_winner`). `OllamaBackend`
  (`/api/generate` `format=json`, temp 0) with deterministic `RegexBackend` fallback +
  disk cache (`data/cache/llm`, keyed text+backend+model+schema_ver). **Off the hot
  path**: emits no prob/EV, NOT in `FEATURE_COLS` (verified overlap=∅). No-ops to `{}`
  when `llm.enabled=false` (default) or no text. Added disabled `llm:` block to
  config.yaml + `llm` to config_loader `_KNOWN_TOP_LEVEL_KEYS`. Verified: new
  `tests/test_text_features.py` **11 passed** (stub backend, no live call — incl.
  disabled-no-op, cache hit, None→regex fallback, Ollama-down→None); config still
  validates; config_loader tests **28 passed**.

- 2026-06-14 — Task 20: reporter + retrain-trigger review. **No code bugs — both confirmed
  sound.** `tests/models/test_retrain_trigger.py` **29 passed**. Generated a full bundle via
  `python -m utils.reporter --format all` → `reports/` now holds `bets_*.csv` (0 rows),
  `predictions_*.csv` (21 rows), `snapshot_*.json`, `manifest.json` — all present. **Manifest
  hashing correct**: SHA-256 + row_count + size_bytes per file, atomic `.tmp`→`replace` writes,
  integrity checks (bets/predictions/referential) run. **PDF gracefully skipped** (warning, no
  crash). **Retrain drift logic reads the reference snapshot correctly**: `--check-only` loaded
  `models/reference_features.parquet` (198,538 rows), computed per-feature PSI (primary trigger)
  - KS (diagnostic only — KS-OR intentionally not wired, p→0 at scale), wrote
    `models/drift_report.json`, fired triggered=True; AUC-decay compares old/new meta `test_auc`
    per target. **Missing dependency: `reportlab`** — declared in `requirements.txt:25` but NOT
    installed in the current env, so PDF export is skipped and its 3 tests remain skipped;
    `pip install reportlab` enables it (matplotlib 3.10.8 already present for the charts). **Data
    note (not a code bug):** `data/features.parquet` is back to the stale **5-row/3-labelled**
    inference file (Task 05's 291k matrix was overwritten), so the drift PSI is meaningless garbage
    (198k vs 3 rows) until a full labelled matrix is regenerated on disk — revisit in Task 24. No
    code changed.

- 2026-06-14 — Task 19: bet tracker review (`utils/bet_tracker.py`). **No math errors —
  confirmed correct.** Kelly `f=(b·p−q)/b · bankroll` (full), fractional = `full·kelly_fraction`,
  flat = `flat_stake`; all guarded (b≤0/p≤0/p≥1 → 0). **EW settlement verified by hand**:
  half-stake per leg, win leg `half·odds`, place leg `half·((odds−1)·ew_fraction+1)`. Worked
  £5 e/w @ 10.0 (9/1), 1/5: win→£64 (50+14) profit £54, place→£14 profit £4, lose→−£10 —
  **matches the code and the textbook UK/IE each-way math exactly.** Stop-loss = strict
  `bankroll < initial·(1−pct)`, blocks `record_bet` + fires `notify_stop_loss` on floor cross.
  P&L accounting consistent (deduct stake on place, credit full gross on settle → net = profit);
  summary/roi/max-drawdown correct (hand-checked the fixture: £22 profit / £30 staked). No code
  changed. Tests: `test_bet_tracker.py` **38 passed**.

- 2026-06-14 — Task 18: Telegram notifications review (`utils/notifications.py`).
  **No bugs — confirmed sound.** Queue/worker: daemon thread, `enqueue` is
  `put_nowait` (non-blocking), worker `task_done()` in a `finally` so a raising
  channel never deadlocks `join()` and the thread survives for the next message;
  `shutdown()` posts a `None` sentinel and joins. Rate limiting: `_WORKER_SLEEP=1.2s`
  between sends (≤30 msg/min Telegram cap); HTTP errors/network exceptions are caught
  - logged inside `_TelegramChannel.send`, never propagate. Channel only built when
    `enabled` AND `bot_token` AND `chat_id` all set. **Five wired call sites verified:**
    predictor `_fire_alerts` → `notify_race_soon` / `notify_new_top_pick` / `notify_odds_drop`
    (whole block wrapped in try/except, args line up); bet_tracker → `notify_stop_loss`
    (place_bet, on floor breach) + `notify_bet_outcome` (settle_bet). `notify_settle_prompt`
    exists but is intentionally unwired. Tests: `test_notifications.py` **27 passed**.
    **Guarded live send done:** sent one self-test message via the real configured creds
    → HTTP 200, `ok=True`, Telegram returned a `message_id` (delivery confirmed). No
    bot_token/chat_id/URL printed to any output or file. No code changed.

- 2026-06-14 — Task 17: normalizer + storage review. **Chain confirmed sound; one
  real defect fixed.** Pydantic v2 `UnifiedRaceRow` uses `extra="ignore"`, so
  `trainer_name`/`jockey_name` (Tasks 06/07) pass through untouched while
  `odds_decimal` is validated (`Optional[float]`, >1.0); all 7 required str fields
  are derived in the adapters. `_from_betsp`/`_from_timeform` derive canonical
  jockey/trainer ids from the names; `_from_live_odds` carries no jockey/trainer
  (live sources don't emit them — a source limit, not a bug). **Dedup correct:**
  `_DEDUPE_KEY` includes `source` so cross-source rows are preserved and only
  same-source duplicates collapse to latest `fetched_at`. **Parquet read-merge-write
  sound:** `_write` (and `parquet_store`) `rmtree` the partitioned dir before writing
  so pyarrow doesn't append stale files; merge concats existing+new then dedupes.
  **SQLite WAL sound:** pool sets `journal_mode=WAL`/`busy_timeout=5000`/
  `synchronous=NORMAL`/`foreign_keys=ON`, serializes writes via an RLock, one conn
  per thread; `locks` table gives cross-process advisory locks with expiry; upserts
  use `ON CONFLICT`. **Schema matches downstream:** every column `features/derive.py`
  - `builder.py` read (race_date, venue, horse_name, horse_id, jockey_id, trainer_id,
    going, race_class, distance, odds_decimal/sp/odds_finish, timeform_rating,
    recent_form, position) is in `CANONICAL_COLUMNS`. **FIX (carried from Task 10):**
    `_validate` did a row-by-row pydantic `validate_python` over the whole ~582k-row
    frame (~15 CPU-min in a full `normalize()`). Replaced with a vectorized mirror of
    `UnifiedRaceRow` (model kept as the declarative contract). Verified bit-identical
    keep-mask vs the old pydantic loop on a 20k real-data sample, **~194× faster**
    (0.95s→0.005s on 20k). Tests: `test_normalizer.py` + `tests/utils/storage/`
    **46 passed**; full `tests/utils` **330 passed, 3 skipped** (reportlab).

- 2026-06-14 — Task 16: historical scrapers review (`betsp_historical` /
  `timeform_historical` / `results/sporting_life`). **No bugs — chain confirmed end-to-end.**
  SL `__NEXT_DATA__` extractor still parses cleanly from typed JSON (not HTML classes):
  on real raw day 2024-01-06 (44 detail pages → 394 rows) it emits **finish_position
  77.2%** (non-finishers PU/F→`None`, correct), **jockey_name 100%**, **trainer_name 100%**,
  **going 86.3%** (race-level), **time 100%** (folded into the `race_date` minute-key via
  `_local_minute`, which UTC→local-converts so BST races don't zero the join). Confirmed
  trainer & jockey (the history source for Tasks 06/07) ARE emitted here — `ResultRow` carries
  both id+name, joiner `ENRICH_COLS` + writer carry them through. **Betfair-SP join fill on
  2024-01-06** (636 backbone rows = win+place markets × 394 runs): **trainer_name 100%,
  jockey_name 100%, position 95.6%, going 100%.** (A first pass showed 0% — my test artifact
  from `.astype(str)` on backbone `race_date`; `minute_key` only normalizes Timestamps/`T`-ISO,
  not space-separated strings — the real pipeline keeps it a Timestamp, so join is fine.)
  `timeform_historical` is racecard ratings/pace/class only (no jockey/trainer; today-scoped,
  trailing rates gate on `position`) — orthogonal, sound. Tests: `tests/scraper/betsp` +
  `tests/scraper/timeform` **42 passed**.

- 2026-06-14 — Task 15: live scrapers review (paddy*power / livescorebet / boylesports).
  **Rate-limiter + proxy-manager verified used on every request** in all three: PP wraps
  each httpx GET in `get_rate_limiter().acquire_for_url(url)` and rotates proxy +
  report_success/failure per attempt; LSB/BS use `_acquire_slot`→`rate_limiter.acquire(domain)`
  (tests inject a Throttler) with proxy rotation + reporting on every tier. **Cache envelopes
  read/write correctly** (all three via the `utils.cache.Cache` adapter). **Field mapping to
  normalizer is correct**: LSB/BS write the `_PARQUET_COLUMNS` schema (incl. `source`) into
  `live_odds.parquet` → `normalizer._from_live_odds`; PP's nested `races→markets→selections`
  cache → `_from_paddy_power` (keys `horse_name`/`odds_decimal`/`sp`/`each_way_terms.{places,
reduction}` all line up; only `selection_id` is absent so `src_horse_id` is None — harmless,
  `horse_id` is canonical from name). **Fixed one clear bug:** PP `_parse_race_time` called
  `datetime.fromisoformat("")` unguarded, so a single market missing `marketTime` (in-play /
  SP-only markets do) raised a bare `ValueError` out of `scrape()` — not `ScraperError`,
  crashing the caller. Now returns "" on empty/malformed input (matches LSB's guarded
  `_parse_event`). Added 2 tests (missing-marketTime parse + empty/garbage time). LSB and BS
  bot-detection paths already guard correctly (try/except per tier, BS falls back to stale
  cache before raising). **Live-data status (cross-checks Task 08/10):** LivescoreBet =
  **usable** (gateway API, ~297 priced selections); BoyleSports = Cloudflare/empty-DOM, needs
  DevTools re-capture; Paddy Power = Cloudflare 403 on httpx, needs ChromeFallback/re-capture.
  No new sub-tasks. Verified: `test*{paddy_power,livescorebet,boylesports}.py` **94 passed**.

- 2026-06-14 — Task 14: backtester honesty (`utils/backtester.py`). **Confirmed no
  look-ahead** — the scorer (`_score_model`/`_score_market`) only reads features/`implied_prob`;
  `position`/`outcome` never reach selection (documented in module docstring). **Fixed the
  pick==settle dishonesty:** new `_resolve_prices` splits `bet_odds` (genuine board price —
  `odds_decimal`/`board_odds`) from `settle_odds` (executable SP: `bsp`→`betfair_sp`→`sp`→
  `odds_finish`); `_simulate` now gates/stakes on `bet_odds` but **settles returns at
  `settle_odds`**. Added (1) **CLV** (`bet_odds/settle_odds−1`, only where a real board price
  AND a distinct SP exist → `has_clv`; mean% + beat-close% in `_compute_summary`, shown in
  the Overall table); (2) **per-band bootstrapped median ROI + 95% CI** (`_band_ci`,
  deterministic via `bootstrap_seed`); (3) **longshot gate** (`_robustness_gate`: flags
  positive ROI that flips ≤0 after removing top-k winners or is ≥50% longshot-driven).
  Crucial honesty guard: legacy synthesised `decimal_odds` excluded from board cols so CLV
  isn't fabricated against a price derived from the SP. Verified: `test_backtester.py` **63
  passed** (16 new: CLV calc, settle-at-SP, band CI, gate), `tests/utils` **330 passed/3
  skipped**. Real-data run (807 races/49,635 runners): settles at `odds_finish`, correctly
  reports **CLV unavailable** (no independent board price on disk yet — revives when
  `odds_decimal` backfills, Task 24), favourite-band ROI −47% [CI −72%,−12%], gate not
  flagged. Markdown report intact + 2 new sections.

- 2026-06-14 — Task 13: calibration diagnostics in `models/evaluate.py` (measurement
  only, model untouched). Added `_ae_by_band` (Actual/Expected per favourite/mid/underdog
  odds band: expected = Σ predicted prob, actual = Σ y; `flagged` when |A/E-1| > 0.20)
  and `_reliability_table` (10 equal-width pred-prob buckets → n, mean*pred, observed_freq;
  empty buckets omitted, p=1.0 clipped into last bucket). Both logged in `report()` and
  added to its returned metrics as `ae_by_band` / `reliability`; `train.py` now persists
  them per target in `catboost*\*\_meta.json`. Added 7 unit tests on synthetic known-calibration
data (perfect A/E=1 no flag, overconfident A/E=0.33 flagged, NaN-ip ignored, reliability
observed==pred, p=1.0 bucketing). Verified: `test_evaluate.py`13 passed,`test_train.py`9 passed. feature/schema audit. **No`FEATURE_COLS`change — it is exact.**
All 28 whitelisted names exist in`data/features/training.parquet`(verified against`set(df.columns)`); none misnamed vs derive/engine. Reverse check: the only column
produced by the derive/engine pipeline but NOT whitelisted is `runs_in_window`(derive.py:55, horse trailing-runs count) — a deliberate intermediate, correctly
excluded. Dead-on-disk features (fill %):`trainer_win_rate`0,`jockey_win_rate`0,`jt_combo_win_rate`0,`jt_combo_runs`all-zeros (constant),`timeform_rating`/
`rating_rank`/`pace_bias`0 (paywalled),`class_change`/`recent_form_avg`/
`ew_value_index`/`odds_drift` 0. **Recommendation: KEEP all dead columns** — trainer/
  jockey paths are already fixed (Tasks 06–07) and revive on the Task 24 re-normalize/
  backfill; CatBoost tolerates all-NaN, and a stable whitelist keeps the model schema
  identical pre/post-backfill (dropping now forces a needless schema-churn retrain later).
  No code touched; verified via two read-only column/fill-rate smoke checks.

- 2026-06-14 — Task 10: full live path end-to-end on real current data via
  `python -m scripts.refresh` (real scrape, no `--no-scrape`). **PASS — integration
  checkpoint for tasks 03–09 green.** Scrape: livescorebet 297 selections (boylesports
  403/no-DOM, paddy_power Cloudflare — both best-effort, pipeline continued) → normalize →
  `build_inference_matrix` = **297 live runners / 39 races** → `Predictor().load()+predict()`
  wrote `data/predictions.json`. Asserted: `generated_at`=2026-06-14 (today ✓), `total_races`
  =7 (>0 ✓), and **21/21 selections carry decimal_odds AND implied_prob** (post Task 08;
  e.g. Elsie's Smile 2.5→0.40). Odds attach correctly through every stage — nothing drops
  them. Predictor log: `7 races / 297 runners scored — cache written`. **Carry-forward (not a
  Task 10 bug):** `normalizer.normalize()` is slow on the full path (~15+ CPU-min) — the
  culprit is `_validate` (utils/normalizer.py:299) doing a **row-by-row pydantic
  `validate_python` loop over the whole ~582k-row combined frame** after a `to_dict('records')`.
  Correct, just O(n) Python per row; vectorize/batch in Task 17 (normalizer review). LSB
  global-card/finished-race scoping (from Task 08/09) still open for Task 15.

- 2026-06-14 — Task 09: live racecard ingestion. **No ingestion bug — data was stale.**
  Confirmed the intended daily flow: `scraper/livescorebet.py:scrape()` fetches today's
  cards via the gateway API and writes `data/live_odds.parquet`; `utils/normalizer.py:
normalize()` reads it through `_from_live_odds` (race_date from race_time, position null)
  and merges into `unified_races.parquet`. `live_odds.parquet` already held 298 fresh
  rows for 2026-06-14, but `normalize()` had not been re-run since, so unified was stuck at
  06-13 with 0 upcoming. Ran `normalize(write=True)` → `data_health.py` now shows **298
  upcoming rows** (300 odds_decimal) and `build_inference_matrix()` returns **298 live
  runners / 40 races, all priced**. Added one-shot entrypoint `python -m scripts.refresh`
  (scrape→normalize→build→predict, per-scraper best-effort) + `scripts/__init__.py`;
  verified `--no-scrape` end-to-end → 298 live runners → predictor wrote 8 races to
  `data/predictions.json`. Carry to Task 10: full scrape path not exercised here (used
  existing fresh live_odds); LSB global-card/finished-race scoping still open from Task 08.

- 2026-06-14 — Task 08: live odds pipeline. **No mapping bug existed.** `odds_decimal`
  was 2/582k purely because `data/live_odds.parquet` held only 3 synthetic test rows
  (Fast Horse / Odds On Fav / Starfire) — no real scrape was ever persisted (those 2
  surviving unified rows ARE the synthetic boylesports rows). Traced the whole chain and
  confirmed it is correct: LSB gateway API (`gateway-ie.livescorebet.com`) → `_parse_event`
  reads live decimals from `currentEvent[0].markets[].selections[].odds` (verified 259/259
  priced selections parsed) → `normalizer._from_live_odds` maps `odds_decimal` →
  `derive.add_odds_features` derives `implied_prob` → `predictor._decimal_odds` derives
  decimal odds. Paddy Power httpx returns 403 (Cloudflare; ChromeFallback path needed, not
  exercised); BoyleSports is HTML-only (no JSON API). **Fix = replaced the synthetic file
  with a real bounded scrape** (298 runners / 40 races, 298/298 `odds_decimal`) and validated
  end-to-end via `scripts/_validate_live_odds.py`: predictor emitted 8 races / 24 selections,
  **24/24 with `decimal_odds` AND `implied_prob`** (e.g. 2.75→0.3636, 3.5→0.2857). **Carry to
  Task 09:** LSB returns global cards (US/BR/FR) and finished/SP-only races; predictor treats
  them as "live" because position is never extracted and there is no finished/non-priced
  filter — needs racecard-scope + result filtering. No orchestrator wires scrape→normalize→
  predict yet (Task 10).
- 2026-06-14 — Task 07: `jockey_name` end-to-end (mirrors Task 06). **Drop point:**
  results parsers extracted only `jockey_id`, never the name — `ResultRow` had no
  `jockey_name`, and `_from_betsp` hard-wiped `jockey_id` to NA. Fixed the chain:
  added `jockey_name` to `ResultRow` (base.py); extract it in all three results
  sources (SL `jockey.name`, racing_post/at_the_races anchor text); carry it in
  `joiner.ENRICH_COLS` + `writer.FINAL_COLUMNS`; and in `_from_betsp` flow the name
  through and **derive `jockey_id` from it** (was NA) so `jockey_win_rate`/`jt_combo*`
  features — which key on `jockey_id` — revive. Native id preserved as `src_jockey_id`.
  Added `test_from_betsp_carries_jockey_name_and_derives_jockey_id`. Verified on real
  raw SL day 2024-01-06: parse → **394/394** jockey names; `_from_betsp` yields
  `jockey_name="Donagh Meyler"`, derived `jockey_id`, `src_jockey_id=847`. Tests:
  normalizer+scraper 162 passed. **Backfill required → flag Task 24:** on-disk
  betsp/unified parquet stay jockey-name-less (and jockey_id NA) until re-scraped from
  stored raw / re-normalized; same full re-normalize as the trainer fix needs.
- 2026-06-14 — Task 06: `trainer_name` end-to-end. **Drop point:** results parsers
  extracted only the trainer reference _id_, never the name — `ResultRow` had no
  `trainer_name`, so the joiner/writer/unified never carried it (0%). Fixed the whole
  chain: added `trainer_name` to `ResultRow` (base.py); extract it in all three results
  sources (sporting*life `trainer.name`, racing_post/at_the_races anchor text); carry it
  in `joiner.ENRICH_COLS` + `writer.FINAL_COLUMNS`; and in `normalizer._from_betsp` flow
  the name through and **derive `trainer_id` from it** (was hard-wiped to NA) so the
  `trainer_win_rate`/`jt_combo*\*`features — which key on`trainer_id`— revive. Native
source id preserved as`src_trainer_id`. Added
`test_from_betsp_carries_trainer_name_and_derives_trainer_id`. Verified on one real raw
SL day (2024-01-06): parse → **22/22** trainer names; joiner+writer carry it; `\_from_betsp`yields`trainer_name="T Cooper"`, derived `trainer_id`, `src_trainer_id=92`. Tests:
  betsp suite 22 passed, features+normalizer 61 passed. **NB code-path only** — existing
  on-disk betsp/unified parquet stay name-less until re-scraped/re-normalized (no backfill,
  per task scope).

- 2026-06-14 — Task 05: regenerated `data/features.parquet` via
  `features.builder.build_training_matrix(write=True)`. Before: stale **5×47**.
  After: **291,956×65** full derived matrix (training side unchanged at 248,173).
  Verified `ui/predictions.py` reads the same path the builder writes
  (`_FEATURES_PATH == _DEFAULT_FEATURES == data/features.parquet`) and that the
  join key `horse_id` is `str` on both sides (cache selections + features). Join
  check: **9/9** current-cache horse_ids matched in features; sample row populates
  feature values (e.g. `implied_prob`). No code change needed — purely a data
  refresh. The breakdown panel now finds rows.

- 2026-06-14 — Task 04: non-runner filter in `models/predictor.py`. Added `_is_non_runner`
  (matches `jockey_name == "non runner"` case/space-insensitive; no dedicated flag in unified
  schema) applied once in `_build_race` before ranking, so non-runners are dropped from
  selections, `excluded_low_odds`, and `field_size`. Added `test_non_runner_excluded_entirely`
  in `tests/models/test_predictor.py`. Verified: 3 passed (non_runner/unknown_odds/field_size).

- 2026-06-14 — Task 03: fixed "GUI shows nothing" in `ui/app.py`. (1) Date selector now
  defaults to **All upcoming** when the cache holds no race matching today, else **Today**
  (`has_today` probe). (2) Added a staleness banner (`generated_at` date + days-ago) and an
  explicit **0 upcoming races found** empty-state, distinct from **No models loaded** /
  **No prediction cache found** / **No races match the current filters**. (3) `_run_predictor`
  now returns `{status, data, message}`; the Refresh button stores it in `refresh_outcome` and a
  post-rerun banner surfaces "Predictor ran but found 0 upcoming races — scrape today's racecards
  first" instead of a silent blank. Dark theme/markup untouched. Verified with a standalone script
  against the real `data/predictions.json` (06-13 cache vs 06-14 today): old "Today" default → 0
  races, new default → 3 races; all empty-state/staleness assertions passed. `py_compile` clean.

- 2026-06-14 — Task 01 baseline: `python -m pytest -q` → **606 passed, 3 skipped, 0 failed** (~60s).
  Skips are all in `tests/utils/test_reporter.py` (lines 380/393/503) — `reportlab` not installed (PDF export optional).
  Core imports clean (no ImportError): `models.predictor`, `features.builder`, `utils.config_loader`, `utils.normalizer`.
  CHANGELOG.md already existed. No logic changed.
- 2026-06-14 — Task 02: added `scripts/data_health.py` (row/col counts, date range, fill rates
  for position/odds_decimal/implied_prob/sp/jockey_name/trainer_name, + upcoming count) and wrote
  `DATA_HEALTH.md` with its real output. Handles partitioned unified parquet and absent columns.
  Verified by running it: **all figures match the VERIFIED STATE table — zero drift.**
- 2026-06-14 — Audit complete; LightGBM removed; PROGRESS.md + PROMPTS.md seeded. (pre-loop)
