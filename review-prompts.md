# Race Predictor v3 — Review Prompts

Seven sequential prompts to review, verify, and document the project.
Run one per session; each prompt instructs the model to write a `.md` memory file so you can `/clear` safely.

---

## Prompt 1 — Test Suite Health Check

**Effort: LOW** | Model: Sonnet | Est. time: 5–10 min

### Paste this paragraph ABOVE Prompt 1:

> This is Race Predictor v3, a Python horse racing prediction tool at `C:\Users\mshr\Desktop\Race Predictor v3`. It has 100+ test files covering scrapers, features, models, UI, utils, and storage. Python 3.14, Windows 11, PowerShell. No git repo. After this session you will write a memory file so I can `/clear` and continue.

---

### Prompt 1:

```
This is Race Predictor v3, a Python horse racing prediction tool at `C:\Users\mshr\Desktop\Race Predictor v3`. It has 100+ test files covering scrapers, features, models, UI, utils, and storage. Python 3.14, Windows 11, PowerShell. No git repo. After this session you will write a memory file so I can `/clear` and continue.

You are reviewing Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

**Task:** Run the full test suite and produce a health report.

Steps:
1. Run `python -m pytest tests/ -v --tb=short 2>&1` from the project root (PowerShell).
2. Parse the output: count pass/fail/skip/error by test module.
3. For any failure or error, read the relevant source file and identify the root cause (do NOT fix yet — just diagnose).
4. Note any tests marked skip or xfail and why.
5. Check `requirements.txt` for packages that may not be installed and flag them.

At the end, write `review/01-test-health.md` with this exact structure:
- ## Summary (total pass/fail/skip/error counts)
- ## Failures (file, test name, root cause, severity: critical/minor)
- ## Skipped / xfail (reason)
- ## Missing deps (any packages flagged)
- ## Next (one-line recommendation for Prompt 2)

Do not fix anything. Just diagnose and document.
```

---

## Prompt 2 — Utils & Notifications (Telegram) Review

**Effort: MEDIUM** | Model: Sonnet | Est. time: 15–20 min

### Paste this paragraph ABOVE Prompt 2:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Test suite results are in `review/01-test-health.md` — read it first. This session focuses on reviewing the utility layer: config_loader, cache, notifications (Telegram), bet_tracker, and reporter. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

---

### Prompt 2:

```
Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Test suite results are in `review/01-test-health.md` — read it first. This session focuses on reviewing the utility layer: config_loader, cache, notifications (Telegram), bet_tracker, and reporter. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

You are reviewing Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

Read `review/01-test-health.md` first for known failures.

**Task:** Review and verify the utility layer for correctness.

Files to review (read each in full):
- `utils/config_loader.py` — hot-reload singleton; check mtime polling, thread safety, schema validation
- `utils/cache.py` — in-memory cache; check TTL, thread safety, get/set interface
- `utils/notifications.py` — Telegram bot; check bot_token/chat_id wiring from config.yaml, message formatting, daemon thread scheduler, error handling when Telegram is unreachable
- `utils/bet_tracker.py` — check all_bets(), summary(), pl_series(), breakdown(by=) return shapes and edge cases (empty DB)
- `utils/reporter.py` — check generate(), export_csv/json/pdf, integrity checker, SHA-256 manifest
- `config.yaml` — verify notifications.bot_token and chat_id are populated; verify reporter block

For Telegram specifically:
- Confirm the config values (bot_token: `8648732935:...`, chat_id: `6618994572`) are correctly wired into the notification send path
- Trace the code path from `notifications.enabled: true` → daemon thread start → send_message() → Telegram API call
- Identify any bugs that would prevent a real Telegram message from being sent

For each file: list bugs found (if any), confirm what works correctly, note any edge-case gaps.

At the end, write `review/02-utils-review.md` with:
- ## Config Loader (verdict + bugs)
- ## Cache (verdict + bugs)
- ## Notifications / Telegram (verdict + bugs + confirmed send path)
- ## Bet Tracker (verdict + bugs)
- ## Reporter (verdict + bugs)
- ## Critical fixes needed (list, ordered by severity)
- ## Next (one-line recommendation for Prompt 3)
```

---

## Prompt 3 — Scrapers Review

**Effort: MEDIUM** | Model: Sonnet | Est. time: 15–20 min

### Paste this paragraph ABOVE Prompt 3:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Utils layer review is in `review/02-utils-review.md`. This session reviews all scrapers: Paddy Power, LivescoreBet, BoyleSports, Timeform historical, and Betfair SP historical. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

---

### Prompt 3:

```
Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Utils layer review is in `review/02-utils-review.md`. This session reviews all scrapers: Paddy Power, LivescoreBet, BoyleSports, Timeform historical, and Betfair SP historical. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

You are reviewing Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

Read `review/02-utils-review.md` first.

**Task:** Review all scrapers for correctness, error handling, and integration.

Files to review:
- `scraper/paddy_power.py` — note: API endpoints unconfirmed (see warning in file); review parse logic, fallback chain, cache wiring
- `scraper/livescorebet.py` — review parse logic, parquet write, rate limiting wiring
- `scraper/boylesports.py` — review HTML DOM scraping, cross-source EW terms, region filter, tier fallback chain
- `scraper/_selenium_fallback.py` — review CDP XHR capture pattern, reusability
- `scraper/timeform_historical.py` + `scraper/timeform/` — review full subpackage: client, parser, extractor, writer, pages
- `scraper/betsp_historical.py` + `scraper/betsp/` — review betfair SP data fetch, results sources, joiner, writer

For each scraper: does it integrate correctly with `utils/cache.py`, `utils/rate_limiter.py`, `utils/proxy_manager.py`? Does it handle network errors gracefully? Does it write to the correct parquet/JSON paths?

Note any scraper with a known "stub" or unconfirmed endpoint — do not attempt to fix Paddy Power endpoints (they need DevTools confirmation), just flag them clearly.

At the end, write `review/03-scrapers-review.md` with:
- ## Paddy Power (verdict, known gaps, integration status)
- ## LivescoreBet (verdict, bugs)
- ## BoyleSports (verdict, bugs)
- ## Selenium Fallback (verdict)
- ## Timeform Historical (verdict, bugs)
- ## Betfair SP Historical (verdict, bugs)
- ## Integration matrix (table: scraper × cache/rate_limiter/proxy — ✓/✗/partial)
- ## Critical fixes needed
- ## Next (one-line recommendation for Prompt 4)
```

---

## Prompt 4 — Features & Models Pipeline Review

**Effort: HIGH** | Model: Opus | Est. time: 25–35 min

### Paste this paragraph ABOVE Prompt 4:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Scrapers review is in `review/03-scrapers-review.md`. This session reviews the ML pipeline: feature engineering (features/) and model training/prediction (models/). This is the most complex part of the codebase. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

---

### Prompt 4:

```

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. Scrapers review is in `review/03-scrapers-review.md`. This session reviews the ML pipeline: feature engineering (features/) and model training/prediction (models/). This is the most complex part of the codebase. Python 3.14, Windows 11. After this session write a memory file so I can `/clear`.

You are reviewing Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

Read `review/03-scrapers-review.md` first.

**Task:** Review the full ML pipeline — feature engineering → training → prediction.

Files to review (read each in full):
- `features/engine.py` — main entry point; how it wires builder/fuse/derive/labels
- `features/builder.py` — feature construction from raw data
- `features/derive.py` — derived/computed features
- `features/fuse.py` — multi-source data fusion
- `features/labels.py` — target label generation
- `models/train.py` — training loop, LightGBM + CatBoost, Optuna tuning wiring
- `models/tuner.py` — Optuna hyperparameter search
- `models/predictor.py` — inference: load model → feature transform → predict
- `models/evaluate.py` — evaluation metrics, AUC, calibration
- `models/retrain_trigger.py` — PSI/KS drift detection
- `models/targets.py` — target definitions (won, placed_2, showed)
- `models/features.py` — feature list / schema

Check:
1. Does the feature pipeline produce a valid DataFrame that matches what `models/train.py` expects?
2. Are column names consistent from features/ → models/?
3. Does `predictor.py` correctly load the model artefact and return a composite_score in [0,1]?
4. Does `retrain_trigger.py` correctly read PSI threshold from config?
5. Are there any import cycles between features/ and models/?

At the end, write `review/04-features-models-review.md` with:
- ## Feature Pipeline (verdict, column schema produced)
- ## Training Pipeline (verdict, bugs)
- ## Prediction (verdict, bugs)
- ## Drift Detection (verdict)
- ## Data contract (features/ → models/ column alignment: ✓/✗/unknown per column)
- ## Critical fixes needed
- ## Next (one-line recommendation for Prompt 5)
```

---

## Prompt 5 — UI Review

**Effort: MEDIUM** | Model: Sonnet | Est. time: 15–20 min

### Paste this paragraph ABOVE Prompt 5:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. ML pipeline review is in `review/04-features-models-review.md`. This session reviews the Streamlit UI: all pages, navigation, bet placement, and settings wiring. Python 3.14, Windows 11, Streamlit. After this session write a memory file so I can `/clear`.

---

### Prompt 5:

```

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. ML pipeline review is in `review/04-features-models-review.md`. This session reviews the Streamlit UI: all pages, navigation, bet placement, and settings wiring. Python 3.14, Windows 11, Streamlit. After this session write a memory file so I can `/clear`.

You are reviewing Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

Read `review/04-features-models-review.md` first.

**Task:** Review the Streamlit UI for correctness and completeness.

Files to review:
- `ui/app.py` — main Streamlit entry point, page routing, startup hooks (does it call start_scheduler for notifications/reporter?)
- `ui/live_races.py` — live race display, odds table, scraper integration
- `ui/predictions.py` — prediction display, composite score rendering
- `ui/performance_dashboard.py` — performance charts, ReportLab PDF export integration
- `ui/performance.py` — performance data loading
- `ui/bet_placer.py` — bet placement UI, BetTracker wiring
- `ui/race_compare.py` — race comparison view
- `ui/settings.py` — config editor UI, does it correctly write back to config.yaml?

Check:
1. Does `app.py` import and start the notification scheduler? Does it start the reporter scheduler?
2. Does `ui/live_races.py` call the correct scraper(s) and display results correctly?
3. Does `ui/predictions.py` call `models/predictor.py` and display composite_score correctly?
4. Does `ui/bet_placer.py` call `BetTracker` correctly?
5. Are all pages importable without runtime errors (no missing imports, no missing config keys)?
6. Does the app have a clean startup sequence — no circular imports, no blocking calls at import time?

At the end, write `review/05-ui-review.md` with:
- ## App startup (verdict, scheduler wiring)
- ## Live Races page (verdict, bugs)
- ## Predictions page (verdict, bugs)
- ## Performance Dashboard (verdict, bugs)
- ## Bet Placer (verdict, bugs)
- ## Settings (verdict, bugs)
- ## Import health (any circular imports or missing imports found)
- ## Critical fixes needed
- ## Next (one-line recommendation for Prompt 6)
```

---

## Prompt 6 — Fix Critical Bugs

**Effort: HIGH** | Model: Opus | Est. time: 30–40 min

### Paste this paragraph ABOVE Prompt 6:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. All review files are in `review/`. Read them all before starting. This session fixes all critical bugs found across the full stack. Python 3.14, Windows 11. After fixing, re-run pytest and confirm the test count. Write a memory file so I can `/clear`.

---

### Prompt 6:

```

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. All review files are in `review/`. Read them all before starting. This session fixes all critical bugs found across the full stack. Python 3.14, Windows 11. After fixing, re-run pytest and confirm the test count. Write a memory file so I can `/clear`.

You are fixing bugs in Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

First read ALL review files:
- `review/01-test-health.md`
- `review/02-utils-review.md`
- `review/03-scrapers-review.md`
- `review/04-features-models-review.md`
- `review/05-ui-review.md`

Collect every item marked as "critical" across all five files. Prioritise in this order:
1. Bugs that cause ImportError or crash on startup
2. Bugs that break Telegram notifications end-to-end
3. Bugs that break the prediction pipeline (features → models → predictor)
4. Bugs that break the UI pages
5. Test failures (non-skip)

For each critical bug:
- Read the relevant source file in full
- Make the minimal correct fix (no refactoring, no new features)
- Run `python -m pytest tests/ -v --tb=short 2>&1` after all fixes are applied

Do NOT fix:
- Paddy Power API endpoint URLs (they need DevTools confirmation)
- Any item marked "minor" or "nice to have" in the review files
- Timeform session_cookie (requires user account)

At the end, write `review/06-fixes.md` with:
- ## Fixes applied (file, what was wrong, what was changed)
- ## Pytest result after fixes (pass/fail/skip counts)
- ## Unfixed items (and why they were skipped)
- ## Next (one-line recommendation for Prompt 7)
```

---

## Prompt 7 — Generate setup.md

**Effort: LOW** | Model: Sonnet | Est. time: 10–15 min

### Paste this paragraph ABOVE Prompt 7:

> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. All reviews and fixes are in `review/`. The project is now verified. This final session generates a comprehensive setup.md for end-users. Python 3.14, Windows 11. config.yaml and requirements.txt are the source of truth for config. Telegram bot_token and chat_id are already in config.yaml.

---

### Prompt 7:

```
> Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`. All reviews and fixes are in `review/`. The project is now verified. This final session generates a comprehensive setup.md for end-users. Python 3.14, Windows 11. config.yaml and requirements.txt are the source of truth for config. Telegram bot_token and chat_id are already in config.yaml.

You are documenting Race Predictor v3 at `C:\Users\mshr\Desktop\Race Predictor v3`.

Read these files for context:
- `review/06-fixes.md` (or `review/05-ui-review.md` if 06 doesn't exist)
- `config.yaml`
- `requirements.txt`
- `ui/app.py` (just the first 80 lines — for the startup command)

**Task:** Write a comprehensive `setup.md` at the project root covering:

1. **Prerequisites** — Python version, OS, Playwright install command
2. **Installation** — `pip install -r requirements.txt`, `playwright install chromium`
3. **Configuration** — explain every key in `config.yaml` that a user would need to change:
   - proxy_pool (how to add DataImpulse credentials)
   - timeform.session_cookie (where to get it, what it unlocks)
   - notifications.channels.telegram (bot_token + chat_id — already filled in, just explain how to verify)
   - reporter.enabled / interval_hours
4. **Running the app** — exact command to launch the Streamlit UI
5. **Running tests** — `python -m pytest tests/ -v`
6. **Feature walkthrough** — one paragraph per UI page: Live Races, Predictions, Performance Dashboard, Bet Placer, Race Compare, Settings
7. **Scheduled reports** — how to enable background PDF reports (reporter.enabled: true)
8. **Telegram alerts** — how to verify the bot works (what events trigger alerts, how to test)
9. **Known limitations** — Paddy Power endpoints unconfirmed; Timeform requires subscriber cookie for full TFR data; BoyleSports EW terms cross-sourced (requires at least one other bookie to have scraped)
10. **Troubleshooting** — common errors and fixes (Cloudflare 403 → Playwright fallback, empty predictions → model not trained yet, Telegram not sending → check enabled flag + chat_id)

Write clean, user-friendly markdown. No code blocks longer than 5 lines. Target audience: someone setting up the project for the first time who is comfortable with Python but not with this codebase.

Save as `setup.md` in the project root.
```

---

## Quick Reference

| #   | Topic             | Effort | Model  | Output file                           |
| --- | ----------------- | ------ | ------ | ------------------------------------- |
| 1   | Test suite health | Low    | Sonnet | `review/01-test-health.md`            |
| 2   | Utils & Telegram  | Medium | Sonnet | `review/02-utils-review.md`           |
| 3   | Scrapers          | Medium | Sonnet | `review/03-scrapers-review.md`        |
| 4   | Features & Models | High   | Opus   | `review/04-features-models-review.md` |
| 5   | UI                | Medium | Sonnet | `review/05-ui-review.md`              |
| 6   | Fix critical bugs | High   | Opus   | `review/06-fixes.md`                  |
| 7   | Generate setup.md | Low    | Sonnet | `setup.md`                            |

**Total estimated time:** 2–3 hours across 7 sessions.
**Between sessions:** `/clear` after each prompt. Paste the "paragraph above" from the next prompt to restore context.
