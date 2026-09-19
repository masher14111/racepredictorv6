# Race Predictor v3 — Improvement & Cleanup Prompt Pack

> A ready-to-use library of **28 prompts** to clean the project, sharpen the model, rebuild the UI, add paper-betting + bet suggestions, and make it beautiful.
> Designed so you can `/clear` between every task to keep each chat cheap and focused.

---

## How to use this file

1. Pick a prompt below (work top-to-bottom — they're ordered by dependency).
2. Start a **fresh chat** (`/clear`).
3. Set the **model** and **effort** noted on the prompt (e.g. `Opus · low`).
4. Paste the **📋 Memory paragraph** first (it re-orients the fresh chat about the project), then paste the **✍️ Prompt** underneath it. Send as one message.
5. Each prompt ends by telling the chat to **write a new memory file** + add a line to `memory/MEMORY.md`, so context compounds across sessions.

### Model & effort legend

| Tier              | When                                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Sonnet · low**  | Purely mechanical: file moves, `.gitignore`, log rotation, docs. No judgment needed.                              |
| **Opus · low**    | Scans/cleanups that need judgment about what's safe to touch (your hunch is right — Opus-low > Sonnet for these). |
| **Opus · medium** | Standard feature builds, UI pages, dashboards.                                                                    |
| **Opus · high**   | ML reasoning (features, calibration, ensembles), design architecture, leakage hunting.                            |
| **Opus · xhigh**  | The hardest, multi-faceted reasoning: backtesting engine, full model overhaul. Use sparingly.                     |

> Turn on **Fast mode** (`/fast`) for any Opus prompt — it's the same Opus quality, just faster output. Great for the `low`/`medium` ones.

### Recommended order (dependencies)

- **Phase 1 — Hygiene (1–8):** do first; gives you a clean base and a safety net (tests).
- **Phase 2 — Model (9–17):** start with #9 (baseline audit) — every other model prompt depends on it.
- **Phase 3 — UI + Betting (18–24):** #18 audit before #19 redesign.
- **Phase 4 — Graphics (25–26):** can run any time; #25 needs no chat (paste into Gemini/ChatGPT).
- **Phase 5 — Wrap (27–28):** smoke test + docs last.

### Shared base context (already folded into every memory paragraph below)

_Race Predictor v3 is a Python 3.14 horse-racing win/place prediction system for UK & Irish racing (timezone Europe/Dublin). Layout: `scraper/` (live bookmaker odds + historical results scrapers — livescorebet, boylesports, paddy_power, betfair SP, timeform), `features/` (builder.py, derive.py, engine.py build `data/features.parquet` from `data/races.db`), `models/` (CatBoost models for `won` / `placed_2` / `showed` targets with Optuna tuning, time-split CV, probability calibration, underdog sample-weighting; `predictor.py` serves live inference), `utils/` (config_loader, cache, bet_tracker, notifications, reporter), and a UI app. Config lives in `config.yaml`, loaded via `utils/config_loader.py`. Tests run with `pytest`. There is a persistent memory system in `memory/` (MEMORY.md index + one markdown file per fact)._

> **Safety convention used throughout:** for any prompt that deletes or rewrites, the chat is told to **report findings and get your confirmation before destructive changes**, work on a branch, and run `pytest` before declaring done.

---

# PHASE 1 — Cleanup & Hygiene

---

### Prompt 1 — Git hygiene: stop tracking build artifacts

**Model:** Sonnet · **Effort:** low
**Why:** Purely mechanical `.gitignore` + `git rm --cached` work.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish racing, tz Europe/Dublin). Layout: scraper/, features/, models/, utils/, data/, a UI app; config.yaml via utils/config_loader.py; pytest; memory system in memory/. Right now `git status` shows dozens of `__pycache__/*.pyc` files and data artifacts tracked as modified, which is noise. Persistent memory lives in memory/ (MEMORY.md index + one file per fact).

**✍️ Prompt:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish racing, tz Europe/Dublin). Layout: scraper/, features/, models/, utils/, data/, a UI app; config.yaml via utils/config_loader.py; pytest; memory system in memory/. Right now `git status` shows dozens of `__pycache__/*.pyc` files and data artifacts tracked as modified, which is noise. Persistent memory lives in memory/ (MEMORY.md index + one file per fact).

> Clean up Git tracking hygiene. 1) Audit `.gitignore` and add standard Python ignores (`__pycache__/`, `*.pyc`, `.pytest_cache/`, `*.log`, virtualenv dirs) plus any large data/model artifacts that shouldn't be versioned (inspect `data/` and `models/` to decide which `.parquet`/`.db`/`.cbm` files are regenerable vs. source-of-truth — list them for me before deciding). 2) Run `git rm -r --cached` on already-tracked files that now match the ignore rules, **without deleting them from disk**. 3) Show me the proposed `.gitignore` diff and the list of files to untrack, and wait for my OK before committing. 4) Commit on a new branch. When done, write a memory file `memory/cleanup-01-git-hygiene.md` (type: project) summarizing what's now ignored/untracked and which data files are considered source-of-truth, and add a one-line pointer to `memory/MEMORY.md`.

---

### Prompt 2 — Dead-file scan: orphaned modules

**Model:** Opus · **Effort:** low
**Why:** Needs judgment about entry points and dynamic imports before calling a file "unused."

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Layout: scraper/ (odds + results scrapers), features/ (builder/derive/engine → data/features.parquet from data/races.db), models/ (CatBoost won/placed_2/showed + Optuna + predictor.py), utils/, a UI app; config.yaml via utils/config_loader.py; pytest; memory/ for persistent memory.

**✍️ Prompt:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Layout: scraper/ (odds + results scrapers), features/ (builder/derive/engine → data/features.parquet from data/races.db), models/ (CatBoost won/placed_2/showed + Optuna + predictor.py), utils/, a UI app; config.yaml via utils/config_loader.py; pytest; memory/ for persistent memory.

> Find Python modules that are never imported or executed (dead files). Build an import graph across the repo starting from real entry points (the UI app, any `__main__`/CLI scripts, scheduled jobs, and pytest test files). Flag any `.py` that nothing references. Be careful with dynamic imports, plugin-style discovery, and scraper registries — check for `importlib`, string-based imports, and entry-point patterns before flagging. Produce a table: file → reason it looks orphaned → confidence (high/med/low). **Do not delete anything** — just report, then ask me which to remove. After I confirm, delete the high-confidence ones on a branch and run `pytest`. Write a memory file `memory/cleanup-02-dead-files.md` (type: project) listing what was removed and what was kept-but-suspicious, and add a pointer line to `memory/MEMORY.md`.

---

### Prompt 3 — Dead-code scan: unused functions, classes, imports

**Model:** Opus · **Effort:** low
**Why:** Tool-assisted but needs review of false positives (public API, re-exports).

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). scraper/ + features/ + models/ + utils/ + UI app; config.yaml via utils/config_loader.py; pytest; memory/ persistent memory. Codebase has grown across many refactors, so unused functions/imports are likely.

**✍️ Prompt:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). scraper/ + features/ + models/ + utils/ + UI app; config.yaml via utils/config_loader.py; pytest; memory/ persistent memory. Codebase has grown across many refactors, so unused functions/imports are likely.

> Hunt dead code _inside_ files: unused functions, classes, methods, and imports. Use `ruff` (F401/F811/etc.) and `vulture` if available (add to dev deps if not, but don't change runtime deps). Run them, then manually triage the results — exclude false positives like re-exported symbols in `__init__.py`, public API used by the UI, and anything referenced only in tests or via config strings. Give me a triaged list grouped by file with a recommended action each. After my OK, apply the safe removals on a branch and run `pytest` + `ruff` to confirm nothing broke. Write `memory/cleanup-03-dead-code.md` (type: project) summarizing removals + any lint config you added, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 4 — Dependency audit

**Model:** Opus · **Effort:** low
**Why:** Mapping imports → packages needs care (extras, transitive, optional deps).

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). scraper/ + features/ + models/ (CatBoost, Optuna) + utils/ + UI app; config.yaml via utils/config_loader.py; pytest; memory/. Dependencies are pinned in requirements.txt and have drifted over time.

**✍️ Prompt:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). scraper/ + features/ + models/ (CatBoost, Optuna) + utils/ + UI app; config.yaml via utils/config_loader.py; pytest; memory/. Dependencies are pinned in requirements.txt and have drifted over time.

> Audit `requirements.txt`. 1) Cross-reference every listed package against actual imports in the code. 2) Flag packages that are listed but never imported (candidates to remove) and packages imported but missing from requirements (must add). 3) Note any that are only used by removed/dead code from earlier cleanups. 4) Separate true runtime deps from dev/test-only deps and suggest splitting into `requirements.txt` + `requirements-dev.txt` if that's cleaner. Report the analysis first; don't edit until I confirm. After approval, update the files on a branch and verify the app still imports and `pytest` passes. Write `memory/cleanup-04-deps.md` (type: project) with the before/after dependency list and rationale, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 5 — Data & cache artifact cleanup

**Model:** Opus · **Effort:** medium
**Why:** Deleting data is dangerous — needs to distinguish regenerable caches from source-of-truth.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Data lives in data/: races.db (SQLite), features.parquet / features/training.parquet, live_odds.parquet, unified_races.parquet (partitioned by year), and data/cache/\*.json (per-bookmaker scrape caches). Pipelines: scraper/ → data/races.db → features/ → features.parquet → models/. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Data lives in data/: races.db (SQLite), features.parquet / features/training.parquet, live_odds.parquet, unified_races.parquet (partitioned by year), and data/cache/\*.json (per-bookmaker scrape caches). Pipelines: scraper/ → data/races.db → features/ → features.parquet → models/. memory/ persistent memory; pytest.

> Audit the `data/` directory for stale, duplicate, or oversized artifacts. For each file/dir (races.db, the parquet files, partitioned `unified_races.parquet`, `data/cache/*.json`), classify it as: (a) **source-of-truth** that must never be deleted, (b) **regenerable** from a pipeline step (note the exact command to regenerate), or (c) **stale/orphaned** leftover. Report sizes and last-modified. Identify duplicate or superseded parquet partitions. **Do not delete anything yet** — present the classification table and a proposed cleanup (with regeneration commands), and wait for my confirmation. Add or update a `data/README.md` documenting what each artifact is and how to rebuild it. Write `memory/cleanup-05-data-artifacts.md` (type: project) capturing the data inventory + regeneration commands, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 6 — Config consolidation

**Model:** Opus · **Effort:** low
**Why:** Needs to trace config usage across modules; low-risk edits.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). All config in config.yaml, loaded via utils/config_loader.py (recently migrated so all callers use it). scraper/ + features/ + models/ + utils/ + UI app; pytest; memory/. There may be stray hardcoded constants and orphaned config keys.

**✍️ Prompt:**

> Audit configuration consistency. 1) Find any hardcoded values in the code (paths, URLs, thresholds, model hyperparams, timezone strings) that should live in `config.yaml`. 2) Find keys in `config.yaml` that nothing reads anymore (orphaned config). 3) Verify every module reads config through `utils/config_loader.py` (no stray `yaml.safe_load`). Report findings as a table first. After I confirm, move hardcoded values into config + remove dead keys on a branch, and run `pytest`. Write `memory/cleanup-06-config.md` (type: project) summarizing the config surface and any keys added/removed, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 7 — Logging cleanup & rotation

**Model:** Sonnet · **Effort:** low
**Why:** Mechanical logging-config change.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Logging writes to logs/race_predictor.log (currently grows unbounded and gets committed). scraper/ + features/ + models/ + utils/ + UI app; config.yaml via utils/config_loader.py; pytest; memory/.

**✍️ Prompt:**

> Improve logging hygiene. 1) Make sure `logs/` is git-ignored and untracked (without deleting local logs). 2) Add a `RotatingFileHandler` (or timed rotation) so `race_predictor.log` is capped in size with a few backups, configured via `config.yaml`. 3) Sweep the codebase for `print()` calls that should be `logger` calls and noisy debug logs that should be `DEBUG` level. Keep changes minimal and centralized in whatever module sets up logging. Run `pytest`. Write `memory/cleanup-07-logging.md` (type: project) noting the rotation policy + config keys, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 8 — Test-suite health check

**Model:** Opus · **Effort:** low
**Why:** Interpreting failures/coverage gaps needs judgment.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). scraper/ + features/ + models/ + utils/ + UI app; config.yaml via utils/config_loader.py; pytest (suite has been ~600+ tests historically). memory/ persistent memory.

**✍️ Prompt:**

> Take stock of the test suite. Run `pytest -q` and report pass/fail/skip counts and total time. For any failures or persistent skips, diagnose the root cause and tell me whether it's a real bug, a stale test, or an environment issue — fix the quick/safe ones. Then run coverage (`pytest --cov`) and identify the most important **untested** areas (especially in `models/` and `features/` where silent bugs hurt most). Give me a prioritized list of 5–8 high-value tests worth adding (don't write them all now — just the plan, plus implement the top 2). Write `memory/cleanup-08-tests.md` (type: project) with current suite status + the testing backlog, and add a pointer to `memory/MEMORY.md`.

---

# PHASE 2 — Model improvement (more win probability)

---

### Prompt 9 — Model baseline audit (DO THIS FIRST in Phase 2)

**Model:** Opus · **Effort:** high
**Why:** This is the diagnostic foundation — leakage hunting + metric reasoning. Every other model prompt builds on it.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/ trains CatBoost classifiers for three targets — won / placed_2 / showed — using Optuna tuning, time-split (chronological) CV, probability calibration, and underdog sample-weighting. features/ builds data/features.parquet from data/races.db; models/predictor.py serves live inference. Known historical issues from prior reviews: sample-weight misalignment and trailing-rate denominator bugs; never use the horse's own finishing odds as a feature (leakage). memory/ persistent memory; pytest.

**✍️ Prompt:**

> Produce a rigorous baseline audit of the model pipeline — no changes yet, this is the diagnostic that the next prompts depend on. Cover: (1) **Data leakage** — verify NO feature encodes the outcome or post-race info (finishing position, returned SP, in-running). Trace each feature's construction in `features/` and confirm it's known _before_ the off. (2) **Train/validation split** — confirm it's strictly chronological with no future leakage across the split boundary, and that entity stats (jockey/trainer/horse form) are computed as-of-race-date only. (3) **Current metrics** — report won/placed_2/showed AUC, log-loss, and calibration (reliability curve / Brier) on the held-out period. (4) **Probability sanity** — do per-race win probabilities sum sensibly across runners? (5) **Class balance & weighting** — is the underdog weighting actually helping or distorting calibration? Deliver a findings report ranked by impact on win-probability quality, with a concrete fix list. Write `memory/model-09-baseline-audit.md` (type: project) capturing current metrics + the ranked improvement backlog, and add a pointer to `memory/MEMORY.md`. **This memory file becomes the reference for prompts 10–17.**

---

### Prompt 10 — Feature engineering expansion

**Model:** Opus · **Effort:** high
**Why:** Domain-heavy reasoning about predictive racing features + leakage discipline.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). features/ (builder.py, derive.py, engine.py) builds data/features.parquet from data/races.db; models/ trains CatBoost (won/placed_2/showed) on it. Hard rule: every feature must be knowable BEFORE the race goes off — no outcome/SP leakage. Prior baseline audit is in memory/model-09-baseline-audit.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md` first, then expand the feature set to improve win-probability accuracy. Propose and implement high-signal racing features that are leak-free, e.g.: recent speed/class ratings and trend, days-since-last-run (freshness/layoff), course-and-distance suitability, going/ground preference, draw/stall bias by course-distance, weight carried & weight-for-age, headgear changes (first-time blinkers etc.), jockey/trainer strike-rate (as-of-date, properly denominated), trainer-jockey combo, field size, class drop/rise vs last run, and pace/run-style projection. For each: define it precisely, confirm it's pre-race, implement it in `features/`, and **measure its marginal lift** (AUC/log-loss delta + SHAP importance) versus the current model — keep only features that help and don't hurt calibration. Watch for the known trailing-rate-denominator and as-of-date pitfalls. Run `pytest`. Write `memory/model-10-features.md` (type: project) listing features added, their measured lift, and any rejected, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 11 — Probability calibration

**Model:** Opus · **Effort:** high
**Why:** Calibration is subtle (isotonic vs Platt, per-target, reliability curves).

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/ trains CatBoost for won/placed_2/showed with existing calibration work; predictor.py serves probabilities used downstream for value/betting. Baseline metrics + calibration state in memory/model-09-baseline-audit.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md`, then make the model's probabilities **well-calibrated** (a predicted 20% should win ~20% of the time) — this matters more than raw AUC for betting. (1) Plot reliability curves and report Brier score per target on the held-out period. (2) Compare calibration methods (isotonic regression vs. Platt/sigmoid) fit on a proper held-out calibration fold (no leakage into training). (3) Add **within-race normalization** so the field's win probabilities sum to 1 (e.g. normalize after calibration), and check that this improves both calibration and ranking. (4) Persist the chosen calibrator alongside the model and wire it into `predictor.py`. Show before/after reliability curves and Brier. Run `pytest`. Write `memory/model-11-calibration.md` (type: project) with the method chosen, before/after Brier + curves description, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 12 — Hyperparameter tuning expansion

**Model:** Opus · **Effort:** medium
**Why:** Mostly configuring Optuna search well; moderate reasoning.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/tuner.py uses Optuna with time-split CV to tune CatBoost for won/placed_2/showed. Training is GPU-capable. Baseline in memory/model-09-baseline-audit.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md`, then strengthen hyperparameter tuning in `models/tuner.py`. Review the current Optuna search space and: (1) widen/refine it for CatBoost (depth, learning_rate, l2_leaf_reg, bagging_temperature, random_strength, border_count, plus class-weight/scale_pos_weight for the imbalanced targets); (2) optimize for the metric that matters — prefer log-loss / Brier (calibration-aware) over raw accuracy, and confirm CV folds are strictly chronological; (3) add a pruner (e.g. median/Hyperband) and a sensible trial budget + early stopping so it's not wasteful; (4) make the search space and trial count config-driven via `config.yaml`. Run a tuning session, report the best params + CV metric vs. the previous model, and persist them. Run `pytest`. Write `memory/model-12-tuning.md` (type: project) with the new search space, best params, and metric delta, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 13 — Ensemble & model blending

**Model:** Opus · **Effort:** high
**Why:** Multi-model design + blending weight selection without leakage.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/ is currently CatBoost-only for won/placed_2/showed (LightGBM was removed earlier). features.parquet feeds training; predictor.py serves inference. Baseline in memory/model-09-baseline-audit.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md`, then test whether an **ensemble** beats the single CatBoost model on win probability. (1) Train one or two diverse additional learners (e.g. XGBoost and/or a regularized logistic/GLM on a curated feature subset) on the same chronological splits. (2) Blend them — start with simple averaging, then a meta-learner (stacking) fit on out-of-fold predictions to avoid leakage. (3) Optionally blend with the **market** implied probability as a feature/component (this is often the single biggest accuracy gain — but test it separately so we can see model-vs-market contribution). (4) Compare AUC, log-loss, Brier, and calibration of ensemble vs. baseline on held-out data; only adopt the ensemble if it genuinely improves calibrated probability, and keep inference latency acceptable for `predictor.py`. Run `pytest`. Write `memory/model-13-ensemble.md` (type: project) with the ensemble design, weights/meta-learner, and metric deltas, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 14 — Backtesting & walk-forward validation engine

**Model:** Opus · **Effort:** xhigh
**Why:** The hardest single piece — a leak-free, realistic simulator that drives every betting decision.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/ produces calibrated win/place probabilities; data/races.db + data/live_odds.parquet + features.parquet hold history and odds. The system's purpose is to find value bets, so we need a realistic, leak-free backtester. Value-betting principles to honor (from prior reference): no odds-leakage in features, prefer CLV (closing-line value) and A/E (actual-vs-expected) over raw ROI, calibrate before betting, gate longshots. Baseline in memory/model-09-baseline-audit.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md`, then build a rigorous **backtesting / walk-forward engine** in a new `backtest/` module (or `models/backtest.py` if that fits better). Requirements: (1) **Walk-forward** — repeatedly train on data up to date T, predict races on (T, T+window], roll forward; never let future data touch a prediction. (2) **Realistic execution** — bet at the odds actually available (BSP or pre-race price from `live_odds.parquet`), account for the over-round/margin, and support stake schemes (flat, fractional-Kelly). (3) **Proper metrics** — ROI, yield, hit-rate, max drawdown, Sharpe-like ratio, **CLV** vs closing/SP, and **A/E ratio** by probability bucket and by odds band. (4) **Staking/strategy hooks** — pluggable so we can test "bet when model prob > implied prob by X%", longshot gating, min-EV thresholds. (5) Output a clean results report + per-strategy comparison, and save runs so the UI can read them later. Add tests for the no-leakage guarantees and the EV/Kelly math. Run `pytest`. Write `memory/model-14-backtest.md` (type: project) documenting the engine's API, assumptions, and headline results, and add a pointer to `memory/MEMORY.md`. **Prompts 16 & 22 depend on this.**

---

### Prompt 15 — Feature selection & SHAP explainability

**Model:** Opus · **Effort:** medium
**Why:** Pruning + SHAP setup; moderate reasoning, sets up UI explanations.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/ trains CatBoost on features.parquet (won/placed_2/showed); predictor.py serves inference. After feature expansion the feature set may be large/redundant. We also want per-prediction explanations for the UI later. Baseline in memory/model-09-baseline-audit.md, features in memory/model-10-features.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-09-baseline-audit.md` and `memory/model-10-features.md`, then (1) compute global feature importance via SHAP and CatBoost importances, identify redundant/low-signal features, and test pruning them — keep the model at least as accurate and calibrated while smaller/faster. (2) Add a reusable function that returns **per-prediction SHAP contributions** (top positive/negative drivers for a given horse in a given race) so the UI can later show "why this horse" — make it fast enough for live use (cache the explainer). Report the pruned feature list + accuracy/calibration impact, and demo the per-prediction explanation on a sample race. Run `pytest`. Write `memory/model-15-feature-selection.md` (type: project) with the final feature set + the explanation API, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 16 — Market-aware value detection

**Model:** Opus · **Effort:** high
**Why:** EV math + value-edge calibration; needs the backtester to validate.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/predictor.py outputs calibrated win/place probabilities; data/live_odds.parquet holds current bookmaker odds (livescorebet, boylesports, paddy_power) and BSP history. Goal: find VALUE — where model probability exceeds market-implied probability. Backtester exists (memory/model-14-backtest.md). Value principles: remove over-round before comparing, prefer CLV, gate longshots, don't bet tiny edges. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-14-backtest.md`, then implement a **value-detection layer**. (1) Convert bookmaker odds to implied probabilities and **strip the over-round** (normalize the book) so model-vs-market is apples-to-apples. (2) Compute expected value (EV) and a value edge = model_prob − fair_implied_prob for each runner; compute Kelly fraction. (3) Define value-bet criteria (min edge %, min/max odds band to gate longshots & odds-on shots, min model confidence) — make thresholds config-driven. (4) **Validate with the backtester** — show that bets passing the value filter produce positive CLV and a sane A/E, not just positive backtested ROI (overfit risk). Output a `find_value_bets(race)` API returning ranked value picks with edge, EV, suggested Kelly stake, and confidence. Run `pytest`. Write `memory/model-16-value-detection.md` (type: project) with the value API + validated thresholds, and add a pointer to `memory/MEMORY.md`. **Prompt 22 (suggestion engine) uses this.**

---

### Prompt 17 — Live inference robustness & probability normalization

**Model:** Opus · **Effort:** medium
**Why:** Hardening predictor.py against missing data + ensuring coherent per-race probs.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). models/predictor.py scores upcoming races (race_date >= today) using calibrated CatBoost models; it joins live entries to feature history. Past bugs: it once scored historical join-misses as "live"; entity-pruned derive made it fast (~1.7s). Live data is often incomplete (debut horses, missing jockey/trainer stats). memory/ persistent memory; pytest.

**✍️ Prompt:**

> Harden live inference in `models/predictor.py` for real-world messy data. (1) Make it robust to **missing/partial features** (first-time runners, unknown jockey/trainer, no recent form) — define sensible defaults/imputation that match how the model was trained, and never silently emit garbage probabilities. (2) Confirm it only scores genuinely upcoming races and never resurrects the historical-join-miss bug — add a guard + test. (3) Ensure output probabilities are **calibrated and normalized within each race** (field win probs sum to ~1) and that place/show probs are coherent with win prob. (4) Return a clean, typed result object the UI can consume (per-horse: win/place/show prob, value edge if available, confidence, data-completeness flag). Add tests for the missing-data and normalization paths. Run `pytest`. Write `memory/model-17-inference.md` (type: project) describing the inference contract + imputation rules, and add a pointer to `memory/MEMORY.md`.

---

# PHASE 3 — UI, paper-betting & suggestions

> The UI prompts use the **impeccable** skill. In each, start by detecting the UI framework (likely Streamlit given `page_link`/multipage references, but confirm). For Streamlit, apply impeccable's design principles via custom CSS/theme + Plotly/Altair charts + HTML components; for a standalone web app, use impeccable fully.

---

### Prompt 18 — UI audit: surface every feature, fix what's broken

**Model:** Opus · **Effort:** medium
**Why:** Inventory + diagnosis of broken pages; sets the redesign scope.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). It has a UI app (multipage; refs to page_link, a settings page, race/horse breakdown pages). Backend now exposes: calibrated win/place/show probabilities (models/predictor.py), per-prediction SHAP explanations, value-bet detection, and a backtester. Prior UI bugs: feature-path mismatch made breakdowns silently empty, jockey shown blank everywhere, settings page crashed, page_link broken, and a Today-filter hid races. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Audit the UI end-to-end. First detect the framework and map every page/route and what it currently shows. Then: (1) catalogue which **backend capabilities are NOT surfaced** in the UI (predictions, calibrated probs, SHAP "why", value bets, backtest results, scraper/data freshness, model metrics) — there's likely a lot the model produces that the UI never displays. (2) Reproduce and root-cause the known broken bits (blank jockey, settings crash, broken page links, over-aggressive Today-filter, empty breakdowns) and fix the quick ones now. (3) Run `/impeccable critique` on the main surface for a heuristic UX score and `/impeccable audit` for a11y/responsive/perf issues. Deliver: a page-by-page inventory, a "missing features" list, fixed bugs, and a prioritized redesign brief for the next prompt. Write `memory/ui-18-audit.md` (type: project) with the page map, missing-features list, and redesign brief, and add a pointer to `memory/MEMORY.md`. **Prompt 19 reads this.**

---

### Prompt 19 — Beautiful redesign: design system + core screens

**Model:** Opus · **Effort:** high
**Why:** Design architecture + committed visual direction. This is the centerpiece UI build.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). It's a personal analytics/betting-research tool (NOT real money — paper betting only). Backend exposes calibrated win/place/show probabilities, SHAP explanations, value-bet picks, and backtest results. UI audit + redesign brief are in memory/ui-18-audit.md. We want it to look premium and distinctive — a refined data-driven racing/analytics aesthetic, NOT generic SaaS-cream, NOT default Streamlit gray. memory/ persistent memory; pytest. The impeccable design skill is available.

**✍️ Prompt:**

> Read `memory/ui-18-audit.md`, then run `/impeccable shape` to plan the redesign, followed by `/impeccable craft` to build it. Establish a real **design system first**: a committed color strategy (pick a distinctive direction — e.g. deep racing-green/oxblood/charcoal with a sharp accent, in OKLCH — avoid the cream/SaaS defaults and default Streamlit theme), a type pairing on a contrast axis, spacing scale, and reusable components (race card, runner row, probability bar, value badge, confidence chip). Then build the **core screens**: a Today/race-list landing, a race detail view (runners with calibrated win% bars, value edge, SHAP "why" drivers), and a clean nav. Verify contrast ≥4.5:1 for body text, responsive at mobile/tablet/desktop, and screenshot the result to check it. If the app is Streamlit, implement the system via injected CSS + custom HTML components + Plotly/Altair for charts (themed to match). Honor impeccable's bans (no gradient text, no side-stripe cards, no eyebrow-on-every-section). Write `memory/ui-19-redesign.md` (type: project) documenting the design tokens, components, and screens built, and add a pointer to `memory/MEMORY.md`. **Prompts 20–24 extend this system.**

---

### Prompt 20 — Paper-betting feature (fake bets, no real money)

**Model:** Opus · **Effort:** medium
**Why:** Standard feature build on top of the new design system + existing bet_tracker.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). NOT real money — we want PAPER betting to test if the model works. utils/bet_tracker.py already exists. Backend has calibrated probs + value detection; live + historical odds in data/live_odds.parquet / races.db; results come from the scrapers so bets can be settled automatically. Design system + components from memory/ui-19-redesign.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/ui-19-redesign.md`, then build a **paper-betting** feature (clearly labeled "Paper / Practice — no real money"). (1) Inspect `utils/bet_tracker.py` and reuse/extend it; only build new storage if it's insufficient. A bet records: race, horse, market (win/place), odds taken, stake (from a virtual bankroll), timestamp, and the model's probability/edge at bet time (so we can measure CLV later). (2) UI: a "Place paper bet" action on each runner in the race detail view (styled with the existing component system), a virtual bankroll the user sets, and an open-bets list. (3) **Auto-settle** bets when results arrive from the scrapers (won/lost/void), updating bankroll. (4) Guard against double-betting and betting on started races. Add tests for staking, settlement, and CLV capture. Run `pytest`. Write `memory/ui-20-paper-betting.md` (type: project) describing the bet schema, settlement flow, and UI entry points, and add a pointer to `memory/MEMORY.md`. **Prompt 21 visualizes this data.**

---

### Prompt 21 — Bankroll & performance dashboard

**Model:** Opus · **Effort:** medium
**Why:** Data viz + metric aggregation on top of paper-bet data and backtests.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Paper bets are recorded + auto-settled (memory/ui-20-paper-betting.md) with the model's prob/edge captured at bet time. A backtester produces ROI/CLV/A/E results (memory/model-14-backtest.md). Design system in memory/ui-19-redesign.md. We want to SEE whether the model actually works. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/ui-20-paper-betting.md`, `memory/model-14-backtest.md`, and `memory/ui-19-redesign.md`, then build a **performance dashboard** that shows whether the model is working. Display, using charts themed to the design system (Plotly/Altair or equivalent): bankroll-over-time equity curve, cumulative profit, ROI/yield, hit-rate, max drawdown, **CLV** (did we beat the closing/SP?), and an **A/E (actual vs expected) by probability bucket** chart that doubles as a live calibration check. Let the user filter by date range, market, odds band, and "model picks only vs all bets." Show backtest results alongside paper-trading results for comparison. Avoid the hero-metric SaaS cliché — make the equity curve and calibration the heroes. Run `pytest`. Write `memory/ui-21-dashboard.md` (type: project) listing the metrics + charts and where data comes from, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 22 — Bet-suggestion engine (model tells you what to bet)

**Model:** Opus · **Effort:** high
**Why:** Combines value detection + staking + confidence tiering into recommendations; needs careful reasoning to avoid over-confident picks.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Backend has calibrated win/place/show probs, value detection (model edge vs over-round-stripped market) with fractional-Kelly staking and longshot gating (memory/model-16-value-detection.md), SHAP explanations (memory/model-15-feature-selection.md), and a validated backtester (memory/model-14-backtest.md). Design system in memory/ui-19-redesign.md. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/model-16-value-detection.md`, `memory/model-15-feature-selection.md`, and `memory/ui-19-redesign.md`, then build a **bet-suggestion engine** that recommends what to bet on today's races. (1) For each upcoming race, surface the best value bet(s): horse, market, model prob, fair vs offered odds, edge %, EV, suggested Kelly stake (fractional, capped), and a **confidence tier** (e.g. Strong / Lean / Pass) derived from edge size, model confidence, and data completeness. (2) Attach a short, human-readable **rationale** from SHAP ("backed by: strong recent speed figs, course-and-distance winner, in-form trainer"). (3) Rank/curate — don't spam a pick on every race; show "no value" honestly when there isn't any, and gate longshots/odds-on per the value config. (4) UI: a "Today's suggestions" view (themed to the design system) with one-click "place paper bet" using the suggested stake. Make recommendations reproducible and explainable, never a black box. Run `pytest`. Write `memory/ui-22-suggestions.md` (type: project) describing the suggestion logic, tiers, and rationale source, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 23 — Race & horse detail pages with explanations

**Model:** Opus · **Effort:** medium
**Why:** Rich detail views + explainability presentation.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Backend exposes calibrated probs, per-prediction SHAP drivers (memory/model-15-feature-selection.md), value edges, and historical form from data/races.db. Design system + race detail base in memory/ui-19-redesign.md. Prior bug: breakdown pages were silently empty due to a feature-path mismatch — verify data actually loads. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/ui-19-redesign.md` and `memory/model-15-feature-selection.md`, then build rich **race and horse detail pages**. Race page: full field with calibrated win/place bars, value badges, pace/run-style projection, going/draw context, and a clear "model's view" summary. Horse page: recent form (last runs with finishing positions, going, class, speed figs), trainer/jockey stats, and the **SHAP "why" breakdown** for the current race shown as readable positive/negative drivers (not a raw plot). Confirm the data actually loads (regression-test the old empty-breakdown path). Keep everything on the design system's components; charts themed to match. Verify on real upcoming races by screenshotting. Run `pytest`. Write `memory/ui-23-detail-pages.md` (type: project) describing the detail views + data sources, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 24 — Polish pass: motion, responsive, empty/loading states

**Model:** Opus · **Effort:** medium
**Why:** Final craft pass using impeccable's polish/animate/adapt.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). UI is rebuilt on a committed design system (memory/ui-19-redesign.md) with paper betting, dashboard, suggestions, and detail pages. Now it needs a final craft pass: motion, responsiveness, empty/error/loading states. The impeccable design skill is available. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Read `memory/ui-19-redesign.md`, then run a final polish pass using `/impeccable polish`, then `/impeccable animate` and `/impeccable adapt` as needed. Focus on: (1) **purposeful motion** — staggered reveals on race lists, smooth probability-bar fills, transitions on bet placement/settlement — with a `prefers-reduced-motion` alternative for each (required). (2) **Responsive** — verify every screen at mobile/tablet/desktop; fix any heading overflow or clipped dropdowns. (3) **States** — design real empty states ("no races today", "no value bets found"), loading skeletons for the ~1.7s inference, and clear error states when scrapers/data are stale. (4) Re-check contrast and run `/impeccable audit` for a11y/perf. Screenshot before/after. Run `pytest`. Write `memory/ui-24-polish.md` (type: project) listing the motion/states/responsive fixes, and add a pointer to `memory/MEMORY.md`.

---

# PHASE 4 — Graphics & imagery

---

### Prompt 25 — Image generation prompts for Gemini/ChatGPT

**Model:** _(no Claude chat needed — paste the prompts in Appendix A directly into Gemini/ChatGPT/Imagen)_
**Effort:** n/a

This one doesn't need a Claude session. **The ready-to-paste image prompts are in Appendix A** at the bottom of this file — copy them straight into your image model. They're tuned to match a premium, dark, data-driven racing aesthetic so they fit the redesign. Generate, then drop the assets into your UI's static/assets folder and reference them from the design system.

> If you'd rather have a Claude chat wire the generated images in (optimize, convert to WebP, set responsive `srcset`, add as CSS backgrounds): **Sonnet · low** with the memory paragraph: _"Race Predictor v3 UI (see memory/ui-19-redesign.md for the design system). I've added raw image files to <folder>. Optimize them (WebP, sized variants), and integrate them into the UI as backgrounds/hero/empty-state art per the design system. Write memory/ui-25-images.md and add a MEMORY.md pointer."_

---

### Prompt 26 — In-house CSS/SVG graphics (no external images)

**Model:** Opus · **Effort:** medium
**Why:** Generative visual craft — often looks more cohesive than stock art.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). UI is on a committed OKLCH design system (memory/ui-19-redesign.md). We prefer crisp, lightweight, on-brand graphics over stock photos where possible. The impeccable design skill is available. memory/ persistent memory.

**✍️ Prompt:**

> Read `memory/ui-19-redesign.md`, then create **in-house visual assets** with CSS/SVG (no external images) that match the design system: a subtle hero background (e.g. a layered mesh-gradient or fine track-rail line motif rendered in the brand OKLCH colors), a set of cohesive **SVG icons** (race, runner, value, bankroll, calibration, trophy), probability/strength meter graphics, an empty-state illustration (simple line-art horse or finish-line in brand colors), and a favicon/logo mark. Keep them lightweight and theme-driven (use CSS variables so they recolor with the theme). Use blur/mask/clip-path as premium materials where they help, but obey impeccable's bans (no gradient text, no decorative glassmorphism by default). Show them in context with a screenshot. Write `memory/ui-26-css-graphics.md` (type: project) cataloguing the assets + how to reuse them, and add a pointer to `memory/MEMORY.md`.

---

# PHASE 5 — Wrap-up

---

### Prompt 27 — End-to-end smoke test

**Model:** Opus · **Effort:** medium
**Why:** Orchestrating the full pipeline + UI and diagnosing breaks.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Full pipeline: scraper/ → data/races.db + live odds → features/ → features.parquet → models/ (train/calibrate) → predictor.py → UI (paper betting, suggestions, dashboard). After many changes we need to confirm the whole chain still runs. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Run an end-to-end smoke test of the whole system and report exactly what works and what's broken — faithfully, with the actual output. Steps: (1) run the scrapers (or a small/limited fetch) and confirm data lands in `data/races.db` / live odds; (2) build features and confirm `features.parquet` regenerates; (3) train (or load) models and confirm calibrated predictions come out of `predictor.py` for today's races; (4) launch the UI and click through race list → race detail → place a paper bet → settle → dashboard → suggestions, screenshotting each; (5) run `pytest`. For anything that breaks, root-cause and fix the quick ones, and list the rest as a prioritized backlog. Don't claim success on steps you skipped. Write `memory/wrap-27-smoke-test.md` (type: project) with the end-to-end status + remaining issues, and add a pointer to `memory/MEMORY.md`.

---

### Prompt 28 — Documentation refresh

**Model:** Sonnet · **Effort:** low
**Why:** Mechanical doc writing from existing state.

**📋 Memory paragraph:**

> Race Predictor v3 — Python 3.14 horse-racing predictor (UK/Irish, tz Europe/Dublin). Pipeline: scraper/ → data/races.db → features/ → features.parquet → models/ → predictor.py → UI (paper betting, suggestions, dashboard). There are PROGRESS.md and PROMPTS.md files and a memory/ system with many memory files documenting recent work. memory/ persistent memory; pytest.

**✍️ Prompt:**

> Refresh the project documentation to match its current state. Read the recent memory files in `memory/` to understand what changed, then update (or create) a clear top-level `README.md` covering: what the project is, the data→features→models→UI pipeline, how to set up + run each stage, how to run the UI, how paper betting works, and how to run tests. Update `PROGRESS.md` with what's now done and what's next. Keep it accurate and concise — no aspirational claims about things that don't work. Write `memory/wrap-28-docs.md` (type: project) noting what docs were updated, and add a pointer to `memory/MEMORY.md`.

---

# Appendix A — Ready-to-paste image prompts (Gemini / ChatGPT / Imagen)

Copy any of these directly into your image model. **House style** (prepend or keep consistent): _"Premium dark UI asset for a horse-racing analytics app. Refined, modern, editorial — deep charcoal/racing-green/oxblood palette with a single sharp accent. Clean, high-end, data-driven. No text, no watermark, no logos."_ Generate at 2× the display size and export PNG/WebP.

1. **Hero background (dashboard) —** _"Abstract premium background for a horse-racing analytics dashboard: subtle layered gradient mesh in deep racing green and charcoal with a faint oxblood glow, very soft fine diagonal track-rail lines, lots of negative space, dark, elegant, minimal, no text. 16:9, ultra-clean, suitable as a web hero background."_

2. **Login / splash —** _"Cinematic, moody wide shot of an empty racecourse at dawn, low fog over the turf, soft rim light on the running rail, desaturated deep-green and charcoal tones with a single warm accent of sunrise, photographic, premium, depth of field, no people, no text. 16:9."_

3. **Empty-state illustration —** _"Minimal line-art illustration of a single galloping racehorse, thin elegant strokes in a sharp accent color on transparent background, lots of whitespace, modern editorial style, no shading, no text. Square, for an app empty state."_

4. **Finish-line empty state —** _"Simple geometric line illustration of a finish line and post with a subtle checkered motif, thin strokes, brand green and charcoal on transparent, calm and minimal, no text. Square."_

5. **Icon set base —** _"A cohesive set of minimal line icons on transparent background, 2px uniform stroke, rounded joins, single accent color: a racehorse, a jockey cap, a trophy, a coin/value tag, a bar-chart, a target/edge symbol, a wallet/bankroll. Flat, modern, consistent grid, no text."_

6. **Value badge texture —** _"Subtle premium foil/metallic texture swatch in deep emerald-green with a faint sheen, seamless, dark, understated, for use as a small UI badge background, no text. Square, tileable."_

7. **Card background accent —** _"Very subtle abstract topographic/contour line pattern in dark charcoal with barely-visible green lines, seamless tile, low contrast so text remains readable on top, premium and quiet, no text."_

8. **Calibration / data motif —** _"Abstract elegant visualization motif: smooth flowing probability curve and scattered glowing dots over a dark background, deep green to oxblood gradient on the dots, data-art aesthetic, minimal, no axes, no text. 16:9."_

9. **Loading / spinner concept —** _"A minimal circular motion graphic concept: a thin racing-rail loop with a single bright dot tracing it, on dark transparent background, sleek and modern, no text. Square (will be animated in CSS)."_

10. **Favicon / app mark —** _"A simple, bold app icon mark for a horse-racing prediction app: an abstract monogram combining a horseshoe and an upward trend line, single accent color on dark rounded-square background, flat, crisp at small sizes, no text."_

---

# Appendix B — Tips for running this pack

- **Always paste the memory paragraph first**, then the prompt, in the same message. The fresh chat needs that orientation after `/clear`.
- **Do Phase 2 in order from #9.** The baseline audit (`model-09`) is the file every other model prompt reads — skipping it makes the rest guess.
- **#14 (backtester) is the linchpin** of the betting half — it's the only honest judge of whether the model "works." Give it `xhigh` and don't rush it.
- **The memory files chain the work together** — by the time you reach the UI prompts, `memory/` holds the model contracts the UI needs. That's why each prompt writes one.
- **Branch + `pytest` before "done"** is baked into every destructive/code prompt — keep that discipline.
- **Fast mode** (`/fast`) pairs well with all the Opus `low`/`medium` prompts for quicker output at the same quality.
