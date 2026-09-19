# Race Predictor v3 — Repair & Hardening Prompts

25 self-contained prompts to drive Opus/Sonnet through fixing, verifying, and documenting the
project. Run **one prompt per `/clear` cycle**. State carries over via `PROGRESS.md` (the agent's
memory) and `CHANGELOG.md`.

---

## HOW TO USE THIS (read once)

**The paste order — important:** Paste the **PRIMER + the prompt as ONE message**, primer on top,
then a blank line, then the prompt. Send it in one go. **Do not** paste the primer first and wait —
on its own it does nothing useful. The primer simply re-establishes context that `/clear` wiped, so
the model re-reads `PROGRESS.md` before each task.

**Loop per task:**

1. `/clear`
2. Paste **[PRIMER] + blank line + [PROMPT N]** as a single message → send.
3. Wait for the model to finish (it edits code, runs tests, ticks the task in `PROGRESS.md`, appends to `CHANGELOG.md`).
4. Skim its summary. If good → `/clear` and go to PROMPT N+1. If not → reply in the same session to correct it (no `/clear` yet).

**Model / effort guidance (your preference: Opus-low when it beats Sonnet — it usually does here):**

- **Default = Opus, effort `low`** for review/verify/doc/mechanical tasks. It matches or beats Sonnet on these and the token cost is fine because the primer keeps scope tight.
- **Opus, effort `medium`** for multi-file fixes that touch the data pipeline or UI logic.
- **Opus, effort `high`** only for the 3 genuinely architectural ones: **08** (odds pipeline), **12** (leakage / price-free model), **14** (backtest honesty). Each prompt states its level.
- Sonnet is fine for 18/20/22/25 if you want to save cost, but Opus-low is the safer default.

**Effort is set with `/model` or the effort toggle before you paste.** Each prompt header tells you which to use.

---

## THE PRIMER (paste this above every prompt)

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices..

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.
```

---

## PROMPTS

### 01 — Bootstrap & baseline · Opus / low

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 01. Ensure PROGRESS.md and CHANGELOG.md exist (create CHANGELOG.md with a header if missing).
Run the full suite: `python -m pytest -q`. Record in PROGRESS.md under the run log: total tests,
passed/failed/skipped, and the names of any failing tests (do not fix them yet). Confirm the repo
imports cleanly: `python -c "import ui.app"` is not meaningful for Streamlit, so instead import the
core modules: models.predictor, features.builder, utils.config_loader, utils.normalizer. Note any
ImportError. This task only establishes the baseline — change no logic.
```

### 02 — Data-health report · Opus / low

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 02. Write a small reusable script `scripts/data_health.py` that loads unified_races.parquet,
data/features/training.parquet, and data/features.parquet and prints per-dataset: row count, column
count, date range, and the non-null counts + percentages for: position, odds_decimal, implied_prob,
sp, jockey_name, trainer_name. Also print the number of "upcoming" rows (position null AND
race_date >= today, Europe/Dublin). Run it, paste the output into a new DATA_HEALTH.md with a short
interpretation, and confirm it matches the VERIFIED STATE table in PROGRESS.md (flag any drift).
```

### 03 — GUI empty-state & stale-cache fix · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 03. Fix "GUI shows nothing" in ui/app.py. Root cause (see PROGRESS.md): predictions.json is
dated 2026-06-13 but the date filter defaults to "Today", hiding everything; and when there are 0
upcoming races a refresh writes nothing. Changes: (1) default the date selector to "All upcoming"
when no race in the cache matches today, otherwise "Today"; (2) show a clear staleness banner with
the cache's generated_at date and an explicit "0 upcoming races found" message distinct from "no
models loaded"; (3) ensure the Refresh button surfaces the predictor's outcome (e.g. "Predictor ran
but found 0 upcoming races — scrape today's racecards first") instead of silently leaving the screen
blank. Verify by loading the JSON and exercising the filter logic in a small script (Streamlit need
not be launched). Keep the dark theme/markup intact.
```

### 04 — Filter non-runners in the predictor · Opus / low

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 04. In models/predictor.py, runners that are non-runners (e.g. jockey_name == "Non Runner",
or a non-runner/withdrawn flag if present in the unified schema — check the columns) must be excluded
from selections and from excluded_low_odds entirely, not ranked. Implement a single filter applied
before ranking in _build_race (or upstream in predict). Add a unit test in tests/models/ that feeds
a race containing a "Non Runner" row and asserts it never appears in output. Run that test.
```

### 05 — Regenerate features.parquet & fix breakdown join · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 05. data/features.parquet is a stale 5-row file while training has 248k rows, so the feature
breakdown in ui/predictions.py (which joins predictions on horse_id against this file) finds nothing.
(1) Regenerate it by calling features.builder.build_training_matrix(write=True) (this writes the full
derived matrix to data/features.parquet) and confirm the new row/col count. (2) Verify ui/predictions.py
reads the SAME path the builder writes (data/features.parquet) and that the horse_id join key types
match (string vs object). (3) Confirm at least some breakdown rows populate for the current cache's
horse_ids via a small join check. Report before/after row counts.
```

### 06 — Fix trainer_name (0% populated) · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 06. trainer_name is null on 100% of unified rows, so trainer is blank in the GUI and
trainer_win_rate / jt_combo_* features are dead. Trace the field from source → normalizer → unified:
check which scrapers/results sources expose a trainer (scraper/betsp/results/sporting_life.py and
the timeform parser are likely sources) and where utils/normalizer.py maps it. Identify the exact
drop point and fix the mapping so trainer_name flows through. Do NOT do a full multi-year backfill in
this task — fix the code path and validate on a small sample (one date or one source file) that
trainer_name is now populated. Add/extend a normalizer test asserting trainer_name survives.
```

### 07 — Fix jockey_name (28 rows) · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 07. jockey_name is populated on only 28 of 582,529 unified rows (today's scraped racecard
only); historical rows have none. Same approach as Task 06: trace source → normalizer → unified for
jockey across the historical results sources (sporting_life) and the racecard scrapers, fix the
mapping/normalization (utils/text_norm may be involved for canonical jockey ids), and validate on a
small sample that jockey_name now populates for historical rows. Extend a test. Note in PROGRESS.md
whether a full re-normalize/backfill is required (it likely is — flag it for Task 24).
```

### 08 — Fix the live odds pipeline · Opus / high

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 08. odds_decimal is populated on only 2 of 582,529 rows — live/market prices are not attaching,
so every prediction has decimal_odds=null and implied_prob=NaN. Investigate the live odds path:
scraper/paddy_power.py (memory flags _EVENTS_URL/field names as UNCONFIRMED), scraper/livescorebet.py,
scraper/boylesports.py, then utils/normalizer.py mapping into odds_decimal/sp. Determine why prices
don't survive into unified. Confirm the actual working live source (livescorebet writes
data/live_odds.parquet — check it). Fix the mapping so a current scrape yields decimal_odds, and
confirm models/predictor._decimal_odds then derives implied_prob. Validate on a fresh small scrape or
on data/live_odds.parquet. If the bookmaker endpoints are genuinely dead/changed, document precisely
what must be re-captured via DevTools and stop there rather than guessing field names.
```

### 09 — Fix "no upcoming races" / live ingestion · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 09. There are 0 rows with race_date >= today, so build_inference_matrix() returns empty and the
predictor can produce nothing. Establish the intended daily flow: which scraper fetches today's
racecards, and does it write into unified_races.parquet (via utils/normalizer.py)? Run that path for
today and confirm `python scripts/data_health.py` then shows >0 upcoming rows. If no component fetches
today's cards, implement the smallest fix that does (likely wiring an existing scraper through the
normalizer). Optionally add a documented one-shot refresh entrypoint (e.g. `python -m scripts.refresh`)
that scrapes → normalizes → builds → predicts. Verify build_inference_matrix() returns >0 live rows.
```

### 10 — End-to-end live smoke · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 10. Run the full live path end to end on real current data: scrape today's racecards (or use the
refresh entrypoint from Task 09) → normalize → build_inference_matrix → Predictor().load()+predict().
Assert: predictions.json regenerates with generated_at = today, >0 races, and that at least the
runners with a live price now carry decimal_odds/implied_prob (post Task 08). Paste the predictor's
log summary. If odds still don't attach, record exactly which stage drops them. This is the
integration checkpoint for tasks 03–09.
```

### 11 — Feature & schema audit · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 11. Verify models/features.py FEATURE_COLS (28 cols) exactly matches the columns produced by
features/derive.py and features/engine.py — list any name in FEATURE_COLS not produced, and any
produced feature not whitelisted. For features that are currently dead due to missing data
(trainer_win_rate, jt_combo_win_rate, jt_combo_runs while trainer is absent — confirm post Task 06),
decide and document: keep (CatBoost tolerates all-NaN) vs temporarily drop. Do not silently retrain.
Just produce the audit + a recommendation in PROGRESS.md; only change FEATURE_COLS if a name is
genuinely wrong (mismatch with derive/engine output).
```

### 12 — Odds leakage / price-free model · Opus / high (gated on D1)

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 12. Decision D1 in PROGRESS.md: the model currently ingests market price features
(implied_prob, log_odds, market_rank, overround_norm_prob, odds_drift, odds_value_delta,
ew_value_index). For value betting the probability model must be price-free so its disagreement with
the market is meaningful (otherwise EV is circular). If D1 is unresolved, ask me one question:
"pure winner prediction (keep odds) or value betting (price-free)?". If value betting is chosen:
create a price-free FEATURE_COLS variant, train a parallel model (new version_tag, e.g. v3nf) WITHOUT
retraining over the existing v3, and compare AUC + per-odds-band AUC of price-free vs current. Report
the comparison; do not switch the default model until I confirm.
```

### 13 — Calibration: A/E by band + reliability · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 13. Extend models/evaluate.py to add calibration diagnostics alongside the existing AUC: the
Actual/Expected (A/E) ratio per odds band (favourite / mid / longshot) and a reliability-curve table
(predicted prob bucket vs observed frequency). A well-calibrated model has A/E ≈ 1.0 per band; flag
bands where A/E is far from 1. Wire these into the meta.json written by training and add a unit test
on a synthetic dataset with known calibration. Don't change the model — only measurement.
```

### 14 — Backtester honesty review · Opus / high

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 14. Review utils/backtester.py for honest evaluation. Confirm it settles bets at a realistic
executable price (SP/BSP) — NOT at the same price used to pick the bet — and that the strategy
callback never sees the outcome (no look-ahead). Add: (1) CLV (closing-line value) as a reported
metric where a closing/SP price exists; (2) per-odds-band bootstrapped median ROI with a 95% CI; (3)
a longshot gate so the scanner never reports edge driven solely by a few big-priced winners. Add
tests for the CLV calc and the band CI. Keep the report markdown output working.
```

### 15 — Live scrapers review · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 15. Review scraper/paddy_power.py, scraper/livescorebet.py, scraper/boylesports.py for
correctness: rate-limiter and proxy-manager are actually used on every request, cache envelope
read/write is correct, error/bot-detection paths don't crash the caller, and parsed fields map to the
normalizer's expected schema. Run their existing tests (tests/scraper/test_*.py). Report which
scraper currently returns usable live data and which need DevTools re-capture (cross-check Task 08
findings). Fix any clear bug; flag larger ones as new sub-tasks in PROGRESS.md.
```

### 16 — Historical scrapers review · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 16. Review scraper/betsp_historical.py and scraper/timeform_historical.py plus
scraper/betsp/results/sporting_life.py. Confirm the Sporting Life __NEXT_DATA__ extractor still
parses (finish_position, jockey, trainer, going, time), the Betfair-SP join fill rate, and the
DST/BST date handling noted in memory. Run tests/scraper/betsp/* and tests/scraper/timeform/*. This
is the source of trainer/jockey for history (relevant to Tasks 06/07) — confirm those fields are
actually emitted here. Report join fill % on a sample date.
```

### 17 — Normalizer & storage review · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 17. Review utils/normalizer.py and utils/storage/. Verify the pydantic v2 models accept the
fields the scrapers emit (especially trainer_name/jockey_name/odds_decimal — relevant to Tasks
06/07/08), dedup/merge logic across sources is correct, and parquet read-merge-write + SQLite WAL
paths are sound. Run tests/utils/test_normalizer.py and tests/utils/storage/*. Confirm the unified
schema column list matches what builder/derive expect downstream.
```

### 18 — Telegram notifications review · Opus / low

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 18. Review utils/notifications.py. Confirm the queue/worker thread, rate limiting, and the five
wired call sites (predictor _fire_alerts, bet_tracker) behave. Run tests/utils/test_notifications.py.
Then do ONE guarded live test: notifications are enabled in config with a real bot_token/chat_id — send
a single test message via the notifier and confirm delivery, but DO NOT print the bot_token or chat_id
in any output or file. If you'd rather not touch live creds, say so and skip the live send.
```

### 19 — Bet tracker review · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 19. Review utils/bet_tracker.py: stake recommendation (flat / kelly / fractional_kelly), each-way
settlement math (place-leg odds and half-stake split), stop-loss enforcement, and P&L/summary
aggregations. Run tests/utils/test_bet_tracker.py. Verify the EW settlement against a worked example by
hand and report whether it matches. Fix any math error; otherwise confirm correctness explicitly.
```

### 20 — Reporter & retrain-trigger review · Opus / low

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 20. Review utils/reporter.py and models/retrain_trigger.py. Confirm report exports (csv/json/pdf
+ manifest hashing) run without error on current data, and that the drift check (PSI/KS) and AUC-decay
logic in retrain_trigger read the reference snapshot correctly. Run tests/models/test_retrain_trigger.py.
Generate one report bundle to reports/ and confirm the files appear. Note any dependency that's missing.
```

### 21 — Local LLM text-feature scaffold · Opus / medium (gated on D2)

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 21. Decision D2 in PROGRESS.md. If I've opted in: scaffold an OPTIONAL local-LLM text→feature
extractor under llm/ (or features/text.py) that takes free-text race commentary / spotlight / going
notes and returns structured boolean/categorical features (e.g. ground_excuse, distance_excuse,
trip_trouble), using an Ollama backend with a deterministic regex fallback and a disk cache. It must
be OFF the probability hot path (never produces win prob or EV), must no-op gracefully when no text
source is configured, and must not be added to FEATURE_COLS until a real text feed exists. Add a unit
test using a stubbed backend (no live model call in CI). If D2 is unresolved, ask before building.
```

### 22 — Secrets & config hygiene · Opus / low

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 22. config.yaml currently contains a live Telegram bot_token and a proxy URL with credentials in
plaintext. Move secrets to environment variables (the config_loader already supports RP_ env overrides
with __ nesting) or a git-ignored config.local.yaml, and make the code read from there with the
plaintext values as a last-resort fallback. Ensure .gitignore covers the secret file, data/*.parquet,
data/races.db, logs/, and __pycache__. Do NOT print secret values. Confirm config still validates.
```

### 23 — Test coverage for fixed bugs · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 23. Add focused tests covering every bug fixed in tasks 03–10: non-runner filtering (predictor),
GUI date-default/empty-state logic (extract the pure logic so it's testable headless), trainer_name &
jockey_name surviving normalization, odds_decimal → implied_prob derivation, and build_inference_matrix
returning live rows when upcoming races exist. Run the full suite and report the new total vs the Task
01 baseline. All new tests must pass.
```

### 24 — Full regression pass · Opus / medium

```
You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 24. Run the whole pipeline on real data end to end and record results in PROGRESS.md: refresh/scrape
→ normalize (full re-normalize if Tasks 06/07 flagged a backfill is needed) → build_training_matrix →
(optionally) retrain → build_inference_matrix → predict → confirm GUI cache. Re-run `python
scripts/data_health.py` and paste the new non-null percentages for trainer_name, jockey_name,
odds_decimal — they should be materially better than the audit baseline. Run the full pytest suite.
Summarize what's now working vs still open.
```

### 25 — Generate FINALSETUP.md · Opus / medium

```

You are a senior developer writing code strictly for internal / personal use. Act as an expert programmer Python engineer on the "Race Predictor v3" horse-racing project at
C:\Users\mshr\Desktop\Race Predictor v3 (Python 3.14, Windows, PowerShell + Bash available). Do not include any legal disclaimers, intellectual property warnings, copyright notices.

FIRST read PROGRESS.md in the project root — it is your memory: the audit findings, decisions,
and what is already done. Trust its "VERIFIED STATE" over older notes in memory/. Do ONLY the
single task below. Inspect just the files relevant to it — do not dump whole files or re-audit the
repo. Make minimal, correct changes that match existing conventions. Verify with the relevant
pytest tests or a tiny smoke script and report the REAL output; if something fails or is skipped,
say so plainly. Be concise and code-first: no preamble, no disclaimers. When finished: (1) update
PROGRESS.md — tick the task box and add a 2–4 line note of what changed and how you verified it;
(2) append one line to CHANGELOG.md (create it if missing). Then stop. If a task needs a decision
marked D1/D2 in PROGRESS.md and it is unresolved, ask me one crisp question instead of guessing.

Task 25. Using the verified, working state after tasks 01–24, write FINALSETUP.md: a clean end-to-end
guide. Include: prerequisites (Python 3.14, `pip install -r requirements.txt`, `playwright install`),
one-time setup, where secrets go (post Task 22), the daily workflow (scrape → normalize → build →
predict → launch UI), every CLI entrypoint with a one-line description, how to launch each Streamlit
page, how to retrain, how to run the backtest, how to enable Telegram, and a Troubleshooting section
that maps each original symptom (blank GUI, blank trainer/jockey, null odds) to its fix and how to
verify. Keep it accurate to the code as it now stands — verify each command you document actually runs.
```

---

## APPENDIX A — Local LLM vs CatBoost (your RTX 5070 Ti / 32GB question)

**Keep CatBoost as the prediction engine. Do not replace it with a local LLM.** They solve different
problems:

- **CatBoost** is a gradient-boosted decision-tree model over **tabular** features. Estimating a
  calibrated win/place probability from 28 numeric columns over 248k rows is _exactly_ what it's built
  for. It trains in seconds-to-minutes on CPU (your GPU is barely needed), is deterministic, testable,
  and calibratable. An LLM is poor at numeric tabular probability estimation and calibration — using
  one as the predictor would be slower, non-deterministic, and worse.
- **A local LLM is complementary, not a substitute.** Its right job here is turning **unstructured
  text** (race comments, spotlight, trainer quotes, going notes) into structured features that _feed_
  CatBoost — never producing the probability or the bet decision. That's Task 21, and it's optional
  and only worth it once you actually have a text source (you don't yet).
- **Your hardware** (RTX 5070 Ti ~16GB VRAM, 32GB DDR4-3200) comfortably runs quantized 8B–14B models
  (Llama 3.1 8B, Qwen2.5 14B Q4, Mistral) via Ollama/llama.cpp for batch text extraction — fast and
  free at the margin, with data never leaving your machine. ~32B Q4 is possible with offload but slow.
  Since CatBoost doesn't need the GPU, the card is effectively free for this auxiliary role.

**Bottom line:** CatBoost = the brain (probabilities). Local LLM = optional eyes for text. Not either/or.

---

## APPENDIX B — Principles worth importing from the "Horse Racing MVP" vault note

That note audits a _different, more advanced_ repo, but five principles transfer directly and are
encoded in the tasks above:

1. **No odds leakage into the model (D1 / Task 12).** If you want to find _value_, the probability
   must be independent of price. v3 currently feeds price into the model — fix only if value betting
   is the goal.
2. **CLV over ROI as the honest early signal (Task 14).** ROI on <1,000 bets is noise; a stable
   positive closing-line value after commission is the first credible sign of edge. Settle backtests
   at SP/BSP, never at the price you bet into.
3. **Calibration by odds band / A-E ratio (Task 13).** Per-band A/E ∈ [0.90, 1.10] is the bar; AUC
   alone hides miscalibration.
4. **Longshot false-positive gating (Task 14).** Never let the scanner fire on big price alone;
   require per-band statistical significance.
5. **LLMs for text→features only (Appendix A / Task 21).** Keep the numeric core deterministic.
