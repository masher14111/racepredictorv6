# Race Predictor v3 — Final Setup & Operations Guide

End-to-end guide for the working state after repair tasks 01–24. Every command
below was run against the live repo (`--help`/smoke-verified) on 2026-06-14.

- Project root: `C:\Users\mshr\Documents\Race Predictor v4\Race Predictor v4`
- Python: **3.14** · OS: Windows · Shell: PowerShell (Bash also available)
- Timezone: **Europe/Dublin** (all race times are localised through `utils.timezone`)
- Launch UI: `streamlit run ui/app.py`
- Tests: `python -m pytest -q` (baseline: **658 passed, 3 skipped** — the 3 skips are
  the optional `reportlab` PDF path)

---

## 1. Prerequisites

```powershell
# from the project root
python --version                 # expect 3.14.x
pip install -r requirements.txt  # installs streamlit, pandas, catboost, optuna, playwright, etc.
playwright install               # one-time: download the browser binaries used by scraper fallbacks
```

`requirements.txt` includes `reportlab` (PDF reports) and `matplotlib` (charts). If
`reportlab` is missing the pipeline still runs — PDF export is skipped with a warning
and 3 tests skip. Everything else is required.

A virtual environment is recommended but optional (personal/internal use):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install
```

---

## 2. One-time setup

```powershell
# 1. Initialise / migrate the SQLite schema (data/races.db)
python -m utils.storage migrate
python -m utils.storage version      # -> current=2 target=2

# 2. Put secrets in place (see §3)

# 3. Confirm config loads and validates
python -c "from utils.config_loader import get_config; print('config OK', bool(get_config()))"

# 4. (only if you have no trained models / no historical parquet)
#    Full cold-start build. Skips network scrape if parquets already exist.
python pipeline.py --skip-scrape    # normalize -> build training matrix -> train CatBoost
#    Cold-start WITH a fresh 3-year historical scrape (slow, Cloudflare-gated):
# python pipeline.py
```

The repo already ships trained models (`models/catboost_{won,placed_2,showed}_v3.bin`

- `*_meta.json`) and historical parquet, so a normal install only needs §1–§3 and then
  the daily workflow in §4.

---

## 3. Where secrets go (post Task 22)

Real credentials never live in the committed `config.yaml`. They go in **either** of:

1. **`config.local.yaml`** (git-ignored) — deep-merged over `config.yaml` at load time.
   This is the recommended place. Structure mirrors `config.yaml`:

   ```yaml
   proxy_pool:
     proxies:
       - http://LOGIN:PASSWORD@gw.dataimpulse.com:823
   timeform:
     session_cookie: "<timeform session cookie>" # unlocks TFR / pace ratings
   notifications:
     channels:
       telegram:
         bot_token: "<telegram bot token>"
         chat_id: "<telegram chat id>"
   ```

2. **`RP_`-prefixed environment variables** — `__` (double underscore) descends one
   config level. These win over `config.local.yaml`, which wins over `config.yaml`.

   ```powershell
   $env:RP_NOTIFICATIONS__CHANNELS__TELEGRAM__BOT_TOKEN = "<token>"
   $env:RP_NOTIFICATIONS__CHANNELS__TELEGRAM__CHAT_ID   = "<chat id>"
   ```

Precedence: **env (`RP_*`) > `config.local.yaml` > `config.yaml`**. On load you will
see `Merged secrets overlay from config.local.yaml` in the log when the overlay exists.
`config.local.yaml`, `.env`, and `*.local.yaml` are all in `.gitignore` — never commit
them.

---

## 4. Daily workflow

The single command that keeps everything current (scrape → normalize → build → predict):

```powershell
python -m scripts.refresh        # full daily flow
# python -m scripts.refresh --no-scrape   # re-normalize + predict only (skip scraping)
```

What it does, in order:

1. **scrape** — live odds scrapers (`livescorebet` = verified racecard source;
   `boylesports`, `paddy_power` = best-effort) write `data/live_odds.parquet`. Each
   scraper is wrapped so one dead bookie never blocks the run.
2. **normalize** — `utils.normalizer.normalize(write=True)` merges all sources into
   `data/unified_races.parquet` (race_date derived from race_time, `position` null ⇒ "upcoming").
3. **build** — `features.builder.build_inference_matrix()` derives today's live runners.
4. **predict** — `models.predictor` scores them and writes `data/predictions.json`.

Then launch the dashboard:

```powershell
streamlit run ui/app.py
```

Health check at any time:

```powershell
python scripts/data_health.py    # row/col counts, date range, fill rates, upcoming count
```

---

## 5. CLI entrypoints

| Command                                                                                                                                                                                                          | What it does                                                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python -m scripts.refresh [--no-scrape]`                                                                                                                                                                        | **Daily flow:** scrape live → normalize → build → predict → `data/predictions.json`.                                                              |
| `python pipeline.py [--skip-scrape] [--no-tune]`                                                                                                                                                                 | **Cold start:** migrate DB → (scrape 3yr history) → normalize → build training matrix → train CatBoost.                                           |
| `python scripts/data_health.py`                                                                                                                                                                                  | Data-health snapshot (rows, dates, fill rates, upcoming count) for the core datasets.                                                             |
| `python -m utils.storage [migrate\|version]`                                                                                                                                                                     | Run / inspect SQLite schema migrations (`data/races.db`).                                                                                         |
| `python -m utils.normalizer` _(module API)_                                                                                                                                                                      | `normalize(write=True)` merges sources → `unified_races.parquet` (called by refresh/pipeline).                                                    |
| `python -m models.train [--targets won placed_2 showed] [--trials N] [--no-tune] [--model-dir PATH]`                                                                                                             | Train/retune the CatBoost models for each target; writes `*.bin` + `*_meta.json`.                                                                 |
| `python -m models.predictor [--model-dir PATH] [--min-odds 2.5]`                                                                                                                                                 | Score upcoming races → `data/predictions.json`.                                                                                                   |
| `python -m models.retrain_trigger [--force] [--check-only] [--model-dir PATH]`                                                                                                                                   | PSI/KS drift check; archives + retrains when triggered. Writes `models/drift_report.json`.                                                        |
| `python -m utils.backtester [--strategy flat\|fractional_kelly] [--stake EUR] [--kelly F] [--bankroll EUR] [--min-odds N] [--top-n N] [--bet-type win\|each_way] [--split F] [--model-dir PATH] [--output PATH]` | Walk-forward backtest → markdown report (default `reports/backtest.md`).                                                                          |
| `python -m utils.reporter [--output-dir DIR] [--format csv\|json\|pdf\|all]`                                                                                                                                     | Export predictions/bets/performance bundle (+ optional PDF) and a `manifest.json` to `reports/`.                                                  |
| `python -m scripts.backfill_betsp_from_raw [--limit-days N]`                                                                                                                                                     | Offline replay of stored raw results to repopulate `trainer_name`/`jockey_name` in `betsp.parquet` (see Troubleshooting).                         |
| `python -m scripts.compare_pricefree`                                                                                                                                                                            | Task-12 diagnostic: trains a price-free (`v3nf`) variant vs full-feature control and prints per-band AUC. Does **not** change the deployed model. |

---

## 6. Streamlit pages

Each page is a standalone Streamlit app — launch the one you want:

```powershell
streamlit run ui/app.py                    # main dashboard (predictions + refresh + filters)
streamlit run ui/live_races.py             # live race cards with status (Upcoming/Starting/In-Play/Finished)
streamlit run ui/race_compare.py           # head-to-head comparison of two selections
streamlit run ui/bet_placer.py             # record a bet (stake, win/each-way) via BetTracker
streamlit run ui/performance_dashboard.py  # interactive P&L/ROI dashboard with filters + PDF export
streamlit run ui/settings.py               # edit proxy/scrape/model/retrain config; writes config.yaml
```

Default URL: `http://localhost:8501` (use `--server.port` to run several at once).

---

## 7. Retraining

```powershell
# Full retrain across all targets (with Optuna tuning — slow):
python -m models.train

# Fast retrain, no hyperparameter search:
python -m models.train --no-tune

# Drift-gated retrain (only retrains if PSI/KS thresholds are exceeded):
python -m models.retrain_trigger
python -m models.retrain_trigger --check-only   # report drift only, no retrain
python -m models.retrain_trigger --force        # always retrain
```

`train()` rebuilds the training matrix from `unified_races.parquet`, writes
`models/catboost_<target>_v3.bin` + `<target>_v3_meta.json` (AUC, calibration A/E by
odds band, reliability table — Task 13), and saves a reference feature snapshot used by
the drift trigger. `retrain_trigger` archives the prior models under
`models/archive/<timestamp>/` (keeps the last `retrain.archive_keep`).

> **Note:** the deployed `v3` models were trained while trainer/jockey features were
> dead. After the Task 06/07 backfill those features are now live in the matrix, so a
> fresh `python -m models.train` will let the model actually use that signal. See D1 in
> `PROGRESS.md` before deciding price-free vs full-feature (`scripts.compare_pricefree`).

---

## 8. Backtest

```powershell
python -m utils.backtester                                   # flat €10, min-odds 2.5, top-3, win
python -m utils.backtester --strategy fractional_kelly --kelly 0.25 --bankroll 1000
python -m utils.backtester --bet-type each_way --min-odds 8 --output reports/backtest.md
```

Honesty guarantees baked in (Task 14): no look-ahead in scoring; bets are **selected/staked
at the board price but settled at SP/BSP**; reports CLV, per-odds-band bootstrapped median
ROI with 95% CI, and a longshot-robustness gate. Output is a markdown report
(default `reports/backtest.md`).

> CLV shows as "unavailable" until an independent live board price (`odds_decimal`) is on
> disk alongside the SP — historical rows settle at `odds_finish`. This is by design, not a bug.

---

## 9. Enabling Telegram notifications

1. Create a bot via @BotFather, get the **bot token**, and your numeric **chat_id**.
2. Put them in `config.local.yaml` (or `RP_` env vars) — see §3.
3. In `config.yaml`, `notifications.enabled` and `channels.telegram.enabled` are already
   `true`; the channel only activates once both `bot_token` **and** `chat_id` resolve.
4. Quick self-test (sends one real message):

   ```powershell
   python -c "from utils.notifications import notify_race_soon; notify_race_soon('Test', 'Self-test', 5); import time; time.sleep(2)"
   ```

Wired alerts (Task 18): `race_soon` / `new_top_pick` / `odds_drop` (predictor) and
`stop_loss` / `bet_outcome` (bet tracker). Sends are rate-limited (~1.2 s apart) on a
daemon worker thread; failures are logged, never fatal. Tune timing under
`notifications.events` in `config.yaml`.

---

## 10. Troubleshooting (original symptoms → fix → verify)

### Blank GUI / "no predictions load"

- **Cause:** (a) no upcoming races in `unified_races.parquet` for today, so the predictor
  produces nothing; (b) the UI used to default the date filter to _Today_, hiding a stale
  cache from a previous day.
- **Fix (Task 03):** `ui/app.py` now defaults to **"All upcoming"** when no cached race
  matches today, shows an explicit empty-state, and surfaces a staleness banner with the
  cache's `generated_at` date. The real remedy for empty data is to refresh:
  ```powershell
  python -m scripts.refresh
  ```
- **Verify:**
  ```powershell
  python -c "import json; d=json.load(open('data/predictions.json')); print(d['generated_at'], d['total_races'])"
  ```
  `generated_at` should be today and `total_races > 0`. Then `streamlit run ui/app.py`.

### Blank trainer / jockey everywhere

- **Cause:** the results parsers only ever extracted `*_id`, never the names; the on-disk
  `betsp.parquet`/`unified_races.parquet` predated the fix (`trainer_name` 0%,
  `jockey_name` 28 rows).
- **Fix (Tasks 06/07 + 24):** parsers/joiner/writer now carry `trainer_name`/`jockey_name`
  and derive the canonical ids from them. Existing on-disk data is repopulated **offline**
  from stored raw HTML (no re-scrape needed):
  ```powershell
  python -m scripts.backfill_betsp_from_raw          # rewrites betsp.parquet with names
  python -m scripts.refresh --no-scrape              # flow names into unified + predictions
  ```
- **Verify:**
  ```powershell
  python scripts/data_health.py    # trainer_name / jockey_name should read ~90% (was 0%)
  ```

### Null / missing odds

- **Cause:** `odds_decimal` is a **live board price** and was 2/582k because no real live
  scrape had ever been persisted; the file held only synthetic test rows.
- **Fix (Task 08):** the live scrape → normalize → derive chain is correct; running a real
  refresh attaches live decimals, and `derive.add_odds_features` produces `implied_prob`.
  Historical rows have no live board price by design — they carry `odds_finish`, which
  feeds `implied_prob` in the training matrix (100% filled).
  ```powershell
  python -m scripts.refresh        # real live scrape populates data/live_odds.parquet
  ```
- **Verify:** every selection in the cache carries `decimal_odds` and `implied_prob`:
  ```powershell
  python -c "import json; r=json.load(open('data/predictions.json'))['races']; s=[x for race in r for x in race['selections']]; print(len(s), 'selections;', sum('decimal_odds' in x and x.get('implied_prob') is not None for x in s), 'priced')"
  ```
  Note: `data_health.py` will still show `odds_decimal ≈ 0%` on the **training** matrix —
  that is expected (historical price lives in `odds_finish`/`implied_prob`, not the
  live-only `odds_decimal` column).

### "Non Runner" ranked as a pick

- **Fix (Task 04):** `models/predictor.py` drops non-runners (`jockey_name == "non runner"`,
  case/space-insensitive) before ranking, so they never appear in selections or field size.

### Feature-breakdown panel shows nothing

- **Cause:** stale 5-row `data/features.parquet`.
- **Fix (Task 05):** regenerated by the builder; `python -m scripts.refresh` keeps it
  current. Verify with `python scripts/data_health.py` (features.parquet row count > 5).

### Full sanity check

```powershell
python -m pytest -q              # expect 658 passed, 3 skipped (reportlab PDF)
```
