# Race Predictor v3

A horse-racing win/place probability model and paper-betting dashboard for UK & Irish
racing. It scrapes racecards and historical results, derives point-in-time features,
trains calibrated CatBoost models, and serves ranked selections plus value bets through a
Streamlit UI. Personal/internal use; not financial advice.

- **Python:** 3.14 · **OS:** Windows (PowerShell; Bash also available) · **Timezone:** Europe/Dublin
- **Probability engine:** two model lines run side by side —
  **CatBoost** (per-target gradient-boosted trees) and **LightGBM v3** (a softmax
  win model, `models/lgbm_softmax.py` + `models/train_lgbm.py`). Both are judged on
  one leak-free holdout (`backtest/holdout.py`) against the de-vigged market.
- **UI:** Streamlit (paper betting only — no real money is staked). Includes a
  **Model Honesty** GO/NO-GO panel and a **Model Compare** head-to-head page.

---

## Pipeline

```
scraper/  ──▶  data/unified_races.parquet  ──▶  features/  ──▶  data/features/training.parquet
(live odds +     (normalized, all sources       (derive +        (+ data/features.parquet for the
 3yr history)     merged & deduped)              engine)           live UI breakdown)
                                                                        │
                                                                        ▼
              data/predictions.json  ◀──  models/predictor.py  ◀──  models/  (CatBoost *.bin + *_meta.json)
              (ranked picks + value bets)                                │
                                                                        ▼
                                                                  ui/ (Streamlit)
```

1. **Scrape** (`scraper/`) — live odds (`livescorebet` = verified racecard source;
   `boylesports`, `paddy_power` = best-effort, Cloudflare-gated) write
   `data/live_odds.parquet`; historical results (`betsp_historical`, `timeform_historical`,
   Sporting Life `__NEXT_DATA__` parser) populate `data/betsp.parquet`.
2. **Normalize** (`utils/normalizer.py`) — merges every source into
   `data/unified_races.parquet`. Rows with a null `position` and `race_date >= today` are
   "upcoming"; everything else is settled history.
3. **Features** (`features/derive.py`, `features/engine.py`, `features/builder.py`) —
   build point-in-time features (trailing win rates, going preference, ratings ranks,
   market features). `build_training_matrix()` writes the labelled training set;
   `build_inference_matrix()` derives today's live runners.
4. **Models** (`models/train.py`) — trains one CatBoost model per target
   (`won`, `placed_2`, `showed`) with optional Optuna tuning, isotonic calibration, and
   calibration diagnostics (A/E by odds band + reliability table) saved to `*_meta.json`.
   Two model lines exist: **`v3`** (priced, shown in the UI) and **`v3nf`** (price-free,
   well-calibrated, drives the value layer).
5. **Predict** (`models/predictor.py`) — scores upcoming races, filters non-runners,
   attaches per-bookmaker odds (best price wins), computes value/EV from the price-free
   model, and writes `data/predictions.json`.
6. **UI** (`ui/`) — Streamlit dashboard: predictions, live cards, paper betting,
   performance/bankroll, settings.

---

## The LightGBM v3 line + the honesty machinery

Alongside CatBoost, v4 trains a second, independent win model and judges **both**
on the one test that cannot flatter itself — a leak-free holdout against the
_de-vigged_ market.

- **`models/lgbm_softmax.py`** — a LightGBM softmax (`multiclassova`) win model
  that learns one-runner-per-race winners, then normalises within race to a
  proper per-race probability distribution.
- **`models/train_lgbm.py`** — trains the v3 line on a frozen window, scores its
  own most-recent ~3-week holdout through the shared GO/NO-GO harness, and writes
  `models/lgbm_won_v3.txt` + `models/lgbm_v3_meta.json` (which carries the
  `verdict` block).
- **`models/devig.py`** — strips the bookmaker overround out of a price book
  (`proportional` / `power` / `shin`) so the market benchmark is a _fair_,
  margin-free line — the only honest thing to beat.
- **`models/head_to_head.py`** — scores model vs de-vigged market on log-loss /
  Brier / ECE per odds band; `model_beats_market_logloss` is the headline gate.
- **`backtest/integrity.py`** — the suite that tries to _disprove_ an edge:
  look-ahead bias, course/season/band cherry-picking, CLV leakage, liquidity and
  minimum-backtest-length checks (OK / WARN / FAIL).
- **`backtest/holdout.py`** — the GO/NO-GO orchestrator. Loads a **frozen** model,
  scores the holdout window, runs head-to-head → EV>0 simulation (settled at the
  **pre-off** price, CLV measured against the closing line) → integrity suite, and
  writes `data/backtests/holdout_*/summary.json` + `ledger.csv`. A leakage guard
  hard-fails if the holdout starts on/before the model's train cutoff.

  ```powershell
  python -m backtest.holdout --model models/catboost_won_v3nf.bin \
      --holdout-start 2026-05-23 --holdout-end 2026-06-12
  ```

- **UI honesty pages** — `ui/model_honesty.py` (page `ui/pages/14_Model_Honesty.py`)
  renders the per-line **GO / NO-GO banner**, the **CLV** read, the integrity
  status list and the per-band calibration table. `ui/model_compare.py` (page
  `ui/pages/15_Model_Compare.py`) puts CatBoost vs LightGBM vs the market on one
  scorecard + reliability diagram. Both modules keep a Streamlit-free
  data/formatter layer so they unit-test headless.

> **Honesty rule baked in:** beating the market on log-loss is the gate, but a
> positive verdict paired with **negative CLV is paper-only, never a green light**
> to bet — the v3 holdout does exactly this (GO on log-loss, CLV ≈ −12%). The
> panels say so explicitly.

### Full rebuild — "use my full PC"

`pipeline_full.py` is the one-shot orchestrator: normalise → build features →
train **CatBoost (GPU) and LightGBM (CPU) concurrently** → CatBoost holdout
GO/NO-GO → refresh `predictions.json` → print a timing table + both verdicts.

```powershell
python pipeline_full.py                       # full cold rebuild (scrapes first)
python pipeline_full.py --skip-scrape --no-tune   # rebuild from existing parquet, fast
python pipeline_full.py --resume              # skip any stage whose output is up to date
python pipeline_full.py --force               # re-run every stage regardless
```

It writes the additive `verdict` block into `data/predictions.json`, which both
honesty pages read.

---

## Setup

```powershell
python --version                 # expect 3.14.x
pip install -r requirements.txt  # streamlit, pandas, catboost, optuna, playwright, ...
pip install -r requirements-dev.txt   # optional: pytest, pytest-mock, respx, reportlab
playwright install               # one-time: browser binaries for scraper fallbacks

# one-time SQLite schema migration
python -m utils.storage migrate
python -m utils.storage version  # -> current=2 target=2
```

The repo ships trained models (`models/catboost_{won,placed_2,showed}_v3.bin` + meta) and
historical parquet, so a normal install can go straight to the daily workflow. A full cold
start (no models/history) is `python pipeline.py --skip-scrape` (add network scrape by
dropping the flag — slow, Cloudflare-gated).

### Secrets

Real credentials never live in the committed `config.yaml`. Put them in **either**:

- **`config.local.yaml`** (git-ignored, deep-merged over `config.yaml`) — recommended.
- **`RP_`-prefixed env vars**, `__` descends one level (these win over the local file).

```yaml
# config.local.yaml
proxy_pool:
  proxies: ["http://LOGIN:PASSWORD@gw.dataimpulse.com:823"]
timeform:
  session_cookie: "<timeform session cookie>" # unlocks ratings/pace features
notifications:
  channels:
    telegram: { bot_token: "<token>", chat_id: "<chat id>" }
```

Precedence: **env (`RP_*`) > `config.local.yaml` > `config.yaml`**.

---

## Running each stage

```powershell
# Daily flow: scrape live -> normalize -> build -> predict -> data/predictions.json
python -m scripts.refresh
python -m scripts.refresh --no-scrape   # re-normalize + predict only (skip scraping)

# Cold start: migrate -> (scrape 3yr) -> normalize -> build training matrix -> train
python pipeline.py [--skip-scrape] [--no-tune]

# Health snapshot (rows, dates, fill rates, upcoming count)
python scripts/data_health.py

# Train / retrain
python -m models.train                 # all targets, with Optuna tuning (slow)
python -m models.train --no-tune       # fast, no hyperparameter search
python -m models.retrain_trigger [--check-only|--force]   # PSI/KS drift-gated retrain

# Predict only
python -m models.predictor [--min-odds 2.5]

# Backtest (honest: settles at SP/BSP, reports CLV + per-band ROI CI + longshot gate)
python -m utils.backtester [--strategy flat|fractional_kelly] [--bet-type win|each_way]

# Reports bundle (csv/json/pdf + manifest) to reports/
python -m utils.reporter --format all
```

---

## Running the UI

```powershell
streamlit run ui/app.py     # main dashboard (predictions, refresh, filters, per-book odds)
```

Default URL `http://localhost:8501`. Other standalone pages:

| Page                                                | Purpose                                                                       |
| --------------------------------------------------- | ----------------------------------------------------------------------------- |
| `ui/app.py`                                         | Main dashboard: predictions grouped by venue/race, refresh, filters           |
| `ui/live_races.py`                                  | Live race cards with status (Upcoming/Starting/In-Play/Finished)              |
| `ui/bet_placer.py`                                  | Record a paper bet (stake, win/each-way)                                      |
| `ui/performance_dashboard.py`                       | Bankroll curve, P&L, history, model metrics                                   |
| `ui/race_compare.py`                                | Head-to-head comparison of two selections                                     |
| `pages/11_Todays_Suggestions.py`                    | Curated value picks (Strong/Lean) + one-click paper bet                       |
| `pages/13_Yesterdays_Bet_Predictor.py`              | Settle the model's picks on a past day vs real results (profit check)         |
| `pages/14_Model_Honesty.py`                         | GO/NO-GO vs the de-vigged market: verdict banner, CLV, integrity, calibration |
| `pages/15_Model_Compare.py`                         | CatBoost vs LightGBM vs market: scorecard + reliability diagram               |
| `ui/settings.py`                                    | Edit proxy/scrape/model/retrain config                                        |

---

## Paper betting

Betting is **simulated** via `utils/bet_tracker.py` — no real wagers are placed.

- **Staking:** flat, full Kelly, or fractional Kelly (`f = (b·p − q)/b · bankroll`,
  fractional = full × `kelly_fraction`); all guarded against degenerate odds/probabilities.
- **Each-way settlement:** half-stake per leg; win leg pays `half·odds`, place leg pays
  `half·((odds−1)·ew_fraction + 1)` — standard UK/IE math.
- **Stop-loss:** blocks new bets and fires a notification when bankroll falls below
  `initial·(1 − pct)`.
- Record bets in `ui/bet_placer.py`; track bankroll/ROI/drawdown in the performance pages.
- Optional **Telegram** alerts (`utils/notifications.py`): `race_soon`, `new_top_pick`,
  `odds_drop`, `stop_loss`, `bet_outcome`. Enable by setting `bot_token`/`chat_id` (see
  Secrets). Sends are rate-limited on a daemon thread; failures are logged, never fatal.

---

## Tests

```powershell
python -m pytest -q
```

Baseline: **1082 passed, 3 skipped** (~80s). The 3 skips are the optional `reportlab` PDF
export path (`pip install reportlab` to clear them). Coverage of `models/`+`features/`+
`utils/` is ~84%.

---

## Status & known limits

See `PROGRESS.md` for the full repair/hardening history and what's still open. Highlights:

- **The headline "Win %" is the within-race-normalised probability, not the raw model
  marginal.** The raw `v3` `won` calibrator was fit on a narrow historical slice and
  **saturates out-of-sample** (live ECE ≈0.22; the field piled near a ~0.73 ceiling), so
  the UI/API/backtest present `won_prob_normalized` (sum-to-1 per race ⇒ field mean = base
  rate; OOS ECE **0.029**, AUC **0.778**). The raw `won_prob` is retained only as a
  debug / EV-reference column. See `memory/calib-fl-01-headline.md`.
- The old `odds_finish` market-feature leak is **fixed** (commit `3eca232`): market
  features now build from the pre-off `morningwap` (`implied_prob == 1/morningwap`,
  verified empirically in the Stage-4 audit). `v3nf` (price-free) still drives the value
  layer, suggestions, and the Yesterday's Bet Predictor. One feature defect remains
  (`race_complexity` uses dataset-global z-scores — excluded from audit-fitted models);
  see `reports/calibration_audit_20260727.md`.
- **Stage-4 model verdict (2026-07-27): NO-GO — paper-only.** On an untouched
  2026-06-13→07-25 window (904 races) the price-free model loses to the de-vigged
  pre-off market by 0.186 race log-loss (95% CI [−0.225, −0.146]); no model line
  significantly beats the best (shin) de-vig baseline; CLV is −13.4%; every EV>0
  portfolio loses at the pre-off price. The market-adjusted `value_win_prob` is ~93%
  correlated with the offered price — treat it as a market blend, and read the
  price-free line from the new `value_win_prob_independent` field.
- **The value layer is now calibrated but not yet proven profitable at the executable
  price.** The price-free win prob is favourite-longshot recalibrated against the pre-off
  price (`models/fl_oddsband_v3nf_calib.pkl`), which flattens Actual/Expected across odds
  bands (was 2.80 odds-on … 0.17 longshot; now ~0.94 … 0.78) and lifts backtested yield.
  The value band is tightened to decimal odds **[2.0, 4.0]** — the regime with the
  least-negative closing-line value. **But CLV stays negative in every gate (best ~−4%)**
  on the walk-forward frame, and the last-settled-week strategies still lose when settled
  at SP. Calibration ≠ closing-line value, and SP settlement is conservative (the live
  layer takes the best board price, typically more generous than SP/ppwap), so treat all
  picks as **provisional / paper-only** until live CLV (best board price vs eventual SP)
  is measured. See `memory/calib-fl-04-backtest.md`.
- **Results lag the calendar.** Live racecards are scraped daily, but finishing positions
  arrive from the historical feeds a few days later — so backtests/settlement (incl. the
  Yesterday's Bet Predictor) settle the _most recent day that has results_, and only at the
  starting price (SP), since board prices exist for upcoming races only.
- Live racecards from the odds feeds carry **no jockey/trainer** (source limitation), so
  those features are sparse at inference time.
- `boylesports`/`paddy_power` are Cloudflare-gated and best-effort; `livescorebet` is the
  reliable live source.

For a deeper operations guide (troubleshooting symptom→fix tables, retrain/backtest
details, Telegram setup) see `FINALSETUP.md`.
