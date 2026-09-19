# CLAUDE.md

Guidance for Claude Code working in this repo. Read alongside `README.md` (user
docs) and `FINALSETUP.md` (operations). Personal/internal project — paper betting
only, no real money is ever staked.

## What this is

A horse-racing win/place probability model + Streamlit betting dashboard for
UK & Irish racing. Python 3.14, Windows/PowerShell, timezone Europe/Dublin.

## The two model lines

Two **independent** win models are trained and judged side by side against the
_de-vigged_ (margin-free) market — the only honest benchmark:

- **CatBoost** (`models/train.py`) — one model per target (`won`, `placed_2`,
  `showed`). Two variants: `v3` (priced, shown in UI) and `v3nf` (price-free,
  drives the value layer). Since commit `3eca232` the market features build from
  the **pre-off** price (`implied_prob == 1/morningwap`, verified empirically in
  the Stage-4 audit) — the old odds_finish look-ahead is fixed. The remaining
  known feature defect is `race_complexity` (dataset-global z-scores; see
  `memory/live-repair-04-calibration-audit.md`).
- **LightGBM v3** — a softmax win model:
  - `models/lgbm_softmax.py` — the `multiclassova` softmax model + within-race
    normalisation to a proper per-race distribution.
  - `models/train_lgbm.py` — trains on a frozen window, runs its own ~3-week
    holdout through the shared GO/NO-GO harness, writes `models/lgbm_won_v3.txt`
    - `models/lgbm_v3_meta.json` (carries the `verdict` block).

## The honesty / backtest stack

- `models/devig.py` — overround removal (`proportional` / `power` / `shin`).
- `models/head_to_head.py` — model vs de-vigged market on log-loss / Brier / ECE
  per odds band. `model_beats_market_logloss` is the headline GO/NO-GO gate.
- `backtest/integrity.py` — `run_all_integrity_checks`: look-ahead bias,
  course/season/band cherry-picking, CLV leakage, liquidity, min-backtest-length.
  Returns OK / WARN / FAIL rows.
- `backtest/holdout.py` — `run_holdout(...)`: loads a **frozen** model, scores the
  holdout window, runs head-to-head → EV>0 sim (settled at the **pre-off** price;
  CLV vs the closing line) → integrity → writes
  `data/backtests/holdout_*/summary.json` + `ledger.csv`. **Leakage guard**
  hard-fails if the holdout starts on/before the model's train cutoff;
  `evaluate_scored_window(...)` is the model-agnostic core both lines funnel through.
- `models/predict_unified.py` — additively enriches `data/predictions.json` with
  the top-level `verdict` block + per-runner `catboost_win_prob` / `market_prob` /
  `ev_catboost` (and `lgbm_*` when the model file is present). **Additive only** —
  never drops legacy keys (contract in `tests/test_predictions_schema.py`).

## The UI honesty pages

Both keep a **Streamlit-free** data/formatter layer (unit-tested headless), with a
single `render()` Streamlit entry point:

- `ui/model_honesty.py` → `ui/pages/14_Model_Honesty.py` — per-line GO/NO-GO
  banner, CLV read, integrity status list, per-band calibration table. Also exports
  `render_verdict_strip()` for embedding atop the performance pages.
- `ui/model_compare.py` → `ui/pages/15_Model_Compare.py` — CatBoost vs LightGBM vs
  market scorecard + reliability diagram + per-band detail.

## Commands

```powershell
# Full rebuild ("use my full PC"): normalise → build → train CatBoost(GPU)+LightGBM(CPU)
# concurrently → CatBoost holdout → refresh predictions.json → timing + verdicts.
python pipeline_full.py [--skip-scrape] [--no-tune] [--resume] [--force]

# Daily flow / predict only
python -m scripts.refresh [--no-scrape]
python -m models.predictor [--min-odds 2.5]

# Honest holdout GO/NO-GO for one frozen model
python -m backtest.holdout --model models/catboost_won_v3nf.bin \
    --holdout-start 2026-05-23 --holdout-end 2026-06-12

# Tests
python -m pytest -q
```

## Conventions

- `pipeline_full.py` rebuilds features (~6 min) before training; if you wrap it in
  an outer timeout, budget for that or use `--resume` (skips up-to-date stages).
- Frozen-model filename grammar: `catboost_<target>_<tag>.bin` with sibling
  `catboost_<tag>_meta.json` / `*_calib.pkl`.
- Tests must never hit the network (`tests/conftest.py` disables notifications).
  CatBoost-free tests stub `predict_proba` (see `tests/backtest/test_holdout.py`,
  `tests/test_smoke_end_to_end.py`).
