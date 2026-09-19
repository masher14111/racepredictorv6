---
name: model-14-backtest
description: backtest/ walk-forward value-betting engine — API, leak-free design, and headline result (model calibrated but NOT bettable at execution price; CLV negative; favorite-longshot bias)
metadata:
  type: project
---

# Walk-forward backtester (`backtest/` module) — 2026-06-16

A rigorous, leak-free walk-forward / value-betting engine. Built per the task
"Prompts 16 & 22 depend on this". Lives in `backtest/` (NOT `models/backtest.py`).
Tests in `tests/backtest/` (100 tests). Builds on [[model-09-baseline-audit]] and
the value principles in [[reference-horse-racing-mvp-vault]].

**Why:** the whole system exists to find _value bets_, so a calibrated model is
necessary but not sufficient — we need to know whether betting it at realistically
available prices actually makes money after the market's margin. A leak-free
backtester is the only honest way to measure that, and it is the substrate for the
two follow-on prompts (strategy/EV-threshold sweeps and live value selection).

**How to apply:**

## API (entry points)

```python
from backtest import load_panel, Backtester, BacktestConfig, WalkForwardConfig, \
    Strategy, FlatStake, KellyStake, CatBoostFactory, render_report, save_run
panel = load_panel()                                # bet candidates from training.parquet
bt = Backtester(panel, CatBoostFactory(task_type="GPU"),
                WalkForwardConfig(min_train_days=365, test_window_days=90))
run = bt.run({"ev_flat": (Strategy("ev5", min_ev=0.05, min_odds=2, max_odds=26),
                          FlatStake(10))})
print(render_report(run)); save_run(run)            # -> data/backtests/<ts>/ for the UI
```

CLI: `python -m backtest --min-train-days 365 --window-days 90 --iterations 400 --task-type GPU`
(`--rolling`, `--commission`, `--stop-loss`, `--no-save`, etc.). Defaults pull
`value:`/`bet_tracker:`/`model:` from config.yaml via `utils.config_loader`.

## Modules

- `splitter.py` — `walk_forward_folds()`. **Single source of truth for no-look-ahead.**
  train: `race_date <= T`; test: `T < race_date <= T+window`. Expanding (default) or rolling.
- `data.py` — `load_panel()`. WIN-market only (PLACE rows duplicate runners). Bets at the
  **execution price** = `ppwap` (pre-off Betfair WAP), falls back to `morningwap`. Closing
  line = `odds_finish` (BSP) for CLV only. **The training `implied_prob` (== 1/odds_finish,
  audit C2 leak) is never carried into the panel / never drives a bet.**
- `model.py` — `CatBoostFactory` (price-free `PRICE_FREE_FEATURE_COLS`, calibration tail
  carved from TRAIN only) + `CallableModelFactory` escape hatch for tests/external scorers.
- `staking.py` — `Strategy` (pluggable AND-ed gates: min_ev, min_edge_pct, min_abs_edge,
  odds band, min/max prob, or a custom `selector`) + `FlatStake` / `KellyStake` (fractional,
  capped). EV=`p*d-1`, edge=`p-1/d`, Kelly=`(p*d-1)/(d-1)` = EV/(d-1) floored at 0.
- `metrics.py` — NaN-safe, scalar-or-vector betting maths + `summarize_bets` (ROI/yield,
  hit-rate, max drawdown, per-bet & daily Sharpe, CLV, beat-close rate) + `ae_table` (A/E by
  prob bucket and by odds band).
- `engine.py` — fit ONCE per fold, score test rows, dedup overlapping windows, replay every
  strategy over the shared OOS frame; bankroll compounds; optional stop-loss halts new bets.
- `report.py` — `render_report` (text) + `save_run` -> `data/backtests/<run_id>/`
  (summary.json + bets\_<strategy>.parquet + scored.parquet) for the UI to read later.

## Assumptions / caveats

- Features in `training.parquet` are point-in-time leak-safe (audit), so the same matrix is
  reused for every fold — only model _fit/calibration_ respects the train/test cut.
- **Only 21 of ~30 `PRICE_FREE_FEATURE_COLS` are present** in the current training.parquet
  (9 absent: course*\*, distance*\*, days_since_last_run, horse_career_runs, speed_trend). The
  matrix should be rebuilt to restore them; AUC matched baseline regardless.
- `race_time` is 100% null (audit C3) so a true per-race key is unavailable → **de-vig left
  OFF** (raw per-runner implied prob used; conservative for a backer).
- Bankroll caps each stake at funds on hand; flat staking on turnover >> bankroll at a
  negative edge drains to 0 (expected, not a bug).

## Headline result (run `data/backtests/20260616_195957`, 6 folds, 2025-01→2026-06, 77,323 OOS runners)

- **Model OOS: AUC 0.668, logloss 0.341, Brier 0.099, ECE 0.0015** — reproduces the v3nf
  baseline; aggregate calibration by _its own_ probability bucket is excellent (A/E ≈ 1.0
  across 0.02–0.30).
- **But it is NOT bettable at the execution price.** Every strategy loses: yields −2.1%
  (band-all) to −10.8% (edge10_kelly); all bankrolls → €0.
- **CLV is strongly negative (−9% to −12%)** and beat-close rate ~24–27% — the single most
  damning sign: we systematically take worse prices than the BSP close.
- **Favorite-longshot bias is uncorrected** (A/E by odds band): odds-on A/E 2.80, 2.0–4.0
  1.88 (favs win far more than the model says) vs 16–34 A/E 0.44, >34 A/E 0.17 (longshots
  hugely over-predicted). The price-free model leans into longshots, where the market is
  sharpest, hence the negative CLV.

**Takeaway for Prompts 16/22:** calibrated ≠ profitable. The next steps are odds-band
recalibration / favorite-longshot correction and EV/edge-threshold + staking sweeps —
selected on **CLV and A/E**, not raw ROI. Re-run after rebuilding the full feature matrix.
