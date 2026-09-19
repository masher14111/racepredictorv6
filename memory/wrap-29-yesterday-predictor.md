---
name: wrap-29-yesterday-predictor
description: New "Yesterday's Bet Predictor" — one-day blind backtest engine + Streamlit page; results lag means it settles the most recent settled day at SP
metadata:
  type: project
---

Built the **Yesterday's Bet Predictor** (2026-06-17): an honest, fully out-of-sample
one-day backtest that scores a completed racing day blind to the result, places the
model's bets, and settles them against the real finishing positions.

- **Engine:** `models/yesterday.py` — `run_yesterday(YesterdayConfig, df=None) -> YesterdayResult`.
  - Loads the persisted labelled matrix `data/features/training.parquet` (instant; the exact
    output of `build_training_matrix`, 248k rows, dates 2024-01-01→2026-06-12) — falls back to
    a slow rebuild only if missing.
  - Scores via `Predictor` with `_value_enabled=True` forced so the price-free v3nf value
    model loads regardless of `config.value.enabled`.
  - Two strategies: `"value"` (reuses `models.value.find_value_bets` + `ValueConfig.from_config`
    — identical gating to the live Today's Suggestions page) and `"top_pick"` (highest
    `won_prob` per race, min-odds gated). Bet types win / each_way.
  - Settlement reuses `utils.backtester._settle_outcome` / `_gross_return`. Selection never
    sees `position`.
- **UI:** `ui/pages/13_Yesterdays_Bet_Predictor.py` — sidebar controls (strategy / bet-type /
  stake / settled-day select), KPI rail (bets, win rate, staked, profit, ROI), settled-bets
  `st.dataframe`, and an honesty note. Cached with `@st.cache_data`.
- **Tests:** `tests/models/test_yesterday.py`, 13 passing (pure helpers + monkeypatched
  `_score_day` for the end-to-end paths).

**Why the "most recent settled day", not literally yesterday:** finishing positions arrive
from the historical results feeds a few days _after_ the races run. On 2026-06-17 the latest
settled day was **2026-06-12** (today/yesterday rows exist in `unified_races.parquet` but with
0 positions). The engine/page default to `available_dates[-1]` and state the date plainly.

**Why settle at SP:** settled rows carry only the starting price (`odds_finish`); live
bookmaker board prices (livescorebet/paddy/boyle) exist only for _upcoming_ races. So bets are
selected AND settled at SP — CLV is not measurable on this data, and the value-bet profit is
conservative vs. taking a bigger board price. Same constraint the [[model-14-backtest]] /
`utils.backtester` honesty notes describe.

Sanity run on 2026-06-12 (58 races): value/win = 3 value bets, 0 winners, −€30 (tiny sample);
top_pick/win = 58 bets, 18 winners, ROI −43% (favourites at SP lose to the vig). One day is
noise — judge the edge over many days on the dashboard / `utils.backtester`.

Retrain question answered: **NOT needed.** Models were retrained 2026-06-16 (v3 now calibrated
per [[model-11-calibration]], v3nf price-free per [[model-15-feature-selection]]). Retrain only
on drift via `python -m models.retrain_trigger`. UI flow: `streamlit run ui/app.py` → sidebar
**Refresh predictions**; per-book scrape on page **1 Scrape & Refresh**. Relates to
[[ui-22-suggestions]] (the live counterpart this settles) and [[wrap-28-docs]].
