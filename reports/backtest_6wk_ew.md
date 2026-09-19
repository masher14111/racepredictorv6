# Backtest Report

**Generated:** 2026-06-19 01:30 IST  
**Date range:** 2026-04-20 → 2026-06-12  
**Races:** 278 | **Test runners:** 17373 | **Train rows:** 230800 | **Test rows:** 17373

## Configuration

| Parameter | Value |
|-----------|-------|
| Strategy | flat |
| Flat stake | €10.00 |
| Kelly fraction | 0.25 |
| Initial bankroll | €1000.00 |
| Min selection odds | 2.50 |
| Top N per race | 3 |
| Bet type | each_way |
| Train/test split | 93/6 by date |
| Stop-loss | 20% drawdown |

## Overall Performance

| Metric | Model (CatBoost) | Market Baseline |
|--------|-------------|-----------------|
| Bets placed | 75 | 57 |
| Win rate | 20.0% | 15.8% |
| Place rate | 62.7% | 52.6% |
| Total staked | €750.00 | €570.00 |
| Total profit | -€211.22 | -€226.89 |
| ROI | -28.16% | -39.81% |
| Max drawdown | 22.09% | 23.64% |
| CLV (mean) | n/a | n/a |
| CLV > 0 (beat close) | n/a | n/a |
| Final bankroll | €788.78 | €773.11 |

## Monthly P&L — Model (CatBoost)

| Month | Bets | Win % | Staked | Profit | ROI |
|-------|------|-------|--------|--------|-----|
| 2026-04 | 9 | 11.1% | €90.00 | -€23.42 | -26.02% |
| 2026-05 | 60 | 20.0% | €600.00 | -€169.16 | -28.19% |
| 2026-06 | 6 | 33.3% | €60.00 | -€18.64 | -31.07% |

## Performance by Odds Band

| Band | Bets | Win % | Staked | Profit | ROI |
|------|------|-------|--------|--------|-----|
| Favourite (≤4.0) | 67 | 19.4% | €670.00 | -€197.72 | -29.51% |
| Mid (4.1–10.0) | 8 | 25.0% | €80.00 | -€13.50 | -16.88% |

## Per-Band ROI — bootstrapped 95% CI (2000 resamples)

| Band | Bets | ROI | Median ROI | CI low | CI high |
|------|------|-----|-----------|--------|---------|
| Favourite (≤4.0) | 67 | -29.51% | -30.36% | -46.85% | -10.38% |
| Mid (4.1–10.0) | 8 | -16.88% | -17.00% | -81.75% | +66.00% |

_A CI that straddles 0% means the band's edge is not distinguishable from noise at this sample size._

## Edge Robustness (Longshot Gate)

**Status:** ✓ robust

| Check | Value |
|-------|-------|
| ROI (all bets) | -28.16% |
| ROI excl. top 3 winners | -36.19% |
| Top 3 winners' share of winnings | 28.4% |
| Longshot winners (>10) | 0 |
| Longshot share of winnings | 0.0% |

## Top 10 Venues by ROI

| Venue | Bets | Win % | Profit | ROI |
|-------|------|-------|--------|-----|
| Aintree | 3 | 33.3% | +€2.39 | +7.98% |
| Ballinrobe | 9 | 33.3% | -€2.88 | -3.20% |
| Bath | 24 | 20.8% | -€48.95 | -20.40% |
| Beverley | 15 | 20.0% | -€34.70 | -23.13% |
| Carlisle | 9 | 11.1% | -€40.52 | -45.02% |
| Brighton | 6 | 16.7% | -€34.54 | -57.57% |
| Ascot | 9 | 11.1% | -€52.03 | -57.81% |

## Notes

- Scoring mode: **Model (CatBoost)**.
- Market baseline ranks runners by implied probability (favourite first).
- Walk-forward: model evaluated on the last 6% of historical races by date.
- Bets selected/staked at the takeable board price (≥ 2.5); **settled at the realistic executable price (BSP → SP → returned SP)** — not the price used to pick the bet.
- Settlement price source(s) this run: odds_finish.
- **CLV not measurable** this run: no independent board price distinct from the closing/SP price in the data (both collapse to the same source).
- Selection is blind to the result: the scorer never sees finishing position.
