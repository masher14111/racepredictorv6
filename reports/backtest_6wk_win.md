# Backtest Report

**Generated:** 2026-06-19 01:38 IST  
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
| Top N per race | 1 |
| Bet type | win |
| Train/test split | 93/6 by date |
| Stop-loss | 20% drawdown |

## Overall Performance

| Metric | Model (CatBoost) | Market Baseline |
|--------|-------------|-----------------|
| Bets placed | 157 | 39 |
| Win rate | 33.1% | 17.9% |
| Place rate | 33.1% | 17.9% |
| Total staked | €1570.00 | €390.00 |
| Total profit | -€203.80 | -€209.62 |
| ROI | -12.98% | -53.75% |
| Max drawdown | 19.58% | 20.98% |
| CLV (mean) | n/a | n/a |
| CLV > 0 (beat close) | n/a | n/a |
| Final bankroll | €796.20 | €790.38 |

## Monthly P&L — Model (CatBoost)

| Month | Bets | Win % | Staked | Profit | ROI |
|-------|------|-------|--------|--------|-----|
| 2026-04 | 23 | 26.1% | €230.00 | -€62.79 | -27.30% |
| 2026-05 | 101 | 33.7% | €1010.00 | -€134.65 | -13.33% |
| 2026-06 | 33 | 36.4% | €330.00 | -€6.36 | -1.93% |

## Performance by Odds Band

| Band | Bets | Win % | Staked | Profit | ROI |
|------|------|-------|--------|--------|-----|
| Favourite (≤4.0) | 142 | 33.8% | €1420.00 | -€174.49 | -12.29% |
| Mid (4.1–10.0) | 15 | 26.7% | €150.00 | -€29.31 | -19.54% |

## Per-Band ROI — bootstrapped 95% CI (2000 resamples)

| Band | Bets | ROI | Median ROI | CI low | CI high |
|------|------|-----|-----------|--------|---------|
| Favourite (≤4.0) | 142 | -12.29% | -12.23% | -33.10% | +9.02% |
| Mid (4.1–10.0) | 15 | -19.54% | -19.54% | -78.52% | +56.64% |

_A CI that straddles 0% means the band's edge is not distinguishable from noise at this sample size._

## Edge Robustness (Longshot Gate)

**Status:** ✓ robust

| Check | Value |
|-------|-------|
| ROI (all bets) | -12.98% |
| ROI excl. top 3 winners | -19.05% |
| Top 3 winners' share of winnings | 10.6% |
| Longshot winners (>10) | 0 |
| Longshot share of winnings | 0.0% |

## Top 10 Venues by ROI

| Venue | Bets | Win % | Profit | ROI |
|-------|------|-------|--------|-----|
| Hereford | 2 | 100.0% | +€43.90 | +219.50% |
| Cartmel | 3 | 66.7% | +€27.88 | +92.94% |
| Ballinrobe | 3 | 66.7% | +€25.69 | +85.64% |
| Market Rasen | 3 | 66.7% | +€24.82 | +82.75% |
| Curragh | 4 | 50.0% | +€26.00 | +65.00% |
| Limerick | 4 | 50.0% | +€25.98 | +64.94% |
| Huntingdon | 3 | 66.7% | +€16.45 | +54.82% |
| Naas | 3 | 66.7% | +€16.06 | +53.52% |
| Kilbeggan | 3 | 66.7% | +€9.96 | +33.21% |
| Doncaster | 6 | 50.0% | +€15.02 | +25.04% |

## Notes

- Scoring mode: **Model (CatBoost)**.
- Market baseline ranks runners by implied probability (favourite first).
- Walk-forward: model evaluated on the last 6% of historical races by date.
- Bets selected/staked at the takeable board price (≥ 2.5); **settled at the realistic executable price (BSP → SP → returned SP)** — not the price used to pick the bet.
- Settlement price source(s) this run: odds_finish.
- **CLV not measurable** this run: no independent board price distinct from the closing/SP price in the data (both collapse to the same source).
- Selection is blind to the result: the scorer never sees finishing position.
