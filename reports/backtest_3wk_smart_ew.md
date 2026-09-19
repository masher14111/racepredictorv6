# Backtest Report

**Generated:** 2026-06-19 02:48 IST  
**Date range:** 2026-05-17 → 2026-06-12  
**Races:** 141 | **Test runners:** 8687 | **Train rows:** 239486 | **Test rows:** 8687

## Configuration

| Parameter | Value |
|-----------|-------|
| Strategy | flat |
| Flat stake | €4.00 |
| Kelly fraction | 0.25 |
| Initial bankroll | €1000.00 |
| Min selection odds | 4.00 |
| Top N per race | 2 |
| Bet type | each_way |
| Train/test split | 96/3 by date |
| Stop-loss | 20% drawdown |

## Overall Performance

| Metric | Model (CatBoost) | Market Baseline |
|--------|-------------|-----------------|
| Bets placed | 282 | 112 |
| Win rate | 17.7% | 12.5% |
| Place rate | 55.0% | 34.8% |
| Total staked | €1128.00 | €448.00 |
| Total profit | -€191.41 | -€200.41 |
| ROI | -16.97% | -44.73% |
| Max drawdown | 18.91% | 19.72% |
| CLV (mean) | n/a | n/a |
| CLV > 0 (beat close) | n/a | n/a |
| Final bankroll | €808.59 | €799.59 |

## Monthly P&L — Model (CatBoost)

| Month | Bets | Win % | Staked | Profit | ROI |
|-------|------|-------|--------|--------|-----|
| 2026-05 | 160 | 17.5% | €640.00 | -€85.93 | -13.43% |
| 2026-06 | 122 | 18.0% | €488.00 | -€105.48 | -21.61% |

## Performance by Odds Band

| Band | Bets | Win % | Staked | Profit | ROI |
|------|------|-------|--------|--------|-----|
| Longshot (>10.0) | 4 | 0.0% | €16.00 | +€1.44 | +8.98% |
| Mid (4.1–10.0) | 278 | 18.0% | €1112.00 | -€192.85 | -17.34% |

## Per-Band ROI — bootstrapped 95% CI (2000 resamples)

| Band | Bets | ROI | Median ROI | CI low | CI high |
|------|------|-----|-----------|--------|---------|
| Longshot (>10.0) | 4 | +8.98% | +8.98% | -100.00% | +117.97% |
| Mid (4.1–10.0) | 278 | -17.34% | -17.28% | -29.87% | -4.05% |

_A CI that straddles 0% means the band's edge is not distinguishable from noise at this sample size._

## Edge Robustness (Longshot Gate)

**Status:** ✓ robust

| Check | Value |
|-------|-------|
| ROI (all bets) | -16.97% |
| ROI excl. top 3 winners | -22.01% |
| Top 3 winners' share of winnings | 13.8% |
| Longshot winners (>10) | 2 |
| Longshot share of winnings | 2.4% |

## Top 10 Venues by ROI

| Venue | Bets | Win % | Profit | ROI |
|-------|------|-------|--------|-----|
| Huntingdon | 4 | 75.0% | +€41.81 | +261.29% |
| Fontwell | 4 | 75.0% | +€21.84 | +136.48% |
| Plumpton | 2 | 0.0% | +€8.87 | +110.84% |
| Navan | 2 | 50.0% | +€5.36 | +67.00% |
| Fairyhouse | 6 | 50.0% | +€14.23 | +59.30% |
| Tramore | 4 | 50.0% | +€9.44 | +59.03% |
| Beverley | 4 | 25.0% | +€8.56 | +53.49% |
| Thirsk | 4 | 50.0% | +€8.40 | +52.50% |
| Gowran Park | 6 | 50.0% | +€12.30 | +51.24% |
| Nottingham | 8 | 37.5% | +€14.95 | +46.71% |

## Notes

- Scoring mode: **Model (CatBoost)**.
- Market baseline ranks runners by implied probability (favourite first).
- Walk-forward: model evaluated on the last 3% of historical races by date.
- Bets selected/staked at the takeable board price (≥ 4.0); **settled at the realistic executable price (BSP → SP → returned SP)** — not the price used to pick the bet.
- Settlement price source(s) this run: odds_finish.
- **CLV not measurable** this run: no independent board price distinct from the closing/SP price in the data (both collapse to the same source).
- Selection is blind to the result: the scorer never sees finishing position.
