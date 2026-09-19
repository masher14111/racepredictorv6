---
name: ui-21-dashboard
description: Prompt 21 performance dashboard — "does the model work?" Equity curve + A/E calibration are the heroes; CLV the tiebreaker. Paper-ledger metrics filterable (date/market/odds/picks-only) shown alongside walk-forward backtest runs. ui/dashboard.py (testable) + ui/pages/10_Dashboard.py.
metadata:
  type: project
---

# Performance Dashboard (Prompt 21, 2026-06-16)

The project's honest scoreboard — _does the model actually work?_ Built on the
Prompt-19 design system ([[ui-19-redesign]]), reading the paper-betting ledger
([[ui-20-paper-betting]]) and the walk-forward backtest runs ([[model-14-backtest]]).

**Deliberate anti-pattern:** NOT a wall of SaaS hero-number tiles. The two
**heroes** are the **bankroll equity curve** and the **A/E calibration by
probability bucket** (the live calibration check). CLV (did we beat the close?)
is the deciding signal per [[reference-horse-racing-mvp-vault]]. Everything else
(cumulative profit, ROI/yield, hit-rate, max drawdown) is a compact rail _below_
the heroes.

## Files

- **`ui/dashboard.py`** — Streamlit-free data + figure layer (so it's unit
  testable). Loaders, filters, metrics, and Plotly figure builders. Imports
  `plotly_layout`/`PALETTE` from `ui._design` and reuses `backtest.metrics`
  (`ae_table`, `clv_pct`, `max_drawdown`) so paper and backtest A/E share the
  exact same prob-bucketing.
- **`ui/pages/10_Dashboard.py`** — the Streamlit assembly only (sidebar filters,
  app-bar with trust badges, hero charts, KPI rail, backtest comparison). Calls
  `inject_design()` + `pin_sidebar_nav()` like pages 8/9. **Run via
  `streamlit run ui/app.py` → "Dashboard" page**, NOT standalone (the
  `st.page_link("app.py", …)` nav links raise `KeyError: 'url_pathname'` with no
  multipage registry — same constraint as pages 8/9).
- **`tests/ui/test_dashboard.py`** — 25 tests over the pure layer.

## Metrics + charts (and where the data comes from)

Paper metrics are computed from the **filtered** frame (`paper_metrics`), not
`tracker.summary()`, so the filters actually move the numbers.

| Surface                    | Source                                                                                                                        |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Equity curve** (HERO)    | paper ledger: `initial_bankroll + cumsum(profit)` over `settled_at` (reconstructed so filters apply), dashed baseline = start |
| **A/E calibration** (HERO) | paper: `ae_table(won_prob, won)`; backtest: `summary.json → ae_by_prob`. Grouped bars + green `A/E=1` reference               |
| Cumulative profit          | paper settled bets, per-bet outcome markers                                                                                   |
| CLV distribution           | paper `clv_pct` histogram, mean line; green right of 0 = beat the close                                                       |
| KPI rail                   | `paper_metrics`: Net P&L, ROI/yield, hit-rate, avg CLV, beat-close rate, max drawdown                                         |
| Trust badges + footer      | backtest `model_metrics` (OOS AUC, ECE)                                                                                       |
| Paper-vs-backtest equity   | both re-based to 100 at start vs bet # (`normalized_equity_fig`) — comparable despite different epochs                        |
| Strategy comparison table  | `backtest_compare_table(summary)` — every strategy, sorted by yield                                                           |

## Filters (sidebar)

Date range (`placed_date`), **market** (`bet_type` win/each-way), **odds band**
(`ODDS_BANDS`, favourite→longshot), and **"model picks only"** toggle — a _pick_
= a bet the model flagged positive value (`value_edge > 0`) vs all bets. Plus a
backtest **run** + **strategy** selector and a calibration-overlay toggle.

## Gotchas

- **Void bets** are excluded from hit-rate and A/E (no decision) but kept in P&L
  (stake refunded). `n_void` is surfaced separately.
- **Empty-honest by design.** With 0 paper bets the equity/CLV panels show an
  empty state and calibration renders **backtest-only** bars — the page is never
  dead, and never fabricates paper data. (Current `races.db` has no settled paper
  bets, so that's the live state.)
- Backtest A/E is over the **full OOS run** (all scored runners), the proper
  calibration read; paper A/E is over your settled bets only — labelled as such.
- Two **older** light-theme dashboards still exist (`ui/performance.py`,
  `ui/performance_dashboard.py` ← `pages/4_Performance.py`). This is the new
  design-system one and the only one with CLV / A/E / backtest comparison;
  consider retiring the old pair when pages 1-7 migrate off `base="light"`.

## Verification

`pytest` 972 pass / 3 skip (reportlab-only). Rendered in the live app (port via
`ui/app.py`): dark theme + heroes + trust badges (ECE 0.002), no console/runtime
errors beyond the known benign `theme.sidebar.textColor ""` warning. Screenshot:
`.design-md/dashboard-prompt21.png`.

See [[ui-20-paper-betting]], [[model-14-backtest]], [[ui-19-redesign]],
[[reference-horse-racing-mvp-vault]], [[model-09-baseline-audit]].
