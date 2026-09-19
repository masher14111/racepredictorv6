---
name: ui-18-audit
description: Prompt 18 end-to-end UI audit — page map, unsurfaced backend capabilities, fixed bugs, and a prioritized redesign brief for Prompt 19
metadata:
  type: project
---

# UI Audit (Prompt 18, 2026-06-16)

Framework: **Streamlit multipage** (`ui/app.py` is the entry/dashboard; `ui/pages/1..7`
auto-populate the sidebar nav; most pages are thin `runpy.run_path` wrappers around a
sibling module). Tokens/CSS are duplicated inline per page (no shared stylesheet).
Theme: navy sidebar `#172033`, light content `#f6f7f9`/white, blue accent `#2563eb`,
quiet green for value. tz Europe/Dublin. Prompt 19 reads this file.

## Page-by-page inventory

| Nav                   | File(s)                                                           | Shows                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| --------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Dashboard** (entry) | `ui/app.py` → `ui/logic.py`, `ui/_theme.py`                       | KPI strip (Races/Value bets/Selections/Each-Way/Models); venue-grouped expanders of race cards: rank badge, horse, jockey, best odds + per-book price row (★ best), Win% bar, composite Score bar, EV cell (when cache has value layer). Sidebar: Date filter (uses `default_date_index` — Today-filter fix), Venue multiselect, Min-composite slider, Stake-strategy selectbox (session-only, no effect), Refresh button. Stale-cache + refresh-outcome banners; 4-way empty state. |
| 1 Scrape & Refresh    | `pages/1_Scrape_and_Refresh.py` → `ui/refresh_ops.py`             | Per-book scrape buttons (livescorebet/paddy_power/boylesports) + rebuild predictions (normalize → build_inference_matrix → predict).                                                                                                                                                                                                                                                                                                                                                 |
| 2 Live Races          | `pages/2_Live_Races.py` → `ui/live_races.py`                      | Live cards from unified/live_odds/caches; status (Upcoming/Starting Soon/In-Play/Finished by time math); highlights predicted runners.                                                                                                                                                                                                                                                                                                                                               |
| 3 Place Bet           | `pages/3_Place_Bet.py` → `ui/bet_placer.py`                       | Race/horse picker, stake form, win/each-way, confirm flow, BetTracker P&L sidebar, recent bets.                                                                                                                                                                                                                                                                                                                                                                                      |
| 4 Performance         | `pages/4_Performance.py` → `ui/performance_dashboard.py` (Plotly) | Cumulative/monthly P&L, rolling ROI, win-rate-by-odds-band, outcome donut, pending settlement, bet-history CSV, PDF export. **No model metrics.**                                                                                                                                                                                                                                                                                                                                    |
| 5 Compare             | `pages/5_Compare.py` → `ui/race_compare.py`                       | Head-to-head two horses: summary cards, radar (plotly), key diffs, full feature table grouped by `_FEATURE_GROUPS`. Reads `data/features.parquet`.                                                                                                                                                                                                                                                                                                                                   |
| 6 Settings            | `pages/6_Settings.py` → `ui/settings.py`                          | Editable config form (proxy/scrape/bet thresholds/model/drift), atomic save to config.yaml, Reload Services. Crash fixed: `st.page_link` replaced with launch-command list (settings.py:170).                                                                                                                                                                                                                                                                                        |
| 7 Test Results        | `pages/7_Test_Results.py`                                         | Renders pytest JUnit XML from `reports/testrun/junit.xml`.                                                                                                                                                                                                                                                                                                                                                                                                                           |

### Orphaned (NOT in `ui/pages/`, unreachable from nav)

- **`ui/predictions.py`** — the _richest_ prediction view: risk tiers (strong/good/moderate/speculative), confidence labels, best-bets strip, per-selection **feature "why" breakdown** (`_feature_breakdown_html` + `_load_importance` from CatBoost `.bin`). Competes with app.py as a second full dashboard (own CSS+sidebar).
- **`ui/performance.py`** (Altair) — the **only** model-metrics view: `_render_model_perf` = per-target AUC bars + band AUCs + feature-importance chart. Competes with `performance_dashboard.py` (Plotly).
- `backtest/` package (incl. `_explain_demo.py` SHAP) and `llm/text_features.py` — surfaced nowhere.

## (1) Backend capabilities NOT surfaced in the UI

Predictor (`models/predictor.RunnerPrediction`) emits far more than any page shows.

- **In the live cache but unshown** on the dashboard: `value_win_prob`, `value_edge` (only in a `title=` tooltip), `placed_2_prob`, `showed_prob`, `implied_prob`, `low_odds`. EV column appears only when present and only far-right.
- **Not even in the cache (stale `predictions.json`, generated 2026-06-15):** `won_prob_normalized`, `value_supported`, `data_completeness`, `confidence` (high/med/low), `first_time_runner`. The data-quality/confidence layer postdates the cache.
- **Per-prediction feature "why" / SHAP:** only in orphaned `predictions.py`; SHAP demo only in `backtest/_explain_demo.py`.
- **Model quality** (test AUC, band AUCs, calibration/ECE, A/E) from `models/catboost_v3_meta.json`: only in orphaned `performance.py`; the main dashboard shows model _names_ only.
- **Backtest results** (`backtest/` engine/metrics/report): no UI at all.
- **Scraper/data freshness:** only an implicit stale-cache banner; no per-source scrape timestamps or row counts.

## (2) Known bugs — root-caused

- **Blank jockey everywhere — CONFIRMED, upstream data bug (still open).** `predictions.json` has `jockey: ""` / `trainer: ""` for all runners. Predictor maps `jockey=row.get("jockey_name")` (predictor.py:836) but the inference matrix carries `jockey_id`/`trainer_id` and the name columns don't reach the predictor; `build_inference_matrix._fill_last_known_connections` only backfills from history. `data/unified_races.parquet` is currently **empty (0 rows)** so it can't be regenerated as-is. Fix = ensure `jockey_name`/`trainer_name` survive derive→predict, then rebuild cache. NOT a UI-only fix.
- **Empty feature breakdowns — CONFIRMED root cause.** `data/features.parquet` is a **5-row synthetic fixture** (`horse_id='h1'`, `jockey_id`/`trainer_id` but no name cols). `predictions.py` and `race_compare.py` join live horses against it and find nothing → silent empty. Fix = regenerate features.parquet from real data (it's written by `features/builder.build_features(write=True)`).
- **Settings crash — FIXED already.** Was `st.page_link` from a standalone-script page (raises); now lists launch commands (settings.py:170). Parses + imports clean.
- **page_link "broken" — by design.** Standalone pages can't use `st.page_link`; `_theme.pin_sidebar_nav()` + launch-command lists are the deliberate workaround.
- **Over-aggressive Today filter — FIXED.** `app.py` uses `ui.logic.default_date_index` (falls back to "All upcoming" when no race today). `predictions.py` had a hardcoded `index=0` — **fixed this pass** to use the same helper.

## Fixed this pass

- `ui/predictions.py`: Date selectbox `index=0` → `default_date_index(races_all, today_dublin)` (+ import from `ui.logic`). Aligns the orphaned view with the tested Today-filter fix. Verified: empty cache → index 2 ("All upcoming").

Still open (need data pipeline, beyond quick UI fix): blank jockey (cache regen + name passthrough), empty breakdowns (regen features.parquet), stale predictions.json, empty unified_races.parquet.

## /impeccable results (main surface = `ui/app.py`)

- **critique: 30/40 (Good).** Not AI slop (passes the bans). Strengths: honest state handling, restrained trust-signalling, decision density. P0 = blank jockey; P1 = value/confidence layer unsurfaced + two competing/orphaned dashboards. Snapshot in `.impeccable/critique/`.
- **audit: 15/20 (Good).** A11y 3 / Perf 3 / Theming 2 / Responsive 3 / Anti-patterns: app.py clean, but **5 side-stripe-border ban violations** in `predictions.py`(×2), `performance.py`, `live_races.py`, `bet_placer.py`. `--muted #6b7689` on `--bg #f6f7f9` = **4.28:1 (sub-AA)**. Systemic: CSS tokens duplicated per page (no shared stylesheet).

## Prioritized redesign brief (for Prompt 19)

1. **Fix the data so the screen is honest (P0).** Pass `jockey_name`/`trainer_name` through to the predictor and regenerate `predictions.json`; regenerate a real `data/features.parquet`; rebuild `unified_races.parquet`. Without this every prediction view is degraded.
2. **Make value & confidence first-class (P1).** Lead race cards with edge/EV and a confidence indicator (PRODUCT.md principle 2: value over favourites), add place/show columns, allow sort-by-edge. Surface `data_completeness`/`confidence` once the cache carries them.
3. **Consolidate to one prediction surface + one performance surface (P1).** Choose app.py _or_ predictions.py as canonical; fold the unique value (per-feature "why" breakdown, risk tiers, model AUC/calibration metrics from performance.py) into the nav. Kill the duplicate. Pick one chart lib (Plotly vs Altair).
4. **Surface model trust (P2).** A calibration / A-E / AUC badge reachable from the dashboard header.
5. **Surface backtest + data freshness (P2).** A page for `backtest/` results; per-source scrape timestamps/row counts.
6. **Design-system + a11y hygiene (P2).** Extract shared CSS tokens to one stylesheet; remove the 5 side-stripe borders (use full borders / bg tints / leading icons); darken `--muted` to clear AA on `--bg`; move EV-edge out of `title=` into visible text.

See [[review-05-ui-bugs]], [[review-09-audit-2026-06-14]], [[model-09-baseline-audit]], [[calibration-value-work-handoff]].
