---
name: ui-20-paper-betting
description: Prompt 20 paper-betting feature — virtual-bankroll practice betting (NO real money). Bet schema + CLV capture, auto-settlement from scraper results, UI entry points (race-detail per-runner form + Paper Betting page). Prompt 21 visualises this data.
metadata:
  type: project
---

# Paper Betting (Prompt 20, 2026-06-16)

Simulated **paper / practice** betting (clearly labelled "no real money") to test
whether the model's calibrated probs + value edges actually make money, measured
primarily by **CLV** (closing-line value), not just P&L. Built on the existing
`utils/bet_tracker.py` (Prompt-19-era) — extended, not replaced. Prompt 21
visualises this ledger (equity curve, CLV distribution).

## Bet schema (`bets` table, `data/races.db`)

Pre-existing columns: `id, placed_at, race_id, horse_id, horse_name, venue,
race_time, bet_type ('win'|'each_way'), stake, odds_decimal, composite_score,
won_prob, strategy, bankroll_before, outcome, settled_at, gross_return, profit,
notes`. **Migration 3** (`utils/storage/migrations.py`) adds the CLV trio:

- `value_edge` — model edge at bet time (won_prob − implied), captured on record.
- `closing_odds` — the closing price (Betfair SP), captured at settlement.
- `clv_pct` — `(odds_taken / closing_odds − 1) × 100`; **positive = beat the
  line**. Computed in `BetTracker._clv_pct`, stored on settle.

`bankroll_log` (unchanged shape) records every `init` / `bet_placed` /
`bet_settled` / `adjust` event with running `balance` — the equity curve source.

`race_id` for paper bets = the UI **race slug** (`ui._components.race_slug`,
venue+post-time), since predictions have no explicit race id. `horse_id` comes
from the prediction cache; settlement falls back to normalized-name matching when
ids don't line up across sources.

## `utils/bet_tracker.py` additions

- `record_bet(... value_edge=...)` — stores the edge.
- `place_paper_bet(...)` — **the guarded entry point.** Raises `RaceStartedError`
  (race_time ≤ now, via `race_has_started` static method — UTC-normalized,
  unknown time ⇒ False so a missing post-time never blocks), `DuplicateBetError`
  (`has_open_bet(race_id, horse_id, bet_type)` — only fires when both ids present;
  win vs each-way on the same horse are distinct markets), or `StopLossError`.
  Then delegates to `record_bet`.
- `settle_bet(bet_id, outcome, closing_odds=None)` — now accepts **`void`**
  (full stake refund, profit 0) alongside win/place/lose; computes + stores
  `clv_pct` when `closing_odds` given.
- `set_bankroll(amount)` — before any bets, **resets the `init` baseline** (so
  ROI/drawdown measure from the new figure); after bets exist, logs an `adjust`
  event keeping the curve continuous.
- `summary()` gains `avg_clv_pct`; `ew_places` property exposed for settlement.

## Settlement flow (`utils/bet_settlement.py`)

Results arrive from the betsp scrapers → `data/historical/betsp.parquet`
(columns incl. `position`, `odds_finish` = SP). `settle_from_results(tracker,
results)` indexes results by `horse_id` and by `(norm_venue, norm_horse, day)`,
plus a `raced` set of `(venue, day)` races that appeared at all. Per pending bet:

- match by horse_id, else by normalized name+venue+day;
- **no match + race ran** (`(venue, day)` in `raced`) ⇒ **void** (non-runner);
- **no match + race absent** ⇒ leave **pending** (results not scraped yet —
  this is the key guard against premature settlement);
- matched: position 1 ⇒ win; ≤ `ew_places` on an each-way ⇒ place; position
  null/≤0 ⇒ void; else lose. `closing_odds = odds_finish` ⇒ CLV.

Idempotent (settled bets are skipped). `settle_from_storage(tracker, storage=None)`
loads the parquet via `utils.storage.get_storage()` and calls the above; missing/
empty dataset ⇒ `[]` (non-fatal).

## UI entry points

- **`ui/_betting.py`** — shared Streamlit glue. `get_tracker()` (`@st.cache_resource`,
  config-driven from the `bet_tracker:` section). `place_bet_widget(sel, race, ccy)`
  — collapsed per-runner expander → `st.form` (market / odds-taken / recommended
  stake) → `place_paper_bet`, surfacing each guard inline. `PAPER_NOTICE` HTML
  banner. Recommended stake uses `st.session_state["strategy"]` (set in the
  landing sidebar).
- **`ui/pages/8_Race_Detail.py`** — each runner block now renders
  `place_bet_widget` beneath it; `PAPER_NOTICE` above the runners list.
- **`ui/pages/9_Paper_Betting.py`** — the bankroll cockpit: set-bankroll form
  (sidebar), KPI rail (Bankroll / P&L / ROI / Open bets / Avg CLV), **"Settle
  from results"** button (calls `settle_from_storage`), per-bet **manual** settle
  (outcome + closing-odds form), and the settled-history dataframe.
- **`ui/_design.py`** — `.rp-paper-note` amber banner (locked roles: amber =
  caution, fits "this is practice, not real money").

## Verification

`pytest` 947 pass / 3 skip. New: `tests/utils/test_bet_settlement.py` (10:
win/lose/place/EW-outside, void-absent/void-no-position, stay-pending,
name-fallback, idempotent) + paper-betting tests appended to
`tests/utils/test_bet_tracker.py` (value-edge, ±CLV/null-CLV, avg-CLV, void
refund, set-bankroll reset/adjust, the three guards, market distinction).

## Gotchas

- **Started-race guard uses the real wall clock** (`utils.timezone.now`). Tests
  pass `_now=` or use far-future race_times; settlement-test result fixtures
  share the bet's day so name/void matching lines up.
- **`get_tracker()` is `@st.cache_resource`** (one tracker/pool across reruns);
  reads are always live from SQLite so placing/settling needs no cache bust —
  but a server restart is still needed to pick up `_design.py`/`_components.py`
  edits (see [[ui-19-redesign]] gotcha 3).
- **Duplicate guard only fires with both ids present.** Free-form `record_bet`
  calls (no race_id/horse_id) are never deduped — that's deliberate.

See [[ui-19-redesign]], [[review-02-utils-bugs]], [[model-16-value-detection]],
[[reference-horse-racing-mvp-vault]] (CLV > ROI principle).
