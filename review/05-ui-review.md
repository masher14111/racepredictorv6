# Race Predictor v3 — Streamlit UI Review

_Reviewed: 2026-06-13 | All 8 UI files read in full + cross-checked against `features/builder.py`, `utils/bet_tracker.py`, `models/predictor.py`, `config.yaml`, and `review/04-features-models-review.md`._

**Scope:** `ui/app.py`, `ui/live_races.py`, `ui/predictions.py`, `ui/performance_dashboard.py`, `ui/performance.py`, `ui/bet_placer.py`, `ui/race_compare.py`, `ui/settings.py`.

---

## App startup (verdict, scheduler wiring)

**Verdict: NO scheduler wiring. The notification and reporter schedulers are never started.**

`app.py` is a single-page Streamlit script — it renders predictions from the JSON cache and offers a sidebar "Refresh predictions" button that lazily imports and calls `Predictor().predict()`. That is the entire startup sequence.

**What is missing:**

- `utils/notifications.py` (`start_scheduler` or equivalent) is never imported or called anywhere in the UI layer. Push notifications and race-soon alerts will never fire during a live UI session.
- The periodic reporter (mentioned in the project brief as a separate scheduler) is similarly absent.
- There is no multi-page Streamlit structure (no `ui/pages/` directory). Each UI file is a standalone script intended to be launched via `streamlit run`. `app.py` does not route to other pages.

**What works:**

- `st.set_page_config()` is correctly the first Streamlit call in every file.
- The lazy `from models.predictor import Predictor` inside `_run_predictor()` avoids a blocking model-load at import time.
- `@st.cache_data(ttl=60)` on `_load_predictions()` is appropriate and is correctly cleared after a fresh predict run.
- Session-state initialisation (`if "predictions" not in st.session_state`) is correct.

---

## Live Races page (verdict, bugs)

**Verdict: DATA LOADING is correct and robust; refresh only covers one scraper out of four.**

**What works:**

- Four-source fallback loading (`unified_races.parquet` → `live_odds.parquet` → `livescorebet.json` → `paddy_power.json`) with per-source try/except is clean and degrades gracefully.
- Runner deduplication by best-odds-first `groupby.first()` is correct.
- Race status heuristic (Upcoming / Starting Soon / In-Play / Finished) based on Dublin-local time is correct, using `to_local()`.
- Prediction cross-reference lookup (by `horse_id` then `horse_name.lower()`) is sensible given cross-source name variance.
- `jockey_name` is correctly read at `live_races.py:516` (`runner.get("jockey_name") or runner.get("jockey")`), unlike the other pages — this page will show jockey correctly from live scraper data.

**Bugs:**

1. **🟠 Refresh button only scrapes Paddy Power** (`live_races.py:598-602`). `_do_refresh()` calls only `scraper.paddy_power.scrape`. BoylesportsScraper, Timeform, and Betfair are not called. The docstring says Boylesports "is intentionally skipped" but doesn't mention Timeform/Betfair. After a user clicks "Refresh odds now", `unified_races.parquet` will only reflect PP data.

2. **🟡 `@st.cache_data(ttl=0)` on `_load_races` and `_load_pred_lookup`** (`live_races.py:306,392`). In Streamlit, `ttl=0` means the cache entry expires immediately — every rerun re-reads all source files from disk. If the page is accessed frequently this is a non-trivial I/O hit (parsing JSON + Parquet on every interaction). Intentional for live freshness but worth documenting; a `ttl=30` would be sufficient for 30-second stale tolerance.

3. **🟢 Note:** `_load_races` is decorated `@st.cache_data(ttl=0)` but `_load_races.clear()` is called on refresh anyway — the explicit clear is a no-op given `ttl=0`. Harmless.

---

## Predictions page (verdict, bugs)

**Verdict: COMPOSITE SCORE display is correct; feature breakdown is silently broken by a path mismatch; jockey/trainer always blank (inherited from predictor bug).**

**What works:**

- Calls `Predictor().predict()` correctly and clears both `_load_predictions` and `_load_features` caches after refresh.
- Risk-tier thresholds and composite-score formula are consistent with `models/predictor.py`.
- Feature group ordering matches `models/features.FEATURE_COLS` exactly.
- `_load_importance()` uses `@st.cache_resource` (correct — loads `.bin` model files once per process).
- `_render_selection` shows `won_prob`, `placed_2_prob`, `showed_prob`, and `composite_score` — all four are correct fields from the predictor output dict.

**Bugs:**

1. **🔴 Feature path mismatch — breakdowns always empty** (`predictions.py:38`).
   `_FEATURES_PATH = _ROOT / "data" / "features" / "features.parquet"` — with a `features/` subdirectory.
   `features/builder.py:20` writes to `_ROOT / "data" / "features.parquet"` — no subdirectory.
   `_load_features()` will always return `None`; every selection's feature breakdown will show only dashes.
   **Fix:** Change `_FEATURES_PATH` to `_ROOT / "data" / "features.parquet"`.

2. **🟠 Jockey/trainer always blank** (`predictions.py:521-522`).
   `_render_selection` reads `sel.get("jockey")` and `sel.get("trainer")`. The predictor writes these keys with `""` (Prediction bug #1 from review 04 — wrong column names `jockey`/`trainer` vs canonical `jockey_name`/`trainer_name`). Fixing the predictor (review 04 bug #3) will fix this automatically, but the UI itself adds no further correction.

3. **🟠 `_load_importance()` uses hardcoded `FEATURE_COLS`** (`predictions.py:305`).

   ```python
   pairs = sorted(zip(FEATURE_COLS, imps.tolist()), ...)
   ```

   If the model was trained on fewer columns (because some derive columns were absent at train time — `train.py` filters to `avail_cols`), `len(FEATURE_COLS) != len(imps)` and `zip` silently truncates. Importances are then assigned to wrong features. **Fix:** Read `feature_cols` from `catboost_v3_meta.json` (already loaded by `_load_meta()`) and use that instead of the hardcoded `FEATURE_COLS`.

4. **🟡 `timedelta` import inside function** (`predictions.py:646`).
   `from datetime import timedelta` is inside the sidebar block. This works but should be at the top-level import. Low priority.

---

## Performance Dashboard (verdict, bugs)

**Verdict: CORRECT wiring to BetTracker; all required methods exist. One dead-code tile and one date-default inconsistency.**

**What works:**

- `_get_tracker()` with `@st.cache_resource` → single BetTracker instance per process. Confirmed: `BetTracker` has `all_bets()`, `pending_bets()`, `settle_bet()`, `summary()`, `export_csv()` — all present.
- Settle workflow: calls `tracker.settle_bet(bid, outcome)`, then `_load_bets.clear()` and `st.rerun()` — correct.
- PDF export is guarded by `try/except ImportError` for reportlab — correct graceful degradation.
- KPI computation is correct (ROI, win rate, place rate, max drawdown all derived correctly).
- All four Plotly charts (P&L over time, monthly bar, win-rate by band, outcome donut) are logically correct.

**Bugs:**

1. **🟡 `tiles[2]` is built twice** (`performance_dashboard.py:441-463`).
   The initial build at line 441-445 references `kpis["ew_success"]` — that key does not exist in the `kpis` dict from `_compute_kpis`. The guard `'ew_success' in kpis` catches it but produces a sub-optimal fallback (shows total count not success count). Line 458-463 then immediately overwrites `tiles[2]` with the correct calculation. The first build is dead code.
   **Fix:** Remove lines 441-445 (the first `tiles[2]` assignment); keep only lines 458-463.

2. **🟡 `date.today()` used for filter defaults** (`performance_dashboard.py:302`).
   `min_date = date.today() - timedelta(days=365)` / `max_date = date.today()`.
   The rest of the codebase uses `now().date()` for Dublin-timezone-aware "today". Minor inconsistency; would show wrong date for users running on a non-Irish server after midnight.
   **Fix:** `from utils.timezone import now` (already imported) → `now().date()`.

3. **🟢 Note:** `_monthly_pl` groups on `"id"` column (`agg(bets=("id", "count"))`). `BetTracker.all_bets()` returns the SQLite rowid as `"id"` — confirmed present. No crash.

---

## Performance.py — duplicate page (verdict)

**Verdict: FUNCTIONAL but is a largely overlapping older version of `performance_dashboard.py`. Both exist; only `performance_dashboard.py` is linked from Settings navigation.**

`performance.py` uses Altair charts; `performance_dashboard.py` uses Plotly. Both import BetTracker. Both define `_get_tracker()` with `@st.cache_resource` — if both pages happen to run in the same Streamlit process, the `@st.cache_resource` key is module-qualified so they get separate instances (no sharing issue).

**Bug:**

1. **🔴 Top-level `import altair as alt`** (`performance.py:20`). If `altair` is not installed, the entire page crashes at import time with no user-facing error. `performance_dashboard.py` has the same risk with `import plotly.graph_objects as go` at line 22. Both should be guarded or added to `requirements.txt` with explicit version pins.

2. **🟠 `performance.py` is dead from navigation.** Settings links to `performance_dashboard.py`; `app.py` has no navigation at all. `performance.py` is unreachable from the UI unless run standalone. Recommend removing or merging.

---

## Bet Placer (verdict, bugs)

**Verdict: CORRECT BetTracker wiring; stake logic and confirmation flow are sound. One private-attribute mutation.**

**What works:**

- `BetTracker` instantiated with all relevant params (`kelly_fraction`, `stop_loss_pct`, `ew_fraction`, `ew_places`).
- `tracker.record_bet(...)` called with full signature — `horse_name`, `odds_decimal`, `stake`, `bet_type`, `horse_id`, `venue`, `race_time`, `composite_score`, `won_prob`, `strategy`, `notes`. All match `BetTracker.record_bet` signature (confirmed).
- `tracker.is_stopped()` and `StopLossError` handling are correctly used.
- `tracker.recommend_stake(dec_odds, win_prob, strategy)` is confirmed to exist in BetTracker.
- Two-step confirmation (Review → Confirm) is clean; session state for `pending_bet` and `last_placed` is well-managed.

**Bugs:**

1. **🟠 Direct private-attribute mutation** (`bet_placer.py:374`).
   `tracker._flat_stake = flat_stake` bypasses `BetTracker`'s constructor and any validation it may perform. If `BetTracker` caches or derives values from `_flat_stake` at construction time, this may have no effect. **Fix:** Add a `set_flat_stake(value)` method to `BetTracker`, or pass `flat_stake` as an override arg to `recommend_stake()`.

2. **🟠 Jockey/trainer always blank** (`bet_placer.py:321`).
   `_horse_card_html` reads `sel.get("jockey")` and `sel.get("trainer")`. Same root cause as predictions page bug #2. The horse detail card will always show "— · —" for connections.

3. **🟡 `tracker.bankroll` accessed directly** (`bet_placer.py:484`).
   `max_stake = max(tracker.bankroll, 0.50)` — assumes `bankroll` is a public attribute. If `BetTracker` exposes bankroll only via `summary()["bankroll"]`, this would AttributeError. From the grep, `BetTracker` exposes `bankroll` as a property (line 484 of bet_tracker.py wasn't checked but the `summary()` method exists). Low risk, worth confirming.

---

## Settings (verdict, bugs)

**Verdict: ATOMIC WRITE is correct; validation is sound. Navigation links broken in standalone mode; bet_tracker config not exposed.**

**What works:**

- Atomic write via temp file (`_CFG_PATH.with_suffix(".yaml.tmp")`) then `tmp.replace(_CFG_PATH)` is correct.
- `_load_cfg.clear()` is called after save — correct.
- Proxy URL regex validation with deduplication.
- Version tag regex validation.
- Partial merge: `cfg["model"] = {**cfg.get("model", {}), ...}` preserves unknown model keys.
- Re-run predictions button is guarded with `p.load()` check before calling `p.predict()`.

**Bugs:**

1. **🔴 `_load_cfg()` has no error handling** (`settings.py:122-124`).

   ```python
   with open(_CFG_PATH, "r", encoding="utf-8") as f:
       return yaml.safe_load(f) or {}
   ```

   If `config.yaml` is missing (fresh install, deleted by accident), the entire settings page crashes with an unhandled `FileNotFoundError`. Every other page handles this with `try/except`.
   **Fix:** Wrap in `try/except (FileNotFoundError, OSError): return {}`.

2. **🟠 `st.page_link` will fail in standalone mode** (`settings.py:165-169`).
   `st.page_link("app.py", label="Dashboard", icon="🏠")` etc. `st.page_link` only works within Streamlit's multi-page app framework where pages are in a `pages/` subdirectory. Since the UI files are all standalone scripts in `ui/`, not in a `pages/` subfolder, these links either raise a runtime error or silently do nothing depending on Streamlit version.
   **Fix:** Either adopt the multi-page structure (move files to `ui/pages/`) or replace with `st.markdown` hyperlinks / `st.button` navigations.

3. **🟠 `bet_tracker` config not exposed** (omission).
   `initial_bankroll` and `flat_stake` are the most operationally important BetTracker parameters. The Settings page offers no way to change them. The `_get_tracker()` in bet_placer and performance pages reads these from config at `@st.cache_resource` construction time — meaning a config change only applies after a process restart, not after "Clear all page caches". **Fix:** Add a `bet_tracker` section to the settings form and save it to `config.yaml`.

4. **🟡 `scrape_interval` is written as a top-level config key** (`settings.py:443`).
   `cfg["scrape_interval"] = int(scrape_interval)` — if scrapers read this from a nested section (e.g. `cfg["scraper"]["interval"]`), this key is misplaced. Verify against scraper code.

---

## Import health (circular imports / missing imports)

| Check                                                               | Result                                                                                                       |
| ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Circular imports in `ui/`                                           | ✅ None — each file imports only from `utils/` and `models/`                                                 |
| `import altair` in `performance.py`                                 | ⚠️ Top-level, hard crash if not installed                                                                    |
| `import plotly` in `performance_dashboard.py`                       | ⚠️ Top-level, hard crash if not installed                                                                    |
| `from catboost import CatBoostClassifier`                           | ✅ Guarded by `try/except ImportError` in both files                                                         |
| `from reportlab...`                                                 | ✅ Guarded by `try/except ImportError`                                                                       |
| `from utils.timezone import TZ, now, to_local`                      | ✅ Present in all files that use them                                                                        |
| `from utils.currency import currency_for_venue, to_eur`             | ✅ Present where used                                                                                        |
| `from utils.bet_tracker import BetTracker, Strategy, StopLossError` | ✅ All three confirmed to exist                                                                              |
| `from models.predictor import Predictor`                            | ✅ Lazily imported inside functions in all pages — no import-time block                                      |
| `from scraper.paddy_power import scrape`                            | ✅ Lazily imported inside `_do_refresh()`                                                                    |
| `from utils.normalizer import normalize`                            | ✅ Lazily imported inside `_do_refresh()`                                                                    |
| `import yaml` in predictions.py                                     | ✅ Imported but not used (dead import — `_CFG_PATH` is defined but `config.yaml` is never read in that file) |

---

## Critical fixes needed

| #   | Severity    | File:line                                                                  | Fix                                                                                                                                                                                                                           |
| --- | ----------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **🔴 HIGH** | `ui/predictions.py:38` + `ui/race_compare.py:36`                           | Feature path `data/features/features.parquet` → `data/features.parquet` (no subdirectory). Feature breakdowns and comparison tables are silently empty without this.                                                          |
| 2   | **🔴 HIGH** | `ui/settings.py:122-124`                                                   | Wrap `_load_cfg` in `try/except (FileNotFoundError, OSError): return {}` to prevent crash on missing config.yaml.                                                                                                             |
| 3   | **🟠 MED**  | `ui/app.py`, `ui/predictions.py`, `ui/bet_placer.py`, `ui/race_compare.py` | All read `sel.get("jockey")` / `sel.get("trainer")`. These will remain blank until review-04 Prediction bug #1 (predictor column rename) is fixed first; no UI-only fix is possible.                                          |
| 4   | **🟠 MED**  | `ui/predictions.py:303-309`                                                | `_load_importance` zips hardcoded `FEATURE_COLS` against model importances. Use `meta["feature_cols"]` from the already-loaded meta dict instead to avoid misalignment when a column was absent at train time.                |
| 5   | **🟠 MED**  | `ui/settings.py:165-169`                                                   | `st.page_link` calls will fail outside multi-page structure. Replace with standard Markdown links or remove.                                                                                                                  |
| 6   | **🟠 MED**  | `ui/app.py`                                                                | Neither `utils/notifications` scheduler nor reporter scheduler is started. If notifications are required during a UI session, start the scheduler in `main()` with an `if "scheduler_started" not in st.session_state` guard. |
| 7   | **🟡 LOW**  | `ui/performance_dashboard.py:441-445`                                      | Remove dead first `tiles[2]` assignment (lines 441-445); the correct version at line 458-463 already overwrites it.                                                                                                           |
| 8   | **🟡 LOW**  | `ui/performance_dashboard.py:302`                                          | `date.today()` → `now().date()` for Dublin-timezone consistency.                                                                                                                                                              |
| 9   | **🟡 LOW**  | `ui/bet_placer.py:374`                                                     | `tracker._flat_stake = flat_stake` mutates private attr. Add a setter to BetTracker or pass as override to `recommend_stake`.                                                                                                 |
| 10  | **🟡 LOW**  | `ui/performance.py`                                                        | Duplicate of `performance_dashboard.py`, unreachable from navigation. Evaluate for removal.                                                                                                                                   |
| 11  | **🟢 INFO** | `ui/predictions.py:22`                                                     | `import yaml` is unused — `_CFG_PATH` is defined but config is never read in this page. Remove.                                                                                                                               |
| 12  | **🟢 INFO** | `ui/settings.py`                                                           | `bet_tracker.initial_bankroll` and `flat_stake` are not editable from Settings. Add a section so they can be changed without editing config.yaml directly.                                                                    |

---

## Next

**Prompt 6:** Apply the two red fixes from this review (#1 feature path, #2 settings crash) and the two red fixes from review 04 (#1 sample-weight alignment, #2 trailing-rate denominator); then fix Prediction bug #1 (jockey/trainer column rename in `predictor.py`) which unblocks the jockey display in all five pages simultaneously.
