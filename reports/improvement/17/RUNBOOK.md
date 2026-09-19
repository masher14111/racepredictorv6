# Paper-operation runbook (step 17)

PAPER-ONLY. Model verdict **NO-GO**: every decision is a PASS disclosure record, not a bet. Nothing here
places a bet, starts a schedule, or starts the formal forward window. All commands run from the repo root
(`C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4`) with `.venv\Scripts\python.exe`.

## 1. Before a cycle (2 minutes)

| Check | Command | Expect |
|---|---|---|
| Served bundle unchanged | compare sha256 of `models/*.bin`, `*_calib.pkl`, `lgbm_won_v3.txt` to `candidate_manifest.json::serving_status.currently_served` | identical; if not, STOP — the evaluated model changed |
| Paper-only on | `config.yaml` -> `execution.paper_only: true` | true (absent/unreadable also loads as true) |
| Verdict source present | `data/audit/stage4/final_evaluation.json` exists | missing/corrupt loads as NO-GO, never GO |
| Config not loosened | any `execution config tightened —` line in the log | none expected; a clamp line means someone loosened YAML |

Model binaries are **gitignored**: `git status` can never show they changed. Use the hashes.

## 2. One collection cycle (manual; do not schedule from a code prompt)

1. `python -m scripts.refresh` — scrape odds + declared cards (network), normalise, predict -> `data/predictions.json`.
2. `python -m scripts.daily_paper_loop --no-scrape` — gap ledger, gate, issue PASS records, settle, report.

Isolated rehearsal that touches no real store: `python reports/improvement/17/06_paper_replay.py --label <name>`
(writes only under `data/audit/17/replay/<name>/`; proves isolation by hashing the real stores).

## 3. After a cycle — what to read

| Artifact | Healthy looks like | Act if not |
|---|---|---|
| `data/predictions.json::generated_at` | today, minutes old | refresh wrote NOTHING when every source was stale — it exits 0 and prints "predicted 0 races" but leaves the OLD file. Do not run the loop on it (it would re-ticket yesterday's card: B6) |
| `data/execution/daily_loop_run.json` | `deployment: PAPER-ONLY`, `issue.issued` ~ runners, every decision PASS, `capture_active_today: true` | `capture_active_today: false` = a GAP day; it costs the window a qualifying week by design |
| `data/execution/gap_ledger.json` | today `captured` | a gap is recorded, never back-filled |
| `data/source_health.json` | each odds source `ok`, age < 900 s | a stale/erroring source makes every runner it prices a PASS (`source_health:*`) |
| `data/historical/timeform.parquet` max `race_date` | today | if still 2026-06-13, declared cards are NOT flowing (B8): every jockey/trainer shown is a historical fallback |
| PASS reasons in `paper_tickets.pass_reasons` | `model_validation:NO-GO` on all; few `executable_price:*` | many `executable_price:age_unknown` = snapshot store not being written; many `race_gate:*` = partial/stale books |
| Duplicate physical races | each venue+field once | same horses at off-times a minute apart = B5; treat both ticket sets as one race |

Read-only ledger summary: `reports/improvement/17/09_live_store_readonly.json` shows the query set
(decision counts, `paper_only != 1` must be 0, settled count, snapshot date range, future-stamped rows must be 0).

## 4. Fail-closed rules (verified 30/30 in `06_paper_replay_after_fix.json`)

Unavailable data is a PASS reason, never a waiver: stale quote (>900 s), future-stamped quote, unreadable
timestamp, consensus/SP/Timeform price (no named bookmaker), incomplete reference book, unpriced runner,
stale / erroring / missing source health, only a market-adjusted probability, field < 5, book sum > 1.25,
edge < 0.02, EV < 0.05, verdict missing/corrupt/NO-GO. `gates.require_model_validation` cannot be switched
off from YAML (clamped at load, step 17).

**Unsupported terms — never treat as evidence:** Rule 4 deductions, dead heats, non-runner voids, each-way/place,
exchange commission, stake sizing and P&L are NOT applied by the live loop (B2/B3). PLACE odds are not captured.

## 5. Alerts

Only external channel: Telegram (`utils/notifications.py`), active only if a token and chat id are supplied
locally; `RP_DISABLE_NOTIFICATIONS=1` or running under pytest disables it. Race-soon / new-top-pick / odds-drop
are de-duplicated for 12 h (`data/cache/sent_alerts.json`); **bet-settled and stop-loss are not**, and a
fingerprint is marked sent before delivery, so a failed send is not retried. The drift/retrain trigger
(`models/retrain_trigger.py`) has **no caller** — drift surfaces only through the static stage-4 verdict reasons.
There is no alert for a gap day or a stale card: read section 3 every cycle.

## 6. Pause, resume, rollback

- **Pause:** stop running the two commands. Nothing runs unattended; a missed day is logged as a gap.
- **Resume:** run section 1, then section 2. Re-running the loop the same day is idempotent (0 issued, N duplicates).
- **Rollback of a future promotion:** set `value.model_tag: v3nf` in `config.yaml`. The procedure in
  `candidate_manifest.json` only ever COPIES files under a new tag, so the served champion is never overwritten.
- **Rollback of step 17's two code fixes:** revert the `require_model_validation` clamp in `execution/config.py`
  and the `format="mixed"` / `unreadable_price_timestamp` lines in `models/predictor.py::_race_ev_gate`. Both
  only ever turn a candidate into a PASS; reverting re-opens a fail-open path, so do not.
- **Never:** set `execution.paper_only: false`, lower any `execution.forward_gate` value (8 weeks / 200 qualified
  bets / 150 races / CLV CI > 0 / A/E 0.90-1.10 / ECE <= 0.03 / drawdown <= 10% are floored in code), call
  `execution.window.start_window` while NO-GO, or edit a row in `odds_snapshots` / `text_archive` (append-only triggers).

## 7. When does this stop being capture-only?

Only after, in order: (1) B1-B3 closed by their own work orders; (2) a candidate beats the de-vigged market on
PROSPECTIVE data — the last untouched holdout was consumed on 2026-09-19 and it failed; (3) `execution.model_gate`
returns GO on that evidence; (4) a formal window is started and then satisfies every forward-gate criterion.
A single session can satisfy none of the duration requirements.
