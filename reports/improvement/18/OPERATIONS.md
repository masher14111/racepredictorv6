# Operations handoff — daily paper capture (step 18)

PAPER-ONLY, capture-only. Model verdict **NO-GO** (`data/audit/stage4/final_evaluation.json`).
No formal forward window is running (`window_status: NOT_STARTED`). Nothing here places a bet,
starts a schedule, or changes the evaluated model. Full technical detail, fail-closed proof and
rollback procedure: `reports/improvement/17/RUNBOOK.md` and `READINESS.md` — this file is the
day-to-day checklist plus what today's worked cycle actually showed.

All commands run from the repo root with `.venv\Scripts\python.exe`.

## 1. Collect (once per day, manually — do not schedule from a code prompt)

```
.\.venv\Scripts\python.exe -m scripts.refresh
.\.venv\Scripts\python.exe -m scripts.daily_paper_loop --no-scrape
```

`scripts.refresh` scrapes odds + declared cards, normalises, and writes `data/predictions.json`.
`scripts.daily_paper_loop` reads that file, gates every runner, issues PASS/CANDIDATE tickets,
settles anything already resolved, and writes the day's report. Combine as one manual session;
do not leave a scheduler invoking these unattended (AGENTS.md: no new recurring jobs from a code
prompt). If you want OS-level scheduling later, that is a human decision outside this session's
authorization — Windows Task Scheduler running the two commands above, once daily, after UK/IRE
racing has been declared (mid-morning Europe/Dublin) is the natural slot, but nothing here creates it.

## 2. Check freshness before trusting a cycle

| Check | Command / file | Healthy | Today (2026-09-19) |
|---|---|---|---|
| Served bundle unchanged | sha256 vs `reports/improvement/17/candidate_manifest.json::serving_status` | identical | **verified identical**, 11/11 files |
| `data/predictions.json::generated_at` | today, minutes old | fresh | 2026-09-19T15:17:47+01:00 |
| `data/source_health.json` | each source `ok`, age < 900s | ok | livescorebet/paddy_power ok; boylesports `ok` but 2.7% request success (proxy blocks retried, not fatal) |
| `data/historical/timeform.parquet` max `race_date` | today | fresh | **still 2026-06-13** — B8 open, declared cards NOT flowing (403 on racecards this cycle too) |
| `data/execution/gap_ledger.json` | today `captured` | captured | 2026-09-19 `captured` |

## 3. Review decisions

Read `data/execution/daily_loop_run.json` (`deployment`, `capture_active_today`, per-runner
`decision`) and the day's `reports/forward_validation_YYYYMMDD.md`. Every decision must be `PASS`
while the model verdict is NO-GO — a `CANDIDATE` under NO-GO would itself be a defect (guarded by
the step-17 clamp on `execution.gates.require_model_validation`). PASS tickets are disclosure
records, never bets, and never count toward the forward gate's qualified-bet total.

## 4. Reconcile results and closing prices

`scripts.daily_paper_loop` settles already-resolved races against SP on the same run. Read
`data/execution/daily_loop_run.json::issue.settled_count` / `still_open_count` /
`unmeasurable_clv_count`. Do not hand-edit `paper_tickets`, `odds_snapshots` or `text_archive`
(append-only triggers). B2-B4 (no stake sizing, settlement bypasses `settle_ticket`, PASS never
settled) stay dormant while every decision is PASS — they activate the moment a verdict turns GO
and must be fixed by a future work order before that evidence would count.

## 5. Inspect failures

- Refresh log ends without "predicted N races" → it exited having written nothing; do **not** run
  the loop against a stale `predictions.json` (re-tickets yesterday's card, see B6).
- A source in `data/source_health.json` reads `stale`/`erroring` → every runner it prices becomes a
  PASS with `source_health:*`; expected under NO-GO, not itself a bug.
- `timeform.parquet` date not advancing → B8 stays open; treat every jockey/trainer field shown as
  a historical fallback, not a live declaration.
- Any traceback during `[4/5] predict` → check `models/predictor.py::_is_non_runner`-style dtype
  issues first (fixed once this stage, see stage 18 note); an all-null pyarrow-backed column is the
  known failure class.

## 6. Pause and resume

- **Pause:** stop running section 1's two commands. A skipped day is logged as a gap in
  `gap_ledger.json` and costs the eventual formal window a qualifying day — it does not corrupt
  anything.
- **Resume:** run section 1 again. Re-running the same day's loop is idempotent (0 newly issued,
  N duplicates suppressed).
- **Rollback:** unchanged from `reports/improvement/17/RUNBOOK.md` section 6 — one config line
  (`value.model_tag: v3nf`) reverts any future promotion; nothing this stage did touches serving.

## 7. What today's cycle actually was

One bounded capture cycle, run manually in this session, 2026-09-19:

- `scripts.refresh`: 38 races / 432 runners scored (2 attempts — first crashed on a genuine live
  defect, fixed below, then re-run cleanly).
- `scripts.daily_paper_loop --no-scrape`: 432 tickets issued, 0 duplicates, all `PASS`
  (`model_validation:NO-GO`), 0 settled this cycle, forward gate 9/9 criteria failed (unchanged
  from before this cycle — no criterion can pass on a single day).
- Ledger total after this cycle: 1,420 tickets (988 prior + 432 today), all `paper_only=1`, 0
  `CANDIDATE`, 0 settled — over 3 capture days (2026-07-28, 2026-09-18, 2026-09-19; one prior gap
  day 2026-07-28 recorded as `scraper_outage` in the gap ledger despite having tickets — pre-dates
  this stage, not investigated further here).
- No formal window started (`execution.window.start_window` never called). Formal qualification
  stays pending: model verdict NO-GO, and even a GO verdict today could not satisfy the 8-week /
  200-qualified-bet / 150-race duration floors in a single session.

## 8. Configured forward-release gates (unchanged, floored in code)

From `reports/forward_validation_20260919.md` — all 9 failed today because tracking has not
started, not because of any single bad reading:

| criterion | required |
|---|---|
| model_go | Stage-4 model verdict == GO |
| min_weeks | >= 8 weeks of forward tracking |
| min_qualified_bets | >= 200 qualified bets |
| min_qualified_races | >= 150 qualified races |
| positive_mean_clv | mean CLV > 0 |
| clv_ci_lower | 95% race-clustered CI lower bound > 0 |
| ae_stable | 0.90 <= A/E <= 1.10 |
| calibration | ECE <= 0.03 |
| drawdown | max drawdown <= 10% of bankroll |

A single session cannot satisfy the duration/count floors regardless of outcome. PASS tickets do
not count as qualified bets.
