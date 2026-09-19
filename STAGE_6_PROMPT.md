# Stage 6 prompt — start live point-in-time capture and the daily paper loop

Successor to `RACE_PREDICTOR_REPAIR_PROMPTS.md` (Stages 1–5, all complete as of
2026-07-27, commits `d7a7058` / `f2d25dd`).

**Claude:** Claude Sonnet 5 (`claude-sonnet-5`) · `high`
**Do NOT use `ultracode` for this stage.** Stage 5 needed multi-agent breadth
because it was inventing a measurement apparatus. Stage 6 is plumbing against an
apparatus that already exists and is tested — the reasoning is bounded, the
contracts are written down, and a workflow fan-out would cost 10–20× the tokens
for no additional correctness.

In a fresh chat:

```text
/model claude-sonnet-5
/effort high
```

Escalate to `claude-opus-5` · `xhigh` **only** if the freeze/reset logic
(requirement 4) or the gap accounting (requirement 5) comes back weak — those are
the two places where a subtle mistake silently manufactures evidence. Everything
else in this stage is wiring.

---

Paste everything below into the fresh chat:

```text
STAGE: 6 — live point-in-time capture and the daily paper loop
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

VERIFIED STARTING FACTS — TREAT AS GIVEN, DO NOT RE-DERIVE
Stages 1-5 are complete. Do not re-audit them; the facts below were verified on
2026-07-27 and are recorded in HANDOFF.md and memory/live-repair-0*.md.

- Branch chore/config-audit. HEAD f2d25dd. Stage 5 feature commit d7a7058
  (47 files, 21517 insertions).
- MODEL GATE: NO-GO. Source data/audit/stage4/final_evaluation.json. The
  price-free line loses to the de-vigged pre-off market by 0.186 race log-loss
  (95% CI excludes zero) and CLV is -13.4%.
- DEPLOYMENT: PAPER-ONLY, enforced in three independent places -
  execution.paper_only clamped True in config loading, CHECK (paper_only = 1) on
  the paper_tickets table, TicketStore.issue raising RealMoneyTicketRefused.
- FORWARD GATE: 9 of 9 criteria failed. 0.0 weeks elapsed, 0 qualified bets.
  execution/forward_gate.py refuses evidence_kind == "backtest" by construction.
- The Stage-5 walk found all three strategies (model_only, devigged_market,
  favourite) with CLV intervals entirely below zero and every ROI interval
  spanning zero. The model was the worst of the three.
- Full test suite: 1966 passed. Run it with `python -m pytest -q`. NEVER pass
  --timeout; pytest-timeout is not installed in this venv.
- The append-only snapshot store already exists and is tested:
  execution/snapshots.py, table odds_snapshots (migration 4 in
  utils/storage/migrations.py), keyed (race_uid, horse_key, bookmaker,
  market_type, fetched_at), RAISE(ABORT) triggers on UPDATE and DELETE,
  latest_quote enforces fetched_at <= as_of in SQL, FuturePriceError on
  violation. You are wiring a live producer into it, not building it.

BOUNDED STARTUP READING - READ THESE AND STOP
Do not read the whole repository. Do not read HANDOFF.md in full (1600+ lines).

1. HANDOFF.md - the Stage 6 section only. Find it with a grep for
   "## Stage 5" and read from there to the end of file.
2. memory/live-repair-05-betting-validation.md (whole file, it is short).
3. execution/snapshots.py, execution/gates.py, execution/tickets.py,
   execution/forward_gate.py, scripts/paper_betting.py.
4. scripts/refresh.py and scripts/daily_clv.cmd - the existing daily entry
   points whose conventions you must follow.
5. The `execution:` block of config.yaml.

Spot-check exactly three claims in code rather than re-verifying the programme:
that paper_only cannot be flipped at runtime; that forward_gate rejects backtest
evidence; that latest_quote cannot return a row newer than as_of. If any of the
three is false, stop and report STAGE 6 BLOCKED with the evidence.

CONTEXT AND STANDING CONSTRAINTS
This is paper betting only. No real money is ever staked. The model gate is
NO-GO and nothing in this stage may change that, route around it, or soften it.
"No bet" remains a valid and expected output on every card.

The Stage-5 backtest could model price movement but could not measure it - the
historical warehouse holds one pre-off quote per runner. This stage exists to
start measuring it for real. Every price captured now is reusable by any future
model and cannot be reconstructed later, which is why capture starts immediately
and independently of whether the current model is worth validating.

Be clear-eyed about what this is worth: an eight-week forward window on a model
already measured NO-GO will most likely confirm the NO-GO. The capture
infrastructure is the durable deliverable. Do not write copy that implies the
forward window is expected to produce a GO.

TASK
1. LIVE CAPTURE. Wire execution.snapshots into the live scrape path so every
   poll appends an immutable row per runner per bookmaker per market. Multiple
   polls per race across the day must produce a real price series, not one row
   overwritten. Capture must be additive and must not change what the existing
   scrapers return to their current callers - if capture fails, the scrape still
   succeeds and the failure is recorded.

2. DAILY LOOP. One command that is idempotent and safe to re-run any number of
   times on the same day: capture, run the PASS-by-default gate over today's
   card, issue paper tickets for anything that passes, settle yesterday's open
   tickets against results, recompute forward metrics, write the dated reports.
   Re-running must not double-issue tickets, double-settle, or double-count.

3. SCHEDULING. Register it for unattended daily execution following the existing
   scripts/daily_clv.cmd convention. Document the exact schtasks command in
   HANDOFF.md. Do not silently install a scheduled task - print the command and
   let the user run it.

4. WINDOW FREEZE AND RESET. Write data/execution/forward_window.json recording
   the window start date, content hashes of the frozen model files, the hash of
   selection_lock.json, and a hash of the execution config block. On every run,
   re-verify those hashes. If any changed, the validation window RESETS to zero
   and both the report and the dashboard must say so prominently, naming what
   changed. A model retrain or a threshold edit invalidates accumulated forward
   evidence - the system must enforce that mechanically rather than trusting
   whoever made the change to remember. Starting the formal validation window
   must be a deliberate, explicit action, separate from starting capture.

5. GAP ACCOUNTING. A day with no capture is a GAP, not a day with no qualifying
   bets. The two are not interchangeable and conflating them inflates the
   evidence base. Record every gap with its cause (scraper outage, machine off,
   no racing, partial card). Gaps must appear in the report and must NOT count
   toward the forward gate's minimum-weeks or qualified-days requirements.

6. NO BACKFILL, EVER. A missed decision point stays missed. Never write a
   snapshot with a fetched_at earlier than the moment it was actually observed,
   never settle a decision against a price captured after the decision time, and
   never reconstruct a skipped day from later data. This is the single defect
   that would make every forward number meaningless.

7. SETTLEMENT AND CLV. Settle real paper tickets against results, and compute
   CLV against the closing price captured live rather than an archive proxy.
   Where the closing price was not captured, the ticket's CLV is unmeasurable
   and must be reported as such, not silently dropped or substituted.

8. REPORTING. The forward-validation report and dashboard must now show real
   paper evidence in Lane 2, with the capture history, gap ledger, window state
   (running / reset / not started), and days elapsed against the eight-week
   requirement. Keep the three lanes separate. Expect FORWARD GATE NOT MET and
   render it plainly.

TESTS AND ACCEPTANCE
1. Test: idempotent re-run issues and settles nothing twice; a gap is recorded
   as a gap and does not satisfy the gate; a hash change resets the window and
   says why; no snapshot can be written with a backdated fetched_at; capture
   failure does not break the scrape; unmeasurable CLV is reported, not dropped.
2. Run the targeted execution and UI tests first. Run the full suite ONCE at the
   end: python -m pytest -q. Do not re-run the full suite between edits.
3. Run the daily loop end to end twice against the same day and show that the
   second run is a no-op.

TOKEN DISCIPLINE
Do not use subagents, the Agent tool, workflows, or ultracode. Do not re-read
files you just edited - the harness tracks them. Do not print large file
contents into the transcript to "verify" them. Prefer targeted greps over full
reads. Ask before any exploration that would exceed the bounded reading list.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
1. Append a dated Stage 6 section to HANDOFF.md. Preserve Stages 1-5 exactly.
2. Record: files and schemas changed, the capture wiring points, the scheduling
   command, the window-freeze hash set, gap-accounting rules, commands and
   tests, reports produced, unresolved risks, and confirmation that deployment
   remains PAPER-ONLY.
3. Create memory/live-repair-06-forward-capture.md with standard YAML
   frontmatter (metadata type: project) and verified facts only.
4. Add its pointer line to memory/MEMORY.md.
5. Record the commit hash.
6. Create a scoped commit: `feat: start live point-in-time capture and daily
   paper loop`. Never mix unrelated files - the working tree contains untracked
   user work (.claude/, .design-md/, images/, review/, tools/, data/audit/,
   data/backups/, implementation_plan.md, v4prompts.md,
   RACE_PREDICTOR_REPAIR_PROMPTS.md, tests/test_train_lgbm.py, and several
   reports/*) which must be left exactly as found.
7. End with:
   - STAGE 6 PASS/FAIL/BLOCKED
   - MODEL GO/NO-GO
   - CAPTURE ACTIVE/INACTIVE
   - FORWARD WINDOW NOT STARTED/RUNNING (day N of 56)/RESET
   - DEPLOYMENT PAPER-ONLY/CANDIDATE-ELIGIBLE
```

---

## Why Sonnet 5 · high, and what it saves

Stage 5 ran Opus 5 at `ultracode` and cost a large multiple of what this stage
needs. Three things drive the difference:

1. **The apparatus exists.** Stage 5 had to invent the measurement contracts;
   Stage 6 wires a producer into a store that already enforces its own
   invariants in SQL and triggers. The correctness burden sits in code that is
   already written and tested.
2. **The facts are handed over.** The Stage-5 prompt spent an enormous amount of
   context re-verifying four prior stages from scratch. This prompt states the
   verified findings up front and asks for three targeted spot-checks instead.
3. **The reading is bounded.** `HANDOFF.md` alone is now 1600+ lines. The prompt
   names the files and the ranges, and forbids exploratory sweeps.

Escalate to Opus 5 `xhigh` if requirements 4 or 5 come back weak. Those are the
two places where a plausible-looking implementation quietly manufactures
evidence — a window that fails to reset on a model change, or a gap counted as a
clean no-bet day, both make the eight-week gate passable without eight weeks of
honest data.

## What Stage 6 is deliberately not

Not model improvement. If you intend to change features, retrain, or re-select a
strategy, that is a separate stage and it resets any validation window that has
started — which is exactly why requirement 4 enforces the reset mechanically.
Capture, by contrast, is model-independent: start it now, because a price not
recorded today cannot be recovered tomorrow.
