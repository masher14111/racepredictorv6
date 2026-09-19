# Next prompt — generated from STATE.md

Select the model/effort below, then paste the block in a fresh project chat.

# 21 — Make paper staking, settlement and shadow observations reliable

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 20

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 21 only: Make paper staking, settlement and shadow observations reliable.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Reproduce B2/B3/B4 with the step 17 settlement probe on isolated stores. Trace execution/staking.py, settlement.py, tickets.py, race_facts.py, report.py, gates.py and scripts/daily_paper_loop.py. Wire the existing tested settlement engine into the real daily path, not a parallel calculator.
2. Persist nonzero deterministic stakes only for eligible paper CANDIDATE decisions, honour bankroll/exposure limits, available executable prices, selection-lock odds bands and supported WIN/EW terms. A NO-GO model must continue producing PASS with zero stake. Test GO only via isolated fixtures, never by modifying the real gate or verdict.
3. Handle winner/loser, DNF (loss, not void), explicit non-runner/abandoned race void, Rule 4, dead heats, EW legs and rounding through settle_ticket. Missing/ambiguous facts or unsupported terms stay pending with a reason. Matching includes canonical race and runner, never horse alone. Preserve source facts and settled-at times; restart/repeated result fetch cannot apply P&L twice.
4. Reconcile results and actual closing-price observations for PASS/shadow predictions in a separate observation path. PASS records remain zero-stake non-bets and never count towards qualified-bet thresholds, bankroll, ROI or drawdown. Separate probability scoring, CLV coverage, candidate-ticket P&L and forward-qualified evidence in both payload and UI. Label reference/closing/executable sources and log versus percentage CLV accurately.
5. Make report totals, stakes, returns, drawdown and open/pending counts reconcile exactly to the canonical ledger. Label any ui/bet_placer.py manual ledger as separate and exclude it from validation unless explicitly reconciled. Add local dedupe for notifications/events where this path can repeat them, but do not send any real external message during development or smoke runs.
6. Regression-test real loop issuance-to-settlement and rerun/crash recovery using isolated DBs, with stake/profit arithmetic assertions for all supported cases. Verify the real NO-GO verdict and formal-window state remain unchanged. Record unresolved facts as data gaps, not successful settlements.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/21/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/21.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
