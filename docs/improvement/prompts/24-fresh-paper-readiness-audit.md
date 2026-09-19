# 24 — Audit daily operation and hand off fresh paper evaluation

**Model:** Claude Opus 5
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 19, 20, 21, 22, 23

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 24 only: Audit daily operation and hand off fresh paper evaluation.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Independently inspect implementations and fresh evidence from 19-23. Re-run the original B1-B10 repros where applicable and publish each disposition: fixed with evidence, still open, or external data gap. Do not inherit previous green tests as proof. Reproduce critical acceptance through the actual daily entrypoints on isolated stores before any live operation.
2. Test a complete cycle, repeated same-cycle invocation, interrupted/resumed cycle, source outage, stale/already-off data, capped hosted outage, model-hash mismatch and delayed settlement. Required checks include no duplicate canonical decisions, no changed old evidence, correct stake/result/CLV reconciliation, NO-GO means zero candidate bets, and separately scored PASS observations. Run the full test suite serially at this review milestone.
3. Repair bounded integration defects within this stage with regression checks and re-audit. A large unresolved correctness defect must leave this stage NEEDS_FIX with exact file/repro/repair scope, not DONE. Do not change earlier stage ledger statuses or manufacture a green operational verdict. External provider gaps may remain declared only when excluded/fail-closed behavior itself is verified.
4. Provide a single practical daily command/launcher that collects current data, records grounded extraction, creates eligible frozen shadow predictions, reconciles available results/closing quotes and writes a clear health/evaluation summary. Include a run lock, bounded timeouts, atomic checkpoint, resumable stages and actionable failure output; do not silently repeat expensive training daily. Test it headlessly with no external notifications. No recurring scheduler/service is created by this coding stage.
5. After readiness checks, run one bounded real capture-only cycle with notifications disabled and prove its counts/hashes/timestamps from actual stored records. Use the new prospective manifest, never backfill older records into its cohort. If no race is eligible now or an indispensable live feed is missing, report that precise condition and use a labelled offline replay; do not claim fresh observations exist.
6. Write reports/improvement/24/READINESS.md and OPERATIONS.md with exact daily start/status/resume commands, what the user sees, safe recovery and remaining external actions. Distinguish engineering completion, source readiness, shadow collection started, model acceptance and prospective evidence complete. Include the shared hosted budget balance and how to disable hosted calls. Record the next review date/horizon from the frozen protocol; real calendar time and qualified sample sizes cannot be completed in a coding session.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/24/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/24.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
