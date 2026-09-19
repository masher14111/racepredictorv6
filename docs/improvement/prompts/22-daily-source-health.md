# 22 — Make current declarations and daily source health trustworthy

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 21

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 22 only: Make current declarations and daily source health trustworthy.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Recheck B8 using current files and bounded permitted source reads. Timeform was HTTP403 and its card file stale at 2026-06-13; do not assume a wired scraper has working data. Inspect existing authorized provider paths before adding a source. Do not bypass paywalls/access blocks or purchase access. A licensed feed may need user-supplied access; state that concrete dependency honestly.
2. Make declaration ingestion preserve provider IDs, fetched/published times, race date, runner status and current jockey/trainer/draw/weight/equipment. Explicitly distinguish declared, historical_fallback and missing per field all the way through prediction payloads and UI. Do not turn absent runner status into a claimed confirmed runner. Preserve the null[pyarrow] regression fix.
3. Add reliable stale/partial/error detection and bounded retry/backoff; a failed refresh must preserve good stored data without labelling it current. Exclude ineligible races from formal evidence, while allowing clearly labelled capture-only diagnostics. Blocked providers must not stall the whole cycle indefinitely. Capture failure reasons, coverage/field completeness and timestamps in a daily health report.
4. Wire the existing results and source-comment archive into bounded daily collection where currently permitted and available. Extraction uses stage 19's shared cap and timing contract. Never backdate acquisition of old comments; newly observed historical text is not pre-race evidence.
5. Test stale source, changed declaration, late withdrawal, provider outage, partial field, null-only columns and same-day later refresh. Verify history fallback cannot acquire a declared label or fresh timestamp accidentally.
6. Run at most one bounded source-health smoke after offline tests, with notifications disabled. If legitimate current declarations are unavailable, finish the safe failure handling and configuration/runbook, record B8 OPEN_EXTERNAL with exact feed/credential requirement, and leave affected evaluation eligibility disabled. Engineering DONE must explicitly distinguish that unresolved live-data dependency from data readiness.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/22/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/22.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
