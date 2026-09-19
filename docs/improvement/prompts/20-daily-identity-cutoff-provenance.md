# 20 — Repair daily race identity, operating cutoff and prediction provenance

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 19

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 20 only: Repair daily race identity, operating cutoff and prediction provenance.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Reproduce B5/B6/B7 from reports/improvement/17/READINESS.md on isolated fixtures/copies. Inspect utils/normalizer.py, features/fuse.py, execution/snapshots.py, tickets.py, race_facts.py, storage migrations and the daily loop before choosing one compatible canonical identity contract.
2. Resolve a physical race across source off-time disagreements using venue/date and reliable source IDs/runner evidence. Never blindly round neighbouring races into one. Preserve WIN/PLACE association; different venues at the same instant must stay distinct. Ambiguous mappings must be quarantined with a reason rather than guessed.
3. Implement an additive, versioned, restart-safe mapping/migration for persisted snapshots/tickets/results/text references. Preserve historical identifiers and immutable payloads; no silent rewriting of prior research artifacts or destructive DB reset. Back up the live store before any authorized local migration, verify row counts and integrity, and document rollback. Prove repeated migration/capture is idempotent and legacy ambiguity fails closed.
4. Wire configured started-race buffer and real wall-clock decision cutoff into daily issuance. Reject prior-day predictions, stale/future/unreadable timestamps and already-started races, including Dublin DST/midnight cases. Keep actual prediction issuance separate from any date key used for dedupe. Do not regenerate old tickets with today's timestamp.
5. Attach model artifact content hashes, feature/schema version, configuration hash, prediction/cycle ID, source as-of times and decision time to fresh predictions and decisions; versions must resolve to a saved manifest. Missing or inconsistent provenance must exclude formal eligibility. Preserve additive compatibility with existing UI readers.
6. Test the actual daily loop entrypoint on isolated stores: same-time different venues, one-minute source disagreement, adjacent distinct races, repeat run/restart, stale-next-day replay, DST, mixed timestamps and changed model content. Assert unique tickets, correctly associated markets, no retroactive issuance and truthful provenance. Record before/after audit repro results.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/20/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/20.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
