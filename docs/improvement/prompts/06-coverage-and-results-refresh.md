# 06 — Refresh results and repair measurable feature coverage

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 03, 04, 05
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 06 only: Refresh results and repair measurable feature coverage.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 04, 05.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect current artifact dates and result-ingestion paths, then refresh the accessible missing historical window using existing sources with resumable, bounded acquisition and preserved provenance.
2. Measure raw and derived coverage by date, source and racing code. Investigate entirely empty Timeform/class/pace/form fields and the low going_speed fill rate; distinguish parser/join defects from genuinely unavailable source data.
3. Normalize UK/Irish going descriptions without inventing physical speeds. Add missingness indicators or disable unsupported candidate features; never forward-fill future declarations/results into historical races.
4. Repair available feature derivations and canonical result joins, and write refreshed candidate matrices with deterministic schema and coverage reports. Update dependency paths consistently rather than silently replacing the frozen baseline.
5. Implement a reusable coverage/freshness report that detects stale data, all-null features and sudden provider changes. Do not claim every paywalled field must be acquired for the step to succeed.

Acceptance:
- Date-range and field-fill changes are measured before/after, with reasons for every still-unavailable feature.
- Results join to the correct race and horse, historical features remain point-in-time, and schema checks pass.
- Freshness/coverage reports are reproducible; any external acquisition block is recorded separately from completed engineering.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/06.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/06/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
