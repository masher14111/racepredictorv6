# 09 — Independently audit the repaired data and evaluation foundations

**Model:** Claude Opus 5
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 02, 03, 04, 05, 06, 07, 08
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 09 only: Independently audit the repaired data and evaluation foundations.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 02, 03, 04, 05, 06, 07, 08.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Act as an independent reviewer. Read actual diffs, code and stage evidence for steps02..08 instead of accepting their summaries. Reproduce important claims and challenge plausible hidden failures.
2. Inspect race/market identity, split/calibration contamination, as-of availability, missing-field/fallback behavior, snapshot pricing, settlement idempotency and result/report reconciliation.
3. Run adversarial regression checks and an appropriate full offline suite. Check non-runner/full-field invariants, independent/market feature provenance and preservation of unrelated user changes.
4. Write a severity-ranked report with code locations, reproductions and acceptance verdicts. Fix small clear scoped defects with checks. For substantial defects, keep this audit NEEDS_FIX or BLOCKED, queue step09, and name the earlier owning repair stage in its evidence; stop the supervisor for supervised repair routing. Do not silently rewrite previously completed stage statuses or approve the audit while required defects remain.
5. Update the shared contract if evidence requires clarification, and record unresolved external-data limitations. Approve the foundation only when required correctness criteria pass; do not infer profitability from engineering tests.

Acceptance:
- A reviewer-reproduced foundation verdict exists with actual check results and remaining data limits.
- Unresolved correctness failures block dependent model comparisons and identify an exact repair step.
- Ready model work has a consistent data/split contract and no unsupported claims of completed forward validation.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/09.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/09/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
