# 08 — Repair settlement identity and forward-report accounting

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 03, 06, 07
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 08 only: Repair settlement identity and forward-report accounting.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 06, 07.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Trace scripts/daily_paper_loop.py and execution settlement/evaluation/report/gate paths. Investigate horse-ID-first historical result lookup and use canonical race-plus-horse identity with verified fallback matching.
2. Separate recorded PASS decisions, qualified candidates, issued paper tickets, fills, voids, settled outcomes and gate-qualified observations. Reconcile the reported open/settled/qualified totals from the underlying ledger.
3. Retain existing Rule4, dead-heat, each-way, commission and non-runner support; test their interactions with identifiers and dates. Make settlement idempotent and fail ambiguous matching rather than selecting another run.
4. Work against a copied candidate ledger before applying any repair to existing records. Preserve an audit trail and original rows; do not fabricate results or rewrite historical prices.
5. Test repeated horses across dates/courses, late results, duplicate result fetches, open PASS records, refunds and missing closing quotes. Rebuild a forward report with correct units and denominators.

Acceptance:
- No result for another race can settle a ticket; repeating settlement has no extra financial effect.
- Decision counts, qualified bets and settlements reconcile without treating PASS tickets as wagers.
- Report retains NO-GO/paper-only and clearly separates no evidence from a passed gate.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/08.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/08/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
