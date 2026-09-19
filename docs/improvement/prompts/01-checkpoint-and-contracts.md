# 01 — Checkpoint the project and agree data/evaluation contracts

**Model:** Codex — GPT-6 Astra
**Thinking level:** Medium
**Required prior steps:** None
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 01 only: Checkpoint the project and agree data/evaluation contracts.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: none.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect the current Git state, Python environment, relevant tests, sources and saved model/forward reports. Record a timestamped baseline manifest in reports/improvement/01/ without resetting or sweeping up the user's existing changes.
2. Reproduce the key dated review findings on current inputs: shared race IDs across split boundaries, market-type counts, date/feature coverage, text coverage and decision/settlement totals. Separate historical evidence from today's observations. This step is an audit and contract, not a model rewrite.
3. Write docs/improvement/CONTRACTS.md below 200 lines: canonical race/horse/market identity, timezone and publication/fetch/as-of semantics, independent versus market features, missingness, candidate artifact ownership, complete-race splitting and final-test policy.
4. Select the intended prediction cutoff for the experiments from existing configuration; if unspecified, document a provisional ten-minutes-before-off research cutoff as an assumption, without silently changing the live service.
5. Record available data rights/access and sample coverage. Mark unavailable timing/text/odds history as data gaps, not a reason to invent values. Choose bounded offline baseline checks and record actual results.

Acceptance:
- Current dirty-tree and environment fingerprints, dated measurements and commands are saved; no unrequested application changes.
- One shared contract resolves the identity/timing terminology used by later prompts; no secrets copied.
- Memory lists genuinely verified facts, external-data gaps and the exact next step.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/01.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/01/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
