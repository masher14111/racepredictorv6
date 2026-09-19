# 18 — Start frozen paper collection and hand over daily operation

**Model:** Codex — GPT-6 Astra
**Thinking level:** Medium
**Required prior steps:** 17
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 18 only: Start frozen paper collection and hand over daily operation.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 17.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Read the paper-readiness decision and frozen manifest. If technical blockers remain, repair only bounded integration issues that do not change the evaluated model; document any change requiring renewed validation. Do not start a formal evidence window with an unready pipeline.
2. When technical readiness and the existing formal-start conditions pass, record the actual start time and immutable prospective-run identifier. If the model gate remains NO-GO, retain capture-only/shadow observation and leave formal qualification/start pending according to existing rules. Run one bounded current paper capture cycle and settle only already verified results.
3. Write an operations handoff: collect, check freshness, review decisions, reconcile results and closing prices, inspect failures, pause and resume. Include manual scheduling instructions if needed, but do not create an automation, leave an unbounded process running or enable real-money betting.

Acceptance:
- Verify the first cycle's actual records, hashes and timestamps; report exactly what ran, which races were eligible and which decisions/results are still pending. If no live eligible race exists, complete a labelled dry run and leave the formal start pending.
- Document the configured gates, including at least eight weeks, 200 qualified bets and 150 races where still applicable. A single session cannot satisfy future-duration requirements, and PASS tickets are not bets.
- Keep rules frozen during evaluation, preserve promoted artifacts and report deviations. State that collection has started or remains blocked; do not declare validation or profitability complete.
- Engineering setup can be DONE while Model verdict stays NO-GO and Forward validation remains PENDING. Record these independently; never override gates to manufacture qualified bets.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/18.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/18/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
