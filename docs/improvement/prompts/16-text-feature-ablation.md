# 16 — Measure whether dated text features improve future forecasts

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 10, 15
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 16 only: Measure whether dated text features improve future forecasts.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10, 15.
Exception: step15 may be DEFERRED_DATA for disabled integration/harness work only; then this experiment must also remain DEFERRED_DATA, not DONE.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. If step13 found a supported improvement, use that frozen numerical candidate; otherwise use step10's verified baseline. If step15 lacks reviewed labels or valid history, implement only the guarded harness and record DEFERRED_DATA.
2. Use the winning or provisional extraction configuration, the immutable archive and numerical benchmark manifests. Join text to canonical race/runner records using source availability at the decision cutoff. Exclude later race reports, edited-after-cutoff comments and records whose eligibility cannot be established.
3. Add a versioned optional text feature group to the existing feature pipeline. Preserve unknown/missing flags and provenance, and separate odds references and subjective tips from factual fields in the independent branch. Keep the default promoted path unchanged.
4. Pre-register a chronological comparison of no text, repaired regex text and the chosen LLM text. Use identical eligible races, model families, folds and reasonable fixed tuning budgets. Report both the common-coverage comparison and full operational coverage so missing text cannot improve results merely by filtering difficult races.
5. Include the hosted DeepSeek candidate only when step15 provides valid reviewed evidence; compare local and hosted configurations on identical eligible races when available. Reuse frozen cached extractions within the same cumulative hosted budget. Retrospective text scores cannot establish freedom from memorized outcomes; retain prospective validation as a distinct requirement.

Acceptance:
- Report incremental race log loss, Brier score, calibration, subgroup counts and whole-race uncertainty across later development windows. Include cost, latency and feature availability at prediction time.
- Test that future comments and subsequent source edits cannot change historical features. Save all candidates separately; do not overwrite promoted artifacts, retrofit retrospective publication times or tune against the final holdout.
- If sufficiently broad timestamped history is absent, complete the guarded integration and reproducible experiment harness, mark forecast validation pending prospective data, and keep text features disabled rather than claiming an improvement.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/16.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/16/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
