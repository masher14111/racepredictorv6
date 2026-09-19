# 04 — Repair independent features and historical feature invariance

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 02, 03
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 04 only: Repair independent features and historical feature invariance.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 02, 03.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect features/engine.py, derive.py, builder.py and models/features.py. Repair race_complexity's dataset-global standardization and market entropy in the nominally independent branch.
2. Use train-fitted parameters or race-local quantities where appropriate. Audit rolling form, trainer/jockey aggregates, target encodings and transforms for actual availability before the intended cutoff.
3. Keep market-assisted features in an explicit branch; disclose any market-informed training weights instead of calling them feature leakage automatically. Test unweighted/race-weighted alternatives later, not by silently changing today's model.
4. Add append-future invariance and historical as-of replay checks, including same-day races, late-published results and missing values. Verify train/inference feature parity.
5. Version changed feature definitions; write candidate feature artifacts. Keep old model bundles bound to their original feature schema rather than serving them incompatible new inputs.

Acceptance:
- Appending later observations does not change earlier admissible feature values.
- Independent features do not depend on odds, market shape or outcomes unavailable at the cutoff.
- Feature schemas, required preprocessing and serving compatibility are explicit and tested.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/04.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/04/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
