# 11 — Pilot measured speed, sectionals and normalized performance

**Model:** Claude Opus 5
**Thinking level:** High
**Required prior steps:** 10
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 11 only: Pilot measured speed, sectionals and normalized performance.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect the repaired feature pipeline, source inventory and benchmark manifests. The existing horse_speed is a finishing-position proxy; preserve its meaning and add separately named measured-performance fields only where actual historical times, margins or sectionals exist.
2. Implement a small experiment using course, distance, surface and going adjustments. Fit pars and transformations on earlier training races only. Define units, non-finisher handling, missingness and publication timestamps; prohibit current-race outcomes in pre-race features.
3. If testing a duration predictor, generate its downstream training features from chronological out-of-fold predictions. Keep it behind an experimental switch in the existing pipeline and write to a new experiment directory. Do not overwrite promoted artifacts or tune against the reserved holdout.

Acceptance:
- Report coverage by date/source/race type, unit checks, timestamp eligibility and invariance of historical features when future records are appended.
- Compare the unchanged baseline and the added feature family on the same development windows using race log loss, Brier score and whole-race uncertainty. Retain a feature only with documented evidence.
- If trustworthy timing data are unavailable, deliver the tested ingestion contract and coverage report, mark the experiment DEFERRED_DATA with Candidate enabled: no and a concrete Resume condition, and avoid fabricating timings, buying data or claiming measured gains.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/11.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/11/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
