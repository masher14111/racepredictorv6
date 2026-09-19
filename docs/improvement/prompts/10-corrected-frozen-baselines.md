# 10 — Build comparable baselines and freeze the experiment protocol

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 09
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 10 only: Build comparable baselines and freeze the experiment protocol.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 09.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Freeze a run manifest before training: data/config/code hashes, eligible racing population, intended decision cutoff, development folds, calibration windows, untouched final-window policy and candidate output paths.
2. Using repaired data/features, train comparable market-only baselines, a simple race conditional-logit baseline, existing CatBoost and grouped-softmax LightGBM. Compare independent and market-assisted branches honestly on identical eligible races.
3. Use bounded matched tuning budgets and race-level log loss as the primary development score, with Brier/reliability and uncertainty. Compare existing class/odds weights with defensible unweighted/race-weighted alternatives inside development folds.
4. Do not consume the reserved final holdout yet: step17 evaluates the frozen selection once. If later data have already been inspected or adequate final data do not exist, record that and reserve prospective evaluation instead.
5. Keep candidate models, calibrators, schemas and metadata in distinct directories; verify reload/inference parity. Save out-of-sample development predictions for later feature/blend comparisons and record missing market execution evidence.

Acceptance:
- Reproducible baseline scorecard and out-of-sample predictions share one eligible-race/cutoff protocol.
- Final-test policy is frozen and its data have not been used to choose hyperparameters or thresholds.
- Champion artifacts are preserved; performance may remain NO-GO and is reported without threshold manipulation.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/10.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/10/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
