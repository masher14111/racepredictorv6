# 13 — Test XGBoost as a third numerical model and measure ensemble gain

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 10, 12
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 13 only: Test XGBoost as a third numerical model and measure ensemble gain.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10, 12.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Use measured-performance features from step11 only if that candidate has valid evidence. Missing sectionals do not block the unchanged verified baseline comparison.
2. Inspect the existing CatBoost, grouped-softmax LightGBM and older ensemble code, together with the repaired benchmarks and market combination. Reuse the current training and prediction interfaces rather than building a parallel production pipeline.
3. Add XGBoost explicitly as the third boosted-tree candidate alongside CatBoost and grouped-softmax LightGBM, reusing models/ensemble_experiment.py where sound. Keep regularized conditional logit and market-only as controls. Match eligible races, features, chronological folds and tuning budgets. Fit a simple blend before a stacker using chronological out-of-fold predictions only.
4. Compare each individual learner, CatBoost+LightGBM, and CatBoost+LightGBM+XGBoost on the same later development windows. Measure out-of-sample error diversity and the incremental gain from adding/removing XGBoost. Learn blend weights from development predictions; allow XGBoost zero weight instead of forcing an equal third.
5. Keep independent and market-assisted comparisons explicit. Calibrate the combined race probabilities on the reserved calibration slice. Register the experiment before fitting; write candidate artifacts separately and preserve the final holdout for step17. If using ranking scores, validate their probability conversion explicitly.
6. XGBoost is a numerical learner, not a coding assistant. Use CPU or sequential GPU training within the existing hardware budget; loading/training all three simultaneously is not required. Record runtime, inference latency and memory.

Acceptance:
- Produce a comparable score table with race log loss, Brier score, calibration, sample counts and whole-race or whole-day bootstrap intervals. Avoid treating runners from one race as independent observations.
- If coherent place probabilities are already supported, check win/top-two/top-three ordering and applicable paid-place terms; describe unsupported place markets honestly without expanding this into a separate new modelling project.
- Report runtime, memory, missing-feature robustness and uncertainty limitations. Recommend the simplest supported candidate, including retaining the current baseline when improvements are absent or inconclusive.
- Report the measured delta from two-model to three-model blending, its race/day-clustered interval, calibration and stability across development periods. Retain XGBoost only if the evidence supports it; no guaranteed accuracy/profit increase.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/13.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/13/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
