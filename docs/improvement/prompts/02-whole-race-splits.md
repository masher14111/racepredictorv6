# 02 — Repair chronological splits throughout training and tuning

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 01
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 02 only: Repair chronological splits throughout training and tuning.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 01.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect models/train.py, models/tuner.py, backtest/splitter.py and model-specific calibration/early-stopping paths. Replace row-index boundaries with a shared splitter over whole race/date blocks consistent with the step01 contract.
2. Reserve disjoint chronological fit, early-stopping/tuning, calibration and outer-test periods. Ensure hyperparameter selection cannot consume the reserved calibration/final labels; feature selectors and transforms fit inside the appropriate earlier slice.
3. Audit existing TimeSeriesSplit use and ensemble out-of-fold predictions: time ordering alone does not preserve race groups. Do not require all appearances of a horse to stay in one partition; prior-to-future horse history is realistic.
4. Preserve complete race fields, handle sparse periods, equal dates and unknown race keys explicitly. Return split manifests with actual IDs/cutoffs and determinism independent of input row order.
5. Use offline synthetic edge cases plus the current matrix to reproduce the previous overlap and demonstrate the repair. Do not launch a full production retrain here; reserve that comparison for step10.

Acceptance:
- Zero shared race IDs across every fit/tune/early-stop/calibration/test boundary, with strictly admissible date ordering.
- Shuffling rows, multiple races on one day and repeated horses do not change partition membership improperly.
- Both production training and relevant evaluation/tuning paths use the repaired contract, with regression tests and split evidence.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/02.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/02/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
