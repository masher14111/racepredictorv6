# 23 — Build a consistent shadow model bundle and freeze fresh evaluation

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 22

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 23 only: Build a consistent shadow model bundle and freeze fresh evaluation.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Read B1/B9, stage 10 frozen protocol, stage 17 candidate_manifest/selection evidence and the used-window register. Rebuild a COMPLETE compatible won/placed_2/showed candidate bundle from the corrected feature pipeline into a new isolated directory. Reuse verified immutable matrices and valid checkpoints; do not rerun the entire historical rebuild or a broad tuning sweep without a demonstrated input/version need.
2. Freeze data cutoff, whole-race train/calibration splits, features, hyperparameters, runtime budget and target definitions before fitting. Only historical/development data and already-observed evaluation windows may be used; label all retrospective comparisons as diagnostic. Stage17 consumed the final holdout. Do not rerun its once-only final script, retune on its outcomes or call any old period untouched.
3. Make training and shadow inference use the identical feature definitions/order/transforms and metadata. Validate independent-feature price invariance, complete-field normalization, non-runner handling, target nesting and training/serving parity. Keep market-assisted outputs disclosed. Ensure F-L/reference remapping uses the same reference input as fitting while executable quotes only affect EV; align UI eligibility/EV labels with the canonical gate.
4. Expose a separately tagged shadow prediction route with full bundle/config hashes; do not overwrite, auto-promote or mix targets from the served champion. Preserve its rollback hashes and make corrupt/missing shadow targets fail closed. Capture champion versus candidate provenance distinctly. Text extraction remains sidecar-only pending genuine multi-day ablation; rejected XGBoost is not reintroduced without new evidence.
5. Create an immutable new prospective SHADOW evaluation manifest before new evaluated predictions: exact start instant/run ID, candidate hashes, race eligibility/cutoff, metrics (race log loss/Brier/calibration and separate CLV), market baseline, race/day-clustered uncertainty, data-quality exclusions, review horizon and minimum evidence. Lock rules against outcome-driven changes. Model NO-GO and the formal qualified-bet forward window remain unchanged; this observational cohort is separate.
6. Verify model load/score smoke, provenance and parity offline and run appropriate model/integration tests. Report fit time and actual outputs, not estimates as completion. If a required target cannot be trained correctly, mark NEEDS_FIX/BLOCKED with concrete evidence and stop; never complete this stage with a partial or mixed bundle.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/23/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/23.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
