# 17 — Review the integrated candidate for paper operation

**Model:** Claude Fable 5.1
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 09, 10
**Optional steps requiring a recorded disposition:** 11, 12, 13, 14, 15, 16
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 17 only: Review the integrated candidate for paper operation.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 09, 10.
Also review the recorded disposition of optional steps: 11, 12, 13, 14, 15, 16.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Verify every optional step11..16 already has an evidenced disposition (DONE, EVALUATED_NO_GAIN, or DEFERRED_DATA with candidate off). Do not invent or rewrite another stage's disposition. Missing or contradicted evidence keeps this review NEEDS_FIX/BLOCKED with step17 queued and an exact owning repair stage named; stop for supervised repair routing. A missing timing/text corpus is not completed validation. Required correctness defects block readiness.
2. Review all prerequisite evidence and select the simplest supported candidate; optional experiments lacking data need not block a sound baseline. Trace the existing route from current declarations and timestamped odds through features, probability estimates, candidate/PASS records, results, closing prices and settlement.
3. Freeze one candidate manifest with model/data hashes, features, operating cutoff, calibration, selection threshold, staking rule and supported market terms. Verify freshness, complete runner sets, race-plus-horse settlement keys, quote availability, commission, deductions, non-runners and dead heats wherever the system claims support.
4. Run bounded replay and failure checks in paper mode with isolated artifacts. Define fail-closed behavior for unavailable data or unsupported terms, monitoring, rollback and an operator runbook. Preserve promoted artifacts unless an explicit, evidence-backed promotion procedure records a reversible change.

Acceptance:
- Produce a readiness decision distinguishing technical readiness for paper collection from evidence of predictive improvement or profitability. List open blockers with concrete reproduction details.
- Use a reserved final holdout at most once for the frozen candidate, only if its availability and eligibility are documented. If none remains untouched, say so and require prospective evaluation; never relabel inspected July data as fresh.
- Verify end-to-end identity, provenance, settlement reconciliation and alert behavior. Retain existing minimum evidence gates and keep real-money execution disabled.
- Run the full offline suite and distinguish existing unrelated failures from introduced regressions. Independently verify evidence rather than trusting another model's completion claim.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/17.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/17/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
