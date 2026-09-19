# 03 — Preserve WIN and PLACE market association when fusing data

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 01, 02
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 03 only: Preserve WIN and PLACE market association when fusing data.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 01, 02.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Trace scraper/normalizer -> features/fuse.py -> builder -> CatBoost/LightGBM training. Recheck current WIN/PLACE observations; the review found mixed market rows, not duplicate runner keys.
2. Build a consistent canonical race/runner identity and merge shared horse attributes without collapsing market-specific price, field, result or terms provenance. Attach WIN odds to win models and retain separate PLACE terms/data for place evaluation.
3. Make CatBoost and LightGBM consume equivalent valid WIN populations for win comparisons; do not let arbitrary same-source row order pick the market. Preserve non-runner and complete-field contracts.
4. Version any changed dataset schema and rebuild into candidate outputs first. Maintain compatible serving inputs or an explicit migration, without modifying old frozen artifacts in place.
5. Add adversarial tests with one horse in WIN and PLACE, swapped source ordering, repeated races/venues, conflicting IDs and partial books. Check sample weights and implied probabilities inherit the correct market.

Acceptance:
- Reordering source rows does not change market assignment; no place price or terms contaminate a win record.
- Canonical uniqueness and source/market lineage are demonstrated on fixtures and sampled current data.
- Both win-training lines use the same eligible-race rules, and place data remain correctly available.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/03.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/03/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
