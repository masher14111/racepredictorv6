# 05 — Acquire current race declarations with source provenance

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 03, 04
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 05 only: Acquire current race declarations with source provenance.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 04.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect existing racecard/enrichment providers before adding another source. Populate actual declared jockey/trainer, carried weight/claims, draw, age/sex, rating/class, surface/going, equipment and runner status where legitimate available sources provide them.
2. Keep provider IDs and fetched/published times; reconcile them with canonical race/horse identity. Preserve source freshness and field-level conflicts rather than merging unrelated current and past facts.
3. Repair the live last-known-connections behavior: today's declared jockey must not be invented from the previous run. Store declared and historical-fallback values distinctly with availability/quality flags.
4. Use available authorized source access and saved fixtures; do not purchase subscriptions or treat inaccessible/paywalled fields as populated. Finish the source adapter and missing-data behavior independently of unavailable credentials.
5. Add parser/normalizer fixtures for late jockey changes, non-runners, conflicting cards and absent fields. Run a bounded live coverage check separately from the offline test suite if sources are accessible.

Acceptance:
- At least one real available card can be traced through IDs/timestamps to inference, or the external block is documented precisely.
- Current declarations take precedence; last-known fallback remains distinguishable and missingness stays honest.
- Coverage, parsing checks and unavailable fields are reported without fabricated completeness.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/05.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/05/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
