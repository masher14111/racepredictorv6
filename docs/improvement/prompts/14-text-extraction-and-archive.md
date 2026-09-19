# 14 — Repair text extraction and preserve dated source comments

**Model:** Claude Sonnet 5
**Thinking level:** Medium
**Required prior steps:** 01, 07
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 14 only: Repair text extraction and preserve dated source comments.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 01, 07.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect llm/text_features.py, its configuration, tests and the Spotlight scraper. Repair explicit Boolean parsing, negation and the ground-excuse pattern that can misread 'suited by the soft ground'. Preserve unknown separately from false and do not infer an unmentioned fact.
2. Define and validate a strict extraction schema with evidence phrases, source identifiers and source availability times. Require extracted claims to be supported by the supplied text. Record the backend that actually produced each result, including regex fallback, and keep failure reasons visible.
3. Archive incoming text append-only with content hashes, observation/publication timestamps where available, source, race and runner keys. Cache by text hash plus actual backend/model, prompt and schema versions. Integrate with existing scraper and extractor interfaces while keeping predictive text features disabled pending evaluation.

Acceptance:
- Add focused tests for false strings, true strings, unknowns, negation, contradictory comments, missing evidence, schema failures and correctly labelled fallback cache records.
- Show that two versions of a comment remain retrievable and that later edits cannot silently alter historical pre-race inputs. Unknown historical publication times must stay unknown.
- Report the actual archive coverage; the review found only 243 comments from one day, so do not claim a historical corpus exists. Preserve original files and promoted numerical artifacts.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/14.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/14/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
