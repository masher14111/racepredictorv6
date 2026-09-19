# 07 — Verify timestamped odds capture and executable-price replay

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 03, 04
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 07 only: Verify timestamped odds capture and executable-price replay.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 04.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Review execution/snapshots.py, current capture loops and pricing consumers. Reuse the existing immutable snapshot store; avoid a second incompatible archive.
2. Make historical and live reads honor the step01 decision cutoff, provider/fetch timestamps, market type, full runner set, freshness and withdrawal status. A price called pre-off can still be later than an earlier decision.
3. Preserve reference-market prices separately from executable quotes and their bookmaker/exchange terms. Closing prices are evaluation-only and must never price a historical fill or decision.
4. Record actual available odds and, where accessible, spread/liquidity. Do not assert a weighted average ppwap/morningwap was a fillable quote at an arbitrary instant; isolate such data as a diagnostic proxy.
5. Test future quote rejection, stale/missing timestamps, mixed-source incomplete books, non-runners, repeated capture idempotency and append-only behavior. A short bounded current capture is enough; do not create a recurring automation.

Acceptance:
- As-of replay cannot access later or stale quotes and preserves correct WIN/PLACE association.
- Reference probabilities, executable odds and closing-line observations remain distinguishable.
- Capture works or fails with an actionable source reason; missing executable history is not presented as realistic backtest evidence.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/07.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/07/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
