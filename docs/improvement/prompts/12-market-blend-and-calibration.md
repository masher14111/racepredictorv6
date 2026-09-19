# 12 — Learn market combination and calibrate race probabilities

**Model:** Claude Opus 5
**Thinking level:** High
**Required prior steps:** 10
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 12 only: Learn market combination and calibrate race probabilities.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Use the repaired independent-model and market-only benchmark outputs, canonical WIN records and chronological split manifest. Audit the existing calibration and price-adjustment code before extending it; preserve the distinction between independent probabilities, reference market probabilities and executable quotes.
2. Fit a small probability combination, such as a normalized power blend, using chronological out-of-sample development predictions. Select blend settings on development data, then fit probability calibration on the later reserved calibration period. Keep the final test period untouched during all selection.
3. Compare a small predetermined set of simple calibrators and an uncalibrated control. Normalize valid WIN probabilities within each complete race. Save candidate coefficients, calibration artifacts, input hashes and cutoff rules in a separate experiment directory; do not replace promoted artifacts.

Acceptance:
- Compare market-only, independent-only and combined probabilities on identical eligible races. Report log loss, Brier score and calibration by odds band, field size, racing regime and selected-bet subset, with sample counts.
- Demonstrate that changing only an executable bookmaker quote cannot change estimated horse ability or the reference probability. Test missing/incomplete reference books and non-runners explicitly.
- Record insufficient calibration coverage as a limitation. A disappointing candidate is a valid result; do not repeatedly alter methods after inspecting holdout results.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/12.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/12/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
