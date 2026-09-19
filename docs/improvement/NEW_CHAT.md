# Reusable new-chat starter

**Model / thinking:** Select the recommendation shown in the current
[memory/improvement/NEXT_PROMPT.md](../../memory/improvement/NEXT_PROMPT.md).
For the first run: **Codex — GPT-6 Astra / Medium**.
Paste this same block in each new project chat; its instructions read the latest files.

```text
Continue the Race Predictor v4 improvement programme using its filesystem memory.
Project Git root: C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Use my deliberately opened worktree instead if applicable; verify its data/output paths.

Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read the latest memory/improvement/NEXT_PROMPT.md and its numbered source prompt.
Execute that ONE ready step completely, including implementation and relevant verification.
If a stage is ACTIVE/NEEDS_FIX, resume its recorded work before starting another.
Verify prerequisites and current code; do not rely on assumed prior-chat memory.
Preserve unrelated changes, original model artifacts and paper-only gates.
Never claim absent data, unrun tests, improved forecasts or future validation are complete.

Before finishing, record stage evidence, update STATE/DECISIONS/HANDOFF as appropriate,
keep each active memory file below 200 lines, and run the memory export and check utility.
Report results, remaining limits, and the next prompt number/model/thinking level.
If you cannot access the repository, say so and request the current context/prompt/source
files; do not pretend to read, edit or test my local files.
```
