# 15 — Benchmark local and hosted LLM extraction against repaired regex

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 14
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 15 only: Benchmark local and hosted LLM extraction against repaired regex.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 14.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Use the repaired strict schema, immutable comments and available runtime. Compare repaired regex, local Qwen3.5 9B and hosted deepseek/deepseek-v4.1-flash via OpenRouter. Hermes 4 14B is an optional local challenger; do not expand into a large model tournament. Verify actual model/provider identities and measured resources; do not silently substitute models or call model-card claims benchmark results. Reuse installed local models first; scoped setup may fetch the named quantized models from official/trusted publishers when disk/runtime permit.
2. Build a reproducible labelled evaluation set targeting 500–1,000 distinct comments across dates and sources, including rare events, negations and unknowns. Separate prompt-development and blind evaluation records, prevent duplicate leakage, and retain evidence-span annotations. Automated labels need an explicit review status rather than being called human gold labels.
3. Freeze extraction prompts and schemas before scoring repaired regex and each available model. Use bounded context and structured output, cache actual backend provenance, and schedule local inference separately from GPU training. Keep predictions and promoted numerical models unchanged.
4. Implement a provider-neutral hosted adapter with strict output validation, evidence spans, true/false/unknown, timeouts, bounded retries and per-record caching by text/model/provider/prompt/schema version. Do not silently cache fallback results as DeepSeek. No browsing or tools for extraction; suppress unnecessary identities, require source-supported fields and test identity substitutions to reduce historical-outcome memorization risks.
5. Read hosted_trial limits from docs/improvement/runner.json. Across all attempts/resumes, enforce US$5 total and 1,200 requests maximum using a persistent ledger and conservative pre-request cost reservation including maximum output/reasoning tokens and retries. Enforce configured provider price ceilings; fail closed if cost cannot be bounded. Record actual billed usage, never reset spend on resume, and never buy/top-up credits. Prove budget enforcement using mocked responses before live calls.
6. Use OPENROUTER_API_KEY from the process environment or Windows user environment, never from prompt text; never print/store its value in logs or memory. Missing key, exhausted credits/budget or absent labels means a runnable offline harness and an explicit deferred hosted result. Continue any valid local comparison, distinguishing per-backend status from whole-stage acceptance.

Acceptance:
- Report per-field precision, recall, false-positive counts, unknown handling, evidence support, schema validity, latency and peak memory, with denominators and model/runtime versions.
- Report actual cost per 1,000 comments, input/output/reasoning tokens, failure/retry counts and tail latency; test lowest supported reasoning against any higher-effort candidate only within the same cap.
- If the corpus, reviewed labels or a requested model cannot be obtained, deliver the runnable harness and completed subset results, state the exact gap, and mark the comparison DEFERRED_DATA with Candidate enabled: no and a concrete Resume condition. Reserve BLOCKED for an implementation/correctness failure. Never manufacture gold labels or a winning model.
- Select a provisional extractor only on the frozen extraction evaluation; better extraction does not yet establish better racing predictions.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/15.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/15/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
