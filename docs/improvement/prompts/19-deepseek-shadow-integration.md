# 19 — Integrate capped OpenRouter DeepSeek into daily shadow extraction

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 18, 15

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 19 only: Integrate capped OpenRouter DeepSeek into daily shadow extraction.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Inspect llm/hosted_adapter.py, hosted_budget.py, text_features.py, text_archive.py, scripts/benchmark_extraction.py, scripts/verify_hosted_pricing.py, stage 15 scorecards and the actual daily acquisition path. Reuse the implemented backend rather than adding a competing client or ledger. Record what the existing benchmark actually establishes and its one-day/label-review limitations.
2. Add an explicit configurable OpenRouter shadow extractor using deepseek/deepseek-v4.1-flash. Wire dated archived comments to it in the bounded daily workflow; enable shadow extraction with the existing key when available. Keep numeric forecast influence disabled and llm.enabled false if that switch enables unvalidated model features. Store source identity, original text hash, publication/acquisition time, extraction availability time, schema/prompt/model versions and grounded evidence. A post-cutoff extraction must never masquerade as pre-race information.
3. Reuse data/cache/llm/hosted_budget_ledger.json and existing caches. The US$5/1,200-request ceiling is cumulative across the old benchmark and ALL new cycles/retries, not renewed per day/stage. Inspect actual balances (last recorded $0.2836 and 498 requests are historical), verify current provider pricing and reserve worst-case input/output/reasoning/retry cost before any request. Restrict provider routing to the verified ceilings. Never reset accounting, print credentials, buy credits or substitute a model silently.
4. Fail closed on unavailable key, funds, provider, pricing, timeout, malformed/ungrounded output or exhausted caps. Use clearly labelled regex/unknown results as appropriate; a hosted outage must not crash paper collection or claim hosted success. Implement resumable dedupe, bounded latency and a visible non-secret extraction/budget health summary. Do not send the whole database to a provider; submit only selected source comments needed for extraction.
5. Prove cache/retry/resume accounting, concurrent reservation safety, missing credential, malformed evidence, exhaustion and numerical-probability invariance using offline tests. One minimal live smoke is authorized within the remaining shared cap after verification; use cached output where available. Report actual live-call/cost status separately. Missing external access can be recorded as an integration limitation only after all offline acceptance is met; it is not evidence that hosted extraction works.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/19/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/19.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
