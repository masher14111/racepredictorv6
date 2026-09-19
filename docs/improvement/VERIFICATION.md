# Prompt-pack setup verification
Date: 2026-09-18
Scope: documentation, cross-chat memory and its export/check utility. Predictor training/inference code unchanged.

## Verified
- 18 individual copyable prompts, each with explicit model and thinking level above the text block.
- Manifest IDs and dependency ordering validated; all current implementation steps remain PENDING.
- Shared instructions, design, state, decisions, handoff and portable context below 200 physical lines.
- Current memory exporter and checker run successfully using the existing project Python.
- Generated next prompt selects step01 / Codex GPT-6 Astra / Medium.
- Combined prompt pack is regenerated from individual prompts, including explicit XGBoost step13.
- Original DESIGN and CLAUDE bytes verified against SHA-256 hashes.
- An external edit changed active CLAUDE and the first CLAUDE archive during setup; these edits were retained.
- The exact original CLAUDE was recovered from unchanged Git HEAD as CLAUDE.original.md and hash-verified.
- Independent review found missing readiness validation and inconsistent optional-data dispositions; both repaired.
- Optional data/runtime gaps use DEFERRED_DATA, keep the candidate off and require a concrete resume condition.
- Required correctness failures remain BLOCKED/NEEDS_FIX; engineering and predictive evidence remain distinct.
- Negative checks rejected skipped prerequisites, an active stage without an owner, and unresolved optional dispositions.
- All 31 checked documentation files have resolving local links.
- Scoped whitespace checks passed; the final memory check passed with the largest active memory file at 152 lines.

## Commands
```powershell
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
```

## Boundaries
- No numbered implementation prompt has been executed by this setup.
- No datasets, model weights, prediction outputs or betting configurations were changed here.
- No local LLM downloaded; no recurring automation, message or real-money action created.
- Full application tests were not rerun for documentation setup; each implementation prompt specifies its checks.
- A whole-repository whitespace check encountered extensive unrelated pre-existing/parallel changes; those were preserved.
- Active compact memory excludes original archives, long evidence and the combined work-order reference.
