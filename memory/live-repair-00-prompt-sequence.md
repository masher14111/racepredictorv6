---
name: live-repair-00-prompt-sequence
description: Five sequential fresh-chat prompts for repairing live market ingestion, proxy freshness, full-field probability/EV, calibration validity, and realistic paper-betting safeguards
metadata:
  type: project
---

# Five-stage live-odds and betting repair prompt sequence

Created `RACE_PREDICTOR_REPAIR_PROMPTS.md` as the canonical five-stage repair
sequence. This is a prompt/workflow artifact only; it does **not** claim that any
of the five implementation stages has passed.

Every stage is intended to run in a fresh chat against the same repository,
sequentially:

1. Primary WIN-market ingestion and contaminated-selection rejection.
2. Rotating proxy, source freshness, stale-cache, and atomic snapshot handling.
3. Full-field probability, reference-market, executable-price, and EV
   reconciliation.
4. Leak-free calibration/model-validity audit with an explicit GO/NO-GO.
5. Point-in-time execution, forward validation, conservative paper staking, and
   deployment safeguards.

Because fresh chats cannot see previous conversations, each prompt mandates:

- reading and verifying `HANDOFF.md`, git status/history/diffs, reports, tests,
  and earlier `memory/live-repair-*` notes;
- stopping when a required earlier stage has not genuinely passed;
- preserving existing/unrelated changes;
- appending a factual handoff and a YAML-frontmatter memory note;
- updating `memory/MEMORY.md`;
- running earlier regressions plus the current stage’s acceptance suite; and
- creating only a safely scoped stage commit.

Recommended model/effort choices are embedded for both Codex and Claude. The
workflow is intentionally fail-closed: model NO-GO or negative/uncertain forward
evidence keeps the product paper-only and allows honest PASS/no-bet output.

Related project truth remains authoritative: [[model-14-backtest]],
[[model-16-value-detection]], [[calib-fl-04-backtest]], and
[[calib-fl-05-final]] already document negative CLV and a paper-only verdict.
Future stage notes must report newly measured evidence rather than carrying those
figures forward without revalidation.

