# All 24 implementation prompts

Choose the model and thinking setting above a prompt, then paste its text into a fresh project chat.
Read START_HERE.md first. This long work-order collection is not a compact memory file.

# 01 — Checkpoint the project and agree data/evaluation contracts

**Model:** Codex — GPT-6 Astra
**Thinking level:** Medium
**Required prior steps:** None
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 01 only: Checkpoint the project and agree data/evaluation contracts.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: none.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect the current Git state, Python environment, relevant tests, sources and saved model/forward reports. Record a timestamped baseline manifest in reports/improvement/01/ without resetting or sweeping up the user's existing changes.
2. Reproduce the key dated review findings on current inputs: shared race IDs across split boundaries, market-type counts, date/feature coverage, text coverage and decision/settlement totals. Separate historical evidence from today's observations. This step is an audit and contract, not a model rewrite.
3. Write docs/improvement/CONTRACTS.md below 200 lines: canonical race/horse/market identity, timezone and publication/fetch/as-of semantics, independent versus market features, missingness, candidate artifact ownership, complete-race splitting and final-test policy.
4. Select the intended prediction cutoff for the experiments from existing configuration; if unspecified, document a provisional ten-minutes-before-off research cutoff as an assumption, without silently changing the live service.
5. Record available data rights/access and sample coverage. Mark unavailable timing/text/odds history as data gaps, not a reason to invent values. Choose bounded offline baseline checks and record actual results.

Acceptance:
- Current dirty-tree and environment fingerprints, dated measurements and commands are saved; no unrequested application changes.
- One shared contract resolves the identity/timing terminology used by later prompts; no secrets copied.
- Memory lists genuinely verified facts, external-data gaps and the exact next step.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/01.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/01/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 02 — Repair chronological splits throughout training and tuning

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 01
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 02 only: Repair chronological splits throughout training and tuning.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 01.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect models/train.py, models/tuner.py, backtest/splitter.py and model-specific calibration/early-stopping paths. Replace row-index boundaries with a shared splitter over whole race/date blocks consistent with the step01 contract.
2. Reserve disjoint chronological fit, early-stopping/tuning, calibration and outer-test periods. Ensure hyperparameter selection cannot consume the reserved calibration/final labels; feature selectors and transforms fit inside the appropriate earlier slice.
3. Audit existing TimeSeriesSplit use and ensemble out-of-fold predictions: time ordering alone does not preserve race groups. Do not require all appearances of a horse to stay in one partition; prior-to-future horse history is realistic.
4. Preserve complete race fields, handle sparse periods, equal dates and unknown race keys explicitly. Return split manifests with actual IDs/cutoffs and determinism independent of input row order.
5. Use offline synthetic edge cases plus the current matrix to reproduce the previous overlap and demonstrate the repair. Do not launch a full production retrain here; reserve that comparison for step10.

Acceptance:
- Zero shared race IDs across every fit/tune/early-stop/calibration/test boundary, with strictly admissible date ordering.
- Shuffling rows, multiple races on one day and repeated horses do not change partition membership improperly.
- Both production training and relevant evaluation/tuning paths use the repaired contract, with regression tests and split evidence.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/02.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/02/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

# 04 — Repair independent features and historical feature invariance

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 02, 03
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 04 only: Repair independent features and historical feature invariance.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 02, 03.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect features/engine.py, derive.py, builder.py and models/features.py. Repair race_complexity's dataset-global standardization and market entropy in the nominally independent branch.
2. Use train-fitted parameters or race-local quantities where appropriate. Audit rolling form, trainer/jockey aggregates, target encodings and transforms for actual availability before the intended cutoff.
3. Keep market-assisted features in an explicit branch; disclose any market-informed training weights instead of calling them feature leakage automatically. Test unweighted/race-weighted alternatives later, not by silently changing today's model.
4. Add append-future invariance and historical as-of replay checks, including same-day races, late-published results and missing values. Verify train/inference feature parity.
5. Version changed feature definitions; write candidate feature artifacts. Keep old model bundles bound to their original feature schema rather than serving them incompatible new inputs.

Acceptance:
- Appending later observations does not change earlier admissible feature values.
- Independent features do not depend on odds, market shape or outcomes unavailable at the cutoff.
- Feature schemas, required preprocessing and serving compatibility are explicit and tested.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/04.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/04/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

# 06 — Refresh results and repair measurable feature coverage

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 03, 04, 05
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 06 only: Refresh results and repair measurable feature coverage.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 04, 05.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect current artifact dates and result-ingestion paths, then refresh the accessible missing historical window using existing sources with resumable, bounded acquisition and preserved provenance.
2. Measure raw and derived coverage by date, source and racing code. Investigate entirely empty Timeform/class/pace/form fields and the low going_speed fill rate; distinguish parser/join defects from genuinely unavailable source data.
3. Normalize UK/Irish going descriptions without inventing physical speeds. Add missingness indicators or disable unsupported candidate features; never forward-fill future declarations/results into historical races.
4. Repair available feature derivations and canonical result joins, and write refreshed candidate matrices with deterministic schema and coverage reports. Update dependency paths consistently rather than silently replacing the frozen baseline.
5. Implement a reusable coverage/freshness report that detects stale data, all-null features and sudden provider changes. Do not claim every paywalled field must be acquired for the step to succeed.

Acceptance:
- Date-range and field-fill changes are measured before/after, with reasons for every still-unavailable feature.
- Results join to the correct race and horse, historical features remain point-in-time, and schema checks pass.
- Freshness/coverage reports are reproducible; any external acquisition block is recorded separately from completed engineering.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/06.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/06/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

# 08 — Repair settlement identity and forward-report accounting

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 03, 06, 07
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 08 only: Repair settlement identity and forward-report accounting.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 03, 06, 07.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Trace scripts/daily_paper_loop.py and execution settlement/evaluation/report/gate paths. Investigate horse-ID-first historical result lookup and use canonical race-plus-horse identity with verified fallback matching.
2. Separate recorded PASS decisions, qualified candidates, issued paper tickets, fills, voids, settled outcomes and gate-qualified observations. Reconcile the reported open/settled/qualified totals from the underlying ledger.
3. Retain existing Rule4, dead-heat, each-way, commission and non-runner support; test their interactions with identifiers and dates. Make settlement idempotent and fail ambiguous matching rather than selecting another run.
4. Work against a copied candidate ledger before applying any repair to existing records. Preserve an audit trail and original rows; do not fabricate results or rewrite historical prices.
5. Test repeated horses across dates/courses, late results, duplicate result fetches, open PASS records, refunds and missing closing quotes. Rebuild a forward report with correct units and denominators.

Acceptance:
- No result for another race can settle a ticket; repeating settlement has no extra financial effect.
- Decision counts, qualified bets and settlements reconcile without treating PASS tickets as wagers.
- Report retains NO-GO/paper-only and clearly separates no evidence from a passed gate.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/08.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/08/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 09 — Independently audit the repaired data and evaluation foundations

**Model:** Claude Opus 5
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 02, 03, 04, 05, 06, 07, 08
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 09 only: Independently audit the repaired data and evaluation foundations.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 02, 03, 04, 05, 06, 07, 08.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Act as an independent reviewer. Read actual diffs, code and stage evidence for steps02..08 instead of accepting their summaries. Reproduce important claims and challenge plausible hidden failures.
2. Inspect race/market identity, split/calibration contamination, as-of availability, missing-field/fallback behavior, snapshot pricing, settlement idempotency and result/report reconciliation.
3. Run adversarial regression checks and an appropriate full offline suite. Check non-runner/full-field invariants, independent/market feature provenance and preservation of unrelated user changes.
4. Write a severity-ranked report with code locations, reproductions and acceptance verdicts. Fix small clear scoped defects with checks. For substantial defects, keep this audit NEEDS_FIX or BLOCKED, queue step09, and name the earlier owning repair stage in its evidence; stop the supervisor for supervised repair routing. Do not silently rewrite previously completed stage statuses or approve the audit while required defects remain.
5. Update the shared contract if evidence requires clarification, and record unresolved external-data limitations. Approve the foundation only when required correctness criteria pass; do not infer profitability from engineering tests.

Acceptance:
- A reviewer-reproduced foundation verdict exists with actual check results and remaining data limits.
- Unresolved correctness failures block dependent model comparisons and identify an exact repair step.
- Ready model work has a consistent data/split contract and no unsupported claims of completed forward validation.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/09.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/09/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 10 — Build comparable baselines and freeze the experiment protocol

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 09
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 10 only: Build comparable baselines and freeze the experiment protocol.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 09.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Freeze a run manifest before training: data/config/code hashes, eligible racing population, intended decision cutoff, development folds, calibration windows, untouched final-window policy and candidate output paths.
2. Using repaired data/features, train comparable market-only baselines, a simple race conditional-logit baseline, existing CatBoost and grouped-softmax LightGBM. Compare independent and market-assisted branches honestly on identical eligible races.
3. Use bounded matched tuning budgets and race-level log loss as the primary development score, with Brier/reliability and uncertainty. Compare existing class/odds weights with defensible unweighted/race-weighted alternatives inside development folds.
4. Do not consume the reserved final holdout yet: step17 evaluates the frozen selection once. If later data have already been inspected or adequate final data do not exist, record that and reserve prospective evaluation instead.
5. Keep candidate models, calibrators, schemas and metadata in distinct directories; verify reload/inference parity. Save out-of-sample development predictions for later feature/blend comparisons and record missing market execution evidence.

Acceptance:
- Reproducible baseline scorecard and out-of-sample predictions share one eligible-race/cutoff protocol.
- Final-test policy is frozen and its data have not been used to choose hyperparameters or thresholds.
- Champion artifacts are preserved; performance may remain NO-GO and is reported without threshold manipulation.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/10.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/10/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 11 — Pilot measured speed, sectionals and normalized performance

**Model:** Claude Opus 5
**Thinking level:** High
**Required prior steps:** 10
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 11 only: Pilot measured speed, sectionals and normalized performance.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Inspect the repaired feature pipeline, source inventory and benchmark manifests. The existing horse_speed is a finishing-position proxy; preserve its meaning and add separately named measured-performance fields only where actual historical times, margins or sectionals exist.
2. Implement a small experiment using course, distance, surface and going adjustments. Fit pars and transformations on earlier training races only. Define units, non-finisher handling, missingness and publication timestamps; prohibit current-race outcomes in pre-race features.
3. If testing a duration predictor, generate its downstream training features from chronological out-of-fold predictions. Keep it behind an experimental switch in the existing pipeline and write to a new experiment directory. Do not overwrite promoted artifacts or tune against the reserved holdout.

Acceptance:
- Report coverage by date/source/race type, unit checks, timestamp eligibility and invariance of historical features when future records are appended.
- Compare the unchanged baseline and the added feature family on the same development windows using race log loss, Brier score and whole-race uncertainty. Retain a feature only with documented evidence.
- If trustworthy timing data are unavailable, deliver the tested ingestion contract and coverage report, mark the experiment DEFERRED_DATA with Candidate enabled: no and a concrete Resume condition, and avoid fabricating timings, buying data or claiming measured gains.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/11.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/11/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

# 13 — Test XGBoost as a third numerical model and measure ensemble gain

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 10, 12
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 13 only: Test XGBoost as a third numerical model and measure ensemble gain.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10, 12.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Use measured-performance features from step11 only if that candidate has valid evidence. Missing sectionals do not block the unchanged verified baseline comparison.
2. Inspect the existing CatBoost, grouped-softmax LightGBM and older ensemble code, together with the repaired benchmarks and market combination. Reuse the current training and prediction interfaces rather than building a parallel production pipeline.
3. Add XGBoost explicitly as the third boosted-tree candidate alongside CatBoost and grouped-softmax LightGBM, reusing models/ensemble_experiment.py where sound. Keep regularized conditional logit and market-only as controls. Match eligible races, features, chronological folds and tuning budgets. Fit a simple blend before a stacker using chronological out-of-fold predictions only.
4. Compare each individual learner, CatBoost+LightGBM, and CatBoost+LightGBM+XGBoost on the same later development windows. Measure out-of-sample error diversity and the incremental gain from adding/removing XGBoost. Learn blend weights from development predictions; allow XGBoost zero weight instead of forcing an equal third.
5. Keep independent and market-assisted comparisons explicit. Calibrate the combined race probabilities on the reserved calibration slice. Register the experiment before fitting; write candidate artifacts separately and preserve the final holdout for step17. If using ranking scores, validate their probability conversion explicitly.
6. XGBoost is a numerical learner, not a coding assistant. Use CPU or sequential GPU training within the existing hardware budget; loading/training all three simultaneously is not required. Record runtime, inference latency and memory.

Acceptance:
- Produce a comparable score table with race log loss, Brier score, calibration, sample counts and whole-race or whole-day bootstrap intervals. Avoid treating runners from one race as independent observations.
- If coherent place probabilities are already supported, check win/top-two/top-three ordering and applicable paid-place terms; describe unsupported place markets honestly without expanding this into a separate new modelling project.
- Report runtime, memory, missing-feature robustness and uncertainty limitations. Recommend the simplest supported candidate, including retaining the current baseline when improvements are absent or inconclusive.
- Report the measured delta from two-model to three-model blending, its race/day-clustered interval, calibration and stability across development periods. Retain XGBoost only if the evidence supports it; no guaranteed accuracy/profit increase.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/13.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/13/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

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
---

# 16 — Measure whether dated text features improve future forecasts

**Model:** Codex — GPT-6 Astra
**Thinking level:** High
**Required prior steps:** 10, 15
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 16 only: Measure whether dated text features improve future forecasts.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 10, 15.
Exception: step15 may be DEFERRED_DATA for disabled integration/harness work only; then this experiment must also remain DEFERRED_DATA, not DONE.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. If step13 found a supported improvement, use that frozen numerical candidate; otherwise use step10's verified baseline. If step15 lacks reviewed labels or valid history, implement only the guarded harness and record DEFERRED_DATA.
2. Use the winning or provisional extraction configuration, the immutable archive and numerical benchmark manifests. Join text to canonical race/runner records using source availability at the decision cutoff. Exclude later race reports, edited-after-cutoff comments and records whose eligibility cannot be established.
3. Add a versioned optional text feature group to the existing feature pipeline. Preserve unknown/missing flags and provenance, and separate odds references and subjective tips from factual fields in the independent branch. Keep the default promoted path unchanged.
4. Pre-register a chronological comparison of no text, repaired regex text and the chosen LLM text. Use identical eligible races, model families, folds and reasonable fixed tuning budgets. Report both the common-coverage comparison and full operational coverage so missing text cannot improve results merely by filtering difficult races.
5. Include the hosted DeepSeek candidate only when step15 provides valid reviewed evidence; compare local and hosted configurations on identical eligible races when available. Reuse frozen cached extractions within the same cumulative hosted budget. Retrospective text scores cannot establish freedom from memorized outcomes; retain prospective validation as a distinct requirement.

Acceptance:
- Report incremental race log loss, Brier score, calibration, subgroup counts and whole-race uncertainty across later development windows. Include cost, latency and feature availability at prediction time.
- Test that future comments and subsequent source edits cannot change historical features. Save all candidates separately; do not overwrite promoted artifacts, retrofit retrospective publication times or tune against the final holdout.
- If sufficiently broad timestamped history is absent, complete the guarded integration and reproducible experiment harness, mark forecast validation pending prospective data, and keep text features disabled rather than claiming an improvement.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/16.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/16/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 17 — Review the integrated candidate for paper operation

**Model:** Claude Fable 5.1
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 09, 10
**Optional steps requiring a recorded disposition:** 11, 12, 13, 14, 15, 16
**Fallback:** If this exact Claude model is unavailable, use Claude Opus 5 or Codex GPT-6 Astra at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 17 only: Review the integrated candidate for paper operation.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 09, 10.
Also review the recorded disposition of optional steps: 11, 12, 13, 14, 15, 16.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Verify every optional step11..16 already has an evidenced disposition (DONE, EVALUATED_NO_GAIN, or DEFERRED_DATA with candidate off). Do not invent or rewrite another stage's disposition. Missing or contradicted evidence keeps this review NEEDS_FIX/BLOCKED with step17 queued and an exact owning repair stage named; stop for supervised repair routing. A missing timing/text corpus is not completed validation. Required correctness defects block readiness.
2. Review all prerequisite evidence and select the simplest supported candidate; optional experiments lacking data need not block a sound baseline. Trace the existing route from current declarations and timestamped odds through features, probability estimates, candidate/PASS records, results, closing prices and settlement.
3. Freeze one candidate manifest with model/data hashes, features, operating cutoff, calibration, selection threshold, staking rule and supported market terms. Verify freshness, complete runner sets, race-plus-horse settlement keys, quote availability, commission, deductions, non-runners and dead heats wherever the system claims support.
4. Run bounded replay and failure checks in paper mode with isolated artifacts. Define fail-closed behavior for unavailable data or unsupported terms, monitoring, rollback and an operator runbook. Preserve promoted artifacts unless an explicit, evidence-backed promotion procedure records a reversible change.

Acceptance:
- Produce a readiness decision distinguishing technical readiness for paper collection from evidence of predictive improvement or profitability. List open blockers with concrete reproduction details.
- Use a reserved final holdout at most once for the frozen candidate, only if its availability and eligibility are documented. If none remains untouched, say so and require prospective evaluation; never relabel inspected July data as fresh.
- Verify end-to-end identity, provenance, settlement reconciliation and alert behavior. Retain existing minimum evidence gates and keep real-money execution disabled.
- Run the full offline suite and distinguish existing unrelated failures from introduced regressions. Independently verify evidence rather than trusting another model's completion claim.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/17.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/17/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

# 18 — Start frozen paper collection and hand over daily operation

**Model:** Codex — GPT-6 Astra
**Thinking level:** Medium
**Required prior steps:** 17
**Fallback:** Use the available Codex coding model at the same supported effort.

Select the model/effort in the application. Paste the entire block below into a new chat with repository access.

```text
Work in the Race Predictor v4 Git repository at:
C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4
If I deliberately opened an isolated worktree, use that worktree's Git root and verify its data/output paths.

Execute improvement step 18 only: Start frozen paper collection and hand over daily operation.
Read AGENTS.md, CLAUDE.md, DESIGN.md, memory/improvement/STATE.md,
memory/improvement/DECISIONS.md and memory/improvement/HANDOFF.md.
Read this numbered prompt, the shared contract if created, and prerequisite stage evidence.
Required prerequisites: 17.
Recheck current code/artifacts before acting on dated review observations.
Preserve unrelated dirty changes; follow shared paper-only, evidence and artifact-isolation rules.
If this stage is already evidenced complete, verify and report that rather than blindly rerunning it.
If a required prerequisite fails, record the exact gap and next repair without pretending it passed.

Scope:
1. Read the paper-readiness decision and frozen manifest. If technical blockers remain, repair only bounded integration issues that do not change the evaluated model; document any change requiring renewed validation. Do not start a formal evidence window with an unready pipeline.
2. When technical readiness and the existing formal-start conditions pass, record the actual start time and immutable prospective-run identifier. If the model gate remains NO-GO, retain capture-only/shadow observation and leave formal qualification/start pending according to existing rules. Run one bounded current paper capture cycle and settle only already verified results.
3. Write an operations handoff: collect, check freshness, review decisions, reconcile results and closing prices, inspect failures, pause and resume. Include manual scheduling instructions if needed, but do not create an automation, leave an unbounded process running or enable real-money betting.

Acceptance:
- Verify the first cycle's actual records, hashes and timestamps; report exactly what ran, which races were eligible and which decisions/results are still pending. If no live eligible race exists, complete a labelled dry run and leave the formal start pending.
- Document the configured gates, including at least eight weeks, 200 qualified bets and 150 races where still applicable. A single session cannot satisfy future-duration requirements, and PASS tickets are not bets.
- Keep rules frozen during evaluation, preserve promoted artifacts and report deviations. State that collection has started or remains blocked; do not declare validation or profitability complete.
- Engineering setup can be DONE while Model verdict stays NO-GO and Forward validation remains PENDING. Record these independently; never override gates to manufacture qualified bets.

Finish this work order, including relevant verification and fixes, rather than stopping at a plan.
Do not start the next numbered step, invent unavailable data, or report unrun checks as passed.
Write memory/improvement/stages/18.md with files, commands/results, versions,
acceptance, unresolved limits and exact next action; link long output under reports/improvement/18/.
Update STATE.md, DECISIONS.md when needed, and HANDOFF.md; preserve all deferred work.
Keep every active memory/stage file below 200 physical lines.
Set the next ready step in STATE.md, then run:
.\.venv\Scripts\python.exe tools/improvement_memory.py export
.\.venv\Scripts\python.exe tools/improvement_memory.py check
Use an existing working Python if this venv is unavailable; do not replace the environment blindly.
Return implementation status, actual checks, evidence, model/forward status and the next prompt/model/effort.
```
---

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
---

# 20 — Repair daily race identity, operating cutoff and prediction provenance

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 19

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 20 only: Repair daily race identity, operating cutoff and prediction provenance.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Reproduce B5/B6/B7 from reports/improvement/17/READINESS.md on isolated fixtures/copies. Inspect utils/normalizer.py, features/fuse.py, execution/snapshots.py, tickets.py, race_facts.py, storage migrations and the daily loop before choosing one compatible canonical identity contract.
2. Resolve a physical race across source off-time disagreements using venue/date and reliable source IDs/runner evidence. Never blindly round neighbouring races into one. Preserve WIN/PLACE association; different venues at the same instant must stay distinct. Ambiguous mappings must be quarantined with a reason rather than guessed.
3. Implement an additive, versioned, restart-safe mapping/migration for persisted snapshots/tickets/results/text references. Preserve historical identifiers and immutable payloads; no silent rewriting of prior research artifacts or destructive DB reset. Back up the live store before any authorized local migration, verify row counts and integrity, and document rollback. Prove repeated migration/capture is idempotent and legacy ambiguity fails closed.
4. Wire configured started-race buffer and real wall-clock decision cutoff into daily issuance. Reject prior-day predictions, stale/future/unreadable timestamps and already-started races, including Dublin DST/midnight cases. Keep actual prediction issuance separate from any date key used for dedupe. Do not regenerate old tickets with today's timestamp.
5. Attach model artifact content hashes, feature/schema version, configuration hash, prediction/cycle ID, source as-of times and decision time to fresh predictions and decisions; versions must resolve to a saved manifest. Missing or inconsistent provenance must exclude formal eligibility. Preserve additive compatibility with existing UI readers.
6. Test the actual daily loop entrypoint on isolated stores: same-time different venues, one-minute source disagreement, adjacent distinct races, repeat run/restart, stale-next-day replay, DST, mixed timestamps and changed model content. Assert unique tickets, correctly associated markets, no retroactive issuance and truthful provenance. Record before/after audit repro results.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/20/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/20.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
---

# 21 — Make paper staking, settlement and shadow observations reliable

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 20

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 21 only: Make paper staking, settlement and shadow observations reliable.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Reproduce B2/B3/B4 with the step 17 settlement probe on isolated stores. Trace execution/staking.py, settlement.py, tickets.py, race_facts.py, report.py, gates.py and scripts/daily_paper_loop.py. Wire the existing tested settlement engine into the real daily path, not a parallel calculator.
2. Persist nonzero deterministic stakes only for eligible paper CANDIDATE decisions, honour bankroll/exposure limits, available executable prices, selection-lock odds bands and supported WIN/EW terms. A NO-GO model must continue producing PASS with zero stake. Test GO only via isolated fixtures, never by modifying the real gate or verdict.
3. Handle winner/loser, DNF (loss, not void), explicit non-runner/abandoned race void, Rule 4, dead heats, EW legs and rounding through settle_ticket. Missing/ambiguous facts or unsupported terms stay pending with a reason. Matching includes canonical race and runner, never horse alone. Preserve source facts and settled-at times; restart/repeated result fetch cannot apply P&L twice.
4. Reconcile results and actual closing-price observations for PASS/shadow predictions in a separate observation path. PASS records remain zero-stake non-bets and never count towards qualified-bet thresholds, bankroll, ROI or drawdown. Separate probability scoring, CLV coverage, candidate-ticket P&L and forward-qualified evidence in both payload and UI. Label reference/closing/executable sources and log versus percentage CLV accurately.
5. Make report totals, stakes, returns, drawdown and open/pending counts reconcile exactly to the canonical ledger. Label any ui/bet_placer.py manual ledger as separate and exclude it from validation unless explicitly reconciled. Add local dedupe for notifications/events where this path can repeat them, but do not send any real external message during development or smoke runs.
6. Regression-test real loop issuance-to-settlement and rerun/crash recovery using isolated DBs, with stake/profit arithmetic assertions for all supported cases. Verify the real NO-GO verdict and formal-window state remain unchanged. Record unresolved facts as data gaps, not successful settlements.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/21/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/21.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
---

# 22 — Make current declarations and daily source health trustworthy

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 21

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 22 only: Make current declarations and daily source health trustworthy.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Recheck B8 using current files and bounded permitted source reads. Timeform was HTTP403 and its card file stale at 2026-06-13; do not assume a wired scraper has working data. Inspect existing authorized provider paths before adding a source. Do not bypass paywalls/access blocks or purchase access. A licensed feed may need user-supplied access; state that concrete dependency honestly.
2. Make declaration ingestion preserve provider IDs, fetched/published times, race date, runner status and current jockey/trainer/draw/weight/equipment. Explicitly distinguish declared, historical_fallback and missing per field all the way through prediction payloads and UI. Do not turn absent runner status into a claimed confirmed runner. Preserve the null[pyarrow] regression fix.
3. Add reliable stale/partial/error detection and bounded retry/backoff; a failed refresh must preserve good stored data without labelling it current. Exclude ineligible races from formal evidence, while allowing clearly labelled capture-only diagnostics. Blocked providers must not stall the whole cycle indefinitely. Capture failure reasons, coverage/field completeness and timestamps in a daily health report.
4. Wire the existing results and source-comment archive into bounded daily collection where currently permitted and available. Extraction uses stage 19's shared cap and timing contract. Never backdate acquisition of old comments; newly observed historical text is not pre-race evidence.
5. Test stale source, changed declaration, late withdrawal, provider outage, partial field, null-only columns and same-day later refresh. Verify history fallback cannot acquire a declared label or fresh timestamp accidentally.
6. Run at most one bounded source-health smoke after offline tests, with notifications disabled. If legitimate current declarations are unavailable, finish the safe failure handling and configuration/runbook, record B8 OPEN_EXTERNAL with exact feed/credential requirement, and leave affected evaluation eligibility disabled. Engineering DONE must explicitly distinguish that unresolved live-data dependency from data readiness.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/22/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/22.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
---

# 23 — Build a consistent shadow model bundle and freeze fresh evaluation

**Model:** Claude Sonnet 5
**Thinking level:** High
**Required prior steps:** 22

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 23 only: Build a consistent shadow model bundle and freeze fresh evaluation.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Read B1/B9, stage 10 frozen protocol, stage 17 candidate_manifest/selection evidence and the used-window register. Rebuild a COMPLETE compatible won/placed_2/showed candidate bundle from the corrected feature pipeline into a new isolated directory. Reuse verified immutable matrices and valid checkpoints; do not rerun the entire historical rebuild or a broad tuning sweep without a demonstrated input/version need.
2. Freeze data cutoff, whole-race train/calibration splits, features, hyperparameters, runtime budget and target definitions before fitting. Only historical/development data and already-observed evaluation windows may be used; label all retrospective comparisons as diagnostic. Stage17 consumed the final holdout. Do not rerun its once-only final script, retune on its outcomes or call any old period untouched.
3. Make training and shadow inference use the identical feature definitions/order/transforms and metadata. Validate independent-feature price invariance, complete-field normalization, non-runner handling, target nesting and training/serving parity. Keep market-assisted outputs disclosed. Ensure F-L/reference remapping uses the same reference input as fitting while executable quotes only affect EV; align UI eligibility/EV labels with the canonical gate.
4. Expose a separately tagged shadow prediction route with full bundle/config hashes; do not overwrite, auto-promote or mix targets from the served champion. Preserve its rollback hashes and make corrupt/missing shadow targets fail closed. Capture champion versus candidate provenance distinctly. Text extraction remains sidecar-only pending genuine multi-day ablation; rejected XGBoost is not reintroduced without new evidence.
5. Create an immutable new prospective SHADOW evaluation manifest before new evaluated predictions: exact start instant/run ID, candidate hashes, race eligibility/cutoff, metrics (race log loss/Brier/calibration and separate CLV), market baseline, race/day-clustered uncertainty, data-quality exclusions, review horizon and minimum evidence. Lock rules against outcome-driven changes. Model NO-GO and the formal qualified-bet forward window remain unchanged; this observational cohort is separate.
6. Verify model load/score smoke, provenance and parity offline and run appropriate model/integration tests. Report fit time and actual outputs, not estimates as completion. If a required target cannot be trained correctly, mark NEEDS_FIX/BLOCKED with concrete evidence and stop; never complete this stage with a partial or mixed bundle.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/23/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/23.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
---

# 24 — Audit daily operation and hand off fresh paper evaluation

**Model:** Claude Opus 5
**Thinking level:** Extra high (xhigh)
**Required prior steps:** 19, 20, 21, 22, 23

Select the model/effort above in a manual chat; the automatic runner sets them itself.

```text
Execute improvement step 24 only: Audit daily operation and hand off fresh paper evaluation.
Work in C:/Users/mshr/Documents/Race Predictor v4/Race Predictor v4.
Read AGENTS.md, CLAUDE.md, DESIGN.md and memory/improvement/{STATE,DECISIONS,HANDOFF}.md.
Read docs/improvement/DAILY_PAPER_FOLLOWUP.md, this prompt and the prerequisite stage notes.
Read reports/improvement/17/READINESS.md and reports/improvement/18/OPERATIONS.md.
Recheck actual code/artifacts; preserve unrelated dirty work and all previous stage evidence.
Implement ONLY this stage; the supervisor launches the next fresh CLI session.
Paper-only, model NO-GO and formal-window gates remain in force. No bets, promotion or external messages.

Scope and acceptance:
1. Independently inspect implementations and fresh evidence from 19-23. Re-run the original B1-B10 repros where applicable and publish each disposition: fixed with evidence, still open, or external data gap. Do not inherit previous green tests as proof. Reproduce critical acceptance through the actual daily entrypoints on isolated stores before any live operation.
2. Test a complete cycle, repeated same-cycle invocation, interrupted/resumed cycle, source outage, stale/already-off data, capped hosted outage, model-hash mismatch and delayed settlement. Required checks include no duplicate canonical decisions, no changed old evidence, correct stake/result/CLV reconciliation, NO-GO means zero candidate bets, and separately scored PASS observations. Run the full test suite serially at this review milestone.
3. Repair bounded integration defects within this stage with regression checks and re-audit. A large unresolved correctness defect must leave this stage NEEDS_FIX with exact file/repro/repair scope, not DONE. Do not change earlier stage ledger statuses or manufacture a green operational verdict. External provider gaps may remain declared only when excluded/fail-closed behavior itself is verified.
4. Provide a single practical daily command/launcher that collects current data, records grounded extraction, creates eligible frozen shadow predictions, reconciles available results/closing quotes and writes a clear health/evaluation summary. Include a run lock, bounded timeouts, atomic checkpoint, resumable stages and actionable failure output; do not silently repeat expensive training daily. Test it headlessly with no external notifications. No recurring scheduler/service is created by this coding stage.
5. After readiness checks, run one bounded real capture-only cycle with notifications disabled and prove its counts/hashes/timestamps from actual stored records. Use the new prospective manifest, never backfill older records into its cohort. If no race is eligible now or an indispensable live feed is missing, report that precise condition and use a labelled offline replay; do not claim fresh observations exist.
6. Write reports/improvement/24/READINESS.md and OPERATIONS.md with exact daily start/status/resume commands, what the user sees, safe recovery and remaining external actions. Distinguish engineering completion, source readiness, shadow collection started, model acceptance and prospective evidence complete. Include the shared hosted budget balance and how to disable hosted calls. Record the next review date/horizon from the frozen protocol; real calendar time and qualified sample sizes cannot be completed in a coding session.

Complete implementation and meaningful regression/integration checks, not just a plan.
Store long outputs under reports/improvement/24/ and new model/data artifacts in isolated candidate paths.
Reuse completed work on resume. Wait inside this CLI session for every owned process to finish;
never yield a final response with an unfinished background job or schedule a wakeup.
Record real commands/results, versions/hashes, limitations and acceptance in memory/improvement/stages/24.md.
Immediately before saving, reread current STATE/HANDOFF. Update only this stage's ledger disposition,
set Active step: none, and queue the first unfinished numbered stage (none after all dispositions).
Preserve stages 01-18 and deferred stage 16. Required correctness failures are NEEDS_FIX/BLOCKED.
Keep all active memory and stage notes below 200 physical lines; compact by linking archived evidence.
Run .venv/Scripts/python.exe tools/improvement_memory.py export and then check.
Return actual implementation/checks, evidence paths, unresolved limitations and next step/model/effort.
```
