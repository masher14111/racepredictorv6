# Race Predictor — Five-Stage Live Odds, Probability, and Betting Repair

These prompts are designed to be run **one at a time, in order**, with each prompt
in a fresh chat operating on the same repository:

`C:\Users\mshr\Desktop\Race Predictor v4`

A new chat cannot see earlier conversations. It can only see the repository,
git history, saved reports, `HANDOFF.md`, and files under `memory/`. For that
reason, every prompt below contains its own startup verification and mandatory
handoff/memory instructions.

Do not run Prompts 3–5 in parallel. Prompt 2 also should follow Prompt 1 because
both affect the live snapshot contract. Commit or otherwise checkpoint each
successful stage before starting the next one.

## Recommended models

| Stage | Codex model | Codex effort | Claude model | Claude effort |
| --- | --- | --- | --- | --- |
| 1 — market ingestion | GPT-5.6 Sol | high | Claude Opus 5 | xhigh |
| 2 — proxy and freshness | GPT-5.6 Terra | high | Claude Sonnet 5 | high |
| 3 — probability and EV | GPT-5.6 Sol | max | Claude Fable 5 | xhigh |
| 4 — calibration audit | GPT-5.6 Sol | ultra | Claude Fable 5 | max |
| 5 — realistic betting | GPT-5.6 Sol | max | Claude Opus 5 | ultracode |

Claude fallbacks:

- If Fable 5 is unavailable, use Claude Opus 5 at `max`.
- If Opus 5 is unavailable, use Claude Opus 4.8 at `xhigh` or `max`.
- If Sonnet 5 is unavailable for Prompt 2, use Claude Sonnet 4.6 at `high`.
- Claude Code `ultracode` is a workflow mode built on `xhigh`; if unavailable,
  use `xhigh`.
- Do not use Haiku for these repairs.

The model/effort labels below are instructions for the user. A model name inside
the pasted prompt does not switch the active model automatically.

---

## Prompt 1 — Repair contaminated odds and racecards

**Codex:** GPT-5.6 Sol · `high`  
**Claude:** Claude Opus 5 (`claude-opus-5`) · `xhigh`

For Claude Code:

```text
/model claude-opus-5
/effort xhigh
```

Paste the following into a fresh chat:

```text
STAGE: 1 of 5 — primary WIN-market ingestion repair
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

MEMORY / CONTEXT
This is the first stage of a sequential repair programme. The live snapshot has
demonstrated that sportsbook specials are entering the model as if they were
horses. Examples include “Both To Finish In The Top 3”, horse-pair selections,
“Betting Without”, and winning-distance combinations. On the inspected
2026-07-25 snapshot, LivescoreBet had 51 races but a median 30 selections in
markets labelled WIN and a median booksum around 3.68. BoyleSports also contained
special-market selections. The goal of this stage is a strict, evidence-backed
primary-WIN contract that fails closed. It is not to make the downstream model
look profitable.

FRESH-CHAT STARTUP — DO THIS BEFORE EDITING
1. Read README.md, PRODUCT.md, PROGRESS.md, memory/MEMORY.md, the relevant
   scraper/normalizer memory notes, and HANDOFF.md if it exists.
2. Inspect `git status --short`, recent commits, and the current diff. The
   repository may already contain user or earlier-chat changes. Preserve them.
3. If HANDOFF.md contains an earlier attempt at this stage, verify its claims
   against the actual code, data, and test output. Repository state is
   authoritative.
4. Run the existing targeted scraper/normalizer tests to establish a baseline.
5. Do not discard, reset, overwrite, or silently absorb unrelated changes. If
   overlapping uncommitted work makes safe implementation impossible, stop and
   identify the exact overlap.

TASK
Read and diagnose the live horse-racing market ingestion before editing.
Inspect scraper/livescorebet.py, scraper/boylesports.py,
scraper/paddy_power.py, utils/normalizer.py, features/fuse.py, related cache and
storage code, their fixtures/tests, and current data/live_odds.parquet.

Implement a strict primary-race-WIN-market contract:

1. LivescoreBet must whitelist the confirmed primary WIN market ID/type. An
   unknown marketGroupId must be skipped or quarantined, never defaulted to WIN.
2. BoyleSports must scope parsing to the primary race-winner market container
   rather than selecting every odds button on the page.
3. Give Paddy Power the same market-identity audit and contract.
4. Preserve source market_id, market_name, selection_id, race_id, fetched_at,
   and validation status through normalization/storage where available.
5. Do not rely mainly on runner-name regexes. Regex checks may be a secondary
   quarantine guard, not the primary market classifier.
6. Add race-level validation for duplicate selections, implausible field size,
   extreme booksum, multiple primary WIN markets, and specials masquerading as
   runners. Make thresholds source-aware/configurable where appropriate.
7. A contaminated or ambiguous race must fail closed: it cannot produce a
   prediction, EV, suggestion, or paper/real bet.
8. Do not delete historical source-of-truth data blindly. Remove or quarantine
   only verified contaminated live/cache rows, then rebuild the live snapshot
   and predictions from clean source data.
9. Add an audit command/report that shows race count, runner-count distribution,
   booksum distribution, rejected markets, rejection reasons, and suspicious
   selection examples by source.

TESTS AND ACCEPTANCE
1. Add adversarial fixtures containing ordinary horses plus specials and prove
   that only genuine primary WIN runners survive.
2. Add regression tests for unknown market IDs, multiple markets, duplicate
   selections, contaminated books, and fail-closed downstream behaviour.
3. Run targeted scraper, market-validation, normalizer, storage, and end-to-end
   prediction tests.
4. Run the full pytest suite with the repository’s .venv interpreter.
5. Run the live-data audit and show before/after runner counts and booksums.
6. Do not declare success merely because unit tests pass. Validate the produced
   live artifact and include concrete clean/contaminated examples.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
Before finishing:

1. Create HANDOFF.md if absent. Otherwise append a dated “Stage 1” section;
   preserve all useful earlier content.
2. Record: stage number/status, files changed, schema/config changes, data
   rebuilt/quarantined, exact commands and test results, before/after audit
   figures, unresolved issues, and whether Stage 2 is safe to start.
3. Create or update `memory/live-repair-01-market-ingestion.md` using the
   repository’s YAML-frontmatter memory format:

   ---
   name: live-repair-01-market-ingestion
   description: <one-sentence factual result>
   metadata:
     type: project
   ---

   Record only implemented and verified facts, important decisions, commands,
   measured results, limitations, and links/relations to prior memory notes.
4. Add or update its pointer in memory/MEMORY.md without removing existing
   entries.
5. Record the current git commit hash in HANDOFF.md. If changes are uncommitted,
   say so plainly and list them.
6. After all acceptance gates pass, create a scoped commit containing only this
   stage’s files with message `fix: validate primary win markets`. Never include
   unrelated pre-existing changes. If clean scoping is not possible, do not
   commit; document why and provide the exact safe next step.
7. End with an explicit verdict: STAGE 1 PASS, STAGE 1 FAIL, or STAGE 1 BLOCKED.
```

---

## Prompt 2 — Fix proxy rotation, stale caches, and snapshot reliability

**Codex:** GPT-5.6 Terra · `high`  
**Claude:** Claude Sonnet 5 (`claude-sonnet-5`) · `high`

For Claude Code:

```text
/model claude-sonnet-5
/effort high
```

Paste the following into a fresh chat:

```text
STAGE: 2 of 5 — proxy, freshness, and atomic snapshot repair
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

MEMORY / CONTEXT
Stage 1 established a strict primary-WIN market contract and should have
quarantined phantom selections. This stage makes the network and cache path
reliably deliver a fresh, internally consistent snapshot to that contract. The
configured endpoint is a rotating DataImpulse-style gateway, but earlier code
treated it as an ordinary proxy: repeated failures could blacklist the only
gateway and silently fall back to a direct connection. Stale or partially
updated bookmaker data must never be represented as fresh executable odds.

DEPENDENCY GATE
Stage 1 must be complete. Confirm that HANDOFF.md and
memory/live-repair-01-market-ingestion.md exist and that the primary-WIN
regressions pass. Confirm contaminated races fail closed. If Stage 1 failed,
is uncommitted and unsafe, or its contract is absent, do not start downstream
implementation. Report the dependency failure and return STAGE 2 BLOCKED.

FRESH-CHAT STARTUP — DO THIS BEFORE EDITING
1. Read README.md, PRODUCT.md, PROGRESS.md, memory/MEMORY.md,
   memory/live-repair-01-market-ingestion.md, and the complete HANDOFF.md.
2. Inspect `git status --short`, recent commits, the Stage 1 commit/diff, and
   all reports referenced by the handoff.
3. Verify Stage 1’s claims in the actual code/data and run its key regressions.
4. Run existing proxy, cache, source-health, refresh, and scraper tests for a
   baseline.
5. Preserve earlier fixes and unrelated user changes. Do not reset or duplicate
   Stage 1 work. Stop if unsafe overlapping edits cannot be isolated.

TASK
Audit config.yaml/config.local.yaml handling, utils/proxy_manager.py, all live
scrapers, utils/cache.py, storage code, scripts/refresh.py, ui/refresh_ops.py,
settings UI, and source-health reporting.

Implement:

1. Configure rotating-gateway semantics and explicit country targeting where
   appropriate.
2. For proxy-required domains, never silently fall back to direct after the
   gateway fails. Return a typed failure and mark the source unavailable.
3. Add destination-specific health checks; a successful generic IP probe does
   not prove the bookmaker endpoint works.
4. Distinguish 403/challenge, timeout, TLS, invalid JSON, empty card, parser
   rejection, market-validation rejection, and rate-limit failures.
5. Track per-source success rate, last attempt, last successful fetch, response
   age, row/race counts, validation counts, proxy route, and rejection reason.
6. Make proxy configuration reload safely in the running Streamlit process;
   the singleton must not retain obsolete settings indefinitely.
7. Never log/display proxy credentials. Remove credentials from committed
   config, use config.local.yaml or environment variables, redact URLs in logs,
   health reports, exceptions, tests, and UI, and instruct the user to rotate
   any exposed credential without reproducing it.
8. Propagate `stale`, `age_seconds`, `fetched_at`, and source health. Above a
   configurable TTL, block EV and suggestions rather than presenting stale odds
   as live.
9. Make each source update atomic. A partial failed scrape must not mix with a
   previous snapshot or erase a known-good snapshot without an explicit state.
10. Define and test how multi-source snapshots are assembled when one bookmaker
    is fresh, one stale, and one unavailable.

TESTS AND ACCEPTANCE
1. Test rotating failures never blacklist the gateway, proxy-required domains
   never fall back to direct, destination health, runtime reconfiguration,
   stale-cache blocking, credential redaction, atomic writes, and mixed-source
   refreshes.
2. Run Stage 1 regressions plus targeted proxy/cache/refresh tests.
3. Run the full pytest suite.
4. Perform a bounded health check/live refresh where network access permits.
   Report each source independently; do not turn skipped network validation into
   a pass.
5. Prove stale/failed sources cannot create executable EV or suggestions.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
Before finishing:

1. Append a dated “Stage 2” section to HANDOFF.md; preserve Stage 1.
2. Record files changed, config/secrets migration, runtime behaviour, health
   results, commands/tests, unresolved failures, and whether Stage 3 is safe.
3. Create/update `memory/live-repair-02-proxy-freshness.md` with the standard
   YAML frontmatter and only verified implementation facts.
4. Add/update its pointer in memory/MEMORY.md.
5. Record the current commit hash or an exact uncommitted-file list.
6. After acceptance passes, create a scoped commit containing only Stage 2
   changes with message `fix: harden proxy and odds freshness`. Never include
   unrelated changes; document why if safe commit scoping is impossible.
7. End with STAGE 2 PASS, STAGE 2 FAIL, or STAGE 2 BLOCKED.
```

---

## Prompt 3 — Make win probability, fair odds, and EV consistent

**Codex:** GPT-5.6 Sol · `max`  
**Claude:** Claude Fable 5 (`claude-fable-5`) · `xhigh`

For Claude Code:

```text
/model claude-fable-5
/effort xhigh
```

If Fable is unavailable:

```text
/model claude-opus-5
/effort max
```

Paste the following into a fresh chat:

```text
STAGE: 3 of 5 — full-field probability, fair-price, and EV reconciliation
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

MEMORY / CONTEXT
Stages 1–2 should now provide genuine primary-WIN runners in fresh, atomic,
source-labelled snapshots. This stage repairs the structural probability-to-EV
path. Earlier code normalized win probabilities before removing non-runners,
exposed only the top three plus short-price exclusions to the value layer, and
then de-vigged that partial set. It also risked treating a synthetic best-price-
per-runner board as if it were one bookmaker’s complete market. The objective is
one auditable calculation contract, not a larger number of “value” badges.

DEPENDENCY GATE
Stages 1 and 2 must be complete. Confirm the two live-repair memory files,
HANDOFF.md PASS verdicts, relevant commits/diffs, clean primary-WIN audit, source
freshness metadata, atomic snapshot behaviour, and fail-closed stale/contaminated
races. If either prerequisite is not actually satisfied, do not repair EV on
bad input. Return STAGE 3 BLOCKED with the failed dependency.

FRESH-CHAT STARTUP — DO THIS BEFORE EDITING
1. Read README.md, PRODUCT.md, PROGRESS.md, memory/MEMORY.md,
   memory/live-repair-01-market-ingestion.md,
   memory/live-repair-02-proxy-freshness.md, model/value/calibration memory notes,
   and all of HANDOFF.md.
2. Inspect git status, recent commits, current diffs, and reports referenced by
   both earlier stages.
3. Verify prior claims in code/data and run their key regressions.
4. Run current predictor, devig, value, suggestion, feature-fusion, and UI logic
   tests for a baseline.
5. Preserve earlier fixes and unrelated changes. Do not reset or duplicate work.

TASK
Audit models/predictor.py, models/value.py, models/devig.py,
models/predict_unified.py, models/suggestions.py, features/fuse.py,
features/builder.py, relevant storage schemas, and every UI consumer.

Fix the following:

1. Remove non-runners before within-race probability normalization, market
   de-vigging, field size, ranking, and suggestion generation.
2. Preserve every valid runner in a full internal/API `runners` collection.
   `selections` can remain a display-only top three, but value detection must
   consume the complete validated field.
3. Use robust race identity including source race_id or canonical venue plus
   exact race time. Do not collapse venue-day races or key odds only by
   date/venue/horse.
4. Never de-vig a partial field. Without a complete reference book, fair_prob,
   market edge, and confidence must be unavailable and the race must be PASS.
5. Do not de-vig synthetic best prices as a single bookmaker book. Derive fair
   probability from one complete, fresh reference bookmaker snapshot or a
   documented consensus method. Keep best executable odds separate for
   EV = model_probability * executable_decimal_odds - 1.
6. Require probability, reference odds, and executable odds to map to the same
   race/runner and compatible timestamps. Carry source and age into diagnostics.
7. Consolidate duplicate value rules in Predictor._build_race, models.value,
   models.suggestions, and UI filters so all surfaces agree.
8. Define separate typed fields for model_win_prob, empirical_win_rate,
   raw_implied_prob, devigged_market_prob, fair_decimal_odds, reference_odds,
   executable_odds, edge_pp, relative_edge, EV, and calculation timestamp.
9. Add invariants: valid-runner model probabilities sum to 1 within tolerance;
   complete reference books de-vig to 1; odds are finite and >1; no phantom/NR
   runners; no EV from stale, contaminated, mismatched, or incomplete cards.
10. Produce a reconciliation script/report showing every calculation input and
    output for representative races, including explicit PASS reasons.

TESTS AND ACCEPTANCE
1. Add regressions reproducing the top-three partial-field de-vig bug,
   normalize-before-NR bug, race-key collision, incomplete reference book,
   synthetic-best-price book, stale timestamp mismatch, and UI/backend rule
   disagreement.
2. Run all Stage 1–2 regressions plus targeted predictor/value/devig/suggestion
   and UI tests.
3. Run the full pytest suite.
4. Rebuild predictions only from validated fresh data and run the reconciliation
   report.
5. Demonstrate that complete displayed fields sum correctly and that every EV
   can be recomputed exactly from persisted inputs.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
Before finishing:

1. Append a dated Stage 3 section to HANDOFF.md without deleting Stages 1–2.
2. Record the canonical formulas, schemas, race identity, reference/executable
   price policy, files changed, commands/tests, reconciliation results, known
   limitations, and whether Stage 4 is safe.
3. Create/update `memory/live-repair-03-probability-ev.md` with standard YAML
   frontmatter and factual verified results.
4. Add/update its pointer in memory/MEMORY.md.
5. Record commit hash or exact uncommitted changes.
6. After acceptance passes, create a scoped commit with message
   `fix: reconcile full-field probability and ev`. Exclude all unrelated
   pre-existing changes; document why if safe scoping is impossible.
7. End with STAGE 3 PASS, STAGE 3 FAIL, or STAGE 3 BLOCKED.
```

---

## Prompt 4 — Recalibrate and verify the claimed win rate honestly

**Codex:** GPT-5.6 Sol · `ultra`  
**Claude:** Claude Fable 5 (`claude-fable-5`) · `max`

For Claude Code:

```text
/model claude-fable-5
/effort max
```

If Fable is unavailable:

```text
/model claude-opus-5
/effort max
```

Paste the following into a fresh chat:

```text
STAGE: 4 of 5 — leak-free calibration and model-validity audit
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

MEMORY / CONTEXT
Stages 1–3 should have produced clean primary-WIN data, fresh atomic odds
snapshots, complete race fields, correct non-runner handling, a defensible
reference market, and one reproducible probability-to-EV contract. This stage
does not assume that the model is profitable. It investigates exact-zero value
probabilities, disagreement among priced/normalized/price-free/LightGBM/market
lines, calibration drift, and potential circularity in the odds-conditioned
favourite-longshot recalibrator. The required output is an honest GO/NO-GO.

DEPENDENCY GATE
Stages 1–3 must be PASS in HANDOFF.md and supported by their memory notes,
commits/diffs, tests, clean-data audit, and probability reconciliation. Re-run
key invariants. Do not train or evaluate on contaminated, stale, partial-field,
or unreprocessed artifacts. If prerequisites fail, return STAGE 4 BLOCKED.

FRESH-CHAT STARTUP — DO THIS BEFORE EDITING
1. Read README.md, PRODUCT.md, PROGRESS.md, DATA_HEALTH.md,
   memory/MEMORY.md, all three live-repair memory notes, calibration/model
   memory notes, and the complete HANDOFF.md.
2. Inspect git status/history/diffs, model metadata, artifact dates/hashes,
   training cutoffs, data date ranges, prior holdout reports, and referenced
   audit/reconciliation outputs.
3. Verify earlier claims in actual code/data and run their core tests.
4. Establish a current calibration/holdout baseline before changing training,
   calibration, thresholds, or artifacts.
5. Preserve earlier fixes and unrelated changes. Never overwrite model artifacts
   without retaining reproducibility metadata and a recoverable prior version.

TASK
Perform a leak-free calibration and model-validity audit across training,
feature generation, calibration, prediction, holdout, drift, value detection,
and favourite-longshot recalibration.

Requirements:

1. Train/evaluate only on rows that pass the new market/race validation contract.
   Reprocess historical/live-derived artifacts where required and document scope.
2. Prove the independent price-free model contains no odds, SP, finishing odds,
   market rank derived from finishing prices, outcome, or future information.
3. Treat the odds-dependent favourite-longshot output as market-adjusted, not
   price-free. Persist independent and market-adjusted probabilities separately.
   Use cross-fitting and an ablation to test same-price circularity.
4. Investigate exact-zero/one probabilities, clipping, extrapolation, missing
   support, calibrator portability, and out-of-distribution behaviour.
5. Compare coherent race-level calibration approaches such as grouped
   softmax/temperature scaling against marginal isotonic followed by
   normalization. Do not select by the final test set.
6. Use expanding-window walk-forward evaluation with all preprocessing,
   imputation, feature selection, model fitting, and calibration inside each
   training fold.
7. Reserve a final untouched recent test period. Tune nothing on it.
8. Report model and de-vigged-market log loss, Brier, ECE/reliability, A/E,
   discrimination, CLV where valid, and calibration by odds band, field size,
   race type, venue, data completeness, and month. Include uncertainty intervals.
9. Define “win rate” correctly in API/UI: predicted probability is not realized
   win rate. Realized win rate must show settled wins/bets, denominator, date
   window, selection rule, and confidence interval.
10. Compare independent model, market-adjusted model, LightGBM, priced CatBoost,
    and market baselines on identical eligible races.
11. If the model fails the market-relative or drift gate, issue NO-GO and disable
    real-money recommendations. Do not tune filters until ROI looks positive.

TESTS AND ACCEPTANCE
1. Add leakage, fold-boundary, cross-fitting, probability-extreme, calibration,
   race-normalization, artifact-compatibility, and UI-label regressions.
2. Run all earlier-stage integrity tests and the full pytest suite.
3. Generate a dated calibration report containing commands, config, data hashes,
   model/calibrator hashes, train cutoff, validation/test windows, sample sizes,
   excluded-row reasons, metrics with intervals, and GO/NO-GO.
4. Reproduce the report from documented commands.
5. Do not mark Stage 4 PASS merely because code/tests run. PASS means the audit is
   complete and reproducible; the model verdict may legitimately be NO-GO.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
Before finishing:

1. Append a dated Stage 4 section to HANDOFF.md, preserving all earlier stages.
2. Record audit status separately from model GO/NO-GO, artifacts produced,
   frozen windows, exact metrics, exclusions, tests, and whether Stage 5 may
   proceed in paper-only or candidate mode.
3. Create/update `memory/live-repair-04-calibration-audit.md` with standard YAML
   frontmatter, verified findings, artifact hashes, and the explicit model verdict.
4. Add/update its pointer in memory/MEMORY.md.
5. Record commit hash or exact uncommitted changes.
6. After the audit acceptance gate passes, create a scoped commit with message
   `test: recalibrate and validate model`. Do not include unrelated changes or
   claim a profitable model when the verdict is NO-GO.
7. End with both:
   - STAGE 4 PASS/FAIL/BLOCKED (audit completion)
   - MODEL GO/NO-GO (betting validity)
```

---

## Prompt 5 — Build realistic, conservative bet recommendations

**Codex:** GPT-5.6 Sol · `max`  
**Claude:** Claude Opus 5 (`claude-opus-5`) · `ultracode`

For Claude Code:

```text
/model claude-opus-5
/effort ultracode
```

If `ultracode` is unavailable:

```text
/model claude-opus-5
/effort xhigh
```

Paste the following into a fresh chat:

```text
STAGE: 5 of 5 — realistic execution, forward validation, and betting safeguards
PROJECT: C:\Users\mshr\Desktop\Race Predictor v4

MEMORY / CONTEXT
Stages 1–4 should have repaired live market identity, proxy/freshness integrity,
full-field probability/EV calculations, and completed a reproducible calibration
audit. Earlier project evidence reported negative CLV, so this stage defaults to
paper-only. It must obey Stage 4’s current model GO/NO-GO rather than searching
for thresholds that make historical profit appear. “No bet” is a valid and
expected output.

DEPENDENCY GATE
Stages 1–4 must be complete and verified from HANDOFF.md, their four memory
notes, commits/diffs, tests, audit reports, and artifacts. Read Stage 4’s dated
calibration report and exact MODEL GO/NO-GO. If the model is NO-GO, implement
paper betting, forward tracking, safeguards, and PASS output only. Do not enable
or present real-money recommendations. If Stage 4 is incomplete, return
STAGE 5 BLOCKED.

FRESH-CHAT STARTUP — DO THIS BEFORE EDITING
1. Read README.md, PRODUCT.md, PROGRESS.md, memory/MEMORY.md, all four
   live-repair notes, paper-betting/backtest/suggestion memory notes, HANDOFF.md,
   and the Stage 4 calibration report.
2. Inspect git status/history/diffs and verify every prior stage’s key contract
   against actual code, data, tests, and artifacts.
3. Run the earlier integrity/calibration regressions and current betting,
   settlement, staking, suggestion, dashboard, and end-to-end tests.
4. Preserve prior fixes and unrelated user work. Do not weaken a fail-closed or
   NO-GO gate to produce recommendations.

TASK
Design realistic execution and bankroll protection:

1. Add append-only point-in-time odds snapshots keyed by race, runner,
   bookmaker, primary market, and fetched_at. Never backtest using a later or
   closing price as the price supposedly available earlier.
2. Build a walk-forward execution simulator covering odds age/movement,
   request/bet latency, rejected/suspended markets, non-runners, Rule 4
   deductions, bookmaker limits, exchange commission where applicable,
   each-way terms captured at bet time, dead heats, voids, and no assumed BOG
   unless recorded.
3. Compare model-only, de-vigged market, and simple favourite baselines on the
   same eligible races. Report CLV, ROI/yield, A/E, hit rate, drawdown, turnover,
   losing streaks, sample size, and uncertainty intervals.
4. Prevent threshold mining: tune on training folds, select once, and evaluate
   once on an untouched test window. Report the number of strategies tried and
   multiple-testing/selection-bias risk.
5. Default to PASS unless the complete fresh card, supported runner history,
   calibrated independent probability, reference market, executable price,
   minimum edge/EV, source health, and model-validation gate all pass.
6. While provisional, use conservative configurable staking: default 0.10
   fractional Kelly, capped at 0.5% bankroll per bet, 0.5% per race, and 3%
   daily exposure. No accumulators or correlated bets. These are risk ceilings,
   not profitability claims, and must not be raised without forward evidence.
7. Define a forward-release gate requiring at least eight weeks and a meaningful
   qualified sample, positive mean CLV with a 95% interval above zero, stable
   A/E/calibration, and acceptable drawdown. Make criteria config-driven but do
   not silently loosen them.
8. Every candidate/paper ticket must show horse, race, bookmaker, offered odds,
   timestamp/age, independent model probability, market-adjusted probability if
   used, fair odds, market probability, edge, EV, maximum stake, data quality,
   validation state, and reasons it passed. Show explicit PASS reasons otherwise.
9. Add bankroll stop-loss, daily loss/exposure limits, duplicate prevention,
   started-race blocking, source-stale blocking, and a permanent paper-only
   override.
10. Build a forward-validation dashboard/report that separates historical
    backtest, shadow/paper bets, and any future real execution.

TESTS AND ACCEPTANCE
1. Test point-in-time integrity, no future-price use, latency, rejections,
   non-runners/Rule 4, commission, each-way/dead-heat/void settlement, exposure
   caps, duplicate/started-race blocking, NO-GO propagation, and honest PASS.
2. Run every earlier-stage regression plus the complete pytest suite.
3. Run a deterministic end-to-end paper/shadow workflow and produce a dated
   forward-validation report and “today’s candidates” report.
4. If MODEL NO-GO or forward gate fails, the candidates report must contain no
   real-money recommendations and must explain the failed criteria.
5. Do not claim a strategy is safe or profitable from a short backtest.

MANDATORY HANDOFF, MEMORY, AND CHECKPOINT
Before finishing:

1. Append a dated Stage 5 section and final programme summary to HANDOFF.md.
2. Record files/schemas changed, execution assumptions, staking/exposure limits,
   forward-gate state, commands/tests, reports, unresolved risks, and whether the
   system is PAPER-ONLY or eligible for candidate display.
3. Create/update `memory/live-repair-05-betting-validation.md` with standard YAML
   frontmatter, verified facts, and the final programme/model/deployment verdicts.
4. Add/update its pointer in memory/MEMORY.md.
5. Record commit hash or exact uncommitted changes.
6. After acceptance passes, create a scoped commit with message
   `feat: add realistic paper-betting safeguards`. Never mix unrelated files.
7. End with:
   - STAGE 5 PASS/FAIL/BLOCKED
   - MODEL GO/NO-GO
   - DEPLOYMENT PAPER-ONLY/CANDIDATE-ELIGIBLE
```

---

## Expected handoff chain

After the full sequence, the repository should contain:

```text
HANDOFF.md
memory/live-repair-01-market-ingestion.md
memory/live-repair-02-proxy-freshness.md
memory/live-repair-03-probability-ev.md
memory/live-repair-04-calibration-audit.md
memory/live-repair-05-betting-validation.md
```

Each new chat must verify these artifacts rather than trusting them blindly.
If a handoff conflicts with the code, test output, or generated data, the
repository evidence wins and the discrepancy must be recorded.

