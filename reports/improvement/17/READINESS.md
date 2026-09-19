# Step 17 — paper-operation readiness decision

Reviewed 2026-09-19 by Claude Fable 5.1 (independent; every figure below was re-derived
in this session, none copied from a stage note). Frozen manifest: `candidate_manifest.json`.

## Decision — three separate questions, three separate answers

| Question | Answer |
|---|---|
| **A. Technically ready to COLLECT paper data (capture-only / shadow, verdict NO-GO)?** | **READY, with conditions C1-C3.** Fail-closed behaviour, isolation, idempotency and the paper-only guards all verified (30/30 replay checks). |
| **B. Technically ready to start a FORMAL evidence window, or to operate if a model ever reached GO?** | **NOT READY.** Blockers B1-B3 make any settled CANDIDATE evidence meaningless; B4-B8 must close before evidence could count. |
| **C. Evidence of predictive improvement or profitability?** | **NONE.** The frozen candidate FAILS against the market on the untouched holdout. Model verdict stays **NO-GO / PAPER-ONLY**. |

Real-money execution: **disabled**, and there is no bookmaker order-placement code in the repository.

## C in numbers — the single use of the reserved final holdout

708 races, 2026-08-28..2026-09-17, reserved by step 10 (D42), verified never scored by any stage
(`01_dispositions.json`), opened exactly once after the manifest was frozen (`05_final_holdout_once.json`;
the script refuses a second run). The code path was first rehearsed on the development panel and
reproduced it to max|delta| = 0.

| line | race log loss | market | gap | verdict |
|---|---|---|---|---|
| **frozen candidate, price-free (s10idp) — PRIMARY** | 2.01029 | 1.76165 | -0.24864, CI95 of deficit **[+0.204, +0.291]** | does NOT beat market |
| frozen candidate, market-assisted (s10mkt) | 1.76795 | 1.76165 | -0.00630, CI95 [-0.0017, +0.0138] | does NOT beat market |
| served champion, price-free (v3nf) | 2.18711 | 1.76165 | -0.42546 | far worse |
| served champion, market-assisted CatBoost (v3) | 1.92570 | 1.76165 | -0.16405 | far worse |
| served champion, LightGBM (v3) | 2.18954 | 1.76165 | -0.42789 | far worse |
| served champion, F-L market-adjusted value line | 1.77419 | 1.76165 | -0.01254 | does NOT beat market |

**No untouched holdout remains.** Any further evidence must be prospective. July 2026 stays already-observed.

## Why the frozen candidate is the step-10 rebuild, not the served champion (B1)

Rule frozen BEFORE any champion score existed (`02_selection_protocol.json`): keep the champion unless the
same-family rebuild beats its price-free line with a paired race-clustered CI95 excluding zero.

- Development panel (618 races): rebuild 1.93620 vs champion 2.10003, delta **-0.16383, CI95 [-0.222, -0.107]**.
- Replicated on the untouched holdout: delta **-0.17682, CI95 [-0.229, -0.125]**.
- Cause measured, not guessed (`03_feature_skew_psi.csv`): in the champion's own training reference
  `trainer_form_zscore` / `jockey_form_zscore` / `hot_connection_z` have **std 0.000** (D41's dead windows); the
  repaired live pipeline now feeds them std 0.91-0.94. `trainer/jockey_hot_strike_rate` moved with **PSI 0.43-0.44**
  (the model gate's own retrain threshold is 0.2) and carry 16.4% of the champion LightGBM's importance.
  The served bundle is reading features whose meaning changed under it. Its "price-free" v3nf line also still
  includes the market-derived `race_complexity` (D40); the rebuild's 39 features do not.

Nothing was promoted. `candidate_manifest.json` holds a reversible, config-level procedure and the served
bundle's sha256s as the rollback target. The rebuild has only a `won` target, so the headline/place bundle
cannot be swapped without a retrain — that is a new work order, not a step-18 integration fix.

## Fixed in this stage (both strictly fail-closed, both regression-tested)

1. **NO-GO bypass by one YAML line.** `execution.gates.require_model_validation: false` issued a CANDIDATE under the
   real NO-GO verdict (`06_paper_replay_before_fix.json`, check F14), contradicting `execution/gates.py`'s "deliberately
   no override". `execution/config.py` now clamps it on at load and reports the clamp, exactly like
   `forward_gate.require_model_go`. Test: `tests/execution/test_gates.py::test_a_config_file_cannot_switch_model_validation_off`.
2. **EV gate failed OPEN on mixed-precision timestamps.** `models/predictor.py::_race_ev_gate` parsed `fetched_at`
   without `format="mixed"`; under pandas 3 a sub-second row became NaT and `ages.max()` skipped it, so a 2-hour-stale
   row read as 31 s fresh. Now parses mixed, and a present-but-unreadable timestamp is a PASS reason
   (`unreadable_price_timestamp:N`). Tests: two new cases in `tests/models/test_predictor.py`.

## Open blockers (reproduce each with the command shown; run from the repo root with `.venv/Scripts/python.exe`)

| # | Sev | Blocks | Defect | Reproduce | Owner |
|---|---|---|---|---|---|
| B1 | HIGH | B | Served bundle has train/serve feature skew (above) | `reports/improvement/17/03_champion_replay.py` | human promotion decision + retrain order |
| B2 | HIGH | B | Live loop never sizes a stake: a WINNER at 3.75 settles `profit 0.0`, identical to a loser; forward gate's drawdown criterion is unreachable | `reports/improvement/17/08_settlement_probe.py` | new repair order (step 08's area) |
| B3 | HIGH | B | Live settlement never calls `execution.settlement.settle_ticket`: Rule 4, dead heats, non-runner void and each-way are OFFLINE-ONLY; a non-finisher settles `void` not `lost` | same probe (`DNF` row) | same order (deferred by D37) |
| B4 | MED | B | PASS tickets are never settled, so the ledger accrues no CLV under NO-GO (988 PASS, 0 settled) | `09_live_store_readonly.json` | same order |
| B5 | MED | B | One physical race emitted twice when sources disagree on the off-time by a minute (Dundalk 19:30/19:31): duplicate ticket sets, split price books. Plus known F4 (`snapshots.race_uid` drops venue) | `06_paper_replay.py` GAP-C; `07_serving_claims_verified.json` #1 | normalizer/fuse identity order |
| B6 | MED | B | No operating cutoff live: already-off races are ticketed; a stale card is re-ticketed in full next day (308 rows). `block_started_races` is configured but never called | `06_paper_replay.py` GAP-A/GAP-B | live-loop order |
| B7 | MED | B | No model version/hash on `predictions.json` or any ticket; window hashes are recorded only once a window starts (never) | `06_paper_replay.py` R1 provenance | step 18 can record hashes per cycle |
| B8 | MED | A-cond | Declared racecards are not flowing: `timeform.parquet` frozen at 2026-06-13 (28 rows); `runner_status` null on 651,154/651,154 unified rows. Every live jockey/trainer is a historical fallback, not labelled as such in payload/UI; non-runner detection rests on a jockey-name string | `07_serving_claims_verified.json` #5 | first live refresh (step 05's fix is unexercised) |
| B9 | LOW | - | F-L remap fed executable price but fitted on reference price (D44, unchanged). UI flags 5 "value bets" whose EV on the gate's price-free probability is -0.16..-0.38; 16/26 `ev_eligible` races exceed the gate's 1.25 book-sum ceiling | `07_serving_claims_verified.json` #6-7 | serving order |
| B10 | LOW | - | Selection lock's odds band (<=4.0) not enforced live; drift/retrain trigger has no caller; `ui/bet_placer.py` is an unlabelled parallel paper ledger (local SQLite only); bet-settled/stop-loss alerts undeduped | code trace in `stages/17.md` | backlog |

Dormancy note: B2-B4 cannot mis-settle anything today because no CANDIDATE has ever existed (988/988 PASS).
They become live the instant a verdict turns GO, which is why they block B and not A.

## Conditions on A (for step 18's first cycle)

- **C1** Record the served-bundle sha256s (compare to `candidate_manifest.json::serving_status`) at the start of every cycle.
- **C2** After the first live refresh, check whether declared cards arrived (`timeform.parquet` max date = today,
  any non-null `runner_status`). If not, record B8 as still open — do not describe connections as declared.
- **C3** Do not call `execution.window.start_window`. Verdict NO-GO means capture-only; PASS rows are not bets.

## Fail-closed behaviour, verified (`06_paper_replay_after_fix.json`, 30/30)

Stale quote (>900 s), future-stamped quote, non-executable/consensus price source, incomplete reference book,
stale / erroring / absent source health, market-adjusted-only probability, missing or corrupt verdict file,
missing predictions file -> **PASS every time, even under a stub GO verdict.** A non-paper ticket is refused by the
store (`RealMoneyTicketRefused`) and by a schema `CHECK (paper_only = 1)`. A control run proves the gate is not
vacuous (the same clean race yields a CANDIDATE under a stub GO). Every real store was sha256-identical before and after.

Unsupported terms: anything marked UNSUPPORTED in the manifest must not yield settled CANDIDATE evidence.

## Monitoring and rollback

See `RUNBOOK.md`. Rollback of a future promotion is one config line (`value.model_tag: v3nf`); nothing is deleted.
