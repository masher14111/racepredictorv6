---
name: live-repair-04-calibration-audit
description: Stage 4 PASS (audit complete + reproducible) / MODEL NO-GO — on an untouched 2026-06-13..07-25 window the price-free model loses to the de-vigged market by 0.186 log-loss (CI excludes 0), no line significantly beats the shin de-vig baseline, CLV −13.4%, drift gate fails; value_win_prob is ~93% price echo, now persisted separately from value_win_prob_independent; FL recalibrator refit with sigmoid bands (no more exact-0.0 claims)
metadata:
  type: project
---

# Stage 4 — leak-free calibration and model-validity audit (2026-07-27)

Stage 4 of the programme in [[live-repair-00-prompt-sequence]]. Depends on
[[live-repair-01-market-ingestion]], [[live-repair-02-proxy-freshness]],
[[live-repair-03-probability-ev]] (all PASS; commits `9ce4159`, `d7f7872`).
**Stage 4 commit: `e3c705f`** (`test: recalibrate and validate model`,
36 files).
Full detail: `HANDOFF.md` "2026-07-27 — Stage 4" and
`reports/calibration_audit_20260727.md`. Audit artifacts:
`data/audit/stage4/` (panel, leakage, wf_scored, selection, final_evaluation).

## The two verdicts (do not conflate them)

- **AUDIT: COMPLETE / STAGE 4 PASS.** Every phase reproduces from documented
  commands (`python -m scripts.calibration_audit --run stage4 --all`;
  reproduce-check matched frozen-line numbers exactly, GPU-refit lines to 1e-6).
- **MODEL: NO-GO — paper-only.** May legitimately never flip: the independent
  price-free line loses to the de-vigged pre-off market by **0.186 race
  log-loss** (95% CI [−0.225, −0.146], 0/1000 bootstrap resamples favorable) on
  the untouched window; CLV **−13.4%** [−13.9, −13.0]; every EV>0 portfolio
  loses at the pre-off price; the PSI drift gate formally FAILS
  (`going_speed` 4.92 — seasonal + fill-rate artifact on a dead feature, but
  the gate is the gate).

## The finding that reframes everything: the June "GO" stamps were noise

The 2026-06 holdouts (LGBM +0.0115, v3 +0.0225 log-loss vs market, 313 races)
were point estimates without intervals. On 904 fresh races with race-bootstrap
CIs, **every** market-feature line's edge is insignificant vs the proportional
de-vig (LGBM +0.0105 [−0.0046, +0.0268]; v3 +0.0112 [−0.0036, +0.0275]) — and
the **shin** de-vig baseline (1.68568 vs proportional 1.68778) absorbs a third
of even those point deltas. Never quote `verdict.go` without the CI again.

## Circularity of the F-L recalibrator — measured, not argued

`OddsBandCalibrator.predict(prob, SAME price EV uses)` makes `value_win_prob`
**market-adjusted, not price-free** (closes handoff issue #8):

- corr(prob, 1/price): 0.578 independent → **0.925** adjusted;
- **77%** of a +5% price improvement's EV gain is absorbed by the remap;
- within-price-bin AUC drops 0.549 → 0.537 (the "preserves within-price
  discrimination" design claim from [[calib-fl-02-fl-design]] is falsified);
- 2,484 band-[2,4] OOS bets are EV-positive ONLY via the remap: realized A/E
  0.92, flat yield −1.9% — manufactured, losing edge.

**Persisted split (code, tested):** `value_win_prob_independent` (price-free)
now travels beside `value_win_prob` (market-adjusted) through predictor,
`models.value` picks (`model_win_prob_independent`), RunnerPrediction and the
cache. Old caches rehydrate with None (artifact-compat test).

## Probability extremes — root cause and shipped fix

Exact zeros came from the FL isotonic bands (every band's `y[0] == 0.0`,
odds-on bands reaching `y == 1.0`, 8–24 thresholds per band; 3.6% of OOS
inputs land outside support and clamp). Live cache had 20/356 exact-zero
`value_win_prob`. On validation folds sigmoid bands score BETTER race log-loss
(1.7052 vs 1.7432 — zero-claims are catastrically punished when the horse
wins) with zero extreme emissions → **refit shipped** via
`python -m scripts.refit_fl_recalibrator` (fit on the frozen model's own OOS
span 2025-12-03..2026-06-12; old artifact backed up at
`data/backups/fl_oddsband_v3nf_calib_isotonic_20260617.pkl`, sha256 recorded
in `models/fl_oddsband_v3nf_calib_meta.json`).

## Leakage proofs (requirement 2)

Provenance PASS, empirical guards PASS (closing-move partial-corr, outcome
corr over the whole panel), weight channel verified pre-off
(implied_prob == 1/morningwap on 227k rows — the old odds_finish claim in
CLAUDE.md/backtest docstrings was STALE and is now corrected). One genuine
violation: **`race_complexity` is z-scored over the entire dataset** (appending
future data shifted all 247,617 historical values; drift tiny — mean |Δ|
0.0024, corr 1.0 — but it breaks point-in-time construction and one component
is race-level market entropy). Excluded from every audit-fitted model; flagged
for the next retrain.

## Data facts to remember

- `scripts/fetch_results_window.py` (new) catches the betsp store up over an
  explicit window using the canonical append-dedupe writer — the Betfair SP
  backbone alone carries `win_lose` (outcome) + morningwap/ppwap/BSP, so
  results exist even where Sporting Life enrichment (position fill 81.2%)
  misses.
- The audit panel excludes PLACE-market rows: **`models/train.py` trains on
  WIN+PLACE rows together** (each runner duplicated under two sample weights) —
  a data-hygiene defect to fix at the next retrain, not silently.
- Frozen v3nf's real train cutoff is **2025-12-02** (80% chronological split),
  not recorded in its meta — the holdout leakage guard cannot fire for it.

## Stage 5 guidance

Proceed **paper-only**. Consume the machine verdict from
`data/audit/stage4/final_evaluation.json` + the report; do not re-tune value
filters until the market-relative gates pass (requirement 11). The
`value.max_odds 4.0` band and EV≥0.05 gates stay as-is.
