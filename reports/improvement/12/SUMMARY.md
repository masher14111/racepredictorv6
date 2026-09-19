# Step 12 — market combination and probability calibration

Protocol frozen in `blend_manifest.json` **before** any exponent or calibrator
was fitted. Three chronological, whole-race-disjoint periods inside step 10's
frozen development side; the final holdout (2026-08-28..2026-09-17) was never
loaded by any script here.

| period | window | races | used for |
|---|---|---|---|
| `blend_dev` | 2026-02-01..2026-06-12 | 4,138 | select the blend exponents, nothing else |
| `calib` | 2026-06-13..2026-08-07 | 1,874 | fit the calibrator candidates |
| `eval` | 2026-08-08..2026-08-27 | 618 (5,561 runners) | report only |

`blend_dev`/`calib` rows are genuine walk-forward out-of-sample predictions: 8
expanding-window blocks, each scored by a conditional logit fitted only on whole
races strictly before the block start, at step 10's already-tuned L2 (nothing
re-tuned, so the panel adds no new selection). Panel:
`data/audit/12/walkforward_oos.parquet` (51,902 rows / 6,012 races).

## 1. Blend selection (blend_dev only)

`p ∝ p_independent**alpha * p_reference**beta`, renormalised within race;
63-point frozen grid; race-level log loss.

| line | selected | blend_dev LL | indep-only (1,0) | market-only (0,1) |
|---|---|---|---|---|
| `combined_indep` (price-free x market) | **alpha 0.0, beta 1.0** | 1.68223 | 1.91358 | 1.68223 |
| `combined_market` (market-assisted x market) | alpha 0.5, beta 0.5 | 1.68165 | 1.68243 | 1.68223 |

**The unconstrained search on 4,138 development races puts zero weight on the
price-free model.** The best point with any model weight at all
(`combined_indep_forced`, alpha 0.1 / beta 1.0) scores 1.68346 — *worse* than
the market alone. `combined_indep_forced` is carried forward as a labelled
diagnostic arm, not as the procedure's choice.

## 2. Calibration (calib only, promotion rule committed before eval was scored)

Fit on the first 70% of calib races, scored on the last 30%; lowest race log
loss wins, `none` allowed to win; winner refit on all of calib.

| line | none | sigmoid | isotonic | odds_band | promoted |
|---|---|---|---|---|---|
| market_only | 1.71248 | 1.71170 | 1.73234 | 1.75537 | sigmoid |
| indep_only | 1.99948 | 1.99956 | 2.00328 | **1.76254** | odds_band |
| combined_indep (= market-only) | 1.71248 | 1.71170 | 1.73234 | 1.75537 | sigmoid |
| combined_indep_forced | 1.71650 | 1.71685 | 1.74017 | 1.76656 | none |
| combined_market | 1.71077 | 1.71062 | 1.72765 | 1.74727 | sigmoid |

The only large calibration effect anywhere is `odds_band` on the *uncalibrated
price-free* line (1.999 -> 1.763): that is the favourite-longshot correction
folding the price in. It is **market-adjusted, not an ability estimate**.

## 3. Evaluation (618 reserved races, identical complete-book race set)

Market reference: LL **1.70403**, Brier 0.08472, ECE 0.00830. Base rate 0.1113.

| candidate | LL | gap vs market | Brier | ECE | race-clustered CI95 of gap |
|---|---|---|---|---|---|
| market_only__none | 1.70403 | +0.00000 | 0.08472 | 0.00830 | — |
| indep_only__none | 1.92141 | -0.21738 | 0.09092 | 0.01209 | [-0.2633, -0.1676] |
| indep_only__odds_band | 1.73541 | -0.03138 | 0.08502 | 0.01136 | [-0.0927, +0.0041] |
| combined_indep__* (promoted) | 1.70340 | +0.00063 | 0.08472 | 0.00764 | — |
| **combined_indep_forced__none** | 1.70255 | +0.00148 | 0.08469 | 0.00901 | **[-0.0021, +0.0051]** |
| combined_market__none | 1.70513 | -0.00110 | 0.08477 | 0.00759 | [-0.0040, +0.0015] |
| combined_market__sigmoid | 1.70492 | -0.00089 | 0.08477 | 0.00750 | [-0.0043, +0.0021] |
| catboost_combined_forced__none (transfer) | 1.70291 | +0.00112 | 0.08469 | 0.00997 | — |
| lgbm_combined_forced__none (transfer) | 1.70259 | +0.00144 | 0.08467 | 0.00973 | — |

**Verdict: no candidate demonstrably beats the market.** The best combined arm's
nominal +0.0015 log loss has a race-clustered bootstrap CI95 that straddles
zero (P(beat) = 0.79), and it is a *diagnostic* arm — the procedure's own
selection was "use the market alone". The full candidate table (24 arms) is in
`scorecard_12.json`. Model verdict stays **NO-GO / paper-only**.

Step 10's independent baselines reproduce exactly here (condlogit 1.92141,
catboost 1.93620, lgbm 1.91176), including LightGBM re-scored from its saved
booster over the full 5,561-row panel — which closes step 10's recorded
3,347/5,561 ledger-coverage gap.

## 4. Where calibration actually matters: the selected-bet subset

`EV = p * bet_price - 1 >= 0.05` (config `value.min_expected_value`):

| candidate | selected runners | wins | actual | mean prob | A/E |
|---|---|---|---|---|---|
| market_only__none | 353 | 48 | 0.1360 | 0.1360 | 1.000 |
| indep_only__none | **3,178** | 176 | 0.0554 | 0.1025 | **0.540** |
| indep_only__odds_band | 1,546 | 116 | 0.0750 | 0.0906 | 0.828 |
| combined_indep_forced__none | 369 | 62 | 0.1680 | 0.1706 | 0.985 |
| combined_market__none | 353 | 53 | 0.1501 | 0.1408 | 1.066 |

The uncalibrated price-free line flags **57% of the whole field** as a value bet
and wins at 54% of its own predicted rate — the longshot trap, quantified. Every
market-combined line selects ~6% of the field at A/E ≈ 1. This is the stage's
practically useful result even though no arm beats the market on log loss.

Per-odds-band, field-size, regime and region tables:
`breakdowns/calibration_by_stratum.csv`. Note A/E is ~1.000 by construction for
any *race-complete* stratum (within-race normalisation forces the field to sum
to 1), so field-size/regime/region rows should be read on `race_log_loss` and
`brier`, not A/E; odds-band and selected-bet rows cut across a race and are the
informative A/E slices.

## 5. Executable-quote / reference-book / non-runner invariance

`price_invariance.txt` — all checks PASS on real data (710 raw races rebuilt
twice through `features.builder._derive_all`):

* perturbing **only** the executable quote (`odds_decimal`, `sp`; both 100% null
  historically, so the probe fills them with a realistic best-of-N overlay and
  then moves it 1.5-4x) leaves the price-free probability, the reference
  probability and the blend **bit-identical** (max|delta| = 0);
* the reference-price control moves the reference line (max|delta| 0.572) while
  the price-free probability stays bit-identical — the probe is not vacuous;
* 13 market-assisted features *do* react (`implied_prob`, `market_rank`,
  `race_complexity`, `price_steam_pct`, …). That is by design for the
  market-assisted branch and is exactly why `combined_market` must never be
  described as an ability estimate;
* one unpriced runner NaNs the whole race's reference probability **and** its
  blend (556/556 rows), leaves other races untouched, and leaves the
  independent ability estimate defined;
* withdrawing a runner renormalises the survivors to 1.0 (max|sum-1| 2.2e-16
  over 200 races) with zero reordering, and imputes nothing for the withdrawn
  horse.

## 6. Audited finding on the SHIPPED price adjustment (not fixed here)

`models.predictor._score_value` feeds `_apply_fl_recalibration` the
**executable** best-of-N board price (`_effective_decimal`), while the shipped
`models/fl_oddsband_v3nf_calib.pkl` was fitted on the pre-off **reference**
price (`docs/calibration/fit_fl_recalibrator.py`, `bet_price` = ppwap). Measured
on the 308 real live runners in `data/predictions.json` that carry both prices
(mean executable/reference ratio 1.0143): the served `value_win_prob` differs
from its reference-price value by mean 0.00092 / max 0.05275, and a **+10% move
in the executable quote alone** shifts it by mean 0.00793 / max 0.04319 — on a
~0.10 base rate that is a ~8% relative swing in a served probability caused by a
bookmaker moving a quote.

`value_win_prob_independent` (what `execution/gates.py` prefers) is unaffected,
and the code documents the adjustment as market-adjusted, so this is a
fit/serve price mismatch rather than a mislabelled probability. Not changed in
this stage: it is a live-serving behaviour change, out of this work order's
experiment scope. `models/blend.py` avoids the class of problem by construction
— `power_blend` has no price argument at all.
