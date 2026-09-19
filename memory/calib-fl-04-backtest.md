---
name: calib-fl-04-backtest
description: Re-validated the F-L recalibration on the 77k walk-forward OOS frame — A/E flattened (2.80..0.17 -> 0.94..0.78), backtest yield lifted (band[2,6]+EV.05 5.3%->7.4%), but CLV STILL NEGATIVE everywhere (best ~-4% on favourite band [2,4], beat-close ~0.31). Recommends max_odds tighten 6->4 (CLV-first) but verdict: NOT yet proven to beat the close — provisional/paper-trade. Edge gates no longer harmful but don't lift CLV (2026-06-17)
metadata:
  type: project
---

# calib-fl-04 — re-validate F-L recalibration + re-tune value gates on CLV/A/E — 2026-06-17

Fourth prompt of the calibration programme ([[calib-fl-00-plan]]). Confirmed whether
the favourite-longshot recalibrator wired in [[calib-fl-03-integrate]] actually lifts
CLV / flattens A/E, and re-tuned the value gates on the **corrected** probabilities,
judged on **CLV + A/E first, yield second** (never raw ROI). No config change here
(that's calib-fl-05); no commit. Reproduce: `python -m scripts.calib_fl_04_eval`.

## Leak-free OOS frame (how & why)

Applied the recalibrator to the saved scored frame rather than re-running `backtest`,
because the F-L recalibrator is wired into `models/predictor`, **not** the backtest
`CatBoostFactory` — so `python -m backtest` would regenerate the same _raw_ price-free
prob, not the recalibrated one. Instead:

```
recal.prob = OddsBandCalibrator.predict(scored.prob, scored.bet_price)
```

on `data/backtests/20260616_195957/scored.parquet` (77,323 walk-forward OOS runners).
This is exactly how live scoring applies it (`predictor._apply_fl_recalibration` feeds
the **same effective decimal the EV uses**). Leak-free because: (1) `prob` is already
walk-forward OOS (model fit train-only per fold); (2) `bet_price` = pre-off ppwap, the
price legitimately available at bet time and the one EV uses — **not** the close;
(3) `close_price` (BSP) is touched only to measure CLV, never fed to the recalibrator
or any gate. **The recalibrator artifact was fit on folds 0–4 (`race_date<2026-03-26`),
so the genuinely-held-out evaluation of the recalibration is the fold-5 TAIL
(`race_date>=2026-03-26`, n≈12.9k).** Full-frame numbers are reported for direct
comparison to the [[model-16-value-detection]] baseline but are partly
recalibrator-in-sample.

Harness validated: RAW `prob`, band[2,6]+EV≥0.05, FULL → n=1665, yield **+5.34%**,
CLV **−7.19%** — reproduces the [[model-16-value-detection]] baseline exactly.

## Headline 1 — A/E is flattened (the win). FULL frame, band-all, EV≥0:

| odds band      | A/E pre (raw) | A/E post (recal) |
| -------------- | ------------- | ---------------- |
| odds-on (<2.0) | **2.803**     | 0.94             |
| 2.0–4.0        | **1.884**     | 0.94             |
| 4.0–8.0        | 1.239         | 0.97             |
| 8.0–16.0       | 0.739         | 0.92             |
| 16.0–34.0      | **0.437**     | 0.97             |
| longshot (>34) | **0.166**     | 0.78             |

The favourite-longshot bias is essentially gone end-to-end (longshot tail still mildly
low at 0.78, same residual as the [[calib-fl-02-fl-design]] held-out fit). This is the
real, robust improvement.

## Headline 2 — yield up, but CLV STILL NEGATIVE EVERYWHERE

Default gate band[2.0,6.0]+EV≥0.05, raw → recal:

| frame | prob  | n    | yield%    | CLV%      | beat_close | A/E (2-4 / 4-8) |
| ----- | ----- | ---- | --------- | --------- | ---------- | --------------- |
| FULL  | raw   | 1665 | +5.34     | −7.19     | 0.27       | 0.89 / 0.84     |
| FULL  | recal | 3088 | **+7.44** | **−6.24** | 0.30       | 0.92 / 0.99     |
| TAIL  | raw   | 252  | +20.47\*  | −5.95     | 0.29       | 1.08 / 0.91     |
| TAIL  | recal | 459  | +8.84     | **−5.74** | 0.28       | 0.88 / 1.01     |

\*tail raw yield inflated by a 252-bet sample; FULL is the reliable yield comparison.

Recalibration lifts yield (+2.1pt full) and improves CLV by only **~+1pt** (−7.19→−6.24
full; −5.95→−5.74 tail). **CLV stays negative and beat-close stays ~0.28–0.30 (<0.5) in
every gate tested.**

### CLV by gate — least-negative configs (recalibrated). Edge gates OFF.

FULL (largest n; partly in-sample):

| gate              | n    | yield% | CLV%      | beat | A/E         |
| ----------------- | ---- | ------ | --------- | ---- | ----------- |
| band[2,4] EV≥0.15 | 238  | +16.46 | **−3.78** | 0.37 | 0.93        |
| band[2,4] EV≥0.0  | 5084 | −2.11  | −4.02     | 0.35 | 0.94        |
| band[2,4] EV≥0.05 | 825  | +5.84  | −4.12     | 0.36 | 0.92        |
| band[2,4] EV≥0.10 | 414  | +13.21 | −4.38     | 0.34 | 0.94        |
| band[2,6] EV≥0.05 | 3088 | +7.44  | −6.24     | 0.30 | 0.92 / 0.99 |

TAIL (honest held-out):

| gate                | n   | yield% | CLV%      | beat | A/E            |
| ------------------- | --- | ------ | --------- | ---- | -------------- |
| band[2,4] EV≥0.0    | 812 | −0.01  | **−4.81** | 0.31 | 0.97           |
| band[1.5,6] EV≥0.10 | 209 | +8.45  | −4.79     | 0.31 | 0.79/0.94/0.97 |
| band[1.5,6] EV≥0.15 | 91  | −2.74  | −4.59     | 0.36 | —              |
| band[2,4] EV≥0.05   | 150 | +2.29  | −6.01     | 0.24 | 0.88           |
| band[2,6] EV≥0.05   | 459 | +8.84  | −5.74     | 0.28 | 0.88 / 1.01    |

**Pattern:** CLV is least-negative on the shortest-priced favourites (band [2,4], CLV
~−4%) and worsens monotonically as longer prices are admitted (band[2,6]→−6%, band-all
→−12 to −15%). Even at its best, CLV is **−4%** and we beat the close only ~1/3 of the
time. Widening max_odds past 4 buys yield/volume but costs CLV.

## Edge gates — re-confirm (the prior "HARMFUL" finding is softened)

Pre-recalibration, `min_edge_pct`/`min_abs_edge` were _harmful_ (they selected the
model's own miscalibration, pushing in-selection A/E below 1). **Post-recalibration that
no longer holds:** mild edge gates (`min_abs_edge 0.02`, `min_edge_pct 0.10`) lift yield
and keep A/E ~0.9 (FULL band[2,6]+EV.05+pct0.1 → yield 13.98, A/E 0.94) — the recalibrated
edge is genuine within-price disagreement, not miscalibration. **But they do NOT improve
CLV** (TAIL pct0.1 CLV −6.00 vs −5.74 off; FULL pct0.1 −5.65 vs −6.24 — noise either way),
and _heavy_ gates (pct 0.25 / abs 0.05) still collapse the sample and drop A/E to 0.68–0.74.
**Verdict: edge gates are no longer harmful, but neutral for CLV → keep OFF for a
CLV-first selection** (they remain a mild yield lever if yield is prioritised). `min_confidence`
cannot be swept in `evaluate_filter` (the OOS panel has no per-race field/support), so it is
left at 0.40 unchanged; its odds-reliability factor overlaps the odds band anyway.

## Recommended value gate set (CLV-first; PROVISIONAL — see verdict)

| `value:` key            | current | **recommended** | reason                                  |
| ----------------------- | ------- | --------------- | --------------------------------------- |
| min_odds                | 2.0     | 2.0             | skip odds-on                            |
| **max_odds**            | **6.0** | **4.0**         | CLV −4.1% vs −6.2% at 6.0; A/E ~0.92    |
| min_expected_value      | 0.05    | 0.05            | +EV at the available price; yield +5.8% |
| min_edge_pct            | 0.0     | 0.0 (off)       | doesn't lift CLV                        |
| min_abs_edge            | 0.0     | 0.0 (off)       | "                                       |
| min_confidence          | 0.40    | 0.40            | not testable here; overlaps odds band   |
| require_support / devig | true    | true            | unchanged                               |

This gives FULL n=825, yield **+5.84%**, A/E **0.92**, CLV **−4.12%** (TAIL n=150, yield
+2.29%, CLV −6.01%, A/E 0.88) — the best CLV/A/E combination that still carries a real EV
gate and a usable sample. **Alternative:** keep `max_odds 6.0` if volume/yield is wanted
over CLV — A/E now supports it (4–8 band A/E ~0.99 post-recal) at yield +7.4% but CLV −6.2%.

## Honest verdict — does CLV now beat the close? **NO.**

The recalibration did what it was designed to: **flattened A/E (2.80…0.17 → 0.94…0.78) and
lifted backtest yield.** It did **not** make the model beat the closing line. CLV remains
negative in every gate (best −4% on the tightest favourite band), beat-close ~0.31 (<0.5).
The recalibration improved CLV by only ~1pt where it overlaps the old bands. **The model is
still NOT proven bettable at the execution price — treat all picks as provisional / paper
only.** Calibration ≠ closing-line value: fixing the probability fixes profitability _at the
price taken_ but not the fact that the BSP close is sharper than our ppwap entry.

**Important live-vs-backtest nuance:** the backtest executes at **ppwap** (Betfair WAP); the
LIVE value layer takes the **best board price across bookmakers**, which is typically more
generous than ppwap and may beat BSP materially more often. So this backtest CLV is a
**conservative lower bound** for the live system — live CLV must be re-measured before
concluding "unbettable."

### Recommended next steps (instead of forcing a profitable-looking config)

1. **Forward / paper-trade** the band[2,4]+EV≥0.05 config and measure _realised_ CLV on
   live best-board prices vs eventual SP (the gap that actually matters).
2. Investigate **price-capture timing** — bet only where the board price sits above a
   threshold over projected SP (turn the negative CLV into a selection signal).
3. **Rebuild the full feature matrix** (9 missing price-free features, [[model-14-backtest]])
   and re-fit; better discrimination may move CLV.
4. Recover a per-race key (race_time, audit C3) to enable **de-vig** in `evaluate_filter`
   and re-validate end-to-end.

## calib-fl-05 (next)

Apply the recommended `value:` gates to `config.yaml`, re-run
`python -m scripts.last_week_backtest`, write the before/after report + known-limits, and
record the final programme verdict. Carry forward: CLV-negative ⇒ ship as
_provisional/advisory_, not a stake-with-confidence signal.

Relates to [[calib-fl-00-plan]], [[calib-fl-03-integrate]], [[calib-fl-02-fl-design]],
[[model-16-value-detection]], [[model-14-backtest]], [[model-09-baseline-audit]].
