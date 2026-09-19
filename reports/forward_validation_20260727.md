# Forward-validation report — 2026-07-27

_Generated 2026-07-27T21:41:12+00:00. Deployment state: **PAPER-ONLY**._

This report keeps three kinds of evidence apart. They are not interchangeable: a backtest says what the mechanics would have done, paper trading says what the system actually decided in real time, and only the second can satisfy the forward-release gate.

---

### Model gate: **NO-GO**

- Source: `data\audit\stage4\final_evaluation.json`
- Window: 2026-06-13 .. 2026-07-25 (904 races, 7922 runners)
- Headline line: `audit_independent`

| line | races | model LL | market LL | delta | delta 95% CI | beats market | beats best de-vig | significant |
|---|---|---|---|---|---|---|---|---|
| audit_independent | 904 | 1.8740 | 1.6878 | -0.1862 | [-0.2251, -0.1462] | FAIL | FAIL | FAIL |
| audit_market_adjusted | 904 | 1.6817 | 1.6878 | 0.0061 | [-0.0064, 0.0180] | PASS | PASS | FAIL |
| audit_market_adjusted_sigmoid | 904 | 1.6850 | 1.6878 | 0.0028 | [-0.0035, 0.0095] | PASS | PASS | FAIL |
| audit_grouped_temperature | 904 | 1.8663 | 1.6878 | -0.1785 | [-0.2182, -0.1377] | FAIL | FAIL | FAIL |
| frozen_v3nf_independent | 904 | 1.8690 | 1.6878 | -0.1812 | [-0.2228, -0.1419] | FAIL | FAIL | FAIL |
| frozen_v3nf_market_adjusted_sigmoid_refit | 904 | 1.6806 | 1.6878 | 0.0072 | [0.0011, 0.0145] | PASS | PASS | PASS |
| frozen_v3nf_market_adjusted | 904 | 1.6792 | 1.6878 | 0.0086 | [-0.0058, 0.0201] | PASS | PASS | FAIL |
| frozen_v3_priced | 904 | 1.6766 | 1.6878 | 0.0112 | [-0.0036, 0.0275] | PASS | PASS | FAIL |
| frozen_lgbm | 904 | 1.6772 | 1.6878 | 0.0105 | [-0.0046, 0.0268] | PASS | PASS | FAIL |

- logloss_edge_not_significant:delta=-0.18621 ci95=[-0.22507,-0.14620]
- loses_to_best_devig:shin=1.68568
- clv_not_positive:mean=-0.13417 ci95_lower=-0.13888
- drift_gate_failed:going_speed=4.916, horse_career_runs=0.301

---

### Forward-release gate

**FORWARD GATE NOT MET (9 of 9 criteria failed)** — evidence kind: `forward`

- Weeks elapsed: 0.00
- Qualified bets: 0 over 0 races

| criterion | result | observed | required |
|---|---|---|---|
| model_go | FAIL | NO-GO | Stage-4 model verdict == GO |
| min_weeks | FAIL | 0.0000 | >= 8 weeks of forward tracking |
| min_qualified_bets | FAIL | 0.0000 | >= 200 qualified bets |
| min_qualified_races | FAIL | 0.0000 | >= 150 qualified races |
| positive_mean_clv | FAIL | n/a | mean CLV > 0 |
| clv_ci_lower | FAIL | n/a | 95% race-clustered CI lower bound > 0 |
| ae_stable | FAIL | n/a | 0.9 <= A/E <= 1.1 |
| calibration | FAIL | n/a | ECE <= 0.03 |
| drawdown | FAIL | n/a | max drawdown <= 0.1 of bankroll |

**Failed criteria:** `model_go`, `min_weeks`, `min_qualified_bets`, `min_qualified_races`, `positive_mean_clv`, `clv_ci_lower`, `ae_stable`, `calibration`, `drawdown`

> Forward-release gate NOT met: 9 of 9 criteria failed (model_go, min_weeks, min_qualified_bets, min_qualified_races, positive_mean_clv, clv_ci_lower, ae_stable, calibration, drawdown). Observed 0.0 weeks of forward evidence, 0 qualified bets across 0 qualified races. 0 criteria were met. The deployment stays paper-only.

---

## Lane 1 — historical backtest

_Evidence kind: `backtest`. Simulated frictions over a fixed historical window. This measures execution mechanics. It is **not** evidence that a strategy is profitable and it cannot satisfy the forward gate._

_Walk ran: **edge0.020_ev0.050_short_lt_4** (source: selected strategy) — min edge 0.020, min EV 0.050, odds band 1.00–4.00. model_only only — baselines bet the whole book_

| strategy | bets | races | ROI | ROI 95% CI | hit rate | mean CLV | CLV 95% CI (log) | A/E | A/E 95% CI | max DD | worst streak | turnover |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| model_only | 63 | 63 | 31.10% | [-9.07%, 70.89%] | 39.68% | -8.25% | [-0.1248, -0.0644] | 1.066 | [0.737, 1.403] | 2.63% | 7 | 302.60 |
| devigged_market | 74 | 74 | 23.33% | [-27.76%, 79.64%] | 21.62% | -5.53% | [-0.0925, -0.0369] | 1.194 | [0.718, 1.714] | 3.83% | 12 | 177.70 |
| favourite | 153 | 153 | 12.13% | [-16.63%, 40.41%] | 36.60% | -2.13% | [-0.0442, -0.0111] | 0.999 | [0.799, 1.200] | 6.02% | 12 | 467.50 |

**ROI read:** 31.10% over 63 bet(s), 95% CI [-9.07%, 70.89%] — the interval spans zero, so this result is **not distinguishable from chance** on 63 bet(s). It is not evidence of profitability and must not be reported as a return.

**CLV read:** mean -0.0937 log units (-8.25%), 95% CI [-0.1248, -0.0644] — negative: bets are being struck at prices the market subsequently beat. Per `CLAUDE.md`, a positive model verdict with negative CLV is paper-only and never a green light to bet.

**model_only** — 63 struck of 66 attempts (archive-settled 100.00%); blocked by: `SUSPENDED` x2, `REJECTED` x1

**devigged_market** — 74 struck of 105 attempts (archive-settled 100.00%); blocked by: `min_stake` x30, `REJECTED` x1

**favourite** — 153 struck of 1480 attempts (archive-settled 100.00%); blocked by: `no_edge` x1247, `min_stake` x65, `REJECTED` x10, `SUSPENDED` x3, `daily_exposure_limit` x2

**What this lane does not exercise:**

- live source-health staleness (scraper telemetry describes today, not the backtest window; quote age is enforced instead)
- price movement between observations (the warehouse holds one pre-off quote per runner, so movement is modelled, not measured)
- each-way terms and best-odds-guaranteed (neither is recorded historically; win-only unless terms are passed explicitly)

### Selection protocol (anti-threshold-mining)

- Strategies tried: **64**
- Multiple-testing correction: **sidak**, alpha 0.050 -> adjusted **0.00080**
- Selected once: **edge0.020_ev0.050_short_lt_4** `{"min_edge": 0.02, "min_expected_value": 0.05, "odds_band": "short_lt_4", "odds_max": 4.0, "odds_min": 1.0}`
- Train score -0.0660 · validation -0.0685 · **test -0.0923**
- Test window scored at: 2026-07-27T21:27:01+00:00 (reused existing lock — NOT re-scored)

| window | start | end | rows |
|---|---|---|---|
| train | 2025-01-01 | 2025-11-05 | 44584 |
| validation | 2025-11-06 | 2026-02-20 | 13204 |
| test | 2026-02-22 | 2026-05-25 | 14229 |

> 64 strategy variant(s) were scored on the training window; 16 were carried to validation and the single winner was scored on the test window exactly once. A family-wise alpha of 0.05 becomes 0.0008011 under the sidak correction for 64 comparison(s). The reported test score is a single draw from one window and does not by itself establish an edge.

**Selection lock was reset 1 time(s).** The test window has therefore been scored more than once across this programme, so the Šidák correction above **understates** the true multiplicity. Treat the current test score as the more optimistic of several draws.

| superseded lock | winner | test score | scored at | why it was reset |
|---|---|---|---|---|
| `data/execution\selection_lock_superseded_20260727.json` | edge0.080_ev0.050_mid_4_12 | -0.14456 | 2026-07-27T21:10:20+00:00 | The odds-band filter was applied to the scored frame instead of to the backable-runner mask. That renormalised the de-vig over a field with its favourites removed (so `edge` was measured against a book no bookmaker offered) and changed the eligible race set per strategy (the `short_lt_4` band reported 0 of 3817 races eligible, and 16 of 64 candidates were never scored at all). Fixed in execution/baselines.py:model_only_selection; the sweep was re-run from scratch. |

---

## Lane 2 — paper / shadow bets

_Evidence kind: `paper`. Tickets issued in real time by the live gate against prices available at decision time, settled against results. **This is the only lane the forward-release gate reads.**_

_No paper tickets recorded yet. The forward window has not started, so the forward gate cannot pass — this is the expected state until the live loop has run for the configured minimum._

- Tickets: 0 (0 open, 0 settled)
- First ticket: n/a · latest: n/a

---

## Lane 3 — real-money execution

**No real-money execution exists.** This system is paper-only by configuration (`execution.paper_only`), enforced in storage by a `CHECK (paper_only = 1)` constraint on the ticket table and in code by `TicketStore` raising `RealMoneyTicketRefused`. This lane is not empty pending data — there is no code path that can fill it.

---

## How to read this

- A short backtest cannot show that a strategy is safe or profitable. Sample size, the ROI interval and the CLV interval are printed for exactly that reason.
- Staking limits in this system are **risk ceilings, not profitability claims**. They must not be raised without forward evidence.
- "No bet" is a valid and expected output.
