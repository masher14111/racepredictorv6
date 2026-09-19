# Forward-validation report — 2026-09-19

_Generated 2026-09-19T18:38:43+00:00. Deployment state: **PAPER-ONLY**._

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

_No backtest ledger._

_Selection protocol did not run._

---

## Lane 2 — paper / shadow bets

_Evidence kind: `paper`. Tickets issued in real time by the live gate against prices available at decision time, settled against results. **This is the only lane the forward-release gate reads.**_

_No paper tickets recorded yet. The forward window has not started, so the forward gate cannot pass — this is the expected state until the live loop has run for the configured minimum._

- Decisions logged: 308 (308 PASS — no stake, not a wager; 0 candidate)
- Candidate tickets (paper wagers): 0 (0 open, 0 settled)
- First ticket: 2026-09-18T00:00:00+00:00 · latest: 2026-09-18T00:00:00+00:00

### Forward-validation window

**NOT STARTED.** Live capture may be running independently of this clock — starting the formal 8-week validation window is a separate, deliberate action (`execution.window.start_window`), not implied by capture running.

### Capture history

- Days recorded: 2 (0 captured, 2 gap)

| date | cause |
|---|---|
| 2026-09-18 | scraper_outage |
| 2026-09-19 | no_racing |

---

## Lane 3 — real-money execution

**No real-money execution exists.** This system is paper-only by configuration (`execution.paper_only`), enforced in storage by a `CHECK (paper_only = 1)` constraint on the ticket table and in code by `TicketStore` raising `RealMoneyTicketRefused`. This lane is not empty pending data — there is no code path that can fill it.

---

## How to read this

- A short backtest cannot show that a strategy is safe or profitable. Sample size, the ROI interval and the CLV interval are printed for exactly that reason.
- Staking limits in this system are **risk ceilings, not profitability claims**. They must not be raised without forward evidence.
- "No bet" is a valid and expected output.
