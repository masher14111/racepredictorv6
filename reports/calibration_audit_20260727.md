# Stage-4 calibration & model-validity audit — 2026-07-27

**AUDIT STATUS: COMPLETE.** **MODEL VERDICT: NO-GO — paper-only.** The independent price-free model is decisively worse than the de-vigged pre-off market; the market-adjusted lines are statistically indistinguishable from the market once compared against the best de-vig baseline; CLV is deeply negative; the drift gate fails formally. No real-money recommendation is supportable.

- Git HEAD at audit: `abdfae421d90851c74bbf4982be185bcc6d3c814`
- Reproduce: `python -m scripts.calibration_audit --run stage4 --all` (phases: panel → leakage → walkforward → select → final → report), then `python -m scripts.refit_fl_recalibrator` for the recalibrator swap. Data refresh that preceded it: `python -m scripts.fetch_results_window --start 2026-06-13 --end 2026-07-26` + normalizer + `build_training_matrix`.
- Seeds: CatBoost 42, bootstrap 42 (B=1000). GPU fits (fold timings in `wf_meta.json`).

## Data, windows, cutoffs

- Training matrix: `C:\Users\mshr\Desktop\Race Predictor v4\data\features\training.parquet` (sha256 `3E7CDE9869D86ED2…`), rebuilt this session after a 44-day results catch-up (Betfair SP backbone + Sporting Life enrichment, 31,584 rows merged, position fill 81.2%).
- Audit panel: **143,364 WIN-market runner rows / 16,039 races**, 2024-01-01 → 2026-07-25; complete pre-off books 79.8%.
- Excluded rows (requirement 1, reason-coded):

| reason | rows | races |
|---|---:|---:|
| non_win_market_row | 111,630 | 12,522 |
| no_valid_preoff_price | 11 | 6 |
| implausible_field_size(<2|>40) | 7 | 7 |
| extreme_booksum(<0.8|>1.8) | 5,478 | 946 |

  (PLACE-market rows are the deliberate bulk: `models/train.py` historically trained on WIN+PLACE rows together — every runner duplicated with a different sample weight. The audit evaluates on the WIN market only. The 946 extreme-booksum races are almost all early-morning thin exchange books.)

- Frozen artifact cutoffs: v3nf/v3 chronological 80% split → **2025-12-02** (June matrix); LightGBM **2026-05-22**; every published holdout ended **2026-06-12**; value gates tuned on frames ≤ 2026-06-16.
- Walk-forward validation folds: 17 expanding folds, test windows ≤ 2026-06-12 (72,017 scored rows). All preprocessing/fitting/calibration inside each fold's train slice; fixed hyperparameters from the shipped v3nf meta (documented contamination channel for pre-2026-06 folds; the final window is clean of it).
- **Final untouched window: 2026-06-13 → 2026-07-25** — results fetched 2026-07-27, postdating every model, holdout and gate-tuning frame. Scored once (plus one prespecified addendum line; nothing tuned on it). 7,922 runners / 904 races on identical complete-book races for every line.

## Requirement 2 — price-free feature proof

- Structural provenance: **PASS** — 40 price-free columns disjoint from every price/post-off/outcome column.
- Empirical guards (closing-move partial-corr ≤ 0.30, |corr won| ≤ 0.70): **PASS** over the full panel.
- Append-future invariance: **FAIL for exactly one feature** — `race_complexity` is z-scored over the whole dataset (its market-entropy component also injects race-level market shape). Drift is cosmetic (mean |Δ| 0.0024 on a 0.574-std feature, rank-corr 1.0) but it violates point-in-time construction, so it is **excluded from every audit-fitted model**; frozen artifacts keep it (documented caveat). All other 53 features are bit-invariant when future data is appended.
- Sample-weight channel: `models/train.py` weights rows by 1/implied_prob = morningwap (pre-off, verified exact on 227k rows) — market-informed training, not a feature leak; documented.

## Requirement 5 — calibration-method comparison (validation folds only)

| arm | race log-loss | Brier | ECE |
|---|---:|---:|---:|
| raw_normalized | 1.91399 | 0.09368 | 0.01102 |
| marginal_calib_then_normalize (production) | 1.89429 | 0.09319 | 0.00679 |
| grouped_softmax_temperature | 1.89485 | 0.09332 | 0.00625 |
| fl_market_adjusted_then_normalize | 1.74317 | 0.08684 | 0.00447 |

Among PRICE-FREE approaches, marginal-calibration-then-normalize (production) and grouped softmax temperature scaling are equivalent (Δ ≈ 0.0006; fitted T ≈ 0.92–1.02). The F-L market-adjusted arm wins only because it imports the market price. Production recipe retained.

**F-L band method (extremes fix), selected on validation folds:** sigmoid bands 1.70523 vs isotonic 1.74317 race log-loss, equal Brier, and **0 exact 0/1 emissions vs 3,154 zeros + 21 ones** — sigmoid adopted.

## Requirement 3 — F-L circularity (cross-fitted ablation)

- corr(prob, 1/price): independent 0.578 → adjusted **0.925** — the adjusted probability substantially collapses onto the price.
- Within-price-bin AUC: 0.5489 → 0.5365 — the adjustment does NOT preserve within-price discrimination (design claim falsified), and the independent signal is itself thin (~0.55).
- Manufactured edge: 4,932 OOS rows turn EV-positive ONLY via the price-conditioned remap; in the betting band [2,4]: 2,484 rows, realized A/E 0.922, flat yield -0.0194 — phantom edge that loses.
- Price-shock absorption: **77%** of a +5% price improvement's EV gain is absorbed by the recalibration (price-free behaviour would absorb 0%).
- In-sample vs cross-fit optimism: Brier +0.0016.
- Consequence: `value_win_prob` is market-adjusted and is now persisted alongside the price-free `value_win_prob_independent` (predictor/value/picks/schema; regression-tested).

## Requirement 4 — probability extremes / support / portability

- Base v3nf calibrator: SigmoidCalibrator (a=1.100, b=0.484) — cannot emit 0/1.
- Shipped isotonic F-L bands (pre-refit): exact-0.0 floor in EVERY band, exact-1.0 ceilings in odds-on bands (8–24 fitted thresholds per band); 3.6% of OOS inputs fall outside fitted support and are answered by endpoint clamping; live cache carried 20/356 exact-zero `value_win_prob`.
- Portability: the artifact is fit on exchange ppwap but applied to best-board bookmaker prices; band-occupancy shift is recorded in `selection.json::extremes.portability`.
- **Fix shipped:** `models/fl_oddsband_v3nf_calib.pkl` refit with sigmoid bands on 26,252 rows of the frozen model's own OOS span (2025-12-03 → 2026-06-12); previous isotonic artifact backed up at `C:\Users\mshr\Desktop\Race Predictor v4\data\backups\fl_oddsband_v3nf_calib_isotonic_20260617.pkl` (sha256 `548BB776C71C33FD…`). New artifact emits no exact 0/1 anywhere on the sanity grid; final-window check below.

## Requirements 7/8/10 — final untouched window, identical races

Race-level log-loss vs the proportional de-vigged pre-off market, race-bootstrap 95% CIs (B=1000). ‘boot+’ = fraction of resamples with the model ahead.

| line | log-loss | Δ (mkt−model) | 95% CI | boot+ | sig. beats mkt |
|---|---:|---:|---|---:|---|
| audit_independent | 1.87400 | -0.18621 | [-0.22507, -0.14620] | 0.00 | NO |
| frozen_v3nf_independent | 1.86902 | -0.18124 | [-0.22283, -0.14192] | 0.00 | NO |
| audit_grouped_temperature | 1.86632 | -0.17853 | [-0.21816, -0.13767] | 0.00 | NO |
| audit_market_adjusted | 1.68173 | +0.00605 | [-0.00644, +0.01796] | 0.84 | no (ns) |
| audit_market_adjusted_sigmoid | 1.68499 | +0.00280 | [-0.00354, +0.00954] | 0.80 | no (ns) |
| frozen_v3nf_market_adjusted | 1.67919 | +0.00859 | [-0.00577, +0.02007] | 0.89 | no (ns) |
| frozen_v3nf_market_adjusted_sigmoid_refit | 1.68058 | +0.00720 | [+0.00107, +0.01454] | 0.99 | YES |
| frozen_v3_priced | 1.67660 | +0.01118 | [-0.00358, +0.02754] | 0.92 | no (ns) |
| frozen_lgbm | 1.67724 | +0.01055 | [-0.00463, +0.02681] | 0.93 | no (ns) |

Market de-vig baselines on the same races: proportional 1.68778, power 1.68631, **shin 1.68568**. Every ‘model beats market’ delta above is measured against the WEAKEST baseline (proportional); against shin, the best line's edge shrinks to ≈ +0.005 and no line is significant. The one nominally-significant delta (sigmoid-refit, +0.0072 [+0.0011, +0.0145]) is a price-echo line (corr 0.93 with 1/price) evaluated against the weakest baseline — not evidence of bettable edge.

### CLV and EV simulation (final window)

- CLV (pre-off ppwap vs BSP), all common rows: mean log CLV **-0.1342** [-0.1389, -0.1295], beat rate 0.265.
- audit_independent: 4,597 EV>0 flat bets → ROI -0.234, strike 0.060 [0.053, 0.067], CLV -0.1746.
- audit_market_adjusted: 2,962 EV>0 flat bets → ROI -0.085, strike 0.101 [0.090, 0.112], CLV -0.1384.
- frozen_v3nf_market_adjusted: 2,490 EV>0 flat bets → ROI -0.179, strike 0.116 [0.104, 0.130], CLV -0.1294.
- frozen_v3nf_market_adjusted_sigmoid_refit: 2,621 EV>0 flat bets → ROI -0.168, strike 0.103 [0.092, 0.115], CLV -0.1353.

Every EV-selected portfolio loses at the pre-off price and shows deeply negative CLV. (Integrity suite: OK apart from the standing liquidity WARN — ppwap volume is not modelled.)

### Drift gate

- Worst PSI: **4.92 → FAIL** (`going_speed`, a 79%-null feature already catalogued empirically dead — the shift is summer-going seasonality plus an enrichment fill-rate change; next worst `horse_career_runs` 0.30).
- Monthly race log-loss delta vs market is stable (2026-06: −0.189, 2026-07: −0.185): performance is consistently behind the market, not degrading — but the formal gate FAILS and requirement 11 therefore forces NO-GO regardless of the head-to-head.

## Requirement 9 — win-rate semantics

Every realized win rate in the UI now renders as `rate% (wins/denominator)` with a Wilson 95% CI and window/rule label (`ui/_winrate.py`; performance dashboard + performance page + bet placer + Yesterday's predictor). Predicted probabilities are labelled `Model win %`, never bare “win rate”. Tracker summaries now carry raw `wins`/`places` counts. Regression: `tests/ui/test_winrate.py`.

## Artifact hashes (SHA-256)

| artifact | sha256 |
|---|---|
| `data/features/training.parquet` | `3E7CDE9869D86ED2BB8701CB5BE6F5244F6A728B20B869857662AC8C1E54AD99` |
| `data/backups/training_20260618_pre_stage4.parquet` | `1C20FA97BDC982F5CC3DAEF5F8881D88E300B16A3C0891219575C27B031B9246` |
| `models/catboost_won_v3nf.bin` | `9D45EF36CAE6FEB26DDD8C5C3130E3423B736B24E1960E46033359523BCB518B` |
| `models/catboost_won_v3nf_calib.pkl` | `27C992ADCDDCC22E531E25DDC37899ADDD07728C4F8688C7F3C4D04EF7A91FD3` |
| `models/fl_oddsband_v3nf_calib.pkl` | `ECCED5EEE2CD7B609FE0A09D5B5A61D96B2B745A28BAB4CFAABEA300555631B1` |
| `models/catboost_won_v3.bin` | `A064701968D1023B88483C60D6BEB04246DB4C44191B9250FDADBD51A8715E32` |
| `models/lgbm_won_v3.txt` | `2CB869453916C6B55DFA80F5C48B7C435CA1147C9181C7E2F10ECFA0EE848948` |
| `data/audit/stage4/panel.parquet` | `D8521BE44E54291A2616BBAF7922AA8B055ABE56E270995B2C37A8A5D498FEEE` |
| `data/audit/stage4/wf_scored.parquet` | `DA8A6CED6A9D7BBAFC495293C934D24997F344A837991AC55C589BC64414D0FE` |

## Verdicts

- **AUDIT: COMPLETE and reproducible** (commands above; every phase artifact under `data/audit/stage4/`).
- **MODEL: NO-GO — paper-only.** Grounds: (1) the independent price-free line loses to the market by 0.186 log-loss [CI −0.225, −0.146], 0/1000 favorable resamples; (2) no line significantly beats the best de-vig market baseline; (3) the market-adjusted ‘edge’ is ~93% price echo and manufactures losing bets; (4) CLV −13.4% [−13.9, −13.0], beat rate 26.5%; (5) every EV portfolio loses at the pre-off price; (6) the PSI drift gate fails. Real-money recommendations remain disabled; do not tune filters until the market-relative gates pass.
