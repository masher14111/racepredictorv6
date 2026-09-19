---
name: calib-fl-05-final
description: Final prompt of the calibration programme — shipped the calib-fl-04 value gate set (max_odds 6.0->4.0, CLV-first) to config.yaml, regenerated predictions (headline sane, no 0.73 saturation, sum-to-1), re-ran last-week backtest, wrote the honest before/after verdict. Bottom line — ranks well, headline calibrated, F-L bias fixed, but CLV still negative => paper-only/not yet bettable (2026-06-17)
metadata:
  type: project
---

# calib-fl-05 — land the programme + honest before/after verdict — 2026-06-17

Fifth/final prompt of the calibration programme ([[calib-fl-00-plan]]). Applied the
recommended value gates to `config.yaml`, regenerated live predictions, re-ran the
last-week backtest, updated the docs, and recorded the bottom-line verdict. Suite
green (**1082 passed, 3 skipped**). **No commit yet — awaiting the user's go-ahead to
commit the whole programme to `chore/config-audit`.**

## Config change (config.yaml `value:` — the one functional change)

| key                   | before | after   | why                                                                                                            |
| --------------------- | ------ | ------- | -------------------------------------------------------------------------------------------------------------- |
| **max_odds**          | 6.0    | **4.0** | CLV-first ([[calib-fl-04-backtest]]): A/E now flat, so the cap is CLV-driven; CLV least-negative on band [2,4] |
| min_odds              | 2.0    | 2.0     | unchanged (skip odds-on)                                                                                       |
| min_expected_value    | 0.05   | 0.05    | unchanged                                                                                                      |
| min_edge_pct          | 0.0    | 0.0     | off — no longer "harmful" post-recal, but CLV-neutral                                                          |
| min_abs_edge          | 0.0    | 0.0     | off — same                                                                                                     |
| min_confidence        | 0.40   | 0.40    | unchanged (not sweepable in evaluate_filter)                                                                   |
| require_support/devig | true   | true    | unchanged                                                                                                      |
| fl_recalibration      | true   | true    | unchanged (wired in [[calib-fl-03-integrate]])                                                                 |

Also rewrote the `max_odds` comment (CLV rationale) and the `min_edge_pct/min_abs_edge`
comment (the "VALIDATED HARMFUL" wording is superseded — post-recalibration they're
CLV-neutral, kept off for a CLV-first selection). `config.local.yaml` untouched; no
secrets printed.

## Files touched (no commit)

- **config.yaml** — `max_odds` 6.0→4.0 + the two comment rewrites above.
- **models/value.py** — `ValueConfig.max_odds` default 6.0→4.0 and `from_config` fallback
  6.0→4.0 (the dataclass docstring says defaults _mirror config.yaml_, so they must track
  it), plus the module docstring rewritten for the post-recalibration state.
- **models/predictor.py** — `_load_cfg` `value_max_odds` fallback **26.0→4.0** (stale; only
  the config value is ever used, but it diverged).
- **backtest/**main**.py** — `max_odds` fallback **26.0→4.0** (same stale-default fix).
- **Tests** — `tests/models/test_value.py` (default-assert 6.0→4.0; the OOS-baseline guard
  made explicit `ValueConfig(max_odds=6.0)` so the model-16 [2,6] n=1665/yield+5.34/CLV−7.19
  invariant survives the default change; band comments [2,6]→[2,4]); `test_predictor.py`
  (two `_scored_grp` fixtures moved ip 0.20→0.30 so the price stays in the new [2,4] band and
  the test isolates the gate it names; comments 6.0→4.0); `test_suggestions.py` (band test
  assertion + comment 6.0→4.0).
- **README.md** (known-limits rewritten + test baseline 695→1082), **PROGRESS.md** (RUN LOG),
  **CHANGELOG.md** (2026-06-17 bullet), **data/predictions.json** (regenerated).

## Regenerated predictions (`python -m scripts.refresh --no-scrape`)

111 races / 1223 runners; F-L recalibrator loaded. Headline `won_prob_normalized`:
mean **0.093**, max **0.375**, **0 runners >0.70** (no 0.73 saturation), and the full-field
within-race sum = **1.000 across all 122 races**. The raw `won_prob` debug column still
saturates (max 0.734, 12 runners >0.70) — expected, it's the EV-reference column. The JSON's
per-race headline sums look <1 only because the cache stores the top selections, not the
full field (field_size > stored count for nearly every race).

## Before → after

### 1. Headline win-prob calibration — last-week OOS (2,338 runners, 254 winners, base 0.109)

| column                              | before (calib-fl-00)                    | after (this run)                            |
| ----------------------------------- | --------------------------------------- | ------------------------------------------- |
| **HEADLINE** `won_prob_normalized`  | n/a (raw was headline)                  | mean 0.109, AUC **0.778**, ECE **0.029** ✅ |
| raw marginal `won_prob` (debug now) | mean 0.331, AUC 0.773, **ECE 0.222** ❌ | mean 0.331, ECE 0.222 (demoted to debug)    |

Bug 1 fixed and holding: the headline is the calibrated within-race prob (ECE 0.029, no
saturation); the overconfident marginal is retained only as a debug/EV column.

### 2. Favourite-longshot A/E by odds band — 77k walk-forward OOS frame ([[calib-fl-04-backtest]])

| odds band      | A/E before (raw) | A/E after (recal) |
| -------------- | ---------------- | ----------------- |
| odds-on (<2.0) | **2.80**         | 0.94              |
| 2.0–4.0        | **1.88**         | 0.94              |
| 4.0–8.0        | 1.24             | 0.97              |
| 8.0–16.0       | 0.74             | 0.92              |
| 16.0–34.0      | **0.44**         | 0.97              |
| longshot (>34) | **0.17**         | 0.78              |

Bug 2 fixed: the F-L bias is essentially gone (longshot tail mildly low at 0.78).

### 3. Value gate yield / CLV — 77k frame ([[calib-fl-04-backtest]])

| gate (EV≥0.05)                 | yield%    | CLV%      | beat-close |
| ------------------------------ | --------- | --------- | ---------- |
| band [2,6] raw (old gate)      | +5.34     | −7.19     | 0.27       |
| **band [2,4] recal (shipped)** | **+5.84** | **−4.12** | **0.36**   |

Recalibration + the tighter band lift yield and roughly halve the CLV deficit, but **CLV
stays negative**.

### 4. Last-settled-week SP-settled strategies (calib-fl-00 before → this run after)

| strategy           | before (bets / ROI) | after (bets / ROI) |
| ------------------ | ------------------- | ------------------ |
| value, each-way    | 20 / −51.6%         | 18 / **−60.2%**    |
| value, win         | 20 / −77.2%         | 18 / **−72.4%**    |
| top-pick, each-way | 254 / −25.5%        | 254 / **−25.5%**   |
| top-pick, win      | 254 / −32.7%        | 254 / **−32.7%**   |

The value strategy still **loses at SP** (and the 18-bet week is noise-dominated — one
0-for-5 day is −100%). **SP settlement is conservative**: the backtest settles at the
starting price / ppwap, while the live layer takes the **best board price across
bookmakers**, typically more generous. So this is a lower bound; the real test is CLV, and
CLV is not yet positive.

## Bottom-line verdict

- **Ranks well** — AUC ~0.78 headline, ~0.79 value (within-price discrimination preserved).
- **Headline now calibrated** — ECE 0.222 → 0.029, no saturation, sum-to-1 per race.
- **Favourite-longshot bias fixed** — A/E flattened 2.80…0.17 → 0.94…0.78; value band
  tightened to the best-CLV regime [2,4].
- **Bettable: NO (not yet).** CLV is negative in every gate (best ~−4%), beat-close ~0.36
  (<0.5), and the model loses at SP. Calibration ≠ closing-line value — fixing the
  probability fixes profitability _at the price taken_, not the fact that the BSP close is
  sharper than our entry. **Ship as provisional / advisory / paper-only**, not a
  stake-with-confidence signal, until **live CLV (best board price vs eventual SP)** is
  measured.

## Next (post-programme, from calib-fl-04 step list)

1. Forward/paper-trade band[2,4]+EV≥0.05 and measure _realised_ live CLV (best board vs SP).
2. Price-capture timing — bet only where the board price clears a threshold over projected SP.
3. Rebuild the full price-free feature matrix (9 missing feats) and re-fit — better
   discrimination may move CLV.
4. Recover a per-race key (race_time) to enable de-vig in `evaluate_filter`.

Relates to [[calib-fl-00-plan]], [[calib-fl-01-headline]], [[calib-fl-02-fl-design]],
[[calib-fl-03-integrate]], [[calib-fl-04-backtest]], [[model-16-value-detection]],
[[model-09-baseline-audit]].
