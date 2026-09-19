---
name: model-16-value-detection
description: Value-detection layer (models/value.py) — find_value_bets() API, de-vig/edge/EV/Kelly maths, and the backtester-validated thresholds (favourite band [2,6]; CLV still negative)
metadata:
  type: project
---

# Value-detection layer (`models/value.py`) — 2026-06-16

Turns calibrated price-free probabilities + market prices into ranked **value
bets**. Built per the task "Prompt 22 (suggestion engine) uses this". Reuses the
pure betting maths in [[model-14-backtest]] (`backtest.metrics`) and the honest
calibrated probs from [[model-09-baseline-audit]] (`v3nf`). Tests:
`tests/models/test_value.py` (42). Suite green (892 pass / 3 skip).

**Why:** a calibrated model is necessary but not sufficient — value only exists
where model prob > the market's _fair_ (de-vigged) prob AND the price still pays
after the book's margin. This layer makes that comparison apples-to-apples and
gates the result to the regime the backtester says is trustworthy.

**How to apply:**

## API

```python
from models.value import find_value_bets, ValueConfig, evaluate_filter
picks = find_value_bets(race)          # race = predictor race dict / runner list / DataFrame
# -> ranked list (highest edge first); each pick dict:
#   horse_id, horse_name, decimal_odds, model_prob, fair_prob,
#   market_implied_prob, overround, edge, edge_pct, expected_value,
#   kelly_fraction, suggested_stake, confidence, rank
```

- `find_value_bets(race, config=None, bankroll=None)` — `config` defaults to
  `ValueConfig.from_config()` (reads `value:` + `bet_tracker:` from config.yaml).
  Accepts a predictor race dict (`selections`/`excluded_low_odds`), a list of
  runner dicts, a single runner dict, or a runner DataFrame (resolves
  `value_win_prob`/`model_prob`/`prob` and `best_odds`/`decimal_odds`/`bet_price`).
- `evaluate_filter(scored, config)` — replays the SAME gates over a backtester OOS
  frame (`backtest` `scored.parquet`) and returns yield / CLV / beat-close / A/E by
  odds band / flat+Kelly bankroll. This is the validation entry point.

## Pipeline (per race)

1. **De-vig** (`devig_field`): `fair_i = (1/d_i) / Σ(1/d_j)` over the field →
   sums to 1; `overround = Σ(1/d) - 1`. De-vig needs a real per-race field, which
   `find_value_bets` HAS (its input is one race) — unlike the backtest panel where
   `race_time` is null and de-vig stays off (audit C3).
2. **edge = model_prob − fair_prob** (disagreement with the de-vigged market);
   **EV = model_prob·d − 1** uses the _raw_ price (what you get paid, margin
   included) so EV>0 ⇒ +EV at the available price.
3. **Gates** (all config-driven, ANDed): `min_ev`, `min_odds`/`max_odds` band,
   `min_edge_pct`, `min_abs_edge`, `min_prob`, `min_confidence`, `require_support`.
4. **confidence** ∈ [0,1] = support × field × odds-reliability:
   support (prior form) 1/0; field `clip(n_priced/8, 0.3, 1)` (de-vig quality);
   odds-reliability 1.0 ≤ `reliable_max_odds`(8), decaying to 0 by 3×.
5. **stake** = `clip(kelly_fraction·fullKelly, 0, kelly_cap)·bankroll`
   (¼-Kelly, 5% cap), full Kelly = `(p·d−1)/(d−1)`.

## Validated thresholds (config.yaml `value:`)

Validated on the saved walk-forward run `data/backtests/20260616_195957`
(77,323 OOS runners, 2025-01→2026-06; v3nf price-free, executed at `ppwap`).
Selected on **CLV + A/E**, not raw ROI.

| `value:` key       | value     | rationale                                 |
| ------------------ | --------- | ----------------------------------------- |
| min_expected_value | 0.05      | +EV at the available price                |
| min_odds           | 2.0       | skip odds-on shots                        |
| **max_odds**       | **6.0**   | **tightened from 26.0** — see below       |
| min_edge_pct       | 0.0 (off) | edge gating validated HARMFUL — see below |
| min_abs_edge       | 0.0 (off) | ""                                        |
| min_confidence     | 0.40      | drops thin fields / formless / longshots  |
| require_support    | true      | blocks the formless-longshot trap         |
| devig              | true      | strip over-round before edge              |

`kelly_fraction` (0.25) + `initial_bankroll` (1000) come from `bet_tracker:`;
`kelly_cap` 0.05.

## Headline validation result (`evaluate_filter`, validated defaults)

- **The whole field loses** (yield −11%, CLV −14%) — reproduces [[model-14-backtest]].
- **The favourite band `[2.0, 6.0]` + EV≥0.05 is the only profitable, sane regime:**
  n=1665, **yield +5.34%**, A/E 0.84–0.89, flat-1% bankroll 1000→1889, ¼-Kelly
  1000→2134. Above 6.0 the bet falls into the model's longshot-over-prediction zone
  (A/E 0.74 @8–16, 0.44 @16–34, 0.17 @>34) and yield + Kelly collapse ([2,8] ¼-Kelly
  → €31). Hence `max_odds 6.0`, not 26.
- **A probability-edge gate is HARMFUL.** `min_abs_edge`/`min_edge_pct` select the
  runners the model most over-rates vs the price (its own miscalibration), pushing
  in-selection A/E _below_ 1 and collapsing ¼-Kelly. Left OFF (kept config-driven
  for a future recalibrated model).
- **CLV stays NEGATIVE (~−7%) in every band**, beat-close ~27%. So the layer is
  profitable in backtest but **NOT yet proven to beat the closing line** — treat
  picks as provisional. The fix is the favourite-longshot recalibration flagged in
  [[model-14-backtest]] (the model is under-confident on favourites, over-confident
  on longshots); once corrected, re-run `evaluate_filter` and expect CLV to lift.

## Notes / caveats

- `models.predictor._runner_dict` now emits `value_supported` so the live race
  dict feeds the support gate. The predictor's own inline `value_bet` flag is the
  simple per-runner gate (raw implied, EV+band); `find_value_bets` is the richer
  field-aware **de-vigged** layer Prompt 22 should consume.
- De-vig over a real field RAISES apparent edge vs raw implied (fair < raw for a
  margined book) — correct, but EV (raw price) is the profitability gate, so the
  layer can't be fooled into a −EV bet by de-vig alone.
- `evaluate_filter` uses raw implied (no per-race key in the panel) — strictly
  conservative since raw ≥ fair; re-validate de-vig end-to-end once a per-race key
  (race_time/race_id) is recovered (audit C3).
