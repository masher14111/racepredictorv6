---
name: calib-fl-03-integrate
description: Wired the OddsBandCalibrator F-L recalibrator into models/predictor._score_value so value_win_prob (+ value_edge/expected_value/value_bet/models.value) reflect the price-corrected price-free prob; config flag value.fl_recalibration (default on); 1082✓; full re-validation deferred to calib-fl-04 (2026-06-17)
metadata:
  type: project
---

# calib-fl-03 — integrate the F-L recalibrator into the value layer — 2026-06-17

Third prompt of the calibration programme ([[calib-fl-00-plan]]). Wired the
`OddsBandCalibrator` artifact built offline in [[calib-fl-02-fl-design]] into the
live predictor so the favourite-longshot correction now applies to the served
`value_win_prob`. No commit (per instruction).

## Wiring points (all in `models/predictor.py`)

- **Load** — `_load_value_model()` now loads `models/fl_oddsband_<tag>_calib.pkl`
  (tag = value model tag, `v3nf`) via the existing `_load_calibrator` pickle helper,
  alongside `_value_calibrator`. New instance attrs in `__init__`:
  `self._value_fl_recalibration` (bool, from config) and
  `self._value_fl_calibrator` (the loaded `OddsBandCalibrator` or `None`).
  Absent artifact **or** disabled flag ⇒ `None`, logged at INFO, never crashes —
  same graceful-degrade contract as `_value_calibrator` ([[model-17-inference]]).
- **Apply** — `_score_value()`: after the price-free model + `_value_calibrator`
  produce `vw`, it computes `dec = df.apply(_effective_decimal, axis=1)` **once**
  and calls the new `_apply_fl_recalibration(vw, dec)`. The recalibrated `vw` is
  written to `value_win_prob`, then `value_edge = vw - implied_prob` and
  `expected_value = vw * dec - 1` reuse the same `vw`/`dec`. So **one corrected
  probability** flows to every downstream consumer (edge, EV, `value_bet` gate in
  `_build_race`, the JSON cache, and `models.value.find_value_bets`, which reads the
  already-recalibrated `value_win_prob` from runner dicts — no double-application).
- **Price fed** = the **same effective decimal the EV uses** (`_effective_decimal`:
  best board price across bookmakers when known, else fused market odds). This is a
  pre-off, legitimately-available price — not the `odds_finish` close (CLV-only).

## Config flag

`config.yaml` `value:` → **`fl_recalibration: true`** (default on). Read in
`models/predictor._load_cfg` as `value_fl_recalibration` (`bool(v.get(
"fl_recalibration", True))`). Set `false` to serve the raw market-independent
price-free calibration. (Not added to `models.value.ValueConfig`: the value layer
itself does no recalibration — it consumes the already-corrected `value_win_prob` —
so a ValueConfig field would be dead. evaluate_filter for calib-fl-04 can apply the
artifact to its `scored.prob` directly.)

## Edge cases handled (`_apply_fl_recalibration`)

- **No recalibrator** → returns `prob` unchanged (identity).
- **Missing / invalid price** (NaN or `dec ≤ 1`) → that row keeps its original
  market-independent prob; the recalibrator returns NaN for NaN odds, so rows are
  masked and only finite recalibrated values overwrite. **An odds-less runner never
  loses its `value_win_prob` to a NaN.**
- **Non-finite recalibrated output** → masked out, original prob kept.
- `value_supported` / the longshot-trap support gate are untouched (computed from
  `historical_place_rate`, independent of the prob value).

## Test coverage

- `tests/models/test_predictor.py::TestFLRecalibration` (7): favourite revised
  up / longshot revised down (edge + EV follow); no-op when calibrator absent;
  odds-missing row keeps prob & stays finite; `_apply_fl_recalibration` identity
  without calibrator; config flag defaults on; flag-off skips the load even with the
  artifact present (skips if model bin absent); real saved artifact loads + revises a
  2.0 favourite upward (skips if absent).
- `tests/models/test_value.py::TestRecalibratedValueWinProb` (3): a recalibrated
  favourite surfaces as value through `find_value_bets`; a deflated longshot is gated
  out; same runner flips in/out of selection as its prob crosses the EV gate.
- **Suite: `python -m pytest -q` → 1082 passed, 3 skipped** (the 3 reportlab-only).

## Sanity-run (note only; full re-validation = calib-fl-04)

`python -m scripts.last_week_backtest` runs green (it scores the value strategy
through `models.yesterday._score_day → Predictor._score`, so the backtest's
`value_win_prob` is now recalibrated too). `value_win_prob` A/E by odds band on the
last settled week (2,338 runners, base 0.109), banded by SP:

| odds band      | A/E raw (pre-FL) | A/E post-FL |
| -------------- | ---------------- | ----------- |
| odds-on (<2.0) | 1.833            | 0.581       |
| 2.0–4.0        | 0.813            | 0.358       |
| 4.0–8.0        | 0.594            | 0.454       |
| 8.0–16.0       | 0.463            | 0.664       |
| 16.0–34.0      | 0.244            | 0.636       |
| longshot (>34) | 0.123            | 0.767       |

The extreme tail is hugely improved (longshot 0.12→0.77, 16–34 0.24→0.64), but on
this small week the whole curve sits **below 1** and the field mean overshoots
(0.140→0.228 vs base 0.109), i.e. it now over-predicts favourites. This is a
sample/price-distribution gap, **not** an integration bug: feeding the same artifact
its own held-out tail (`data/backtests/20260616_195957`, 12.9k runners) reproduces
the [[calib-fl-02-fl-design]] result exactly — A/E **1.01 / 0.98 / 1.00 / 0.95 /
0.95 / 0.79**, mean 0.115 vs base 0.112. The last-week frame differs from the
walk-forward fit frame in price source (SP/`implied_prob` vs pre-off `ppwap`),
in-band base rates, and size. Closing that gap + re-tuning the value gates on
**CLV + A/E** is **calib-fl-04** (also: recalibration injects market price, so
post-correction _band-level_ edge ≈ 0 — value now lives in within-price
disagreement; the edge gates must be re-validated against that, per
[[model-16-value-detection]]).

Relates to [[calib-fl-00-plan]], [[calib-fl-02-fl-design]], [[calib-fl-01-headline]],
[[model-17-inference]], [[model-16-value-detection]], [[model-14-backtest]].
