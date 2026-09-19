---
name: calib-fl-00-plan
description: Master plan + diagnosis for fixing the headline won_prob calibration bug and the favorite-longshot recalibration; 5-prompt sequence (calib-fl-01..05), each run in a fresh chat
metadata:
  type: project
---

# Calibration + favorite-longshot fix programme — 2026-06-17

Two model defects were diagnosed (see numbers below) and a 5-prompt plan was written
to fix them across separate chat sessions (token hygiene — clear the chat between
prompts; each prompt re-bootstraps from these memory files). The prompts live in
`CALIBRATION_FIX_PROMPTS.md` at the repo root. Each prompt writes its own
`calib-fl-0N-*.md` memory + a MEMORY.md index line on completion; read those (and
this file) at the start of every prompt to recover state.

**Always run Python as `python` (= 3.14 / numpy 2.4 on this box), not the bare
`streamlit`/`python310`** — calibration numbers must be measured under the official
env. The portable `IsotonicCalibrator`/`SigmoidCalibrator` (plain-float) load in
either env; see [[perf-30-rebuild-speed]] / [[model-11-calibration]].

## Diagnosis (last-week settlement + OOS quality, 2026-06-06..06-12, 7 settled days)

Built `scripts/last_week_backtest.py` (loops `models.yesterday.run_yesterday` over
the last 7 settled days for value/top-pick × win/each-way, plus an OOS calibration
table). Run: `python -m scripts.last_week_backtest`. Results — **every strategy lost**,
settling at SP:

| strategy           | bets | wins | strike | ROI    |
| ------------------ | ---- | ---- | ------ | ------ |
| value, each-way    | 20   | 1    | 5.0%   | −51.6% |
| value, win         | 20   | 1    | 5.0%   | −77.2% |
| top-pick, each-way | 254  | 85   | 33.5%  | −25.5% |
| top-pick, win      | 254  | 85   | 33.5%  | −32.7% |

OOS win-prob quality over the week (2,338 runners, 254 winners, base rate 0.109):

| column                   | mean pred | AUC   | ECE                        |
| ------------------------ | --------- | ----- | -------------------------- |
| `won_prob` (UI headline) | **0.331** | 0.773 | **0.222** ❌ overconfident |
| `won_prob_normalized`    | 0.109     | 0.778 | 0.029 ✅                   |
| `value_win_prob` (v3nf)  | 0.140     | 0.665 | 0.033 ✅                   |

## Bug 1 — headline `won_prob` is overconfident

The loaded `won` calibrator (`models/catboost_won_v3_calib.pkl`, an
`IsotonicCalibrator`) has x-knots only over **[0.252, 0.509]** → y **[0.007, 0.734]**.
At inference it saturates: raw<0.25→0.007, raw>0.51→0.734, so the field piles up at
the 0.734 ceiling (mean 0.331 vs true 0.109). The v3 _priced_ model (28 features,
built on leaked `odds_finish`, audit C2 in [[model-09-baseline-audit]]) emits a
compressed/narrow raw range on the historical calibration slice but a wider range on
recent live data — distribution shift the saturating curve can't absorb. Meanwhile
`won_prob_normalized` (within-race sum-to-1, `models.calibration.normalize_within_race`,
applied at `models/predictor.py` ~L415-417) is already AUC 0.778 / ECE 0.029. The
code comment at predictor.py ~L408-414 claiming "won_prob marginally excellent
(ECE≈0.005), keep as headline" is now **false on OOS** and must be revisited.

Key code: `models/predictor.py` `_score` (calibration L382-383, normalized L415-417),
`_runner_dict` (L856-858 emits both); `models/calibration.py`; calibrators fit in
`models/train.py` L232-288 (time-ordered tail slice, `calibration_size` 0.15).

## Bug 2 — favorite-longshot (F-L) bias, the profitability blocker

From the saved walk-forward run (77k OOS) in [[model-14-backtest]] /
[[model-16-value-detection]]: A/E by odds band — odds-on **2.80**, [2,4] **1.88**
(favs win far more than predicted) vs [8,16] 0.74, [16,34] **0.44**, >34 **0.17**
(longshots hugely over-predicted). The price-free `value_win_prob` (v3nf) leans into
longshots where the market is sharpest → **CLV −9..−12%**, beat-close ~24-27%. The
validated value band [2.0,6.0]+EV≥0.05 yields +5.3% in backtest but **CLV still −7%**,
so it is not proven to beat the close. Fix = odds-band / F-L recalibration of the
price-free win prob, re-validated on CLV + A/E (not ROI).

Entry points for validation: `models.value.evaluate_filter(scored, cfg)` (replays
gates over a backtest `scored.parquet`, returns yield/CLV/beat-close/`ae_by_odds`);
`backtest` CLI `python -m backtest --task-type GPU`; `backtest.metrics.ae_table` /
`_ODDS_BANDS`. Config gates in `config.yaml` `value:` and `models.value.ValueConfig`.

## The 5 prompts (sequential; clear chat between)

1. **calib-fl-01** — fix headline `won_prob` calibration (promote normalized prob
   and/or refit v3 calibrator; ECE target <~0.05 OOS, AUC preserved; fix stale
   comment + UI label + tests).
2. **calib-fl-02** — quantify F-L bias on the full backtest OOS frame; design & fit
   an odds-band recalibrator for the price-free model; save artifact + analysis
   (offline only, no wiring).
3. **calib-fl-03** — integrate the F-L recalibrator into the predictor value layer
   (`value_win_prob`), backward-compatible, with tests.
4. **calib-fl-04** — re-validate via `evaluate_filter` / `backtest`; re-tune value
   gates on the corrected probs, selected on CLV + A/E.
5. **calib-fl-05** — update `config.yaml`, re-run `scripts/last_week_backtest.py`,
   before/after report, update README/known-limits + final verdict.

Relates to [[model-16-value-detection]], [[model-14-backtest]],
[[model-09-baseline-audit]], [[wrap-29-yesterday-predictor]].
