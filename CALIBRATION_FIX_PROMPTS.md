# Calibration + Favorite-Longshot Fix — 5 Prompts for Opus

**How to use this file.** Run the prompts **in order, one per fresh chat session**.
After each prompt finishes, it writes a `memory/calib-fl-0N-*.md` file and a line in
`memory/MEMORY.md` — so you can **clear the chat (`/clear`) between every prompt** to
keep token usage low without losing context. The next session re-bootstraps itself by
reading those memory files. Do **not** run two prompts in the same chat. If a prompt
says it's blocked or a check fails, fix it in that same chat before moving on.

**One-time context (already saved):** the diagnosis and master plan are in
`memory/calib-fl-00-plan.md` (indexed in `memory/MEMORY.md`). Every prompt below
starts by telling Opus to read those, so you don't have to paste any background.

**Environment rule (important):** this machine has two Pythons. Always use `python`
(= Python 3.14 / numpy 2.4, the project-official env). Do **not** use the bare
`streamlit` / `python310`. Run the app with `python -m streamlit run ui/app.py`.

**Verification tools you already have:**

- `python -m scripts.last_week_backtest` — settles value/top-pick × win/each-way over
  the last 7 settled days and prints an OOS win-prob calibration table (AUC / ECE /
  A/E by probability band). This is the before/after yardstick.
- `python -m pytest -q` — full suite (currently ~1056 pass / 3 skip). Keep it green.

---

## Prompt 1 — Fix the headline `won_prob` calibration bug

```
Read memory/MEMORY.md, then memory/calib-fl-00-plan.md and memory/model-11-calibration.md,
before doing anything. Use the `python` interpreter (3.14/numpy-2.4) for all runs.

PROBLEM: the headline win probability `won_prob` shown in the UI is badly
overconfident on out-of-sample data — over 2026-06-06..06-12 it averages 0.331 when
the true win rate is 0.109 (ECE 0.222). Its v3 isotonic calibrator
(models/catboost_won_v3_calib.pkl) only has x-knots over ~[0.252, 0.509] and
saturates everything else to 0.007 / 0.734. Meanwhile `won_prob_normalized`
(within-race sum-to-1) is AUC 0.778 / ECE 0.029 and `value_win_prob` is well
calibrated too. The code comment in models/predictor.py (~L408-414) claiming
won_prob is "marginally excellent (ECE≈0.005), keep as headline" is now false OOS.

TASK:
1. Reproduce: run `python -m scripts.last_week_backtest` and confirm the won_prob vs
   won_prob_normalized vs value_win_prob calibration table. Also print the raw v3
   model output range on recent data vs the calibrator's x-knots to confirm the
   saturation diagnosis.
2. Choose and implement the fix. Evaluate both options on the OOS week and pick the
   one with ECE < ~0.05 AND AUC not worse than current won_prob:
     (a) Promote `won_prob_normalized` to be the headline win probability the UI/API
         present (keep the raw per-runner calibrated value available under a clearly
         named column for debugging), OR
     (b) Refit the v3 `won` calibrator on a larger / more representative time-ordered
         slice so it no longer saturates (see models/train.py L232-288,
         calibration_size), OR a blend.
   Prefer the simplest change that hits the targets; (a) is the likely winner — state
   why in the memory note. Do NOT retrain the CatBoost model unless strictly required;
   recalibration only if possible.
3. Update the now-false comment/docstring in models/predictor.py and the
   RunnerPrediction docstring. If any UI page (ui/*.py, ui/pages/*) displays
   won_prob as the headline win %, point it at the corrected field.
4. Add/adjust tests (tests/models/test_predictor.py, tests/models/test_calibration.py)
   to lock in: headline win prob field-sum per race ≈ field winners expectation, and a
   regression guard that the headline isn't the saturating raw value.
5. Verify: `python -m pytest -q` green; re-run scripts/last_week_backtest and record
   the new headline ECE/AUC.

FINISH: write memory/calib-fl-01-headline.md (type: project) documenting the chosen
fix, before/after ECE+AUC numbers, files touched, and anything deferred. Add a
one-line pointer to memory/MEMORY.md. Do not commit unless I ask.
```

---

## Prompt 2 — Quantify F-L bias & fit an odds-band recalibrator (offline)

```
Read memory/MEMORY.md, then memory/calib-fl-00-plan.md, memory/calib-fl-01-headline.md,
memory/model-14-backtest.md and memory/model-16-value-detection.md, before starting.
Use `python` (3.14/numpy-2.4).

PROBLEM: the price-free value model (value_win_prob / v3nf) has a favorite-longshot
bias — A/E ≈ 2.80 odds-on and 1.88 at [2,4] (favourites win more than predicted) but
0.74 at [8,16], 0.44 at [16,34], 0.17 at >34 (longshots hugely over-predicted). This
drives CLV negative (−9..−12%) so the model is calibrated overall yet not bettable.

TASK (offline analysis + artifact only — do NOT wire anything into the live predictor
this prompt):
1. Get a leak-free OOS scored frame: either reuse the saved backtest run under
   data/backtests/ (load its scored.parquet) or regenerate with
   `python -m backtest --task-type GPU`. Confirm it carries prob / bet_price / won /
   close_price / race_date.
2. Quantify the F-L bias precisely: A/E by odds band and by probability band, plus a
   reliability curve, using backtest.metrics.ae_table / _ODDS_BANDS. Save a small
   report (docs/calibration/ or data/backtests/<run>/) with a plot or table.
3. Design a recalibration that corrects the bias while preserving ranking (must stay
   monotone within band). Candidate approaches — evaluate at least two and pick by
   held-out Brier/ECE AND A/E flattening toward 1.0 across bands:
     - isotonic/logistic recalibration computed PER odds band (with smooth blending
       at band edges), or
     - a 2-feature calibration (raw prob + log odds) via logistic/spline, or
     - a beta-calibration variant.
   Fit on a time-ordered train slice, validate on a held-out tail — NO leakage. Reuse
   models/calibration.py patterns; the artifact MUST pickle as plain Python floats
   (numpy-version-portable), like IsotonicCalibrator/SigmoidCalibrator.
4. Implement the chosen recalibrator as a new portable class in models/calibration.py
   (e.g. `OddsBandCalibrator` with .predict(prob, odds) -> prob), with unit tests
   (monotonicity within band, portability/no-numpy-in-pickle, A/E improvement on a
   synthetic biased fixture). Fit it and save the artifact to models/ with a clear
   name (do not overwrite existing calibrators). Record its measured A/E-by-band
   before vs after on the OOS frame.

FINISH: write memory/calib-fl-02-fl-design.md (type: project) with the bias table,
chosen method + why, the artifact path, before/after A/E by band, and the exact
validation command. Add a MEMORY.md pointer. Do not wire into predictor yet. Do not
commit unless I ask.
```

---

## Prompt 3 — Integrate the F-L recalibrator into the value layer

```
Read memory/MEMORY.md, then memory/calib-fl-00-plan.md, memory/calib-fl-02-fl-design.md,
and memory/model-17-inference.md, before starting. Use `python` (3.14/numpy-2.4).

GOAL: apply the odds-band/F-L recalibrator (built in calib-fl-02) to the live
price-free win probability so `value_win_prob` (and everything derived from it —
value_edge, expected_value, value_bet, models/value.find_value_bets) reflects the
corrected probabilities.

TASK:
1. Load the new recalibrator in models/predictor.py alongside _value_calibrator
   (follow the _load_calibrator / _load_value_model pattern; absent artifact must
   degrade gracefully to current behaviour, logged — never crash).
2. Apply it in _score_value: after the price-free model + existing value calibrator
   produce a probability, pass (prob, effective_decimal_odds) through the
   recalibrator. Be careful about which price to feed (use the same effective decimal
   the EV uses) and about NaN/odds-missing rows. Keep value_supported / gating intact.
3. Make it config-toggleable under config.yaml `value:` (e.g.
   `fl_recalibration: true`) read via ValueConfig / _load_cfg, default on.
4. Tests: extend tests/models/test_predictor.py + tests/models/test_value.py to cover
   the recalibrated path (recalibrated value_win_prob differs as expected on
   favourite vs longshot rows; layer disabled when artifact/flag absent).
5. Verify: `python -m pytest -q` green. Sanity-run `python -m scripts.last_week_backtest`
   and note the new value_win_prob A/E by odds band (full re-validation is the next
   prompt).

FINISH: write memory/calib-fl-03-integrate.md (type: project) with the wiring points,
config flag, test coverage, and any edge cases handled. Add a MEMORY.md pointer.
Do not commit unless I ask.
```

---

## Prompt 4 — Re-validate on the backtester & re-tune the value gates

```
Read memory/MEMORY.md, then memory/calib-fl-00-plan.md, memory/calib-fl-03-integrate.md,
memory/model-16-value-detection.md and memory/model-14-backtest.md, before starting.
Use `python` (3.14/numpy-2.4).

GOAL: confirm the F-L recalibration actually lifts CLV / flattens A/E, and re-tune the
value gates on the corrected probabilities. Judge on CLV + A/E first, yield second —
NOT raw ROI (per the audits).

TASK:
1. Produce a leak-free OOS scored frame whose `prob` is the RECALIBRATED price-free
   probability (regenerate via `python -m backtest --task-type GPU` if the model/
   calibration path now differs, or apply the recalibrator to the saved scored.parquet
   consistently with how live scoring does it — state which and why it's leak-free).
2. Run models.value.evaluate_filter across a grid of gates (min_ev, min_odds,
   max_odds, min_confidence; edge gates were HARMFUL before — re-confirm). Report for
   each: n_bets, yield, CLV mean, beat_close_rate, A/E by odds band, flat + ¼-Kelly
   bankroll. Compare against the pre-recalibration baseline in
   memory/model-16-value-detection.md (band [2,6]+EV≥0.05: yield +5.3%, CLV −7%).
3. Pick the gate set that maximises CLV while keeping A/E near 1.0 and a usable
   sample size. If CLV is still negative everywhere, say so plainly — that means not
   yet bettable; recommend next steps rather than forcing a profitable-looking config.
4. Do NOT change config.yaml here (that's prompt 5) — just produce the recommended
   numbers and the evidence table.

FINISH: write memory/calib-fl-04-backtest.md (type: project) with the full
before/after metrics table, the recommended value: gate set, and an honest verdict on
whether CLV now beats the close. Add a MEMORY.md pointer. Do not commit unless I ask.
```

---

## Prompt 5 — Apply config, re-run the last-week test, final report

```
Read memory/MEMORY.md, then memory/calib-fl-00-plan.md, memory/calib-fl-01-headline.md,
memory/calib-fl-02-fl-design.md, memory/calib-fl-03-integrate.md and
memory/calib-fl-04-backtest.md, before starting. Use `python` (3.14/numpy-2.4).

GOAL: land the recommended settings and produce a clean before/after verdict for the
whole programme (headline calibration + F-L recalibration).

TASK:
1. Apply the recommended value: gate set from calib-fl-04 to config.yaml (and update
   the stake/Kelly/odds-band tooltips if the bands changed). Keep config.local.yaml
   untouched and never print secrets.
2. Regenerate live predictions so the UI reflects the fixes:
   `python -m scripts.refresh --no-scrape` (or the rebuild entry point), then confirm
   data/predictions.json win % are sane (headline win probs per race ≈ sum to ~1, no
   0.73 saturation).
3. Re-run `python -m scripts.last_week_backtest` and build a before/after table vs the
   numbers in memory/calib-fl-00-plan.md (ROI per strategy; headline ECE/AUC; value
   A/E by band; CLV from calib-fl-04). Be honest: if it still loses at SP, state that
   SP-settlement is conservative and that CLV is the real test.
4. Update README.md / the known-limits section and PROGRESS.md to reflect the corrected
   calibration and the current honest verdict on profitability. Run `python -m pytest -q`
   one last time (green).

FINISH: write memory/calib-fl-05-final.md (type: project) with the final before/after
table, config changes, files touched, and the bottom-line verdict ("ranks well /
headline now calibrated / bettable: yes-no"). Add a MEMORY.md pointer. Then tell me
the one-paragraph summary and ask whether to commit the whole programme to the
chore/config-audit branch.
```

---

### After all 5

You'll have, in `memory/`: `calib-fl-00-plan` (this diagnosis) through
`calib-fl-05-final` (the result). Ask me to review `calib-fl-05-final.md` and then to
commit. If at any point a prompt's verification fails, don't proceed to the next —
paste the failure back into that same chat and fix it first.
