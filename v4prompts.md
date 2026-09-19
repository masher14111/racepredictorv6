# v4prompts.md — Porting racing_ingestion's strengths into Race Predictor v4

**Goal:** take the things `racing_ingestion` does _better_ than this project and rebuild them here, train a LightGBM win-probability model line ported from **racing_ingestion v1** plus its **planned v3 retrain**, surface everything in the built-in **Streamlit UI** (`ui/`), and use the **full PC** (12 logical cores + GPU CatBoost).

**Two project paths you will see repeated below:**

- This project (target): `C:\Users\mshr\Desktop\Race Predictor v4`
- Source to port from: `C:\Users\mshr\Documents\racing_ingestion`

---

## Why we are doing this (the honest framing — read once)

A same-window 3-week out-of-sample test (2026-05-22 → 2026-06-12) showed:

|                                                   | racing_ingestion v1  | racing_ingestion v2 | **Race Predictor v4**  |
| ------------------------------------------------- | -------------------- | ------------------- | ---------------------- |
| Win-prob log-loss **vs its own de-vigged market** | beaten by **0.0029** | 0.0042              | **0.188** (≈65× worse) |
| EV-bet ROI                                        | +2.3%                | −6.2%               | **−17.8%**             |
| Mean CLV / beat-close                             | −0.07 / 44%          | −0.08 / 45%         | **−0.16 / 24%**        |

The point of this rebuild is **not** "switch models and start winning." Neither system beats the market yet. What racing*ingestion does \_better* is **measure honestly**: its headline metric is "do we beat the de-vigged market line," it tracks CLV, runs integrity checks (look-ahead, cherry-picking, minimum-backtest-length), and shows a NO-GO banner when there is no proven edge. v4's own README already admits the matching flaws (look-ahead leak from `odds_finish` in the priced model, negative CLV in every gate, OOS calibration saturation). So we port the **evaluation spine first**, then the **model line**, then **wire it into the UI** — so this project finally tells you the truth and gives you a second, market-relative model to compare.

**Expectation to keep:** when prompt 7 finishes, the most likely result is the v4 LightGBM v3 line is _close to the market but does not beat it_ (like racing_ingestion). That is success — an honest, calibrated baseline — not failure.

---

## How to use these prompts

- **One prompt per fresh Opus chat.** Each prompt is self-contained (it restates paths, files, and constraints) so you can `/clear` between them to save tokens.
- **The paragraph above each prompt is the "memory"** — paste it into the chat first (or keep it as the project memory note for that step) so the model knows where the step sits in the sequence and what must not break.
- **Effort tier** is the recommended Opus reasoning level (`low` / `medium` / `high` / `xhigh`). Set it before sending.
- **Run them roughly in order** — phases build on each other. Within a phase, dependencies are noted.
- **Every prompt ends with an acceptance gate** (tests / a command to run). Don't move on until it's green.
- A **DeepSeek v4 Pro** section at the end lists the prompts you can offload to DeepSeek (mechanical / test-heavy work) to save Opus budget.

**Global guardrails baked into every prompt:** Python 3.14, Windows/PowerShell. The decision/EV price is the **pre-off** price (`morningwap` or `ppwap`) — **never `odds_finish`** (the settling price = look-ahead leak). Keep `pytest -q` green (baseline 1082 passed, 3 skipped). Add `lightgbm` to `requirements.txt` and verify the wheel imports on 3.14 before using it — if there's no 3.14 wheel, fall back to a dedicated `.venv-lgbm` (Python 3.10) used only for the LightGBM line, driven via subprocess (see Prompt 5, Step 0).

---

# PHASE A — Port the evaluation spine (the part racing_ingestion does best)

---

### Prompt 1 — De-vig market baselines · effort: `medium`

**Memory / context:** First step of the port. racing_ingestion's whole edge is that it compares the model to the **de-vigged market line**, not to race outcomes. We need that de-vig math in v4 before anything else can be measured. This is a self-contained, pure-function module with no dependencies on the rest of the rebuild. Source of truth is `racing_ingestion/ml/model/baselines.py` (`DevigMarketBaseline`, methods `proportional` / `power` / `shin`). Everything in Phase A and the UI later consumes this.

```text
First step of the port. racing_ingestion's whole edge is that it compares the model to the **de-vigged market line**, not to race outcomes. We need that de-vig math in v4 before anything else can be measured. This is a self-contained, pure-function module with no dependencies on the rest of the rebuild. Source of truth is `racing_ingestion/ml/model/baselines.py` (`DevigMarketBaseline`, methods `proportional` / `power` / `shin`). Everything in Phase A and the UI later consumes this.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, Windows/PowerShell)
Source to port from: C:\Users\mshr\Documents\racing_ingestion\ml\model\baselines.py

TASK: Create models/devig.py — a market de-vigging module ported and adapted from the
DevigMarketBaseline class in the source file above. Implement three margin-removal methods
that turn per-runner decimal odds into a market-implied probability vector that sums to 1.0
within each race:
  - "proportional" : p_i ∝ 1/o_i, normalised by sum.
  - "power"        : p_i ∝ (1/o_i)^k, exponent k solved numerically so Σp = 1.
  - "shin"         : Shin (1992/93) insider-fraction fixed point; fall back to "power"
                     when the fixed point is ill-defined.
Public API:
  devig(odds: pd.Series, race_ids: pd.Series, method: str = "proportional") -> np.ndarray
returning probabilities aligned to the input rows. It must handle NaN odds (a race with any
missing price is NOT a complete book — exclude or flag it, never de-vig a partial book),
single-runner races, and odds <= 1.0 defensively.

CONSTRAINTS: pure functions, no I/O, no look-ahead. Decimal odds in, probabilities out.
Match the source's numeric behaviour (compare against it on a few hand cases in the test).

ACCEPTANCE: add tests/test_devig.py with cases for all three methods (sum-to-1 per race,
power/shin remove more longshot margin than proportional, NaN handling, single-runner).
Run: python -m pytest tests/test_devig.py -q  → all green. Then run full: python -m pytest -q.
```

---

### Prompt 2 — Market-relative head-to-head evaluator + GO gate · effort: `high`

**Memory / context:** This is the metric that exposed the truth in the 3-week test. It scores the model's win probabilities against the de-vigged market over the SAME races, on log-loss / Brier / ECE, and returns a single GO/NO-GO boolean (`model beats market on log-loss`). Depends on Prompt 1 (`models/devig.py`). Sources: `racing_ingestion/ml/evaluation/metrics.py` (`evaluate_all`), `.../calibration.py` (`expected_calibration_error`), `.../odds_bands.py` (`model_vs_market_by_odds_band`), and the `head_to_head_logloss` function in `racing_ingestion/backtest/phase4_holdout_backtest.py`. Everything downstream (holdout runner, UI banner, v3 verdict) reads this.

```text

This is the metric that exposed the truth in the 3-week test. It scores the model's win probabilities against the de-vigged market over the SAME races, on log-loss / Brier / ECE, and returns a single GO/NO-GO boolean (`model beats market on log-loss`). Depends on Prompt 1 (`models/devig.py`). Sources: `racing_ingestion/ml/evaluation/metrics.py` (`evaluate_all`), `.../calibration.py` (`expected_calibration_error`), `.../odds_bands.py` (`model_vs_market_by_odds_band`), and the `head_to_head_logloss` function in `racing_ingestion/backtest/phase4_holdout_backtest.py`. Everything downstream (holdout runner, UI banner, v3 verdict) reads this.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14)
Depends on: models/devig.py (already built).
Sources to port from:
  C:\Users\mshr\Documents\racing_ingestion\ml\evaluation\metrics.py        (evaluate_all)
  C:\Users\mshr\Documents\racing_ingestion\ml\evaluation\calibration.py    (expected_calibration_error)
  C:\Users\mshr\Documents\racing_ingestion\ml\evaluation\odds_bands.py     (model_vs_market_by_odds_band)
  C:\Users\mshr\Documents\racing_ingestion\backtest\phase4_holdout_backtest.py  (head_to_head_logloss)

TASK: Create models/head_to_head.py exposing:
  head_to_head(df, prob_col, odds_col, race_id_col, devig_method="proportional") -> dict
where df has one row per runner with the win label (0/1), the model win prob, the pre-off
decimal odds, and a race id. It must:
  1. Restrict to COMPLETE-ODDS races only (every runner priced) — de-vig needs a full book.
  2. Compute, for BOTH the model probs and the de-vigged market probs: race-level log-loss
     (-log of the winner's normalised prob, averaged over races), runner-level Brier, and ECE.
  3. Return per-odds-band calibration (actual vs model vs market) like the source's odds_bands.
  4. Return model_beats_market_logloss: bool and the numeric gaps.
The v4 model probs must be normalised within race (sum-to-1) before scoring — reuse
models/calibration.normalize_within_race if present.

CONSTRAINTS: odds_col must be a PRE-OFF price (morningwap/ppwap), NEVER odds_finish. No look-ahead.

ACCEPTANCE: tests/test_head_to_head.py — a synthetic race set where the model == de-vigged
market gives model_beats_market = False with ~equal log-loss; a deliberately better model
gives True. Run python -m pytest tests/test_head_to_head.py -q, then python -m pytest -q.
```

---

### Prompt 3 — Backtest integrity suite · effort: `high`

**Memory / context:** racing_ingestion refuses to trust a backtest until it passes integrity checks. We port that suite so v4's ROI numbers can no longer flatter themselves. Independent of the model work — only needs a bet ledger / scored card DataFrame. Sources: `racing_ingestion/backtest/integrity.py` (`run_all_integrity_checks`) and `racing_ingestion/backtest/overfitting.py` (`minimum_backtest_length`). Consumed by the holdout runner (Prompt 4) and the UI (Prompt 9).

```text
racing_ingestion refuses to trust a backtest until it passes integrity checks. We port that suite so v4's ROI numbers can no longer flatter themselves. Independent of the model work — only needs a bet ledger / scored card DataFrame. Sources: `racing_ingestion/backtest/integrity.py` (`run_all_integrity_checks`) and `racing_ingestion/backtest/overfitting.py` (`minimum_backtest_length`). Consumed by the holdout runner (Prompt 4) and the UI (Prompt 9).

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14)
Sources to port from:
  C:\Users\mshr\Documents\racing_ingestion\backtest\integrity.py     (run_all_integrity_checks)
  C:\Users\mshr\Documents\racing_ingestion\backtest\overfitting.py   (minimum_backtest_length / deflated Sharpe)

TASK: Create backtest/integrity.py in this project, porting the integrity checks and adapting
them to v4's bet-ledger schema (a DataFrame with at least: race_date, won, stake, profit,
bet_price/pre-off odds, close_price, plus an odds-band column). Implement:
  - lookahead bias guard (first bet date must be strictly after the model's train cutoff),
  - cherry-picking checks (course / season / odds-band: is profit concentrated in one slice?),
  - CLV-leakage check (mean log CLV and beat-close rate),
  - non-runner impact, liquidity-unknown warning,
  - minimum-backtest-length / deflated-Sharpe credibility from overfitting.py.
Return a list of structured results: {name, status: OK|WARN|FAIL, message}.

CONSTRAINTS: read-only over the ledger; no network; no look-ahead. Where v4 lacks a column
(e.g. matched volume), emit a WARN ("liquidity unknown"), don't crash.

ACCEPTANCE: tests/test_integrity.py with a clean ledger (all OK), a leaky ledger (lookahead
FAIL), and a cherry-picked ledger (cherry-pick WARN/FAIL). python -m pytest tests/test_integrity.py -q
then python -m pytest -q.
```

---

### Prompt 4 — Frozen-model walk-forward holdout runner · effort: `medium`

**Memory / context:** This is v4's version of racing_ingestion's `phase4_holdout_backtest.py` — the honest GO/NO-GO evaluation. It loads a FROZEN model trained strictly before a holdout window, scores the window, runs the head-to-head (Prompt 2), simulates EV>0 bets settled at the pre-off price with CLV vs the closing line, and runs the integrity suite (Prompt 3). Depends on Prompts 2 and 3. This is the harness the v3 retrain (Prompt 7) reports through.

```text

**Memory / context:** This is v4's version of racing_ingestion's `phase4_holdout_backtest.py` — the honest GO/NO-GO evaluation. It loads a FROZEN model trained strictly before a holdout window, scores the window, runs the head-to-head (Prompt 2), simulates EV>0 bets settled at the pre-off price with CLV vs the closing line, and runs the integrity suite (Prompt 3). Depends on Prompts 2 and 3. This is the harness the v3 retrain (Prompt 7) reports through.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14)
Depends on: models/head_to_head.py, backtest/integrity.py.
Reference design: C:\Users\mshr\Documents\racing_ingestion\backtest\phase4_holdout_backtest.py
v4 data: data/features.parquet (has race_uid, race_date, morningwap, ppwap, odds_finish, won).

TASK: Create backtest/holdout.py with a CLI:
  python -m backtest.holdout --model <path> --holdout-start YYYY-MM-DD --holdout-end YYYY-MM-DD \
      --price-col ppwap --close-col odds_finish --out data/backtests/holdout_<stamp>
It must:
  1. Load the model's train cutoff from its metadata and HARD-FAIL if the holdout starts on or
     before the cutoff (leakage guard).
  2. Score every runner in the window; normalise win probs within race.
  3. Call head_to_head(...) and print model-vs-market log-loss/Brier/ECE + GO/NO-GO.
  4. Simulate EV>0 flat bets: EV = norm_prob * price - 1; bet when > 0; £10 flat; 2% commission
     on winnings; settle at --price-col (pre-off); compute CLV = log(price/close).
  5. Run backtest/integrity.run_all_integrity_checks over the ledger and print results.
  6. Write summary.json (window, n_races, head-to-head dict, ROI, strike, CLV, integrity) and ledger.csv.

CONSTRAINTS: decision/settlement price = ppwap or morningwap; odds_finish is CLOSE-only (CLV ref),
never a feature, never the decision price.

ACCEPTANCE: runs end-to-end on the existing CatBoost v3nf model over a recent 3-week window and
writes a summary.json containing model_beats_market_logloss. python -m pytest -q stays green.
```

---

# PHASE B — Port racing_ingestion v1 + do the v3 retrain

---

### Prompt 5 — LightGBM softmax-by-race win-prob model line (port of racing_ingestion v1) · effort: `high`

**Memory / context:** racing_ingestion v1 was the best performer in the 3-week test. Its model is a LightGBM **ranker with softmax-by-race normalisation** — it outputs one win probability per runner that sums to 1 within a race, trained with market features. We add this as a SECOND model line in v4 (alongside CatBoost), not a replacement. Sources: `racing_ingestion/ml/model/lgbm_model.py`, `.../softmax.py` (`SoftmaxWrapper`, `softmax_by_race`), `.../objective.py`. Uses all 12 CPU cores (`num_threads=-1`). Depends on nothing in Phase A but is consumed by Prompts 6–8.

```text

**Memory / context:** racing_ingestion v1 was the best performer in the 3-week test. Its model is a LightGBM **ranker with softmax-by-race normalisation** — it outputs one win probability per runner that sums to 1 within a race, trained with market features. We add this as a SECOND model line in v4 (alongside CatBoost), not a replacement. Sources: `racing_ingestion/ml/model/lgbm_model.py`, `.../softmax.py` (`SoftmaxWrapper`, `softmax_by_race`), `.../objective.py`. Uses all 12 CPU cores (`num_threads=-1`). Depends on nothing in Phase A but is consumed by Prompts 6–8.


Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, Windows/PowerShell, 12 cores).
Sources to port from:
  C:\Users\mshr\Documents\racing_ingestion\ml\model\lgbm_model.py
  C:\Users\mshr\Documents\racing_ingestion\ml\model\softmax.py     (SoftmaxWrapper, softmax_by_race)
  C:\Users\mshr\Documents\racing_ingestion\ml\model\objective.py

STEP 0 — ENVIRONMENT (do this FIRST, before writing any model code):
  1. Try the main interpreter: add `lightgbm` to requirements.txt and run
     `python -c "import lightgbm, sys; print(lightgbm.__version__, sys.version)"`.
  2. IF that imports cleanly on 3.14 → use it; the LightGBM line runs in the main env. Done.
  3. IF the wheel is unavailable / fails to build on 3.14 (no cp314 wheel yet) → DO NOT pin the
     whole project back to 3.10 and DO NOT try to force-build. Instead create a DEDICATED 3.10
     venv used ONLY for the LightGBM line (racing_ingestion already runs LightGBM 4.6 on 3.10):
        py -3.10 -m venv .venv-lgbm
        .\.venv-lgbm\Scripts\python -m pip install lightgbm pandas numpy pyarrow scikit-learn
     Then make models/lgbm_softmax.py importable AND runnable as a subprocess by that interpreter:
       - Keep the module pure (pandas/numpy/lightgbm only — no v4-3.14-only deps) so .venv-lgbm
         can run it directly.
       - Add a small helper utils/lgbm_env.py exposing LGBM_PYTHON (the .venv-lgbm python path,
         configurable via config.yaml `lgbm.python_path`, default ".venv-lgbm/Scripts/python.exe",
         falling back to "python" when step 2 succeeded).
       - train_lgbm.py and the predictor (Prompts 7/8) shell out via LGBM_PYTHON for training and
         batch scoring (subprocess + parquet/json hand-off), so the 3.14 main app never imports
         lightgbm directly. Inference for the UI reads the model's batch-scored parquet/json output.
     Record which path you took (3.14-native vs .venv-lgbm) in the model metadata json and print it.
  4. Add .venv-lgbm/ to .gitignore if you created it.

TASK: Create models/lgbm_softmax.py — a LightGBM win-probability model line ported from
racing_ingestion v1. It must:
  - Train a LightGBM model whose raw scores are converted to win probabilities via
    softmax_by_race (probabilities sum to 1.0 within each race).
  - Expose a save/load bundle (model + feature_name list + metadata json: train window,
    feature list, n_rows, n_races) mirroring how v4's CatBoost meta json works.
  - predict_proba(X, race_ids) -> np.ndarray of within-race win probs.
  - Use num_threads=-1 (all 12 cores). No GPU needed for LightGBM here.
  - Be robust to a feature-count change (store the trained feature_name list and select exactly
    those columns, in order, at predict time — racing_ingestion's v1 broke when the pipeline grew
    from 43→45 features; do NOT repeat that, select by stored name).

CONSTRAINTS: market features must come from a PRE-OFF price (morningwap/ppwap), never odds_finish.

ACCEPTANCE: tests/test_lgbm_softmax.py trains on a tiny synthetic frame, checks probs sum to 1
per race and that save→load→predict reproduces probabilities. python -m pytest tests/test_lgbm_softmax.py -q
then python -m pytest -q.
```

---

### Prompt 6 — Feature adapter + look-ahead audit (v4 features → LightGBM schema) · effort: `high`

**Memory / context:** The LightGBM line needs v4's features mapped into a clean, leak-free matrix. v4's feature columns live in `models/features.py` (`FEATURE_COLS`, `PRICE_FREE_FEATURE_COLS`) and `data/features.parquet`. This step builds the adapter AND audits for look-ahead — the single most important correctness step, because v4's README admits the priced model leaks `odds_finish`. Depends on Prompt 5. Feeds Prompt 7 (training).

```text

**Memory / context:** The LightGBM line needs v4's features mapped into a clean, leak-free matrix. v4's feature columns live in `models/features.py` (`FEATURE_COLS`, `PRICE_FREE_FEATURE_COLS`) and `data/features.parquet`. This step builds the adapter AND audits for look-ahead — the single most important correctness step, because v4's README admits the priced model leaks `odds_finish`. Depends on Prompt 5. Feeds Prompt 7 (training).

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14)
Key files: models/features.py (FEATURE_COLS, PRICE_FREE_FEATURE_COLS, EMPIRICALLY_DEAD_COLS),
features/builder.py (build_training_matrix / build_inference_matrix), data/features.parquet.
Reference (how racing_ingestion picks leak-free market features):
  C:\Users\mshr\Documents\racing_ingestion\ml\features\market.py

TASK: Create features/lgbm_adapter.py with:
  build_lgbm_matrix(df, *, inference: bool) -> (X, y, race_ids)
that selects v4 features for the LightGBM line. Market features must derive ONLY from a pre-off
price (morningwap → ppwap fallback), and the adapter must EXCLUDE any column derived from
odds_finish / SP / finishing position. Also write tools/audit_lgbm_features.py that prints, per
feature: null %, and a hard assertion that no selected feature correlates with the outcome via a
post-off price (flag any feature whose values are only known after the off).

CONSTRAINTS: chronological integrity — features must be point-in-time (lagged). If a v4 feature
is built with future data, exclude it and log why.

ACCEPTANCE: python tools/audit_lgbm_features.py prints a clean report with ZERO post-off features
selected and reasonable null rates. tests/test_lgbm_adapter.py asserts odds_finish-derived columns
are absent from the matrix. python -m pytest -q green.
```

---

### Prompt 7 — Train v3: fresh-window LightGBM retrain with GO/NO-GO verdict · effort: `xhigh`

**Memory / context:** THE headline step — "racing_ingestion v1 + its planned v3 retrain, on this project." racing_ingestion's v3 was blocked because its holdout was spent; v4 has its own data to a recent date, so we can cut a FRESH out-of-sample window. Train the LightGBM line (Prompt 5) on v4 data with strict chronological splits, then evaluate on the untouched window via the holdout runner (Prompt 4) and emit a GO/NO-GO verdict from the head-to-head (Prompt 2). Uses full PC. Depends on Prompts 2, 4, 5, 6. Be rigorous about leakage — this is where a mistake silently inflates results.

```text

**Memory / context:** THE headline step — "racing_ingestion v1 + its planned v3 retrain, on this project." racing_ingestion's v3 was blocked because its holdout was spent; v4 has its own data to a recent date, so we can cut a FRESH out-of-sample window. Train the LightGBM line (Prompt 5) on v4 data with strict chronological splits, then evaluate on the untouched window via the holdout runner (Prompt 4) and emit a GO/NO-GO verdict from the head-to-head (Prompt 2). Uses full PC. Depends on Prompts 2, 4, 5, 6. Be rigorous about leakage — this is where a mistake silently inflates results.


Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, 12 cores, GPU available for CatBoost)
Depends on: models/lgbm_softmax.py, features/lgbm_adapter.py, models/head_to_head.py, backtest/holdout.py.
Reference: C:\Users\mshr\Documents\racing_ingestion\ml\train.py (chronological splits, holdout guard).
Data: data/features.parquet (race_date runs to ~2026-06-12). Determine the true max date at runtime.
ENV NOTE: Prompt 5 may have put LightGBM in a dedicated 3.10 venv (.venv-lgbm) if no 3.14 wheel
exists. Check models/lgbm_softmax.py / utils/lgbm_env.py: if training must run under that interpreter,
launch it via LGBM_PYTHON (subprocess + parquet/json hand-off), not by importing lightgbm in 3.14.

TASK: Create models/train_lgbm.py with a CLI:
  python -m models.train_lgbm --max-date <D-21d> --split-date <auto> --holdout-start <D-20d> \
      --model-version v3-lgbm-<stamp> --output models/lgbm_won_v3.txt --importance
It must:
  1. Build the matrix via features/lgbm_adapter (leak-free, pre-off market features).
  2. Split CHRONOLOGICALLY: train < split-date < validation < holdout-start; never random.
  3. Train LightGBM with softmax-by-race; early-stop on the validation slice; num_threads=-1.
  4. Save model + metadata (train window, holdout_start, feature list, n rows/races).
  5. Reserve the most recent ~3 weeks as an UNTOUCHED holdout — do NOT load it in training.
  6. Immediately call backtest.holdout on that window and print the GO/NO-GO verdict
     (model-vs-market log-loss/Brier/ECE, EV-bet ROI, CLV, integrity).
  7. Write models/lgbm_v3_meta.json including the verdict.

CONSTRAINTS: leakage is the enemy. Re-state in the run log: train max date, split date,
holdout start, and assert train_max < holdout_start. Market features pre-off only.

EXPECTED OUTCOME: most likely the model tracks the market but does NOT beat it (NO-GO) — that is
an honest, correct result, not a bug. Report it plainly; do not tune against the holdout to force GO.

ACCEPTANCE: training completes using all cores; a models/lgbm_v3_meta.json with a verdict is written;
the holdout summary.json exists. python -m pytest -q green.
```

---

### Prompt 8 — Unified prediction: CatBoost + LightGBM + market, with verdict in predictions.json · effort: `medium`

**Memory / context:** Now make both model lines usable together. v4's `models/predictor.py` writes `data/predictions.json` consumed by the UI. We extend it to also score the LightGBM v3 line, attach the de-vigged market prob per runner, and stamp each race/file with the latest GO/NO-GO verdict so the UI can show honesty. Depends on Prompts 1, 2, 5, 7. Pure wiring — no new modelling.

```text

**Memory / context:** Now make both model lines usable together. v4's `models/predictor.py` writes `data/predictions.json` consumed by the UI. We extend it to also score the LightGBM v3 line, attach the de-vigged market prob per runner, and stamp each race/file with the latest GO/NO-GO verdict so the UI can show honesty. Depends on Prompts 1, 2, 5, 7. Pure wiring — no new modelling.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14)
Key files: models/predictor.py (writes data/predictions.json), models/value.py, models/devig.py,
models/lgbm_softmax.py, models/head_to_head.py, models/lgbm_v3_meta.json.
ENV NOTE: if Prompt 5 put LightGBM in a dedicated 3.10 venv (.venv-lgbm), score the LightGBM line by
shelling out via LGBM_PYTHON (utils/lgbm_env.py) and reading its batch-scored parquet/json — the
3.14 app must not import lightgbm. If lgbm scoring is unavailable, omit the lgbm_* keys gracefully.

TASK: Extend models/predictor.py (or add models/predict_unified.py that it calls) so each upcoming
runner in data/predictions.json carries:
  - catboost_win_prob (existing v3nf normalised win prob),
  - lgbm_win_prob (from the LightGBM v3 line, within-race normalised),
  - market_prob (de-vigged from the best available pre-off board price via models/devig),
  - ev_catboost / ev_lgbm (prob*price - 1 at the executable board price).
Add a top-level "verdict" block to predictions.json: the latest holdout GO/NO-GO summary for each
line (read from the meta json + most recent data/backtests/holdout_* summary.json).

CONSTRAINTS: don't break the existing predictions.json keys the UI already reads (additive only).
Pre-off prices only. If the LightGBM model file is missing, degrade gracefully (omit lgbm_* keys).

ACCEPTANCE: python -m models.predictor runs and data/predictions.json validates (additive keys
present, old keys intact). Add tests/test_predictions_schema.py. python -m pytest -q green.
```

---

# PHASE C — Built-in UI + full-PC orchestration

---

### Prompt 9 — UI "Model Honesty" panel (GO/NO-GO + CLV + integrity + calibration) · effort: `high`

**Memory / context:** The single most valuable UI change: a panel that tells the truth. It shows, for each model line, whether it beats the de-vigged market (GO/NO-GO banner), the CLV / beat-close rate, the integrity check results, and the per-odds-band model-vs-market calibration table. This is what racing_ingestion's dashboard banner does and v4 lacks. Depends on Prompts 2, 3, 4, 8. UI lives in `ui/` (Streamlit, `streamlit run ui/app.py`).

```text


**Memory / context:** The single most valuable UI change: a panel that tells the truth. It shows, for each model line, whether it beats the de-vigged market (GO/NO-GO banner), the CLV / beat-close rate, the integrity check results, and the per-odds-band model-vs-market calibration table. This is what racing_ingestion's dashboard banner does and v4 lacks. Depends on Prompts 2, 3, 4, 8. UI lives in `ui/` (Streamlit, `streamlit run ui/app.py`).

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, Streamlit in ui/)
Data: data/predictions.json (now has verdict block), data/backtests/holdout_*/summary.json.
UI entry: ui/app.py; standalone pages pattern already exists (ui/performance.py etc.).

TASK: Create ui/model_honesty.py and surface it in the UI (a new section in ui/app.py or a new
page). It must render, reading from the latest holdout summary.json + predictions.json verdict:
  - A prominent GO / NO-GO banner per model line: green "Beats market (log-loss X vs Y)" or
    amber/red "Does NOT beat market (X vs Y) — paper-only".
  - CLV: mean CLV and beat-close rate, with a one-line plain-English explanation.
  - Integrity checks as a status list (OK / WARN / FAIL with messages).
  - The per-odds-band model-vs-market calibration table.
Design: match the existing Streamlit look; no real-money language; clear that this is paper-only.
Follow good contrast (body text >=4.5:1) — no light-gray-on-tint.

CONSTRAINTS: read-only display; if a file is missing show a graceful "no holdout yet — run
python -m backtest.holdout" message, never a stack trace.

ACCEPTANCE: streamlit run ui/app.py launches; the panel renders the verdict from a real
holdout summary.json without errors. python -m pytest -q green (add a smoke import test for the page).
```

---

### Prompt 10 — UI model-comparison page + wire holdout/integrity into performance pages · effort: `medium`

**Memory / context:** Complements Prompt 9: a page that puts CatBoost vs LightGBM vs market side by side (log-loss, ROI, CLV, calibration), and wires the holdout runner output into the existing performance pages so the bankroll/metrics views reflect the honest numbers. Depends on Prompts 4, 8, 9.

```text

**Memory / context:** Complements Prompt 9: a page that puts CatBoost vs LightGBM vs market side by side (log-loss, ROI, CLV, calibration), and wires the holdout runner output into the existing performance pages so the bankroll/metrics views reflect the honest numbers. Depends on Prompts 4, 8, 9.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, Streamlit in ui/)
Files: ui/performance.py / ui/performance_dashboard.py, ui/model_honesty.py, data/backtests/holdout_*/.

TASK: Create ui/model_compare.py — a head-to-head page showing CatBoost v3nf vs LightGBM v3 vs
de-vigged market over the latest holdout window: a table of log-loss / Brier / ECE / EV-ROI / CLV,
plus the per-odds-band calibration for both models on one chart. Add a control to pick which
holdout run (folder) to view. Then update the performance pages to display the latest GO/NO-GO
verdict at the top so the bankroll view is read in context.

CONSTRAINTS: additive UI; reuse existing chart/style helpers; graceful when only one model exists.

ACCEPTANCE: streamlit run ui/app.py → the compare page renders both lines from real holdout data.
python -m pytest -q green.
```

---

### Prompt 11 — Full-PC orchestration: one-shot fast rebuild · effort: `medium`

**Memory / context:** "Use my full PC." A single command that rebuilds everything as fast as the machine allows: parallel feature build, CatBoost on GPU and LightGBM on all 12 cores trained concurrently, walk-forward folds parallelised, Optuna with n_jobs. v4 already sets `task_type: GPU` and `thread_count: -1` in config.yaml; this ties it together. Depends on Prompts 5–8 existing.

```text

**Memory / context:** "Use my full PC." A single command that rebuilds everything as fast as the machine allows: parallel feature build, CatBoost on GPU and LightGBM on all 12 cores trained concurrently, walk-forward folds parallelised, Optuna with n_jobs. v4 already sets `task_type: GPU` and `thread_count: -1` in config.yaml; this ties it together. Depends on Prompts 5–8 existing.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, 12 logical cores, GPU for CatBoost)
Existing: pipeline.py, scripts/refresh, config.yaml (model.task_type=GPU, model.thread_count=-1),
models/train.py (CatBoost), models/train_lgbm.py (LightGBM).

TASK: Create pipeline_full.py — a one-shot orchestrator:
  python pipeline_full.py [--skip-scrape] [--no-tune]
that runs, with maximum safe parallelism on this machine:
  1. Feature build parallelised across cores (joblib / ProcessPool) where safe.
  2. Train CatBoost (GPU) and LightGBM v3 (num_threads=-1) CONCURRENTLY (they use different
     hardware — GPU vs CPU — so run them in parallel processes).
  3. Run the holdout GO/NO-GO for both lines, then models/predictor to refresh predictions.json.
  4. Print a final timing + verdict summary.
Add a config.yaml block (e.g. orchestration: { max_workers: 10, lgbm_threads: -1 }) and read it.
Leave 2 cores free by default (10 workers) so the UI stays responsive.

CONSTRAINTS: don't oversubscribe — GPU job + CPU job should not both grab all cores. Make it
idempotent and resumable; never delete existing models without --force.

ACCEPTANCE: python pipeline_full.py --skip-scrape --no-tune completes, trains both lines, writes
both verdicts and a fresh predictions.json, and prints wall-clock timings. python -m pytest -q green.
```

---

### Prompt 12 — End-to-end smoke test, docs, and final verification · effort: `low`

**Memory / context:** Final wiring pass. Confirm the whole chain runs, the UI shows the new panels, tests pass, and the docs/README/CLAUDE.md describe the new model line, the GO/NO-GO honesty panel, and the full-PC command. Low effort but do it carefully — it's the gate that says "the rebuild works." Depends on everything above.

```text

**Memory / context:** Final wiring pass. Confirm the whole chain runs, the UI shows the new panels, tests pass, and the docs/README/CLAUDE.md describe the new model line, the GO/NO-GO honesty panel, and the full-PC command. Low effort but do it carefully — it's the gate that says "the rebuild works." Depends on everything above.

Project: C:\Users\mshr\Desktop\Race Predictor v4  (Python 3.14, Streamlit in ui/)

TASK:
  1. Run the full chain on existing data: python pipeline_full.py --skip-scrape --no-tune, then
     streamlit run ui/app.py — confirm the Model Honesty panel and Model Compare page render with
     a real GO/NO-GO verdict and CLV.
  2. Add tests/test_smoke_end_to_end.py: imports the new modules, runs backtest.holdout on a tiny
     fixture, asserts a verdict dict is produced and predictions.json gets the additive keys.
  3. Update README.md and CLAUDE.md: document models/lgbm_softmax.py + train_lgbm.py (the
     LightGBM v3 line), models/devig.py + head_to_head.py + backtest/integrity.py + backtest/holdout.py,
     the UI honesty/compare pages, and the pipeline_full.py full-PC command.
  4. Run python -m pytest -q and ruff/lint if configured; fix anything red.

ACCEPTANCE: pytest green (baseline 1082 passed + the new tests), UI launches and shows the panels,
README/CLAUDE.md updated. Print a short "what changed / how to run it" summary at the end.
```

---

# DeepSeek v4 Pro — extra prompts (offload the mechanical work)

DeepSeek v4 Pro is strong and cheap on **well-specified, mechanical, test-heavy** work where the spec is unambiguous. Use it to save Opus budget on these; keep the architecture/leak-judgment prompts (2, 3, 7, 9) on Opus. Each is self-contained.

**DS-1 — Port the de-vig math (alternative executor for Prompt 1).**
The math (proportional / power / Shin) is fully specified, so DeepSeek can port it cleanly.

```text
Port the class DevigMarketBaseline from C:\Users\mshr\Documents\racing_ingestion\ml\model\baselines.py
into a new module models/devig.py in C:\Users\mshr\Desktop\Race Predictor v4 (Python 3.14). Expose
devig(odds, race_ids, method="proportional"|"power"|"shin") -> np.ndarray of within-race probabilities
summing to 1.0. Handle NaN/partial books, single-runner races, odds<=1. Reproduce the source's numeric
behaviour. Include a docstring per method explaining the margin-removal formula.
```

**DS-2 — Exhaustive unit tests for the ported modules.**
Test generation is mechanical once the API exists; great DeepSeek task after Prompts 1, 3, 5.

```text
Write thorough pytest suites for these v4 modules (Python 3.14): models/devig.py,
backtest/integrity.py, models/lgbm_softmax.py. Cover edge cases: empty/partial books, single-runner
races, NaN odds, leaky vs clean ledgers, softmax sum-to-1 per race, save/load round-trip. Use small
synthetic DataFrames (no network, no real DB). Target the public API only.
```

**DS-3 — Feature-name mapping table (supports Prompt 6).**
A mechanical column-to-column mapping between v4 features and the LightGBM schema.

```text
Given v4 feature columns in models/features.py (FEATURE_COLS, PRICE_FREE_FEATURE_COLS) and
data/features.parquet, and the racing_ingestion v1 feature list in
C:\Users\mshr\Documents\racing_ingestion\models\v1_realdata.metadata.json, produce a markdown table
mapping each racing_ingestion v1 feature to its v4 equivalent column (or "no equivalent"). Flag every
v4 column derived from odds_finish / SP / finishing position as POST-OFF (must be excluded). Output
only the table + a short list of post-off columns to exclude.
```

**DS-4 — Docstrings, type hints, and ruff cleanup pass.**
Pure polish across the new modules — ideal cheap DeepSeek finishing pass before Prompt 12.

```text
Add complete type hints and concise docstrings to these new v4 modules and fix any ruff lint:
models/devig.py, models/head_to_head.py, backtest/integrity.py, backtest/holdout.py,
models/lgbm_softmax.py, features/lgbm_adapter.py, models/train_lgbm.py. Do not change behaviour;
only annotations, docstrings, and lint fixes. Keep pytest green.
```

---

## Build order at a glance

```
Phase A (eval spine):   1 → 2 → 3 → 4
Phase B (model + v3):   5 → 6 → 7 → 8        (5 can start in parallel with A)
Phase C (UI + full PC):  9 → 10 → 11 → 12

DeepSeek offload: DS-1 ⇒ Prompt 1 · DS-2 after 1/3/5 · DS-3 ⇒ Prompt 6 · DS-4 before 12
```

**Reminder:** the win condition is an _honest, market-calibrated_ second model line and a UI that tells you the truth — not a forced positive ROI. If the v3 line comes back NO-GO, that is the system working.
