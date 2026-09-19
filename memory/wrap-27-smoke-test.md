---
name: wrap-27-smoke-test
description: End-to-end smoke test 2026-06-17 — full chain runs green; remaining issues are model-quality (null live features → win-prob quantization), not pipeline breakage
metadata:
  type: project
---

End-to-end smoke test of Race Predictor v3, run 2026-06-17. All five steps
exercised against live data; the whole chain runs. Screenshots in `.smoke-shots/`.

## Status by step — all PASS

1. **Scrape** — boylesports re-scraped 622 rows in ~11s (curl_cffi Chrome tier;
   proxy blacklisted then recovered, fell back to direct — no data loss). Live
   odds land in `data/live_odds.parquet`. (Other scrapers not re-run this pass;
   boylesports was the one under fix.)
2. **Features** — `features.builder` regenerated `data/features.parquet`
   (293,532 × 76) + `data/features/training.parquet` (248,173 × 76) in **335.6s**.
3. **Predict** — `predictor.py` loaded v3 (won/placed_2/showed) + v3nf value model,
   all calibrators, scored **108 races / 1184 runners** → `predictions.json` +
   `inference_features.parquet`. Telegram channel active. Calibrated probs emit.
4. **UI walkthrough** (Streamlit @127.0.0.1:8501, all pages OK):
   - Race list → Race Detail (Ascot 14:30, 27 runners, 2 value bets).
   - Placed £10 win paper bet on Victorious @5.00 → bankroll £1,000 → £990. ✓
   - Paper Betting page: open bet listed; "Settle from results" ran (no scraped
     result for a same-day race, so nothing to settle — expected, not a bug).
   - Performance dashboard: manual settle (win) → bankroll €1,040, P&L +€40,
     win-rate 100%, 1 settled/0 pending, **all 5 Plotly charts populated**. ✓
   - Today's Suggestions: 4 LEAN picks from 108 races (4 with value), each with
     edge/EV/win%/fractional-Kelly stake + SHAP rationale + 1-click bet widget. ✓
5. **pytest** — **1033 passed in 90.4s**, 0 failed.

## Bugs fixed this pass

- **Placeholder runners** ("1st Favourite"/"2nd Favourite" etc. with null odds)
  were scraped by boylesports and scored as real horses, topping race cards.
  Fixed at source (`scraper/boylesports.py` `_parse_event` regex skip) AND
  defensively in `utils/normalizer.py` (`_drop_placeholders` applied in both
  `_from_live_odds` and post-merge in `_write`, since read-merge-write re-reads
  the prior unified parquet and would otherwise resurrect baked-in placeholders).
  Flushed on-disk unified (584,393 → 584,381 rows).

## Remaining issues — prioritized backlog

- **P1 — win-prob quantization / null live features (STRUCTURAL).** Live
  inference features are ~100% null (distance, going, class, form, ratings, pace)
  because no racecard source feeds today's runners — only odds + partial
  historical connection rates (Timeform dormant). Symptom: only ~11 distinct
  win-probs across 368 runners; suggestion #1 (Paddy De Pole) shows EDGE +140% /
  EV +109% in a 4-runner field — the model is guessing from sparse signal in
  tiny fields. Fix = wire a live racecard feed (Timeform revival or alt source).
  Not a quick fix.
- **P2 — sklearn version skew.** Calibrators pickled under sklearn 1.8.0, runtime
  is 1.9.0 → `InconsistentVersionWarning` ("might lead to invalid results").
  Re-fit/re-pickle calibrators under 1.9.0, or pin sklearn==1.8.0.
- **P3 — Suggestions page cold-cache latency.** `suggest_from_cache` takes ~58s
  cold (SHAP explainer load + 108-race scan); page shows a skeleton spinner the
  whole time. Acceptable but worth a warm-cache/precompute step.
- **P3 — Same-day settlement gap.** "Settle from results" can't settle a bet on a
  race that hasn't run; only manual settle works intraday. Expected behaviour,
  documented so it isn't mistaken for breakage.

## Key operational lesson

`normalize` takes ~7min and feature build ~5.6min — **never wrap them in short
timeouts** (a 300s timeout killed a normalize mid-write last session, leaving an
empty unified parquet). Correct re-run order: scrape → normalize → predict.

See [[review-09-audit-2026-06-14]] (which features are null/sparse) and
[[model-09-baseline-audit]] (v3nf is the trustworthy calibrated model; v3 market
features are leaked). Quantization here is the live-inference face of those.
