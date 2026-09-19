---
name: perf-30-rebuild-speed
description: Refresh/rebuild sped up ~5x (470s→93s) by vectorizing normalize date coercion + removing a duplicate inference build; also fixed the win calibrator failing to unpickle on numpy<2
metadata:
  type: project
---

2026-06-17. The UI "Scrape all + rebuild" / `scripts.refresh --no-scrape` rebuild
(normalize → build → predict) was ~8 min. Profiled and cut to ~1.5 min. GPU is
**not** the lever — scraping is network/Cloudflare-bound and the rest is pandas/IO;
CatBoost training already runs on GPU (task_type GPU). Three fixes:

1. **`utils/normalizer.py` `_reindex_canonical` was the monster (194s, run twice
   per `normalize` = ~388s).** It did `out["race_date"].map(_date_str)` — a scalar
   `pd.to_datetime` per row over the ~584k-row unified frame. Added
   `_date_str_series()` using `pd.to_datetime(s, utc=True, errors="coerce",
format="mixed").dt.strftime(...)` — byte-for-byte identical to the scalar path
   (verified on real data + edge cases incl. slash-dates/ISO-datetimes), 193s→3.2s.
   normalize 408s → **32.7s**.
2. **Duplicate inference build.** `ui/refresh_ops.rebuild_predictions` called
   `build_inference_matrix` (just to count runners) and then `predictor.predict()`,
   which calls `build_inference_matrix` AGAIN internally (~57s each). Added an
   optional `live=` param to `Predictor.predict()` so the caller passes the matrix
   it already built; predict reuses it (still takes the un-fused per-book odds map
   from `unified`). predict step 57s → **2s**.
3. **Win calibrator unreadable on numpy<2** (the log warning
   `calibrator catboost_won_v3_calib.pkl unreadable: No module named
'numpy._core.numeric'`). v3 win probs were silently falling back to RAW
   (uncalibrated) whenever Streamlit ran under the user's Python 3.10/numpy-1.x
   env. Root cause: `sklearn.IsotonicRegression` pickles numpy arrays whose module
   path changed in numpy 2.0 (`numpy.core`→`numpy._core`). Added portable
   `IsotonicCalibrator` (stores threshold lists as plain floats, predicts via
   `np.interp`, like the already-portable `SigmoidCalibrator`); `_fit_one` now
   returns it. Converted the existing `catboost_won_v3_calib.pkl` in place (loaded
   under numpy 2.4, re-saved portable — predictions matched to 1e-9, no numpy in
   the pickle). The 89-byte sigmoid calibrators were always fine. See
   [[model-11-calibration]].

Net rebuild: ~470s → **93s** (normalize 33s + build 57s + load 1s + predict 2s),
identical output (584,475 rows, 119 races / 1284 runners). Full suite **1056✓/3 skip**.
Remaining hot spot = `build_inference_matrix` (57s, `_derive_all` over the pruned
frame) — next lever if more speed is needed. Two Python envs exist on this box:
3.14/numpy-2.4 (Bash, README-official) and a 3.10/numpy-1.x the user launched
Streamlit with — the portable calibrator means either env now serves calibrated
win probs. Relates to [[wrap-29-yesterday-predictor]], [[wrap-27-smoke-test]].

**Follow-up (same day): Live Races page "didn't load" = a ~237s hang, not an
error.** `ui/live_races.py::_load_races` read the ENTIRE `unified_races.parquet`
(~584k rows) and `_build_races` looped over all ~31.5k historical race-groups
(per-group `sort_values`+`groupby().first()`) before the status filter threw all
but ~123 upcoming away. Added `_live_only(df)` (keep `race_date >= today`) applied
right after the parquet read → ~2k rows / 123 races, build 237s→1.3s. The
`FutureWarning: concatenation with empty or all-NA entries` the user kept seeing
was unrelated noise from `normalize`: an absent source's adapter returns a 0-row
all-NA-column frame; now skipped (`if not adapted.empty`) before `pd.concat` →
0 warnings. 1056✓ still.
