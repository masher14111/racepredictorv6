---
name: ui-23-detail-pages
description: Race + Horse detail pages — calibrated bars, model's-view, recent form, SHAP "why"; data sources + the empty-breakdown fix
metadata:
  type: project
---

Rich **race & horse detail pages** built on the Prompt-19 design system ([[ui-19-redesign]]). Both pages are pure-HTML-string renderers fed by cached loaders, so the data→markup contract is testable headless (Streamlit only does layout + `st.markdown(..., unsafe_allow_html=True)`).

## Pages

- **`ui/pages/8_Race_Detail.py`** — full field. Per runner: calibrated win/place/show bars, value badge (+pp edge), each-way badge, EV, confidence chip, best-price book chips. Adds `C.model_view(_form.race_shape(selections))` (a plain-English "model's view" — top pick, win%, race shape from the top-two `won_prob` gap, value-bet count), context chips (going/field), runner names link to `Horse_Detail?horse=<id>&race=<slug>`, and an honest SHAP source note.
- **`ui/pages/12_Horse_Detail.py`** (NEW) — one runner. Sections: header app-bar ("Runs on record" badge), "Today's race" panel (connections, price, win/place/show bars, EV, confidence) when the horse is in a loaded race, context chips (form trend / career runs / days-since-run / going-pref win%), **Recent form** `C.form_table` (last 10 runs: finish-position chips, date, course, going, class, distance, speed fig, SP), **Connections & record** `C.stat_grid` (jockey/trainer/combo/horse/course/distance win%), **"Why the model rates this runner"** SHAP drivers as readable +/- bars (not a raw plot) with provenance note, and the paper-bet widget.

## Data sources (keyed on hashed `horse_id`)

- **`ui/_form.py`** (NEW) is the form/feature layer. `data/features/training_full.parquet` (248k labelled rows) is the real form + SHAP feature source — `data/races.db` is empty, so the prompt's "form from races.db" does not hold. `horse_history_from()` (recent-first runs), `connection_stats_from()` (aggregates off latest row), `feature_row_from(matrix, inference, id)` → `(row, "live"|"history")`, `race_shape()`, `trend_label()`.
- `data/predictions.json` supplies the live card (probs, odds, value, EV).
- `data/inference_features.parquet` (NEW) — written best-effort by `models/predictor._write_inference_features()` after scoring, so future live runs resolve a real **live** feature row for SHAP.

## The empty-breakdown fix (regression-tested)

Old bug: SHAP read `data/features.parquet`, a synthetic 5-row fixture (ids `h1/h2`) with **zero** overlap with live hashed ids → breakdowns silently blank. Fix: SHAP feature rows now resolve via `_form.feature_row_from()` — prefer the live inference store, else the latest real matrix row; only a true debutant/unmatched runner shows "unavailable". Verified live (Chepstow `chepstow-20260615T1350`, Pride Of Nepal `6f1a40e1a19311f2`): real drivers render (speed rank, recent place rate, field size…). UI is honest about provenance via `C.driver_source_note(source)` ("from today's race card" vs "most recent run profile, no live feature row yet"). Regression locked in `tests/ui/test_form.py::test_feature_row_resolves_from_history_for_known_horse`.

## Verify

`pytest` → **1018 passed, 3 skipped**. Screenshots: `.design-md/detail-race-chepstow.png`, `.design-md/detail-horse-pride-full.png` + `-bottom.png`. New tests: `tests/ui/test_form.py`, additions to `tests/ui/test_components.py`. Components added in `ui/_components.py` (`model_view`, `context_chips`, `form_table`, `stat_grid`, `driver_source_note`, `fmt_rate`, `fmt_speed`) + CSS in `ui/_design.py` (`.rp-form`, `.rp-modelview`, `.ctx-row`, `.rp-hstat`, `a.rp-hlink`). Gotcha (from [[ui-19-redesign]]): restart the Streamlit server after editing imported modules — page-file autoreload doesn't pick them up.
