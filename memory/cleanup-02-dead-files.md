---
name: cleanup-02-dead-files
description: Dead-file / orphaned-module audit (2026-06-16) — import graph built from all entry points; NO files removed (user kept all); 2 superseded standalone UI scripts + 2 low-confidence suspects recorded
metadata:
  type: project
---

Dead-file audit run 2026-06-16 on branch `chore/git-hygiene`. Built a first-party
import graph (packages: features, models, scraper, utils, llm, ui, tools, scripts,
pipeline) from every entry point: `pipeline.py`, `ui/app.py` + `ui/pages/*`, the
`scripts/*` and `tools/*` CLIs, `utils/storage/__main__.py`, and all `tests/*`.

**Decision: nothing deleted.** User opted to record findings only — all files kept.

Dynamic-reference patterns checked before flagging (these hide edges from import grep):

- `ui/pages/2..6_*.py` execute their real module via `runpy.run_path(str(...))` (string
  path, NOT import): live_races, bet_placer, performance_dashboard, race_compare,
  settings are all LIVE this way.
- `ui/refresh_ops.py` reaches every bookmaker scraper via
  `importlib.import_module(f"scraper.{key}")` — all scrapers LIVE.
- `scripts/*` and `tools/*` are CLI entry points (`__main__` guard or `python -m`), so
  "no inbound import" does NOT mean dead for them.

**No HIGH-confidence dead files exist** — nothing is both unreferenced AND not an
intentional entry point.

Kept-but-suspicious (candidates if cleanup resumes):

- `ui/predictions.py` (MED) — no importer, no page wrapper, not in nav. The prediction
  panel was inlined into `ui/app.py` (`_load_predictions`/`_run_predictor`/
  `_render_race_card`), superseding it. Only reachable via standalone
  `streamlit run ui/predictions.py`.
- `ui/performance.py` (MED) — superseded by `ui/performance_dashboard.py` (wrapped by
  page 4). Only standalone-runnable.
- `llm/text_features.py` + `llm/__init__.py` (LOW) — reached only by
  `tests/test_text_features.py`; not wired into features/builder|derive|engine.
  Deliberate tested WIP module (matches "LLM-for-text-only", see
  [[reference-horse-racing-mvp-vault]]).
- `tools/preview_value_ui.py` (LOW) — self-labeled "throwaway dev tool" (static HTML
  design preview); standalone by design.

Everything else (features/_, models/_, all scrapers, utils/_, storage, the run_path-
wrapped ui/_ modules) is reachable. Related: [[cleanup-01-git-hygiene]].
