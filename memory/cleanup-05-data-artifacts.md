---
name: cleanup-05-data-artifacts
description: data/ inventory — what each artifact is, regeneration commands, and the stale/orphaned ones to clean (2026-06-16)
metadata:
  type: project
---

Audited `data/` (2.2 GB total) on 2026-06-16. Full table + rebuild commands now in `data/README.md`. Pipeline: scraper → `data/historical/*.parquet` (+ `raw/`) → `utils.normalizer` → `unified_races.parquet` → `features.builder` → `features/training.parquet` (+ `features.parquet`) → `models.train`; `models.predictor` → `predictions.json`. One-shot: `python pipeline.py [--skip-scrape]`.

**Source-of-truth (never delete):**

- `races.db` (72 KB) — only artifact with non-regenerable USER state (`bets`, `bankroll_log`). Tables currently near-empty. Schema rebuild: `python -m utils.storage.migrations migrate`.
- `historical/betsp.parquet/` (22 MB, year-partitioned) — Betfair SP + Sporting Life enrichment; core training source. Re-scrape: `python -m scraper.betsp_historical`.
- `historical/raw/sporting_life/` (~2.0 GB, 29.7k `.html.gz`) — write-only archive for later LLM re-extraction (`scraper/betsp/raw_store.py`). **Nothing reads it back today.** ~90% of all of data/.

**Regenerable:**

- `unified_races.parquet/` (27 MB) → `from utils.normalizer import normalize; normalize(write=True)`
- `features/training.parquet` (21 MB) + `features.parquet` → `from features.builder import build_training_matrix; build_training_matrix(write=True)` (writes BOTH).
- `betsp_backbone.parquet` (15 MB) — resume checkpoint for the betsp backfill.
- `predictions.json` → `python -m models.predictor`; `live_odds.parquet` → `python -m scraper.boylesports|livescorebet`; `cache/*.json` → next scrape.

**DELETED 2026-06-16 (user confirmed):**

- `historical/raw/racing_post/` (~35 MB, 1.1k **uncompressed** `.html`) — orphaned: racing_post results source was DROPPED (`config.yaml` `results_sources: [sporting_life]`). No code produced/consumed it.
- `data/_*` scratch (~2.7 MB): `_sample_rp.html`, `_sample_sl.html`, `_sl_race.json`, `_slice_joined.parquet`, `_value_ui_preview.html` — one-off debug files (git-ignored via `data/_*`).

**Inert but kept:**

- `historical/timeform.parquet` (16 KB, ~28 rows) — Timeform dormant (no `session_cookie`); near-empty, left in place.

**OPEN DECISIONS (deferred — do NOT act without asking):**

- `historical/raw/sporting_life/` (~2.0 GB) — keep as LLM re-extraction insurance vs reclaim 2 GB. Nothing reads it today. User undecided.
- `features.parquet` rebuild — it's ~27 KB vs 21 MB `training.parquet` though it should be the LARGER full matrix (history + live). Last written from a small live-only slice → **stale/partial**. Fix is `build_training_matrix(write=True)`, but user said DO NOT regenerate yet. Matches prior note in [[review-09-audit-2026-06-14]] about stale features.parquet.

`data/README.md` written with the full table. Related: [[cleanup-01-git-hygiene]], [[cleanup-04-deps]].
