# Step 07 — bounded live capture evidence

2026-09-19. Command run: `execution.snapshots.ingest_live_odds_parquet()` against
the real `data/races.db` (`cfg.snapshots.db_path`) — the same function
`scripts/refresh.py::_capture_snapshots()` calls after every scrape poll. This is
the "short bounded current capture" required by step 07's scope item 5; a full
live network scrape was not run (WAF-blocked / no browser in this session,
consistent with steps 05/06), so this exercises the actual capture entry point
against the real, already-scraped `data/live_odds.parquet` board on disk
(905 rows, all `validation_status == VALID`, `fetched_at` 2026-09-18T23:46:21Z
.. 2026-09-18T23:49:12Z UTC).

## Run 1 (first ingest of this board)

```
pre count:  2642
inserted:   905
duplicates: 0
rejected:   0
post count: 3547
```

## Run 2 (immediate re-run, same board, no new scrape)

```
inserted:   0
duplicates: 905
rejected:   0
post count: 3547
```

Confirms on real data (not a synthetic fixture): the capture entry point is
idempotent (natural key `(race_uid, horse_key, bookmaker, market_type,
fetched_at)`, `INSERT OR IGNORE`), append-only (row count only grew), and every
row passed field-level validation before being considered (this board had 0
`invalid_status`/`missing_*` rejections). `data/races.db` now carries this
board permanently as part of its immutable history — consistent with the
store's contract (append-only) and with normal production operation (this is
exactly what a real `_capture_snapshots()` call does).

Not exercised by this bounded run: multiple polls of the *same still-open*
board (the on-disk file is a single already-closed poll), so intraday price
movement across two live polls was not observed today. That property is
already covered by `tests/execution/test_snapshots.py` (synthetic multi-poll
fixtures) and by the accumulated production history already in `data/races.db`
prior to this session (2,642 rows before this run).
