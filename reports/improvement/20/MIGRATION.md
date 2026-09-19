# Migration 8 — race_key / ticket provenance columns, applied to the live store

Applied 2026-09-19 to `data/races.db` (the real, live paper-trading store).

## What it does

* `odds_snapshots`: adds nullable `race_key` (venue-qualified, minute-precision
  identity). **Never backfilled** — the table's own `RAISE(ABORT)` triggers
  (migration 4) forbid UPDATE, and the point of that table is that a past price
  observation is physically unrewritable. Existing rows keep `race_key = NULL`
  forever; every row written by `execution.snapshots.SnapshotStore.record`
  from now on computes its own `race_key` at write time (application code, not
  this migration).
* `paper_tickets`: adds nullable `race_key`, `model_content_hash`,
  `feature_schema_version`, `config_hash`, `prediction_cycle_id`,
  `provenance_complete`. This table is not append-only (`TicketStore.settle`
  already updates it), so `race_key` **is** backfilled here from each row's
  own already-stored `(venue, race_time)` — a pure, deterministic function, so
  re-running the backfill (if the migration were ever re-triggered) recomputes
  identical values. The four provenance columns and `provenance_complete` stay
  `NULL`/`0` on every pre-existing row — they predate the contract, and that is
  the honest reading, not a fabricated match.

## Row-count / integrity verification (this run)

| table | before | after |
|---|---|---|
| odds_snapshots | 4806 | 4806 |
| paper_tickets | 1420 | 1420 |
| text_archive | 243 | 243 |
| llm_shadow_extractions | 3 | 3 |

`PRAGMA user_version` 7 → 8. `paper_tickets` backfill: 1420/1420 rows now
carry a non-null `race_key` (spot-checked format:
`ayr|2026-07-27T13:15` for a real historical row, matching
`execution.snapshots.race_key`'s scheme exactly). `odds_snapshots`:
4806/4806 rows carry `race_key IS NULL`, as designed (see above) — this is a
one-time, bounded gap: any snapshot captured by the NEXT live `scripts.refresh`
cycle already has `race_key` populated, since that is application code, not a
property of this migration.

## Backup and rollback

Backup taken **before** migration: `data/audit/20/backups/races.db.pre_migration8_20260919T153817Z`
(byte-for-byte copy of the pre-migration file, 4,603,904 bytes, sha256 verified
against the live file at backup time).

**Rollback** (if ever needed): stop any running loop process, then
`copy data\audit\20\backups\races.db.pre_migration8_20260919T153817Z data\races.db`
— this reverts every schema and data change from migration 8, back to exactly
the pre-migration state (`user_version=7`). No other file changes this stage
made are DB-state-dependent, so a plain file copy is sufficient; there is no
separate "undo" script.

## A real defect found and fixed while applying this migration

The first version of migration 8 tried to `UPDATE ... SET race_key = ?` on
`odds_snapshots` unconditionally for every table, including that append-only
one — it raised `sqlite3.IntegrityError: odds_snapshots is append-only: UPDATE
forbidden` **on this real store** (4,806 existing rows; a fresh empty test db
never exercised the crash, which is why unit tests alone missed it). The
`ALTER TABLE` / `CREATE INDEX` statements in the same `executescript()` call
had already committed by the time the backfill loop raised, leaving the real
db with the new columns present but `user_version` still `7` — a genuinely
partially-applied state. Restored from the backup above, then fixed the
migration to (a) never attempt to update `odds_snapshots` at all, and (b)
guard every `ALTER TABLE ADD COLUMN` with a `PRAGMA table_info` existence
check first, so a second attempt after ANY partial failure resumes cleanly
instead of raising "duplicate column name". Regression tests:
`tests/utils/storage/test_migrations.py::test_migration_8_never_updates_the_append_only_snapshots_table`
and `::test_migration_8_resumes_after_a_partial_prior_attempt`, both of which
reproduce the exact failure mode on a synthetic populated store before
asserting the fix. Re-applied cleanly to the real store afterward (see table
above — row counts identical, no data lost).
