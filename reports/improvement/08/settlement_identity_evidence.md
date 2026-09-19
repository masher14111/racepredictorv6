# Step 08 — supporting evidence

2026-09-19. Commands run against the real repo data on disk (read-only checks
unless stated). Python: `.venv` (3.14.7, unchanged).

## 1. The pre-fix bug, reproduced on the real store

`scripts/daily_paper_loop.py::_index_results` indexed results on `horse_id`
**alone**, globally, no venue/date restriction:

```python
by_id[str(r["horse_id"])] = r   # last row wins, across ALL dates/venues
```

`_settle_open_tickets` then did `by_id.get(str(ticket.horse_id))` first,
before any venue/date check. Any open ticket for a horse would settle
against whichever result row for that `horse_id` happened to be read last
from the `betsp` parquet — including a different race, different day,
different course. Confirmed the same `horse_id` legitimately recurs across
dates on the real store (a horse racing more than once a season):

```
>>> df.groupby('horse_id')['race_date'].nunique().gt(1).sum()
35913   # of 40,633 total horse_id values in data/historical/betsp.parquet
```

Regression test `test_a_result_from_a_different_race_never_settles_a_ticket`
(tests/scripts/test_daily_paper_loop.py) reproduces this directly: an open
ticket's own race result (lost, position 4) is listed *before* a
same-`horse_id`, different-venue, different-day result (won, position 1) in
the results list fed to `_settle_open_tickets`. Verified by re-running the
pre-fix `by_id[str(horse_id)] = r` algorithm standalone against this exact
fixture (last-write-wins over the two rows in list order): it settles the
ticket "won", from the Doncaster row, not its own race. The fixed code
(this session) settles the same ticket "lost", from its own race only.

## 2. `execution.snapshots.race_uid` drops venue once an off-time parses

```python
>>> df.groupby('race_date')['venue'].nunique().gt(1).sum()
4   # of 35,107 unique off-times in data/historical/betsp.parquet
>>> collisions.index[0]
Timestamp('2024-05-17 16:15:00+0100')  # Leopardstown AND York both went off then
```

`execution/snapshots.py::race_uid()` returns `dt.strftime(_RACE_TIME_FMT)`
with **no venue** once the off-time parses — confirmed by reading the
function body (`execution/snapshots.py:182-214`) and `_RACE_TIME_FMT =
"%Y-%m-%dT%H:%M:%S+00:00"`. Two meetings sharing an exact post time (4
confirmed instances out of 35,107, ~0.011%) would collapse to one
`race_uid`, shared by `execution.gates.race_uid`, `execution.tickets`, and
`execution.snapshots.SnapshotStore`'s own keying. Real but low-frequency;
not fixed this stage (see stage note "Deferred, not fixed" for why). The
stage 08 identity fix in `daily_paper_loop.py` does **not** reuse this
function for the results join — it builds its own
`(venue, off-time-to-the-minute, horse)` key so this specific collision
cannot contaminate settlement, but `SnapshotStore.closing_quote` (used only
for the CLV metric, not for win/lose) still keys on the venue-dropping
`race_uid` and remains exposed to it.

## 3. `execution.settlement`/`execution.race_facts` are unused by the live loop

```
$ grep -rn "from execution.settlement" --include="*.py" . | grep -v tests/
./execution/race_facts.py   (defines the bridge, not a caller)
./execution/simulator.py:62:from execution.settlement import EachWayTerms, RaceResult, settle_ticket
```

Only `execution/simulator.py` (the offline backtest/walk-forward harness)
calls `execution.settlement.settle_ticket`. `scripts/daily_paper_loop.py`'s
live `_settle_open_tickets` -> `TicketStore.settle()` computes win/lose
directly via `backtest.metrics.settle(stake, odds, 1.0/0.0, commission)` —
no Rule 4 deduction, no dead-heat divisor, no non-runner void, no each-way
place leg. `execution.race_facts` (the Sporting Life archive parser that
`RaceResult` needs for that data) is verified to have real, recent coverage:

```
$ ls data/historical/raw/sporting_life/2026/2026-09-17 | wc -l
32
>>> parse_raw_file(...) for that day -> real finish positions, ride_status
```

i.e. the data needed to wire this in is present as of 2 days before this
session's "today" (2026-09-19) — this is a real, actionable gap, not a
missing-data blocker, and not attempted this stage (see stage note).

## 4. Test runs

- `tests/scripts/test_daily_paper_loop.py`: 13 passed (6 pre-existing + 7
  new for this stage).
- `tests/execution tests/scripts tests/ui` (`-n auto --dist loadscope`, 2
  workers): 687 passed.
- `tests/execution/test_settlement.py tests/execution/test_race_facts.py`
  (isolated re-check that Rule4/dead-heat/each-way/non-runner logic and its
  own date/identity handling are unaffected — this stage did not edit either
  file): 83 passed.
- Full repository suite, serial: **2,336 passed**, 1 pre-existing unrelated
  warning (same as step 07: `tests/utils/test_timezone_bucketing.py`
  dateutil format warning). Up from step 07's 2,329 (+7 new tests here).

## 5. Live ledger state (unchanged by this stage — no repair needed)

```
>>> TicketStore(cfg=cfg).to_frame()
988 rows, decision counts: {'PASS': 988}, settled rows: 0
```

Zero tickets have ever been settled in production (model verdict is NO-GO,
so every decision is PASS). There is no corrupted historical settlement to
repair — the identity fix is prophylactic for the first real CANDIDATE
settlement, not a repair of past financial figures. Confirms the "988 open
tickets" previously reported (STATE.md, DECISIONS.md D-series) was always
988 **PASS decisions**, not open wagers — exactly the reporting conflation
fixed in `_paper_summary` (item 2 below).

```
>>> loop._paper_summary(store)   # after the fix, same real data
{'n_tickets': 988, 'n_pass': 988, 'n_candidates': 0,
 'n_settled': 0, 'n_open': 0, ...}
```
