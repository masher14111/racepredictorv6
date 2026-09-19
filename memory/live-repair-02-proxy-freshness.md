---
name: live-repair-02-proxy-freshness
description: Stage 2 PASS — proxy rotation/redaction, destination health, staleness-TTL gating, and atomic per-source writes are implemented and tested; the decisive proof that a stale source can never produce EV or a suggestion is now a green end-to-end test, not just an architecture claim
metadata:
  type: project
---

# Stage 2 — proxy, freshness, and atomic snapshot repair (2026-07-27)

Stage 2 of the programme in [[live-repair-00-prompt-sequence]]. Depends on
[[live-repair-01-market-ingestion]] (Stage 1, verified DONE-but-uncommitted).
Full detail in `HANDOFF.md`, section "2026-07-27 — Stage 2".

## The decisive proof (requirement 8 / acceptance item 4)

Stage 0's triage found a project report,
`reports/calibration_dependency_failure_20260725.md`, recording a **Fail**
row: "Partial or freshness-unknown snapshot cannot produce any EV" —
i.e. the project's own evidence said this was _not_ actually proven. This
session closed that gap for real:
`tests/models/test_predictor.py::test_stale_price_cannot_reach_value_bet_or_suggestion_end_to_end`
walks one runner with a stale 50.0 price and a fresh 3.0 price through the
unmocked chain `_odds_by_book_map` → `_attach_odds_by_book` →
`_effective_decimal` → `Predictor._build_race` → `check_output_invariants` →
`find_value_bets` → `suggest_bets`, and proves the stale price never
participates: `best_odds` resolves to 3.0, `expected_value` is negative,
`value_bet` is `False`, zero invariant violations, zero picks, zero
suggestions.

**Architectural finding, not a bug — must not be "fixed" by a future
refactor without re-reading this note first.** `find_value_bets`'s own
`too_stale` gate (in `models/value.py`) never fires on this specific
predictor→UI→suggestions path, because `models/predictor.py::_runner_dict`
never propagates `stale`/`fetched_at` into the race-dict runner objects it
builds. That gate is still real and load-bearing for any _other_ caller that
constructs a runner dict with those fields directly — it is not dead code in
general, only unreachable on this one path. All protection on this path is
upstream, in `_odds_by_book_map`'s whole-(source,race)-group rejection. See
[[per-source-race-id-fragments-oddsmap]] for the related finding about
`race_id` not being a safe join key across books.

## What Stage 2 actually shipped

- Rotating-gateway semantics, country targeting, typed failure reasons,
  and safe runtime config reload were **already correct** before this
  session (re-verified, no changes needed).
- **Real gap #1, fixed:** partitioned parquet writes (the year-partitioned
  `data/unified_races.parquet`) were not crash-safe — the old code
  `rmtree`'d the live directory before writing the replacement. New
  `write_partitioned_parquet()` / `_atomic_replace_dir()` in
  `utils/storage/parquet_store.py` (sibling `.tmp` dir + two-rename swap +
  `.bak` rollback) closes it, with 5 crash-injection tests proving the prior
  snapshot survives a simulated kill mid-write.
- **Real gap #2, fixed:** `check_destinations()`/`set_proxy_reachable()`
  existed and were unit-tested in isolation but were never called from any
  production code path — `data/source_health.json`'s `proxy_reachable` field
  was permanently `null`. `scripts/refresh.py::_check_destination_health()`
  now runs it after every scrape and records the result.
- **Real gap #3, fixed:** the CLI/cron refresh path
  (`python -m scripts.refresh`) had zero source-health visibility, unlike the
  Streamlit page. `scripts/refresh.py::_print_source_health()` closes it.
- **Real gap #4, fixed (defense-in-depth):** `redact_proxy_url()` only masks
  a single, already-isolated proxy URL string. New `redact_secrets(text)` in
  `utils/proxy_manager.py` masks any `scheme://user:pass@host` found anywhere
  inside free text — for the case where a raw HTTP-client exception embeds a
  full proxy URL in its own `str()`. Wired into the catch-all exception
  handlers in `scripts/refresh.py` and `ui/refresh_ops.py`.
- **Real gap #5, fixed:** the unit-test suite was writing the **real**
  `data/source_health.json` — only `tests/utils/test_source_health.py`
  isolated `source_health._PATH`; scraper tests that transitively call
  `record_success`/`record_failure` did not. New autouse fixture
  `tests/conftest.py::_isolate_source_health` fixes this for every test in
  the suite at once. The polluted file was deleted; the copy now on disk was
  produced only by a genuine bounded `check_destinations()` run outside
  pytest (see below).
- Mixed multi-source snapshot assembly (one fresh book, one stale book, one
  book with no rows for a race) is now a named, tested contract, not an
  implicit property.

## Bounded live health check (real network)

```
livescorebet   reachable=True
paddy_power    reachable=True
boylesports    reachable=False   <- confirmed genuine 403 via direct no-proxy GET
```

BoyleSports failing _through the proxy pool_ and returning 403 _without_ a
proxy is the **correct** outcome for "a proxy-required domain must never
silently fall back to direct" (requirement 2) — it is evidence the design
works, not a defect.

## Security incident this session — do not repeat

A diagnostic shell command against the git-ignored `config.local.yaml`
(`grep -A2 "^proxy_pool" config.local.yaml`) echoed a live DataImpulse proxy
URL — including its embedded username and password — into this session's
tool-call transcript. **The credential is not reproduced in this file, in
HANDOFF.md, in any test, or in the Stage 1+2 commit; it must never be.**

Mitigations taken:

- Confirmed `config.local.yaml` has never been committed
  (`git log --all --oneline -- config.local.yaml` → nothing) — exposure is
  confined to the local session transcript, not the repository.
- `redact_secrets()` above is the durable code-level mitigation for the
  general failure mode (a raw exception string carrying a credential).
- **The operator must rotate the exposed DataImpulse credential** — treat it
  as no longer confidential. This is a manual step outside this session's
  ability to perform.

**Guidance for future sessions:** never run a raw `grep`/`cat`/similar
command against `config.local.yaml` (or any `*.local.yaml`) and let its
output land in a tool result. If you need to confirm a key exists, check for
the key name only, or read the file with a tool whose output you control and
can redact before it is echoed back.

## Commit

Joint Stage 1+2 commit, per the plan of record in
[[live-repair-01-market-ingestion]] and [[live-repair-triage-20260727]].
Message: `fix: validate primary win markets and harden proxy and odds
freshness` — deviates from the raw Stage 2 prompt's suggested
`fix: harden proxy and odds freshness` because the commit necessarily also
carries Stage 1's scraper/normalizer work (inseparable at hunk level); a
message naming only Stage 2 would misrepresent the commit's actual contents.

**Deliberately excluded from this commit, deferred to Stage 3's own
checkpoint:** `models/predictor.py`, `models/value.py`,
`models/suggestions.py`, `models/predict_unified.py`, `features/fuse.py`,
and their mirroring test files (`tests/models/test_predictor.py`,
`tests/models/test_value.py`, `tests/features/test_fuse.py`) — these mix
Stage 1/2 work with substantial, not-yet-checkpointed Stage 3 work
(`_build_race`, `_race_uid`, `check_output_invariants`, reference/executable
odds split), confirmed by Stage 0's own hunk-level analysis. Including them
would ship unfinished Stage 3 work under a Stage 1+2 label. This means the
end-to-end stale-price proof above, while fully verified in the working
tree and documented here, is **not** part of this commit's snapshot — it
lands with Stage 3. Also excluded as generated/runtime artifacts, not code:
`data/predictions.json`, `data/source_health.json`.

Commit hash: `9ce4159bcf86da633bebe3d8a7763d3585957c07` (parent
`6d6efc263f23eff45b6f613b73f60dc632c4cfe1`); full rationale in `HANDOFF.md`,
section "2026-07-27 — Stage 2", under "Commit".

## Next

Stage 3 is next. Its own two open items (from Stage 0/1's unresolved-issues
log, both still open): `data/predictions.json` has no `runners` key (170 of
514 runners serialized), and `scripts/reconcile_prob_to_ev.py` rewrites
`fetched_at`/`stale` so it cannot evidence freshness. Stage 3's commit will
also be the one that finally lands `models/predictor.py`'s Stage 1
fail-closed hunks and `features/fuse.py`'s Stage 2 staleness-gate hunks,
since none of those files could be split cleanly at any stage so far.
