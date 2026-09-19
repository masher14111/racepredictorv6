# Stage 19 — capped OpenRouter DeepSeek shadow extraction, integration evidence

PAPER-ONLY. Model verdict stays NO-GO (unchanged by this stage). `llm.enabled` stays `false` —
nothing here influences a probability, EV, or any `models.features` column.

## 1. What was inspected before writing anything

- `llm/hosted_adapter.py` (`OpenRouterBackend`) — Stage 15's hosted transport: strict grounding
  validation, bounded retries, reserve-then-commit cost accounting, per-record cache keyed on
  (schema/prompt version, provider, model, reasoning_effort, max_output_tokens, text). Reused
  as-is, unmodified.
- `llm/hosted_budget.py` (`HostedBudgetLedger`) — the durable, file-backed, PROGRAMME-lifetime
  spend/request cap at `data/cache/llm/hosted_budget_ledger.json`. Reused as-is, unmodified;
  no new ledger file, no reset.
- `llm/text_features.py` / `llm/text_archive.py` — the tri-state grounded-feature schema and the
  append-only source-text archive (Stage 14). `llm.enabled`/`llm.backend` (the switch that lets
  `TextFeatureExtractor` influence a forecast) are untouched by this stage.
- `scripts/benchmark_extraction.py`, `scripts/verify_hosted_pricing.py`, Stage 15's scorecards.
  **What Stage 15 actually established:** a ONE-DAY labelled benchmark (243 comments, 21 races,
  all from 2026-06-17) scoring regex vs. hosted DeepSeek against a small REVIEWED label subset,
  finding hosted `schema_valid_rate` 86.01% at `reasoning_effort=low`/`max_output_tokens=1600`
  (D47) and total hosted spend $0.2836/498 requests. It did **not** establish multi-day
  reliability, did not touch a live daily pipeline, and the reviewed labels are a small manual
  subset, not the full 243-comment corpus — this stage inherits those limits verbatim; it does
  not re-benchmark accuracy.
- **The actual daily acquisition path, before this stage:** `scraper/spotlight.py::scrape()` (the
  only comment source that feeds `llm/text_archive.py`) was **never called from
  `scripts/refresh.py` or `scripts/daily_paper_loop.py`** — grepped, confirmed absent. The archive
  was frozen at its single 2026-06-17 backfilled sample (D46/D48) because nothing in the daily
  loop had ever invoked it. This is a real, previously-undocumented gap in "the bounded daily
  workflow" that this stage's scope requires wiring into, so it is fixed here (section 3), not
  merely worked around.

## 2. New production code

- `llm/shadow_extraction.py` — new orchestration module. Reuses `OpenRouterBackend`,
  `HostedBudgetLedger`, `TextArchive`, `Cache(data/cache/llm)` unmodified; adds no competing
  client or ledger.
  - `ShadowExtractionStore` — new append-only SQLite table `llm_shadow_extractions` (migration 7,
    same `data/races.db`, same RAISE(ABORT) UPDATE/DELETE-refusal pattern as `text_archive`).
    Columns cover the required identity/timing fields: `archive_row_id` + `content_hash` +
    `source` (source identity + original text hash), `text_published_at`/`text_fetched_at`
    (publication/acquisition time, copied verbatim from the archive row), `extraction_time`
    (extraction availability time — see below), `schema_version`/`prompt_version`/`model`
    (versions), `features` (grounded evidence JSON).
  - **Extraction-availability honesty**: `extraction_time` is recorded separately from
    `text_fetched_at` and is the ONLY field `ShadowExtractionStore.as_of()` filters on — a batch
    run well after a race has gone off can never be read back as though it were available at
    scrape time. Proven in `tests/llm/test_shadow_extraction.py::test_as_of_hides_an_extraction_recorded_after_the_query_instant`
    and `::test_extraction_time_never_precedes_text_fetched_at`.
  - `verify_current_pricing()` — fetches the live OpenRouter `/models` listing (falls back to a
    cached snapshot on a transient fetch failure; fails closed with no live fetch AND no cache),
    computes the worst-case in/out price across any time-of-day overrides, and fails closed
    (`PricingUnbounded`) if it exceeds the configured ceiling — mirrors
    `scripts/verify_hosted_pricing.py`'s Stage-15 logic, run fresh on every invocation rather than
    from a frozen snapshot.
  - `run_shadow_extraction()` — the bounded batch orchestrator: disabled config / absent
    `OPENROUTER_API_KEY` / unverifiable or over-ceiling pricing / an exhausted programme cap each
    stop with a labelled, non-secret reason and touch neither the ledger nor the network.
    Candidate selection is a single indexed SQL join (`text_archive` LEFT JOIN
    `llm_shadow_extractions` WHERE no match), oldest `fetched_at` first, bounded to
    `batch_size_per_run` — never loads the whole archive, never re-submits an already-recorded
    row (resumable dedupe), never sends the whole database to the provider.
  - `write_health_report()` — a non-secret `data/execution/shadow_extraction_health.json`
    (model, batch counts, cache hits, cost this run, ledger snapshot, stop reason) — no API key,
    no raw text, no prompt content.
- `utils/storage/migrations.py` — migration 7 (`llm_shadow_extractions`), additive, append-only.
- `config.yaml` — new `llm.hosted_shadow.*` section, independent of `llm.enabled`/`llm.backend`.
  `enabled: true` (the key IS present in this environment) so the daily workflow attempts
  extraction when possible; every failure mode still fails closed.
- `scripts/refresh.py` — two new best-effort steps inside `_scrape()`:
  1. `_scrape_spotlight()` — calls `scraper.spotlight.scrape()` (previously never invoked from any
     daily entrypoint), archiving today's comments.
  2. `_shadow_extract_text()` — calls `run_shadow_extraction()` + `write_health_report()`, prints
     a one-line non-secret summary. Both are wrapped exactly like the existing odds scrapers: a
     failure is logged and printed, never raised, never blocks the rest of the refresh.

## 3. Offline acceptance (all in `tests/llm/test_shadow_extraction.py` unless noted)

19 new tests, all offline (fake `http_post`/`fetch_listing`, injected stores/ledger/cache):
disabled config, missing credential, pricing unverifiable (no live + no cache), pricing recovers
from a cached snapshot on live failure, pricing-exceeds-ceiling fails closed, model absent from
listing fails closed, successful batch + resumable dedupe across 3 separate `run_shadow_extraction`
calls with a NEW `HostedBudgetLedger` instance each time (genuine resume, not a shared in-memory
object), provenance fields copied correctly from the source archive row, extraction-time vs.
fetched-time monotonicity, the `as_of` point-in-time contract, identical-text cache-hit reuse
across two different archive rows (proves the adapter's cache is respected, no double-spend),
bounded retries (3 attempts = 1 + 2 retries, then a recorded failure, $0 spent), budget exhaustion
mid-batch (deterministic via `max_requests=1`) leaving the remainder for a later resume, an
ungrounded claim forced to unknown before it reaches the store, a malformed schema entry recorded
(not dropped) with `schema_valid=False`, the append-only table refusing UPDATE/DELETE, the health
report never containing the API key, and two invariance checks that `models.features` /
`features/text_features_v1.py` never reference `shadow_extraction` and that `llm.enabled=False`
still disables `TextFeatureExtractor` regardless of the shadow switch.

`tests/llm/test_hosted_budget.py` — added one concurrency test: 25 threads racing `reserve()`
against a cap sized for exactly 5 reservations; exactly 5 succeed, `reserved_cost_usd` never
exceeds the cap (proves the existing lock in `HostedBudgetLedger.reserve` is what this stage
relies on for concurrent-safe shared-ledger reservation — not new locking code, an existing
property proven directly).

`tests/test_refresh.py` — 4 new tests for `_scrape_spotlight`/`_shadow_extract_text`: both are
non-fatal on failure, both print recognisable status lines, and a mocked summary's fields reach
the console without going through the real network.

Results: `pytest_targeted.log` (124 passed), `pytest_full_suite.log` (**2,558 passed**, up from
Stage 18's 2,528 by the 30 new tests here).

## 4. Live evidence (within the shared $5/1,200 cap; nothing reset)

- **Pricing verification, fetched live today**: `hosted_pricing_verification.json` —
  `deepseek/deepseek-v4.1-flash` worst-case $0.30/$1.20 per million in/out tokens (6 time-of-day
  overrides), **exactly at** the configured ceiling, `within_ceiling: true`. Raw listing saved at
  `openrouter_models_raw.json` (447 models).
- **Production batch run** (`refresh_batch_health.json`): ran `run_shadow_extraction()` against
  the REAL archive/store/ledger with `batch_size_per_run=3`. All 3 selected rows were cache hits
  from Stage 15's benchmark cache (same text, same schema/prompt version, same
  `reasoning_effort`/`max_output_tokens`) — **$0 new spend**, `llm_shadow_extractions` now holds
  3 real rows (`production_shadow_store_coverage.json`). This is genuine evidence that "reuse
  existing caches" works end-to-end in production, not merely in a mocked test.
- **One genuine fresh network call** (`live_smoke_health.json`), on an ISOLATED store
  (`data/audit/19/smoke_live.db`, not the production `data/races.db`) seeded with one new,
  clearly-labelled probe comment (source=`stage19_live_smoke`) containing an unambiguous ground
  excuse and a first-time-headgear mention: the model correctly returned
  `ground_excuse=True` (evidence: "was unsuited by the soft ground last time") and
  `headgear_first_time=True` (evidence: "Wears cheekpieces for the first time today"),
  `schema_valid=True`, cost **$0.000125145**, committed to the REAL shared ledger. Ledger before
  this stage: $0.2835941872 / 498 requests (historical). Ledger after (production batch: $0 / 0
  new; live smoke: +$0.000125145 / +1): **$0.2837193322 / 499 requests**. Remaining:
  **$4.7162806678 / 701 requests**.
- No cap reset, no top-up, no model substitution (provider echoed back the exact requested model
  id both times), no credential printed anywhere in these artifacts.

## 5. Limitations, disclosed not hidden

- The text archive is still overwhelmingly a single historical day (2026-06-17, 243 rows) plus
  today's 1 isolated smoke probe (not counted in the 243) and whatever `_scrape_spotlight()`
  captures on future `scripts.refresh` runs when UK/IRE racecards carry spotlight commentary —
  this stage did not and could not fabricate additional real trading days. Stage 16's
  DEFERRED_DATA text-feature-forecast-value verdict is unchanged and not reopened by this stage.
  `llm.enabled` stays `false`.
  - `_scrape_spotlight()`'s real effect on `scripts.refresh` was **not exercised live this
    session** (no UK/IRE spotlight page was actually scraped) — proven only by the offline
    tests in `tests/test_refresh.py` (mocked `scraper.spotlight.scrape`). Whether Sporting Life
    actually serves fresh spotlight commentary on the next real cycle (and whether it is
    WAF-blocked like Timeform/B8) is unverified and should be checked at the next live
    `scripts.refresh` run, alongside the existing freshness checklist in
    `reports/improvement/18/OPERATIONS.md`.
  - Stage 15's reviewed-label subset (used for accuracy scoring) does NOT cover today's smoke
    probe or the 3 cache-hit rows freshly written into `llm_shadow_extractions` — this stage adds
    NO new accuracy evidence, only integration/collection evidence. Extraction accuracy claims
    remain exactly what Stage 15 measured.
- `ShadowExtractionStore.select_unprocessed` is a per-call bounded batch; nothing in this stage
  schedules it — `scripts.refresh` must actually run (manually, per `OPERATIONS.md` section 1) for
  any further batch to happen. No recurring job/service was created.
