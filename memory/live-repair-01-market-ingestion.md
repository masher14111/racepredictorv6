---
name: live-repair-01-market-ingestion
description: Stage 1 PASS — market-identity whitelisting (not name regexes) now gates live WIN ingestion; LivescoreBet median booksum fell 3.68 to 1.206 and no INVALID race can produce a prediction, but the stage is still uncommitted because its scraper files interleave PARTIAL Stage 2 work
metadata:
  type: project
---

# Stage 1 — primary WIN-market ingestion repair (2026-07-27)

Stage 1 of the programme in [[live-repair-00-prompt-sequence]]. Builds directly
on the verification in [[live-repair-triage-20260727]], whose Stage 1 claims were
re-checked against code and data this session and held. Full detail in
`HANDOFF.md`, section "2026-07-27 — Stage 1".

## The design decision that matters

**Market identity is the primary classifier; runner-name regexes are only a
secondary quarantine guard.** This is the whole point of the stage and must not
be softened later.

The proof is `tests/scraper/fixtures/livescorebet_adversarial.json`: alongside
the genuine `To win` market it carries a `Without 1 (Alpha)` market whose
selections are named plainly **"Bravo"** and **"Charlie"** — byte-identical to
real runners in the genuine market. They are still excluded, because the
whitelist (group `758` + type `1001558122` + name `to win`) rejects the market
before any name is examined. A name-based classifier could not possibly get this
case right.

Per-source contracts:

- **LivescoreBet** — whitelist group `758` + type `1001558122` + name `to win`.
  An unknown market inside the primary group is logged and quarantined, never
  defaulted to WIN.
- **BoyleSports** — parsing scoped to `#racingNavMiniMenu` (the active primary
  market) plus `#RacingMarketSelections` (its runner container). Price boosts and
  specials live under `#RacingMarketRefresh` and cannot participate in discovery.
  This replaced "select every odds button on the page".
- **Paddy Power** — restricted to `marketType == "WIN"`.

## Fail-closed contract

`utils/market_validation.py` validates at **race level** — any issue invalidates
the whole source race and callers must never salvage individual rows. Reason
codes: `primary_win_market_count:N`, `primary_win_market_ids:N`,
`non_primary_win_market_type`, `missing_market_id`, `missing_selection_id`,
`duplicate_selection_id`, `duplicate_selection_name`, `missing_selection_name`,
`implausible_field_size:N`, `special_selection_in_primary_win`,
`extreme_booksum:X`. Thresholds: field 2–40, booksum 0.80–1.80.

## Measured results

- LivescoreBet **median booksum 3.6804 → 1.2058** (max 9.3627 → 1.4694), median
  field **30 → 10**. A booksum of 3.68 is three-plus books stacked on each other;
  1.206 is an ordinary ~20.6% UK/IRE overround.
- BoyleSports 46 VALID / 9 INVALID; Paddy Power 38 VALID / 1 INVALID. All ten
  rejections are priceless overseas fixtures (Del Mar, Gulfstream, Prairie
  Meadows, Enghien) with `booksum 0.0`.
- `data/live_odds.parquet`: 948 rows × 21 cols, **zero INVALID rows persisted**.
- Verified on the shipped artifact, not just in unit tests: 51 predicted races,
  **0** with no VALID live counterpart, **0** INVALID races producing a
  prediction, **0** specials among the 170 serialized selections. Note the
  parquet is UTC and predictions are Europe/Dublin — a naive string join on
  `race_time` produces a bogus 43-race mismatch.
- `python -m pytest -q` → **1365 passed** (was 1351 before this session).

## Work done this session

Requirement 9's reporting gap was the only genuinely open item.
`scripts/audit_live_markets.py` now emits `validation_reasons`, a reason-family
rollup, suspicious-selection examples, and runner/booksum quantile
distributions. Added `parse_validation_reasons()` / `reason_family()` to
`utils/market_validation.py`, plus `tests/test_audit_live_markets.py` (12 tests).

Extended the secondary regex to catch `"<horse> by N Lengths"` winning-distance
selections. Market identity already excluded these, so it was never a live
contamination hole. **False-positive check: 0 of 741 distinct real runner names
flagged** — worth re-running that check before ever tightening this regex again,
since a false positive fails a clean race closed.

## Limitations

- Markets rejected at **parse** time (unknown `marketGroupId`, non-primary
  container) never reach a cache or the parquet, so they are visible only in
  scraper logs and the audit cannot count them. Only race-level rejections and
  survivors are reportable.
- Paddy Power contributes no rows to `data/live_odds.parquet`; the audit reads it
  from its cache. Worth confirming in Stage 2.

## Why it is still uncommitted

`scraper/livescorebet.py`, `boylesports.py`, `paddy_power.py` and
`utils/normalizer.py` carry Stage 1 **and** PARTIAL Stage 2 edits;
`models/predictor.py` carries Stage 1 **and** Stage 3. By `git diff -U0` hunk
count the scrapers are _majority_ Stage 2 (e.g. livescorebet 13 Stage 1 vs 20
Stage 2). Stage 1 code paths call Stage 2 symbols (`source_health.record_*`,
`next(required=True)`) and share the import block, so hunk extraction would not
even import. Committing the files whole would ship unfinished Stage 2/3 work
under `fix: validate primary win markets`.

This session's **own** increment (`utils/market_validation.py`,
`scripts/audit_live_markets.py`, their tests, the fixtures, the after-audit CSV)
is entirely additive and _is_ cleanly committable — left to the user's call
rather than done unilaterally, since that commit message would name a fix whose
scraper integration is absent from it.

Plan of record, unchanged from [[live-repair-triage-20260727]]: land Stages 1+2
as one joint commit when Stage 2 closes, stating the joint scope plainly.

## Next

Stage 2 is safe to start. Fix `utils.source_health._PATH` isolation first — the
scraper tests write the real `data/source_health.json`, so its contents are test
residue, not telemetry.
