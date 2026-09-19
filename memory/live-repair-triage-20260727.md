---
name: live-repair-triage-20260727
description: 2026-07-27 triage of the uncommitted five-stage repair work — Stage 1 acceptance genuinely passes but is inseparable from PARTIAL Stage 2/3 edits in the same files, so no stage commit was created
metadata:
  type: project
---

# Live-repair programme triage — 2026-07-27

Verification-only checkpoint of the in-flight work from `RACE_PREDICTOR_REPAIR_PROMPTS.md`
(see [[live-repair-00-prompt-sequence]]). No functionality was implemented, no
code changed, no commit made. Full detail is in `HANDOFF.md`, section
"2026-07-27 — Stage 0".

## State found

Branch `chore/config-audit`, HEAD `6d6efc2`, working tree dirty (26 tracked files
modified, 19 untracked paths). A previous session ran Stages 1→3 on 2026-07-25,
hit a Stage 4 dependency failure, and stopped **without** writing `HANDOFF.md`,
any `memory/live-repair-01..05` note, or any stage commit.

## Verdicts

- **Stage 1 (primary WIN market) — DONE on acceptance.** Not committable alone.
- **Stage 2 (proxy/freshness/atomic) — PARTIAL.**
- **Stage 3 (probability/EV) — PARTIAL.**
- **Stage 4 (calibration audit) — NOT STARTED**, explicitly NO-GO in
  `reports/calibration_dependency_failure_20260725.md`.
- **Stage 5 (paper-betting safeguards) — NOT STARTED**, zero artifacts.

## Measured evidence

- Full suite `1351 passed in 121.87s`; targeted 12-file stage suite
  `426 passed in 26.41s`. Both with `.venv\Scripts\python.exe -m pytest -q`.
  `--timeout=` is unavailable: `pytest-timeout` is not installed in the `.venv`.
- Stage 1 before/after live audit — LivescoreBet went from 51 races at a median
  30 selections and median booksum **3.6804** (max 9.363) to 51 VALID races at
  median 10 runners and booksum **1.2058**. BoyleSports 46 VALID / 9 INVALID,
  Paddy Power 38 VALID / 1 INVALID.
- `data/live_odds.parquet` is 948 rows × 21 cols carrying `market_id`,
  `market_name`, `selection_id`, `validation_status`, `validation_reasons`,
  `field_size`, `booksum`; **zero INVALID rows persisted**.
- The LivescoreBet adversarial fixture contains a "Without 1 (Alpha)" market
  whose selections are named plainly "Bravo" and "Charlie" — ordinary horse
  names. Its test therefore proves market-identity whitelisting, not a runner-name
  regex, is the primary classifier (Stage 1 requirement 5).

## Why no commit was created

`scraper/livescorebet.py`, `scraper/boylesports.py`, `scraper/paddy_power.py`
and `utils/normalizer.py` each carry Stage 1 **and** Stage 2 edits;
`models/predictor.py` carries Stage 1 **and** Stage 3 edits. They share import
blocks, the `BotDetectedError`/`ScraperError` definitions, and the `scrape()`
bodies, so `git add <file>` cannot scope a stage. Stage 1's code paths call
Stage 2 symbols (`source_health.record_*`, `next(required=True)`), so a
hunk-level extraction would not even import. Since Stages 2 and 3 are PARTIAL,
committing those files under `fix: validate primary win markets` would ship
unfinished work under a pass label — which the prompt file forbids.

**Decision: land Stages 1+2 as one joint commit once Stage 2 closes**, with a
body stating plainly that the two stages are inseparable in these files.

## Defect found while verifying

Unit tests write the **real** `data/source_health.json`.
`tests/utils/test_source_health.py` monkeypatches `sh._PATH` to `tmp_path`, but
the scraper tests do not, so `source_health.record_*` fired from scraper code
under test hits the live path. Proven by mtime: `07:02:21` → run
`tests/scraper/test_livescorebet.py` (44 passed) → `07:03:58`. The file's
contents (livescorebet 36 successes / 167 failures / status "failing",
`proxy_reachable: null` everywhere) are therefore test residue, **not** a live
health reading — so Stage 2 has no valid evidence for its bounded-health-check
acceptance item.

## Other open items worth remembering

- `data/predictions.json` (`generated_at 2026-07-25T13:21:22`) predates every
  Stage 3 edit: it has **no `runners` key**, and serializes 170 rows against a
  declared `total_runners: 514`. Stage 3's central deliverable exists in code and
  is absent from the shipped cache.
- `scripts/reconcile_prob_to_ev.py` sets `fetched_at = now` and `stale = False`
  by design, so `reports/prob_to_ev_reconciliation.md` (51 races, 514 runners,
  0 invariant violations) evidences the arithmetic identities and **nothing about
  freshness**.
- EV is still computed before any shared race-level complete/fresh contract in
  `models.predictor._score_value` and `models.predict_unified.enrich`.
- `value_win_prob` is market-adjusted, not price-free — `OddsBandCalibrator`
  overwrites the v3nf output using the same offered odds EV uses, with an exact
  zero floor. Consistent with the paper-only verdicts in
  [[calib-fl-04-backtest]] and [[calib-fl-05-final]]; the CLV-negative reading in
  [[v3-lgbm-holdout-verdict]] still stands and was not revalidated here.
- Race identity keyed on `race_time` rather than the per-source `race_id` is
  confirmed correct in `models/predictor.py::_race_uid`, matching
  [[per-source-race-id-fragments-oddsmap]].

## Next action

Run Prompt 2 (Stage 2 of 5) from `RACE_PREDICTOR_REPAIR_PROMPTS.md`, pointing its
dependency gate at the 2026-07-27 section of `HANDOFF.md` as the Stage 1
evidence, and telling it Stage 1 is verified-but-uncommitted by design.
