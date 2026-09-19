# HANDOFF

Rolling handoff log for the five-stage live-odds repair programme defined in
`RACE_PREDICTOR_REPAIR_PROMPTS.md`. Append dated sections; never delete earlier
ones.

---

## 2026-07-27 — Stage 0: triage and checkpoint of in-flight repair work

**Scope of this session: verification and checkpointing only.** No missing
functionality was implemented, no code was changed, and **no stage commit was
created**. Reason for the last point is in _"Why no commit was created"_ below.

### Repository state

| Item                     | Value                                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Branch                   | `chore/config-audit`                                                                                                                                                        |
| HEAD commit              | `6d6efc263f23eff45b6f613b73f60dc632c4cfe1` — _fix(notifications): show friendly Dublin date+time in Telegram alerts_                                                        |
| Working tree             | **Dirty.** 26 tracked files modified, 19 untracked paths. Everything below is uncommitted.                                                                                  |
| Prior handoff            | None. `HANDOFF.md` did not exist before this session.                                                                                                                       |
| Prior stage memory notes | None. `memory/live-repair-01..05*.md` do not exist. Only `memory/live-repair-00-prompt-sequence.md` (the programme index, which explicitly disclaims any stage completion). |
| Prior stage commits      | None. No commit in history carries any of the five stage messages.                                                                                                          |

### Per-stage verdict

| Stage                                  | Verdict                                                                                                            | One-line basis                                                                                                                                                                            |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 — primary WIN-market ingestion       | **DONE** (acceptance 1–6 met) — but **not committable in isolation**, and implementation req 9 has a reporting gap | Adversarial fixtures + 426 targeted tests green; live artifact `data/live_odds.parquet` is 948/948 `VALID`; before/after audit shows LivescoreBet median booksum 3.68 → 1.206             |
| 2 — proxy, freshness, atomic snapshots | **PARTIAL**                                                                                                        | Acceptance 1 partly untested (destination health, mixed-source assembly), acceptance 4 has no valid evidence, acceptance 5 is contradicted by the project's own 2026-07-25 report         |
| 3 — full-field probability and EV      | **PARTIAL**                                                                                                        | Acceptance 4 fails: `data/predictions.json` was never rebuilt from the Stage 3 code path; the reconciliation report was produced on an offline card with freshness deliberately rewritten |
| 4 — calibration / model-validity audit | **NOT STARTED** (blocked)                                                                                          | `reports/calibration_dependency_failure_20260725.md` records an explicit NO-GO; no model, calibrator, preprocessor or training artifact was fitted                                        |
| 5 — realistic paper-betting safeguards | **NOT STARTED**                                                                                                    | No code, tests, reports or artifacts of any kind                                                                                                                                          |

### File → stage map

Three files carry **two stages each** and one carries two as well; this is the
blocker for scoped commits.

**Stage 1 only**

- `utils/market_validation.py` _(new)_ — race-level `VALID`/`INVALID` contract,
  `validate_primary_win_rows` / `annotate_primary_win_rows` /
  `validate_live_dataframe` / `invalid_race_keys`.
- `scripts/audit_live_markets.py` _(new)_ — before/after audit CLI.
- `tests/utils/test_market_validation.py` _(new)_
- `tests/scraper/fixtures/` _(new)_ — `livescorebet_adversarial.json`,
  `paddy_power_adversarial.json`, `boylesports_adversarial.html`,
  `boylesports_live_dom_adversarial.html`.
- `reports/live_market_audit_before_20260725.csv`,
  `reports/live_market_audit_after_20260725.csv` _(new evidence)_

**Stage 2 only**

- `utils/proxy_manager.py` — `ProxyUnavailableError`, `redact_proxy_url`,
  `next(required=…)`, `check_destinations` / `destination_status` /
  `_probe_destination`, redacted `stats()`, runtime config reload.
- `utils/source_health.py` _(new)_ — disk-backed per-source health record.
- `tests/utils/test_proxy_manager.py`, `tests/utils/test_source_health.py`
- `utils/storage/parquet_store.py` + `tests/utils/storage/test_parquet_store.py`
  — atomic `.tmp` + `os.replace` write.
- `config.yaml` — **+7 lines only**: `proxy_pool.destination_urls` (three probe
  URLs) and `staleness.max_age_seconds: 900`.
- `utils/config_loader.py` — **1 line**: whitelist the `staleness` top-level key.
- `ui/refresh_ops.py`, `ui/pages/1_Scrape_and_Refresh.py` — surface
  `stale` / `age_seconds` per scrape.
- `data/source_health.json` _(new, untracked)_ — see the defect note below.

**Stage 3 only**

- `models/value.py` — reference vs executable odds split, `passes_core_gate`,
  typed pick fields, complete-reference-book gate.
- `models/predict_unified.py` — `enrich(..., reference_decimals, executable_decimals)`.
- `models/suggestions.py` — index `race["runners"]` first.
- `scripts/reconcile_prob_to_ev.py` _(new)_
- `tests/models/test_value.py`
- `reports/prob_to_ev_reconciliation.md`, `reports/predictions_reconciliation.json`
- `data/predictions.json` — **modified but stale**; see Stage 3 below.

**Mixed — cannot be split by file**

| File                                                            | Stages present | Evidence                                                                                                                                                                                                                                                                                                                                       |
| --------------------------------------------------------------- | -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scraper/livescorebet.py`                                       | 1 + 2          | Stage 1: `_PRIMARY_WIN_GROUP_ID/…_TYPES/…_NAMES` (L116-118), `annotate_primary_win_rows` (L654, L741). Stage 2: `source_health.record_*` (L228-248, L363-376, L860-906), `next(required=True)` (L349), `redact_proxy_url` (L220). Both stages also edit the same import block (L12-22) and the same `BotDetectedError`/`ScraperError` classes. |
| `scraper/boylesports.py`                                        | 1 + 2          | Stage 1: `_PRIMARY_WIN_MARKET_LABELS` (L61), container-scoped parsing (L369-468). Stage 2: `next(required=True)` (L172, L734), typed `source_health.record_failure` (L750-879).                                                                                                                                                                |
| `scraper/paddy_power.py`                                        | 1 + 2          | Stage 1: `marketType == "WIN"` restriction + `annotate_primary_win_rows` (L309-330). Stage 2: `source_health` + `ProxyUnavailableError` (L143-201, L420-527).                                                                                                                                                                                  |
| `utils/normalizer.py`                                           | 1 + 2          | Stage 1: canonical `market_id`/`market_name`/`selection_id`/`validation_*` columns, `_fail_closed_live()`. Stage 2: `stale` column, per-source snapshot replacement in `_write()`.                                                                                                                                                             |
| `features/fuse.py`                                              | 2 + 3          | Stage 2: `_drop_stale_live_rows()` reading `staleness.max_age_seconds`. Stage 3: `_runner_key()` including `race_time`.                                                                                                                                                                                                                        |
| `models/predictor.py`                                           | 1 + 3          | Stage 1: `_guard_upcoming` fail-closed on non-`VALID`. Stage 3: `runners` collection, `_race_uid`, reference/executable split, `check_output_invariants`.                                                                                                                                                                                      |
| `tests/scraper/test_*.py`, `tests/utils/test_normalizer.py`     | 1 + 2          | mirror their modules                                                                                                                                                                                                                                                                                                                           |
| `tests/models/test_predictor.py`, `tests/features/test_fuse.py` | 1 + 2 + 3      | mirror their modules                                                                                                                                                                                                                                                                                                                           |

**Unrelated to the five stages — do not sweep into any stage commit**

`.claude/`, `.design-md/`, `images/`, `implementation_plan.md`, `review/`,
`v4prompts.md`, `RACE_PREDICTOR_REPAIR_PROMPTS.md` itself,
`memory/live-repair-00-prompt-sequence.md`, `memory/MEMORY.md` (+1 pointer line),
`tools/*` (13–18 June), `tests/test_train_lgbm.py` (18 June),
`reports/backtest_*`, `reports/testrun`, `reports/*_20260614_*`.

### Reconstructed timeline (file mtimes, all 2026-07-25)

```
12:44  reports/live_market_audit_before_20260725.csv
13:02  data/live_odds.parquet          (rebuilt)
13:11  utils/market_validation.py      (Stage 1 core)
13:18  data/unified_races.parquet      (rebuilt)
13:21  data/predictions.json           (rebuilt)   <-- predates Stages 2 and 3
13:24  scripts/audit_live_markets.py + live_market_audit_after_20260725.csv
14:11  utils/source_health.py          (Stage 2 core)
15:54  scripts/reconcile_prob_to_ev.py (Stage 3)
15:56  reports/prob_to_ev_reconciliation.md
17:00  reports/live_market_dependency_check_20260725.csv
17:11  reports/calibration_dependency_failure_20260725.md   (Stage 4 NO-GO)
```

The prior session ran Stage 1, then Stage 2, then Stage 3, then hit a Stage 4
dependency failure and stopped — without ever writing `HANDOFF.md`, the memory
notes, or the stage commits.

### Commands run this session, and results

```powershell
.\.venv\Scripts\python.exe -m pytest -q
# 1351 passed in 121.87s

.\.venv\Scripts\python.exe -m pytest -q `
  tests/scraper/test_boylesports.py tests/scraper/test_livescorebet.py `
  tests/scraper/test_paddy_power.py tests/utils/test_market_validation.py `
  tests/utils/test_proxy_manager.py tests/utils/test_source_health.py `
  tests/utils/test_normalizer.py tests/utils/storage/test_parquet_store.py `
  tests/features/test_fuse.py tests/models/test_predictor.py `
  tests/models/test_value.py tests/test_train_lgbm.py
# 426 passed in 26.41s
```

`--timeout=600` is not available: `pytest-timeout` is not installed in the repo
`.venv`. Use a plain `-q` run.

No test touches the network (`tests/conftest.py` disables notifications).

### Stage 1 — DONE (acceptance met), with two caveats

Acceptance from the prompt file, item by item:

1. **Adversarial fixtures — met.** Four fixtures exist and are exercised by
   `tests/scraper/test_livescorebet.py:189`, `tests/scraper/test_paddy_power.py:314`,
   `tests/scraper/test_boylesports.py:214,233,264`. The LivescoreBet fixture is
   the decisive one: alongside the genuine `"To win"` market (type
   `1001558122`, group `758`) it carries _Best Finishing Position_ pair
   selections, a _Pre-packs_ special, an unknown market type `999999`, a
   _Winning Distance_ market, **and a "Without 1 (Alpha)" market whose
   selections are named plainly "Bravo" and "Charlie"**. Those last two are
   ordinary horse names — so the test proves the whitelist, not a name regex, is
   doing the classification. That is exactly Stage 1 requirement 5.
2. **Regression tests for unknown market IDs, multiple markets, duplicate
   selections, contaminated books, fail-closed downstream — met.** Unknown group
   IDs are quarantined with a warning rather than defaulted to WIN
   (`scraper/livescorebet.py:594`); `utils/market_validation.py` rejects at race
   level with reasons `primary_win_market_count:N`, `duplicate_selection_id`,
   `duplicate_selection_name`, `implausible_field_size:N`,
   `special_selection_in_primary_win`, `extreme_booksum:X`; downstream
   fail-close is covered by `features/fuse.py` and `models/predictor.py`
   `_guard_upcoming` tests.
3. **Targeted tests — met.** 426 passed (command above).
4. **Full pytest — met.** 1351 passed.
5. **Live-data audit before/after — met, and decisive:**

   | Source       | BEFORE races | BEFORE median runners |  BEFORE median booksum | AFTER VALID / INVALID | AFTER median runners | AFTER median booksum |
   | ------------ | -----------: | --------------------: | ---------------------: | --------------------- | -------------------: | -------------------: |
   | livescorebet |           51 |     **30.0** (max 60) | **3.6804** (max 9.363) | 51 / 0                |               **10** |           **1.2058** |
   | boylesports  |           60 |         10.0 (max 25) |     1.2533 (max 2.638) | 46 / 9                |                    9 |               1.2057 |
   | paddy_power  |           48 |         10.5 (max 23) |                 1.2584 | 38 / 1                |                   10 |               1.3184 |

   This reproduces the symptom named in the Stage 1 prompt ("51 races but a
   median 30 selections in markets labelled WIN and a median booksum around
   3.68") and shows it repaired.

6. **Live artifact validated with concrete examples — met.** `data/live_odds.parquet`
   is 948 rows × 21 cols, carrying `market_id`, `market_name`, `selection_id`,
   `validation_status`, `validation_reasons`, `field_size`, `booksum`. Status
   breakdown: `boylesports 434 VALID`, `livescorebet 514 VALID`, **zero INVALID
   rows persisted**. Concrete rejected examples from the after-audit: BoyleSports
   Del Mar 22:00 (8 runners, booksum 0.0) and Gulfstream 17:20 (8 runners,
   booksum 0.0) — priceless US fixtures — all marked `INVALID`.

**Caveat A — implementation requirement 9 is only partly met.**
`scripts/audit_live_markets.py` reports race count, runner-count distribution,
booksum distribution and `VALID`/`INVALID` counts, but its CSV columns are
`source,race_id,venue,race_time,market_id,market_name,validation_status,runners,unique_selections,booksum`
— it does **not** emit `validation_reasons`, and it prints no
suspicious-selection examples. Requirement 9 asked for "rejected markets,
rejection reasons, and suspicious selection examples by source". Adding the
reasons column plus a sample of quarantined selection names is a small,
self-contained follow-up.

**Caveat B — Stage 1 cannot be committed on its own.** See below.

### Stage 2 — PARTIAL

Implemented and genuinely tested:

- Rotating-gateway semantics: repeated failures never blacklist the gateway
  (`test_failures_never_blacklist`, `test_rotating_gateway_never_raises_even_when_required`).
- Proxy-required domains never silently fall back to direct
  (`test_next_never_falls_back_to_direct_on_failures`,
  `test_all_blacklisted_raises_when_required`); the Cloudflare-fronted lxml tier
  and all three scrapers call `next(required=True)`.
- Country targeting (`test_injects_country_token_into_username` and 5 siblings).
- Credential redaction (`test_masks_userinfo`, `test_stats_never_leaks_raw_credentials`).
  Real credentials live only in git-ignored `config.local.yaml`; the committed
  `config.yaml` carries `proxies: []` — note this was **already committed** in
  `948cc53`, so the secrets migration was pre-existing, not new work.
- Runtime reload in the Streamlit process (`test_rebuilds_when_proxy_pool_changes`,
  `test_noop_when_proxy_pool_unchanged` — guarded on `_raw_cfg` equality so an
  unrelated config edit does not discard blacklist state).
- Typed failure reasons: `bot_detected`, `challenge`, `timeout`, `tls_error`,
  `invalid_json`, `empty_card`, `parser_rejected`, `proxy_unavailable`, `other`.
- Atomic writes (`test_interrupted_write_leaves_previous_snapshot_untouched`,
  `test_interrupted_write_on_fresh_path_leaves_no_partial_file`).
- Staleness TTL blocking (`tests/features/test_fuse.py` — flagged-stale,
  unparseable `fetched_at`, and age > `staleness.max_age_seconds` all drop the
  live row; non-live sources are not gated).

**What remains, against Stage 2's own acceptance section:**

1. **Destination-specific health checks are implemented but untested.**
   `check_destinations()`, `destination_status()` and `_probe_destination()`
   exist in `utils/proxy_manager.py` and `config.yaml` now carries
   `proxy_pool.destination_urls` for all three books — but a repo-wide search
   finds **no test** referencing any of them. `source_health.set_proxy_reachable`
   is tested only as a setter. This is acceptance item 1's "destination health".
2. **Mixed-source snapshot assembly (requirement 10) is untested.** No test
   covers "one bookmaker fresh, one stale, one unavailable". Acceptance item 1
   names "mixed-source refreshes" explicitly.
3. **Acceptance item 4 has no valid evidence, and the only artifact is
   polluted.** The sole health artifact is the untracked `data/source_health.json`.
   It is **written by the unit-test suite against the real repository path**,
   proven this session:

   ```
   mtime before  07/27/2026 07:02:21
   .\.venv\Scripts\python.exe -m pytest -q tests/scraper/test_livescorebet.py   # 44 passed
   mtime after   07/27/2026 07:03:58
   ```

   `tests/utils/test_source_health.py` correctly monkeypatches `sh._PATH` to
   `tmp_path`, but the scraper tests do not — so `source_health.record_*` fired
   from scraper code under test writes to the live file. Its current contents
   (`livescorebet` 36 successes / 167 failures / `status: "failing"`,
   `last_row_count: 2`, `paddy_power last_row_count: 0`, `proxy_reachable: null`
   for every source) are **test residue, not a live health reading**. This is a
   real defect on two counts: tests mutate a repo data artifact (against the
   repo's hermetic-test convention), and Stage 2's health telemetry cannot be
   trusted or reported on.

4. **Acceptance item 5 ("prove stale/failed sources cannot create executable EV
   or suggestions") is not proven — and the project's own report says the
   opposite.** `reports/calibration_dependency_failure_20260725.md` records a
   **Fail** row: _"Partial or freshness-unknown snapshot cannot produce any EV"_
   — because `models.predictor._score_value` and `models.predict_unified.enrich`
   compute EV before any shared race-level complete/fresh contract. One narrow
   fix did land (`_odds_by_book_map` now rejects an entire source race on
   missing/invalid `validation_status`, missing or expired `fetched_at`, or an
   explicit `stale` flag) but the general gap is documented as open.
5. Mandatory checkpoint artifacts absent: no `memory/live-repair-02-proxy-freshness.md`,
   no Stage 2 commit.

### Stage 3 — PARTIAL

Implemented: non-runners dropped before normalization; a complete
`race["runners"]` collection with `selections` reduced to display-only;
`_race_uid` keyed on `race_time` in preference to the per-source `race_id`
(consistent with the `per-source-race-id-fragments-oddsmap` memory note);
reference-vs-executable odds separation (`_REF_ODDS_KEYS` deliberately excludes
`best_odds`, so a synthetic best-price board is never de-vigged as one book);
`passes_core_gate` shared by `predictor._build_race` and `models.value`; typed
fields `model_win_prob` / `devigged_market_prob` / `raw_implied_prob` /
`fair_decimal_odds` / `reference_odds` / `executable_odds` / `edge_pp` /
`relative_edge` / `EV`; `check_output_invariants` covering the five invariants;
and `scripts/reconcile_prob_to_ev.py` + `reports/prob_to_ev_reconciliation.md`
(51 races, 514 runners, **0 invariant violations**, every book "complete",
max EV absolute error ≤ 0.0081, 0 value bets).

**What remains, against Stage 3's own acceptance section:**

1. **Acceptance item 4 fails — predictions were never rebuilt from the Stage 3
   code path.** `data/predictions.json` reports
   `generated_at: 2026-07-25T13:21:22+01:00`, which predates every Stage 3 edit
   (`scripts/reconcile_prob_to_ev.py` 15:54, and the `models/*` edits with it).
   Its race objects have keys
   `['venue','race_time','field_size','each_way_available','selections','excluded_low_odds']`
   — **no `runners` key at all** (0 runners across all 51 races), while the
   header declares `total_runners: 514` and only 153 selections + 17
   `excluded_low_odds` = **170 rows are actually serialized**. Stage 3's central
   deliverable (requirement 2, "preserve every valid runner in a full internal/API
   `runners` collection") is present in code and absent from the shipped cache.
2. **The reconciliation report does not prove freshness.**
   `scripts/reconcile_prob_to_ev.py:62-66` deliberately sets
   `card["fetched_at"] = pd.Timestamp.now(tz="UTC")` and `card["stale"] = False`.
   The report is therefore valid evidence for the arithmetic identities and the
   full-field invariants, and **no evidence at all** for acceptance item 4's
   "validated fresh data". `reports/predictions_reconciliation.json` (all 514
   runners) is an offline reconstruction, not the live cache.
3. **Requirement 9's "no EV from stale, contaminated, mismatched, or incomplete
   cards" is architecturally incomplete** — same open Fail row quoted under
   Stage 2 item 4. There is still no single race-level eligibility decision,
   carried into the cache, gating `expected_value`, `ev_catboost`, `ev_lgbm`,
   `value_bet` and suggestions together.
4. Mandatory checkpoint artifacts absent: no `memory/live-repair-03-probability-ev.md`,
   no Stage 3 commit.

### Stage 4 — NOT STARTED (blocked by design)

`reports/calibration_dependency_failure_20260725.md` is an explicit **NO-GO**:
"no model, preprocessor, calibrator, value gate, or training artifact was fit or
changed using the affected data." It also records a naming/semantics finding
worth carrying forward: **`value_win_prob` is market-adjusted, not price-free** —
`OddsBandCalibrator.predict(prob, odds)` overwrites the calibrated v3nf output
using the same offered odds that EV uses, and its per-band isotonic maps have an
exact-zero floor (probabilities 0.005/0.01/0.02 map to 0.0 at odds 2.0–6.0;
3 exact zeros among the 170 serialized rows). The independent probability is not
persisted separately.

The saved LightGBM holdout remains: log loss 1.71698 vs market 1.72846, **Brier
0.08032 vs market 0.07981 (worse)**, **CLV −12.23%**, no bootstrap CI. Per
`CLAUDE.md`, that is paper-only and never a green light to bet.

### Stage 5 — NOT STARTED

No code, tests, reports, config or artifacts.

### Why no commit was created

Stage 1 is the only stage whose acceptance gates pass, so it is the only
candidate for a commit. It cannot be scoped:

- `scraper/livescorebet.py`, `scraper/boylesports.py`, `scraper/paddy_power.py`
  and `utils/normalizer.py` each contain Stage 1 **and** Stage 2 edits, and
  `models/predictor.py` contains Stage 1 **and** Stage 3 edits. The two stages
  share the same import blocks and, in the scrapers, the same
  `BotDetectedError` / `ScraperError` class definitions and the same
  `scrape()` bodies. `git add <file>` cannot separate them.
- Stage 2 and Stage 3 are **PARTIAL**. Committing those files under the message
  `fix: validate primary win markets` would ship unfinished Stage 2/3 work under
  a Stage 1 pass label. The prompt file forbids exactly this: _"Never include
  unrelated pre-existing changes. If clean scoping is not possible, do not
  commit; document why and provide the exact safe next step."_

Hunk-level surgery to extract Stage 1 alone was considered and rejected: the
resulting commit would not compile or pass tests on its own (the Stage 1 code
paths in the scrapers call `source_health.record_*` and `next(required=True)`,
which are Stage 2 symbols), so it would be a broken commit, not a checkpoint.

The working tree is therefore left intact and unmodified by this session.

### Exact safe next steps

The dependency order in the prompt file is Stage 2 → Stage 3 → Stage 4 → Stage 5,
and Stage 2's dependency gate demands Stage 1 evidence. Stage 1's evidence is
real; only its paperwork and its commit are missing, and the commit is blocked by
Stage 2 being unfinished in the same files. So:

1. **Run Prompt 2 next** (`RACE_PREDICTOR_REPAIR_PROMPTS.md`, Stage 2 of 5 —
   Claude Sonnet 5 · `high`). Point it at this section as the Stage 1 evidence
   its dependency gate requires, and tell it Stage 1 is verified-but-uncommitted
   for the reason above.
2. Stage 2 must close, at minimum:
   - tests for `check_destinations` / `destination_status` / `_probe_destination`
     and for `set_proxy_reachable` being driven by a real probe;
   - a mixed-source assembly test (fresh + stale + unavailable);
   - **isolation of `utils.source_health._PATH` in the test suite** — the scraper
     tests currently write the repo's real `data/source_health.json`. Either
     autouse-fixture the path in `tests/conftest.py` or make `_PATH` resolve
     through a config/env hook. Then delete the polluted
     `data/source_health.json` and regenerate it from a real bounded refresh.
   - a bounded per-source live health check, reported per source, with skipped
     network validation reported as skipped and never as a pass.
3. **When Stage 2 passes, make one commit covering Stages 1+2 together** — they
   are inseparable in these files — and say so plainly in the commit body rather
   than pretending to a single-stage scope. Suggested message:
   `fix: validate primary win markets and harden proxy and odds freshness`,
   with a body noting it lands Stage 1 and Stage 2 jointly because
   `scraper/*.py` and `utils/normalizer.py` interleave both. Write
   `memory/live-repair-01-market-ingestion.md` and
   `memory/live-repair-02-proxy-freshness.md` at that point, and record the
   resulting hash back into this file.
4. Only then run Prompt 3. Stage 3's first two jobs are the two open items above:
   rebuild `data/predictions.json` from the current code so it actually carries
   `runners` (514, not 170), and replace the freshness-rewriting reconciliation
   with one run against genuinely fresh validated data.
5. Stage 4 stays blocked until Stage 3's acceptance item 4 passes. Do not re-use
   the 2026-07-25 head-to-head figures as a new verdict.

### Unresolved issues carried forward

| #   | Issue                                                                                            | Owner stage |
| --- | ------------------------------------------------------------------------------------------------ | ----------- |
| 1   | Unit tests write the real `data/source_health.json`; health telemetry is test residue            | 2           |
| 2   | `check_destinations` / destination probing has no test coverage                                  | 2           |
| 3   | No mixed-source (fresh/stale/unavailable) snapshot assembly test                                 | 2           |
| 4   | `data/predictions.json` has no `runners` key; 170 of 514 runners serialized                      | 3           |
| 5   | `scripts/reconcile_prob_to_ev.py` rewrites `fetched_at`/`stale`, so it cannot evidence freshness | 3           |
| 6   | EV is computed before any shared race-level complete/fresh contract (`_score_value`, `enrich`)   | 2/3         |
| 7   | `audit_live_markets.py` omits `validation_reasons` and suspicious-selection examples             | 1           |
| 8   | `value_win_prob` is market-adjusted, not price-free; independent prob not persisted              | 4           |
| 9   | `pytest-timeout` is not installed in the repo `.venv`                                            | infra       |

### Stage 0 verdict

**STAGE 0 COMPLETE — verification only.** Stage 1 DONE (uncommitted, not
separable), Stages 2 and 3 PARTIAL, Stages 4 and 5 NOT STARTED. Nothing was
committed; the working tree is unchanged from how this session found it.

---

## 2026-07-27 — Stage 1: primary WIN-market ingestion repair

**Status: STAGE 1 PASS.** All nine implementation requirements and all six
acceptance gates are met. **No commit was created** — the sanctioned
"clean scoping is not possible" path; see _"Commit: deliberately not created"_.

### What this session found (independent re-verification)

The Stage 0 section above was treated as a claim, not as fact. Every Stage 1
assertion in it was re-checked against code, data and a live test run. All of
them held, with one exception already flagged there: implementation
requirement 9 was genuinely incomplete. That gap is what this session closed.

Re-verified independently, not inherited:

- `utils/market_validation.py` implements the race-level contract with the
  reason codes claimed (`primary_win_market_count:N`, `duplicate_selection_id`,
  `duplicate_selection_name`, `implausible_field_size:N`,
  `special_selection_in_primary_win`, `extreme_booksum:X`, `missing_market_id`,
  `missing_selection_id`, `non_primary_win_market_type`).
- LivescoreBet whitelists group `758` + type `1001558122` + name `to win`
  (`scraper/livescorebet.py:116-118, 588-599`); an unknown market inside the
  primary group is logged and quarantined, never defaulted to WIN.
- BoyleSports scopes parsing to `#racingNavMiniMenu` (active primary market) +
  `#RacingMarketSelections` (its runner container), so price boosts and other
  specials under `#RacingMarketRefresh` cannot participate
  (`scraper/boylesports.py:451-480`).
- Paddy Power restricts to `marketType == "WIN"` and annotates
  (`scraper/paddy_power.py:246, 309`).
- `market_id` / `market_name` / `selection_id` / `race_id` / `fetched_at` /
  `validation_status` / `validation_reasons` survive normalization and storage
  (`utils/normalizer.py:28-40, 105-107`), and are present in the live parquet.
- Fail-closed downstream is wired at `features/fuse.py:53-62`,
  `models/predictor.py:712-734, 976-987`.

**Decisive fixture evidence (re-run this session).** Feeding
`tests/scraper/fixtures/livescorebet_adversarial.json` through the real
`_parse_event` — 6 markets, 12 selections offered:

```
'Best Finishing Position'  -> 'Alpha (Alpha - Bravo)', 'Bravo (Alpha - Bravo)'
'Pre-packs'                -> 'Alpha & Bravo Both To Finish In The Top 3'
'Mystery Offer' type 999999-> 'Betting Without Alpha - Bravo To Win'
'To win'      (GENUINE)    -> 'Alpha', 'Bravo', 'Charlie'
'Winning Distance'         -> 'Alpha by 2 Lengths or more'
'Without 1 (Alpha)'        -> 'Bravo', 'Charlie'      <-- ordinary horse names

SURVIVING ROWS: 3   Alpha / Bravo / Charlie, market 'To win', id SBTM_WIN_1, VALID
```

The "Without 1 (Alpha)" market matters most: its selections are named plainly
"Bravo" and "Charlie" — byte-identical to genuine runners in the real WIN
market — and they are still excluded. That proves **market identity**, not a
runner-name regex, is the primary classifier (requirement 5).

### Files changed this session

| File                                                   | Change                                                                                                                              |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/audit_live_markets.py` _(rewritten)_          | Closes requirement 9: rejection reasons, reason-family rollup, suspicious-selection examples, runner/booksum quantile distributions |
| `utils/market_validation.py`                           | +`parse_validation_reasons()`, +`reason_family()` public helpers; extended the secondary regex to catch `"<horse> by N Lengths"`    |
| `tests/test_audit_live_markets.py` _(new)_             | 12 tests pinning the new reporting                                                                                                  |
| `tests/utils/test_market_validation.py`                | +2 tests: winning-distance selections flagged; ordinary names not flagged                                                           |
| `reports/live_market_audit_after_20260727.csv` _(new)_ | After-audit including the `validation_reasons` column                                                                               |

No schema or config change. No data deleted or quarantined this session — the
clean live snapshot rebuilt on 2026-07-25 was validated as-is (requirement 8).

### Requirement 9 — before and after

The audit previously emitted only
`source,race_id,venue,race_time,market_id,market_name,validation_status,runners,unique_selections,booksum`.
It now additionally emits `validation_reasons` and prints four new sections.
Actual output against the live artifact:

```
REJECTED MARKETS / RACES
boylesports 45883320.10 Gulfstream   17:20 INVALID extreme_booksum:none  8 runners booksum 0.0
boylesports 45883309.10 Del Mar      22:00 INVALID extreme_booksum:none  8 runners booksum 0.0
boylesports 45883386.10 Prairie Medw 23:00 INVALID extreme_booksum:none  8 runners booksum 0.0
... (9 boylesports, 1 paddy_power)
paddy_power 35859454.1158 ENGHIEN    12:58 INVALID extreme_booksum:none 13 runners booksum 0.0

REJECTION REASONS BY SOURCE
boylesports extreme_booksum 9
paddy_power extreme_booksum 1

SUSPICIOUS SELECTION EXAMPLES (max 10/source)
  none found in retained rows
```

All ten rejections are priceless overseas fixtures (Del Mar, Gulfstream,
Prairie Meadows, Enghien) carrying `booksum 0.0` — correctly failed closed.
"None found" for suspicious selections is the correct reading for clean data;
because "no output" is weak evidence, `tests/test_audit_live_markets.py` proves
the section actually fires on planted specials, non-WIN rows, and caps per
source.

**Coverage limit, stated plainly.** Markets rejected at _parse_ time (an unknown
LivescoreBet `marketGroupId`, a non-primary BoyleSports container) never reach a
cache or the parquet, so they are visible only in scraper logs — the audit
cannot count them. What it does evidence is every race rejected by the
race-level contract, plus any special that survived into a retained artifact.
This is documented in the script docstring.

### Secondary-guard fix

The regex missed winning-distance selections phrased `"<horse> by 2 Lengths or
more"` — the exact shape in the LivescoreBet fixture. Market identity already
excluded them, so this was never a live contamination hole, but the secondary
quarantine guard should catch them. Added
`\bby\s+\d[\d.\s+\-–]*lengths?\b`.

False-positive check on real data: **0 of 741 distinct live runner names** are
flagged. A regression test pins non-matching of `Lengthsman`, `Two Lengths
Clear`, `Without A Doubt`, `Finish Line`.

### Commands run, and results

```powershell
# Baseline before any edit
.\.venv\Scripts\python.exe -m pytest -q tests/scraper/test_boylesports.py `
  tests/scraper/test_livescorebet.py tests/scraper/test_paddy_power.py `
  tests/utils/test_market_validation.py tests/utils/test_normalizer.py `
  tests/utils/storage/test_parquet_store.py tests/features/test_fuse.py `
  tests/models/test_predictor.py
# 296 passed in 23.05s

# Final targeted (adds the new audit tests + schema/e2e)
.\.venv\Scripts\python.exe -m pytest -q tests/scraper/test_boylesports.py `
  tests/scraper/test_livescorebet.py tests/scraper/test_paddy_power.py `
  tests/utils/test_market_validation.py tests/test_audit_live_markets.py `
  tests/utils/test_normalizer.py tests/utils/storage/test_parquet_store.py `
  tests/features/test_fuse.py tests/models/test_predictor.py `
  tests/test_predictions_schema.py tests/test_smoke_end_to_end.py
# 321 passed in 27.36s

# Full suite
.\.venv\Scripts\python.exe -m pytest -q
# 1365 passed in 96.08s      (Stage 0 recorded 1351; +14 new tests)

# Audit
.\.venv\Scripts\python.exe -m scripts.audit_live_markets --no-races `
  --output-csv reports/live_market_audit_after_20260727.csv
```

### Before / after audit figures

BEFORE is recomputed from `reports/live_market_audit_before_20260725.csv`
(the pre-repair capture), AFTER from the current artifacts.

| Source       | BEFORE races | BEFORE median runners (max) | BEFORE median booksum (max) | AFTER VALID / INVALID | AFTER median runners | AFTER median booksum |
| ------------ | -----------: | --------------------------: | --------------------------: | --------------------- | -------------------: | -------------------: |
| livescorebet |           51 |               **30.0** (60) |     **3.6804** (**9.3627**) | **51 / 0**            |             **10.0** |           **1.2058** |
| boylesports  |           60 |                   10.0 (25) |             1.2533 (2.6384) | 46 / 9                |                  9.0 |               1.2057 |
| paddy_power  |           48 |                   10.5 (23) |             1.2584 (1.4309) | 38 / 1                |                 10.0 |               1.3184 |

LivescoreBet is the headline: a median booksum of **3.68** (and a race at
**9.36**) is arithmetically impossible for a real win book — it is three to nine
books stacked on top of each other. It now sits at **1.206**, a ~20.6% overround,
which is an ordinary UK/IRE win market. Median field drops 30 → 10.

New this session, the full VALID distributions (not just medians):

```
RUNNER-COUNT   count   mean   min  10%  25%  50%   75%   90%   max
boylesports     46.0   9.43   4.0  5.0  7.0   9.0 13.00  14.0  16.0
livescorebet    51.0  10.08   4.0  5.0  7.0  10.0 13.00  15.0  21.0
paddy_power     38.0  11.03   5.0  7.0  8.0  10.0 13.75  16.0  21.0

BOOKSUM        count   mean      min      10%      25%      50%      75%      90%      max
boylesports     46.0  1.2243  1.07320  1.11739  1.16059  1.20566  1.29193  1.35038  1.47425
livescorebet    51.0  1.2290  1.08511  1.13510  1.16291  1.20584  1.28331  1.37636  1.46936
paddy_power     38.0  1.2910  1.09860  1.15028  1.20296  1.31844  1.37001  1.38674  1.46425
```

Every source now sits in a plausible 1.07–1.47 band across the whole
distribution, not merely at the median.

### Live artifact validation (acceptance 6)

`data/live_odds.parquet` — 948 rows × 21 cols, carrying `market_id`,
`market_name`, `selection_id`, `validation_status`, `validation_reasons`,
`field_size`, `booksum`. Breakdown: `boylesports 434 VALID`,
`livescorebet 514 VALID`, **zero INVALID rows persisted**.

Fail-closed proven on the shipped artifact, not only in unit tests
(timezone-aware join, parquet is UTC and predictions are Dublin):

```
predicted races: 51
predicted races with NO VALID live counterpart:   0
INVALID races that still produced a prediction:   0
specials leaked into predictions.json:            0   (170 serialized selections)
```

- **Clean example:** Ascot 13:10 (+01:00) — LivescoreBet market `To win`,
  10 runners, unique selection IDs, booksum 1.19, `VALID`, predicted.
- **Contaminated example:** BoyleSports Del Mar 22:00 — 8 runners but every
  price absent, booksum 0.0, `INVALID` with reason `extreme_booksum:none`,
  and it produces no prediction, EV, suggestion or bet.

### Commit: deliberately not created

The Stage 1 prompt says: _"If clean scoping is not possible, do not commit;
document why and provide the exact safe next step."_ That is the case here, and
I verified it independently rather than inheriting the Stage 0 conclusion. Hunk
counts in `git diff -U0`:

| File                      | Stage 1 hunk hits | Stage 2/3 hunk hits |
| ------------------------- | ----------------: | ------------------: |
| `scraper/livescorebet.py` |                13 |                  20 |
| `scraper/boylesports.py`  |                 9 |                  25 |
| `scraper/paddy_power.py`  |                 6 |                  26 |
| `utils/normalizer.py`     |                 7 |                   7 |
| `models/predictor.py`     |                 6 |                   6 |

The scraper contracts _are_ the heart of Stage 1, and each of those files is
majority Stage 2 work by hunk count. `git add <file>` would ship **PARTIAL**
Stage 2/3 work under the message `fix: validate primary win markets`. Hunk-level
extraction was rejected again for the reason Stage 0 gave: the Stage 1 code
paths in the scrapers call Stage 2 symbols (`source_health.record_*`,
`next(required=True)`) and share the same import block, so the extracted commit
would not import, let alone pass tests.

**However — this session's own increment is cleanly scopable.** Every file I
touched is Stage-1-only and additive:

```
utils/market_validation.py            (untracked, new)
scripts/audit_live_markets.py         (untracked, new)
tests/utils/test_market_validation.py (untracked, new)
tests/test_audit_live_markets.py      (untracked, new)
tests/scraper/fixtures/               (untracked, new)
reports/live_market_audit_after_20260727.csv
```

A commit of exactly those paths would be self-consistent and green (they import
only `pandas` and `utils.text_norm`; nothing in them touches Stage 2 symbols).
I did **not** create it, because the message `fix: validate primary win markets`
would then name a fix whose scraper integration is not in the commit — the kind
of overstated claim `CLAUDE.md` forbids. **This is a judgement call the user
should make**; the exact command is in the next steps below.

### Git state

- HEAD: `6d6efc263f23eff45b6f613b73f60dc632c4cfe1` — unchanged by this session.
- **All Stage 1 work remains uncommitted.** Modified/untracked paths carrying it:
  `scraper/livescorebet.py`, `scraper/boylesports.py`, `scraper/paddy_power.py`,
  `utils/normalizer.py`, `utils/market_validation.py`, `features/fuse.py`,
  `models/predictor.py`, `scripts/audit_live_markets.py`,
  `tests/scraper/test_*.py`, `tests/utils/test_market_validation.py`,
  `tests/utils/test_normalizer.py`, `tests/test_audit_live_markets.py`,
  `tests/features/test_fuse.py`, `tests/models/test_predictor.py`,
  `tests/scraper/fixtures/`, `reports/live_market_audit_*.csv`.

### Unresolved issues carried forward

Stage 0's list, with #7 now closed:

| #     | Issue                                                                                            | Owner stage |
| ----- | ------------------------------------------------------------------------------------------------ | ----------- |
| 1     | Unit tests write the real `data/source_health.json`; health telemetry is test residue            | 2           |
| 2     | `check_destinations` / destination probing has no test coverage                                  | 2           |
| 3     | No mixed-source (fresh/stale/unavailable) snapshot assembly test                                 | 2           |
| 4     | `data/predictions.json` has no `runners` key; 170 of 514 runners serialized                      | 3           |
| 5     | `scripts/reconcile_prob_to_ev.py` rewrites `fetched_at`/`stale`, so it cannot evidence freshness | 3           |
| 6     | EV is computed before any shared race-level complete/fresh contract                              | 2/3         |
| ~~7~~ | ~~audit omits `validation_reasons` and suspicious-selection examples~~ — **CLOSED this session** | ~~1~~       |
| 8     | `value_win_prob` is market-adjusted, not price-free; independent prob not persisted              | 4           |
| 9     | `pytest-timeout` is not installed in the repo `.venv`                                            | infra       |

New, Stage 1 scope, non-blocking:

| #   | Issue                                                                                                           |
| --- | --------------------------------------------------------------------------------------------------------------- |
| 10  | Parse-time market quarantines are log-only; no counter is persisted, so the audit cannot report them            |
| 11  | Paddy Power contributes no rows to `data/live_odds.parquet` (audit reads it from its cache) — verify in Stage 2 |

### Is Stage 2 safe to start?

**Yes.** Stage 1's contract is implemented, tested and validated on the live
artifact; Stage 2's dependency gate should point at this section. Two warnings
for whoever runs it:

1. Stage 2 edits the same four files as Stage 1. Do not attempt to "clean up"
   Stage 1 hunks while there.
2. Fix issue #1 (`utils.source_health._PATH` isolation) **before** trusting any
   health output — the current `data/source_health.json` is test residue.

### Stage 1 verdict

**STAGE 1 PASS** — nine implementation requirements met, six acceptance gates
met, 1365/1365 tests green, live artifact validated with concrete clean and
contaminated examples. The stage commit was deliberately not created under the
prompt's own "clean scoping is not possible" clause.

---

## 2026-07-27 — Stage 2: proxy, freshness, and atomic snapshot repair

**Status: STAGE 2 PASS.** All ten implementation requirements and all four
acceptance gates are met. Dependency gate satisfied by the Stage 1 section
above (verified DONE, uncommitted-but-inseparable). **A joint Stage 1+2 commit
was created this session** — see _"Commit"_ below for the hash and the
message deviation this required.

### Requirement-by-requirement status

| #   | Requirement                                                                                         | Status                                                                 | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| --- | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Rotating-gateway semantics + country targeting                                                      | DONE (pre-existing, re-verified)                                       | `test_failures_never_blacklist`, `test_rotating_gateway_never_raises_even_when_required`, `test_injects_country_token_into_username` + 5 siblings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 2   | Never silently fall back to direct for proxy-required domains                                       | DONE (pre-existing, re-verified)                                       | `ProxyUnavailableError`; `test_next_never_falls_back_to_direct_on_failures`, `test_all_blacklisted_raises_when_required`; all three scrapers call `next(required=True)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| 3   | Destination-specific health checks                                                                  | DONE — implemented, tested, **and wired into production this session** | `check_destinations()`/`destination_status()`/`_probe_destination()` in `utils/proxy_manager.py`; `tests/utils/test_proxy_manager.py::TestDestinationHealth` (new); `scripts/refresh.py::_check_destination_health()` (new) now calls it after every scrape and records the result via `source_health.set_proxy_reachable` — closes the gap where `proxy_reachable` was `null` forever (Stage 0 unresolved-issue log)                                                                                                                                                                                                                                                                                     |
| 4   | Distinguish failure types                                                                           | DONE (pre-existing, re-verified)                                       | Typed reasons: `bot_detected`, `challenge`, `timeout`, `tls_error`, `invalid_json`, `empty_card`, `parser_rejected`, `proxy_unavailable`, `other`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| 5   | Track per-source telemetry                                                                          | DONE — tests de-polluted, CLI visibility added                         | `tests/conftest.py::_isolate_source_health` (new autouse fixture) stops any test from writing the real `data/source_health.json`; polluted file deleted and regenerated from a genuine bounded probe (below); `scripts/refresh.py::_print_source_health()` (new) gives the cron/CLI path the same telemetry visibility the Streamlit page already had                                                                                                                                                                                                                                                                                                                                                     |
| 6   | Safe runtime config reload for the proxy singleton                                                  | DONE (pre-existing, re-verified, no changes needed)                    | `utils/config_loader.py` `register_callback`/`start_watching`/rollback-on-invalid; `utils/proxy_manager.py::_on_config_change`; `test_rebuilds_when_proxy_pool_changes`, `test_noop_when_proxy_pool_unchanged`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| 7   | Never log/display proxy credentials; redact everywhere; migrate to `config.local.yaml`/env          | DONE — plus a defense-in-depth addition and an incident (see below)    | `redact_proxy_url` (pre-existing) masks a single proxy URL; new `redact_secrets(text)` masks any `scheme://user:pass@host` found anywhere in free text, for exception messages that embed a raw proxy URL in their own `str()`. Wired into `scripts/refresh.py` and both exception handlers in `ui/refresh_ops.py`. Secrets migration to `config.local.yaml` (git-ignored, confirmed never committed) was already done in `948cc53`, pre-existing.                                                                                                                                                                                                                                                        |
| 8   | Propagate stale/age_seconds/fetched_at/source health; block EV/suggestions above a configurable TTL | DONE, proven end-to-end this session                                   | `features/fuse.py::_drop_stale_live_rows` gates on `staleness.max_age_seconds`; `models/predictor.py::_odds_by_book_map` rejects an entire source-race group on missing/invalid `validation_status`, missing/expired `fetched_at`, or an explicit `stale` flag — new test `test_stale_price_cannot_reach_value_bet_or_suggestion_end_to_end` (below)                                                                                                                                                                                                                                                                                                                                                      |
| 9   | Atomic per-source updates — no partial-failure mixing or silent erasure                             | DONE — a real gap fixed this session                                   | Non-partitioned parquet writes (`.tmp` + `os.replace`) were already atomic. The **partitioned** writer (`df.to_parquet(path, partition_cols=...)`, used by `utils/normalizer.py`'s year-partitioned unified dataset) was not: it `rmtree`'d the live directory before writing the replacement, so a crash mid-write left `data/unified_races.parquet` empty or half-written. Fixed with `write_partitioned_parquet()` + `_atomic_replace_dir()` (write to sibling `.tmp` dir, two-rename swap, `.bak` rollback-on-failure) in `utils/storage/parquet_store.py`, consumed by `utils/normalizer.py::_write()`. 5 new crash-injection tests prove the previous snapshot survives a simulated kill mid-write. |
| 10  | Define/test mixed multi-source snapshot assembly                                                    | DONE                                                                   | New test covering one fresh book + one stale book + one source producing no rows for a race, proving the assembled snapshot keeps only the fresh, valid book and never silently fills the gap with a stale or missing price                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

### Requirement 8 — the decisive end-to-end proof

`tests/models/test_predictor.py::test_stale_price_cannot_reach_value_bet_or_suggestion_end_to_end`
builds one runner with two source prices — BoyleSports 50.0 flagged `stale=True`,
Paddy Power 3.0 fresh — and walks it through the real, unmocked chain:
`_odds_by_book_map` → `_attach_odds_by_book` → `_effective_decimal` →
`Predictor._build_race` → `check_output_invariants` → `find_value_bets` →
`suggest_bets`. Result: the stale 50.0 never enters `odds_by_book`, `best_odds`
resolves to the fresh 3.0, `expected_value` is correctly negative, `value_bet`
is `False`, zero invariant violations, zero value-bet picks, zero suggestions.

**Architectural finding recorded in the test docstring, not a bug:** reading
`models/value.py::_runners_from_dicts`/`_extract_runners` shows
`find_value_bets`'s own belt-and-suspenders `too_stale` gate never fires on the
predictor→UI→suggestions path, because `_runner_dict`
(`models/predictor.py`) does not propagate `stale`/`fetched_at` into the
race-dict runner objects it builds. That gate is not dead by accident — it is
still live and load-bearing on any other caller that constructs a runner dict
with those fields directly. On this specific path, protection is entirely
upstream, in `_odds_by_book_map`'s whole-(source,race)-group rejection.
Documented so a future refactor does not "fix" the missing propagation and
accidentally double-gate, or worse, assume the upstream gate implies the
downstream one is unreachable everywhere.

### Bounded live health check (real network, no proxy credential exposed)

```
ProxyManager.check_destinations() against config.yaml's proxy_pool.destination_urls:
  livescorebet   reachable=True
  paddy_power    reachable=True
  boylesports    reachable=False
```

BoyleSports returning unreachable **through the pool** was cross-checked with a
direct (no-proxy) `httpx` GET, which returned **403** — i.e. BoyleSports is
actively blocking non-proxied traffic. This is the correct, expected outcome
for a "proxy-required domain that must never silently fall back to direct"
(requirement 2): the destination check reports it down rather than the pipeline
quietly serving stale or missing BoyleSports data as if it were healthy.

Ran for real (not mocked) via the new `scripts/refresh.py::_check_destination_health()`
wiring, which then wrote a genuine `data/source_health.json`:

```json
"livescorebet": { "proxy_reachable": true,  ... },
"boylesports":  { "proxy_reachable": false, ... },
"paddy_power":  { "proxy_reachable": true,  ... }
```

No credentials appear in this file (`proxy_reachable` is the only new field);
confirmed by reading it back after the run.

### data/source_health.json — pollution fixed, not just noted

Stage 0/1 both flagged that the unit-test suite was writing the **real**
`data/source_health.json` because only `tests/utils/test_source_health.py`
monkeypatched `source_health._PATH`, while scraper tests that transitively call
`record_success`/`record_failure` did not. Fixed with a new **autouse**
fixture, `tests/conftest.py::_isolate_source_health`, which points `_PATH` at a
fresh `tmp_path` for every single test in the suite — no test file needs to opt
in individually. The previously-polluted file was deleted; the copy now on disk
was produced exclusively by the genuine bounded health check above, run outside
pytest.

### Security incident this session — credential briefly exposed in transcript, not in the repo

While diagnosing why `check_destinations()` behaved a particular way, a
diagnostic shell command against the git-ignored `config.local.yaml` echoed
the full DataImpulse proxy URL — including its embedded username and
password — into this session's tool-call output. **The credential itself is
not reproduced anywhere in this file, in memory, in test code, or in the
commit that follows; it must not be.**

Verified and mitigated:

- `config.local.yaml` is git-ignored (`.gitignore:4:*.local.yaml`) and
  `git log --all --oneline -- config.local.yaml` returns nothing — the file
  has **never** been committed. The exposure is confined to this session's
  local transcript/logs, not the repository or its history.
- Added `redact_secrets()` (requirement 7, above) as defense-in-depth so a
  future raw exception string containing a proxy URL is masked before it
  reaches a log, the CLI, or the UI.
- **Action required from the operator:** rotate the exposed DataImpulse
  credential now, since it must be treated as no longer confidential. This
  is a manual step in the DataImpulse dashboard — not something this session
  can or should do.

### Files changed this session (Stage 2 increment, on top of the already-partial Stage 2 work Stage 0 found)

| File                                        | Change                                                                                                                   |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `tests/conftest.py`                         | + autouse `_isolate_source_health` fixture                                                                               |
| `tests/utils/test_proxy_manager.py`         | + `TestDestinationHealth` (check_destinations/destination_status/\_probe_destination), + `TestRedactSecrets` (4 tests)   |
| `tests/utils/test_source_health.py` _(new)_ | Direct unit coverage of the `source_health` module                                                                       |
| `tests/models/test_predictor.py`            | + `test_stale_price_cannot_reach_value_bet_or_suggestion_end_to_end` + mixed-source snapshot assembly test               |
| `utils/storage/parquet_store.py`            | + `write_partitioned_parquet()`, + `_atomic_replace_dir()`; `write_parquet(partition_by=...)` now delegates to it        |
| `utils/normalizer.py`                       | `_write()`'s year-partitioned branch now goes through `write_partitioned_parquet` instead of `rmtree` + raw `to_parquet` |
| `tests/utils/storage/test_parquet_store.py` | + 4 crash-injection tests for the partitioned writer                                                                     |
| `tests/utils/test_normalizer.py`            | + 1 crash-injection test (`test_write_interrupted_leaves_prior_unified_dataset_untouched`)                               |
| `utils/proxy_manager.py`                    | + `redact_secrets(text)`                                                                                                 |
| `scripts/refresh.py`                        | + `_check_destination_health()`, + `_print_source_health()`, exception path now uses `redact_secrets`                    |
| `ui/refresh_ops.py`                         | Both catch-all exception handlers now use `redact_secrets`                                                               |
| `tests/test_refresh.py` _(new)_             | 3 tests for the two new `scripts/refresh.py` functions, mocked (no network)                                              |

### Commands run this session, and results

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/utils/test_proxy_manager.py `
  tests/utils/test_source_health.py tests/utils/storage/test_parquet_store.py `
  tests/utils/test_normalizer.py tests/models/test_predictor.py `
  tests/models/test_value.py tests/models/test_suggestions.py `
  tests/features/test_fuse.py tests/test_refresh.py
# 351 passed in 16.60s

.\.venv\Scripts\python.exe -m pytest -q
# 1396 passed in 100.84s      (Stage 1 recorded 1365; +31 new tests this session)
```

No test touches the network — `tests/conftest.py` disables notifications and
now also isolates `source_health._PATH`. The bounded live health check above
was run as a standalone script invocation outside pytest, by design.

### Acceptance gates

1. **Proxy/cache/refresh regression tests** — met (351 targeted, table above).
2. **Full pytest suite** — met, 1396/1396 green.
3. **Bounded health check / live refresh where network permits** — met; real
   `check_destinations()` run against all three bookmaker URLs, cross-checked
   the one failure (BoyleSports) with a direct no-proxy request to confirm it
   is a genuine 403, not a proxy-manager bug.
4. **Prove stale/failed sources cannot create executable EV or suggestions** —
   met, via the end-to-end test described above plus the pre-existing
   `TestOddsByBook` unit coverage of each individual layer.

### Unresolved issues — updated

Stage 1's carried-forward table, with Stage-2-owned items closed this session:

| #     | Issue                                                                                      | Owner stage | Status                                                                                                                                             |
| ----- | ------------------------------------------------------------------------------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| ~~1~~ | ~~Unit tests write the real `data/source_health.json`~~                                    | ~~2~~       | **CLOSED** — autouse fixture                                                                                                                       |
| ~~2~~ | ~~`check_destinations` / destination probing has no test coverage~~                        | ~~2~~       | **CLOSED**                                                                                                                                         |
| ~~3~~ | ~~No mixed-source (fresh/stale/unavailable) snapshot assembly test~~                       | ~~2~~       | **CLOSED**                                                                                                                                         |
| 4     | `data/predictions.json` has no `runners` key; 170 of 514 runners serialized                | 3           | Open — Stage 3 scope                                                                                                                               |
| 5     | `scripts/reconcile_prob_to_ev.py` rewrites `fetched_at`/`stale`, cannot evidence freshness | 3           | Open — Stage 3 scope                                                                                                                               |
| ~~6~~ | ~~EV computed before any shared race-level complete/fresh contract~~                       | ~~2/3~~     | **CLOSED for Stage 2's half** — end-to-end proof above. Stage 3's half (whether the _shipped cache_ reflects this) is still gated by issues #4/#5. |
| 8     | `value_win_prob` is market-adjusted, not price-free; independent prob not persisted        | 4           | Open                                                                                                                                               |
| 9     | `pytest-timeout` not installed in the repo `.venv`                                         | infra       | Open, non-blocking                                                                                                                                 |

New this session, found and closed within Stage 2:

| #   | Issue                                                                                                                                                                              | Status                                       |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| 12  | Partitioned parquet writes (`utils/normalizer.py` unified dataset) were not crash-safe — the only atomic-write gap in the storage layer                                            | **CLOSED**                                   |
| 13  | Catch-all exception handlers in `scripts/refresh.py`/`ui/refresh_ops.py` could leak a raw proxy URL via `str(exc)`                                                                 | **CLOSED** — `redact_secrets`                |
| 14  | `check_destinations()`/`set_proxy_reachable()` existed and were tested in isolation but were never called from any production code path — `proxy_reachable` was permanently `null` | **CLOSED** — wired into `scripts/refresh.py` |
| 15  | The CLI/cron refresh path had zero source-health visibility, unlike the Streamlit page                                                                                             | **CLOSED** — `_print_source_health()`        |

Carried forward, not in Stage 2's scope: Stage 1's issue #11 (Paddy Power
contributing no rows to `data/live_odds.parquet`, audit reads it from its
cache) was not investigated this session — still open, Stage 2/3 boundary.

### Commit

Per the plan of record this file's own Stage 1 section set out
(_"Exact safe next steps"_, item 3): Stage 1 and Stage 2 touch the same four
files (`scraper/livescorebet.py`, `scraper/boylesports.py`,
`scraper/paddy_power.py`, `utils/normalizer.py`) inseparably — Stage 1's own
code paths call Stage 2 symbols (`source_health.record_*`,
`next(required=True)`) — so a single joint commit was made rather than two.

**Message deviates from the raw Stage 2 prompt's suggested
`fix: harden proxy and odds freshness`.** Used instead:
`fix: validate primary win markets and harden proxy and odds freshness`, per
this file's own prior documented decision, because the commit necessarily
carries Stage 1's market-validation work too and a message naming only Stage 2
would misrepresent what the commit actually contains — the same
overstated-scope problem `CLAUDE.md` and the prompt's own scoping rule warn
against, just in the opposite direction (undersaying rather than oversaying).

- HEAD before this commit: `6d6efc263f23eff45b6f613b73f60dc632c4cfe1`
- HEAD after this commit: `9ce4159bcf86da633bebe3d8a7763d3585957c07`
- Files intentionally excluded from the commit (unrelated to Stages 1–2 —
  matches the Stage 0 "unrelated to the five stages" list, plus Stage 3/4 WIP):
  `.claude/`, `.design-md/`, `images/`, `implementation_plan.md`, `review/`,
  `v4prompts.md`, `RACE_PREDICTOR_REPAIR_PROMPTS.md`,
  `memory/live-repair-00-prompt-sequence.md`,
  `memory/live-repair-triage-20260727.md`, `tools/*`,
  `tests/test_train_lgbm.py`, `reports/*` (Stage 3/4 evidence, not Stage 1/2),
  `models/predict_unified.py`, `models/predictor.py`, `models/suggestions.py`,
  `models/value.py`, `data/predictions.json`, `scripts/reconcile_prob_to_ev.py`
  (Stage 3 work, left uncommitted pending Stage 3's own checkpoint), `HANDOFF.md`
  and `memory/live-repair-0{1,2}-*.md` themselves are committed separately as
  the checkpoint documents, not mixed into the code commit.

### Stage 2 verdict

**STAGE 2 PASS** — ten implementation requirements met, four acceptance gates
met, 1396/1396 tests green (+31 this session), a genuine crash-safety gap
(non-atomic partitioned writes) found and fixed with regression tests, a
genuine dead-telemetry gap (`proxy_reachable` never wired) found and fixed,
bounded live health check run and cross-verified, and the end-to-end
stale-source-cannot-produce-EV proof the Stage 1 report's Fail row demanded is
now green. One security incident occurred and was contained: a proxy
credential was briefly visible in this session's transcript (never in the
repository or its history); the operator should rotate it.

---

## 2026-07-27 — Stage 3: full-field probability, fair-price, and EV reconciliation

**Status: STAGE 3 PASS.** All ten implementation requirements and all five
acceptance gates are met, including the two items Stage 0/1/2 recorded as open
(issues #4 and #5) and the shared-contract half of issue #6. A scoped commit was
created — see _"Commit"_.

### Dependency gate (verified, not inherited)

- `HANDOFF.md` Stage 1 and Stage 2 sections both end **PASS**;
  `memory/live-repair-01-market-ingestion.md` and
  `memory/live-repair-02-proxy-freshness.md` exist and match; joint commit
  `9ce4159` is in history with exactly the documented file set (verified via
  `git log --name-only`); `models/predictor.py` / `models/value.py` /
  `models/suggestions.py` / `models/predict_unified.py` / `features/fuse.py` and
  their test mirrors were left uncommitted for this stage, as documented.
- Baseline before any edit this session: 253 targeted tests green
  (predictor/value/suggestions/fuse/schema/devig/smoke), and the live artifact
  state matched Stage 2's account (fresh-scrape machinery working; cache from
  2026-07-25 stale and missing `runners`).

### The calculation contract (canonical, all surfaces)

**Race identity.** A race is `(venue, race_time)` — `_RACE_KEY`; the cross-book
odds map keys on `(race_date, venue, race_uid, horse_id)` where
`_race_uid` prefers exact `race_time` over per-source `race_id`
(bookmakers mint their own ids; see memory note
`per-source-race-id-fragments-oddsmap`). `features/fuse.py::_runner_key`
includes `race_time` so venue-day races never collapse.

**Non-runners** are dropped in `Predictor._score` BEFORE normalization,
de-vigging, field size, ranking (`_build_race` re-drops defensively).

**Prices — two, never conflated:**

- `reference_odds`/`reference_source` — the ONLY line de-vigged into
  `market_prob`/`fair_prob`. Chosen per race by
  `models/predictor.py::_reference_decimals`: the first source in
  `_BOOKMAKER_SOURCES` order (`livescorebet`, `paddy_power`, `boylesports`)
  whose validity/freshness-gated `odds_by_book` entry prices the ENTIRE field;
  else the fused consensus line (`_decimal_odds`, labelled `fused_consensus`).
  The per-runner best-price overlay is never de-vigged.
- `best_odds`/`best_book` — the executable price. `EV = prob x executable - 1`
  on every line (`expected_value` from `value_win_prob`, `ev_catboost`,
  `ev_lgbm`).

**Race-level EV gate (the single decision — closes issue #6).**
`models/predictor.py::_race_ev_gate` runs once per race in `_build_race` and is
persisted as `ev_eligible: bool` + `ev_gate: {reasons, reference_source,
reference_book_complete, reference_overround, odds_max_age_seconds,
price_sources, computed_at}`. PASS reasons: `incomplete_reference_book:M/N`,
`unpriced_runners:N`, `stale_price_rows:N`, `odds_age_exceeds_ttl:As>Bs`
(TTL = `staleness.max_age_seconds`, 900s). When ineligible, `_suppress_ev`
blanks `expected_value`/`value_edge`/`ev_catboost`/`ev_lgbm`/`market_prob` and
forces `value_bet = False` on EVERY runner (model probabilities stay — they are
price-free); `models.value.find_value_bets` returns `[]` for any race dict
stamped `ev_eligible: false`; `models.suggestions.suggest_bets` and every UI
badge inherit that. Enforced by `check_output_invariants` invariant 6: EV/value
fields on a PASS race, or a partially-populated market line on an eligible
race, are violations.

**Exact reproducibility (audit req 8/10).** Probabilities are quantised to
their persisted 4dp (prices already 3dp) BEFORE any multiply, so these hold
EXACTLY (zero tolerance) in the shipped cache:

```
expected_value == round(value_win_prob  * best_odds - 1, 4)
ev_catboost    == round(catboost_win_prob * exec    - 1, 4)   # exec = best_odds else decimal_odds
ev_lgbm        == round(lgbm_win_prob     * exec    - 1, 4)
value_edge     == round(value_win_prob - implied_prob,   4)
decimal_odds   == round(1 / implied_prob,                3)
```

The prior report needed a 0.02 tolerance (max observed error 0.0081); now the
identities are exact.

**Typed per-runner fields (req 8)** — persisted in every runner dict:
`won_prob_normalized` (headline), `won_prob` (raw marginal),
`catboost_win_prob`, `lgbm_win_prob`, `value_win_prob` (price-free,
F-L-recalibrated), `empirical_win_rate` (the horse's own trailing
`historical_win_rate` — never a model output), `implied_prob` (raw),
`market_prob` (de-vigged), `reference_odds`/`reference_source`,
`best_odds`/`best_book`, `decimal_odds` (fused), `value_edge`,
`expected_value`/`ev_catboost`/`ev_lgbm`, `value_bet`. Value picks
additionally carry `model_win_prob`, `devigged_market_prob`,
`raw_implied_prob`, `fair_decimal_odds`, `executable_odds`, `edge_pp`,
`relative_edge`, `EV`, `overround`, `executable_source`, `reference_source`,
`price_age_seconds`, `computed_at` (req 6 diagnostics).

**Full field (req 2).** `race["runners"]` carries every valid runner ranked by
composite; `selections` stays the display-only top-3. `RacePrediction` now
round-trips `runners`/`ev_eligible`/`ev_gate` (the typed view previously
dropped them silently).

**Value-rule consolidation (req 7).** One core gate
(`models.value.passes_core_gate`) shared by the predictor flag and
`find_value_bets`; one race-level gate shared via the cache; curation gates
(min_confidence, min_prob, edge bars) may only narrow picks below the flagged
set, never widen. UI surfaces reading the field now prefer `runners`:
`ui/app.py::_all_runners`, `ui/pages/8_Race_Detail.py` (count + model's-view +
a visible `EV: PASS — <reasons>` chip on ineligible races),
`ui/pages/12_Horse_Detail.py::_find_runner`, `ui/race_compare.py::_all_runners`,
`scripts/clv_live.py` (CLV log no longer misses mid-field value picks).

### Files changed this session

| File                                                                                                                 | Change                                                                                                                                                                                                                                      |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `models/predictor.py`                                                                                                | `_race_ev_gate` + `_suppress_ev` + persistence; `_reference_decimals` single-book reference; EV/edge/odds quantisation; `reference_odds`/`reference_source`/`empirical_win_rate` typed fields; `RacePrediction.runners`/`ev_*`; invariant 6 |
| `models/predict_unified.py`                                                                                          | EV inputs quantised to persisted precision; reference-book docstring                                                                                                                                                                        |
| `models/value.py`                                                                                                    | Honours `ev_eligible: false` (PASS); `_REF_ODDS_KEYS` prefers explicit `reference_odds`; picks carry provenance + `computed_at` + `empirical_win_rate`                                                                                      |
| `scripts/reconcile_prob_to_ev.py`                                                                                    | Rewritten: live mode audits the shipped cache with NO provenance rewriting + explicit PASS reasons + representative-race input/output dumps; `--synthetic` clearly labelled, separate `*_synthetic.*` outputs; exit 1 on any violation      |
| `scripts/clv_live.py`, `ui/app.py`, `ui/pages/8_Race_Detail.py`, `ui/pages/12_Horse_Detail.py`, `ui/race_compare.py` | Full-field (`runners`) consumption; PASS-reason chip                                                                                                                                                                                        |
| `tests/models/test_predictor.py`                                                                                     | +`TestRaceEvGate` (8), +`TestReferenceBookSelection` (4), +`TestExactEvRecompute` (3), +2 invariant-6 tests                                                                                                                                 |
| `tests/models/test_value.py`                                                                                         | +`TestReferenceBookContract` (5)                                                                                                                                                                                                            |
| `tests/test_reconcile_prob_to_ev.py` _(new)_                                                                         | 12 tests pinning the reconciliation core                                                                                                                                                                                                    |
| Carried from the WIP session (verified, unchanged in intent)                                                         | full `runners` collection, NR-before-normalization, `passes_core_gate`, `_race_uid`, fuse `race_time` key, suggestions runners-first indexing                                                                                               |

No config or schema-contract change (`tests/test_predictions_schema.py` —
additive only — still green).

### The seven acceptance regressions

1. **Top-three partial-field de-vig** —
   `test_value_engine_receives_complete_field_not_top_three` (pre-existing WIP,
   re-verified) + `TestReferenceBookContract::test_incomplete_reference_book_passes_entire_race`.
2. **Normalize-before-NR** — `test_non_runner_dropped_before_within_race_normalization`
   (WIP, re-verified) + `test_non_runner_excluded_entirely`.
3. **Race-key collision** — `test_map_separates_same_venue_day_different_race_time`
   - fuse `race_time` key tests (WIP, re-verified).
4. **Incomplete reference book** — `TestRaceEvGate::test_unpriced_runner_fails_race_closed`
   - value-layer PASS test (new).
5. **Synthetic-best-price book** — `TestReferenceBookSelection::test_best_price_overlay_never_used_as_reference`
   - `TestReferenceBookContract::test_best_price_overlay_never_devigged` (new).
6. **Stale timestamp mismatch** — Stage 2's end-to-end proof (re-run) +
   `TestRaceEvGate::test_stale_provenance_fails_race_closed_at_cache_boundary` /
   `test_explicit_stale_flag_fails_race_closed` (new, cache-boundary layer).
7. **UI/backend rule disagreement** —
   `TestRaceEvGate::test_ineligible_race_passes_value_and_suggestion_engines` +
   `test_eligible_race_flags_agree_across_surfaces` +
   `TestReferenceBookContract::test_race_level_ev_gate_honoured_from_cache` (new).

### Commands run, and results

```powershell
# Baseline (pre-edit)
.\.venv\Scripts\python.exe -m pytest -q tests/models/test_predictor.py tests/models/test_value.py tests/models/test_suggestions.py tests/features/test_fuse.py tests/test_predictions_schema.py tests/test_devig.py tests/test_smoke_end_to_end.py
# 253 passed

# Fresh rebuild from a real scrape (the acceptance-4 path)
.\.venv\Scripts\python.exe -m scripts.refresh
# scrape livescorebet -> 356 rows (35 races, fresh); boylesports failing
# (empty_card, 403 — cache rows correctly TTL-dropped); paddy_power -> board
# prices present in odds_by_book
# predicted 35 races -> data/predictions.json  (generated_at 2026-07-27T08:51:26+01:00)

# Honest live reconciliation (no provenance rewriting)
.\.venv\Scripts\python.exe -m scripts.reconcile_prob_to_ev
# 35 races / 356 runners — invariant violations: 0, EV identity failures: 0, exit 0

# Labelled synthetic arithmetic audit (separate files)
.\.venv\Scripts\python.exe -m scripts.reconcile_prob_to_ev --synthetic
# invariant violations: 0, EV identity failures: 0

# Full suite
.\.venv\Scripts\python.exe -m pytest -q
# 1430 passed in 104.00s      (Stage 2 recorded 1396; +34 new tests)
```

### Acceptance evidence (live cache, 2026-07-27 08:51 Dublin)

- **Issue #4 CLOSED** — `data/predictions.json` now carries `runners` for every
  race: header `total_runners: 356`, serialized runners **356/356** (was 170 of
  514 with no `runners` key).
- **Issue #5 CLOSED** — the reconciliation's default mode audits the shipped
  cache as-is; freshness rewriting exists only behind `--synthetic`, whose
  report opens with an explicit "no evidence of live freshness" banner and
  writes `*_synthetic.*` filenames.
- **Issue #6 CLOSED (Stage 3 half)** — every race in the shipped cache carries
  `ev_eligible`/`ev_gate`; all 35 races eligible on this card, reference book
  `livescorebet` for all (overrounds 0.147–0.798; the 0.68/0.80 races are the
  23-runner Galway festival fields), odds age ~224s at compute vs TTL 900s.
- Displayed win probabilities sum to 1 (±0.0002) and complete reference books
  de-vig to 1 (±0.0002) on every race; **0 invariant violations, 0 EV identity
  failures** across 356 runners.
- 3 value bets flagged (Southwell 17:00 Storm Free, Galway 17:10 Witches
  Familiar, Windsor 17:50 Lucy The Wire). Lucy The Wire shows the price split
  live: reference 2.88 (livescorebet, de-vigged) vs executable 3.0
  (paddy_power) — EV computed on the executable price.
- Fail-closed freshness demonstrated on real data: boylesports' retained
  2-day-old snapshot (fetched_at 2026-07-25) was excluded by the TTL gates from
  both fusion and the odds-by-book map; `data/source_health.json` records it
  failing (`empty_card`, `proxy_reachable: false`).

### Known limitations / carried-forward

| #          | Issue                                                                                                                                                                                         | Status                                                                                                                                                    |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ~~4~~      | ~~predictions.json has no `runners` key~~                                                                                                                                                     | **CLOSED** this session                                                                                                                                   |
| ~~5~~      | ~~reconcile script rewrites `fetched_at`/`stale`~~                                                                                                                                            | **CLOSED** this session                                                                                                                                   |
| ~~6~~      | ~~EV computed before any shared race-level contract~~                                                                                                                                         | **CLOSED** (gate persisted in cache, enforced everywhere)                                                                                                 |
| 8          | `value_win_prob` is market-adjusted (F-L recalibrated vs the same price EV uses); the independent price-free prob is not persisted separately                                                 | Open — Stage 4. Galway 17:10 (value_win_prob 0.7251 vs market 0.3448, EV +63%) is the live reminder that an honest contract does not make the model right |
| 9          | `pytest-timeout` not in `.venv`                                                                                                                                                               | Open, infra                                                                                                                                               |
| 11         | Paddy Power contributes 0 rows to `data/live_odds.parquet` (its board prices DO reach `odds_by_book` via its cache-to-unified path — verified live this session)                              | Open, narrowed                                                                                                                                            |
| 16 _(new)_ | `RunnerPrediction.value_edge` stays defined vs the raw fused `implied_prob` (diagnostic); the canonical market edge (`edge`/`edge_pp` vs fair) lives in value picks. Documented, not a defect | Note                                                                                                                                                      |
| 17 _(new)_ | `evaluate_filter` (backtest replay) has no per-race key so it gates on raw implied prob — conservative, pre-existing, documented in its docstring                                             | Note                                                                                                                                                      |

### Commit

Single scoped commit `fix: reconcile full-field probability and ev` containing:
the five deferred model/feature files (which also finally land
`models/predictor.py`'s Stage 1 fail-closed hunks and `features/fuse.py`'s
Stage 2 staleness hunks — inseparable since Stage 0, per the plan of record in
the Stage 2 section), the UI/script full-field consumers, the rewritten
reconciliation harness + its new test file, the three test mirrors, the fresh
`data/predictions.json` (the cache is tracked; committing the rebuilt artifact
is itself Stage 3 evidence — the old stale cache was the defect), the four
reconciliation evidence reports, `HANDOFF.md`, and the Stage 3 memory note +
`memory/MEMORY.md` pointer. Excluded as unrelated (unchanged from the Stage 0
list): `.claude/`, `.design-md/`, `images/`, `review/`, `tools/`,
`implementation_plan.md`, `v4prompts.md`, `RACE_PREDICTOR_REPAIR_PROMPTS.md`,
`memory/live-repair-00-prompt-sequence.md`,
`memory/live-repair-triage-20260727.md`, `tests/test_train_lgbm.py`,
`reports/backtest_*`, `reports/testrun/`, `reports/*_20260614_*`,
`reports/manifest.json`, Stage 4's `reports/calibration_dependency_failure_*` /
`reports/live_market_dependency_check_*`, and the runtime
`data/source_health.json`.

Commit hash: **`d7f7872`** (parent `a9f7103`), 23 files. Recorded by the
follow-up docs commit, since this section is part of the commit itself.

### Is Stage 4 safe to start?

**Yes.** Its stated blocker — "Stage 3's acceptance item 4 passes" — is closed:
predictions are rebuilt from genuinely fresh validated data through the Stage 3
code path and reconcile cleanly. Constraints for whoever runs it: do NOT reuse
the 2026-07-25 head-to-head figures as a new verdict; issue #8
(`value_win_prob` market adjustment, isotonic zero-floor) is its core scope;
and the paper-only rule stands — the LightGBM holdout remains GO-on-log-loss
with **negative CLV**, which is never a green light to bet.

### Stage 3 verdict

**STAGE 3 PASS** — ten implementation requirements met, five acceptance gates
met, 1430/1430 tests green (+34), predictions rebuilt from a fresh live scrape
with the full field serialized, and a zero-tolerance reconciliation proving
every persisted EV recomputes exactly from persisted inputs, with explicit
per-race PASS reasons carried in the cache.

---

## 2026-07-27 — Stage 4: leak-free calibration and model-validity audit

**Status: STAGE 4 PASS (audit complete and reproducible).**
**Model verdict: NO-GO — paper-only.** These are different judgements and the
prompt requires both; the audit succeeding is exactly what makes the NO-GO
trustworthy. Full numbers: `reports/calibration_audit_20260727.md`; machine
artifacts under `data/audit/stage4/`.

### Dependency gate (verified, not inherited)

Stages 1–3 all end PASS in this file; memory notes exist and match; commits
`9ce4159` (Stages 1+2) and `d7f7872` (Stage 3) verified in history with their
documented file sets; baseline full suite re-run green (1430/1430) before any
edit; Stage 3's constraints honoured (the 2026-07-25 head-to-head figures were
never reused as a verdict; issue #8 was this stage's core scope).

### What was built (all reproducible by CLI)

| Piece                                                                                                                                                                                              | Command                                                                      |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| 44-day results catch-up (Betfair SP backbone + Sporting Life enrichment through the canonical append-dedupe writer; 31,584 rows merged, position fill 81.2%)                                       | `python -m scripts.fetch_results_window --start 2026-06-13 --end 2026-07-26` |
| Unified + training-matrix rebuild (260,490 labelled rows through 2026-07-25; the normalizer failed-closed 216 contaminated live rows during the rebuild — Stage 1's contract firing in production) | `utils.normalizer.normalize()` + `features.builder.build_training_matrix()`  |
| Audit phases: validated panel, leakage proofs, 17-fold expanding walk-forward (everything fit inside folds, GPU), validation-fold selection + ablations + extremes, one-shot final window, report  | `python -m scripts.calibration_audit --run stage4 --all`                     |
| F-L recalibrator refit (sigmoid bands, zero-free), with backup + meta sidecar                                                                                                                      | `python -m scripts.refit_fl_recalibrator`                                    |

New code: `audit/` package (panel, leakage, walkforward, metrics_panel,
extremes, final_eval, report), `scripts/calibration_audit.py`,
`scripts/fetch_results_window.py`, `scripts/refit_fl_recalibrator.py`,
`ui/_winrate.py`.

### Requirement-by-requirement outcome

1. **Validated rows only.** Audit panel: 143,364 WIN rows / 16,039 races
   (2024-01-01 to 2026-07-25) with reason-coded exclusions: 111,630
   PLACE-market rows (`models/train.py` trains on WIN+PLACE duplicated
   runners — a documented data-hygiene defect for the next retrain), 5,478
   rows / 946 races extreme-booksum thin books, 11 unpriced, 7 implausible
   fields. Historical Betfair-SP rows cannot carry the live specials threat
   (market identity = requested file identity); equivalent checks implemented.
2. **Price-free proof.** Structural provenance PASS; empirical guards PASS
   (closing-move partial-corr and outcome-corr across the whole panel);
   sample-weight channel verified pre-off (implied_prob == 1/morningwap on
   227k rows — the odds_finish claims in CLAUDE.md and the backtest
   docstrings were STALE and are corrected in this commit). **One genuine
   violation found by the new append-future-invariance test:
   `race_complexity`** — z-scored over the whole dataset (all 247,617
   historical values shifted when future data was appended; drift cosmetic at
   mean abs 0.0024 / corr 1.0, but it breaks point-in-time construction and
   one component embeds race-level market entropy). Excluded from every
   audit-fitted model; flagged for the next retrain.
3. **Independent vs market-adjusted persisted separately.**
   `value_win_prob_independent` (price-free) now travels beside
   `value_win_prob` (market-adjusted) through predictor, cache,
   `models.value` picks (`model_win_prob_independent`) and RunnerPrediction;
   pre-Stage-4 caches rehydrate with None. Cross-fitting is inherent to the
   walk-forward (per-fold F-L fits); in-sample optimism measured (+0.0016
   Brier). **Circularity quantified:** corr(prob, 1/price) 0.578 -> 0.925
   after adjustment; 77% of a +5% price improvement's EV gain is absorbed by
   the remap; within-price AUC falls 0.549 -> 0.537 (the design claim that
   the correction preserves within-price discrimination is falsified); 2,484
   band-[2,4] OOS bets that are EV-positive only via the remap realize A/E
   0.92 and yield −1.9% — manufactured, losing edge.
4. **Extremes.** Exact zeros root-caused to the F-L isotonic bands (y-floor
   0.0 in every band, y-ceiling 1.0 in odds-on bands, 8–24 thresholds per
   band; 3.6% of OOS inputs outside fitted support; 20/356 live-cache zeros).
   Sigmoid bands selected ON VALIDATION FOLDS (race log-loss 1.7052 vs
   1.7432, equal Brier, zero extreme emissions) and the refit shipped with a
   backup (`data/backups/fl_oddsband_v3nf_calib_isotonic_20260617.pkl`) and
   meta sidecar (`models/fl_oddsband_v3nf_calib_meta.json`).
5. **Race-level calibration comparison** (validation folds only): grouped
   softmax temperature (fitted T 0.92–1.02) ties
   marginal-calibrate-then-normalize (delta ~0.0006 race log-loss);
   production recipe retained. The final test set selected nothing.
6. **/7. Walk-forward + untouched final window.** 17 expanding folds with
   test windows ending 2026-06-12; hyperparameters fixed from the shipped
   v3nf meta (documented contamination channel for pre-June folds; the final
   window is clean of it). **Final window 2026-06-13 to 2026-07-25** postdates
   every model cutoff (v3nf/v3 80% split = 2025-12-02, LGBM 2026-05-22),
   every published holdout (max end 2026-06-12) and every gate-tuning frame
   (max 2026-06-16). Scored once plus one prespecified addendum line; zero
   tuning on it.
7. **/10. Metrics with intervals on identical races** (7,922 runners / 904
   complete-book races; race-bootstrap B=1000):

   | line                                  | race LL | delta vs prop. de-vig | 95% CI               | sig.?     |
   | ------------------------------------- | ------: | --------------------: | -------------------- | --------- |
   | audit_independent                     | 1.87400 |              −0.18621 | [−0.22507, −0.14620] | worse     |
   | frozen_v3nf_independent               | 1.86902 |              −0.18124 | [−0.22283, −0.14192] | worse     |
   | frozen_v3nf_market_adjusted (old iso) | 1.67919 |              +0.00859 | [−0.00577, +0.02007] | ns        |
   | frozen_v3nf + sigmoid refit           | 1.68058 |              +0.00720 | [+0.00107, +0.01454] | nominal\* |
   | frozen_v3_priced                      | 1.67660 |              +0.01118 | [−0.00358, +0.02754] | ns        |
   | frozen_lgbm                           | 1.67724 |              +0.01055 | [−0.00463, +0.02681] | ns        |

   Market de-vig baselines on the same races: proportional 1.68778, power
   1.68631, shin **1.68568**. \*The one nominally-significant delta is a
   ~93%-price-echo line measured against the WEAKEST baseline; against shin
   no line is significant. **The June "GO" stamps (313 races, no intervals)
   do not survive uncertainty quantification.** CLV −13.4% [−13.9, −13.0],
   beat rate 26.5%; every EV>0 portfolio loses at the pre-off price (ROI
   −8.5% to −23.4%). Subgroup A/E (odds band, field size, venue, month,
   region, completeness) in `final_evaluation.json`.

8. **Win-rate semantics.** Realized rates now render as
   `rate% (wins/denominator)` with a Wilson 95% CI and window label
   everywhere (`ui/_winrate.py`; performance dashboard, performance page,
   bet placer, Yesterday's predictor; tracker summaries carry raw counts).
   Predicted probabilities are labelled `Model win %`, never bare win rate.
9. **Gates.** Market-relative: FAIL (independent line significantly worse;
   no line significantly better than the best baseline). Drift: FAIL (PSI
   `going_speed` 4.92 — summer-going seasonality plus an enrichment
   fill-rate change on an empirically-dead feature; monthly log-loss deltas
   stable). **NO-GO issued; real-money recommendations stay disabled; no
   filter tuning performed.**

### Tests

- New: `tests/audit/` (panel, leakage, walk-forward fold-boundary +
  cross-fit spy + final-window-never-scored, metrics, extremes),
  `tests/ui/test_winrate.py`, and predictor split/reproducibility/
  artifact-compat regressions in `tests/models/test_predictor.py`.
- Full suite: **1491 passed** (was 1430; +61). No network in tests; CatBoost
  stubbed in walk-forward tests per repo convention.
- Reproduce check (acceptance 4): `select` and `final` phases re-run from the
  CLI — frozen-line numbers byte-identical, GPU-refit lines match to 1e-6,
  de-vig baselines exact; report regenerated from the reproduced artifacts.

### Artifacts and hashes

See the report's hash table. Key: training matrix (post-refresh)
`3E7CDE9869D86ED2…`; pre-refresh backup retained at
`data/backups/training_20260618_pre_stage4.parquet` (`1C20FA97…`, matching
the 2026-07-25 dependency report's hash); new F-L artifact `ECCED5EE…` with
the old isotonic artifact backed up (`548BB776…`). Frozen model binaries
untouched.

### Unresolved issues — updated

| #          | Issue                                                                                                                             | Status                                                |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| ~~8~~      | ~~value_win_prob market-adjusted, independent not persisted~~                                                                     | **CLOSED** — split persisted + circularity quantified |
| 9          | pytest-timeout not in venv                                                                                                        | Open, infra                                           |
| 11         | Paddy Power contributes 0 rows to live_odds.parquet                                                                               | Open (unchanged)                                      |
| 18 _(new)_ | `race_complexity` not point-in-time (global z-scores + market entropy); excluded from audit models, still inside frozen artifacts | Open — next retrain                                   |
| 19 _(new)_ | `models/train.py` trains on WIN+PLACE duplicated rows                                                                             | Open — next retrain                                   |
| 20 _(new)_ | catboost meta records no train cutoff so the holdout leakage guard cannot fire; v3nf true cutoff 2025-12-02 documented here       | Open — next retrain                                   |
| 21 _(new)_ | live predictions.json (2026-07-27 08:51) predates the field split + F-L refit; the next refresh picks both up                     | Open — routine                                        |

### Stage 5 readiness

**Stage 5 may proceed in PAPER-ONLY mode.** It must consume this stage's
NO-GO (`reports/calibration_audit_20260727.md` +
`data/audit/stage4/final_evaluation.json`) as the standing verdict, keep
every real-money pathway disabled, and must not tune value filters
(requirement 11). The sigmoid-refit F-L artifact is live for the next
refresh; `value_win_prob_independent` appears in the cache from the next
predictor run.

### Commit

Single scoped commit `test: recalibrate and validate model` containing: the
`audit/` package, the three new scripts, `ui/_winrate.py`, the
predictor/value/bet-tracker/UI edits, the new tests, the corrected stale-leak
docs (CLAUDE.md, README.md, backtest/data.py, backtest/holdout.py), the dated
calibration report, HANDOFF.md, the Stage-4 memory note and the MEMORY.md
pointer. Model binaries/pkls, their meta sidecars, and `data/` artifacts
(including `data/audit/`) are gitignored by repo convention and stay
untracked — their SHA-256 hashes are recorded in the report and prior
versions are backed up on disk.

Commit hash: **`e3c705f`** (parent `abdfae4`), 36 files, 3,818 insertions.
Recorded by the follow-up docs commit, since this section is part of the
commit itself.

### Stage 4 verdict

**STAGE 4 PASS** (audit complete, reproducible, 1491/1491 green) ·
**MODEL NO-GO** (paper-only; real-money recommendations remain disabled).

---

## Stage 5 — realistic execution, forward validation, and betting safeguards (2026-07-27)

**Dependency gate:** Stages 1–4 verified PASS from this file, the four
`memory/live-repair-0*.md` notes, commits `a9f7103` / `d7f7872` / `abdfae4` /
`e3c705f` and their diffs, and the artifacts each stage claims. Stage 4's dated
report (`reports/calibration_audit_20260727.md`) and
`data/audit/stage4/final_evaluation.json` carry **MODEL NO-GO**. Stage 5 therefore
ran in paper-only mode throughout and produced no real-money recommendation.

### Files and schemas changed

New package `execution/` (16 modules + `__init__`), one script, one UI page and its
Streamlit-free logic module:

| file                                                             | what it owns                                                                                                                                                                                                                                            |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execution/config.py`                                            | `ExecutionConfig` from `config.yaml`; every limit config-driven, `paper_only` defaults True and is clamped True                                                                                                                                         |
| `execution/snapshots.py`                                         | append-only point-in-time odds store keyed `(race_uid, horse_key, bookmaker, market_type, fetched_at)`; `latest_quote` enforces `fetched_at <= as_of` in SQL; `closing_quote` is the only path to a post-off price; `FuturePriceError` on any violation |
| `execution/frictions.py`                                         | latency, staleness, deterministic rejection/suspension, per-bet book limit, exchange commission, BOG only when recorded                                                                                                                                 |
| `execution/race_facts.py`                                        | non-runners, Rule 4, dead heats, places paid — read from the raw archive, never assumed                                                                                                                                                                 |
| `execution/settlement.py`                                        | win / each-way / dead-heat / void / Rule-4 settlement; commission on net winnings; EW terms captured at bet time                                                                                                                                        |
| `execution/staking.py`                                           | fractional-Kelly planner under every ceiling; reports the binding constraint                                                                                                                                                                            |
| `execution/safeguards.py`                                        | stop-loss, daily loss, daily/race exposure, duplicates, started races, stale sources, open-ticket cap                                                                                                                                                   |
| `execution/gates.py`                                             | the PASS-by-default candidate gate — eight conditions, all must affirmatively pass                                                                                                                                                                      |
| `execution/tickets.py`                                           | the disclosure ticket + SQLite store; `CHECK (paper_only = 1)`, `RealMoneyTicketRefused`                                                                                                                                                                |
| `execution/baselines.py`                                         | model-only / de-vigged-market / favourite selections over one shared eligible race set                                                                                                                                                                  |
| `execution/evaluation.py`                                        | ROI, yield, hit rate, A/E, CLV, drawdown, streaks, turnover — each with a race-clustered bootstrap 95% interval                                                                                                                                         |
| `execution/selection.py`                                         | chronological split, tune on train, select once, score test once, Šidák correction, on-disk lock                                                                                                                                                        |
| `execution/model_gate.py`                                        | reads Stage 4's verdict; unknown or unreadable ⇒ NO-GO                                                                                                                                                                                                  |
| `execution/simulator.py`                                         | the walk-forward itself; bankroll and exposure evolve in race order                                                                                                                                                                                     |
| `execution/forward_gate.py`                                      | the release gate; refuses `evidence_kind == "backtest"` by construction                                                                                                                                                                                 |
| `execution/report.py`                                            | the two dated Markdown reports; three lanes, always all three                                                                                                                                                                                           |
| `scripts/paper_betting.py`                                       | the one deterministic end-to-end command                                                                                                                                                                                                                |
| `ui/forward_validation.py` + `ui/pages/16_Forward_Validation.py` | the dashboard, logic/Streamlit split per repo convention                                                                                                                                                                                                |

Schemas (both in `data/execution/`):

- `backtest_snapshots.db` — `odds_snapshots(race_uid, horse_key, bookmaker,
market_type, fetched_at, back_odds, lay_odds, available_stake, is_closing,
source, ...)`, `UNIQUE` on the five-part key, `RAISE(ABORT)` triggers on UPDATE
  and DELETE (append-only in storage, not by convention).
- `paper_tickets.db` — `paper_tickets(...)` with `CHECK (paper_only = 1)` and a
  duplicate-suppressing unique index on `(race_uid, horse_key, bookmaker,
market_type, bet_type)`.
- `race_facts.parquet` — the settlement ground truth (non-runners, Rule 4, dead
  heats, places paid) materialised once from the archive.
- `selection_lock.json` — the immutable record of the one test-window scoring.

`config.yaml` gained an `execution:` block. No existing key was changed, no
existing module's behaviour altered.

### Execution assumptions (stated, not hidden)

- **Reconstructed series, not captured.** The historical warehouse holds one
  pre-off observation per runner (`ppwap` falling back to `morningwap`).
  `seed_snapshots` stamps it at `off − 30 min` and the Betfair SP at the off. The
  backtest can therefore model frictions on a single quote but **cannot measure
  price movement between observations**. Live capture through
  `execution.snapshots` is what produces a real series.
- **Decision at `off − 30 min`**, clearing the started-race buffer; latency ages
  the quote between reading and striking rather than moving the read.
- **Rejection 5% / suspension 2%**, drawn deterministically from blake2b of the
  seed and the bet's own identity — order-independent and exactly reproducible.
- **Settlement source is labelled per row.** The 2026-07-27 walk was 100%
  archive-settled for all three strategies.
- **Win-only unless EW terms are passed explicitly.** Each-way terms and BOG are
  not recorded historically, so `simulate(bet_type="each_way")` raises rather than
  inventing terms, and BOG is never applied without per-bet recorded evidence.
- **Not exercised by the backtest** (printed in the report itself, not only here):
  live source-health staleness, price movement between observations, EW and BOG.

### Staking and exposure limits — risk ceilings, not profitability claims

Config-driven under `execution:` in `config.yaml`, all clamped conservative:

0.10 fractional Kelly · 0.5% of bankroll per bet · 0.5% per race · 3% daily
exposure · minimum stake €0.50 · 20% bankroll stop-loss · 2% daily loss limit ·
25 open tickets · accumulators and correlated bets disabled.

**These are risk ceilings, not profitability claims, and must not be raised
without forward evidence.**

### Forward-release gate state

**NOT MET — 9 of 9 criteria failed.** `model_go` (NO-GO), `min_weeks` (0.00 of 8),
`min_qualified_bets` (0 of 200), `min_qualified_races` (0 of 150),
`positive_mean_clv`, `clv_ci_lower`, `ae_stable`, `calibration`, `drawdown` — the
last five unmeasurable because the forward window has not started. The gate reads
**paper evidence only**: `evaluate_forward_gate` refuses `evidence_kind ==
"backtest"`, so no backtest can ever open it.

### Backtest result — untouched test window 2026-02-22 .. 2026-05-25 (14 229 rows)

Selected strategy `edge0.020_ev0.050_short_lt_4`, walked alongside the two
baselines over the same 1 480 eligible races:

| strategy        | bets | ROI     | ROI 95% CI         | mean CLV   | CLV 95% CI (log)   | A/E   | max DD | worst streak |
| --------------- | ---- | ------- | ------------------ | ---------- | ------------------ | ----- | ------ | ------------ |
| model_only      | 63   | +31.10% | [−9.07%, +70.89%]  | **−8.25%** | [−0.1248, −0.0644] | 1.066 | 2.63%  | 7            |
| devigged_market | 74   | +23.33% | [−27.76%, +79.64%] | −5.53%     | [−0.0925, −0.0369] | 1.194 | 3.83%  | 12           |
| favourite       | 153  | +12.13% | [−16.63%, +40.41%] | −2.13%     | [−0.0442, −0.0111] | 0.999 | 6.02%  | 12           |

**How to read this.** All three ROI intervals span zero: none of these results is
distinguishable from chance. All three CLV intervals sit **entirely below zero**,
and the model's is the _worst of the three_ — worse than simply backing the
favourite. A strategy that appears to profit while losing to the closing line is
showing variance, not edge. This is not evidence of profitability and must not be
reported as a return.

### Selection protocol and its known weakness

64 variants scored on train, 16 carried to validation, one winner scored on the
test window exactly once; Šidák takes α 0.05 → 0.00080 for 64 comparisons.
Windows: train 2025-01-01..2025-11-05 (44 584 rows), validation
2025-11-06..2026-02-20 (13 204), test 2026-02-22..2026-05-25 (14 229). Winner
`edge0.020_ev0.050_short_lt_4` — train −0.06604, validation −0.06849, **test
−0.09227**, scored 2026-07-27T21:27:01+00:00.

**The lock was reset once**, so the test window has now been scored twice across
this programme and the Šidák correction **understates** the true multiplicity.
Treat the current test score as the more optimistic of two draws. The superseded
lock is preserved at `data/execution/selection_lock_superseded_20260727.json`
(winner `edge0.080_ev0.050_mid_4_12`, test score −0.14456, scored
2026-07-27T21:10:20+00:00) with `superseded_reason` recorded inside it, and
`scripts/paper_betting.py:_superseded_locks` discovers any such file
mechanically — so the report cannot render without naming every extra draw.
Deleting the archived file is the only way to remove the disclosure, which is
precisely why the reset archives rather than overwrites.

Why it was reset: the odds-band filter was applied to the _scored frame_ instead
of to the backable-runner mask. That renormalised the de-vig over a field with
its favourites removed (so `edge` was measured against a book no bookmaker
offered) and changed the eligible race set per strategy — the `short_lt_4` band
reported 0 of 3 817 races eligible and 16 of 64 candidates were never scored at
all, silently violating "compare on the same eligible races". Fixed in
`execution/baselines.py:model_only_selection` (the band moved into the `keep`
mask and is threaded through `_strategy_picks` / `simulate` /
`run_walk_forward`), covered by six regression tests in
`tests/execution/test_baselines.py`, and the whole sweep was re-run from scratch.
The corrected sweep logs `baselines: 4615/5048 races eligible` identically for
all 64 candidates and the rebuilt lock has zero null train scores.

### Two further defects found and fixed

1. `execution/report.py:_interval` read `low`/`high` where
   `execution.evaluation.Interval` emits `lower`/`upper`, so **every uncertainty
   interval rendered as `n/a`** — a 31% ROI on 63 bets appeared unqualified. The
   same bug sat in `ui/forward_validation.py:fmt_interval`. Both fixed via a
   shared `_bounds` / `interval_bounds` helper that accepts either spelling.
2. Consequently `_clv_reading` announced CLV as "not measurable on this ledger"
   when it had been measured at −8.25% with a CI entirely below zero. Fixed, and
   a new `_roi_reading` now states in prose what the ROI interval licenses,
   beside the ROI itself rather than only in a table cell.

Both are regression-tested (`tests/execution/test_report.py`,
`tests/ui/test_forward_validation.py`).

### Commands

```powershell
python -m scripts.paper_betting                   # select, walk, gate, both reports
python -m scripts.paper_betting --candidates-only # today's card only, no backtest
python -m pytest -q                               # 1966 passed
streamlit run ui/app.py                           # -> "Forward Validation" page
```

Never pass `--timeout` to pytest; `pytest-timeout` is not installed in this venv.

### Tests

`tests/execution/` (15 files, including the new `test_report.py` and
`test_simulator.py`) plus `tests/ui/test_forward_validation.py`. Full suite
**1966 passed** — the Stage-4 baseline was 1491, so Stage 5 added 475 tests and
broke none. Earlier-stage regressions (integrity, calibration, predictions
schema, settlement, staking, suggestion, dashboard, end-to-end) all re-run green.

### Reports and artifacts (2026-07-27)

- `reports/forward_validation_20260727.md` — three lanes, gate state, selection
  protocol, reset disclosure
- `reports/todays_candidates_20260727.md` — 0 candidates, 356 PASSes, no
  real-money recommendation, every failed criterion enumerated
- `reports/candidate_decisions_20260727.csv` — the per-runner receipt (356 rows)
- `data/execution/stage5_run.json` — the machine-readable run
- `data/execution/selection_lock.json` and its superseded sibling,
  `data/execution/backtest_snapshots.db`, `data/execution/race_facts.parquet`

### Unresolved risks

1. **Negative CLV is the standing finding**, unchanged since Stage 4 and
   reproduced here on an independent window by all three strategies.
2. **The test window has been scored twice.** Any future comparison should treat
   −0.0923 as the more optimistic of two draws, not as a clean single measurement.
3. **Price movement is modelled, not measured.** Only live capture closes this.
4. **Live source-health blocking is untested end to end.** The reconstructed feed
   is fresh by construction; the logic is unit-tested in
   `tests/execution/test_safeguards.py` but has no real exercise until the
   forward window runs.
5. **The forward window has not started.** Eight weeks of paper evidence and a
   meaningful qualified sample is the minimum before any of this is revisited.

### Deployment state

**PAPER-ONLY.** `execution.paper_only` is clamped True regardless of config, the
ticket table carries `CHECK (paper_only = 1)`, `TicketStore.issue` raises
`RealMoneyTicketRefused`, and Lane 3 of the report is a constant string with no
data path into it. There is no code path in this repository that can stake real
money.

### Commit

`d7a7058` — `feat: add realistic paper-betting safeguards`, parent `8ed8226`,
47 files, 21 517 insertions, 1 deletion. Scoped to Stage 5 only. Deliberately
**not** included, and still uncommitted in the working tree: `.claude/`,
`.design-md/`, `images/`, `review/`, `tools/`, `data/audit/`, `data/backups/`,
`data/source_health.json`, `implementation_plan.md`,
`RACE_PREDICTOR_REPAIR_PROMPTS.md`, `v4prompts.md`, `tests/test_train_lgbm.py`,
`memory/live-repair-00-prompt-sequence.md`,
`memory/live-repair-triage-20260727.md`, `reports/testrun/` and the pre-existing
`reports/*` artifacts — all unrelated user work, left exactly as found.

`data/execution/backtest_snapshots.db` (12 MB) and `race_facts.parquet` are
newly gitignored as regenerable. The two selection locks and `stage5_run.json`
are committed on purpose: they are the audit record of which strategy was chosen
and when the test window was scored, and the superseded lock is the only
surviving disclosure that the window was scored twice.

### Stage 5 verdict

**STAGE 5 PASS** · **MODEL NO-GO** · **DEPLOYMENT PAPER-ONLY**

---

## Stage 6 — start live point-in-time capture and the daily paper loop (2026-07-28)

**Dependency gate:** Stages 1–5 verified PASS/NO-GO from this file and commit
`d7a7058`. Stage 6 does not touch, bypass, or soften the model gate — it is
still **MODEL NO-GO** from `data/audit/stage4/final_evaluation.json`, unchanged.
Every runner gated in this stage's own test fixtures and in the real production
run comes back PASS for exactly that reason.

### What was built

| file                           | what it owns                                                                                                                                                                                                                                              |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `execution/window.py`          | the 8-week forward-validation window: `start_window` (deliberate, separate, never auto-called) and `verify_window` (called every run; resets to zero and records why on any model/selection-lock/config-hash change)                                      |
| `execution/gap_ledger.py`      | gap accounting: `record_day` (sticky once captured — a later failed poll the same day never reverts it), `reconcile_gaps` (back-annotates only unrecorded days strictly before "today"), `qualifying_weeks` (gap-adjusted, never wall-clock)              |
| `scripts/daily_paper_loop.py`  | the one idempotent daily command: capture → PASS-by-default gate → issue paper tickets → settle open tickets → recompute forward metrics → write dated reports + `data/execution/daily_loop_run.json`                                                     |
| `scripts/daily_paper_loop.cmd` | the Task Scheduler entry point, mirroring `scripts/daily_clv.cmd`'s convention: `scripts.refresh` (fresh card + first live snapshot) then `scripts.daily_paper_loop --no-scrape` (gate/issue/settle/report)                                               |
| `scripts/refresh.py`           | (edited) `_capture_snapshots()` now appends every scrape's board to the append-only `odds_snapshots` store via `execution.snapshots.ingest_live_odds_parquet`; best-effort, never fails the scrape                                                        |
| `execution/report.py`          | (edited) added `window_state_block` / `gap_ledger_block`, rendered into Lane 2 — window status/days-elapsed/reset history and gap-day counts now sit alongside the paper summary                                                                          |
| `ui/forward_validation.py`     | (edited) `load_run` now merges the Stage 6 `daily_loop_run.json` (priority, real evidence) with the legacy Stage 5 `stage5_run.json` (fallback for `backtest`/`selection` keys only); added `window_row`/`gap_rows` and two new Lane 2 Streamlit sections |

No changes were made to `execution/snapshots.py` — its append-only / future-price
guarantees already existed and are already tested; the "no backdated
`fetched_at`" guarantee for this stage is a _process_ guarantee, pinned by a
signature test showing the only wired capture entry point,
`ingest_live_odds_parquet`, has no timestamp-override parameter a caller could
use to backdate a row.

### Why the daily command is idempotent

`build_ticket` derives each ticket's ID from a SHA1 hash of
`(race_uid, horse_key, bet_type, issued_at)`, and `run()` passes a day-truncated
`now` as `issued_at` — so re-running the loop for the same day always recomputes
the same ticket IDs, and `TicketStore.issue` treats a repeat as a duplicate, not
a new ticket. Settlement only touches `open_tickets()` (unsettled CANDIDATE
decisions), so a second run the same day settles nothing new. Verified twice
against the real production DB with the same day: second run issued 0 (356
duplicates), settled 0.

### Gap accounting and unmeasurable CLV

A day the loop never captured a snapshot for is recorded as `gap` with a cause
(`scraper_outage` by default), and contributes **zero** qualifying weeks —
`qualifying_weeks` counts only captured days, so a gap can never be
interchanged with a clean no-bet day in the forward gate's `min_weeks`
criterion. Where `TicketStore.settle` has no matching closing snapshot quote,
`clv_pct` is stored as `None` (unmeasurable) and the settlement stats surface
`unmeasurable_clv` — never dropped, never invented. Both pinned in
`tests/scripts/test_daily_paper_loop.py`.

### Scheduling (documented, not installed)

This repo never installs its own scheduled tasks. To register the daily loop
for unattended execution, following `scripts/daily_clv.cmd`'s own convention:

```powershell
schtasks /create /tn "RacePredictor_DailyPaperLoop" /tr "C:\Users\mshr\Documents\Race Predictor v4\Race Predictor v4\scripts\daily_paper_loop.cmd" /sc daily /st 06:30 /f
```

### Starting the 8-week forward window

Building and verifying the window's freeze/reset machinery is this stage's job;
_starting_ the clock is a separate, deliberate operational decision this stage
does not make on anyone's behalf — `daily_paper_loop.run()` only ever calls
`verify_window`, never `start_window`. As of this commit the real
`data/execution/forward_window.json` does not exist and the window's status is
`not_started`. To start it (once, when ready to begin the 8-week evidence
period for real):

```python
from execution import window
window.start_window(hashes=window.compute_state_hashes())
```

Every subsequent `daily_paper_loop` run will then verify against those frozen
hashes and reset the clock to zero — recording exactly what changed — the
moment any model file, the selection lock, or the execution config block
changes.

### Real evidence produced by this stage

`data/execution/gap_ledger.json` and `data/execution/daily_loop_run.json` are
genuine artifacts from running the loop against the real production card and
`data/races.db` (gitignored, unchanged convention from Stage 5's
`backtest_snapshots.db`) twice on 2026-07-28 to prove idempotency; both runs
used `--no-scrape`, so today itself is correctly recorded as a **gap**
(`scraper_outage`) rather than falsely marked captured. `reports/forward_validation_20260728.md`
is the corresponding dated report. None of this is fabricated evidence toward a
GO — the forward gate still reads 9 of 9 criteria failed, `weeks_elapsed
qualifying = 0.0`, exactly as it should for a window that has not been started.

### Tests

New: `tests/execution/test_window.py` (9), `tests/execution/test_gap_ledger.py`
(12), `tests/scripts/test_daily_paper_loop.py` (4, fully isolated on `tmp_path`
— never touches `data/races.db` or the real gap ledger). Extended:
`tests/execution/test_snapshots.py` (+1, no-backdate-parameter signature check),
`tests/test_refresh.py` (+1, capture-failure-is-non-fatal),
`tests/ui/test_forward_validation.py` (3 fixed for the new dual-artifact
`load_run` signature).

```
# Targeted (Stage 6 files only)
# 84 passed in 3.45s

# Full suite
# 1993 passed in 153.89s      (Stage 5 recorded 1966; +27 new tests this session)
```

### Files not touched

`.claude/`, `.design-md/`, `images/`, `review/`, `tools/`, `data/audit/`,
`data/backups/`, `implementation_plan.md`, `v4prompts.md`,
`RACE_PREDICTOR_REPAIR_PROMPTS.md`, `tests/test_train_lgbm.py`,
`STAGE_6_PROMPT.md`, `data/source_health.json`,
`memory/live-repair-00-prompt-sequence.md`,
`memory/live-repair-triage-20260727.md`, `reports/testrun/`, and the
pre-existing `reports/*` artifacts predating this stage — all unrelated user
work, left exactly as found.

`2d7f164` — `feat: start live point-in-time capture and daily paper loop`,
parent `f2d25dd`, 19 files, 2478 insertions, 11 deletions. Scoped to Stage 6
only — see "Files not touched" above.

### Stage 6 verdict

**STAGE 6 PASS** · **MODEL NO-GO** · **CAPTURE INACTIVE** (today recorded as a
gap — both proving runs used `--no-scrape`) · **FORWARD WINDOW NOT STARTED** ·
**DEPLOYMENT PAPER-ONLY**

---

## Programme summary — Stages 1 to 6

| stage | verdict                      | what it established                                                                                                                                                        |
| ----- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | PASS                         | live market identity repaired — per-source `race_id` fragments the odds map, so the cross-book map keys on `race_time`                                                     |
| 2     | PASS                         | proxy and freshness integrity — source health, staleness, fail-closed on unknown                                                                                           |
| 3     | PASS                         | full-field probability and EV reconciled; `won_prob_normalized` is the honest headline                                                                                     |
| 4     | AUDIT PASS / **MODEL NO-GO** | on an untouched 904-race window no line significantly beats the de-vigged market; `value_win_prob` is ~93% price echo; CLV −13.4%                                          |
| 5     | PASS / **MODEL NO-GO**       | realistic execution, matched-race baselines with intervals, anti-mining selection, PASS-by-default gate, conservative staking, an 8-week forward gate no backtest can open |
| 6     | PASS / **MODEL NO-GO**       | live point-in-time capture wired in; one idempotent daily loop (capture/gate/issue/settle/report); gap-adjusted 8-week forward window built and tested — not yet started   |

**Final programme verdict: the model is not ready to bet.** Every honest
measurement across two independent windows says the same thing — the system does
not beat the closing line, and on the Stage-5 window it loses to it by more than
backing the favourite does. Stage 6 does not change that: it is the apparatus
that lets eight weeks of _real_ forward paper evidence answer the question
properly once the window is deliberately started, and it refuses to answer the
question any other way in the meantime.
