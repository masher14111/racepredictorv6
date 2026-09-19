---
name: live-repair-03-probability-ev
description: Stage 3 PASS — one race-level EV eligibility decision (ev_eligible/ev_gate) now gates every EV surface from inside the cache; the fair line de-vigs one complete bookmaker board (never the best-price overlay); every persisted EV recomputes exactly from persisted inputs; predictions.json rebuilt from a genuinely fresh scrape with all runners serialized
metadata:
  type: project
---

# Stage 3 — full-field probability, fair-price, and EV reconciliation (2026-07-27)

Stage 3 of the programme in [[live-repair-00-prompt-sequence]]. Depends on
[[live-repair-01-market-ingestion]] and [[live-repair-02-proxy-freshness]] (both
PASS, joint commit `9ce4159`). Full detail in `HANDOFF.md`, section
"2026-07-27 — Stage 3". **Stage 3 commit: `d7f7872`**
(`fix: reconcile full-field probability and ev`, 23 files — also finally lands
predictor's Stage 1 hunks and fuse's Stage 2 hunks, inseparable since Stage 0).

## The calculation contract (do not fork it again)

- **One race-level EV decision.** `models/predictor.py::_race_ev_gate` runs once
  per race in `_build_race` and is persisted as `ev_eligible` + `ev_gate`
  (PASS reasons, reference source, overround, max odds age, computed_at). When
  ineligible, `_suppress_ev` blanks `expected_value`/`value_edge`/`ev_catboost`/
  `ev_lgbm`/`market_prob` and forces `value_bet=False` on **every** runner;
  `models.value.find_value_bets` returns `[]` for any race dict stamped
  `ev_eligible: false`, so the suggestion engine and every UI badge inherit the
  same PASS. No surface may re-derive an EV the cache refused to carry.
- **Reference vs executable prices.** The fair line (`market_prob`, `fair_prob`)
  de-vigs ONE complete bookmaker board — the first source in
  `_BOOKMAKER_SOURCES` order that prices the entire field
  (`_reference_decimals`), falling back to the fused consensus line, and NEVER
  the per-runner best-price overlay. EV always uses the executable best board
  price: `EV = model_prob × executable − 1`. Both are persisted per runner
  (`reference_odds`/`reference_source` vs `best_odds`/`best_book`).
- **Exact reproducibility.** Probabilities are quantised to their persisted 4dp
  (and prices to 3dp) BEFORE any multiply, so
  `expected_value == round(value_win_prob*best_odds−1, 4)`,
  `ev_catboost == round(catboost_win_prob*exec−1, 4)`, `ev_lgbm` likewise,
  `value_edge == round(value_win_prob−implied_prob, 4)`, and
  `decimal_odds == round(1/implied_prob, 3)` hold **exactly** in the cache
  (zero-tolerance identities; the old report needed a 0.02 tolerance).
- **Full field.** `race["runners"]` carries every valid runner; `selections`
  stays display-only top-3. Non-runners are dropped before normalization
  (Stage 3 WIP, retained). `check_output_invariants` invariant 6 flags any EV
  leak on a PASS race and any partial market line on an eligible race.

## Verified live evidence (2026-07-27 08:51 Dublin)

Fresh scrape → rebuild → `python -m scripts.reconcile_prob_to_ev` (live mode,
no provenance rewritten): **35 races / 356 runners, all 356 serialized** (closes
handoff issue #4 — was 170/514 with no `runners` key), all races EV-eligible,
reference book `livescorebet` everywhere (overrounds 0.147–0.798, the big ones
being 23-runner Galway festival fields), odds ~224s old at compute (TTL 900s),
**0 invariant violations, 0 EV identity failures**, 3 value bets. The Windsor
17:50 pick shows the split working: reference 2.88 (livescorebet) de-vigged,
EV on executable 3.0 (paddy_power).

`scripts/reconcile_prob_to_ev.py` no longer rewrites `fetched_at`/`stale` in
its default mode (closes issue #5): live mode audits `data/predictions.json`
as shipped; `--synthetic` (dates +1d, provenance rewritten) writes separate
`*_synthetic.*` files under an explicit banner and evidences arithmetic only.

## Boundaries that must hold

- `find_value_bets`' complete-reference-book gate applies when `cfg.devig` is
  true; with `devig=False` there is no fair line and per-runner price gaps skip
  only that runner (tests pin both).
- Curation gates (min_confidence, min_prob, edge bars) legitimately narrow
  picks below the flagged `value_bet` set — flags are core-gate + race gate;
  picks are flags ∩ curation. They can never widen it.
- `value_win_prob` remains market-adjusted (F-L recalibrated) — issue #8, Stage
  4 scope. Galway 17:10 "Witches Familiar" (value_win_prob 0.7251 vs market
  0.3448, EV +63%) is the reminder that an honest calculation contract does not
  make the model right. Paper-only; CLV still unproven.

## Sources on the day

livescorebet OK (356 rows / 35 races, fresh), boylesports failing (403 through
and without proxy — its 2-day-old cache rows were correctly TTL-dropped),
paddy_power reachable and contributing board prices to `odds_by_book` (Windsor
best price above) but still 0 rows in `data/live_odds.parquet` (issue #11,
open).
