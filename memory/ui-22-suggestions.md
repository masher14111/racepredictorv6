---
name: ui-22-suggestions
description: Bet-suggestion engine (Prompt 22) — tiers, rationale source, curation, and the Today's Suggestions page
metadata:
  type: project
---

Prompt 22 — the **bet-suggestion engine**: turns a day's scored races into a
short, ranked, explainable list of "what should I bet on today?". Curation layer
on top of [[model-16-value-detection]] (value gates), [[model-15-feature-selection]]
(SHAP) and the price-free `v3nf` model. Done 2026-06-16; suite 998 pass / 3 skip.

**Backend — `models/suggestions.py`**

- `suggest_bets(races, *, value_config, config, bankroll, feature_rows, explainer, generated_at)` → `SuggestionBook`. `suggest_from_cache()` is the zero-arg UI wrapper (reads `data/predictions.json`, wires `_load_feature_rows` + `_load_explainer`).
- **Selection is NOT re-implemented** — every gate (de-vig, odds band [2.0,6.0], EV floor, support, confidence) comes from `find_value_bets`/`ValueConfig`. The layer only **tiers, curates, explains**. This keeps the suggestion set exactly the backtester-validated favourite band — do not widen it here.
- **Tiers** (`assign_tier`): `Strong` requires ALL three bars — `edge_pct ≥ strong_edge_pct (0.15)` AND `value_confidence ≥ strong_confidence (0.60)` AND complete-enough data (`completeness ≥ strong_completeness 0.55`, not first-time, `data_confidence != "low"`). Anything that clears the value gates but misses a Strong bar is a `Lean`. `Pass` is the implicit state of a runner that clears NO value gate (never surfaced). Unknown completeness ⇒ Lean (honest, never Strong). Returns `(tier, reasons)` — human strings shown in the card tooltip so the tier is never opaque.
- **Rationale** (`build_rationale`): "Backed by {plain-English phrases}." from the model's strongest **positive** SHAP drivers (`_DRIVER_PHRASES` maps feature→phrase, e.g. `horse_speed`→"strong recent speed figures"), then a value clause "Model rates it X% vs the market's Y% — a +Zpp edge at O.OO (EV +E%)." When no feature row exists for the runner (live `features.parquet` is still the synthetic fixture — see [[model-09-baseline-audit]]) it **degrades to the value clause alone** rather than fabricating form.
- **Curation**: `max_per_race` (default 1, anti-spam) + optional `max_suggestions` global cap. Ranked Strong-before-Lean, then edge desc, then EV desc. `SuggestionBook` reports honest coverage: `n_races`, `n_races_with_value`, `n_strong`, `n_lean` — a no-value day yields an empty book, not an invented pick.
- Config: optional `suggestions:` block in config.yaml (`SuggestionConfig.from_config`); all keys optional, absent ⇒ the validated defaults above.

**UI**

- Page `ui/pages/11_Todays_Suggestions.py`: KPI rail (Suggestions / Strong / Lean / races-with-value of-scanned), ranked `C.suggestion_card(...)` + one-click `place_suggestion_widget(...)` per pick, honest `empty_state` when no value. Cached on `round(tracker.bankroll,2)` so stakes track the live bankroll. Per-card currency via `currency_for_venue` (a day spans UK + IRE).
- `ui/_components.py`: `tier_badge` (Strong = filled green / Lean = quiet outline — both on the **green value channel**, word carries meaning), `suggestion_card` (`sg-*` markup: header rank+tier+market, connections sub-line, `sg-stats` grid Offered/Fair/Edge/EV/Win%/Stake/Confidence, rationale, optional SHAP "why" bars).
- `ui/_betting.py`: `place_suggestion_widget(s, ccy)` pre-fills the engine's **fractional-Kelly suggested_stake** + offered price (unlike `place_bet_widget`, which re-derives from the sidebar strategy), routes through `tracker.place_paper_bet` (all guards) with `notes=f"suggestion:{tier}"`.
- CSS in `ui/_design.py` (`.tier*`, `.sg-*`). Colour roles honoured: green = value only.

**Known live-data caveat**: with the current stale `predictions.json` + synthetic `features.parquet`, `suggest_from_cache()` returns few/zero suggestions and rationales are value-only — engine degrades gracefully. Tested end-to-end on synthetic races (`tests/models/test_suggestions.py`, 26 tests). Real output needs the data fix in [[model-09-baseline-audit]] (fresh predictions + real feature store).
