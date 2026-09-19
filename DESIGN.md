# Race Predictor v4 — compact design
Keep this file below 200 physical lines. Stable contract; progress belongs in memory/improvement/STATE.md.
The full prior 863-line UI specification is preserved at docs/archive/memory-setup-20260918/DESIGN.md.
Read that reference before visual changes; its historical implementation claims still need current verification.

## Architecture
- Product: UK/Irish racing win/place probabilities, transparent model evidence and paper execution.
- Runtime: Python/Streamlit on Windows; use project environment; EUR; display Europe/Dublin times.
- Flow: sources -> canonical race/runner/market records -> point-in-time features -> models -> predictions -> UI/paper ledger.
- Source rows retain provider IDs, acquisition/publication times, provenance and market type.
- Canonical runner identity includes race and horse; a horse ID alone cannot identify a result.
- Keep actual declarations distinct from historical fallback values.
- WIN and PLACE are different price books; runner attributes may be shared but market facts must stay associated.
- Preserve immutable odds/text observations; as-of reads cannot access later values.
- CatBoost: independent binary target models, with disclosed independent and priced variants.
- LightGBM: per-race grouped-softmax objective, not generic multiclassova.
- Calibrate and combine on chronological out-of-sample predictions; preserve one probability distribution per full race.
- Reference market probabilities and executable prices have different roles and timestamps.
- LLM branch: archived text -> strict grounded extraction -> optional model features, never direct probability overrides.
- Candidate models/artifacts are isolated; preserve existing champion and rollback metadata.
- Prediction JSON changes stay additive unless an explicit versioned migration is required.
- Execution distinguishes PASS decisions, issued paper tickets, fills, settlements and forward-qualified evidence.
- Engineering readiness does not imply market outperformance or complete prospective validation.
- Model/release gates remain fail-closed; missing data supports no-bet, not fabricated confidence.
- A chronological development/final/forward protocol is frozen before candidate comparisons.
- Preserve testable headless helpers; separate formatting logic from Streamlit rendering.

## Code map
| Area | Main locations |
|---|---|
| Source acquisition / normalization | scraper/, utils/normalizer.py, features/fuse.py |
| Historical / live feature parity | features/builder.py, features/derive.py, features/engine.py |
| Model definitions / training | models/train.py, models/tuner.py, models/lgbm_softmax.py, models/train_lgbm.py |
| Calibration / market comparison | models/calibration.py, models/devig.py, models/head_to_head.py |
| Serving / value | models/predictor.py, models/predict_unified.py, models/value.py |
| Historical evaluation | backtest/, scripts/calibration_audit.py |
| Capture / settlement / gates | execution/, scripts/daily_paper_loop.py |
| Text features | llm/text_features.py, scraper/spotlight.py |
| User interface | ui/app.py, ui/pages/, ui/_components.py, ui/_design.py |

## UI invariants
- Streamlit remains the application, not a standalone HTML rewrite.
- Canonical palette/tokens: ui/_design.py; native themes: .streamlit/config.toml.
- Components: ui/_components.py; icons: ui/_icons.py; no duplicated page palettes.
- Keep the shipping dark indigo identity and the current shared design injection.
- Preserve working current user edits; the archive is not an instruction to restore older code.
- Green means positive EV; amber means caution/each-way; red means loss/danger.
- Chart series use the separate categorical palette, not semantic status colors.
- Pair color with a label/icon/shape; never rely on color alone.
- One typeface: Inter, everywhere. Weight, size and tracking carry the hierarchy.
- `--f-display`/`--f-num` are roles that resolve to Inter; `--f-code` is the only real monospace, for code blocks.
- Matches the style reference, verified glyph-for-glyph at matched size, not by eye.
- Wordmark: 800 weight, tracking -0.035em, product name in `--brand-text`, version in a pill.
- Tabular numerals for aligned tables/odds/times; avoid forcing them on large standalone values.
- Text contrast at least 4.5:1 (large text 3:1); visible keyboard focus.
- Preserve responsive containers/cards and usable 375px mobile layouts.
- Native charts live in actual Streamlit containers, not embedded inside HTML strings.
- No fake persistent AI side rail, draggable dashboard, casino urgency or decorative emoji.
- No glass/blur, gradient text, unnecessary animation or new decorative asset systems.
- Reduced-motion users see all content immediately.
- Probability meters retain --pbw behavior and always-visible numeric labels.
- Predicted headline win % uses the full-race-normalized probability.
- Realized rates show wins/denominator, interval and evaluation window; distinguish predicted from realized.
- Show paper-only/NO-GO, stale data, incomplete books and missing coverage plainly.
- Prices are ratios, stakes are EUR money; retain the existing fractional/decimal presentation.
- Keep established semantic colors out of unrelated price/bookmaker highlights.
- Chart labels use neutral ink; identity resides in marks; use readable mark size and surface gaps.
- A/E charts center on 1.0; both directions indicate miscalibration, not automatic positive/negative value.
- Equity fills use starting bankroll as baseline; never imply profit by filling from zero.
- Current charts/theme helpers own axes, legend and palettes; do not restore deleted legacy UI modules.

## Scientific invariants
- Every training/tuning/calibration boundary keeps complete races together.
- Fit transforms and historical statistics only on admissible earlier information.
- Preserve actual non-runner and full-field behavior.
- Compare all candidates on the same eligible races and intended decision cutoff.
- Scoring: race log loss, Brier, reliability; execution: CLV, costs, ROI, drawdown and counts.
- Report probability quality separately from bet selection and realized returns.
- Bootstrap races or days; raw/log/percentage quantities must be labelled accurately.
- Keep a candidate off when missing input evidence or comparative validation.
- Historical low errors from the supplied article are an unverified research lead, not an acceptance target.
