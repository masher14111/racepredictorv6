---
name: ui-19-redesign
description: Prompt 19 UI redesign — the dark "betting-desk" design system (ui/_design.py), reusable components (ui/_components.py), rebuilt landing + new race-detail page; tokens, roles, screens, gotchas. Prompts 20-24 extend this.
metadata:
  type: project
---

# UI Redesign (Prompt 19, 2026-06-16)

Replaced the per-page duplicated CSS (Prompt-18 audit finding) with **one shared
design system** and rebuilt the two canonical prediction surfaces on it. Register
= **product** (design serves the tool); single power-user, numbers-forward, paper
betting research (NOT real money).

## Direction — dark "betting-desk"

Deep charcoal ground, **oxblood** brand accent (racing silks / turf-club leather),
a precise **green reserved strictly for value / positive-EV**, **amber** for
each-way, cool **blue** for probability bars + links. Premium, the opposite of
both casino neon and default-Streamlit grey. Serif (Spectral) for titles/venues,
sans (Inter) for UI, mono (IBM Plex Mono) for all numerics — paired on a contrast
axis, strict roles.

## Files

- **`ui/_design.py`** — the single stylesheet. `_TOKENS` (OKLCH, source of truth),
  `_CSS` (Google Fonts @import + tokens + native-widget overrides + the `rp-*`
  component layer), `inject_design()` (call once after `set_page_config`),
  `plotly_layout(**o)` (themed Plotly layout, resolved hex since Plotly can't parse
  oklch), `PALETTE` (hex dict for non-CSS paths).
- **`ui/_components.py`** — Streamlit-free HTML-string builders over the `rp-*`
  classes (headless-testable). Marks (`mark_finishline`, `mark_loader` — crafted
  inline SVG, not the heavy raster source), `app_bar`, `brand_lockup`,
  `trust_badge`, `kpi`/`kpi_rail`, `rank_badge`, `prob_bar`, `value_badge`,
  `ew_badge`, `ev_text`, `confidence_for`/`confidence_chip`, `book_chips`,
  `why_drivers`, `empty_state`, `footer`, `section`, `race_slug`/`find_race`,
  formatters.
- **`ui/app.py`** — rebuilt **Today / race-list landing**: app-bar (serif title +
  model-trust badges), KPI rail (Races / Value bets / Selections / Each-way),
  venue-grouped expanders of race cards (rank, runner+jockey, best price + per-book
  chips, win% bar, confidence, EV), **sort-by Rank | Value edge**, honest
  empty/stale states.
- **`ui/pages/8_Race_Detail.py`** — new **race-detail** page, reached from each
  card's "Card & drivers →" link (`?race=<slug>`). Per-runner win/place/show bars,
  price + books, EV, confidence, and **SHAP "why" drivers** via `models/explain.py`
  with an honest fallback.
- **`.streamlit/config.toml`** — `base="light"` (intentionally; see gotchas),
  `primaryColor="#bc484b"`, `[server] enableStaticServing=true`.
- **`ui/static/racecourse-dawn.webp`** — dawn-racecourse hero photo (copy of
  `images/loginsplash.webp`) for the cold-start empty state, served at
  `app/static/`.
- **`tests/ui/test_components.py`** — 12 tests over the pure helpers.

## Design tokens (all text AA-verified, OKLCH→sRGB)

Surfaces (depth via lightness, not shadow): `--bg #0c1015`, `--surface #13181e`,
`--surface-2 #1b2128`, `--surface-3 #262c33`, `--border #31363d`. Ink ramp:
`--ink #edeff1` (15-16:1), `--ink-2 #babec3` (10.2:1), `--muted #969ca3`
(6.9:1 on bg, **6.46:1 on surface** — fixes the old sub-AA #6b7689 4.28:1).
Accents: `--oxblood #bc484b` (white-on 5.0:1), `--value #5bcc80` (8.85:1),
`--info #549de5` (6.2:1), `--amber #ebae51` (9.1:1), `--danger #ec5448`. Fixed-rem
type scale, 4px spacing, semantic z-index scale, ease-out-quint motion with a
`prefers-reduced-motion` branch.

## Colour roles (locked — do not let drift in Prompts 20-24)

- **oxblood** → brand marks, **rank-1 badge** (every other rank is a neutral
  outline — no rainbow), primary action / buttons.
- **green** → value / positive-EV **ONLY** (keeps the signal uncontested).
- **blue (info)** → probability bars + links.
- **amber** → each-way / caution.
- Colour never carries meaning alone — value pairs with the word VALUE, best price
  with ★, each-way with E/W, confidence with a label + dot count.

## Verification

- Contrast: every body pair ≥4.5:1 (computed in-browser + Python). Muted-on-surface
  6.46:1.
- Responsive: desktop 1440, mobile 390 — KPI rail collapses to 2-col, detail
  two-column runner block (`.rp-detail-grid`) stacks. Screenshots in `.design-md/`
  (`final-landing*.png`, `final-detail-mobile.png`).
- `pytest`: 44 ui/logic + 12 new component tests pass; no console errors (one
  benign Streamlit `theme.sidebar.textColor ""` warning).
- impeccable bans honored: no side-stripes, no gradient text, no per-section
  eyebrow, no hero-metric template, rank-1-only brand colour.

## Gotchas (bit me — read before extending)

1. **`base="light"` stays.** Pages 1-7 each force their own LIGHT `.stApp` inline;
   flipping to `base="dark"` re-introduces white-on-white on them. The dark theme
   is delivered by scoped CSS that explicitly recolours every native widget, so it
   doesn't need the Streamlit base. Revisit when 1-7 migrate.
2. **Never set `font-family` on bare `span`.** `.stApp span { font-family }`
   clobbers Streamlit's Material-icon ligature spans (`[data-testid="stIconMaterial"]`)
   so the collapse chevron renders as raw text "keyboard_double_arrow_left". Set the
   font on text containers only; spans inherit it. (Belt-and-suspenders rule
   restoring the icon font is also in `_design.py`.)
3. **Streamlit caches imported modules in `sys.modules`.** A browser reload re-runs
   `app.py` but NOT edits to imported `ui/_design.py` / `ui/_components.py` —
   **restart the server** to see stylesheet changes. And `pkill -f streamlit` does
   NOT kill the process on Windows; use PowerShell `Stop-Process` on the
   `Win32_Process` whose CommandLine matches, or stale servers pile up on the port.
4. **SHAP drivers are honestly empty for live runners.** `data/features.parquet` is
   a 5-row synthetic fixture, so `explain()` finds no feature row for live
   `horse_id`s → "driver breakdown unavailable". Fixing it = regenerate the feature
   store (Prompt-18 P0, out of scope here). Same root cause as blank jockey/trainer.

## Backend metrics surfaced

Trust badges read **Win AUC** (`models/catboost_v3nf_meta.json` →
`targets.won.test_auc` = 0.670) and **calibration ECE**
(`docs/calibration/metrics.csv`, v3nf/won/calibrated = 0.002) — the price-free
calibrated model. The cache (`predictions.json`) is stale (2026-06-15) and predates
the `confidence`/`data_completeness` fields, so `confidence_for` falls back to a
transparent proxy until the cache carries them.

See [[ui-18-audit]], [[model-15-feature-selection]], [[model-17-inference]],
[[model-11-calibration]], [[model-16-value-detection]].
