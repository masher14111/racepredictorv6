---
name: ui-24-polish
description: Final /impeccable polish pass — purposeful motion, loading skeletons, responsive verify, a11y audit on the dark betting-desk UI (2026-06-17)
metadata:
  type: project
---

Final UI polish pass (`/impeccable polish` → `animate`/`adapt`/`audit`) over the
Prompt-19 dark "betting-desk" surface. Builds on [[ui-19-redesign]]. Suite green:
**1021 passed / 3 skipped** (3 new component tests). Screenshots in `.design-md/`
(`polish-after-landing*.png`, `polish-after-suggestions.png`, `polish-audit-sidebar.png`).

## Motion (all gated under `@media (prefers-reduced-motion: no-preference)`)

The default / reduced-motion state is the fully-visible END state, so headless
renders and pinned tabs never ship blank (the reveal-trap the skill warns about).

- **Smooth prob-bar fills** — `ui/_components.py:prob_bar` writes the target as a
  CSS custom property `--pbw` (was inline `width`); `.pb-fill` keyframes `pb-grow`
  from 0→`var(--pbw)`. Same pattern for SHAP "why" bars via `--bw` + `why-grow`
  (diverging bars sweep from centre). Base `width:var(--pbw)` lives OUTSIDE any
  media query → correct under reduced motion.
- **Staggered reveals** on true sibling groups only (not a uniform reflex): KPI
  tiles (`.rp-kpis > .rp-kpi`, nth-child 1–6 delays), table rows (`.rp-tbl tbody
tr`, 1–6 then n+7), `.why > .why-row`. Keyframes `rp-rise`/`rp-rise-sm`/`rp-fade`.
- Alerts (`[data-testid="stAlert"]`) rise in; paper-note fades in.
- Keyframes verified present in live DOM: `pb-grow, why-grow, rp-rise, rp-shimmer,
rp-fade`. `prob_bar(0.9)` → computed width 48.6px, `animationName: pb-grow`.

## States

- **Loading skeletons** for the ~1.7s inference / value scan. New `_components`
  helpers `skeleton_rail(n)` (clamped 1–6), `skeleton_card()` (header + 4 runner
  rows), `loading_skeleton(label, n_cards, rail)` — pure-CSS shimmer (`.sk::after`,
  `@keyframes rp-shimmer`). Painted into an `st.empty()` placeholder that flushes
  to the browser BEFORE the blocking call, then `.empty()`'d. Wired in
  `ui/app.py` (Refresh → `_refreshing` rerun → skeleton → `_run_predictor`) and
  `ui/pages/11_Todays_Suggestions.py` (value scan). Removed the old `st.spinner`.
- **Empty states** were already honest (`ui/logic.py:empty_state`, suggestions'
  "No value bets today" scanned-count copy) — left as-is, they pass the bar.

## Responsive (verified mobile→desktop)

- Added `@media (max-width:980px)` tablet breakpoint: KPI rail 4→3 col, detail-grid
  →1 col, relaxed padding. KPI rail collapses to 2 col at phone width.
- `overflow-wrap:anywhere` on long-word headings (`.rp-appbar h1`,
  `.rp-card-hdr .venue`, `.rp-empty h3`, `.sg-sub .venue`) — no heading overflow.
- Runner tables: `.rp-body` is `overflow-x:auto` → tables SCROLL within the card on
  mobile (all columns reachable, never clipped). Document-level horizontal overflow
  = 0px at phone width.

## A11y / audit findings

- **Content contrast is strong, all AA-pass**: venue 14:1, h1 16.5:1, table 9.5:1,
  th 6.4:1, in-card `rp-open` link 5.66:1.
- **False alarm caught**: a naive contrast walk reported sidebar nav links at
  3.3:1 / 1:1 — a DOM-layering artifact (the `effBg` walk hit the light outer
  `section[stSidebar]` wrapper `rgb(240,242,246)` and the translucent oxblood
  active-pill, not the dark navy surface actually behind the text). Screenshot
  (`polish-audit-sidebar.png`) confirms nav links are clearly legible light-on-navy.
  Lesson: trust the rendered pixel over a computed-bg walk on translucent/layered DOM.
- **Real fix**: Streamlit logged `Invalid color passed for textColor in
theme.sidebar: ""` (9×). Added `[theme.sidebar] textColor = "#c8d3e8"` to
  `.streamlit/config.toml` to match the forced nav text — silences the warning and
  hardens nav legibility if the `!important` CSS override ever regresses.
  **NOTE: `[theme]` changes need a server restart to take effect.**

## Files touched

`ui/_design.py` (motion + skeleton CSS + tablet breakpoint + wrap), `ui/_components.py`
(custom-prop bars + 3 skeleton builders), `ui/app.py`, `ui/pages/11_Todays_Suggestions.py`,
`.streamlit/config.toml`, `tests/ui/test_components.py` (+3 tests, updated bar assertions).
