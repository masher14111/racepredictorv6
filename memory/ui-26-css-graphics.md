---
name: ui-26-css-graphics
description: In-house CSS/SVG visual assets (no stock photos) for the betting-desk design system — ui/_icons.py icon set + illustrations + meters, _design.py hero/meter/icon-plate CSS, favicon.svg; how to reuse them.
metadata:
  type: project
---

# In-house CSS/SVG Graphics (Prompt 26, 2026-06-17)

Built a cohesive set of **in-house visual assets** — pure CSS/SVG, **no external
image files** — so the app's chrome is on-brand and recolours with the theme
instead of leaning on stock photos. Extends the Prompt-19 system ([[ui-19-redesign]]);
honours impeccable bans (no gradient text, no decorative glassmorphism — the
mask/blur used is _structural background texture_, not glass over content).

## What was added

- **`ui/_icons.py`** (new) — Streamlit-free SVG-string builders, same pattern as
  `ui/_components.py`:
  - `icon(name, size=18, stroke_width=1.6, cls="")` — the cohesive **icon set**
    (`_PATHS`): `race · runner · value · bankroll · calibration · trophy ·
selection · clock`. 24-grid line-art, round joins, **`stroke="currentColor"`**
    so a glyph inherits its container's `color` (theme-driven for free).
    `aria-hidden` (decorative; adjacent text is the label). Unknown name → valid
    empty svg (never raises).
  - `logo_mark(size, rounded=True)` — oxblood rounded tile + white finish-flag;
    self-contained colours (token + literal fallback, not currentColor) so it's a
    brand mark anywhere. **Source of `ui/static/favicon.svg`** (keep in sync).
  - `horse_illustration(size)` — line-art **horse head in profile** facing a
    checker finish-flag, ink silhouette + oxblood mane flicks + value-dim turf
    line. (A profile head reads as "horse" far more reliably than a small gallop —
    the first gallop attempt was an unreadable tangle; profile was the fix.)
  - `finishline_scene(size)` — lighter checker-ribbon-between-posts illustration
    for the "racing's done" empty state.
  - `strength_meter(value, segments=5, tone, label)` — segmented pip meter; the
    **fill count** carries the read (colour only reinforces), honest `—` on None.
  - `prob_gauge(prob, size, tone, label)` — semicircular arc dial via
    `stroke-dasharray` (`pathLength`-normalised); honest `—` on None.
- **`ui/_design.py`** — new CSS block "IN-HOUSE GRAPHICS": `.rp-ico`,
  `.rp-ico-plate{,.info,.value,.amber,.oxblood}` (tinted icon plate),
  `.rp-kpi .k-head` (KPI label + leading plate), `.rp-meter*`, `.rp-gauge*`,
  `.rp-illus`, and the **hero motif**: `.rp-empty.hero-motif` + `.rp-hero` — a
  layered OKLCH mesh-gradient with a concentric **track-rail** line motif
  (`repeating-radial-gradient`) faded by a radial `mask-image`. All tokens.
- **`ui/static/favicon.svg`** — standalone logo mark (literal colours).
- **Wired in context** (`ui/_components.py` + `ui/app.py`): `kpi(..., icon=,
icon_tone=)` adds a plate; `empty_state(..., illustration=)` switches to the
  in-house motif hero. Landing KPI rail now: Races(race/info) · Value bets
  (value/value) · Selections(trophy/oxblood) · Each-way(selection/amber). Cold
  start → horse hero; filtered-empty → finish-line scene.

## How to reuse

- **An icon next to text:** `f'<span class="rp-ico-plate value">{I.icon("value",16)}</span>'`
  — pick the plate class on the **locked colour role** (value→value, info→prob/race,
  amber→each-way/caution, oxblood→brand/rank-1). Or bare `I.icon(name)` inside any
  element with a `color:` to tint a line glyph.
- **A KPI tile:** `C.kpi(label, value, sub, icon="trophy", icon_tone="oxblood")`.
- **A strength read:** `I.strength_meter(0.62, tone="info", label="62%")`; a single
  hero number → `I.prob_gauge(0.62, 84, tone="value")`.
- **An empty/hero state:** `C.empty_state(title, body, illustration=I.horse_illustration(150))`
  (motif background auto-applied) or a standalone splash with the `.rp-hero` class.
- Add a glyph: drop a 24-grid `currentColor` path into `_icons._PATHS`; it's
  auto-covered by `test_icons.py`'s "every named icon" contract.

**Why:** the redesign brief prefers crisp, on-brand graphics over stock photos;
in-house SVG also recolours with the theme, ships in the markup (no static-serving
dependency or extra request), and is headless-testable as a string.

**How to apply:** reach for these instead of emoji/photos in any new surface; keep
glyphs `currentColor` + on the locked roles so colour never carries meaning alone.

## Verify / regenerate

- Tests: `tests/ui/test_icons.py` (12) — valid SVG, currentColor, fill-count read,
  honest `—`; full `tests/ui` green (81). Screenshots: `.design-md/assets-gallery.png`
  (catalog, real stylesheet) + `.design-md/assets-in-app.png` (live landing KPI rail).
- Regenerate gallery: `python .design-md/_assets_gallery.py && python .design-md/_shoot.py`.
- Streamlit gotchas unchanged ([[ui-19-redesign]]): restart the server to pick up
  `_design.py`/`_icons.py` edits (sys.modules cache); on Windows stop it via
  PowerShell `Stop-Process`, not `pkill`.

See [[ui-19-redesign]], [[ui-24-polish]], [[ui-23-detail-pages]].
