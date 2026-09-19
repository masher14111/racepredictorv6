# DESIGN.md — Race Predictor v4

The visual contract for the Streamlit dashboard. Every rule here is a token in
`ui/_design.py`, a class in the `rp-*` component layer, or a line in
`.streamlit/config.toml`. **There is no standalone HTML page** — this app is Streamlit,
and the design system is injected CSS over markup that `ui/_components.py` builds as
pure strings.

Colour values were validated with the dataviz skill's `validate_palette.js` against this
app's own surfaces, not eyeballed. The commands and their recorded output are in §2.

---

## 1. Visual theme & atmosphere

**A betting desk at dusk.**

Near-black ground with a violet ambient field bleeding in from the corners. Opaque,
hairline-bordered cards float on that field. Inside the cards nothing decorates — the
numbers are the only loud thing on the page.

| | |
|---|---|
| Mood | Quiet, instrument-grade, nocturnal. A trading terminal that happens to be about horses. |
| Register | Professional, never casino. No neon, no gradients on text, no glassmorphism, no emoji. |
| Density | Dense where it is scannable, airy where it is structural. Charts fill their cards edge to edge; cards are separated by real space. |
| Motion | Purposeful only. Things arrive; nothing loops. |

**One sentence:** *dark, precise, and boring on purpose, so that a price, an edge and a
stake are the three things your eye lands on.*

### Taken from the style reference

- The **KPI stat-tile row** — a horizontal rail of bordered tiles, each with a tone accent.
- The **bordered card grid** — cards on a field rather than stacked full-width blocks.
- The **ambient glow** — radial violet wash on the page ground, outside the cards.
- The **chart density** — titles live in the card header, not in the plot; tight margins;
  small muted tick text; horizontal legend above the plot.

### Deliberately *not* taken

- **The right-hand AI panel.** Streamlit has no persistent side rail; faking one with
  columns breaks on every rerun and destroys the mobile layout.
- **Draggable, resizable widgets.** Streamlit's layout is declarative and re-executed
  top-to-bottom. There is no stable widget geometry to drag.

Do not attempt either in a later pass.

---

## 2. Colour palette & roles

### 2.1 The contract

Three colours **carry meaning and are reserved**. They may never be used for decoration,
for chart series identity, or for anything that is not the thing they mean:

| token | meaning | never |
|---|---|---|
| `--value` | positive expected value | "the second series", "best price", decoration |
| `--amber` | caution / each-way | a chart series |
| `--danger` | danger, loss, negative | a chart series, the brand |

And the standing rule from the original system, which survives unchanged:
**colour never carries meaning alone.** Every signal pairs with a word, a glyph or a
shape — `VALUE +9pp`, `STRONG`, `★`, the `1ˢᵗ` chip, the numeric `%` beside every
probability bar.

### 2.2 Tokens — dark (shipping)

Contrast measured against the card surface `#13181e`; all text tokens clear WCAG AA (4.5:1),
all mark tokens clear 3:1.

**Surfaces & ink** *(unchanged from the previous system — they were already correct)*

| token | hex | rgb | role | contrast |
|---|---|---|---|---|
| `--bg` | `#0c1015` | 12, 16, 21 | app ground | — |
| `--surface` | `#13181e` | 19, 24, 30 | card / panel | — |
| `--surface-2` | `#1b2128` | 27, 33, 40 | raised / header rail | — |
| `--surface-3` | `#262c33` | 38, 44, 51 | hover / selected base | — |
| `--border` | `#31363d` | 49, 54, 61 | hairline | — |
| `--border-strong` | `#484e55` | 72, 78, 85 | structural boundary | — |
| `--ink` | `#edeff1` | 237, 239, 241 | primary text | 16.5:1 |
| `--ink-2` | `#babec3` | 186, 190, 195 | secondary text | 10.2:1 |
| `--muted` | `#969ca3` | 150, 156, 163 | metadata, axis labels | 6.9:1 |

**Brand — indigo/violet (shipping)**

| token | hex | rgb | role | contrast |
|---|---|---|---|---|
| `--brand` | `#6d5cf0` | 109, 92, 240 | fills, primary button, rank-1 badge | white-on 4.73:1 |
| `--brand-text` | `#9085e9` | 144, 133, 233 | brand-coloured **text** and links | 5.71:1 ✓ AA |
| `--brand-deep` | `#5138c9` | 81, 56, 201 | pressed, borders on brand fills | — |
| `--brand-wash` | `#6d5cf0` @ 16% | — | tinted backgrounds | — |

> `--brand` itself is **3.77:1** on the card — legal as a fill or a mark, **not** as text.
> Any brand-coloured text uses `--brand-text`. White on `--brand` is 4.73:1, so a primary
> button with a white label is AA.

**Semantic — unchanged**

| token | hex | rgb | role | contrast |
|---|---|---|---|---|
| `--value` | `#5bcc80` | 91, 204, 128 | positive EV | 8.83:1 |
| `--value-dim` | `#44a264` | 68, 162, 100 | de-emphasised positive | 5.3:1 |
| `--amber` | `#ebae51` | 235, 174, 81 | caution / each-way | 9.10:1 |
| `--danger` | `#ec5448` | 236, 84, 72 | danger / loss / negative | 5.04:1 |
| `--info` | `#549de5` | 84, 157, 229 | probability bars, links, info | 6.23:1 |

Each `--*-wash` is its parent at 14–16% for tinted backgrounds.

### 2.3 Chart palette

Chart colour is a **separate namespace** from UI chrome. A chart series never wears a
semantic token, and a semantic token never becomes "series 4".

**Categorical — identity (which series).** Five hues, fixed order, assigned in sequence,
**never cycled**. Green, amber and red are deliberately absent: they are reserved.

| slot | token | dark | light (pages 2–6) | hue |
|---|---|---|---|---|
| 1 | `--series-1` | `#9085e9` | `#4a3aa7` | violet |
| 2 | `--series-2` | `#d95926` | `#c2521f` | orange |
| 3 | `--series-3` | `#199e70` | `#0f8a5f` | aqua |
| 4 | `--series-4` | `#3987e5` | `#2a78d6` | blue |
| 5 | `--series-5` | `#d55181` | `#c93f76` | magenta |

A sixth series does not get a generated hue. It folds into "Other", or the chart becomes
small multiples.

**Ordinal — position in a sequence** (odds bands, tiers). One hue, monotone lightness, so
the order is visible in the colour:

| | dark | light |
|---|---|---|
| step 1 (nearest surface) | `#4d40ab` | `#b3a9ef` |
| step 2 | `#6759d2` | `#8a7be2` |
| step 3 | `#8a7de8` | `#5b4ec6` |
| step 4 | `#b3abf2` | `#382e86` |

**Diverging — polarity around a baseline** (A/E around 1.0). Two hues plus a **neutral
grey** midpoint — never a hue at the midpoint, and deliberately **not** green/red:

| | dark | light |
|---|---|---|
| positive pole | `#9085e9` (slot 1) | `#4a3aa7` |
| midpoint | `#4a5565` | `#a8aeb9` |
| negative pole | `#d95926` (slot 2) | `#c2521f` |

> Why not green/red for A/E: **both** directions of miscalibration are bad. A/E 1.4 is not
> "good"; it is wrong in the other direction. A good/bad palette would lie about that.

**Chart chrome**

| token | dark | light | role |
|---|---|---|---|
| `--chart-ink` | `#babec3` | `#3b4658` | legend and value text |
| `--chart-axis` | `#969ca3` | `#6b7689` | tick labels |
| `--chart-grid` | `#1b2128` | `#edf1f6` | hairline gridlines, **solid, never dashed** |

### 2.4 Measured collisions, and what mitigates them

These pairs sit below the ΔE 15 normal-vision floor. They are documented rather than
"fixed", because the fix in each case is a placement rule, not a hex change.

| pair | ΔE | rule |
|---|---|---|
| `--value` ↔ slot-3 aqua | **14.4** | A chart that also carries EV-green marks caps at **two** categorical series. Status always ships icon + label. |
| `--danger` ↔ slot-2 orange | **5.0** | `--danger` is a chip/badge role only. **Never a chart mark.** |
| `--info` ↔ slot-4 blue | **6.7** | `--info` is UI chrome (links, probability bars). Slot 4 is chart identity. They never compete as identities in one view. |

### 2.5 The alternate identity — oxblood

The previous brand, kept fully specified and validated so it can be swapped back by
changing three token values and the chart slot order. Nothing else moves.

| token | hex | role | contrast |
|---|---|---|---|
| `--brand` | `#bc484b` | fills, primary button | white-on 5.04:1 |
| `--brand-text` | `#dc5e5b` | brand-coloured text | 4.92:1 ✓ AA |
| `--brand-deep` | `#902f33` | pressed, borders | — |

**Oxblood never enters the data layer.** It measures 8.4 ΔE from the chart orange and 8.9
from the magenta, and it shares a hue family with `--danger`. Under this identity the
chart palette leads **cool** so the warm chrome and the cool data never argue:

| slot | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| dark | `#3987e5` | `#d95926` | `#199e70` | `#9085e9` | `#d55181` |
| light | `#2a78d6` | `#c2521f` | `#0f8a5f` | `#4a3aa7` | `#c93f76` |

Same five hues as the shipping palette — only slots 1 and 4 swap. That is why the chart
palette works with either identity. Its ordinal ramp:
dark `#8c3539, #b04448, #d0585b, #e8999b` · light `#dd9698, #c96b6e, #bc484b, #7d2528`.

**Why indigo ships instead.** `#bc484b` is 3.54:1 on the card surface, below AA for text,
so it always needs a second lifted step to say anything. It sits 8.4 ΔE from the chart
orange. And it shares a hue family with `--danger #ec5448` — in an app where red means
losing money, a red brand competes with the warning. Indigo collides with nothing
semantic, so one hue can be chrome, ambient glow *and* chart slot 1. That single-hue
coherence is what makes the style reference work.

### 2.6 Validator — re-run these, don't trust the table

```bash
# categorical, shipping (indigo) — adjacent pairlist, real card surface
node scripts/validate_palette.js "#9085e9,#d95926,#199e70,#3987e5,#d55181" \
  --mode dark --surface "#13181e"

# same, all-pairs — the cap for scatter / bubble / small-multiples forms
node scripts/validate_palette.js "#9085e9,#d95926,#199e70" \
  --mode dark --surface "#13181e" --pairs all

# light twin, for pages 2–6 on white
node scripts/validate_palette.js "#4a3aa7,#c2521f,#0f8a5f,#2a78d6,#c93f76" \
  --mode light --surface "#ffffff"

# ordinal ramps (odds bands)
node scripts/validate_palette.js "#4d40ab,#6759d2,#8a7de8,#b3abf2" \
  --mode dark --surface "#13181e" --ordinal
node scripts/validate_palette.js "#b3a9ef,#8a7be2,#5b4ec6,#382e86" \
  --mode light --surface "#ffffff" --ordinal
```

Recorded results at time of writing:

```
PASS  categorical dark  adjacent        worst CVD ΔE 9.4 (deutan) · normal-vision 20.9
PASS  categorical dark  all-pairs[1-3]  worst CVD ΔE 9.4 (deutan) · normal-vision 24.6
PASS  categorical light adjacent        worst CVD ΔE 9.0 (deutan) · normal-vision 20.8 · all ≥3:1
PASS  ordinal violet dark               monotone · ΔL ≥ 0.06 · light-end 2.19:1
PASS  ordinal violet light              monotone · ΔL ≥ 0.06 · light-end ≥2:1
```

The categorical set also passes unchanged on `--surface "#0c1015"` (the ground) and
`--surface "#1b2128"` (the raised rail), so a chart is legal on any of the three planes.

---

## 3. Typography

Unchanged from the existing system — it was already right. Restated so it is enforceable.

```
Spectral    — titles, venue names, the wordmark. Editorial weight, racing-programme feel.
Inter       — all UI text, labels, body.
IBM Plex Mono — every number that is data: prices, EV, stakes, times, probabilities.
```

Loaded by an `@import` at the top of the injected stylesheet, with the stacks as
fallback. (`theme.fontFaces` in `.streamlit/config.toml` could self-host these
through `server.enableStaticServing`, which is already enabled — worth doing to
drop a render-blocking external request, but it is not done yet.)

```css
--f-sans:  'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
--f-serif: 'Spectral', Georgia, 'Times New Roman', serif;
--f-mono:  'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;
```

### Scale — fixed rem, ratio ≈1.2

| token | px | used for |
|---|---|---|
| `--t-xs` | 11 | labels, pills, axis ticks, book chips |
| `--t-data` | 13 | table cells, EV, probability values |
| `--t-sm` | 14 | secondary body, card header time |
| `--t-body` | 15 | body |
| `--t-md` | 17 | card header venue, wordmark |
| `--t-lg` | 22 | section titles |
| `--t-xl` | 28 | page title |
| `--t-2xl` | 36 | KPI value |

### Figures — the rule that is most often got wrong

- **`tabular-nums` where numbers must align vertically:** table columns, axis ticks,
  prices, stakes, times. These are the columns your eye runs down.
- **Proportional figures for large standalone numbers:** the KPI tile value, any hero
  figure. Tabular gives every digit the width of a `0`, which reads loose at 36px.

### Forbidden

No display faces. No script or handwriting faces. No gradient text, no text-shadow on
body copy. Headings are `--f-sans` except where explicitly `--f-serif` (appbar `h1`,
card-header `.venue`, `.rp-brand .wm`).

---

## 4. Component stylings

Every interactive element ships **default / hover / active / focus-visible / disabled**.
`:focus-visible` is never removed — it is a 2px `--brand` ring at 2px offset.

```css
:where(a, button, [role="button"], summary, input, select, textarea):focus-visible {
  outline: 2px solid var(--brand);
  outline-offset: 2px;
  border-radius: var(--r-sm);
}
```

### 4.1 KPI stat tile — `.rp-kpi`

The reference's device: a bordered tile with a **2px tone accent on the top edge**, so a
rail of six tiles is scannable by colour before it is readable by label.

```css
.rp-kpi {
  position: relative; overflow: hidden;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4);
  box-shadow: var(--shadow-card);
  transition: border-color var(--dur) var(--ease), transform var(--dur) var(--ease);
}
.rp-kpi::before {                       /* the tone accent */
  content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: var(--border-strong);
}
.rp-kpi.t-info::before   { background: var(--info); }
.rp-kpi.t-value::before  { background: var(--value); }
.rp-kpi.t-amber::before  { background: var(--amber); }
.rp-kpi.t-brand::before  { background: var(--brand); }

.rp-kpi:hover        { border-color: var(--border-strong); }
.rp-kpi .k-lab       { font-size: var(--t-xs); color: var(--muted);
                       text-transform: uppercase; letter-spacing: .08em; }
.rp-kpi .k-val       { font-size: var(--t-2xl); font-weight: 600; color: var(--ink);
                       font-family: var(--f-mono); line-height: 1; letter-spacing: -.02em; }
                       /* NOTE: no tabular-nums — see §3 Figures */
.rp-kpi .k-val.value { color: var(--value); }
.rp-kpi .k-sub       { font-size: var(--t-sm); color: var(--muted); margin-top: var(--s1); }
```

The tone class reuses the `icon_tone` argument `C.kpi()` already accepts. No API change.

**Stat-tile contract:** `label` (sentence case, no trailing colon) · `value`
(auto-compact) · `sub` (optional; a denominator, a period, a qualifier).
A rate **never travels without its denominator** — `ui/_winrate.py` exists for this.

### 4.2 Race card — `.rp-card`

```css
.rp-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: border-color var(--dur) var(--ease);
}
.rp-card:hover { border-color: var(--border-strong); }
.rp-card-hdr {
  display: flex; align-items: center; gap: var(--s2); flex-wrap: wrap;
  padding: var(--s3) var(--s4);
  border-bottom: 1px solid var(--border);
  background: var(--surface-2);
}
.rp-card-hdr .venue { font-family: var(--f-serif); font-weight: 600; font-size: var(--t-md);
                      color: var(--ink); }
.rp-card-hdr .rtime { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
                      font-size: var(--t-sm); color: var(--ink-2); }
```

### 4.3 The runner row — the scanning surface

The user's priority, in order: **price, EV, stake**. Each gets its own distinct channel,
so no two compete.

| cell | channel |
|---|---|
| **Price** | Loudest. `--f-mono`, weight 500, `--ink`. The **fraction is the headline** (`7/1`); the decimal rides alongside in `--t-xs` `--muted` (`8.00`). Never a currency symbol — a price is a ratio. |
| **EV** | The *only* green in the row. `--value` when flagged, `--muted` when flat. Signed, zero decimals: `+14%`. |
| **Stake** | `--f-mono` `--ink`, with a `--value` leading hairline when it is a live recommendation. Money, so it **keeps** its `€`. The sub-label is the **bet type** — `Win` or `E/W` — not the Kelly size band: "Token" read as a bet type to anyone who had not seen the sizing ladder. The band still travels, in the tooltip. |
| Win % | `--info` meter + always-visible numeric label. |
| Confidence | Neutral chip, dots carry the level, word carries it too. |

**The green-appears-once rule.** Previously `.bk-best` and `.rp-stake` were both
`var(--value)`, alongside the EV chip — three greens in one row, on a surface where green
means one specific thing. Now:

```css
/* best per-book price: weight and the ★ carry it, not the reserved green */
.bk-best { color: var(--ink); font-weight: 600; }

/* stake: ink, with a value-tinted leading rule when it is a live recommendation */
.rp-stake {
  font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  color: var(--ink); font-weight: 600; white-space: nowrap;
  border-left: 2px solid var(--value); padding-left: var(--s2);
}
```

### 4.4 Probability meter — `.pb`

The fill's width rides on the `--pbw` custom property set inline by
`_components.prob_bar()`, so the mount keyframe and the reduced-motion end state agree.
**Do not replace this mechanism.**

Per the meter spec, the unfilled track is a **lighter step of the fill's own ramp**, not
neutral grey, so state reads across the whole bar:

```css
.pb-track { background: color-mix(in oklab, var(--info) 18%, var(--surface)); }
.pb-fill        { background: var(--info); width: var(--pbw, 0%); }
.pb-fill.place  { background: oklch(0.68 0.10 250); }
.pb-fill.show   { background: oklch(0.66 0.06 250); }
.pb-val         { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
```

The numeric `%` beside the bar is **not optional** — it is the secondary encoding.

### 4.5 Buttons

```css
.stButton > button, [data-testid="stBaseButton-primary"] {
  background: var(--brand); color: #fff;            /* 4.73:1 */
  border: 1px solid var(--brand-deep);
  border-radius: var(--r-sm); font-weight: 600;
  transition: background var(--dur-fast) var(--ease);
}
:hover          { background: var(--brand-text); }
:active         { background: var(--brand-deep); transform: translateY(1px); }
:focus-visible  { outline: 2px solid var(--brand); outline-offset: 2px; }
:disabled       { background: var(--surface-3); color: var(--muted);
                  border-color: var(--border); cursor: not-allowed; }
```

Most native widget colour now comes from `.streamlit/config.toml` rather than CSS
overrides — see §10.

### 4.6 Pills, badges, chips

All share: `--r-pill` or `--r-sm`, `--t-xs`, a 1px border, a tinted wash background, and
**a word**. `.pill-value` (`VALUE +9pp`), `.pill-ew` (`E/W`), `.tier-strong` (`STRONG`),
`.tier-lean` (`LEAN`), `.conf` (dots + word), `.bk` (book label + fraction, `★` on best).

---

## 5. Layout

### The desk view — the page's rhythm

The reference is a **chart dashboard**: a KPI strip, then six-to-eight chart cards of
varied size, all visible at once. Matching its colours without matching that rhythm is
what made the first pass feel wrong — we had the skin and not the substance. The landing
page now follows the same beat:

```
app bar                    identity, date, model-trust badges
KPI rail            ×4     races · value bets · selections · each-way
"Today's shape"     ×3     value by odds band | model vs market | best price held by
racecard grid       ×2     the dense block — the reference's bubble matrix
"Today's winners"   wide   ranked picks vs where they actually finished
"Is the model honest?" ×2  bankroll equity curve | A/E calibration
footer
```

The shell runs at **1560px** here (`inject_design(width="wide")`), not the 1240 reading
measure: the racecard block and a three-up chart row both need the room to stay dense.

**"Today's winners" is empty most of the time, and says why.** A card that has not run
yet looks identical to a results feed that has stopped, so the empty state always names
which — and when the feed is stale it says how many days. At the time of writing that
reads *"last recorded a finish on 25 Jul 2026 — 55 days ago. Nothing here is waiting on
today's racing; the feed itself is stale."* Blurring those two states is exactly the
failure this project's honesty rule exists to prevent.

Two questions govern a betting desk, so the page answers both **before** you scroll:
*where is today's value* (pre-race, from `data/predictions.json`) and *should I trust the
model at all* (post-race, from the settled ledger). `ui/_today.py` holds the first,
`ui/dashboard.py` the second.

**`st.columns` already reflows.** Streamlit ships `flex-wrap: wrap` on
`stHorizontalBlock`, so a 3-up chart row stacks to one column on a phone with no help —
measured at 375px, three full-width cards, no overflow. The only thing it needs is a
`min-width: 260px` floor above 420px so charts don't squeeze into slivers on the way down.

**Chart cards use `st.container(border=True)`** — a Streamlit chart cannot live inside an
HTML string, so this is the only way to get a real card around one.
`[data-testid="stVerticalBlockBorderWrapper"]` is styled to match `.rp-card`, and the
title sits in an `.rp-chart-hd` header rather than inside the plot.



- Shell max-width **1240px**, padding `--s5 --s8 --s16`.
- Spacing scale: `--s1` 4 · `--s2` 8 · `--s3` 12 · `--s4` 16 · `--s5` 20 · `--s6` 24 ·
  `--s8` 32 · `--s10` 40 · `--s12` 48 · `--s16` 64. Nothing off-scale.
- Radii: `--r-sm` 6 · `--r-md` 10 · `--r-lg` 14 · `--r-pill` 999.
- **KPI rail:** `repeat(auto-fit, minmax(150px, 1fr))`, gap `--s3`.
- **Card grid:** `repeat(auto-fit, minmax(480px, 1fr))`, gap `--s3`. Two-up on desktop,
  one column below ~990px. **480, not 520:** the shell is capped at 1240px and its padding
  leaves ~1032px of track, so a 520px minimum silently never reaches two columns — it was
  measured at 520 and the grid stayed one-up at every width.

```css
.rp-card-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
                gap: var(--s3); align-items: start; }
.rp-card      { container-type: inline-size; }
```

**A card answers to its own width, not the viewport's.** In a two-up grid a card is ~510px
even on a 1440px screen, so the runner table's column priority is a *container* query, not
a media query — otherwise a wide viewport would keep all seven columns crammed into half
the width:

```css
@container (max-width: 560px) {
  .rp-tbl th:nth-child(5), .rp-tbl td:nth-child(5) { display: none; }  /* Conf. */
  .rp-odds + .rp-sub { display: none; }                                /* decimal */
}
```

`_render_race_card()` **returns** its HTML so a venue's cards can be joined into one
`rp-card-grid` and flushed in a single `st.markdown`. Never grid Streamlit's internal
`stVerticalBlock` wrappers — those testids are not a public API.

---

## 6. Depth & elevation

**Depth comes from lightness, not shadow.** `--bg` → `--surface` → `--surface-2` →
`--surface-3` is the elevation ladder. Shadows only stop a card from floating loose:

```css
--shadow-card: 0 1px 0 rgba(255,255,255,.03) inset, 0 1px 2px rgba(0,0,0,.4);
--shadow-pop:  0 12px 32px rgba(0,0,0,.5), 0 2px 8px rgba(0,0,0,.4);
```

### The ambient glow

```css
.stApp {
  background:
    radial-gradient(90% 60% at 8% -10%,  var(--glow-a), transparent 60%),
    radial-gradient(80% 55% at 100% 8%,  var(--glow-b), transparent 58%),
    var(--bg);
  background-attachment: fixed;
}
```

with `--glow-a: oklch(0.42 0.15 288 / .30)` and `--glow-b: oklch(0.38 0.13 300 / .22)`.

**Never `filter: blur()`.** Blur over painted area is the performance red line; a fixed
radial gradient is one cheap paint and looks the same. `background-attachment: fixed`
stops the wash tiling as the page scrolls.

Cards are **opaque**, so the glow only shows in the gutters and cannot erode text
contrast. The glow peak is capped dark enough that `--ink` over it still clears AA even
if a surface were ever made translucent.

`.rp-appbar` paints the same two radials at card scale, in the same hues, so the appbar
and the page ground read as one continuous field rather than two unrelated washes.

---

## 7. Motion & interaction

**Level 1–2.** Things arrive; nothing loops; nothing follows the cursor.

```css
--ease: cubic-bezier(0.22, 1, 0.36, 1);
--dur-fast: 140ms;  --dur: 200ms;  --dur-slow: 320ms;
```

- Entrances: `rp-rise` / `rp-rise-sm` / `rp-fade`, staggered across the first six KPI
  tiles and the first six table rows, then flat.
- `pb-grow` sweeps probability bars 0 → `--pbw` on mount. `why-grow` does the same for
  SHAP driver bars.
- Hover: border lightens one step; never a scale or a shadow bloom.

**Every entrance lives inside `@media (prefers-reduced-motion: no-preference)`, and the
element's default state is its visible end state.** A reduced-motion user sees the final
layout, never an empty one. A global `prefers-reduced-motion: reduce` block zeroes any
remaining animation and transition.

Charts: hover is the default interaction layer — crosshair + unified tooltip on time
series, per-mark tooltip on bars. Hit targets are larger than the marks.

---

## 8. Do's and don'ts

**Do**

1. Assign categorical hues in **fixed order, never cycled**. A sixth series folds into
   "Other" or becomes small multiples.
2. Give every chart with ≥2 series a legend, and direct-label selectively — the endpoint,
   the extreme, the one series the story is about. Never a number on every point.
3. Keep bars ≤24px thick with a 4px rounded data-end and a square baseline end.
4. Put a 2px surface gap between touching marks and a 2px surface ring on overlapping dots.
5. Pair every colour signal with a word or glyph.
6. Show a rate with its denominator (`37% (74/200)`), and a confidence interval where one
   exists.

**Don't**

1. **Never colour a chart series with `--value`, `--amber` or `--danger`.** They mean
   something. The old `colorway` did exactly this and it was the worst bug in the system.
2. **Never show green twice in one runner row.** EV owns the green.
3. **Never put a currency symbol on a price.** A price is a ratio (`7/1`, `6/4`, `Evens`).
   Money — the stake — keeps its `€`.
4. **Never a dual-axis chart.** Two measures of different scale → two charts, small
   multiples, or index to a common base.
5. **Never replace `won_prob_normalized` with raw `won_prob`** in the headline Win %.
   The raw marginal saturates out-of-sample (mean 0.33 vs a true 0.11 win rate, ECE 0.22).
   `C.headline_win_prob()` is the only correct accessor.
6. **Never a rainbow sequential ramp, and never a hue at a diverging midpoint.** One hue
   light→dark for magnitude; two hues plus neutral grey for polarity.
7. **Never dashed gridlines**, and never a gridline darker than one step off the surface.
8. **Never `filter: blur()` on a painted region**, never gradient text, never
   glassmorphism, never emoji as an icon.
9. Never style Streamlit internal testids where a `[theme]` config option exists — the
   testids break on upgrade, the config options do not.
10. **Never rename a colour token by find-and-replace without re-reading each use.** The
    old brand was a red, so some `--oxblood-bright` uses meant *brand* and others meant
    *this went the wrong way* — the losing-bet marker, negative CLV, and the negative arm
    of the SHAP driver bars. A blanket rename turned all three violet. Polarity belongs to
    `--danger`; identity belongs to `--brand`.

---

## 9. Responsive behaviour

Breakpoints: **980px**, **640px**, **420px**. Mobile is a hard requirement — the app must
be fully usable at **375px**.

| width | behaviour |
|---|---|
| ≥1090px | Card grid two-up. KPI rail auto-fit. Full 7-column runner table. |
| ≤980px | Shell padding tightens. KPI rail → 3 columns. `.rp-detail-grid` → 1 column. |
| ≤640px | KPI rail → 2 columns. Runner table hides **Conf.** and the decimal beside the fraction, leaving `# · Runner · Price · Win% · EV · Stake` — fits with no sideways scroll. |
| ≤420px | **Runner rows stack.** KPI rail → 1 column. `.sg-stats` → 2 columns. Glow reduces to one radial. |

### The stacked runner row

```css
@media (max-width: 420px) {
  .rp-body { overflow-x: visible; }
  .rp-tbl, .rp-tbl tbody, .rp-tbl tr, .rp-tbl td { display: block; width: 100%; }
  .rp-tbl thead { position: absolute; width: 1px; height: 1px;
                  overflow: hidden; clip-path: inset(50%); }   /* visually hidden */
  .rp-tbl tr { border: 1px solid var(--border); border-radius: var(--r-md);
               padding: var(--s3); margin-bottom: var(--s2); }
  .rp-tbl td { border: none; padding: var(--s1) 0;
               display: flex; align-items: center; justify-content: space-between; gap: var(--s3); }
  .rp-tbl td::before { content: attr(data-label); color: var(--muted);
                       font-size: var(--t-xs); text-transform: uppercase;
                       letter-spacing: .06em; }
  .rp-tbl td[data-label="Runner"]::before { display: none; }
  .rp-tbl td.num { text-align: left; }
}
```

This requires `data-label` attributes on the `<td>`s — added in `ui/app.py`. It is the
only render-logic change the mobile pass needs.

### Charts at 375px

Python cannot know the viewport width, so CSS does the degrading:

```css
@media (max-width: 420px) {
  .js-plotly-plot .xtick text, .js-plotly-plot .ytick text { font-size: 10px !important; }
  .js-plotly-plot .modebar { display: none !important; }
}
```

Figures help by carrying `automargin=True` on both axes, no in-plot title, and a
horizontal legend above the plot.

### Touch

Every interactive target is **≥44×44px**: `a.rp-open`, `.bk` chips, expander headers,
sidebar nav links, buttons.

**No horizontal page scroll at 375px.** Long venue and horse names use
`overflow-wrap: anywhere`.

---

## 10. Charts

Charts are most of what makes this dashboard work. Read this before writing a line of
chart code.

### 10.1 Procedure

1. **Pick the form from the data's job** — magnitude, identity, polarity, change over
   time, or a single headline number. Sometimes the answer is a stat tile, not a chart.
2. **Assign colour by job** — categorical, ordinal, sequential, diverging, or status.
3. **Validate the palette with the script** (§2.6). Never reason about ΔE by eye.
4. **Apply the mark specs** below.
5. **Ship the hover layer** — it is the default, not an enhancement.
6. **Accessibility pass** — legend for ≥2 series, numbers available as text, never
   colour alone.
7. **Render it and look at it** at 1440px and at 375px.

### 10.2 Mark specs — fixed across every chart

| mark | spec |
|---|---|
| Bar / column | **≤24px thick**; 4px rounded data-end, square at the baseline; grows from one baseline |
| Line | **2px**, round join and cap |
| Marker / end-dot | **≥8px** (r ≥ 4), filled with the series colour, **2px ring in the surface colour** |
| Area fill | the series hue at **~10%** — a wash, never a saturated block |
| Gridlines / axes | one step off surface, **hairline 1px, solid**, recessive |
| Surface gap | **2px** of surface between touching marks — stacked segments and adjacent bars alike |

**Text never wears the data colour.** Values, labels, legends and axis text use
`--chart-ink` / `--chart-axis`. Identity comes from the coloured mark *beside* the text.

### 10.3 Layout defaults — `plotly_layout(mode="dark")`

- `colorway` = the five validated slots. **Not** the semantic trio.
- **No in-plot title.** The card header carries it — better on mobile, and it frees the
  top margin for the legend.
- `automargin=True` on both axes; tight margins.
- `hovermode="x unified"` on time series; per-mark hover on bars.
- Legend horizontal, above the plot, `--chart-ink`.
- Gridlines `--chart-grid`, solid.

### 10.4 The four charts

#### Bankroll equity curve — `ui/dashboard.py`

*Job: change over time, single series, with polarity that genuinely means good/bad.*

Money is the one place where green-up / red-down is honest, so the line keeps
`--value` above the starting bankroll and `--danger` below it. The area fill takes **the
line's own hue at ~10%** — previously it was a fixed blue regardless, which read as a
third, meaningless colour.

`fill="tozeroy"` is wrong here: it fills from zero, which is off-scale when the bankroll
starts at €1,000. Use an invisible baseline trace at the starting bankroll plus
`fill="tonexty"`, so the fill reads as profit or drawdown *against the start*.

Dashed baseline at the starting bankroll in `--muted`. One ≥8px end marker with a 2px
surface ring, and a single direct end-label with the closing figure. No legend — one
series, and the card header names it.

#### A/E calibration by probability bucket — `ui/dashboard.py`

*Job: polarity around a baseline, across an ordered set of buckets.*

A/E = actual ÷ expected. 1.0 is honest; below is over-confident, above is
under-confident. **Bars are anchored at the 1.0 baseline** (`go.Bar(base=1.0)` on
`y = ae − 1.0`) so the direction of the miss is the shape of the chart, not something to
be read off an axis.

The reference line at 1.0 is **neutral ink** — previously it was `--value` green, which
implied that perfect calibration is a *positive-EV* signal. It is not; it is a different
kind of claim entirely.

Backtest takes slot 1, Paper slot 2. Legend present. `n=` counts ride the Paper bars only
as sparse direct labels. Grey `--muted` is not used for a series — it falls below the
chroma floor and reads as "no data".

#### Per-odds-band bars — `ui/performance_dashboard.py`

*Job: magnitude across an **ordinal** sequence.*

Bands run Fav (<3.0) → Mid (3–8) → Each-Way (8–20) → Longshot (20+). The x-axis carries
that order, so colour must not re-encode it — and colouring nominal bars by their value
spends the identity channel on something bar length already shows.

Win% in slot 1, **place-only stacked above it** in an ordinal step, with the 2px surface
gap between segments. That decomposition is honest (`win + place-only = win+place`) and
replaces `barmode="overlay"` with a 25%-alpha second bar, which was both hard to read and
below contrast. Values on the caps. Light mode.

#### Win-probability bars in race cards — `ui/_components.py`

*Job: a single magnitude, inline, at row scale.*

**Not a chart library.** CSS, via `--pbw` (§4.4). At row scale a Plotly figure would be
heavier, blurrier and would break the stacked mobile layout. Track is a lighter step of
the fill's own ramp; the numeric label is always visible.

### 10.5 Native chart theming

`.streamlit/config.toml` sets `chartCategoricalColors`, `chartSequentialColors` and
`chartDivergingColors` per mode. Streamlit applies these to **Plotly, Altair and
Vega-Lite**, so the validated palette reaches every chart library from one place and the
Altair reliability diagram in `ui/model_compare.py` needs no palette of its own.

An explicit `colorway` in `plotly_layout()` still governs the four charts above — they
are specified, not defaulted.

---

## 11. Where this lives

| file | holds |
|---|---|
| `ui/_design.py` | the colour source of truth, the `:root` tokens, the stylesheet, `PALETTE`, `plotly_layout()` |
| `.streamlit/config.toml` | native theme for both modes, semantic palette, chart palettes, fonts |
| `ui/_components.py` | every `rp-*` markup builder — Streamlit-free, unit-tested |
| `ui/_icons.py` | in-house inline SVG; no external image files |

`PALETTE`, the CSS custom properties and the Plotly theme are all **generated from one
Python dict**. They were three hand-synced copies and they had already drifted. If you
add a colour, add it there and nowhere else.

---

## 12. Migration status

**The app is fully on this system, and `[theme] base` is `"dark"`.**

| surface | state |
|---|---|
| `ui/app.py`, pages 8–16 | on the system, dark |
| `settings.py` (6), `live_races.py` (2), `bet_placer.py` (3), `performance_dashboard.py` (4), `race_compare.py` (5) | **migrated** — own stylesheets deleted, `inject_design()` |
| pages 1, 7 | native widgets only; they take the whole appearance from `config.toml` |
| `ui/_theme.py` | **deleted** |
| `ui/performance.py`, `ui/predictions.py` | **deleted** — 1,553 orphaned lines holding the last two duplicate `:root` palettes. See the note below. |

`ui/_design.py` is the **only** module in the app that defines a palette, and every page
calls `inject_design()`.

> **What the orphan deletion cost.** `ui/predictions.py` and `ui/performance.py` were
> unreachable from `ui/pages/` and imported by nothing, but they were not merely dead
> duplicates. `ui/performance.py` held the only **per-target AUC bars and feature-importance
> chart** in the app — no live page renders feature importance. `ui/predictions.py` held a
> per-selection feature "why" breakdown, which page 8 covers with SHAP drivers. If the model
> metrics view is wanted back, rebuild `_render_model_perf` against `altair_theme()` rather
> than restoring the module: it carried its own palette and its own sidebar.

### What the flip fixed

Pages 1 and 7 inject no CSS, so before the flip their nav rendered `#c8d3e8`
(the old `[theme.sidebar] textColor`) on Streamlit's light `#f0f2f6` sidebar — about
**1.3:1**, effectively invisible. They are dark and legible now, and `st.dataframe`
follows the base, which no CSS could reach.

### Three rules enforced on the way through, not preserved

1. **The rank ramp was a rainbow** — `rk1` blue, `rk2` violet, `rk3` cyan, carrying
   meaning by hue alone. Rank 1 is filled with the brand; every other rank is neutral.
2. **`.stake`, `.bk-best` and `.pill-src` were green.** Green is positive EV. A stake, a
   best price and a data-source tag are none of those. They lost it, exactly as the runner
   table did.
3. **Horse A / Horse B were `#2563eb` / `#7c3aed`.** `#7c3aed` is a near neighbour of the
   new brand, so "Horse B" read as "the brand". They are two equal peers — nominal
   identity — so they now take categorical slots 1 and 2, which are validated as an
   adjacent pair.

### The legacy class map

Pages 2–5 emit class names that predate the `rp-*` layer, sometimes the *same* name with
a different child contract (`.rp-kpi > .kpi-label` where this system expects `.k-lab`).
Rather than rewrite ~3,000 lines of render logic, those names are mapped onto tokens in
`ui/_design.py` under **"legacy page class map"**. It is a bridge: when a page's markup is
ported to the `ui/_components.py` builders, delete its rows. Nothing else depends on it.

### Native widgets the system had never styled

Those pages were light, so the dark stylesheet never needed to cover the widgets they use:

- `[data-baseweb="base-input"]` — **the one that actually paints.** The outer
  `[data-baseweb="input"]` / `"textarea"` divs were already themed, so a light plate showed
  through from the wrapper *between* them. Any input that looks stubbornly light is this.
- `.stTextArea textarea`, `[data-testid="stCode"] pre`, checkbox / toggle / radio labels,
  `[data-testid="stForm"]`.
- `st.dataframe` paints to a `<canvas>` that CSS cannot reach. Its chrome is declared in
  `config.toml` (`dataframeHeaderBackgroundColor`, `dataframeBorderColor`) and its body
  follows `base`.
- Vega writes `style="background: white"` straight onto its SVG, which no spec property
  reliably clears — a stylesheet rule does.

### Editing this system

Streamlit caches `ui/_design.py` across reruns. **Restart the server after changing the
stylesheet** — a page reload alone serves the old CSS and will make a correct change look
broken.
