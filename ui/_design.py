"""Race Predictor v4 — shared design system.

One stylesheet, one set of tokens, injected by every surface via
:func:`inject_design`. See ``DESIGN.md`` at the repo root for the spec this
implements; this module is that spec made executable.

Direction: a dark **"betting desk at dusk"** — near-black ground with a violet
ambient field, opaque hairline-bordered cards floating on it, and the numbers as
the only loud thing on the page.

**Colour lives in exactly one place: the ``_DARK`` / ``_LIGHT`` dicts below.**
They generate the ``:root`` custom properties, the :data:`PALETTE` dict and the
Plotly/Altair chart themes. Previously those were three hand-maintained copies
and they had already drifted. If you add a colour, add it there and nowhere
else.

Locked semantic roles — these carry meaning and are reserved:

* ``--value``  positive expected value. Never "series 2", never "best price".
* ``--amber``  caution / each-way.
* ``--danger`` danger, loss, negative.

Chart series colours are a **separate namespace** (``--series-1..5``): a series
never wears a semantic token and a semantic token is never "series 4". Every
categorical value was validated for colourblind separation against this app's
own surfaces — see ``DESIGN.md`` §2.6 for the commands.

And the standing rule: **colour never carries meaning alone.** Every signal
pairs with a word, glyph or shape — ``VALUE +9pp``, ``STRONG``, ``★``, the
``1ˢᵗ`` chip, the numeric ``%`` beside every probability bar.
"""
from __future__ import annotations

from functools import lru_cache

import streamlit as st

# ── colour: the single source of truth ───────────────────────────────────────
# Authored in OKLCH, stored as resolved hex because Plotly cannot parse oklch().
# Alpha values use rgba(), which is valid in both CSS and Plotly.
# Contrast ratios in the comments are measured against `surface`.

_DARK: dict[str, str] = {
    # surfaces — depth comes from lightness, not shadow
    "bg":            "#0c1015",
    "surface":       "#13181e",
    "surface_2":     "#1b2128",
    "surface_3":     "#262c33",
    "border":        "#31363d",
    "border_strong": "#484e55",

    # ink
    "ink":   "#edeff1",   # 16.5:1
    "ink_2": "#babec3",   # 10.2:1
    "muted": "#969ca3",   #  6.9:1

    # brand — indigo/violet. `brand` is a FILL (3.77:1, legal as a mark, not as
    # text); brand-coloured text always uses `brand_text`.
    "brand":      "#6d5cf0",   # white-on 4.73:1
    "brand_text": "#9085e9",   # 5.71:1 — AA
    "brand_deep": "#5138c9",
    "on_brand":   "#ffffff",

    # semantic — RESERVED, do not reuse for chart identity
    "value":     "#5bcc80",   # 8.83:1
    "value_dim": "#44a264",
    "amber":     "#ebae51",   # 9.10:1
    "danger":    "#ec5448",   # 5.04:1
    "info":      "#549de5",   # 6.23:1 — probability bars, links
    "info_2":    "#679dd4",   # place bar
    "info_3":    "#7695b6",   # show bar

    # chart categorical — fixed order, never cycled. Validated: worst adjacent
    # CVD ΔE 9.4, normal-vision 20.9; all-pairs on slots 1-3 passes too.
    "series_1": "#9085e9",    # violet
    "series_2": "#d95926",    # orange
    "series_3": "#199e70",    # aqua
    "series_4": "#3987e5",    # blue
    "series_5": "#d55181",    # magenta

    # chart ordinal — one hue, monotone lightness (odds bands, tiers)
    "ord_1": "#4d40ab", "ord_2": "#6759d2", "ord_3": "#8a7de8", "ord_4": "#b3abf2",

    # chart diverging — two hues + a NEUTRAL grey midpoint (A/E around 1.0)
    "div_pos": "#9085e9", "div_mid": "#4a5565", "div_neg": "#d95926",

    # chart chrome
    "chart_ink":  "#babec3",
    "chart_axis": "#969ca3",
    "chart_grid": "#1b2128",

    # washes & hairlines
    "brand_wash":     "rgba(109, 92, 240, 0.16)",
    "value_wash":     "rgba(91, 204, 128, 0.14)",
    "value_wash_2":   "rgba(91, 204, 128, 0.20)",
    "value_line":     "rgba(91, 204, 128, 0.40)",
    "value_line_2":   "rgba(91, 204, 128, 0.50)",
    "value_dim_line": "rgba(68, 162, 100, 0.55)",
    "amber_wash":     "rgba(235, 174, 81, 0.14)",
    "amber_line":     "rgba(235, 174, 81, 0.35)",
    "danger_wash":    "rgba(236, 84, 72, 0.14)",
    "info_wash":      "rgba(84, 157, 229, 0.16)",
    "info_line":      "rgba(84, 157, 229, 0.35)",
    "info_track":     "rgba(84, 157, 229, 0.18)",

    # atmosphere — the ambient glow. Fixed radial gradients, never filter:blur()
    # (blur over painted area is the performance red line). Cards are opaque so
    # the glow only shows in the gutters and cannot erode text contrast.
    "glow_a":       "rgba(84, 42, 186, 0.30)",
    "glow_b":       "rgba(104, 46, 170, 0.20)",
    "mesh_brand":   "rgba(109, 92, 240, 0.22)",
    "mesh_info":    "rgba(84, 157, 229, 0.18)",
    "mesh_value":   "rgba(91, 204, 128, 0.12)",
    "hero_scrim_a": "rgba(12, 16, 21, 0.55)",
    "hero_scrim_b": "rgba(8, 11, 15, 0.92)",
    "texture":      "rgba(237, 239, 241, 0.025)",
    "texture_2":    "rgba(237, 239, 241, 0.05)",
    "shimmer":      "rgba(237, 239, 241, 0.07)",

    # elevation
    "shadow_card": "0 1px 0 rgba(255,255,255,.03) inset, 0 1px 2px rgba(0,0,0,.4)",
    "shadow_pop":  "0 12px 32px rgba(0,0,0,.5), 0 2px 8px rgba(0,0,0,.4)",
    "glow_brand":  "0 0 0 1px #5138c9, 0 6px 18px rgba(109, 92, 240, 0.35)",
}

# The light binding. Same role names, so the stylesheet body is mode-agnostic.
# Used by the not-yet-dark pages; every text token clears WCAG AA on white.
_LIGHT: dict[str, str] = {
    "bg":            "#f6f7f9",
    "surface":       "#ffffff",
    "surface_2":     "#f2f4f8",
    "surface_3":     "#e9edf3",
    "border":        "#d8dee8",
    "border_strong": "#b9c2d0",

    "ink":   "#172033",   # 16.3:1
    "ink_2": "#3b4658",   #  9.5:1
    "muted": "#5d687a",   #  5.6:1  (the legacy #6b7689 was 4.28:1 on --bg)

    "brand":      "#5b4ec6",   # white-on 6.27:1
    "brand_text": "#4a3aa7",   # 8.56:1
    "brand_deep": "#382e86",
    "on_brand":   "#ffffff",

    # the legacy light semantics all FAILED AA on white (green 3.30, amber 2.15)
    "value":     "#15803d",   # 5.02:1
    "value_dim": "#1a9e52",
    "amber":     "#8f5d00",   # 5.62:1
    "danger":    "#c9252b",   # 5.55:1
    "info":      "#1d4ed8",   # 6.70:1
    "info_2":    "#4a7fd8",
    "info_3":    "#6f92c4",

    "series_1": "#4a3aa7", "series_2": "#c2521f", "series_3": "#0f8a5f",
    "series_4": "#2a78d6", "series_5": "#c93f76",

    "ord_1": "#b3a9ef", "ord_2": "#8a7be2", "ord_3": "#5b4ec6", "ord_4": "#382e86",

    "div_pos": "#4a3aa7", "div_mid": "#a8aeb9", "div_neg": "#c2521f",

    "chart_ink":  "#3b4658",
    "chart_axis": "#5d687a",
    "chart_grid": "#edf1f6",

    "brand_wash":     "rgba(91, 78, 198, 0.10)",
    "value_wash":     "rgba(21, 128, 61, 0.10)",
    "value_wash_2":   "rgba(21, 128, 61, 0.16)",
    "value_line":     "rgba(21, 128, 61, 0.35)",
    "value_line_2":   "rgba(21, 128, 61, 0.45)",
    "value_dim_line": "rgba(26, 158, 82, 0.50)",
    "amber_wash":     "rgba(143, 93, 0, 0.10)",
    "amber_line":     "rgba(143, 93, 0, 0.30)",
    "danger_wash":    "rgba(201, 37, 43, 0.10)",
    "info_wash":      "rgba(29, 78, 216, 0.10)",
    "info_line":      "rgba(29, 78, 216, 0.30)",
    "info_track":     "rgba(29, 78, 216, 0.14)",

    "glow_a":       "rgba(91, 78, 198, 0.07)",
    "glow_b":       "rgba(120, 70, 190, 0.05)",
    "mesh_brand":   "rgba(91, 78, 198, 0.10)",
    "mesh_info":    "rgba(29, 78, 216, 0.08)",
    "mesh_value":   "rgba(21, 128, 61, 0.07)",
    "hero_scrim_a": "rgba(246, 247, 249, 0.55)",
    "hero_scrim_b": "rgba(246, 247, 249, 0.92)",
    # on light the "texture" overlay has to darken, not lighten
    "texture":      "rgba(23, 32, 51, 0.04)",
    "texture_2":    "rgba(23, 32, 51, 0.06)",
    "shimmer":      "rgba(23, 32, 51, 0.06)",

    "shadow_card": "0 1px 0 rgba(255,255,255,.6) inset, 0 1px 2px rgba(23,32,51,.08)",
    "shadow_pop":  "0 12px 32px rgba(23,32,51,.16), 0 2px 8px rgba(23,32,51,.10)",
    "glow_brand":  "0 0 0 1px #382e86, 0 6px 18px rgba(91, 78, 198, 0.28)",
}

_MODES = {"dark": _DARK, "light": _LIGHT}

# Mode-independent scale. Type, space, radius, z and motion do not change with
# the surface, so they live outside the colour dicts.
_SCALE = """
  /* type scale — fixed rem (product register), ratio ~1.2 */
  --t-xs:   0.6875rem;  /* 11px  labels, pills, axis ticks   */
  --t-data: 0.8125rem;  /* 13px  table cells, EV, prob values */
  --t-sm:   0.875rem;   /* 14px  secondary body               */
  --t-body: 0.9375rem;  /* 15px  body                         */
  --t-md:   1.0625rem;  /* 17px  card header venue, wordmark  */
  --t-lg:   1.375rem;   /* 22px  section titles               */
  --t-xl:   1.75rem;    /* 28px  page title                   */
  --t-2xl:  2.25rem;    /* 36px  KPI value                    */

  /* ONE typeface. The style reference sets everything in Inter — headings,
     labels and its big figures alike — and lets weight, size and tracking carry
     the hierarchy. Verified by rendering its own strings at matched size and
     comparing glyph for glyph rather than by eye: the flat-top 5 in "5,279",
     the double-storey a and the angled t all match Inter exactly.
     `--f-display` and `--f-num` are ROLES, not faces, so the call sites still
     read honestly; `--f-code` is the only real monospace left. */
  --f-sans:    'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --f-display: var(--f-sans);   /* headings — weight + tracking, not a 2nd face */
  --f-num:     var(--f-sans);   /* data — always paired with tabular-nums       */
  --f-code:    'IBM Plex Mono', ui-monospace, 'SFMono-Regular', Menlo, monospace;
  /* back-compat aliases so every existing call site keeps resolving */
  --f-serif:   var(--f-display);
  --f-mono:    var(--f-num);

  --s1: 4px;  --s2: 8px;  --s3: 12px; --s4: 16px; --s5: 20px;
  --s6: 24px; --s8: 32px; --s10: 40px; --s12: 48px; --s16: 64px;

  --r-sm: 6px; --r-md: 10px; --r-lg: 14px; --r-pill: 999px;

  /* semantic z-index scale */
  --z-base: 1; --z-dropdown: 1000; --z-sticky: 1100;
  --z-modal: 1300; --z-toast: 1400; --z-tooltip: 1500;

  /* motion — ease-out-quint; product timings 140-320ms */
  --ease: cubic-bezier(0.22, 1, 0.36, 1);
  --dur-fast: 140ms; --dur: 200ms; --dur-slow: 320ms;
"""

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Inter:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,600&"
    "family=IBM+Plex+Mono:wght@400;500&display=swap');\n"
)

# "wide" is the desk view: the racecard block and the chart row both
# need more than the 1240 reading measure to stay dense.
_WIDTHS = {"default": "1240px", "narrow": "920px", "wide": "1560px"}


def _tokens_css(mode: str) -> str:
    """The ``:root`` block for one mode, generated from its colour dict."""
    rows = "\n".join(
        f"  --{name.replace('_', '-')}: {value};" for name, value in _MODES[mode].items()
    )
    return ":root {\n" + rows + "\n" + _SCALE + "}\n"


_BODY = """
/* ══ base surface ══════════════════════════════════════════════════════════ */
.stApp {
  background:
    radial-gradient(90% 60% at 8% -10%, var(--glow-a), transparent 60%),
    radial-gradient(80% 55% at 100% 8%, var(--glow-b), transparent 58%),
    var(--bg) !important;
  background-attachment: fixed !important;
}
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stMarkdown, [data-testid="stMarkdownContainer"] {
  color: var(--ink);
}
/* font-family is set on text containers only — NOT on `span`, so Streamlit's
   Material icon spans keep their ligature font and don't render as raw text
   ("keyboard_double_arrow_left"). Spans inherit the sans body font anyway. */
.stApp, .stApp p, .stApp li, .stApp label,
.stMarkdown, [data-testid="stMarkdownContainer"] {
  font-family: var(--f-sans);
}
/* light-on-dark legibility compensation: a touch more leading + tracking */
.stApp { font-size: var(--t-body); -webkit-font-smoothing: antialiased;
         text-rendering: optimizeLegibility; }
.block-container { letter-spacing: 0.006em; }
/* keep Streamlit's Material icon ligatures resolving — the blanket span
   font-family above otherwise renders them as raw text ("keyboard_double_arrow_…") */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded,
.material-symbols-outlined, span[class*="material-symbols"] {
  font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
    'Material Icons' !important; letter-spacing: normal !important;
}

.main .block-container,
[data-testid="stMainBlockContainer"] {
  padding: var(--s5) var(--s8) var(--s16) !important;
  max-width: 1240px !important;
}
/* kill the default top header bar; keep it as a thin transparent rail so the
   sidebar collapse control still works */
[data-testid="stHeader"] { background: transparent !important; height: 0 !important; }
[data-testid="stToolbar"] { right: var(--s4); }
[data-testid="stDecoration"] { display: none !important; }

h1, h2, h3, h4 { color: var(--ink); font-family: var(--f-sans); }
hr { border-color: var(--border); }
a, a:visited { color: var(--info); text-decoration: none; }
a:hover { color: var(--ink); text-decoration: underline; text-underline-offset: 2px; }
code, pre, kbd { font-family: var(--f-code); }
::selection { background: var(--brand-wash); }

/* tabular numerics everywhere numbers matter */
.mono, .tnum { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }

/* ══ scrollbars ════════════════════════════════════════════════════════════ */
* { scrollbar-width: thin; scrollbar-color: var(--border-strong) transparent; }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: var(--r-pill);
  border: 2px solid var(--bg); }
*::-webkit-scrollbar-track { background: transparent; }

/* ══ sidebar ═══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] > div:first-child {
  background: var(--surface) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink-2); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--muted) !important; font-family: var(--f-sans) !important;
  font-size: var(--t-xs) !important; font-weight: 600 !important;
  text-transform: uppercase !important; letter-spacing: 0.1em !important;
  margin: var(--s5) 0 var(--s2) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; opacity: 1 !important; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }

/* auto page-nav links on the dark sidebar */
[data-testid="stSidebarNav"] a { border-radius: var(--r-sm); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a *,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] span,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] p {
  color: var(--ink-2) !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover *,
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] [aria-current="page"] * {
  color: var(--ink) !important;
}
[data-testid="stSidebarNav"] a span, [data-testid="stSidebarNav"] a p {
  color: var(--ink-2) !important; font-weight: 500 !important;
}
[data-testid="stSidebarNav"] a:hover { background: var(--surface-2) !important; }
[data-testid="stSidebarNav"] a[aria-current="page"] {
  background: var(--brand-wash) !important;
}
[data-testid="stSidebarNav"] a[aria-current="page"] span,
[data-testid="stSidebarNav"] a[aria-current="page"] p { color: var(--ink) !important; }

/* ══ buttons ═══════════════════════════════════════════════════════════════ */
.stButton > button, .stDownloadButton > button,
[data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-primary"] {
  font-family: var(--f-sans) !important; font-weight: 600 !important;
  font-size: var(--t-sm) !important; border-radius: var(--r-sm) !important;
  background: var(--brand) !important; color: var(--on-brand) !important;
  border: 1px solid var(--brand-deep) !important;
  padding: var(--s2) var(--s4) !important; width: 100%;
  transition: box-shadow var(--dur) var(--ease), transform var(--dur-fast) var(--ease);
}
.stButton > button:hover, .stDownloadButton > button:hover {
  background: var(--brand) !important; box-shadow: var(--glow-brand) !important;
  transform: translateY(-1px);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid var(--brand-text) !important; outline-offset: 2px !important;
}
/* secondary button (kind="secondary") reads as a quiet outline */
[data-testid="stBaseButton-secondary"] {
  background: var(--surface-2) !important; color: var(--ink) !important;
  border: 1px solid var(--border-strong) !important;
}
[data-testid="stBaseButton-secondary"]:hover {
  background: var(--surface-3) !important; box-shadow: none !important;
}

/* ══ inputs / selects / multiselect (BaseWeb) ══════════════════════════════ */
[data-baseweb="input"], [data-baseweb="base-input"],
[data-baseweb="select"] > div, [data-baseweb="textarea"],
.stTextInput input, .stNumberInput input, .stDateInput input {
  background: var(--surface-2) !important; border-color: var(--border-strong) !important;
  color: var(--ink) !important; border-radius: var(--r-sm) !important;
  font-family: var(--f-sans) !important;
}
/* BaseWeb wraps the real <textarea> in its own div, so the wrapper rule above
   does not reach the element that actually paints. Same for st.code. */
.stTextArea textarea, [data-baseweb="textarea"] textarea {
  background: var(--surface-2) !important; color: var(--ink) !important;
  font-family: var(--f-code) !important; border-radius: var(--r-sm) !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
  color: var(--muted) !important; opacity: 1;
}
[data-testid="stCode"], [data-testid="stCode"] pre, .stCode pre, pre[class*="language-"] {
  font-family: var(--f-code) !important;
  background: var(--surface-2) !important; color: var(--ink-2) !important;
  border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important;
}
[data-testid="stCode"] code, .stCode code { color: var(--ink-2) !important; }

/* checkbox / toggle / radio — used by the pages migrated off their own light
   stylesheets, and never covered here before because those pages were light */
[data-testid="stCheckbox"] label, [data-testid="stToggle"] label,
[data-testid="stRadio"] label { color: var(--ink-2) !important; }
[data-testid="stRadio"] [role="radiogroup"] label > div:first-child,
[data-testid="stCheckbox"] label > span:first-child {
  border-color: var(--border-strong) !important;
}
[data-testid="stForm"] {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important; padding: var(--s4) !important;
}
/* st.dataframe paints to a <canvas> themed from `[theme] base`, not from CSS —
   only its frame can be reached from here. It follows the config, which is the
   remaining reason to flip base to "dark". */
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important;
  border-radius: var(--r-sm) !important; overflow: hidden; }

[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus, .stNumberInput input:focus {
  border-color: var(--brand-text) !important;
  box-shadow: 0 0 0 2px var(--brand-wash) !important;
}
[data-baseweb="tag"] {
  background: var(--brand-wash) !important; color: var(--ink) !important;
  border: 1px solid var(--brand-deep) !important;
}
[data-baseweb="tag"] svg { fill: var(--ink-2) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
  background: var(--surface-2) !important; border: 1px solid var(--border-strong) !important;
}
[data-baseweb="popover"] [role="option"] { color: var(--ink-2) !important; }
[data-baseweb="popover"] [role="option"]:hover { background: var(--surface-3) !important; }

/* slider — oxblood track + handle */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--brand) !important; border-color: var(--on-brand) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div { background: var(--brand) !important; }
[data-testid="stSlider"] [data-testid="stTickBar"] { color: var(--muted) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: var(--ink) !important; }

/* ══ expander — venue group container ══════════════════════════════════════ */
[data-testid="stExpander"] { border: none !important; background: transparent !important; }
[data-testid="stExpander"] details {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r-lg) !important; overflow: hidden; margin-bottom: var(--s3);
}
[data-testid="stExpander"] summary {
  background: var(--surface-2) !important; padding: var(--s3) var(--s4) !important;
  font-family: var(--f-sans) !important;
}
[data-testid="stExpander"] summary:hover { background: var(--surface-3) !important; }
[data-testid="stExpander"] summary p { color: var(--ink) !important; font-weight: 600 !important;
  font-size: var(--t-body) !important; }
[data-testid="stExpander"] svg { fill: var(--muted) !important; }

/* ══ native alerts — dark variants (no side stripes) ═══════════════════════ */
[data-testid="stAlert"] { border-radius: var(--r-md) !important; border: 1px solid var(--border-strong);
  background: var(--surface-2) !important; color: var(--ink) !important; }
[data-testid="stAlert"] * { color: var(--ink) !important; }

/* ══ tooltips / captions ═══════════════════════════════════════════════════ */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important;
  font-size: var(--t-sm) !important; }

/* ══════════════════════════════════════════════════════════════════════════
   COMPONENT LAYER — race-predictor primitives (rp-*)
   ══════════════════════════════════════════════════════════════════════════ */

/* ── top app bar ── */
.rp-appbar {
  position: relative; overflow: hidden;
  border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: var(--s5) var(--s6); margin-bottom: var(--s5);
  background:
    radial-gradient(120% 180% at 12% 0%, var(--mesh-brand), transparent 55%),
    radial-gradient(120% 180% at 88% 120%, var(--mesh-info), transparent 55%),
    linear-gradient(180deg, var(--surface-2), var(--surface));
}
.rp-appbar::after {  /* fine diagonal rule lines, echoing the brand texture */
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background: repeating-linear-gradient(115deg, transparent 0 38px,
    var(--texture) 38px 39px);
}
.rp-appbar-row { position: relative; z-index: var(--z-base);
  display: flex; align-items: flex-end; justify-content: space-between; gap: var(--s4);
  flex-wrap: wrap; }
/* The wordmark. One face, so weight and tracking do the work the serif used
   to: 800 at display size, tracked in hard, with the product name in the brand
   step and the version as a pill rather than an italic afterthought. */
.rp-appbar h1 {
  font-family: var(--f-display) !important; font-weight: 800;
  font-size: var(--t-2xl); margin: 0; color: var(--ink);
  letter-spacing: -0.035em; line-height: 1.02;
  display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 .34ch;
}
.rp-appbar h1 .wm-a { color: var(--ink); }
.rp-appbar h1 .wm-b { color: var(--brand-text); }
.rp-appbar h1 .v {
  align-self: center; margin-left: var(--s2);
  font-size: var(--t-xs); font-weight: 700; font-style: normal;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--brand-text); background: var(--brand-wash);
  border: 1px solid var(--brand); border-radius: var(--r-pill);
  padding: 3px 10px; line-height: 1.5;
}
.rp-appbar .sub { color: var(--ink-2); font-size: var(--t-sm); margin-top: var(--s1);
  font-variant-numeric: tabular-nums; }

/* brand lockup (sidebar) */
.rp-brand { display: flex; align-items: center; gap: var(--s2);
  padding-bottom: var(--s4); margin-bottom: var(--s1);
  border-bottom: 1px solid var(--border); }
.rp-brand .mk { flex: none; }
.rp-brand .wm { font-family: var(--f-serif); font-weight: 600; font-size: var(--t-md);
  color: var(--ink); letter-spacing: -0.01em; line-height: 1; }
.rp-brand .wm em { font-style: italic; color: var(--brand-text); }
.rp-brand .tag { font-size: var(--t-xs); color: var(--muted); letter-spacing: 0.08em;
  text-transform: uppercase; margin-top: 2px; }

/* model-trust badge (app-bar right) */
.rp-trust { display: inline-flex; align-items: center; gap: var(--s2);
  background: var(--surface); border: 1px solid var(--border-strong);
  border-radius: var(--r-pill); padding: var(--s2) var(--s3); }
.rp-trust .dot { width: 7px; height: 7px; border-radius: 50%; flex: none; }
.rp-trust .lab { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.08em; }
.rp-trust .val { font-size: var(--t-sm); color: var(--ink); font-family: var(--f-mono);
  font-variant-numeric: tabular-nums; font-weight: 500; }
.rp-trust + .rp-trust { margin-left: var(--s2); }

/* ── KPI rail ── */
.rp-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--s3); margin-bottom: var(--s6); }
.rp-kpi { position: relative; overflow: hidden; container-type: inline-size;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4); box-shadow: var(--shadow-card);
  transition: border-color var(--dur) var(--ease); }
.rp-kpi:hover { border-color: var(--border-strong); }
/* 2px tone accent on the top edge - makes a rail of tiles scannable by colour
   before it is readable by label. Reuses the existing icon_tone argument. */
.rp-kpi::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: var(--border-strong); }
.rp-kpi.t-info::before    { background: var(--info); }
.rp-kpi.t-value::before   { background: var(--value); }
.rp-kpi.t-amber::before   { background: var(--amber); }
.rp-kpi.t-brand::before,
.rp-kpi.t-oxblood::before { background: var(--brand); }
.rp-kpi .k-lab { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.08em; margin-bottom: var(--s2); }
/* proportional figures: tabular-nums gives every digit the width of a zero,
   which reads loose at 36px. Tabular stays in table columns and axis ticks. */
/* the value answers to ITS TILE's width: a six-up rail gives each tile ~160px,
   where a fixed 36px figure wraps "€-27.50" onto two lines */
.rp-kpi .k-val { font-size: clamp(1.4rem, 13cqw, var(--t-2xl)); font-weight: 600;
  color: var(--ink); font-family: var(--f-mono); line-height: 1.05;
  letter-spacing: -0.02em; white-space: nowrap; }
.rp-kpi .k-val.sm { font-size: var(--t-md); font-family: var(--f-sans); letter-spacing: 0; }
.rp-kpi .k-val.value { color: var(--value); }
.rp-kpi .k-sub { font-size: var(--t-sm); color: var(--muted); margin-top: var(--s1); }

/* ── race card ── */
.rp-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); overflow: hidden; margin-bottom: var(--s3);
  box-shadow: var(--shadow-card); }
.rp-card-hdr { display: flex; align-items: center; gap: var(--s2);
  padding: var(--s3) var(--s4); border-bottom: 1px solid var(--border);
  background: var(--surface-2); flex-wrap: wrap; }
.rp-card-hdr .venue { font-family: var(--f-serif); font-weight: 600; font-size: var(--t-md);
  color: var(--ink); }
.rp-card-hdr .rtime { font-size: var(--t-sm); color: var(--ink-2); font-family: var(--f-mono);
  font-variant-numeric: tabular-nums; margin-right: var(--s1); }
.rp-card-hdr .spacer { flex: 1; }
.rp-card-hdr a.rp-open { font-size: var(--t-sm); color: var(--info); font-weight: 500;
  display: inline-flex; align-items: center; gap: 4px; padding: 2px 4px; border-radius: var(--r-sm); }
.rp-card-hdr a.rp-open:hover { color: var(--ink); text-decoration: none; background: var(--surface-3); }

/* ── pills ── */
.pill { display: inline-flex; align-items: center; gap: 4px; font-size: var(--t-xs);
  padding: 2px 9px; border-radius: var(--r-pill); white-space: nowrap; line-height: 1.6;
  border: 1px solid transparent; }
.pill-field { background: var(--surface-3); color: var(--ink-2); border-color: var(--border); }
.pill-ccy   { background: var(--surface-3); color: var(--ink-2); border-color: var(--border);
  font-family: var(--f-mono); }
.pill-ew    { background: var(--amber-wash); color: var(--amber); border-color: var(--amber-line);
  font-weight: 600; }
.pill-value { background: var(--value-wash); color: var(--value); border-color: var(--value-line);
  font-weight: 700; }

/* ── runner table ── */
.rp-body { padding: var(--s2) var(--s2) var(--s3); overflow-x: auto; }
.rp-tbl { width: 100%; border-collapse: collapse; font-size: var(--t-data); }
.rp-tbl th { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600; padding: var(--s2) var(--s2);
  border-bottom: 1px solid var(--border); text-align: left; white-space: nowrap; }
.rp-tbl th.num, .rp-tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }
.rp-tbl td { padding: var(--s3) var(--s2); border-bottom: 1px solid var(--border);
  vertical-align: middle; color: var(--ink-2); }
.rp-tbl tr:last-child td { border-bottom: none; }
.rp-tbl tr.is-value td { background: var(--value-wash); }
.rp-tbl tr:hover td { background: var(--surface-2); }
.rp-tbl tr.is-value:hover td { background: var(--value-wash-2); }
.rp-horse { color: var(--ink); font-weight: 600; font-size: var(--t-sm); }
.rp-sub { color: var(--muted); font-size: var(--t-xs); }
.rp-odds { font-family: var(--f-mono); font-variant-numeric: tabular-nums; color: var(--ink);
  font-weight: 500; }
/* recommended paper stake — money, so it keeps its currency symbol */
.rp-stake { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  color: var(--ink); font-weight: 600; white-space: nowrap;
  border-left: 2px solid var(--value); padding-left: var(--s2); }

/* ── rank badge ── */
.rk { display: inline-flex; align-items: center; justify-content: center;
  font-family: var(--f-mono); font-size: var(--t-xs); font-weight: 600;
  width: 22px; height: 22px; border-radius: var(--r-sm); }
.rk1 { background: var(--brand); color: var(--on-brand); box-shadow: 0 0 0 1px var(--brand-deep); }
.rk-n { background: var(--surface-3); color: var(--ink-2); border: 1px solid var(--border); }

/* ── probability bar ── */
.pb { display: flex; align-items: center; gap: var(--s2); min-width: 96px; }
.pb-track { flex: 1; height: 6px; background: var(--info-track); border-radius: var(--r-pill);
  min-width: 48px; max-width: 110px; overflow: hidden; }
/* width is carried on the `--pbw` custom property the component sets inline, so
   the fill can grow from 0 → target on mount (keyframe below) AND tween live
   updates (transition). Under reduced motion the keyframe is absent and the bar
   simply renders at its final width. */
.pb-fill { height: 100%; border-radius: var(--r-pill); background: var(--info);
  width: var(--pbw, 0%); transition: width var(--dur-slow) var(--ease); }
.pb-fill.win  { background: var(--info); }
.pb-fill.place{ background: var(--info-2); }
.pb-fill.show { background: var(--info-3); }
.pb-val { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  font-size: var(--t-data); color: var(--ink); min-width: 34px; text-align: right; }

/* ── value badge ── */
.val-badge { display: inline-flex; align-items: center; gap: 5px; font-size: var(--t-xs);
  font-weight: 700; letter-spacing: 0.04em; padding: 2px 7px; border-radius: var(--r-sm);
  background: var(--value-wash); color: var(--value); border: 1px solid var(--value-line); }
.val-badge .vd { width: 5px; height: 5px; border-radius: 50%; background: var(--value); flex: none; }
.ew-badge { display: inline-flex; align-items: center; gap: 4px; font-size: var(--t-xs);
  font-weight: 600; padding: 2px 7px; border-radius: var(--r-sm);
  background: var(--amber-wash); color: var(--amber); border: 1px solid var(--amber-line); }

/* ── EV figure ── */
.ev { font-family: var(--f-mono); font-variant-numeric: tabular-nums; font-size: var(--t-data); }
.ev-pos { color: var(--value); font-weight: 600; }
.ev-flat { color: var(--muted); }

/* ── confidence chip ── */
.conf { display: inline-flex; align-items: center; gap: 5px; font-size: var(--t-xs);
  padding: 2px 8px; border-radius: var(--r-pill); border: 1px solid var(--border);
  background: var(--surface-3); color: var(--ink-2); white-space: nowrap; }
.conf .cdots { display: inline-flex; gap: 2px; }
.conf .cdots i { width: 5px; height: 5px; border-radius: 50%; background: var(--border-strong); display: block; }
.conf.high .cdots i { background: var(--value); }
.conf.med  .cdots i:nth-child(-n+2) { background: var(--amber); }
.conf.low  .cdots i:nth-child(1)   { background: var(--muted); }

/* ── per-book price chips ── */
.bk-row { display: flex; flex-wrap: wrap; gap: 3px 8px; margin-top: 5px; }
.bk { font-family: var(--f-mono); font-size: var(--t-xs); color: var(--ink-2);
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.bk .bkn { color: var(--muted); }
/* best price is carried by weight and the star glyph, never by the
   reserved value green - see DESIGN.md §4.3 */
.bk-best { color: var(--ink); font-weight: 600; }

/* ── SHAP driver bars (detail view) ── */
.why { display: grid; gap: var(--s2); }
.why-row { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: var(--s3);
  font-size: var(--t-sm); }
.why-lab { color: var(--ink-2); }
.why-lab .wv { color: var(--muted); font-family: var(--f-mono); font-size: var(--t-xs); }
.why-bar { position: relative; height: 8px; background: var(--surface-3); border-radius: var(--r-pill);
  width: 160px; overflow: hidden; }
.why-bar i { position: absolute; top: 0; bottom: 0; border-radius: var(--r-pill); display: block;
  width: var(--bw, 0%); }
.why-bar i.pos { left: 50%; background: var(--value); }
.why-bar i.neg { right: 50%; background: var(--danger); }
.why-bar::before { content: ""; position: absolute; left: 50%; top: -2px; bottom: -2px;
  width: 1px; background: var(--border-strong); }

/* ── footer / status ── */
.rp-footer { margin-top: var(--s10); padding: var(--s4) 0; border-top: 1px solid var(--border);
  display: flex; justify-content: space-between; gap: var(--s2); flex-wrap: wrap;
  font-size: var(--t-sm); color: var(--muted); }
.rp-footer .mono { font-family: var(--f-mono); color: var(--ink-2); }
.sdot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px;
  vertical-align: middle; }
.sdot-ok { background: var(--value); } .sdot-warn { background: var(--amber); }
.sdot-err { background: var(--danger); }

/* ── empty / hero state ── */
.rp-empty { position: relative; overflow: hidden; text-align: center;
  border: 1px solid var(--border); border-radius: var(--r-lg);
  padding: var(--s16) var(--s6); background: var(--surface); }
.rp-empty.hero {
  background-image:
    linear-gradient(180deg, var(--hero-scrim-a), var(--hero-scrim-b)),
    url("app/static/racecourse-dawn.webp");
  background-size: cover; background-position: center 38%;
}
.rp-empty .mk { margin-bottom: var(--s4); opacity: .92; }
.rp-empty h3 { font-family: var(--f-serif); font-size: var(--t-lg); color: var(--ink);
  margin: 0 0 var(--s2); font-weight: 600; text-wrap: balance; }
.rp-empty p { font-size: var(--t-body); color: var(--ink-2); margin: 0 auto; max-width: 52ch;
  line-height: 1.6; }
.rp-empty p code { background: var(--surface-3); padding: 1px 6px; border-radius: var(--r-sm);
  color: var(--ink); font-size: var(--t-sm); }

/* ── loading mark (track oval + running dot) ── */
.rp-loader { display: inline-grid; place-items: center; }
.rp-loader .dot { animation: rp-orbit 1.4s linear infinite; transform-origin: center; }
@keyframes rp-orbit { to { transform: rotate(360deg); } }

/* ── back link ── */
.rp-back { display: inline-flex; align-items: center; gap: 6px; font-size: var(--t-sm);
  color: var(--muted); margin-bottom: var(--s3); }
.rp-back:hover { color: var(--ink); }

/* ── detail view: two-column runner block (price/probs | why) ── */
.rp-detail-grid { display: grid; grid-template-columns: 1fr 1.2fr; gap: var(--s6);
  padding: var(--s5); align-items: start; }

/* ── section label (used sparingly, NOT a per-section eyebrow) ── */
.rp-sec { font-family: var(--f-sans); font-size: var(--t-md); font-weight: 600;
  color: var(--ink); margin: var(--s6) 0 var(--s3); display: flex; align-items: baseline;
  gap: var(--s2); }
.rp-sec .n { color: var(--muted); font-size: var(--t-sm); font-family: var(--f-mono); }

/* ── paper-betting reminder banner (never let this read as real money) ── */
.rp-paper-note { display: flex; align-items: center; gap: var(--s2);
  background: var(--amber-wash, rgba(235,174,81,.10)); color: var(--ink);
  border: 1px solid var(--amber); border-radius: var(--r-md);
  padding: var(--s2) var(--s4); font-size: var(--t-sm); margin-bottom: var(--s4); }
.rp-paper-note .dot { width: 8px; height: 8px; border-radius: 50%;
  background: var(--amber); flex: 0 0 auto; }
.rp-paper-note b { color: var(--amber); font-weight: 700; }

/* ── confidence-tier badge (Today's suggestions) ── */
.tier { display: inline-flex; align-items: center; gap: 5px; font-size: var(--t-xs);
  font-weight: 700; letter-spacing: 0.05em; padding: 2px 8px; border-radius: var(--r-sm);
  white-space: nowrap; }
.tier .td { width: 5px; height: 5px; border-radius: 50%; flex: none; }
.tier-strong { background: var(--value-wash); color: var(--value);
  border: 1px solid var(--value-line-2); }
.tier-strong .td { background: var(--value); }
.tier-lean { background: transparent; color: var(--value-dim);
  border: 1px dashed var(--value-dim-line); }
.tier-lean .td { background: var(--value-dim); }
.tier-pass { background: var(--surface-3); color: var(--muted); border: 1px solid var(--border); }

/* ── suggestion card ── */
.sg-card { margin-bottom: var(--s4); }
.sg-card .rp-card-hdr .venue { font-family: var(--f-serif); }
.sg-body { padding: var(--s4); }
.sg-sub { display: flex; align-items: baseline; gap: var(--s2); flex-wrap: wrap;
  margin-bottom: var(--s4); }
.sg-sub .venue { font-family: var(--f-serif); font-weight: 600; color: var(--ink);
  font-size: var(--t-body); }
.sg-sub .rtime { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  color: var(--ink-2); font-size: var(--t-sm); }
.sg-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr));
  gap: var(--s2) var(--s3); padding: var(--s3) 0; border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border); }
.sg-stat .sg-lab { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.07em; margin-bottom: 3px; }
.sg-stat .sg-val { font-size: var(--t-sm); color: var(--ink); font-variant-numeric: tabular-nums; }
.sg-stat.value .sg-val { color: var(--value); font-weight: 600; }
.sg-rationale { margin-top: var(--s4); color: var(--ink-2); font-size: var(--t-sm);
  line-height: 1.55; }
.sg-why { margin-top: var(--s4); }

/* ── recent-form table (horse detail) ── */
.rp-form td { font-size: var(--t-data); }
.rp-form .mono { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.fp { display: inline-flex; align-items: baseline; justify-content: center;
  font-family: var(--f-mono); font-weight: 600; font-size: var(--t-data);
  min-width: 30px; padding: 2px 6px; border-radius: var(--r-sm); }
.fp sup { font-size: 0.62em; margin-left: 1px; font-weight: 500; }
.fp1  { background: var(--brand); color: var(--on-brand); box-shadow: 0 0 0 1px var(--brand-deep); }
.fpp  { background: var(--amber-wash); color: var(--amber); border: 1px solid var(--amber-line); }
.fp-n { background: var(--surface-3); color: var(--ink-2); border: 1px solid var(--border); }

/* horse-name link (race detail → horse detail) */
a.rp-hlink, a.rp-hlink:visited { color: var(--ink) !important; }
a.rp-hlink:hover { color: var(--brand-text) !important; text-decoration: none;
  text-underline-offset: 3px; }

/* ── model's view panel (race detail) ── */
.rp-modelview { background:
    linear-gradient(180deg, var(--surface-2), var(--surface));
  border: 1px solid var(--border); border-left: 3px solid var(--brand);
  border-radius: var(--r-md); padding: var(--s4) var(--s5); margin-bottom: var(--s4); }
.rp-modelview .mv-lab { font-size: var(--t-xs); color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
  margin-bottom: var(--s2); }
.rp-modelview .mv-lead { font-size: var(--t-md); color: var(--ink);
  font-family: var(--f-serif); }
.rp-modelview .mv-lead b { font-weight: 600; }
.rp-modelview .mv-num { font-family: var(--f-mono); color: var(--info);
  font-variant-numeric: tabular-nums; font-size: var(--t-sm); margin-left: 4px; }
.rp-modelview .mv-shape { color: var(--muted); font-family: var(--f-sans);
  font-size: var(--t-sm); }
.rp-modelview .mv-row { display: flex; align-items: center; gap: 6px;
  margin-top: var(--s2); font-size: var(--t-sm); color: var(--value); }
.rp-modelview .mv-row .vd { width: 6px; height: 6px; border-radius: 50%;
  background: var(--value); flex: none; }
.rp-modelview .mv-row b { color: var(--value); }
.rp-modelview .mv-note { margin-top: var(--s2); font-size: var(--t-sm);
  color: var(--ink-2); line-height: 1.55; }

/* ── context chips (going / distance / pace) ── */
.ctx-row { display: flex; flex-wrap: wrap; gap: var(--s2); margin: var(--s2) 0 var(--s4); }
.ctx { display: inline-flex; flex-direction: column; gap: 1px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: var(--s2) var(--s3); min-width: 88px; }
.ctx .ctx-l { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.07em; }
.ctx .ctx-v { font-size: var(--t-sm); color: var(--ink); font-weight: 500; }

/* ── horse-detail run-style / aggregate row ── */
.rp-hstat { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--s3); margin-bottom: var(--s5); }

/* ══════════════════════════════════════════════════════════════════════════
   IN-HOUSE GRAPHICS — SVG icon set, illustrations & meters (ui/_icons.py).
   All theme-driven: glyphs stroke in `currentColor` so they inherit the colour
   of their container; illustrations + meters read theme tokens directly. No
   external image files. (Catalogued in memory/ui-26-css-graphics.md.)
   ══════════════════════════════════════════════════════════════════════════ */

/* line-art icons inherit text colour + align to the baseline of adjacent text */
.rp-ico { display: inline-block; vertical-align: -0.18em; flex: none;
  color: inherit; }
/* an icon chip — a tinted square plate behind a glyph (KPI labels, list rows) */
.rp-ico-plate { display: inline-grid; place-items: center; width: 30px; height: 30px;
  border-radius: var(--r-sm); background: var(--surface-3); color: var(--ink-2);
  border: 1px solid var(--border); flex: none; }
.rp-ico-plate.info    { background: var(--info-wash);    color: var(--info);
  border-color: var(--info-line); }
.rp-ico-plate.value   { background: var(--value-wash);   color: var(--value);
  border-color: var(--value-line); }
.rp-ico-plate.amber   { background: var(--amber-wash);   color: var(--amber);
  border-color: var(--amber-line); }
.rp-ico-plate.oxblood, .rp-ico-plate.brand { background: var(--brand-wash);  color: var(--brand-text);
  border-color: var(--brand-deep); }

/* KPI rail with a leading icon plate — extends .rp-kpi without changing it */
.rp-kpi .k-head { display: flex; align-items: center; gap: var(--s2);
  margin-bottom: var(--s2); }
.rp-kpi .k-head .k-lab { margin-bottom: 0; }

/* segmented strength meter — coarse "how strong" read; fill COUNT carries it */
.rp-meter { display: inline-flex; align-items: center; gap: var(--s2);
  white-space: nowrap; }
.rp-meter-pips { display: inline-flex; gap: 3px; }
.rp-meter-pips i { width: 7px; height: 12px; border-radius: 2px;
  background: var(--surface-3); border: 1px solid var(--border); display: block;
  transition: background var(--dur) var(--ease); }
.rp-meter.tone-info    .rp-meter-pips i.on { background: var(--info);
  border-color: var(--info); }
.rp-meter.tone-value   .rp-meter-pips i.on { background: var(--value);
  border-color: var(--value); }
.rp-meter.tone-amber   .rp-meter-pips i.on { background: var(--amber);
  border-color: var(--amber); }
.rp-meter.tone-oxblood .rp-meter-pips i.on,
.rp-meter.tone-brand .rp-meter-pips i.on { background: var(--brand-text);
  border-color: var(--brand-text); }
.rp-meter-val { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  font-size: var(--t-data); color: var(--ink); }

/* semicircular probability gauge (single hero number) */
.rp-gauge { position: relative; display: inline-grid; justify-items: center; }
.rp-gauge svg { display: block; }
/* the filled arc tweens to its value on mount when motion is welcome */
.rp-gauge svg path:last-child { transition: stroke-dasharray var(--dur-slow) var(--ease); }
.rp-gauge-val { margin-top: -0.55em; font-family: var(--f-mono);
  font-variant-numeric: tabular-nums; font-weight: 600; font-size: var(--t-md);
  color: var(--ink); line-height: 1; }

/* ── hero / empty illustration container ── */
.rp-illus { display: block; margin: 0 auto var(--s4); max-width: 100%; height: auto; }

/* ──────────────────────────────────────────────────────────────────────────
   HERO BACKGROUND MOTIF — in-house, no photo. A layered OKLCH mesh-gradient
   ground with a fine concentric "track-rail" line motif (echoing a racecourse
   bend) masked to fade outward. Used by `.rp-empty.hero-motif`. Premium use of
   mask/blur, but it's a structural background texture — NOT decorative
   glassmorphism over content, so it stays inside impeccable's bans.
   ────────────────────────────────────────────────────────────────────────── */
.rp-empty.hero-motif {
  background:
    radial-gradient(120% 150% at 8% -10%, var(--mesh-brand), transparent 50%),
    radial-gradient(120% 150% at 100% 0%, var(--mesh-info), transparent 52%),
    radial-gradient(140% 160% at 70% 120%, var(--mesh-value), transparent 55%),
    linear-gradient(180deg, var(--surface-2), var(--surface));
}
.rp-empty.hero-motif::before {  /* concentric track rails, faded outward by a mask */
  content: ""; position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background:
    repeating-radial-gradient(120% 150% at 50% 130%,
      transparent 0 26px, var(--texture-2) 26px 27px);
  -webkit-mask-image: radial-gradient(120% 120% at 50% 60%, #000 35%, transparent 78%);
          mask-image: radial-gradient(120% 120% at 50% 60%, #000 35%, transparent 78%);
}
.rp-empty.hero-motif > * { position: relative; z-index: 1; }

/* a wider hero band (landing splash) — same motif, taller, with a soft glow */
.rp-hero { position: relative; overflow: hidden; border: 1px solid var(--border);
  border-radius: var(--r-lg); padding: var(--s12) var(--s8); margin-bottom: var(--s5);
  text-align: center;
  background:
    radial-gradient(110% 140% at 6% -10%, var(--mesh-brand), transparent 48%),
    radial-gradient(110% 140% at 100% 10%, var(--mesh-info), transparent 50%),
    linear-gradient(180deg, var(--surface-2), var(--surface)); }
.rp-hero::before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: repeating-radial-gradient(120% 150% at 50% 135%,
    transparent 0 30px, var(--texture-2) 30px 31px);
  -webkit-mask-image: radial-gradient(120% 120% at 50% 55%, #000 30%, transparent 80%);
          mask-image: radial-gradient(120% 120% at 50% 55%, #000 30%, transparent 80%); }
.rp-hero > * { position: relative; z-index: 1; }

/* ══════════════════════════════════════════════════════════════════════════
   LOADING SKELETONS — shown while the ~1.7s inference / value scan runs, so the
   surface reads as "working", not "broken/blank". Shimmer is stilled under
   reduced motion (the shapes stay, the sweep stops).
   ══════════════════════════════════════════════════════════════════════════ */
.rp-skel-rail { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: var(--s3); margin-bottom: var(--s6); }
.rp-skel-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-lg); overflow: hidden; margin-bottom: var(--s3);
  box-shadow: var(--shadow-card); }
.rp-skel-card.kpi { border-radius: var(--r-md); padding: var(--s4); }
.rp-skel-hdr { display: flex; align-items: center; gap: var(--s3); padding: var(--s3) var(--s4);
  border-bottom: 1px solid var(--border); background: var(--surface-2); }
.rp-skel-body { padding: var(--s4); display: grid; gap: var(--s3); }
.sk { position: relative; overflow: hidden; border-radius: var(--r-sm);
  background: var(--surface-3); height: 12px; }
.sk.lg { height: 28px; width: 56%; } .sk.sm { height: 9px; } .sk.pill { border-radius: var(--r-pill); }
.sk.w40 { width: 40%; } .sk.w60 { width: 60%; } .sk.w25 { width: 25%; } .sk.chip { width: 64px; height: 22px; }
.sk::after { content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, var(--shimmer), transparent);
  transform: translateX(-100%); animation: rp-shimmer 1.4s var(--ease) infinite; }
@keyframes rp-shimmer { to { transform: translateX(100%); } }
.rp-skel-cap { display: flex; align-items: center; justify-content: center; gap: var(--s2);
  color: var(--muted); font-size: var(--t-sm); margin: var(--s3) 0 var(--s5); }

/* ══════════════════════════════════════════════════════════════════════════
   PURPOSEFUL MOTION — entrances are scoped under `no-preference` so the default
   (and the reduced-motion default) is the fully-visible end state; the keyframes
   only ever exist when motion is welcome. This satisfies "reveal must enhance an
   already-visible default" — nothing is gated behind a class/JS hook, so a
   headless render or a background tab never ships blank.
   ══════════════════════════════════════════════════════════════════════════ */
@keyframes rp-rise   { from { opacity: 0; transform: translateY(8px);  } to { opacity: 1; transform: none; } }
@keyframes rp-rise-sm{ from { opacity: 0; transform: translateY(4px);  } to { opacity: 1; transform: none; } }
@keyframes rp-fade   { from { opacity: 0; } to { opacity: 1; } }
@keyframes pb-grow   { from { width: 0; } to { width: var(--pbw, 0%); } }
@keyframes why-grow  { from { width: 0; } to { width: var(--bw, 0%); } }

@media (prefers-reduced-motion: no-preference) {
  /* probability + driver bars sweep to their value on mount */
  .pb-fill  { animation: pb-grow  var(--dur-slow) var(--ease) both; }
  .why-bar i { animation: why-grow var(--dur-slow) var(--ease) both; }

  /* KPI rail tiles cascade in (true siblings in one grid → honest stagger) */
  .rp-kpis > .rp-kpi { animation: rp-rise var(--dur) var(--ease) both; }
  .rp-kpis > .rp-kpi:nth-child(1) { animation-delay: 0ms; }
  .rp-kpis > .rp-kpi:nth-child(2) { animation-delay: 45ms; }
  .rp-kpis > .rp-kpi:nth-child(3) { animation-delay: 90ms; }
  .rp-kpis > .rp-kpi:nth-child(4) { animation-delay: 135ms; }
  .rp-kpis > .rp-kpi:nth-child(5) { animation-delay: 180ms; }
  .rp-kpis > .rp-kpi:nth-child(6) { animation-delay: 225ms; }

  /* runner rows reveal as a staggered list — the race-list motion the brief asks
     for. Rows are real <tr> siblings, so the cascade is structural, not a reflex
     pasted onto every section. Capped so long fields don't ripple forever. */
  .rp-tbl tbody tr { animation: rp-rise-sm var(--dur) var(--ease) both; }
  .rp-tbl tbody tr:nth-child(1) { animation-delay: 20ms; }
  .rp-tbl tbody tr:nth-child(2) { animation-delay: 50ms; }
  .rp-tbl tbody tr:nth-child(3) { animation-delay: 80ms; }
  .rp-tbl tbody tr:nth-child(4) { animation-delay: 110ms; }
  .rp-tbl tbody tr:nth-child(5) { animation-delay: 140ms; }
  .rp-tbl tbody tr:nth-child(6) { animation-delay: 170ms; }
  .rp-tbl tbody tr:nth-child(n+7) { animation-delay: 195ms; }

  /* "why" driver rows reveal in sequence under the bar sweep */
  .why > .why-row { animation: rp-rise-sm var(--dur) var(--ease) both; }
  .why > .why-row:nth-child(2) { animation-delay: 55ms; }
  .why > .why-row:nth-child(3) { animation-delay: 110ms; }
  .why > .why-row:nth-child(4) { animation-delay: 165ms; }
  .why > .why-row:nth-child(n+5) { animation-delay: 200ms; }

  /* state feedback — bet placed / settled / stale-cache banners fade-rise in so
     the confirmation registers as a change, not a silent repaint */
  [data-testid="stAlert"] { animation: rp-rise var(--dur) var(--ease) both; }
  .rp-paper-note { animation: rp-fade var(--dur-slow) var(--ease) both; }
  /* skeleton placeholders fade in (and out, via the cap) without a jolt */
  .rp-skel-rail, .rp-skel-card { animation: rp-fade var(--dur) var(--ease) both; }
}

/* ══ responsive ════════════════════════════════════════════════════════════ */
/* tablet / small-laptop: relax the page gutter and step the KPI rail down to a
   3-up before it reaches the 2-up mobile rule, and let long venue/horse titles
   wrap instead of overflowing the app bar (the viewport is part of the design). */
@media (max-width: 980px) {
  .main .block-container, [data-testid="stMainBlockContainer"] {
    padding: var(--s4) var(--s5) var(--s12) !important;
  }
  .rp-kpis { grid-template-columns: repeat(3, 1fr); }
  .rp-detail-grid { grid-template-columns: 1fr; gap: var(--s5); }
}
.rp-appbar h1, .rp-card-hdr .venue, .rp-empty h3, .sg-sub .venue {
  overflow-wrap: anywhere; }

@media (max-width: 640px) {
  .main .block-container, [data-testid="stMainBlockContainer"] {
    padding: var(--s3) var(--s3) var(--s12) !important;
  }
  .rp-appbar { padding: var(--s4); }
  .rp-appbar h1 { font-size: var(--t-xl); letter-spacing: -0.03em; }
  .rp-appbar h1 .v { font-size: 10px; padding: 2px 8px; }
  .rp-kpis { grid-template-columns: repeat(2, 1fr); }
  .rp-tbl { font-size: var(--t-xs); }
  /* column priority: price, EV and stake are the scan targets, so confidence
     and the decimal-alongside give up their space first. */
  .rp-tbl th:nth-child(5), .rp-tbl td:nth-child(5) { display: none; }
  .rp-odds + .rp-sub { display: none; }
  .why-bar { width: 96px; }
  .rp-detail-grid { grid-template-columns: 1fr; gap: var(--s4); }
}

/* ══ reduced motion ════════════════════════════════════════════════════════ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important;
    animation-iteration-count: 1 !important; transition-duration: .001ms !important; }
  .rp-loader .dot { animation: none; }
}

/* ══ legacy page class map ════════════════════════════════════ */
/* Pages 2-5 were written against their own stylesheets and emit class names
   that predate the rp-* layer - sometimes the SAME name with a different child
   contract (`.rp-kpi > .kpi-label` where this system expects `.k-lab`). Rather
   than rewrite ~3,000 lines of render logic, those names are mapped onto the
   tokens here. Everything below is a bridge: when a page's markup is ported to
   the ui/_components.py builders, delete its rows.

   Two rules are enforced on the way through, not preserved:
     - the legacy rank ramp was a rainbow (rk1 blue / rk2 violet / rk3 cyan);
       this system fills rank 1 and leaves the rest neutral.
     - the legacy `.stake` and `.bk-best` were green, which is reserved for
       positive EV. They lose it, exactly as the runner table did. */

/* page header (-> .rp-appbar) */
.rp-header { border-bottom: 1px solid var(--border); padding-bottom: var(--s4);
  margin-bottom: var(--s6); }
.rp-header h1 { font-family: var(--f-serif); font-size: var(--t-lg); font-weight: 600;
  margin: 0 0 var(--s1); color: var(--ink); }
.rp-header .sub { font-size: var(--t-data); color: var(--muted); }

/* KPI strip (-> .rp-kpis / .k-lab / .k-val / .k-sub) */
.rp-kpi-row { display: flex; gap: var(--s3); margin-bottom: var(--s8); flex-wrap: wrap; }
.rp-kpi-row > .rp-kpi { flex: 1; min-width: 120px; padding: var(--s3) var(--s4); }
.rp-kpi .kpi-label { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: 0.06em; margin-bottom: var(--s1); }
.rp-kpi .kpi-val { font-size: var(--t-xl); font-weight: 600; letter-spacing: -0.02em;
  color: var(--ink); font-family: var(--f-mono); }
.rp-kpi .kpi-sub { font-size: var(--t-sm); color: var(--muted); margin-top: 2px; }

/* card body (-> .rp-body) and card footer meta */
.rp-card-body { padding: var(--s3) var(--s4); }
.rp-card-meta { font-size: var(--t-xs); color: var(--muted); text-align: right;
  padding: var(--s1) var(--s4) var(--s3); }

/* currency + source + prediction pills */
.pill-gbp, .pill-eur { background: var(--surface-3); color: var(--ink-2);
  border: 1px solid var(--border); font-family: var(--f-mono); }
/* was green: a data-source tag is not positive EV */
.pill-src  { background: var(--surface-3); color: var(--muted);
  border: 1px solid var(--border); font-size: 10px; padding: 1px 7px; }
.pill-pred { background: var(--amber-wash); color: var(--amber);
  border: 1px solid var(--amber-line); font-weight: 600; }

/* live race status - the one place a pulse is warranted, so it is gated */
.pill-upcoming { background: var(--info-wash);  color: var(--info);
  border: 1px solid var(--info-line); }
.pill-starting { background: var(--amber-wash); color: var(--amber);
  border: 1px solid var(--amber-line); font-weight: 600; }
.pill-inplay   { background: var(--value-wash); color: var(--value);
  border: 1px solid var(--value-line); font-weight: 600; }
.pill-finished { background: var(--surface-3); color: var(--muted);
  border: 1px solid var(--border); }
@media (prefers-reduced-motion: no-preference) {
  .pill-starting { animation: st-pulse 1.4s ease-in-out infinite; }
}
@keyframes st-pulse { 0%, 100% { opacity: 1; } 50% { opacity: .55; } }

/* predicted-runner highlight (-> tr.is-value) */
.rp-tbl tr.pred-row td { background: var(--amber-wash); }
.rp-tbl tr.pred-row td:first-child { border-left: 3px solid var(--amber); }
.pred-badge { font-size: var(--t-xs); padding: 1px 6px; border-radius: var(--r-sm);
  background: var(--amber-wash); color: var(--amber);
  border: 1px solid var(--amber-line); margin-left: 5px; font-weight: 600; }

/* rank badges - rank 1 is filled, the rest are neutral (no rainbow) */
.rk2, .rk3 { background: var(--surface-3) !important; color: var(--ink-2) !important;
  border: 1px solid var(--border); }

/* recommended paper stake - ink, with the value rule, like .rp-stake */
.stake { font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  font-weight: 600; color: var(--ink); white-space: nowrap;
  border-left: 2px solid var(--value); padding-left: var(--s2); }
.stake-band { display: block; font-size: var(--t-xs); font-weight: 500;
  color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }

/* bandwidth / info tip (-> .rp-paper-note) */
.rp-tip { background: var(--info-wash); border: 1px solid var(--info-line);
  border-radius: var(--r-sm); padding: var(--s3) var(--s4); font-size: var(--t-sm);
  color: var(--ink-2); margin-bottom: var(--s3); line-height: 1.5; }

/* empty-state emoji slot the legacy markup uses */
.rp-empty .ei { font-size: 40px; margin-bottom: var(--s3); }

/* page header (bet_placer / race_compare / performance_dashboard) */
.pg-hdr { border-bottom: 1px solid var(--border); padding-bottom: var(--s4);
  margin-bottom: var(--s6); }
.pg-hdr h1 { font-family: var(--f-serif); font-size: var(--t-lg); font-weight: 600;
  margin: 0 0 var(--s1); color: var(--ink); }
.pg-hdr .sub { font-size: var(--t-data); color: var(--muted); }
.section-label, .section-hdr { font-size: var(--t-xs); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted);
  margin: var(--s5) 0 var(--s3); }

/* value / each-way pills under the legacy names */
.pill-val { background: var(--value-wash); color: var(--value);
  border: 1px solid var(--value-line); font-weight: 600; }

/* ── bet_placer: horse detail, confirmation, banners ───────────────── */
.horse-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4) var(--s5); margin: var(--s4) 0;
  border-left: 4px solid var(--info); box-shadow: var(--shadow-card); }
.hc-name { font-size: var(--t-md); font-weight: 700; color: var(--ink);
  margin-bottom: var(--s1); }
.hc-sub { font-size: var(--t-data); color: var(--muted); margin-bottom: var(--s3); }
.hc-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(84px, 1fr));
  gap: var(--s3) var(--s5); }
.hc-stat { display: flex; flex-direction: column; align-items: center; }
.hc-stat-lbl { font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .05em;
  color: var(--muted); margin-bottom: 3px; }
.hc-stat-val { font-size: var(--t-md); font-weight: 600; color: var(--ink);
  font-family: var(--f-mono); font-variant-numeric: tabular-nums; }

.confirm-card { background: var(--info-wash); border: 1px solid var(--info-line);
  border-radius: var(--r-md); padding: var(--s5) var(--s6); margin: var(--s5) 0; }
.confirm-title { font-size: var(--t-sm); font-weight: 700; color: var(--info);
  margin-bottom: var(--s3); }
.confirm-row { display: flex; justify-content: space-between; align-items: center;
  padding: 7px 0; border-bottom: 1px solid var(--border); font-size: var(--t-sm); }
.confirm-row:last-child { border-bottom: none; }
.confirm-lbl { color: var(--muted); }
.confirm-val { font-weight: 600; color: var(--ink);
  font-family: var(--f-mono); font-variant-numeric: tabular-nums; }

.success-card { background: var(--value-wash); border: 1px solid var(--value-line);
  border-radius: var(--r-md); padding: var(--s4) var(--s5); margin: var(--s4) 0;
  display: flex; align-items: center; gap: var(--s3); }
.success-icon { font-size: 26px; }
.success-text h3 { font-size: var(--t-sm); font-weight: 700; color: var(--value);
  margin: 0 0 3px; }
.success-text p { font-size: var(--t-data); color: var(--ink-2); margin: 0; }
.stop-banner { background: var(--danger-wash); border: 1px solid var(--danger);
  border-radius: var(--r-sm); padding: var(--s3) var(--s4); margin: var(--s3) 0;
  font-size: var(--t-data); color: var(--danger); font-weight: 500; }

/* ── race_compare: A/B summary, prob rows, diff strip, table ───────── */
/* A and B are two equal peers, which is nominal identity — so they take the
   validated categorical slots rather than ad-hoc hues. The legacy B was
   #7c3aed, a near neighbour of the new brand: "Horse B" read as "the brand". */
.hc { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4) var(--s5);
  border-top: 4px solid var(--info); box-shadow: var(--shadow-card); }
.hc.side-a { border-top-color: var(--series-1); }
.hc.side-b { border-top-color: var(--series-2); }
.hc-label { font-size: var(--t-xs); font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; margin-bottom: var(--s2); }
.hc.side-a .hc-label { color: var(--series-1); }
.hc.side-b .hc-label { color: var(--series-2); }
.hc-meta { font-size: var(--t-sm); color: var(--muted); margin-bottom: var(--s3); }
.hc-odds { font-size: var(--t-lg); font-weight: 700; color: var(--ink);
  font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.hc-odds-lbl { font-size: var(--t-xs); color: var(--muted); margin-bottom: var(--s4); }

.prob-row { display: flex; align-items: center; gap: var(--s3); margin-bottom: var(--s2); }
.prob-row-lbl { font-size: var(--t-xs); color: var(--muted); width: 48px; flex-shrink: 0; }
.prob-row-bar { flex: 1; height: 6px; background: var(--info-track);
  border-radius: var(--r-pill); overflow: hidden; }
.prob-row-fill { height: 100%; border-radius: var(--r-pill); background: var(--info); }
.prob-row-val { font-size: var(--t-sm); font-weight: 600; color: var(--ink);
  font-family: var(--f-mono); font-variant-numeric: tabular-nums;
  width: 38px; text-align: right; }

.composite-row { display: flex; align-items: center; gap: var(--s2);
  margin-top: var(--s3); padding-top: var(--s3); border-top: 1px solid var(--border); }
.composite-score { font-size: var(--t-lg); font-weight: 700;
  font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.composite-lbl { font-size: var(--t-xs); color: var(--muted); }

.kdiff-strip { display: flex; gap: var(--s3); flex-wrap: wrap;
  margin: var(--s5) 0 var(--s8); }
.kdiff-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: var(--s3) var(--s4); min-width: 140px; flex: 1; }
.kdiff-feat { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: .05em; margin-bottom: var(--s1); }
.kdiff-vals { font-size: var(--t-data); font-weight: 600; color: var(--ink);
  font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.kdiff-adv { font-size: var(--t-xs); margin-top: 3px; font-weight: 500; }
.adv-a, .adv-win-a { color: var(--series-1); font-weight: 600; }
.adv-b, .adv-win-b { color: var(--series-2); font-weight: 600; }
.adv-neutral, .adv-neutral-cell { color: var(--muted); }

.cmp-tbl { width: 100%; border-collapse: collapse; font-size: var(--t-data); }
.cmp-tbl th { font-size: var(--t-xs); color: var(--muted); text-transform: uppercase;
  letter-spacing: .05em; font-weight: 600; padding: 7px 10px;
  border-bottom: 2px solid var(--border); text-align: left; }
.cmp-tbl th.num, .cmp-tbl td.num, .cmp-val { text-align: right;
  font-variant-numeric: tabular-nums; }
.cmp-tbl td { padding: 7px 10px; border-bottom: 1px solid var(--border);
  vertical-align: middle; color: var(--ink-2); }
.cmp-tbl tr:last-child td { border-bottom: none; }
.cmp-grp-hdr td { font-size: var(--t-xs); font-weight: 700; color: var(--ink-2);
  text-transform: uppercase; letter-spacing: .07em; padding: var(--s3) 10px var(--s1);
  background: var(--surface-2); border-bottom: 1px solid var(--border); }
.cmp-feat { color: var(--ink-2); font-size: var(--t-sm); }

.delta-cell { min-width: 90px; }
.delta-wrap { display: flex; align-items: center; gap: var(--s1); justify-content: center; }
.delta-bar { width: 36px; height: 5px; background: var(--surface-3);
  border-radius: var(--r-pill); overflow: hidden; }
.delta-fill-a { height: 100%; border-radius: var(--r-pill); background: var(--series-1); }
.delta-fill-b { height: 100%; border-radius: var(--r-pill); background: var(--series-2); }
.delta-pct { font-size: var(--t-xs); color: var(--muted);
  font-variant-numeric: tabular-nums; }

/* ── performance_dashboard: its own KPI + chart-card family ────────── */
.kpi-wrap { display: flex; gap: var(--s3); flex-wrap: wrap; margin-bottom: var(--s2); }
.kpi { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4) var(--s5); flex: 1; min-width: 130px;
  box-shadow: var(--shadow-card); }
.kpi-lbl { font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .07em;
  color: var(--muted); margin-bottom: var(--s1); }
.kpi .kpi-val { font-size: var(--t-lg); font-weight: 700; color: var(--ink);
  font-family: var(--f-mono); line-height: 1.15; }
.kpi .kpi-sub { font-size: var(--t-xs); color: var(--muted); margin-top: 3px; }
.kpi-pos { color: var(--value); }
.kpi-neg { color: var(--danger); }
.kpi-neu { color: var(--muted); }

.chart-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s4) var(--s4) var(--s3);
  margin-bottom: var(--s4); box-shadow: var(--shadow-card); }
.chart-title { font-size: var(--t-data); font-weight: 600; color: var(--ink-2);
  margin-bottom: var(--s3); }
.pb-card { background: var(--surface); border: 1px solid var(--border);
  border-left: 3px solid var(--amber); border-radius: var(--r-md);
  padding: var(--s3) var(--s4); margin-bottom: var(--s3); }
.pb-meta { font-size: var(--t-sm); color: var(--muted); line-height: 2.1; }
.pb-meta b { color: var(--ink-2); }

/* risk-tier ramp (bet_placer / race_compare). The legacy ramp was a four-step
   rainbow (green/blue/amber/red) carrying meaning by hue alone; it collapses
   onto the three documented tiers, and the word label still travels with it. */
.horse-card.tier-strong,      .pill-tier-strong      { border-color: var(--value) !important; }
.horse-card.tier-good,        .pill-tier-good        { border-color: var(--info) !important; }
.horse-card.tier-moderate,    .pill-tier-moderate    { border-color: var(--amber) !important; }
.horse-card.tier-speculative, .pill-tier-speculative { border-color: var(--border-strong) !important; }
.pill-tier-strong      { background: var(--value-wash);  color: var(--value); }
.pill-tier-good        { background: var(--info-wash);   color: var(--info); }
.pill-tier-moderate    { background: var(--amber-wash);  color: var(--amber); }
.pill-tier-speculative { background: var(--surface-3);   color: var(--muted); }

/* ══ settings / banner family ══════════════════════════════════ */
/* Ported off ui/settings.py's own stylesheet, which carried a seventh copy of
   the palette. Token-driven, so these work in both modes. */
.settings-card { background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r-md); padding: var(--s6) var(--s6) var(--s5);
  margin-bottom: var(--s5); box-shadow: var(--shadow-card); }
.settings-card h4 { font-size: var(--t-data); font-weight: 700;
  text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
  margin: 0 0 var(--s4); padding-bottom: var(--s3);
  border-bottom: 1px solid var(--border); }
.banner-ok, .banner-warn, .banner-err {
  border-radius: var(--r-sm); padding: var(--s3) var(--s5);
  margin-bottom: var(--s4); border: 1px solid transparent;
  font-size: var(--t-sm); }
.banner-ok   { background: var(--value-wash);  color: var(--value);
  border-color: var(--value-line); font-weight: 600; }
.banner-warn { background: var(--amber-wash);  color: var(--amber);
  border-color: var(--amber-line); }
.banner-err  { background: var(--danger-wash); color: var(--danger);
  border-color: var(--danger); }
.reload-card { background: var(--info-wash); border: 1px solid var(--info-line);
  border-radius: var(--r-md); padding: var(--s5) var(--s6); margin-top: var(--s2); }

/* ══ card grid — cards on a field, not a stack ══════════════════════════ */
/* 620px: with the sidebar open, a 1680 screen leaves ~1282px of track, so 620
   is the largest minimum that still fits TWO cards (2x620 + 12 = 1252). Two
   cards of ~635px read denser than one of 1282, and 635 clears the 560px
   container query so all seven columns stay on screen. Below ~1250 of track it
   falls to one full-width card on its own. */
.rp-card-grid { display: grid; gap: var(--s3); align-items: start;
  grid-template-columns: repeat(auto-fit, minmax(620px, 1fr)); }
.rp-card-grid > .rp-card { margin-bottom: 0; }
.rp-card { transition: border-color var(--dur) var(--ease);
  container-type: inline-size; }
.rp-card:hover { border-color: var(--border-strong); }

/* A card in a two-up grid is narrow even on a wide screen, so the runner table
   answers to ITS OWN width rather than the viewport's. Same column priority as
   the 640px breakpoint: price, EV and stake are the scan targets, so confidence
   and the decimal-alongside give up their space first. */
@container (max-width: 560px) {
  .rp-tbl th:nth-child(5), .rp-tbl td:nth-child(5) { display: none; }
  .rp-odds + .rp-sub { display: none; }
  /* numeric columns size to their content so the runner name absorbs the
     leftover — otherwise the stake wraps and clips to "€0." */
  .rp-tbl th.num, .rp-tbl td.num,
  .rp-tbl th:first-child, .rp-tbl td:first-child { width: 1%; white-space: nowrap; }
  .rp-tbl td.num .rp-sub { white-space: nowrap; }
}

/* ══ focus ring — never removed ══════════════════════════════════ */
:where(a, button, [role="button"], summary, input, select, textarea):focus-visible {
  outline: 2px solid var(--brand) !important; outline-offset: 2px !important;
  border-radius: var(--r-sm);
}

/* ══ chart cards — the reference's dense grid ═══════════════════════ */
/* st.container(border=True) is the only way to get a real card around a
   Streamlit chart — a chart cannot live inside an HTML string. Styled to match
   .rp-card so a chart card and a race card read as the same object. */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface); border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
  /* tight: the chart is the content, the card is only its frame */
  padding: var(--s3) var(--s3) 0 !important;
  box-shadow: var(--shadow-card);
  transition: border-color var(--dur) var(--ease);
}
/* strip the stacking gap Streamlit puts between the header and the plot */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stElementContainer"] { margin: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: var(--border-strong) !important; }

/* the card header carries the chart's title, so the plot keeps its top margin */
.rp-chart-hd { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 var(--s2);
  margin-bottom: var(--s1); }
.rp-chart-hd .t { font-family: var(--f-serif); font-size: var(--t-sm); font-weight: 600;
  color: var(--ink); line-height: 1.3; }
.rp-chart-hd .n { font-size: var(--t-xs); color: var(--muted); }

/* Streamlit columns already wrap (flex-wrap: wrap since 1.x), so a chart row
   reflows to one column on a phone on its own — it only needs a floor so the
   charts do not squeeze into unreadable slivers first. */
@media (min-width: 421px) {
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] { min-width: 260px; }
}

/* ══ Vega-Lite / Altair chrome ══════════════════════════════════ */
/* Vega writes `style="background: white"` straight onto the SVG, which no spec
   property reliably clears — so a white plate lands under every Altair chart on
   the dark surface. */
[data-testid="stVegaLiteChart"] svg, .vega-embed svg { background: transparent !important; }

/* ══ Plotly chrome ════════════════════════════════════════ */
.js-plotly-plot .modebar { background: transparent !important; }
.js-plotly-plot .modebar-btn path { fill: var(--muted) !important; }
.js-plotly-plot .modebar-btn:hover path { fill: var(--ink) !important; }

/* ══ mobile 375-420px — a runner is a card, not a table row ════════════ */
@media (max-width: 420px) {
  .main .block-container, [data-testid="stMainBlockContainer"] {
    padding: var(--s3) var(--s3) var(--s12) !important;
  }
  /* one radial instead of two - the gutters the glow lived in are gone */
  .stApp {
    background:
      radial-gradient(120% 45% at 50% -8%, var(--glow-a), transparent 62%),
      var(--bg) !important;
  }
  .rp-kpis      { grid-template-columns: 1fr; }
  .rp-card-grid { grid-template-columns: 1fr; }
  .sg-stats     { grid-template-columns: repeat(2, 1fr); }

  .rp-body { overflow-x: visible; padding: var(--s2); }
  .rp-tbl, .rp-tbl tbody, .rp-tbl tr, .rp-tbl td { display: block; width: 100%; }
  .rp-tbl thead { position: absolute; width: 1px; height: 1px;
    overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
  .rp-tbl tr { position: relative; border: 1px solid var(--border);
    border-radius: var(--r-md); padding: var(--s3) var(--s3) var(--s2);
    margin-bottom: var(--s2); background: var(--surface); }
  .rp-tbl tr.is-value { background: var(--value-wash); border-color: var(--value-line); }
  .rp-tbl tr td, .rp-tbl tr:hover td, .rp-tbl tr.is-value:hover td { background: transparent; }
  .rp-tbl td { border: none; padding: var(--s1) 0; font-size: var(--t-data);
    display: flex; align-items: center; justify-content: space-between; gap: var(--s3); }
  .rp-tbl td::before { content: attr(data-label); color: var(--muted);
    font-size: var(--t-xs); text-transform: uppercase; letter-spacing: .06em; flex: none; }
  /* the horse name is the row heading, and the rank badge floats on it */
  .rp-tbl td[data-label="Runner"] { display: block; margin: 0 0 var(--s2) 30px; }
  .rp-tbl td[data-label="Runner"]::before { display: none; }
  .rp-tbl td[data-label="#"] { position: absolute; top: var(--s3); left: var(--s3);
    padding: 0; width: auto; }
  .rp-tbl td[data-label="#"]::before { display: none; }
  .rp-tbl td.num { text-align: left; }
  .rp-tbl tr th:nth-child(5), .rp-tbl tr td:nth-child(5) { display: flex; }
  .rp-odds + .rp-sub { display: inline; }
  .pb { min-width: 0; flex: 1; justify-content: flex-end; }
  .pb-track { max-width: none; }

  /* charts degrade rather than crowd - Python cannot see the viewport, CSS can */
  .js-plotly-plot .xtick text, .js-plotly-plot .ytick text { font-size: 10px !important; }
  .js-plotly-plot .legendtext { font-size: 10px !important; }
  .js-plotly-plot .modebar { display: none !important; }

  /* Streamlit's floating toolbar overlaps the card header at this width */
  [data-testid="stToolbar"] { display: none !important; }

  /* touch targets */
  .rp-card-hdr a.rp-open { min-height: 44px; display: inline-flex; align-items: center; }
  [data-testid="stSidebarNav"] a { min-height: 44px; display: flex; align-items: center; }
}
"""


@lru_cache(maxsize=None)
def _css(mode: str = "dark", width: str = "default") -> str:
    """The whole stylesheet for one mode. Cached — this runs on every rerun."""
    sheet = "<style>\n" + _FONT_IMPORT + _tokens_css(mode) + _BODY
    if width != "default":
        sheet += (
            "\n.main .block-container, [data-testid=\"stMainBlockContainer\"] {"
            f" max-width: {_WIDTHS[width]} !important; }}\n"
        )
    return sheet + "</style>\n"


# Back-compat: `_CSS` is the dark stylesheet, as it always was. Imported by
# `.design-md/_assets_gallery.py`.
_CSS = _css("dark")


@lru_cache(maxsize=None)
def palette(mode: str = "dark") -> dict[str, str]:
    """Resolved colour for any code path that cannot use CSS vars (Plotly,
    Altair, ReportLab). Generated from the same dict as the CSS — never a
    second copy."""
    return dict(_MODES[mode])


#: The dark palette, for back-compat with ``from ui._design import PALETTE``.
PALETTE = palette("dark")


def inject_design(mode: str = "dark", width: str = "default") -> None:
    """Inject the shared design system. Call once, right after
    ``set_page_config``, on every surface that uses it (idempotent within a run).

    ``mode`` selects the token binding; the stylesheet body is identical for
    both. ``width`` picks the shell width — ``"narrow"`` (920px) suits
    single-column form pages.
    """
    st.markdown(_css(mode, width), unsafe_allow_html=True)


# ── chart themes ─────────────────────────────────────────────────────────────
#: Chart series identity. A separate namespace from the semantic tokens — see
#: the module docstring and DESIGN.md §2.3.
def series(mode: str = "dark") -> list[str]:
    """The categorical colorway, in fixed order. Never cycle it: a sixth series
    folds into "Other" or the chart becomes small multiples."""
    p = _MODES[mode]
    return [p[f"series_{i}"] for i in range(1, 6)]


def ordinal(mode: str = "dark") -> list[str]:
    """The one-hue ordinal ramp, light-end first (odds bands, tiers)."""
    p = _MODES[mode]
    return [p[f"ord_{i}"] for i in range(1, 5)]


def tint(hex_colour: str, alpha: float) -> str:
    """``#rrggbb`` → ``rgba(r, g, b, a)``. For area fills, which the mark spec
    puts at ~10% of the series hue — a wash, never a saturated block."""
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def plotly_layout(mode: str = "dark", density: str = "comfortable",
                  **overrides) -> dict:
    """A Plotly ``layout`` dict themed to the design tokens. Spread into a
    figure: ``fig.update_layout(**plotly_layout())``.

    ``density="compact"`` tightens margins and type for charts that sit in a
    small card. It is a separate argument rather than something you pass through
    ``**overrides`` because ``dict.update`` replaces whole keys — passing
    ``xaxis=dict(...)`` would silently wipe the themed axis colours.

    The ``colorway`` is the validated categorical palette. It deliberately
    contains no green, amber or red: those are reserved semantics, and a chart
    where green means "the second thing I plotted" misleads on a surface where
    green means "this bet has edge".
    """
    p = _MODES[mode]
    compact = density == "compact"
    axis = dict(
        gridcolor=p["chart_grid"], zerolinecolor=p["border"], linecolor=p["border"],
        tickfont=dict(color=p["chart_axis"], size=10 if compact else 11),
        automargin=True,
    )
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=p["chart_ink"],
                  size=11 if compact else 13),
        xaxis=dict(axis), yaxis=dict(axis),
        colorway=series(mode),
        # titles live in the card header, not in the plot — it reads better and
        # frees the top margin on mobile
        margin=(dict(l=8, r=8, t=8, b=8) if compact
                else dict(l=52, r=20, t=16, b=40)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=p["chart_ink"], size=10 if compact else 12)),
        hoverlabel=dict(bgcolor=p["surface_2"], bordercolor=p["border"],
                        font=dict(color=p["ink"], family="Inter")),
    )
    base.update(overrides)
    return base


def altair_theme(mode: str = "dark") -> dict:
    """Colour for the Altair charts (the reliability diagram). Returned as a
    plain dict rather than registered with ``alt.themes.register``, which is
    process-global and would capture every chart in the app."""
    p = _MODES[mode]
    return {
        "series":      {"catboost": p["series_1"], "lgbm": p["series_2"],
                        "market": p["muted"]},
        "axis_label":  p["chart_axis"],
        "legend_label": p["chart_ink"],
        "grid":        p["chart_grid"],
        "reference":   p["border_strong"],
        "surface":     p["surface"],
    }
