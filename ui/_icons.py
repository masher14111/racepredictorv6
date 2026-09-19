"""MASHR’s Predictor v6 — in-house SVG icon set, illustrations & meter graphics.

Pure SVG-string builders, in the same headless / Streamlit-free spirit as
:mod:`ui._components` (callers do the single ``st.markdown(..., unsafe_allow_html
=True)``). No external image files: every mark is hand-crafted inline SVG so it

  • ships in the markup (no extra request, no static-serving dependency),
  • is **theme-driven** — strokes/fills default to ``currentColor``, so wrapping a
    mark in any element with a ``color`` (a CSS var like ``var(--info)``) recolours
    it; it follows the theme for free,
  • is unit-testable as a string.

Cohesion rules (so the set reads as one family):
  • 24×24 viewBox, line-art, ``stroke-width`` ~1.6, round caps/joins,
  • ``fill="none"`` outline by default; fills are used sparingly and deliberately
    (finish-flag checks, coin faces) at high opacity,
  • ``aria-hidden="true"`` — these are decorative; the adjacent text is the label
    (the design system's "colour/'shape never carries meaning alone" rule).

See :data:`_PATHS` for the glyph table. Illustrations (``horse_illustration``,
``finishline_scene``) and meters (``strength_meter``, ``prob_gauge``) live below
the icon set. Catalogued in ``memory/ui-26-css-graphics.md``.
"""
from __future__ import annotations

from html import escape

# ── icon glyph table ──────────────────────────────────────────────────────────
# Each value is the inner SVG markup for a 24×24 viewBox. Strokes are authored as
# ``currentColor`` so a glyph inherits whatever ``color`` its container sets.
# Keep paths line-art + round-joined so the set stays a cohesive family.
_PATHS: dict[str, str] = {
    # race — a pennant on a rail post (the meeting / racecard)
    "race": (
        '<path d="M6 21V3"/>'
        '<path d="M6 4c3.2-1.7 6.4 1.7 9.6 0 1.1-.6 2-.7 2.9-.4v6.2c-1 .4-1.9.4-3.1 0'
        '-3-1-5.9 1.5-9.4 0z"/>'
    ),
    # runner — a horse head in a bridle (the entrant)
    "runner": (
        '<path d="M5.5 20.5c-.6-2.4-.4-4.4.6-6.2.5-.9.4-1.7-.2-2.5l-1.2-1.6c-.6-.8-.4'
        '-1.9.5-2.3l1.9-.9 1 1.6c.4.6 1 1 1.8 1l3.7.1c2.1.1 3.9 1.8 3.9 3.9 0 .9-.3 '
        '1.7-.9 2.4l-2 2.3c-.5.6-.8 1.4-.8 2.2v.5"/>'
        '<path d="M8.3 5.6 7.2 3.3l2.3 1.1"/>'
        '<circle cx="14.4" cy="11.6" r=".9" fill="currentColor" stroke="none"/>'
    ),
    # value — a price tag with a spark (positive edge vs the market)
    "value": (
        '<path d="M3.6 12.4 12 4h6.4v6.4L10 18.8a2 2 0 0 1-2.8 0l-3.6-3.6a2 2 0 0 1 0-2.8z"/>'
        '<circle cx="15.1" cy="8.9" r="1.1"/>'
        '<path d="m8.4 12.6 1.1-1.1m1.6 2.6 1.6-1.6"/>'
    ),
    # bankroll — stacked coins (the practice bankroll)
    "bankroll": (
        '<ellipse cx="12" cy="6.5" rx="7" ry="2.6"/>'
        '<path d="M5 6.5v5c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-5"/>'
        '<path d="M5 11.5v5c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-5"/>'
    ),
    # calibration — A/E scatter on the diagonal (does the model's % hold up)
    "calibration": (
        '<path d="M5 4v15h15"/>'
        '<path d="M5 19 19 5" stroke-dasharray="2.4 2.4"/>'
        '<circle cx="9" cy="15.4" r="1.05" fill="currentColor" stroke="none"/>'
        '<circle cx="12.4" cy="11.2" r="1.05" fill="currentColor" stroke="none"/>'
        '<circle cx="15.8" cy="8.4" r="1.05" fill="currentColor" stroke="none"/>'
    ),
    # trophy — the cup (performance / a winning record)
    "trophy": (
        '<path d="M7 4h10v4.5a5 5 0 0 1-10 0z"/>'
        '<path d="M7 5.5H4.6A1.6 1.6 0 0 0 3 7.1 3.4 3.4 0 0 0 6.4 10.5H7"/>'
        '<path d="M17 5.5h2.4A1.6 1.6 0 0 1 21 7.1 3.4 3.4 0 0 1 17.6 10.5H17"/>'
        '<path d="M12 13.5V17"/>'
        '<path d="M8.5 20.5h7l-1-3.5h-5z"/>'
    ),
    # selection — a check-marked rosette (a flagged pick)
    "selection": (
        '<circle cx="12" cy="9" r="5"/>'
        '<path d="m9.7 9 1.6 1.6L14.4 7.4"/>'
        '<path d="m9.4 13.4-1.6 6 4.2-2.2 4.2 2.2-1.6-6"/>'
    ),
    # clock — post time
    "clock": (
        '<circle cx="12" cy="12" r="8.2"/>'
        '<path d="M12 7.6V12l3 1.8"/>'
    ),
}

_DEFAULT_STROKE = 1.6


def icon(name: str, size: int = 18, stroke_width: float = _DEFAULT_STROKE,
         cls: str = "") -> str:
    """Return one icon from :data:`_PATHS` as an inline ``<svg>`` string.

    The glyph inherits ``color`` from its container (strokes are ``currentColor``),
    so colour it by wrapping/placing it inside an element with a theme var, e.g.
    ``f'<span style="color:var(--value)">{icon("value")}</span>'``. Unknown names
    render an empty (but valid) svg rather than raising, so a template never breaks.
    """
    body = _PATHS.get(name, "")
    klass = f"rp-ico {cls}".strip()
    return (
        f'<svg class="{klass}" width="{size}" height="{size}" viewBox="0 0 24 24" '
        f'fill="none" stroke="currentColor" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" '
        f'xmlns="http://www.w3.org/2000/svg">{body}</svg>'
    )


def icon_names() -> list[str]:
    """The available icon keys (handy for the gallery / tests)."""
    return list(_PATHS)


# ── logo / favicon mark ───────────────────────────────────────────────────────

def logo_mark(size: int = 32, rounded: bool = True) -> str:
    """The app's logo lockup mark: a brand-indigo rounded tile carrying a white
    finish-flag glyph. Self-contained (its own colours, not ``currentColor``) so
    it reads as a brand mark anywhere — sidebar, favicon, an empty state.

    This is the source for ``ui/static/favicon.svg`` (written by the build script);
    keep the two in sync if you change the artwork.
    """
    s = size
    r = "7" if rounded else "0"
    return (
        f'<svg class="rp-logo" width="{s}" height="{s}" viewBox="0 0 32 32" '
        f'fill="none" aria-label="Race Predictor" xmlns="http://www.w3.org/2000/svg">'
        # brand tile with a hairline inner highlight
        f'<rect x="1" y="1" width="30" height="30" rx="{r}" '
        f'fill="var(--brand, #6d5cf0)"/>'
        f'<rect x="1.6" y="1.6" width="28.8" height="28.8" rx="{r}" fill="none" '
        f'stroke="#fff" stroke-opacity=".18"/>'
        # post + flag
        f'<path d="M11 7v18" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>'
        f'<path d="M11 8h12v8H11z" stroke="#fff" stroke-width="1.4" '
        f'stroke-linejoin="round"/>'
        # checker
        f'<path d="M11 8h3v2h-3zM17 8h3v2h-3zM14 10h3v2h-3zM11 12h3v2h-3z'
        f'M17 12h3v2h-3z" fill="#fff" opacity=".95"/>'
        f'</svg>'
    )


# ── illustrations (empty states / hero) ───────────────────────────────────────

def horse_illustration(size: int = 132) -> str:
    """A calm line-art horse-head-in-profile illustration (facing the finish line)
    for empty / hero states. A noble profile reads as "horse" far more reliably
    than a small galloping silhouette, and stays elegant at any size.

    Brand-themed: the head is an ink silhouette with a soft fill, the mane and
    finish flag pick up the brand accent, a value-dim turf line grounds it.
    Every colour is a theme token with a literal fallback, so it recolours with
    the system yet renders standalone.
    """
    s = size
    return (
        f'<svg class="rp-illus" width="{s}" height="{int(s * 0.78)}" '
        f'viewBox="0 0 200 156" fill="none" aria-hidden="true" '
        f'xmlns="http://www.w3.org/2000/svg">'
        # finish post + checker flag (brand accent), on the line ahead
        f'<g stroke="var(--brand-text, #9085e9)" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="M176 16v116"/>'
        f'<path d="M176 20h20v14h-20z"/>'
        f'</g>'
        f'<path d="M176 20h5v4.7h-5zM186 20h5v4.7h-5zM181 24.7h5v4.7h-5z'
        f'M176 29.4h5v4.6h-5zM186 29.4h5v4.6h-5z" '
        f'fill="var(--brand-text, #9085e9)" opacity=".9"/>'
        # horse head + neck in profile, facing right — one closed silhouette
        f'<path d="M58 132 V104 C58 86 60 72 70 58 C66 56 60 58 55 63 '
        'C58 52 66 46 74 45 C73 38 74 31 78 26 C81 33 82 39 82 45 '
        'C86 40 90 36 95 34 C93 41 92 46 92 50 '
        'C104 48 116 52 126 60 C134 66 142 74 150 80 '
        'C154 83 156 87 156 92 C156 96 153 99 149 99 '
        'C146 99 143 97 141 94 L138 96 C140 100 139 104 135 105 '
        'C131 106 127 104 124 100 C120 104 113 105 108 102 '
        'C106 108 104 116 104 124 V132 Z" '
        'fill="var(--surface-3, #262c33)" stroke="var(--ink, #edeff1)" '
        'stroke-width="2.2" stroke-linejoin="round"/>'
        # mane along the crest (brand accent flicks)
        f'<g stroke="var(--brand-text, #9085e9)" stroke-width="2" '
        f'stroke-linecap="round" fill="none">'
        f'<path d="M70 58c-7-1-12 2-16 8"/>'
        f'<path d="M74 70c-7 0-12 4-15 11"/>'
        f'<path d="M80 84c-7 1-12 6-14 13"/>'
        f'</g>'
        # eye + nostril + jaw line (ink details)
        f'<circle cx="118" cy="70" r="2.4" fill="var(--ink, #edeff1)"/>'
        f'<path d="M145 90c-2 .6-4 .4-5.6-.8" stroke="var(--ink, #edeff1)" '
        f'stroke-width="1.6" stroke-linecap="round"/>'
        f'<path d="M104 96c6 2 12 1 16-3" stroke="var(--ink, #edeff1)" '
        f'stroke-width="1.4" stroke-linecap="round" opacity=".5"/>'
        # ground / running rail (soft value-dim turf line)
        f'<path d="M8 138h184" stroke="var(--border-strong, #484e55)" '
        f'stroke-width="1.4" stroke-linecap="round"/>'
        f'<path d="M8 138h66" stroke="var(--value-dim, #44a264)" stroke-width="1.6" '
        f'stroke-linecap="round" opacity=".85"/>'
        f'</svg>'
    )


def finishline_scene(size: int = 120) -> str:
    """A minimal finish-line illustration (checkered ribbon + two posts) for the
    "race is run / nothing upcoming" empty state. Lighter than the horse; same
    token-driven palette."""
    s = size
    return (
        f'<svg class="rp-illus" width="{s}" height="{int(s * 0.66)}" '
        f'viewBox="0 0 180 120" fill="none" aria-hidden="true" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<g stroke="var(--border-strong, #484e55)" stroke-width="2" '
        f'stroke-linecap="round">'
        f'<path d="M30 16v92"/><path d="M150 16v92"/>'
        f'</g>'
        # checkered ribbon between the posts
        f'<rect x="30" y="20" width="120" height="20" rx="2" '
        f'fill="none" stroke="var(--ink-2, #babec3)" stroke-width="1.4"/>'
        + "".join(
            f'<rect x="{30 + i * 15}" y="{20 + (i % 2) * 10}" width="15" height="10" '
            f'fill="var(--ink-2, #babec3)" opacity=".85"/>'
            for i in range(8)
        )
        + f'<path d="M18 108h144" stroke="var(--value-dim, #44a264)" '
        f'stroke-width="1.6" stroke-linecap="round" opacity=".8"/>'
        f'</svg>'
    )


# ── probability / strength meters ─────────────────────────────────────────────

def strength_meter(value: float | None, segments: int = 5, tone: str = "info",
                   label: str | None = None) -> str:
    """A segmented strength meter (filled pips out of ``segments``) for a 0–1
    value — a compact alternative to the linear ``prob_bar`` where a coarse
    "how strong" read is wanted (confidence, field strength, edge size).

    ``tone`` selects the fill channel (info|value|amber|brand) so it stays on
    the locked colour roles; an optional mono ``label`` (e.g. "62%") sits to the
    right. Colour never carries the meaning alone — the fill *count* does, and the
    label reinforces it. Returns an honest dash when ``value`` is None.
    """
    if value is None:
        return ('<span class="rp-meter"><span class="rp-meter-val">—</span></span>')
    v = max(0.0, min(1.0, float(value)))
    filled = round(v * segments)
    pips = "".join(
        f'<i class="{"on" if i < filled else ""}"></i>' for i in range(segments)
    )
    lab = (f'<span class="rp-meter-val">{escape(label)}</span>'
           if label is not None else "")
    return (f'<span class="rp-meter tone-{escape(tone)}">'
            f'<span class="rp-meter-pips">{pips}</span>{lab}</span>')


def prob_gauge(prob: float | None, size: int = 64, tone: str = "info",
               label: str | None = None) -> str:
    """A semicircular gauge arc for a single probability (0–1). The arc sweeps
    from empty to ``prob`` of a half-turn via ``stroke-dasharray``; the track is a
    muted full arc behind it. Theme-driven (token stroke with literal fallback),
    with a centred mono value. A richer, "premium" read for the hero metric on a
    race-detail or suggestion surface where a single number deserves a dial.
    """
    track = "var(--surface-3, #262c33)"
    tone_var = {
        "info": "var(--info, #549de5)", "value": "var(--value, #5bcc80)",
        "amber": "var(--amber, #ebae51)", "brand": "var(--brand-text, #9085e9)",
        # legacy alias — call sites predating the indigo brand still pass this
        "oxblood": "var(--brand-text, #9085e9)",
    }.get(tone, "var(--info, #549de5)")
    # geometry: a 100×60 viewBox half-arc, radius 42, centre (50,52)
    r = 42.0
    import math
    length = math.pi * r  # half-circumference
    p = 0.0 if prob is None else max(0.0, min(1.0, float(prob)))
    dash = f"{length * p:.2f} {length:.2f}"
    arc = "M8 52a42 42 0 0 1 84 0"
    val = (escape(label) if label is not None
           else ("—" if prob is None else f"{p * 100:.0f}%"))
    return (
        f'<span class="rp-gauge" style="width:{size}px">'
        f'<svg width="{size}" height="{int(size * 0.66)}" viewBox="0 0 100 64" '
        f'fill="none" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="{arc}" stroke="{track}" stroke-width="9" stroke-linecap="round"/>'
        f'<path d="{arc}" stroke="{tone_var}" stroke-width="9" stroke-linecap="round" '
        f'stroke-dasharray="{dash}" pathLength="{length:.2f}"/>'
        f'</svg>'
        f'<span class="rp-gauge-val">{val}</span>'
        f'</span>'
    )
