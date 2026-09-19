"""Tests for the in-house SVG asset builders (ui/_icons.py).

These are pure string builders (no Streamlit), so they're exercised headless. The
visual layer is verified by screenshot (.design-md/assets-gallery.png); this locks
the contract that matters for theming + honesty: marks are valid SVG, glyphs use
``currentColor`` so they inherit the theme, meters carry the read in the fill
*count* (not colour alone) and degrade to an honest dash on missing data."""
from __future__ import annotations

from ui import _icons as I


# ── icon set ──────────────────────────────────────────────────────────────────

def test_every_named_icon_renders_valid_currentcolor_svg():
    names = I.icon_names()
    assert {"race", "runner", "value", "bankroll", "calibration", "trophy"} <= set(names)
    for name in names:
        svg = I.icon(name)
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert 'viewBox="0 0 24 24"' in svg
        # theme-driven: strokes inherit the container colour
        assert 'stroke="currentColor"' in svg
        assert 'aria-hidden="true"' in svg   # decorative; text is the label


def test_unknown_icon_is_safe_empty_svg_not_an_error():
    svg = I.icon("does-not-exist")
    assert svg.startswith("<svg") and "viewBox" in svg   # valid, just empty body


def test_icon_size_and_class_are_applied():
    svg = I.icon("trophy", size=40, cls="lead")
    assert 'width="40"' in svg and 'height="40"' in svg
    assert 'class="rp-ico lead"' in svg


# ── logo / favicon mark ───────────────────────────────────────────────────────

def test_logo_mark_is_self_contained_brand_mark():
    svg = I.logo_mark(48)
    assert 'width="48"' in svg and 'aria-label="Race Predictor"' in svg
    # carries its own brand colour (token with literal fallback) — not
    # currentColor, so it reads as a brand mark anywhere (favicon, sidebar)
    assert "var(--brand, #6d5cf0)" in svg
    assert "currentColor" not in svg


# ── illustrations ─────────────────────────────────────────────────────────────

def test_illustrations_use_theme_tokens_with_literal_fallbacks():
    for svg in (I.horse_illustration(), I.finishline_scene()):
        assert svg.startswith("<svg") and svg.endswith("</svg>")
        assert "var(--ink-2, #babec3)" in svg or "var(--ink, #edeff1)" in svg
        assert "var(--brand-text, #9085e9)" in svg or "var(--value-dim, #44a264)" in svg


# ── strength meter: fill count carries the read, honest on None ───────────────

def test_strength_meter_fills_pip_count_for_value():
    html = I.strength_meter(0.6, segments=5, tone="value", label="60%")
    assert html.count("<i") == 5
    assert html.count('class="on"') == 3            # round(0.6*5) = 3 filled
    assert "tone-value" in html and ">60%<" in html


def test_strength_meter_clamps_and_is_honest_on_none():
    assert I.strength_meter(1.5).count('class="on"') == 5   # clamped to full
    assert I.strength_meter(-0.2).count('class="on"') == 0  # clamped to empty
    dash = I.strength_meter(None)
    assert "—" in dash and "rp-meter-pips" not in dash      # no fabricated pips


# ── probability gauge: arc dasharray scales with prob, honest on None ─────────

def test_prob_gauge_dasharray_scales_and_labels():
    full = I.prob_gauge(1.0)
    empty = I.prob_gauge(0.0)
    assert "stroke-dasharray" in full and "100%" in full
    assert "0%" in empty
    # the filled length of a full gauge exceeds that of an empty one
    import re
    def filled(svg):
        return float(re.search(r'stroke-dasharray="([\d.]+)', svg).group(1))
    assert filled(full) > filled(empty)


def test_prob_gauge_none_is_dash_not_zero():
    assert I.prob_gauge(None).count("—") == 1
