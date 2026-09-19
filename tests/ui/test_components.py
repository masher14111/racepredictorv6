"""Tests for the Prompt-19 pure component helpers (ui/_components.py).

These build HTML strings and derive display state with no Streamlit dependency,
so they're exercised headless. The visual layer is verified by screenshot; this
locks the data → markup contract (slugs, confidence derivation, value/EV signals,
the honest SHAP fallback)."""
from __future__ import annotations

from ui import _components as C


# ── race slug / lookup ────────────────────────────────────────────────────────

def test_race_slug_is_stable_and_url_safe():
    race = {"venue": "Bordeaux Le Bouscat", "race_time": "2026-06-15T10:06:00+01:00"}
    slug = C.race_slug(race)
    assert slug == "bordeaux-le-bouscat-20260615T1006"
    assert all(c.isalnum() or c == "-" for c in slug)


def test_find_race_roundtrips_via_slug():
    races = [
        {"venue": "Ascot", "race_time": "2026-06-15T14:00:00+01:00"},
        {"venue": "Naas", "race_time": "2026-06-15T15:30:00+01:00"},
    ]
    target = races[1]
    assert C.find_race(races, C.race_slug(target)) is target
    assert C.find_race(races, "does-not-exist") is None


# ── confidence derivation ─────────────────────────────────────────────────────

def test_confidence_prefers_explicit_field():
    assert C.confidence_for({"confidence": "high"}) == "high"
    assert C.confidence_for({"confidence": "medium"}) == "med"


def test_confidence_falls_back_to_proxy_and_never_fabricates():
    # no signal at all → no chip
    assert C.confidence_for({}) is None
    # complete data + strong prob → high
    assert C.confidence_for({"data_completeness": 0.9, "won_prob": 0.3}) == "high"
    # sparse data, first-timer → low
    assert C.confidence_for(
        {"data_completeness": 0.2, "won_prob": 0.05, "first_time_runner": True}
    ) == "low"


def test_confidence_chip_empty_when_no_level():
    assert C.confidence_chip(None) == ""
    assert "Medium" in C.confidence_chip("med")


# ── headline win prob (calib-fl-01) ───────────────────────────────────────────

def test_headline_win_prob_prefers_normalized():
    """The headline shown to the user is the field-coherent normalized prob, NOT
    the raw calibrated marginal (which saturates badly out-of-sample)."""
    sel = {"won_prob": 0.73, "won_prob_normalized": 0.21}
    assert C.headline_win_prob(sel) == 0.21


def test_headline_win_prob_falls_back_to_marginal():
    """Single-runner rows / pre-normalization caches carry no normalized value →
    fall back to the raw marginal so the cell is never blank."""
    assert C.headline_win_prob({"won_prob": 0.18}) == 0.18
    assert C.headline_win_prob({}) is None


def test_headline_win_prob_zero_normalized_is_used_not_skipped():
    """A genuine 0.0 normalized prob is returned, not treated as missing (guards
    the `is not None` check against a falsy-`or` bug)."""
    assert C.headline_win_prob({"won_prob": 0.4, "won_prob_normalized": 0.0}) == 0.0


# ── value / EV signals pair colour with words ─────────────────────────────────

def test_value_badge_carries_the_word_value():
    html = C.value_badge(0.12)
    assert "VALUE" in html and "+12pp" in html


def test_ev_text_marks_positive_value_bets():
    assert "ev-pos" in C.ev_text(0.2, is_value=True)
    assert "ev-flat" in C.ev_text(-0.5, is_value=False)
    assert C.ev_text(None, False).endswith("—</span>")


# ── rank badge: oxblood rank-1, neutral otherwise (no rainbow) ────────────────

def test_rank_badge_only_rank_one_is_branded():
    assert "rk1" in C.rank_badge(1)
    assert "rk-n" in C.rank_badge(2)
    assert "rk-n" in C.rank_badge(7)
    assert "—" in C.rank_badge(None)


# ── probability bar clamps and labels ─────────────────────────────────────────

def test_prob_bar_clamps_and_shows_numeric_label():
    # width rides on the `--pbw` custom property so the fill can animate from 0
    # to target on mount; the numeric label is always present (never colour-only).
    assert "--pbw:91%" in C.prob_bar(0.91, "win")
    assert ">91%</span>" in C.prob_bar(0.91, "win")
    assert "--pbw:100%" in C.prob_bar(1.5, "win")   # clamped
    assert "—" in C.prob_bar(None)


# ── book chips: best price marked with star, not colour alone ─────────────────

def test_book_chips_star_marks_best():
    html = C.book_chips({"boylesports": 2.5, "paddy_power": 2.3}, "boylesports", "€")
    assert "★" in html and "bk-best" in html
    assert C.book_chips(None, None, "€") == ""


def test_book_chips_show_fractions_not_currency():
    """A price is a ratio. Rendering it as "€2.50" was a category error."""
    html = C.book_chips({"boylesports": 2.5, "paddy_power": 3.0}, "boylesports")
    assert "6/4" in html and "2/1" in html
    assert "€" not in html and "£" not in html
    # the decimal is kept, but only as a tooltip
    assert 'title="Boyle 2.50 decimal"' in html


def test_price_html_has_no_currency():
    html = C.price_html(4.0)
    assert "3/1" in html
    assert "€" not in html and "£" not in html
    assert C.fmt_price(2.0) == "Evens"
    assert C.price_html(None) == '<span class="rp-sub">—</span>'


def test_price_cell_is_a_board_price_not_a_decimal():
    """The dataframe-cell form of a price. Same contract as price_html — a board
    fraction, no currency symbol — because the bet ledger stores a decimal but a
    price is still a ratio."""
    assert C.price_cell(3.75) == "11/4 · 3.75"
    assert C.price_cell(2.0) == "Evens · 2.00"
    assert "€" not in C.price_cell(3.75) and "£" not in C.price_cell(3.75)


def test_price_cell_survives_every_empty_shape_a_dataframe_can_hold():
    """A pandas NaN is truthy AND fails `<= 1.0`, so it slips past to_fraction's
    own guard — it needs an explicit check or the ledger renders garbage."""
    import math
    for empty in (None, float("nan"), math.nan, "", "not a price", 1.0, 0.5):
        assert C.price_cell(empty) == "—", f"{empty!r} should render as a dash"


# ── SHAP drivers: honest fallback, diverging bars ─────────────────────────────

def test_why_drivers_honest_fallback_when_unavailable():
    assert "unavailable" in C.why_drivers(None).lower()
    assert "No strong drivers" in C.why_drivers({"top_positive": [], "top_negative": []})


def test_why_drivers_renders_diverging_bars_scaled_to_peak():
    exp = {
        "top_positive": [{"label": "Speed", "value": 3.0, "contribution": 0.4}],
        "top_negative": [{"label": "Layoff", "value": 40, "contribution": -0.2}],
    }
    html = C.why_drivers(exp)
    assert "Speed" in html and "Layoff" in html
    # peak (0.4) fills the half-track; the −0.2 bar is half of that. Width rides
    # on `--bw` so the diverging bars sweep out from the centre on mount.
    assert 'class="pos" style="--bw:50%"' in html
    assert 'class="neg" style="--bw:25%"' in html


# ── driver provenance note (honest about live vs history) ─────────────────────

def test_driver_source_note_is_honest_about_provenance():
    assert "today" in C.driver_source_note("live").lower()
    assert "most recent" in C.driver_source_note("history").lower()
    assert C.driver_source_note(None) == ""


# ── recent-form table (horse detail) ──────────────────────────────────────────

def test_form_table_renders_rows_with_position_chips():
    runs = [
        {"date": "2025-10-22", "venue": "Ascot", "position": 1, "field_size": 8,
         "going": "Good", "race_class": "Class 3", "distance": "1m",
         "speed": 95.0, "sp": 3.5, "won": True, "placed": True},
        {"date": "2025-04-09", "venue": "Kempton", "position": 7, "field_size": 13,
         "going": "Soft", "race_class": None, "distance": "7f",
         "speed": None, "sp": None, "won": False, "placed": False},
    ]
    html = C.form_table(runs, "£")
    assert "<table" in html and "Ascot" in html and "Kempton" in html
    assert "fp1" in html       # the win → oxblood finish chip
    assert "fp-n" in html      # the 7th → neutral chip
    assert "£3.50" in html     # SP formatted with currency
    assert "—" in html         # missing speed/SP render as em-dash, not fabricated


def test_form_table_empty_is_honest():
    assert "No prior runs" in C.form_table([])


def test_pos_badge_place_is_amber_tier():
    runs = [{"date": "2025-01-01", "position": 2, "field_size": 6, "placed": True}]
    assert "fpp" in C.form_table(runs)


# ── stat grid drops empty values ──────────────────────────────────────────────

def test_stat_grid_drops_empty_pairs():
    html = C.stat_grid([("Jockey", "J One"), ("Trainer", "—"), ("Win%", "")])
    assert "J One" in html and "Jockey" in html
    assert "Trainer" not in html      # dropped (— value)
    assert C.stat_grid([]).startswith("<p") or "No connection" in C.stat_grid([])


def test_fmt_rate_and_speed():
    assert C.fmt_rate(0.123) == "12%"
    assert C.fmt_rate(None) == "—"
    assert C.fmt_rate(1.5) == "100%"   # clamped
    assert C.fmt_speed(91.6) == "92"
    assert C.fmt_speed(None) == "—"


# ── model's view + context chips ──────────────────────────────────────────────

def test_model_view_renders_lead_and_value_line():
    html = C.model_view({"top_name": "Fav", "top_win_pct": "40% win",
                         "shape": "competitive", "n_value": 2, "note": "watch it"})
    assert "Fav" in html and "40% win" in html and "competitive" in html
    assert "2" in html and "value bet" in html
    assert "watch it" in html


def test_context_chips_skip_falsy():
    html = C.context_chips([("Going", "Soft"), ("Draw", ""), ("Pace", None)])
    assert "Going" in html and "Soft" in html
    assert "Draw" not in html and "Pace" not in html


# ── loading skeletons (shown while inference / value scan runs) ────────────────

def test_skeleton_rail_matches_kpi_count_and_is_bounded():
    assert C.skeleton_rail(4).count("rp-skel-card") == 4
    assert C.skeleton_rail(99).count("rp-skel-card") == 6   # clamped to the rail max
    assert C.skeleton_rail(0).count("rp-skel-card") == 1    # at least one
    assert 'class="rp-skel-rail"' in C.skeleton_rail()


def test_skeleton_card_has_header_and_rows():
    html = C.skeleton_card()
    assert "rp-skel-hdr" in html and "rp-skel-body" in html
    assert html.count("sk pill chip") >= 4   # one rank + one per runner row


def test_loading_skeleton_labels_the_wait_and_assembles_parts():
    html = C.loading_skeleton("Scanning today's races…", n_cards=3, rail=5)
    assert "Scanning today&#x27;s races…" in html   # label escaped
    assert "rp-skel-cap" in html and "rp-loader" in html   # captioned running mark
    assert html.count("rp-skel-rail") == 1
    assert html.count("rp-skel-card") == 5 + 3   # 5 rail tiles + 3 body cards
