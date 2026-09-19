"""Today-facing landing-page charts (``ui._today``).

Row builders are pure, so they test headless. The figure builders are checked
for structure and for the colour contract in DESIGN.md, not for pixels.
"""
from __future__ import annotations

import plotly.graph_objects as go
import pytest

from ui import _today as T
from ui._design import PALETTE


@pytest.fixture
def card() -> list[dict]:
    """Two races. Race 1 carries a full `runners` field that is DELIBERATELY
    wider than its `selections` top-3, including a mid-field value bet — the
    case the display list hides. Race 2 is an older-shape cache with no
    `runners` at all."""
    return [
        {
            "venue": "Dundalk",
            "race_time": "2026-09-18T19:30:00+01:00",
            "selections": [
                {"horse_name": "Pavilion End", "best_odds": 3.75, "implied_prob": 0.267,
                 "won_prob_normalized": 0.208, "expected_value": -0.26,
                 "value_bet": False, "best_book": "paddy_power"},
                {"horse_name": "Nakasero", "best_odds": 5.0, "implied_prob": 0.20,
                 "won_prob_normalized": 0.19, "expected_value": -0.05,
                 "value_bet": False, "best_book": "boylesports"},
            ],
            "runners": [
                {"horse_name": "Pavilion End", "best_odds": 3.75, "implied_prob": 0.267,
                 "won_prob_normalized": 0.208, "expected_value": -0.26,
                 "value_bet": False, "best_book": "paddy_power"},
                {"horse_name": "Nakasero", "best_odds": 5.0, "implied_prob": 0.20,
                 "won_prob_normalized": 0.19, "expected_value": -0.05,
                 "value_bet": False, "best_book": "boylesports"},
                # mid-field, outside the display top-3, and the only value bet
                {"horse_name": "Deep Field", "best_odds": 12.0, "implied_prob": 0.083,
                 "won_prob_normalized": 0.15, "expected_value": 0.42,
                 "value_bet": True, "best_book": "paddy_power"},
            ],
        },
        {
            "venue": "Ayr",
            "race_time": "2026-09-18T20:00:00+01:00",
            "selections": [
                {"horse_name": "Old Cache", "best_odds": 2.5, "implied_prob": 0.40,
                 "won_prob_normalized": 0.35, "expected_value": -0.12,
                 "value_bet": False, "best_book": "livescorebet"},
            ],
        },
    ]


# ── the full field, not the display list ──────────────────────────────────────

def test_charts_read_the_full_field_not_the_display_top_three():
    """audit req 2/7: `selections` is the top-3 display list. Counting over it
    makes a mid-field value bet invisible — which is exactly the bug the landing
    page's KPI rail already guards against."""
    race = {
        "selections": [{"horse_name": "A"}, {"horse_name": "B"}],
        "runners": [{"horse_name": "A"}, {"horse_name": "B"}, {"horse_name": "C"}],
    }
    assert [r["horse_name"] for r in T.field(race)] == ["A", "B", "C"]


def test_field_falls_back_for_caches_written_before_runners_existed():
    race = {"selections": [{"horse_name": "A"}], "excluded_low_odds": [{"horse_name": "B"}]}
    assert [r["horse_name"] for r in T.field(race)] == ["A", "B"]
    assert T.field({}) == []


def test_value_bet_outside_the_top_three_is_still_counted(card):
    rows = {r["band"]: r for r in T.odds_band_rows(card)}
    # Deep Field sits at 12.0 — the E/W band — and is only in `runners`
    assert rows["E/W 8–16"]["n_value"] == 1, "a mid-field value bet was dropped"


# ── where is the value? ───────────────────────────────────────────────────────

def test_odds_band_rows_bucket_on_the_price_we_would_actually_take(card):
    rows = {r["band"]: r for r in T.odds_band_rows(card)}
    assert rows["Fav 2–4"]["n"] == 2          # 3.75 and 2.5
    assert rows["Mid 4–8"]["n"] == 1          # 5.0
    assert rows["E/W 8–16"]["n"] == 1         # 12.0
    assert "Odds-on" not in rows              # empty bands are omitted, not zeroed


def test_odds_band_rows_skip_runners_with_no_usable_price():
    card = [{"runners": [
        {"best_odds": None, "expected_value": 0.1},
        {"best_odds": 1.0, "expected_value": 0.1},   # <=1.0 is not a price
        {"best_odds": "n/a", "expected_value": 0.1},
        {"best_odds": 3.0, "expected_value": 0.2},
    ]}]
    rows = T.odds_band_rows(card)
    assert len(rows) == 1 and rows[0]["n"] == 1


def test_ev_chart_uses_the_diverging_pair_never_the_semantic_trio(card):
    fig = T.ev_by_odds_band_fig(T.odds_band_rows(card))
    assert isinstance(fig, go.Figure)
    colours = set(fig.data[0].marker.color)
    assert colours <= {PALETTE["div_pos"], PALETTE["div_neg"]}
    # green/amber/red are reserved: they must never colour a chart series here
    for reserved in ("value", "amber", "danger"):
        assert PALETTE[reserved] not in colours


# ── model vs market ───────────────────────────────────────────────────────────

def test_model_vs_market_rows_pair_each_runner_with_its_price(card):
    rows = T.model_vs_market_rows(card)
    assert len(rows) == 4                      # 3 from race 1 + 1 from race 2
    deep = next(r for r in rows if r["horse"] == "Deep Field")
    assert deep["is_value"] is True and deep["venue"] == "Dundalk"
    assert deep["model"] > deep["market"], "a value bet sits above the diagonal"


def test_model_vs_market_marks_value_by_shape_as_well_as_colour(card):
    """Colour never carries meaning alone — the value split also changes the
    marker symbol and appears in the legend."""
    fig = T.model_vs_market_fig(T.model_vs_market_rows(card))
    marked = [t for t in fig.data if t.mode == "markers"]
    assert len(marked) == 2
    assert {t.marker.symbol for t in marked} == {"circle", "diamond"}
    assert all(t.showlegend is not False for t in marked)
    value_trace = next(t for t in marked if t.marker.symbol == "diamond")
    # `value_bet` literally MEANS positive EV, so the reserved green is correct
    assert value_trace.marker.color == PALETTE["value"]


# ── who is pricing best? ──────────────────────────────────────────────────────

def test_best_book_rows_rank_by_count_and_carry_a_share(card):
    rows = T.best_book_rows(card)
    assert [r["book"] for r in rows][0] == "Paddy"      # 2 of 4
    assert rows[0]["n"] == 2
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-9


def test_best_book_bars_share_one_hue(card):
    """Nominal identity between peers, and bar length already carries the value
    — so colour must not re-encode the count."""
    fig = T.best_book_fig(T.best_book_rows(card))
    assert fig.data[0].marker.color == PALETTE["series_1"]


# ── empty states are calm, never crashes ──────────────────────────────────────

@pytest.mark.parametrize("builder,rows", [
    (T.ev_by_odds_band_fig, []),
    (T.model_vs_market_fig, []),
    (T.best_book_fig, []),
])
def test_empty_states_render_a_figure_with_a_message(builder, rows):
    fig = builder(rows)
    assert isinstance(fig, go.Figure)
    assert fig.layout.annotations and fig.layout.annotations[0].text


@pytest.mark.parametrize("races", [None, [], [{}], [{"runners": []}]])
def test_builders_survive_an_empty_or_malformed_card(races):
    assert T.odds_band_rows(races) == []
    assert T.model_vs_market_rows(races) == []
    assert T.best_book_rows(races) == []


# ── today's winners: did the picks land? ──────────────────────────────────────

@pytest.fixture
def results():
    import pandas as pd
    return pd.DataFrame([
        # the drift that actually occurs between feeds: a country suffix,
        # casing/whitespace on the venue, and seconds on the race time.
        # (Checked against unified_races.parquet: 39,333 distinct horse names,
        # zero of them hyphenated — so hyphen drift is NOT a real case and
        # norm_horse is right not to handle it.)
        {"venue": "Dundalk", "race_time": "2026-09-18T19:30:17+01:00",
         "horse_name": "Pavilion End (IRE)", "position": 1, "sp": 4.0},
        {"venue": "dundalk ", "race_time": "2026-09-18T19:30:00+01:00",
         "horse_name": "Nakasero", "position": 5, "sp": 6.0},
        {"venue": "Dundalk", "race_time": "2026-09-18T19:30:00+01:00",
         "horse_name": "Deep Field", "position": 3, "sp": 11.0},
    ])


@pytest.fixture
def ranked_card():
    return [{
        "venue": "Dundalk", "race_time": "2026-09-18T19:30:00+01:00",
        "runners": [
            {"horse_name": "Pavilion End", "rank": 1, "best_odds": 3.75},
            {"horse_name": "Nakasero", "rank": 2, "best_odds": 5.0},
            {"horse_name": "Deep Field", "rank": 3, "best_odds": 12.0,
             "each_way_value": True},
        ],
    }]


def test_winners_join_survives_real_feed_drift(results, ranked_card):
    """The odds feeds and the results feed disagree on country suffixes, venue
    casing/whitespace and seconds — the join normalises all three or it
    silently returns nothing at all."""
    rows = T.winners_rows(results, ranked_card)
    assert len(rows) == 3
    top = next(r for r in rows if r["rank"] == 1)
    assert top["horse"] == "Pavilion End" and top["position"] == 1
    assert top["won"] is True and top["placed"] is True


def test_winners_rows_mark_place_but_not_win(results, ranked_card):
    third = next(r for r in T.winners_rows(results, ranked_card) if r["rank"] == 3)
    assert third["position"] == 3
    assert third["won"] is False and third["placed"] is True
    assert third["each_way"] is True


def test_winners_summary_counts_top_picks_with_its_denominator(results, ranked_card):
    s = T.winners_summary(T.winners_rows(results, ranked_card))
    assert s == {"races": 1, "picks": 1, "wins": 1, "places": 1}


def test_no_results_yet_is_empty_not_a_zero_score(ranked_card):
    """A card that has not run must return NOTHING, so the caller can say 'not
    yet' — returning zero-scored rows would read as 'every pick lost'."""
    import pandas as pd
    unrun = pd.DataFrame([{"venue": "Dundalk",
                           "race_time": "2026-09-18T19:30:00+01:00",
                           "horse_name": "Pavilion End",
                           "position": None, "sp": None}])
    assert T.winners_rows(unrun, ranked_card) == []
    assert T.last_results_day(unrun) is None


def test_last_results_day_finds_the_newest_finished_day(results):
    import datetime as dt
    assert T.last_results_day(results) == dt.date(2026, 9, 18)


@pytest.mark.parametrize("bad", [None, "not a frame"])
def test_winners_rows_shrug_off_a_missing_results_frame(bad, ranked_card):
    assert T.winners_rows(bad, ranked_card) == []
