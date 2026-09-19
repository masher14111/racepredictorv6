"""Decimal -> traditional fractional odds, and per-book price rows."""

import math

import pytest

from utils.odds_format import book_label, format_book_prices, to_fraction


@pytest.mark.parametrize(
    "decimal,expected",
    [
        (1.20, "1/5"),
        (1.25, "1/4"),
        (1.50, "1/2"),
        (1.80, "4/5"),
        (2.00, "Evens"),
        (2.25, "5/4"),
        (2.375, "11/8"),
        (2.50, "6/4"),     # a board shows 6/4, never the reduced 3/2
        (2.75, "7/4"),
        (3.00, "2/1"),
        (3.50, "5/2"),
        (4.00, "3/1"),
        (5.00, "4/1"),
        (6.00, "5/1"),
        (11.00, "10/1"),
        (26.00, "25/1"),
        (34.00, "33/1"),
    ],
)
def test_ladder_prices_render_exactly(decimal, expected):
    assert to_fraction(decimal) == expected


@pytest.mark.parametrize(
    "decimal,expected",
    [
        (2.26, "5/4"),   # scraped noise snaps to the nearest board price
        (2.24, "5/4"),
        (2.49, "6/4"),
        (3.02, "2/1"),
    ],
)
def test_off_ladder_prices_snap_to_nearest(decimal, expected):
    assert to_fraction(decimal) == expected


@pytest.mark.parametrize("bad", [None, 1.0, 0.0, -3.0, float("nan"), "abc", ""])
def test_unusable_prices_render_as_dash(bad):
    assert to_fraction(bad) == "—"


def test_beyond_the_board_reduces_instead_of_clamping():
    out = to_fraction(2000.0)
    assert out.endswith("/1")
    assert int(out.split("/")[0]) > 1000


def test_book_labels():
    assert book_label("paddy_power") == "Paddy"
    assert book_label("boylesports") == "Boyle"
    assert book_label("livescorebet") == "LSB"
    assert book_label("some_new_book") == "Some New Book"
    assert book_label(None) == "—"


# ---------------------------------------------------------------------------
# format_book_prices
# ---------------------------------------------------------------------------
def test_rows_are_best_price_first_and_flagged():
    rows = format_book_prices({"paddy_power": 1.5, "boylesports": 2.5})
    assert [r["label"] for r in rows] == ["Boyle", "Paddy"]
    assert [r["fraction"] for r in rows] == ["6/4", "1/2"]
    assert rows[0]["is_best"] is True
    assert rows[1]["is_best"] is False


def test_tied_best_price_flags_every_book():
    rows = format_book_prices({"paddy_power": 3.0, "boylesports": 3.0})
    assert all(r["is_best"] for r in rows)


def test_explicit_order_is_respected():
    rows = format_book_prices(
        {"paddy_power": 1.5, "boylesports": 2.5},
        order=["paddy_power", "boylesports"],
    )
    assert [r["label"] for r in rows] == ["Paddy", "Boyle"]
    # Ordering is presentational; the best price is still marked correctly.
    assert rows[1]["is_best"] is True


def test_unusable_entries_are_dropped():
    rows = format_book_prices(
        {"paddy_power": 2.5, "boylesports": None, "livescorebet": 1.0,
         "betfair": float("nan"), "other": "x"}
    )
    assert [r["label"] for r in rows] == ["Paddy"]


@pytest.mark.parametrize("empty", [None, {}, {"a": None}])
def test_no_prices_gives_no_rows(empty):
    assert format_book_prices(empty) == []


def test_decimal_is_preserved_for_downstream_maths():
    rows = format_book_prices({"paddy_power": 2.5})
    assert math.isclose(rows[0]["decimal"], 2.5)
