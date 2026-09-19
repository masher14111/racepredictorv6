import pandas as pd

from utils.market_validation import (
    INVALID,
    VALID,
    annotate_primary_win_rows,
    looks_like_special_selection,
    validate_live_dataframe,
    validate_primary_win_rows,
)


def _rows():
    return [
        {
            "source": "livescorebet",
            "race_id": "r1",
            "market_type": "WIN",
            "market_id": "m1",
            "market_name": "To win",
            "selection_id": "s1",
            "horse_name": "Alpha",
            "odds_decimal": 2.5,
        },
        {
            "source": "livescorebet",
            "race_id": "r1",
            "market_type": "WIN",
            "market_id": "m1",
            "market_name": "To win",
            "selection_id": "s2",
            "horse_name": "Bravo",
            "odds_decimal": 3.5,
        },
        {
            "source": "livescorebet",
            "race_id": "r1",
            "market_type": "WIN",
            "market_id": "m1",
            "market_name": "To win",
            "selection_id": "s3",
            "horse_name": "Charlie",
            "odds_decimal": 4.0,
        },
    ]


def test_valid_primary_win_race():
    result = validate_primary_win_rows(_rows(), primary_market_count=1)
    assert result.status == VALID
    assert result.reasons == ()
    assert result.field_size == 3
    assert 0.8 < result.booksum < 1.8


def test_duplicate_selection_fails_entire_race():
    rows = _rows()
    rows[1]["selection_id"] = "s1"
    result = validate_primary_win_rows(rows, primary_market_count=1)
    assert result.status == INVALID
    assert "duplicate_selection_id" in result.reasons


def test_implausible_field_size_fails_entire_race():
    result = validate_primary_win_rows(_rows() * 14, primary_market_count=1)
    assert result.status == INVALID
    assert any(reason.startswith("implausible_field_size:") for reason in result.reasons)


def test_extreme_booksum_fails_entire_race():
    rows = _rows()
    for row in rows:
        row["odds_decimal"] = 1.1
    result = validate_primary_win_rows(rows, primary_market_count=1)
    assert result.status == INVALID
    assert any(reason.startswith("extreme_booksum:") for reason in result.reasons)


def test_special_inside_confirmed_primary_market_is_secondary_fail_closed_guard():
    rows = _rows()
    rows[1]["horse_name"] = "Alpha & Bravo Both To Finish In The Top 3"
    result = validate_primary_win_rows(rows, primary_market_count=1)
    assert result.status == INVALID
    assert "special_selection_in_primary_win" in result.reasons


def test_winning_distance_selections_are_flagged_as_specials():
    for name in (
        "Alpha by 2 Lengths or more",
        "Bravo by 1-2 lengths",
        "Charlie by 10+ Lengths",
        "Delta by 2.5 lengths",
    ):
        assert looks_like_special_selection(name), name


def test_ordinary_horse_names_are_not_flagged_as_specials():
    """The regex is a secondary guard; false positives would fail clean races."""
    for name in (
        "Alpha", "Lengthsman", "Bravo Charlie", "Without A Doubt",
        "Top Of The Class", "Finish Line", "Two Lengths Clear",
    ):
        assert not looks_like_special_selection(name), name


def test_multiple_primary_win_markets_fail_entire_race():
    rows = _rows()
    rows[2]["market_id"] = "m2"
    result = validate_primary_win_rows(rows, primary_market_count=2)
    assert result.status == INVALID
    assert "primary_win_market_count:2" in result.reasons


def test_dataframe_validator_drops_every_row_in_contaminated_race():
    valid = annotate_primary_win_rows(_rows(), primary_market_count=1)
    invalid_rows = _rows()
    invalid_rows[1]["horse_name"] = "Betting Without Alpha - Bravo To Win"
    invalid = annotate_primary_win_rows(invalid_rows, primary_market_count=1)
    for row in invalid:
        row["race_id"] = "r2"
    out = validate_live_dataframe(pd.DataFrame(valid + invalid))
    assert set(out["race_id"]) == {"r1"}
    assert set(out["validation_status"]) == {VALID}
