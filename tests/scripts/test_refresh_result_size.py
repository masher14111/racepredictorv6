"""Scraper return sizes for the refresh console line.

Regression: Paddy Power returns a dict, so a bare len() counted its two KEYS and
printed "2 rows/selections" on every run no matter what it scraped — which made
the healthiest source look dead.
"""

import pandas as pd
import pytest

from scripts.refresh import _result_size


def _pp(n_races: int, per_race: int) -> dict:
    return {
        "fetched_at": "2026-09-18T17:43:54+01:00",
        "races": [
            {
                "venue": f"Track {i}",
                "markets": [
                    {
                        "market_type": "WIN",
                        "selections": [{"horse_name": f"H{j}"} for j in range(per_race)],
                    }
                ],
            }
            for i in range(n_races)
        ],
    }


def test_paddy_power_dict_counts_selections_not_keys():
    result = _pp(n_races=22, per_race=10)
    assert len(result) == 2          # the old, wrong answer
    assert _result_size(result) == 220


def test_realistic_card_is_not_reported_as_two():
    """The 2026-09-18 run: 45 races, ~10 runners each, printed as "2"."""
    assert _result_size(_pp(45, 10)) == 450


def test_dataframe_sources_count_rows():
    assert _result_size(pd.DataFrame({"a": [1, 2, 3]})) == 3
    assert _result_size(pd.DataFrame()) == 0


def test_empty_and_missing_shapes():
    assert _result_size(None) == 0
    assert _result_size({}) == 0
    assert _result_size({"fetched_at": "x", "races": []}) == 0
    assert _result_size({"races": [{"venue": "X"}]}) == 0          # no markets key
    assert _result_size({"races": [{"markets": []}]}) == 0
    assert _result_size({"races": [{"markets": [{"selections": None}]}]}) == 0


def test_multiple_markets_per_race_all_count():
    result = {
        "races": [
            {"markets": [
                {"selections": [{}, {}]},
                {"selections": [{}]},
            ]}
        ]
    }
    assert _result_size(result) == 3


@pytest.mark.parametrize("weird", [object(), 42])
def test_unsized_objects_do_not_raise(weird):
    assert _result_size(weird) == 0
