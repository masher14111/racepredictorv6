import pytest
from scraper.betsp.results.base import ResultRow, ResultsSource


def test_resultrow_defaults_nullable_fields():
    row = ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                    horse_name="Sharp Romance", source="sporting_life")
    assert row.jockey_id is None and row.trainer_id is None
    assert row.position is None and row.going is None


def test_resultssource_is_abstract():
    with pytest.raises(TypeError):
        ResultsSource()
