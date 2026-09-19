import pytest
from datetime import date
from scraper.betsp.results.base import RawResult, ResultRow, ResultsSource
from scraper.betsp.extractor import PerfExtractor, StubExtractor


class _FakeSource(ResultsSource):
    name = "fake"

    def fetch_raw(self, day, get_html):
        return []

    def parse(self, raw):
        return [ResultRow(race_date="2026-05-31T16:55", venue="Nottingham",
                          horse_name="Sharp Romance", source="fake", position=1)]


def test_stub_extractor_delegates_to_source_parse():
    raw = RawResult("fake", date(2026, 5, 31), "u", "<html></html>")
    rows = StubExtractor().extract(raw, _FakeSource())
    assert len(rows) == 1 and rows[0].position == 1


def test_stub_drops_rows_without_horse_name():
    class _Bad(_FakeSource):
        def parse(self, raw):
            return [ResultRow(race_date="x", venue="y", horse_name="", source="fake")]
    raw = RawResult("fake", date(2026, 5, 31), "u", "<html></html>")
    assert StubExtractor().extract(raw, _Bad()) == []


def test_perfextractor_is_abstract():
    with pytest.raises(TypeError):
        PerfExtractor()
