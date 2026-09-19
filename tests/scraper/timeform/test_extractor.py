import os

from scraper.timeform.extractor import StubExtractor

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "racecard_single.html")


def test_stub_extractor_delegates_to_parser():
    html = open(FIX, encoding="utf-8").read()
    rows = StubExtractor().extract(html, venue="York", going="good")
    assert rows and rows[0].horse_name
