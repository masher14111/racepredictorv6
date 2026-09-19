import gzip
from datetime import date
from scraper.betsp.results.base import RawResult
from scraper.betsp import raw_store


def test_write_and_path(tmp_path):
    raw = RawResult(source="sporting_life", race_date=date(2026, 5, 31),
                    url="http://x/results", html="<html>hi</html>")
    p = raw_store.write(raw, root=str(tmp_path))
    assert p.endswith(".html.gz")
    assert "sporting_life" in p and "2026" in p and "2026-05-31" in p
    with gzip.open(p, "rt", encoding="utf-8") as f:
        assert f.read() == "<html>hi</html>"
