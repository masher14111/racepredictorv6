import datetime
import os

from scraper import timeform_historical as th

FIX = os.path.join(os.path.dirname(__file__), "timeform", "fixtures")


def _index_html():
    return open(os.path.join(FIX, "racecards_index.html"), encoding="utf-8").read()


def _card_html():
    return open(os.path.join(FIX, "racecard_single.html"), encoding="utf-8").read()


def test_fetch_builds_dataframe_and_writes(tmp_path):
    def fake_get_html(url):
        # the index path has fewer segments than a per-race card URL
        return _index_html() if url.rstrip("/").count("/") < 6 else _card_html()

    path = str(tmp_path / "timeform.parquet")
    df = th.fetch(days=[datetime.date(2026, 6, 13)],
                  get_html=fake_get_html, parquet_path=path, max_events=1)
    assert not df.empty
    assert {"timeform_rating", "race_class", "going_speed",
            "historical_win_rate"}.issubset(df.columns)
    assert os.path.exists(path)
