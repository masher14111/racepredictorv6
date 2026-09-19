"""Integration: scraper.spotlight archives scraped commentary (Stage-14 scope 3)."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from llm.text_archive import TextArchive
from scraper import spotlight


@pytest.fixture
def archive(tmp_path):
    a = TextArchive(db_path=str(tmp_path / "text.db"))
    yield a
    a.close()


def test_scrape_archives_every_commentary_row(tmp_path, archive):
    df = pd.DataFrame(
        {
            "race_date": ["2026-06-17T14:30"],
            "race_time": ["14:30"],
            "venue": ["Ascot"],
            "horse_name": ["Little Lady Karen"],
            "horse_id": ["12345"],
            "cloth_number": [4],
            "commentary": ["Needs better ground, may find this too sharp."],
            "race_verdict": ["Open race"],
            "source": ["sporting_life"],
            "fetched_at": [pd.Timestamp("2026-06-17T08:00:00Z")],
        }
    )
    n = spotlight._archive_rows(df, archive_store=archive)
    assert n == 1

    history = archive.history("ascot|2026-06-17T14:30", "Little Lady Karen")
    assert len(history) == 1
    assert history[0].text == "Needs better ground, may find this too sharp."
    assert history[0].source == "sporting_life_spotlight"
    assert history[0].published_at is None  # Sporting Life gives no comment publish time


def test_scrape_writes_parquet_and_archive(monkeypatch, tmp_path, archive):
    """End-to-end ``scrape()`` with a stubbed detail fetch: both the parquet
    "latest view" and the immutable archive get the same comment."""
    index_html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"meetings":[{"meeting_summary":{"course":'
        '{"country":{"short_name":"ENG"}}},"races":[{"race_summary_reference":'
        '{"id":"999"},"course_name":"Ascot","time":"14:30","going":"Soft",'
        '"name":"Test Stakes","verdict":"open race"}]}]}}}</script>'
        '<a href="/racing/racecards/2026-06-17/ascot/racecard/999/test-stakes">x</a>'
    )
    detail_html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        '{"props":{"pageProps":{"race":{"race_summary":{"course_name":"Ascot",'
        '"time":"14:30","date":"2026-06-17"},"rides":[{"horse":{"name":'
        '"Little Lady Karen","horse_reference":{"id":"12345"}},'
        '"cloth_number":4,"commentary":"Needs better ground."}]}}}}</script>'
    )

    def _fake_get_html(url):
        return index_html if "racecard/" not in url else detail_html

    parquet_path = str(tmp_path / "spotlight.parquet")
    df = spotlight.scrape(
        date(2026, 6, 17), get_html=_fake_get_html, parquet_path=parquet_path,
        archive_store=archive,
    )
    assert len(df) == 1
    from execution.race_facts import race_facts_key

    race_uid = race_facts_key(df.iloc[0]["venue"], df.iloc[0]["race_date"])
    history = archive.history(race_uid, "Little Lady Karen")
    assert len(history) == 1
    assert history[0].text == "Needs better ground."


def test_archive_failure_for_one_row_does_not_abort_batch(archive, monkeypatch):
    df = pd.DataFrame(
        {
            "race_date": ["2026-06-17T14:30", "2026-06-17T15:00"],
            "race_time": ["14:30", "15:00"],
            "venue": ["Ascot", ""],  # second row has no venue -> empty race_uid -> skipped
            "horse_name": ["Little Lady Karen", "Another Horse"],
            "horse_id": ["12345", "67890"],
            "cloth_number": [4, 2],
            "commentary": ["Needs better ground.", "Course winner last time."],
            "race_verdict": ["", ""],
            "source": ["sporting_life", "sporting_life"],
            "fetched_at": [pd.Timestamp("2026-06-17T08:00:00Z")] * 2,
        }
    )
    n = spotlight._archive_rows(df, archive_store=archive)
    assert n == 1  # the well-formed row still archives despite the bad one
