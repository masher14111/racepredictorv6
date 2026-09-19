from datetime import date

from scraper.timeform import pages


def test_index_url_for_date():
    assert pages.index_url(date(2026, 6, 13)) == \
        "https://www.timeform.com/horse-racing/racecards/2026-06-13"


def test_index_url_today_has_no_date_suffix():
    assert pages.index_url(None) == \
        "https://www.timeform.com/horse-racing/racecards"


def test_meeting_goings_from_real_index():
    import os
    fix = os.path.join(os.path.dirname(__file__), "fixtures", "racecards_index.html")
    goings = pages.meeting_goings(open(fix, encoding="utf-8").read())
    # keys are normalized venue slugs; base going parsed out of "GoingGood (...)"
    assert goings["york"] == "good"
    assert goings["chester"] == "soft"
    assert goings["goodwood"] == "soft"


def test_event_links_extracts_and_absolutizes():
    html = (
        '<a href="/horse-racing/racecards/york/2026-06-13/1350/62/1/x">A</a>'
        '<a href="/horse-racing/racecards/meeting-summary/chester/2026-06-12/12">skip</a>'
        '<a href="/horse-racing/racecards/bath/2026-06-13/1320/4/1/y">B</a>'
    )
    links = pages.event_links(html)
    assert links == [
        "https://www.timeform.com/horse-racing/racecards/york/2026-06-13/1350/62/1/x",
        "https://www.timeform.com/horse-racing/racecards/bath/2026-06-13/1320/4/1/y",
    ]
