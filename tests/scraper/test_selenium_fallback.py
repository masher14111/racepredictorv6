import pytest


def test_rows_from_logs_uses_parser_on_matching_url():
    from scraper._selenium_fallback import _rows_from_bodies

    bodies = [
        ("https://x/api/event/1", {"events": [{"id": 1}]}),
        ("https://x/static/app.js", {"junk": True}),
    ]
    rows = _rows_from_bodies(
        bodies,
        xhr_url_predicate=lambda u: "/api/event" in u,
        parse_xhr=lambda body: [{"id": body["events"][0]["id"]}],
    )
    assert rows == [{"id": 1}]


def test_rows_from_logs_skips_parser_errors():
    from scraper._selenium_fallback import _rows_from_bodies

    def boom(body):
        raise KeyError("missing")

    bodies = [("https://x/api/event/1", {})]
    rows = _rows_from_bodies(
        bodies, xhr_url_predicate=lambda u: True, parse_xhr=boom
    )
    assert rows == []


def test_fetch_raises_when_no_rows(monkeypatch):
    from scraper._selenium_fallback import SeleniumFallback, ScraperError

    fb = SeleniumFallback(
        url="https://x/horse-racing",
        xhr_url_predicate=lambda u: True,
        parse_xhr=lambda b: [],
    )
    monkeypatch.setattr(fb, "_capture_bodies", lambda: [])
    monkeypatch.setattr(fb, "_dom_rows", lambda: [])
    with pytest.raises(ScraperError):
        fb.fetch()
