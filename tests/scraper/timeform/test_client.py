import httpx
import pytest
import respx

from scraper.timeform.client import TimeformClient, TimeformError


def _cfg(**kw):
    base = {"session_cookie": "", "request_delay": 0}
    base.update(kw)
    return base


@respx.mock
def test_get_html_returns_body_on_200():
    respx.get("https://x.test/card").mock(
        return_value=httpx.Response(200, html="<html>ok rp-horse-row</html>"))
    c = TimeformClient(_cfg(), browser_fetch=None, selenium_fetch=None)
    assert "rp-horse-row" in c.get_html("https://x.test/card")


@respx.mock
def test_waf_challenge_raises_when_no_browser_tier():
    respx.get("https://x.test/r").mock(
        return_value=httpx.Response(200, html="<title>Azure WAF</title>"))
    c = TimeformClient(_cfg(), browser_fetch=None, selenium_fetch=None)
    with pytest.raises(TimeformError):
        c.get_html("https://x.test/r")


@respx.mock
def test_403_falls_through_to_browser_tier():
    respx.get("https://x.test/r").mock(return_value=httpx.Response(403))
    c = TimeformClient(_cfg(), browser_fetch=lambda u: "<html>rp-horse-row via browser</html>",
                       selenium_fetch=None)
    assert "via browser" in c.get_html("https://x.test/r")


@respx.mock
def test_session_cookie_is_sent():
    captured = {}

    def _capture(request):
        captured["cookie"] = request.headers.get("cookie")
        return httpx.Response(200, html="<html>rp-horse-row</html>")

    respx.get("https://x.test/c").mock(side_effect=_capture)
    c = TimeformClient(_cfg(session_cookie="sess=abc"), browser_fetch=None, selenium_fetch=None)
    c.get_html("https://x.test/c")
    assert captured["cookie"] == "sess=abc"
