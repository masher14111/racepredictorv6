"""Firecrawl tier-5 fetcher.

The behaviours worth locking down are the ones that cost money or corrupt odds:
the per-run credit budget, maxAge=0 (never serve cached prices), the IE
geolocation, and treating an upstream challenge as a failure rather than data.
"""

import json

import pytest

from scraper import firecrawl_fetch as fc

_CFG = {
    "enabled": True,
    "max_pages_per_run": 3,
    "workers": 2,
    "proxy": "auto",
    "country": "IE",
    "language": "en-GB",
    "timeout_ms": 60000,
    "wait_for_ms": 0,
}


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Fresh budget + a stub key/config for every test."""
    fc.reset_budget()
    monkeypatch.setattr(fc, "_cfg", lambda: dict(_CFG))
    monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
    yield
    fc.reset_budget()


def _stub_post(monkeypatch, payload, status=200, sink=None):
    def _post(url, json=None, headers=None, timeout=None):  # noqa: A002
        if sink is not None:
            sink.append({"url": url, "json": json, "headers": headers})
        return _Resp(payload, status)

    monkeypatch.setattr(fc.httpx, "post", _post)


def _ok(**data):
    return {"success": True, "data": {"metadata": {"statusCode": 200}, **data}}


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------
def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(fc, "_cfg", lambda: {**_CFG, "api_key": ""})
    ok, why = fc.is_available()
    assert not ok and "FIRECRAWL_API_KEY" in why


def test_unavailable_when_disabled(monkeypatch):
    monkeypatch.setattr(fc, "_cfg", lambda: {**_CFG, "enabled": False})
    ok, why = fc.is_available()
    assert not ok and "disabled" in why


def test_available_with_key():
    ok, why = fc.is_available()
    assert ok and why == "ready"


def test_key_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    monkeypatch.setattr(fc, "_cfg", lambda: {**_CFG, "api_key": "fc-from-config"})
    assert fc.api_key() == "fc-from-config"


# ---------------------------------------------------------------------------
# Credit budget — the guard that stops a bad day becoming an expensive one
# ---------------------------------------------------------------------------
def test_budget_caps_pages_and_then_raises(monkeypatch):
    _stub_post(monkeypatch, _ok(html="<html>card</html>"))
    for _ in range(3):  # max_pages_per_run == 3
        fc.get_html("https://www.boylesports.com/x")
    assert fc.pages_used() == 3

    with pytest.raises(fc.FirecrawlError) as exc:
        fc.get_html("https://www.boylesports.com/x")
    assert exc.value.reason == "budget_exhausted"


def test_budget_zero_disables_tier(monkeypatch):
    monkeypatch.setattr(fc, "_cfg", lambda: {**_CFG, "max_pages_per_run": 0})
    ok, why = fc.is_available()
    assert not ok and "budget" in why


def test_failed_fetch_still_spends_its_page(monkeypatch):
    """A billed call that returns a challenge must still count against the cap,
    otherwise a persistently-blocked book would bill in an unbounded loop."""
    _stub_post(monkeypatch, _ok(html="<html/>"), status=500)
    with pytest.raises(fc.FirecrawlError):
        fc.get_html("https://www.boylesports.com/x")
    assert fc.pages_used() == 1


# ---------------------------------------------------------------------------
# Request shape — the two settings that silently corrupt odds if wrong
# ---------------------------------------------------------------------------
def test_request_never_accepts_cached_content(monkeypatch):
    sink = []
    _stub_post(monkeypatch, _ok(html="<html/>"), sink=sink)
    fc.get_html("https://www.boylesports.com/x")
    assert sink[0]["json"]["maxAge"] == 0, "cached odds would be silently stale"


def test_request_is_geolocated_to_ireland(monkeypatch):
    sink = []
    _stub_post(monkeypatch, _ok(html="<html/>"), sink=sink)
    fc.get_html("https://www.boylesports.com/x")
    assert sink[0]["json"]["location"]["country"] == "IE"
    assert sink[0]["headers"]["Authorization"] == "Bearer fc-test-key"


def test_wait_for_only_sent_when_configured(monkeypatch):
    sink = []
    _stub_post(monkeypatch, _ok(html="<html/>"), sink=sink)
    fc.get_html("https://x.test/a")
    assert "waitFor" not in sink[0]["json"]

    fc.reset_budget()
    monkeypatch.setattr(fc, "_cfg", lambda: {**_CFG, "wait_for_ms": 2500})
    sink.clear()
    fc.get_html("https://x.test/a")
    assert sink[0]["json"]["waitFor"] == 2500


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def test_get_html_returns_markup(monkeypatch):
    _stub_post(monkeypatch, _ok(html="<html>race card</html>"))
    assert fc.get_html("https://x.test/a") == "<html>race card</html>"


def test_empty_html_is_an_error(monkeypatch):
    _stub_post(monkeypatch, _ok(html="   "))
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.get_html("https://x.test/a")
    assert exc.value.reason == "empty_card"


# ---------------------------------------------------------------------------
# JSON (Paddy Power's endpoint, fetched through a browser)
# ---------------------------------------------------------------------------
def test_get_json_parses_bare_body(monkeypatch):
    _stub_post(monkeypatch, _ok(rawHtml=json.dumps({"races": [1, 2]})))
    assert fc.get_json("https://apisms.paddypower.com/x") == {"races": [1, 2]}


def test_get_json_unwraps_browser_pre_viewer(monkeypatch):
    body = '<html><body><pre>{"races": [{"id": 7}]}</pre></body></html>'
    _stub_post(monkeypatch, _ok(rawHtml=body))
    assert fc.get_json("https://apisms.paddypower.com/x") == {"races": [{"id": 7}]}


def test_get_json_unescapes_entities(monkeypatch):
    body = '<pre>{&quot;name&quot;: &quot;Bally &amp; Co&quot;}</pre>'
    _stub_post(monkeypatch, _ok(rawHtml=body))
    assert fc.get_json("https://apisms.paddypower.com/x") == {"name": "Bally & Co"}


def test_get_json_rejects_a_challenge_page(monkeypatch):
    _stub_post(monkeypatch, _ok(rawHtml="<html><body>Just a moment...</body></html>"))
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.get_json("https://apisms.paddypower.com/x")
    assert exc.value.reason == "invalid_json"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "status,reason",
    [
        (402, "payment_required"),
        (429, "rate_limited"),
        (401, "not_configured"),
        (403, "not_configured"),
        (500, "other"),
    ],
)
def test_http_status_maps_to_reason(monkeypatch, status, reason):
    _stub_post(monkeypatch, {"success": False}, status=status)
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.get_html("https://x.test/a")
    assert exc.value.reason == reason


def test_upstream_block_is_bot_detected(monkeypatch):
    """Firecrawl fetched fine, but the book served a 403 — that is not data."""
    _stub_post(
        monkeypatch,
        {"success": True, "data": {"html": "<html/>", "metadata": {"statusCode": 403}}},
    )
    with pytest.raises(fc.FirecrawlError) as exc:
        fc.get_html("https://www.boylesports.com/x")
    assert exc.value.reason == "bot_detected"


def test_unsuccessful_body_is_an_error(monkeypatch):
    _stub_post(monkeypatch, {"success": False, "error": "nope"})
    with pytest.raises(fc.FirecrawlError):
        fc.get_html("https://x.test/a")
