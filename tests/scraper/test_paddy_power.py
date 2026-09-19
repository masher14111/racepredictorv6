import json
import os
from datetime import datetime, timedelta, timezone

import pytest
import json
from copy import deepcopy
from pathlib import Path
from scraper.paddy_power import (
    BotDetectedError, ScraperError, SCRAPE_INTERVAL,
    CacheManager, PaddyPowerClient,
    ChromeFallback, _parse_pp_response, _parse_race_time, scrape,
    _curl_cffi_fetch as _real_curl_cffi_fetch,
)
from utils.proxy_manager import ProxyManager as ProxyRotator


@pytest.fixture(autouse=True)
def _curl_cffi_fails_by_default(monkeypatch):
    """`_curl_cffi_fetch` is now the primary scrape tier. Default it to fail so
    existing scrape() tests exercise their mocked httpx/fallback tiers and none
    accidentally hit the live network. A test that wants the curl_cffi path just
    re-patches `scraper.paddy_power._curl_cffi_fetch` in its own body (the later
    patch wins)."""
    def _raise(*args, **kwargs):
        raise BotDetectedError("curl_cffi disabled in tests")
    monkeypatch.setattr("scraper.paddy_power._curl_cffi_fetch", _raise)


def test_bot_detected_error_is_exception():
    with pytest.raises(BotDetectedError):
        raise BotDetectedError("test")


def test_scraper_error_is_exception():
    with pytest.raises(ScraperError):
        raise ScraperError("test")


def test_scrape_interval_is_int():
    assert isinstance(SCRAPE_INTERVAL, int)
    assert SCRAPE_INTERVAL > 0


def _write_cache(path: str, age_seconds: int, data: dict = None) -> None:
    fetched_at = (
        datetime.now(tz=timezone.utc) - timedelta(seconds=age_seconds)
    ).isoformat()
    payload = {"fetched_at": fetched_at, "races": [], **(data or {})}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f)


def test_cache_miss_when_file_absent(tmp_path):
    cm = CacheManager(path=str(tmp_path / "pp.json"), ttl=3600)
    assert cm.is_fresh() is False
    assert cm.read() is None


def test_cache_fresh_within_ttl(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=100)
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is True


def test_cache_stale_beyond_ttl(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=4000)
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is False


def test_cache_read_returns_data(tmp_path):
    path = str(tmp_path / "pp.json")
    _write_cache(path, age_seconds=10, data={"races": [{"race_id": "1"}]})
    cm = CacheManager(path=path, ttl=3600)
    result = cm.read()
    assert result["races"][0]["race_id"] == "1"


def test_cache_write_creates_directories_and_file(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "pp.json")
    cm = CacheManager(path=path, ttl=3600)
    data = {"fetched_at": "2026-06-12T12:00:00+01:00", "races": []}
    cm.write(data)
    assert os.path.exists(path)
    # Read back via CacheManager to remain format-agnostic
    assert cm.read()["races"] == []


def test_cache_stale_on_corrupt_json(tmp_path):
    path = str(tmp_path / "pp.json")
    with open(path, "w") as f:
        f.write("not json")
    cm = CacheManager(path=path, ttl=3600)
    assert cm.is_fresh() is False
    assert cm.read() is None


def test_proxy_rotator_disabled_returns_none():
    r = ProxyRotator(cfg={"enabled": False, "proxies": ["http://p1:8080"]})
    assert r.next() is None


def test_proxy_rotator_empty_list_returns_none():
    r = ProxyRotator(cfg={"enabled": True, "proxies": []})
    assert r.next() is None


def test_proxy_rotator_single_proxy():
    r = ProxyRotator(cfg={"enabled": True, "proxies": ["http://p1:8080"]})
    assert r.next() == "http://p1:8080"
    assert r.next() == "http://p1:8080"


def test_proxy_rotator_cycles():
    proxies = ["http://p1:8080", "http://p2:8080", "http://p3:8080"]
    r = ProxyRotator(cfg={"enabled": True, "proxies": proxies})
    results = [r.next() for _ in range(5)]
    assert results == [
        "http://p1:8080",
        "http://p2:8080",
        "http://p3:8080",
        "http://p1:8080",
        "http://p2:8080",
    ]


import respx
import httpx as _httpx


@respx.mock
def test_client_successful_get():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(200, json={"events": []})
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    result = client.get("https://api.example.com/data")
    assert result == {"events": []}


@respx.mock
def test_client_raises_bot_detected_on_403():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(403)
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    with pytest.raises(BotDetectedError):
        client.get("https://api.example.com/data")


@respx.mock
def test_client_raises_bot_detected_on_html_body():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(
            200,
            content=b"<html>Cloudflare</html>",
            headers={"content-type": "text/html"},
        )
    )
    client = PaddyPowerClient(proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}))
    with pytest.raises(BotDetectedError):
        client.get("https://api.example.com/data")


@respx.mock
def test_client_retries_on_500_then_succeeds():
    route = respx.get("https://api.example.com/data")
    route.side_effect = [
        _httpx.Response(500),
        _httpx.Response(200, json={"attachments": {}}),
    ]
    client = PaddyPowerClient(
        proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}),
        max_retries=3,
    )
    import scraper.paddy_power as pp_module
    original_sleep = pp_module.time.sleep
    pp_module.time.sleep = lambda x: None
    try:
        result = client.get("https://api.example.com/data")
        assert result == {"attachments": {}}
    finally:
        pp_module.time.sleep = original_sleep


@respx.mock
def test_client_raises_scraper_error_after_max_retries():
    respx.get("https://api.example.com/data").mock(
        return_value=_httpx.Response(500)
    )
    client = PaddyPowerClient(
        proxy_rotator=ProxyRotator(cfg={"enabled": False, "proxies": []}),
        max_retries=2,
    )
    import scraper.paddy_power as pp_module
    original_sleep = pp_module.time.sleep
    pp_module.time.sleep = lambda x: None
    try:
        with pytest.raises(ScraperError):
            client.get("https://api.example.com/data")
    finally:
        pp_module.time.sleep = original_sleep


# ── proxy-required, gateway unavailable — never falls back to direct (req 2) ──

def test_client_get_proxy_unavailable_raises_typed_error_not_direct(mocker):
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("all proxies blacklisted")
    mock_client_cls = mocker.patch("httpx.Client")

    client = PaddyPowerClient(proxy_rotator=mock_rotator, max_retries=3)
    with pytest.raises(ScraperError) as exc_info:
        client.get("https://api.example.com/data")

    assert exc_info.value.reason == "proxy_unavailable"
    mock_rotator.next.assert_called_once_with(required=True)
    mock_client_cls.assert_not_called()


def test_client_get_proxy_unavailable_records_source_health(mocker):
    from scraper.paddy_power import _SOURCE
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("no proxies configured")
    record_unavailable = mocker.patch("scraper.paddy_power.source_health.record_unavailable")

    client = PaddyPowerClient(proxy_rotator=mock_rotator, max_retries=3)
    with pytest.raises(ScraperError):
        client.get("https://api.example.com/data")

    record_unavailable.assert_called_once_with(_SOURCE, reason="proxy_unavailable")


def test_curl_cffi_fetch_proxy_unavailable_raises_typed_error_not_direct(mocker):
    from scraper.paddy_power import _PP_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("all proxies blacklisted")
    mocker.patch("scraper.paddy_power.get_proxy_manager", return_value=mock_rotator)
    mock_get = mocker.patch("curl_cffi.requests.get")

    with pytest.raises(ScraperError) as exc_info:
        _real_curl_cffi_fetch(_PP_URL)

    assert exc_info.value.reason == "proxy_unavailable"
    mock_rotator.next.assert_called_once_with(required=True)
    mock_get.assert_not_called()


def test_curl_cffi_fetch_proxy_unavailable_records_source_health(mocker):
    from scraper.paddy_power import _SOURCE, _PP_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("no proxies configured")
    mocker.patch("scraper.paddy_power.get_proxy_manager", return_value=mock_rotator)
    mocker.patch("curl_cffi.requests.get")
    record_unavailable = mocker.patch("scraper.paddy_power.source_health.record_unavailable")

    with pytest.raises(ScraperError):
        _real_curl_cffi_fetch(_PP_URL)

    record_unavailable.assert_called_once_with(_SOURCE, reason="proxy_unavailable")


_SAMPLE_PP_RESPONSE = {
    "layout": {},
    "attachments": {
        "races": {
            "35700000.1400": {
                "raceId": "35700000.1400",
                "venue": "Ascot",
                "startTime": "2026-06-12T14:00:00.000Z",
                "meetingId": 35700000,
            }
        },
        "markets": {
            "927.000001": {
                "marketId": "927.000001",
                "raceId": "35700000.1400",
                "marketType": "WIN",
                "marketTime": "2026-06-12T14:00:00.000Z",
                "eachwayAvailable": True,
                "numberOfPlaces": 3,
                "placeFraction": {"numerator": 1, "denominator": 4},
                "runners": [
                    {
                        "runnerName": "Mighty Oak",
                        "runnerStatus": "ACTIVE",
                        "winRunnerOdds": {
                            "trueOdds": {
                                "decimalOdds": {"decimalOdds": 6.5}
                            }
                        },
                    }
                ],
            }
        },
    },
}

_FIXTURES = Path(__file__).parent / "fixtures"


def test_adversarial_pp_fixture_keeps_only_primary_win_runners():
    payload = json.loads(
        (_FIXTURES / "paddy_power_adversarial.json").read_text(encoding="utf-8")
    )
    races = _parse_pp_response(payload)

    assert len(races) == 1
    assert len(races[0]["markets"]) == 1
    market = races[0]["markets"][0]
    assert market["market_id"] == "primary-win"
    assert {s["horse_name"] for s in market["selections"]} == {
        "Alpha", "Bravo", "Charlie",
    }
    assert market["validation_status"] == "VALID"


def test_multiple_pp_win_markets_are_invalid():
    payload = json.loads(
        (_FIXTURES / "paddy_power_adversarial.json").read_text(encoding="utf-8")
    )
    duplicate = deepcopy(payload["attachments"]["markets"]["primary-win"])
    duplicate["marketId"] = "primary-win-2"
    for index, selection in enumerate(duplicate["runners"], start=1):
        selection["selectionId"] = f"duplicate-{index}"
    payload["attachments"]["markets"]["primary-win-2"] = duplicate

    races = _parse_pp_response(payload)
    assert races[0]["validation_status"] == "INVALID"
    assert all(
        market["validation_status"] == "INVALID"
        for market in races[0]["markets"]
    )


def test_parse_race_time_converts_to_dublin():
    # 14:00 UTC = 15:00 Europe/Dublin (BST, UTC+1 in June)
    result = _parse_race_time("2026-06-12T14:00:00Z")
    assert "15:00:00" in result
    assert "+01:00" in result


def test_parse_pp_response_structure():
    races = _parse_pp_response(_SAMPLE_PP_RESPONSE)
    assert len(races) == 1
    race = races[0]
    assert race["race_id"] == "35700000.1400"
    assert race["venue"] == "Ascot"
    assert "15:00:00" in race["race_time"]


def test_parse_pp_response_market_fields():
    races = _parse_pp_response(_SAMPLE_PP_RESPONSE)
    market = races[0]["markets"][0]
    assert market["market_type"] == "WIN"
    assert market["each_way_terms"] == {"places": 3, "reduction": 0.25}


def test_parse_pp_response_selection_fields():
    races = _parse_pp_response(_SAMPLE_PP_RESPONSE)
    sel = races[0]["markets"][0]["selections"][0]
    assert sel["horse_name"] == "Mighty Oak"
    assert sel["odds_decimal"] == 6.5
    assert sel["sp"] is None


def test_parse_pp_response_skips_removed_runners():
    data = {
        "attachments": {
            "races": {"35700001.1500": {"venue": "York", "raceId": "35700001.1500"}},
            "markets": {
                "927.000002": {
                    "raceId": "35700001.1500",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T15:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [
                        {"runnerName": "Gone", "runnerStatus": "REMOVED"},
                        {
                            "runnerName": "Runner",
                            "runnerStatus": "ACTIVE",
                            "winRunnerOdds": {
                                "trueOdds": {"decimalOdds": {"decimalOdds": 3.0}}
                            },
                        },
                    ],
                }
            },
        }
    }
    races = _parse_pp_response(data)
    sels = races[0]["markets"][0]["selections"]
    assert len(sels) == 1
    assert sels[0]["horse_name"] == "Runner"


def test_parse_pp_response_null_each_way_when_unavailable():
    data = {
        "attachments": {
            "races": {"35700001.1500": {"venue": "York", "raceId": "35700001.1500"}},
            "markets": {
                "927.000002": {
                    "raceId": "35700001.1500",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T15:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    races = _parse_pp_response(data)
    assert races[0]["markets"][0]["each_way_terms"] is None


def test_parse_pp_response_empty_on_no_markets():
    races = _parse_pp_response({"attachments": {"races": {}, "markets": {}}})
    assert races == []


def test_parse_pp_response_tolerates_missing_market_time():
    # In-play / SP-only markets can omit marketTime — must not crash the parse.
    data = {
        "attachments": {
            "races": {"r1": {"raceId": "r1", "venue": "Ascot"}},
            "markets": {
                "m1": {
                    "raceId": "r1",
                    "marketType": "WIN",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    races = _parse_pp_response(data)
    assert len(races) == 1
    assert races[0]["race_time"] == ""


def test_parse_race_time_empty_returns_empty():
    assert _parse_race_time("") == ""
    assert _parse_race_time("not-a-date") == ""


from unittest.mock import MagicMock


def _make_mock_response(url: str, body: dict):
    resp = MagicMock()
    resp.url = url
    resp.json.return_value = body
    return resp


def test_chrome_fallback_captures_response(mocker):
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser

    captured_handlers = []

    def fake_on(event, handler):
        if event == "response":
            captured_handlers.append(handler)

    mock_page.on.side_effect = fake_on

    def fake_goto(url):
        resp = _make_mock_response(
            "https://apisms.paddypower.com/smspp/content-managed-page/v7?cardsToFetch=21149",
            _SAMPLE_PP_RESPONSE,
        )
        for handler in captured_handlers:
            handler(resp)

    mock_page.goto.side_effect = fake_goto

    cm = MagicMock()
    cm.__enter__.return_value = mock_p
    mocker.patch("scraper.paddy_power.sync_playwright", return_value=cm)

    fallback = ChromeFallback()
    result = fallback.fetch()
    assert "attachments" in result
    assert result["attachments"]["races"]["35700000.1400"]["venue"] == "Ascot"


def test_chrome_fallback_raises_if_no_response_captured(mocker):
    mock_browser = MagicMock()
    mock_page = MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_p = MagicMock()
    mock_p.chromium.launch.return_value = mock_browser
    mock_page.on.return_value = None

    cm = MagicMock()
    cm.__enter__.return_value = mock_p
    mocker.patch("scraper.paddy_power.sync_playwright", return_value=cm)

    fallback = ChromeFallback()
    with pytest.raises(ScraperError, match="no API responses"):
        fallback.fetch()


def _fresh_cache_data():
    return {
        "fetched_at": datetime.now(tz=timezone.utc).isoformat(),
        "races": [{"race_id": "cached"}],
    }


def test_scrape_returns_cached_data_when_fresh(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    data = _fresh_cache_data()
    with open(cache_path, "w") as f:
        json.dump(data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    result = scrape()
    assert result["races"][0]["race_id"] == "cached"


def test_scrape_fetches_when_cache_stale(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    stale_data = {
        "fetched_at": (
            datetime.now(tz=timezone.utc) - timedelta(seconds=5000)
        ).isoformat(),
        "races": [],
    }
    with open(cache_path, "w") as f:
        json.dump(stale_data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.return_value = {
        "attachments": {
            "races": {
                "35700042.1400": {
                    "raceId": "35700042.1400",
                    "venue": "York",
                }
            },
            "markets": {
                "927.042": {
                    "raceId": "35700042.1400",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T14:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)
    mocker.patch("scraper.paddy_power.ChromeFallback")

    result = scrape()
    assert result["races"][0]["race_id"] == "35700042.1400"


def test_scrape_triggers_chrome_fallback_on_bot_detection(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = BotDetectedError("403")
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)

    mock_fallback = MagicMock()
    mock_fallback.fetch.return_value = {
        "attachments": {
            "races": {
                "35700099.1400": {"raceId": "35700099.1400", "venue": "Epsom"}
            },
            "markets": {
                "927.099": {
                    "raceId": "35700099.1400",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T14:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    mocker.patch("scraper.paddy_power.ChromeFallback", return_value=mock_fallback)

    result = scrape()
    assert result["races"][0]["race_id"] == "35700099.1400"
    assert os.path.exists(cache_path)


def test_scrape_raises_scraper_error_when_both_paths_fail(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.side_effect = BotDetectedError("403")
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)

    mock_fallback = MagicMock()
    mock_fallback.fetch.side_effect = ScraperError("Chrome also failed")
    mocker.patch("scraper.paddy_power.ChromeFallback", return_value=mock_fallback)
    mocker.patch(
        "scraper.paddy_power._selenium_fetch",
        side_effect=ScraperError("Selenium also failed"),
    )

    with pytest.raises(ScraperError):
        scrape()


def test_scrape_force_bypasses_fresh_cache(tmp_path, mocker):
    cache_path = str(tmp_path / "pp.json")
    data = _fresh_cache_data()
    with open(cache_path, "w") as f:
        json.dump(data, f)

    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    mock_client = MagicMock()
    mock_client.get.return_value = {
        "attachments": {
            "races": {
                "35700007.1400": {"raceId": "35700007.1400", "venue": "Naas"}
            },
            "markets": {
                "927.007": {
                    "raceId": "35700007.1400",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T14:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)
    mocker.patch("scraper.paddy_power.ChromeFallback")

    result = scrape(force=True)
    assert result["races"][0]["race_id"] == "35700007.1400"


def test_pp_has_selenium_fallback_symbol():
    import scraper.paddy_power as pp
    assert hasattr(pp, "_selenium_fetch")


def test_scrape_uses_curl_cffi_tier_first(tmp_path, mocker):
    """curl_cffi (Chrome impersonation) is the primary tier: on success the
    legacy httpx client is never reached."""
    cache_path = str(tmp_path / "pp.json")
    mocker.patch("scraper.paddy_power._CACHE_PATH", cache_path)
    mocker.patch("scraper.paddy_power.SCRAPE_INTERVAL", 3600)

    raw = {
        "attachments": {
            "races": {"35700007.1400": {"raceId": "35700007.1400", "venue": "Naas"}},
            "markets": {
                "927.007": {
                    "raceId": "35700007.1400",
                    "marketType": "WIN",
                    "marketTime": "2026-06-12T14:00:00.000Z",
                    "eachwayAvailable": False,
                    "runners": [],
                }
            },
        }
    }
    # Override the autouse default so curl_cffi "succeeds".
    mocker.patch("scraper.paddy_power._curl_cffi_fetch", return_value=raw)
    mock_client = MagicMock()
    mocker.patch("scraper.paddy_power.PaddyPowerClient", return_value=mock_client)

    result = scrape(force=True)
    assert result["races"][0]["race_id"] == "35700007.1400"
    mock_client.get.assert_not_called()  # httpx tier skipped entirely
