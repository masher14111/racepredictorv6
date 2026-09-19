import pytest
import time as _time
import json
from copy import deepcopy
from pathlib import Path


def test_imports():
    pass


# ── country filtering (UK & Ireland only) ──────────────────────────────────────

def _racinghome_tree():
    """racinghome shape: Sport → Country → Competition → events (2026-06-19 live)."""
    return {
        "meetingsToday": [{
            "name": "Horse Racing", "type": "Sport",
            "childs": {
                "0": {"name": "UK & Ireland", "type": "Country", "childs": {
                    "0": {"name": "Market Rasen", "type": "Competition",
                          "events": [{"id": "UK_1"}, {"id": "UK_2"}]},
                    "1": {"name": "Limerick", "type": "Competition",
                          "events": [{"id": "IRE_1"}]},
                }},
                "1": {"name": "United States", "type": "Country",
                      "countryCode": "USA", "childs": {
                          "0": {"name": "Churchill Downs", "type": "Competition",
                                "events": [{"id": "US_1"}]},
                      }},
                "2": {"name": "France", "type": "Country", "countryCode": "FRA",
                      "childs": {"0": {"name": "Chantilly", "type": "Competition",
                                       "events": [{"id": "FR_1"}]}}},
            },
        }]
    }


def test_extract_event_ids_keeps_only_uk_ire(monkeypatch):
    import scraper.livescorebet as lsb
    monkeypatch.setattr(lsb, "_LSB_COUNTRIES", {"uk & ireland"})
    ids = lsb._extract_event_ids(_racinghome_tree())
    assert ids == ["UK_1", "UK_2", "IRE_1"]
    assert not any(i.startswith(("US", "FR")) for i in ids)


def test_extract_event_ids_empty_filter_keeps_all(monkeypatch):
    import scraper.livescorebet as lsb
    monkeypatch.setattr(lsb, "_LSB_COUNTRIES", set())
    ids = lsb._extract_event_ids(_racinghome_tree())
    assert set(ids) == {"UK_1", "UK_2", "IRE_1", "US_1", "FR_1"}


def test_extract_event_ids_untyped_tree_unfiltered(monkeypatch):
    """A tree with no `type: Country` layer (older/minimal shape) keeps every event."""
    import scraper.livescorebet as lsb
    monkeypatch.setattr(lsb, "_LSB_COUNTRIES", {"uk & ireland"})
    tree = {"meetingsToday": [{"events": [{"id": "X"}], "childs": {}}]}
    assert lsb._extract_event_ids(tree) == ["X"]


def test_throttler_first_call_is_immediate():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=1.0)
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed < 0.1  # first call never waits


def test_throttler_second_call_waits():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=1.0)
    t.acquire()
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed >= 0.9  # allows 100ms tolerance


def test_throttler_high_rate_does_not_wait():
    from scraper.livescorebet import Throttler
    t = Throttler(rate=100.0)
    t.acquire()
    start = _time.monotonic()
    t.acquire()
    elapsed = _time.monotonic() - start
    assert elapsed < 0.1



import respx
import httpx as _httpx


class _FakeCurlResp:
    """Stand-in for a curl_cffi Response (the client's primary fetch tier)."""

    def __init__(self, status_code=200, text="", json_data=None):
        self.status_code = status_code
        self.text = text
        self._json = json_data

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise _httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=None, response=None)


def test_client_returns_html_on_200(mocker):
    from scraper.livescorebet import LivescoreBetClient, Throttler
    mocker.patch("curl_cffi.requests.get",
                 return_value=_FakeCurlResp(200, text="<html>ok</html>"))
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    html = client.get_page()
    assert "<html>" in html


def test_client_raises_bot_detected_on_403(mocker):
    from scraper.livescorebet import LivescoreBetClient, Throttler, BotDetectedError
    mocker.patch("curl_cffi.requests.get",
                 return_value=_FakeCurlResp(403, text="Forbidden"))
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    with pytest.raises(BotDetectedError):
        client.get_page()


def test_client_raises_on_500(mocker):
    from scraper.livescorebet import LivescoreBetClient, Throttler
    mocker.patch("curl_cffi.requests.get",
                 return_value=_FakeCurlResp(500, text="error"))
    client = LivescoreBetClient(throttler=Throttler(rate=100.0))
    with pytest.raises(Exception):
        client.get_page()


def test_extract_next_data_success():
    from scraper.livescorebet import _extract_next_data
    import json
    payload = {"props": {"pageProps": {"races": [{"id": "1", "name": "Ascot"}]}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    result = _extract_next_data(html)
    assert result == payload


def test_extract_next_data_missing_tag_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    html = "<html><body>no script here</body></html>"
    with pytest.raises(BotDetectedError, match="not found"):
        _extract_next_data(html)


def test_extract_next_data_malformed_json_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    html = '<html><script id="__NEXT_DATA__" type="application/json">{bad json}</script></html>'
    with pytest.raises(BotDetectedError, match="malformed"):
        _extract_next_data(html)


def test_extract_next_data_no_race_data_raises():
    from scraper.livescorebet import _extract_next_data, BotDetectedError
    import json
    payload = {"props": {"pageProps": {"football": []}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    with pytest.raises(BotDetectedError, match="no recognisable"):
        _extract_next_data(html)


def test_find_race_data_nested():
    from scraper.livescorebet import _find_race_data
    obj = {"a": {"b": {"events": [{"id": 1}]}}}
    result = _find_race_data(obj, "events")
    assert result == [{"id": 1}]


def test_find_race_data_empty_list_not_matched():
    from scraper.livescorebet import _find_race_data
    obj = {"events": []}
    result = _find_race_data(obj, "events")
    assert result is None


_FIXTURES = Path(__file__).parent / "fixtures"


def test_adversarial_gateway_fixture_keeps_only_primary_win_runners():
    from scraper.livescorebet import _parse_event

    payload = json.loads(
        (_FIXTURES / "livescorebet_adversarial.json").read_text(encoding="utf-8")
    )
    rows = _parse_event(payload)

    assert {row["horse_name"] for row in rows} == {"Alpha", "Bravo", "Charlie"}
    assert {row["market_id"] for row in rows} == {"SBTM_WIN_1"}
    assert {row["selection_id"] for row in rows} == {
        "runner-1", "runner-2", "runner-3",
    }
    assert {row["validation_status"] for row in rows} == {"VALID"}


def test_unknown_livescorebet_group_never_defaults_to_win():
    from scraper.livescorebet import _parse_event

    payload = json.loads(
        (_FIXTURES / "livescorebet_adversarial.json").read_text(encoding="utf-8")
    )
    event = payload["currentEvent"][0]
    event["markets"] = [
        market for market in event["markets"]
        if market["id"] == "SBTM_UNKNOWN_1"
    ]
    assert _parse_event(payload) == []


def test_multiple_livescorebet_primary_win_markets_fail_closed():
    from scraper.livescorebet import _parse_event, _rows_to_df

    payload = json.loads(
        (_FIXTURES / "livescorebet_adversarial.json").read_text(encoding="utf-8")
    )
    duplicate = deepcopy(next(
        market for market in payload["currentEvent"][0]["markets"]
        if market["id"] == "SBTM_WIN_1"
    ))
    duplicate["id"] = "SBTM_WIN_2"
    for index, selection in enumerate(duplicate["selections"], start=1):
        selection["id"] = f"duplicate-{index}"
    payload["currentEvent"][0]["markets"].append(duplicate)

    rows = _parse_event(payload)
    assert rows and {row["validation_status"] for row in rows} == {"INVALID"}
    df = _rows_to_df(rows, datetime.now(tz=_tz.utc))
    assert df.empty


from datetime import datetime, timezone as _tz

_SAMPLE_RAW = {
    "races": [
        {
            "id": "r1",
            "startTime": "2026-06-12T14:00:00Z",
            "venue": "Ascot",
            "markets": [
                {
                    "id": "m-win",
                    "type": "WIN",
                    "name": "To win",
                    "eachWayTerms": {"places": 3, "reduction": 0.25},
                    "runners": [
                        {"id": "s1", "name": "Thunder Bay", "price": 2.2, "sp": None},
                        {"id": "s2", "name": "Night Owl", "price": 2.2, "sp": 4.1},
                    ],
                },
                {
                    "type": "EACH_WAY",
                    "eachWayTerms": None,
                    "runners": [
                        {"name": "Thunder Bay", "price": 6.5, "sp": None},
                    ],
                },
            ],
        }
    ]
}


def test_parse_races_row_count():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    assert len(rows) == 2  # strict contract keeps only the primary WIN market


def test_parse_races_field_names():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    expected_keys = {
        "race_id", "race_time", "venue", "market_type", "ew_places",
        "ew_reduction", "ew_margin", "horse_name", "odds_decimal", "sp",
        "is_low_odds", "currency",
    }
    assert expected_keys.issubset(rows[0].keys())
    assert {"market_id", "market_name", "selection_id",
            "validation_status", "validation_reasons"}.issubset(rows[0].keys())


def test_lsb_parquet_columns_include_new_fields():
    from scraper.livescorebet import _PARQUET_COLUMNS
    assert "ew_margin" in _PARQUET_COLUMNS
    assert "is_low_odds" in _PARQUET_COLUMNS
    assert "currency" in _PARQUET_COLUMNS


def test_lsb_merge_preserves_other_sources(tmp_path):
    from scraper.livescorebet import _merge_parquet
    import pandas as pd
    path = str(tmp_path / "live.parquet")
    pd.DataFrame([{"source": "boylesports", "race_id": "x"}]).to_parquet(path, index=False)
    _merge_parquet(pd.DataFrame([{"source": "livescorebet", "race_id": "y"}]), path)
    out = pd.read_parquet(path)
    assert set(out["source"]) == {"boylesports", "livescorebet"}


def test_parse_races_each_way_terms():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    win_rows = [r for r in rows if r["market_type"] == "WIN"]
    assert win_rows[0]["ew_places"] == 3
    assert win_rows[0]["ew_reduction"] == 0.25


def test_parse_races_skips_non_win_markets():
    from scraper.livescorebet import _parse_races
    rows = _parse_races(_SAMPLE_RAW)
    assert {r["market_type"] for r in rows} == {"WIN"}


def test_parse_races_sp_none_becomes_nan():
    from scraper.livescorebet import _parse_races
    import math
    rows = _parse_races(_SAMPLE_RAW)
    thunder_win = next(r for r in rows if r["horse_name"] == "Thunder Bay" and r["market_type"] == "WIN")
    assert math.isnan(thunder_win["sp"])


def test_parse_races_empty_payload():
    from scraper.livescorebet import _parse_races
    rows = _parse_races({"other": "data"})
    assert rows == []


def test_rows_to_df_schema():
    from scraper.livescorebet import _parse_races, _rows_to_df, _PARQUET_COLUMNS
    rows = _parse_races(_SAMPLE_RAW)
    fetched_at = datetime(2026, 6, 12, 14, 0, 0, tzinfo=_tz.utc)
    df = _rows_to_df(rows, fetched_at)
    assert list(df.columns) == _PARQUET_COLUMNS
    assert df["source"].iloc[0] == "livescorebet"
    assert str(df["ew_places"].dtype) == "Int64"
    assert str(df["ew_reduction"].dtype) == "Float64"


def test_rows_to_df_empty_returns_correct_columns():
    from scraper.livescorebet import _rows_to_df
    from datetime import datetime, timezone as _tz
    df = _rows_to_df([], datetime(2026, 6, 12, tzinfo=_tz.utc))
    assert len(df) == 0
    assert "horse_name" in df.columns


import os
import tempfile


def test_cache_manager_is_fresh_false_when_missing():
    from scraper.livescorebet import CacheManager
    with tempfile.TemporaryDirectory() as d:
        cm = CacheManager(path=os.path.join(d, "lsb.json"), ttl=3600)
        assert cm.is_fresh() is False


def test_cache_manager_is_fresh_true_within_ttl():
    from scraper.livescorebet import CacheManager
    import json
    from datetime import datetime, timezone as _tz
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        data = {"fetched_at": datetime.now(_tz.utc).isoformat(), "rows": []}
        with open(path, "w") as f:
            json.dump(data, f)
        cm = CacheManager(path=path, ttl=3600)
        assert cm.is_fresh() is True


def test_cache_manager_is_fresh_false_when_stale():
    from scraper.livescorebet import CacheManager
    import json
    from datetime import datetime, timezone as _tz, timedelta
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        old_time = (datetime.now(_tz.utc) - timedelta(hours=2)).isoformat()
        data = {"fetched_at": old_time, "rows": []}
        with open(path, "w") as f:
            json.dump(data, f)
        cm = CacheManager(path=path, ttl=3600)
        assert cm.is_fresh() is False


def test_cache_manager_write_and_read():
    from scraper.livescorebet import CacheManager
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lsb.json")
        cm = CacheManager(path=path, ttl=3600)
        cm.write({"fetched_at": "2026-06-12T14:00:00+00:00", "rows": [{"horse_name": "X"}]})
        data = cm.read()
        assert data["rows"][0]["horse_name"] == "X"


def test_playwright_fallback_returns_payload(mocker):
    from scraper.livescorebet import PlaywrightFallback

    mock_response = mocker.MagicMock()
    mock_response.url = "https://gateway-ie.livescorebet.com/sportsbook/gateway/v1/view/horses/event?eventid=SBTE_2_123"
    mock_response.json.return_value = {
        "currentEvent": [{
            "id": "SBTE_2_123",
            "name": "Ascot: 14:00",
            "startTime": "2026-06-12 14:00:00",
            "racingData": {"racetrackName": "Ascot"},
            "markets": [{
                "id": "SBTM_2_1",
                "name": "To win",
                "type": "1001558122",
                "groupRanks": [{"marketGroupId": 758, "rank": 1}],
                "eachWay": {},
                "selections": [{"id": "SBTS_2_1", "name": "Thunder Bay", "kind": "REGULAR", "odds": 6.5}],
            }],
            "marketGroups": [],
        }]
    }

    mock_page = mocker.MagicMock()
    mock_browser = mocker.MagicMock()
    mock_browser.new_page.return_value = mock_page

    captured_handler = {}

    def fake_on(event, handler):
        if event == "response":
            captured_handler["fn"] = handler

    mock_page.on.side_effect = fake_on

    def fake_goto(url):
        captured_handler["fn"](mock_response)

    mock_page.goto.side_effect = fake_goto

    mock_pw = mocker.MagicMock()
    mock_pw.__enter__ = mocker.MagicMock(return_value=mock_pw)
    mock_pw.__exit__ = mocker.MagicMock(return_value=False)
    mock_pw.chromium.launch.return_value = mock_browser

    mocker.patch("scraper.livescorebet.sync_playwright", return_value=mock_pw)

    fb = PlaywrightFallback()
    result = fb.fetch()
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0]["horse_name"] == "Thunder Bay"
    assert result[0]["venue"] == "Ascot"


def test_playwright_fallback_raises_when_no_data(mocker):
    from scraper.livescorebet import PlaywrightFallback, ScraperError

    mock_page = mocker.MagicMock()
    mock_browser = mocker.MagicMock()
    mock_browser.new_page.return_value = mock_page
    mock_page.on.return_value = None

    mock_pw = mocker.MagicMock()
    mock_pw.__enter__ = mocker.MagicMock(return_value=mock_pw)
    mock_pw.__exit__ = mocker.MagicMock(return_value=False)
    mock_pw.chromium.launch.return_value = mock_browser

    mocker.patch("scraper.livescorebet.sync_playwright", return_value=mock_pw)

    fb = PlaywrightFallback()
    with pytest.raises(ScraperError, match="no race data"):
        fb.fetch()


_SAMPLE_HTML = b"""
<html><body>
  <div class="race-card" data-race-id="r99" data-primary-market-id="m-win"
       data-market-type="WIN" data-market-name="To win">
    <span class="race-title">Cheltenham</span>
    <span class="race-time">15:30</span>
    <div class="runner" data-selection-id="s1">
      <span class="horse-name">Fast Buck</span>
      <span class="odds">5.5</span>
    </div>
    <div class="runner" data-selection-id="s2">
      <span class="horse-name">Silver Star</span>
      <span class="odds">3.0</span>
    </div>
  </div>
</body></html>
"""


@respx.mock
def test_lxml_fallback_extracts_runners(mocker):
    from scraper.livescorebet import LxmlFallback, Throttler, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(200, content=_SAMPLE_HTML)
    )
    fb = LxmlFallback(throttler=Throttler(rate=100.0))
    rows = fb.fetch()
    horse_names = [r["horse_name"] for r in rows]
    assert "Fast Buck" in horse_names
    assert "Silver Star" in horse_names


@respx.mock
def test_lxml_fallback_raises_when_no_races(mocker):
    from scraper.livescorebet import LxmlFallback, Throttler, ScraperError, _HORSE_RACING_URL
    respx.get(_HORSE_RACING_URL).mock(
        return_value=_httpx.Response(200, content=b"<html><body>nothing</body></html>")
    )
    fb = LxmlFallback(throttler=Throttler(rate=100.0))
    with pytest.raises(ScraperError, match="zero races"):
        fb.fetch()


def test_scrape_returns_dataframe_from_cache(tmp_path, mocker):
    from scraper.livescorebet import scrape, _PARQUET_COLUMNS
    import json
    from datetime import datetime, timezone as _tz

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    cache_data = {
        "fetched_at": datetime.now(_tz.utc).isoformat(),
        "rows": [
            {
                "race_id": "r1", "race_time": "2026-06-12T14:00:00+00:00",
                "venue": "Ascot", "market_type": "WIN", "market_id": "m1",
                "market_name": "To win", "selection_id": "s1",
                "ew_places": None, "ew_reduction": None,
                "horse_name": "Speedy", "odds_decimal": 2.0, "sp": None,
            },
            {
                "race_id": "r1", "race_time": "2026-06-12T14:00:00+00:00",
                "venue": "Ascot", "market_type": "WIN", "market_id": "m1",
                "market_name": "To win", "selection_id": "s2",
                "ew_places": None, "ew_reduction": None,
                "horse_name": "Steady", "odds_decimal": 2.0, "sp": None,
            },
        ],
    }
    cache_path.write_text(json.dumps(cache_data))
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    df = scrape()
    assert len(df) == 2
    assert df["horse_name"].iloc[0] == "Speedy"
    assert list(df.columns) == _PARQUET_COLUMNS
    assert not parquet_path.exists()


def test_scrape_tier1_success_writes_parquet(tmp_path, mocker):
    from scraper.livescorebet import scrape, _PARQUET_COLUMNS

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    home_data = {
        "meetingsToday": [{"events": [{"id": "SBTE_2_123"}], "childs": {}}]
    }
    event_data = {
        "currentEvent": [{
            "id": "SBTE_2_123",
            "name": "York: 14:00",
            "startTime": "2026-06-12 14:00:00",
            "racingData": {"racetrackName": "York"},
            "markets": [{
                "id": "SBTM_2_1",
                "name": "To win",
                "type": "1001558122",
                "groupRanks": [{"marketGroupId": 758, "rank": 1}],
                "eachWay": {},
                "selections": [
                    {"id": "SBTS_2_1", "name": "Blaze", "kind": "REGULAR", "odds": 2.0},
                    {"id": "SBTS_2_2", "name": "Ember", "kind": "REGULAR", "odds": 2.0},
                ],
            }],
            "marketGroups": [],
        }]
    }

    mock_client = mocker.MagicMock()
    mock_client.get_racing_home.return_value = home_data
    mock_client.get_event.return_value = event_data
    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback")
    mocker.patch("scraper.livescorebet.LxmlFallback")

    df = scrape(force=True)
    assert len(df) == 2
    assert df["venue"].iloc[0] == "York"
    assert parquet_path.exists()
    import pandas as pd
    written = pd.read_parquet(str(parquet_path))
    assert list(written.columns) == _PARQUET_COLUMNS


def test_scrape_falls_through_to_playwright(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_racing_home.side_effect = BotDetectedError("403")

    playwright_rows = [
        {"race_id": "r2", "race_time": "2026-06-12T15:00:00+01:00", "venue": "Doncaster",
         "market_type": "WIN", "market_id": "m2", "market_name": "To win",
         "selection_id": "s1", "ew_places": None, "ew_reduction": None,
         "horse_name": "Night Run", "odds_decimal": 2.0, "sp": float("nan")},
        {"race_id": "r2", "race_time": "2026-06-12T15:00:00+01:00", "venue": "Doncaster",
         "market_type": "WIN", "market_id": "m2", "market_name": "To win",
         "selection_id": "s2", "ew_places": None, "ew_reduction": None,
         "horse_name": "Day Run", "odds_decimal": 2.0, "sp": float("nan")},
    ]
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.return_value = playwright_rows

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback")

    df = scrape(force=True)
    assert df["venue"].iloc[0] == "Doncaster"


def test_scrape_falls_through_to_lxml(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError, ScraperError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_racing_home.side_effect = BotDetectedError("403")
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.side_effect = ScraperError("no XHR data")
    mock_lxml = mocker.MagicMock()
    mock_lxml.fetch.return_value = [
        {"race_id": "r3", "race_time": "", "venue": "Kempton", "market_type": "WIN",
         "market_id": "m3", "market_name": "To win", "selection_id": "s1",
         "ew_places": None, "ew_reduction": None, "horse_name": "Starfire",
         "odds_decimal": 2.0, "sp": None},
        {"race_id": "r3", "race_time": "", "venue": "Kempton", "market_type": "WIN",
         "market_id": "m3", "market_name": "To win", "selection_id": "s2",
         "ew_places": None, "ew_reduction": None, "horse_name": "Moonfire",
         "odds_decimal": 2.0, "sp": None},
    ]

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback", return_value=mock_lxml)
    _mock_sel = mocker.MagicMock()
    _mock_sel.fetch.side_effect = Exception("no chrome")
    mocker.patch("scraper._selenium_fallback.SeleniumFallback", return_value=_mock_sel)

    df = scrape(force=True)
    assert df["horse_name"].iloc[0] == "Starfire"


# ── proxy-required, gateway unavailable — never falls back to direct (req 2) ──

def test_lxml_fallback_proxy_unavailable_raises_typed_error_not_direct(mocker):
    from scraper.livescorebet import LxmlFallback, ScraperError
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("all proxies blacklisted")
    mock_client_cls = mocker.patch("httpx.Client")

    fb = LxmlFallback(proxy_rotator=mock_rotator)
    with pytest.raises(ScraperError) as exc_info:
        fb.fetch()

    assert exc_info.value.reason == "proxy_unavailable"
    mock_rotator.next.assert_called_once_with(required=True)
    # Never falls through to an unproxied direct request.
    mock_client_cls.assert_not_called()


def test_lxml_fallback_proxy_unavailable_records_source_health(mocker):
    from scraper.livescorebet import LxmlFallback, ScraperError, _SOURCE
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("no proxies configured")
    record_unavailable = mocker.patch("scraper.livescorebet.source_health.record_unavailable")

    fb = LxmlFallback(proxy_rotator=mock_rotator)
    with pytest.raises(ScraperError):
        fb.fetch()

    record_unavailable.assert_called_once_with(_SOURCE, reason="proxy_unavailable")


def test_scrape_raises_when_all_tiers_fail(tmp_path, mocker):
    from scraper.livescorebet import scrape, BotDetectedError, ScraperError

    cache_path = tmp_path / "livescorebet.json"
    parquet_path = tmp_path / "live_odds.parquet"
    mocker.patch("scraper.livescorebet._CACHE_PATH", str(cache_path))
    mocker.patch("scraper.livescorebet._PARQUET_PATH", str(parquet_path))

    mock_client = mocker.MagicMock()
    mock_client.get_racing_home.side_effect = BotDetectedError("403")
    mock_pw = mocker.MagicMock()
    mock_pw.fetch.side_effect = ScraperError("no XHR")
    mock_lxml = mocker.MagicMock()
    mock_lxml.fetch.side_effect = ScraperError("zero races")

    mocker.patch("scraper.livescorebet.LivescoreBetClient", return_value=mock_client)
    mocker.patch("scraper.livescorebet.PlaywrightFallback", return_value=mock_pw)
    mocker.patch("scraper.livescorebet.LxmlFallback", return_value=mock_lxml)
    _mock_sel = mocker.MagicMock()
    _mock_sel.fetch.side_effect = Exception("no chrome")
    mocker.patch("scraper._selenium_fallback.SeleniumFallback", return_value=_mock_sel)

    with pytest.raises(ScraperError, match="All three fetch tiers failed"):
        scrape(force=True)
