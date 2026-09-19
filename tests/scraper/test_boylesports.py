import math as _math
import time as _time

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Imports / components
# ---------------------------------------------------------------------------
def test_imports():
    pass


def test_throttler_first_call_immediate():
    from scraper.boylesports import Throttler
    t = Throttler(rate=1.0)
    start = _time.monotonic()
    t.acquire()
    assert _time.monotonic() - start < 0.1


def test_cache_write_read_roundtrip(tmp_path):
    from scraper.boylesports import CacheManager
    cache = CacheManager(path=str(tmp_path / "bs.json"), ttl=3600)
    cache.write({"fetched_at": "2026-06-12T10:00:00+01:00", "rows": []})
    assert cache.read()["rows"] == []


def test_cache_missing_is_not_fresh(tmp_path):
    from scraper.boylesports import CacheManager
    cache = CacheManager(path=str(tmp_path / "missing.json"), ttl=3600)
    assert cache.is_fresh() is False


# ---------------------------------------------------------------------------
# Client (HTML, with Cloudflare detection)
# ---------------------------------------------------------------------------
import respx
import httpx as _httpx


@respx.mock
def test_client_returns_html():
    from scraper.boylesports import BoyleSportsClient, Throttler, _RACE_CARD_URL
    respx.get(_RACE_CARD_URL).mock(
        return_value=_httpx.Response(200, html="<div>ok</div>")
    )
    client = BoyleSportsClient(throttler=Throttler(rate=100.0))
    assert "ok" in client.get_html(_RACE_CARD_URL)


@respx.mock
def test_client_raises_on_403():
    from scraper.boylesports import (
        BoyleSportsClient, Throttler, BotDetectedError, _RACE_CARD_URL,
    )
    respx.get(_RACE_CARD_URL).mock(return_value=_httpx.Response(403, text="no"))
    with pytest.raises(BotDetectedError):
        BoyleSportsClient(throttler=Throttler(rate=100.0)).get_html(_RACE_CARD_URL)


@respx.mock
def test_client_raises_on_cloudflare_challenge():
    from scraper.boylesports import (
        BoyleSportsClient, Throttler, BotDetectedError, _RACE_CARD_URL,
    )
    respx.get(_RACE_CARD_URL).mock(
        return_value=_httpx.Response(200, html="<title>Just a moment...</title>")
    )
    with pytest.raises(BotDetectedError):
        BoyleSportsClient(throttler=Throttler(rate=100.0)).get_html(_RACE_CARD_URL)


# ── proxy-required, gateway unavailable — never falls back to direct (req 2) ──

def test_get_html_proxy_unavailable_raises_typed_error_not_direct(mocker):
    from scraper.boylesports import BoyleSportsClient, Throttler, ScraperError, _RACE_CARD_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("all proxies blacklisted")
    mock_client_cls = mocker.patch("httpx.Client")

    client = BoyleSportsClient(throttler=Throttler(rate=100.0), proxy_rotator=mock_rotator)
    with pytest.raises(ScraperError) as exc_info:
        client.get_html(_RACE_CARD_URL)

    assert exc_info.value.reason == "proxy_unavailable"
    mock_rotator.next.assert_called_once_with(required=True)
    mock_client_cls.assert_not_called()


def test_get_html_proxy_unavailable_records_source_health(mocker):
    from scraper.boylesports import BoyleSportsClient, Throttler, ScraperError, _SOURCE, _RACE_CARD_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("no proxies configured")
    record_unavailable = mocker.patch("scraper.boylesports.source_health.record_unavailable")

    client = BoyleSportsClient(throttler=Throttler(rate=100.0), proxy_rotator=mock_rotator)
    with pytest.raises(ScraperError):
        client.get_html(_RACE_CARD_URL)

    record_unavailable.assert_called_once_with(_SOURCE, reason="proxy_unavailable")


def test_curl_cffi_get_html_proxy_unavailable_raises_typed_error_not_direct(mocker):
    from scraper.boylesports import ScraperError, _RACE_CARD_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("all proxies blacklisted")
    mocker.patch("scraper.boylesports.get_proxy_manager", return_value=mock_rotator)
    mock_get = mocker.patch("curl_cffi.requests.get")

    from scraper.boylesports import _curl_cffi_get_html
    with pytest.raises(ScraperError) as exc_info:
        _curl_cffi_get_html(_RACE_CARD_URL)

    assert exc_info.value.reason == "proxy_unavailable"
    mock_rotator.next.assert_called_once_with(required=True)
    mock_get.assert_not_called()


def test_curl_cffi_get_html_proxy_unavailable_records_source_health(mocker):
    from scraper.boylesports import ScraperError, _SOURCE, _RACE_CARD_URL
    from utils.proxy_manager import ProxyUnavailableError

    mock_rotator = mocker.MagicMock()
    mock_rotator.next.side_effect = ProxyUnavailableError("no proxies configured")
    mocker.patch("scraper.boylesports.get_proxy_manager", return_value=mock_rotator)
    mocker.patch("curl_cffi.requests.get")
    record_unavailable = mocker.patch("scraper.boylesports.source_health.record_unavailable")

    from scraper.boylesports import _curl_cffi_get_html
    with pytest.raises(ScraperError):
        _curl_cffi_get_html(_RACE_CARD_URL)

    record_unavailable.assert_called_once_with(_SOURCE, reason="proxy_unavailable")


# ---------------------------------------------------------------------------
# Price parsing
# ---------------------------------------------------------------------------
def test_to_decimal_fractional_and_evens():
    from scraper.boylesports import _to_decimal
    assert _to_decimal("5/2") == 3.5
    assert _to_decimal("1/5") == 1.2
    assert _to_decimal("2/7") == round(1 + 2 / 7, 6)
    assert _to_decimal("EVS") == 2.0
    assert _to_decimal("Evens") == 2.0


def test_to_decimal_nr_and_garbage_is_nan():
    from scraper.boylesports import _to_decimal
    assert _math.isnan(_to_decimal("NR"))
    assert _math.isnan(_to_decimal(""))
    assert _math.isnan(_to_decimal(None))
    assert _math.isnan(_to_decimal("not-a-price"))


# ---------------------------------------------------------------------------
# Each-way margin
# ---------------------------------------------------------------------------
def test_ew_margin_basic():
    from scraper.boylesports import _ew_margin
    margin = _ew_margin([5.0, 3.0], places=2, reduction=0.25)
    assert round(margin, 4) == round(0.5 + 1 / 1.5 - 2, 4)


def test_ew_margin_missing_terms_is_nan():
    from scraper.boylesports import _ew_margin
    assert _math.isnan(_ew_margin([5.0], places=None, reduction=0.25))
    assert _math.isnan(_ew_margin([5.0], places=2, reduction=None))


# ---------------------------------------------------------------------------
# HTML fixtures (real BoyleSports structure, confirmed via DevTools 2026-06-12)
# ---------------------------------------------------------------------------
_INDEX_HTML = """
<div class="rc-day-group racecard-meeting-content ukirefeatured_content" data-date="20260612">
  <li class="race-card-group   race-time  ">
    <a class="ui-raceCards flex" href="/sports/horse-racing/uk-ire-featured/120626/chester/23:50"
       data-eventid="45248940.10"><span class="time">23:50</span></a>
  </li>
  <li class="race-card-group   race-time  race-resulted ">
    <a class="ui-raceCards flex" href="/sports/horse-racing/uk-ire-featured/120626/chester/13:30"
       data-eventid="45248941.10"><span class="time">13:30</span></a>
  </li>
</div>
"""

_EVENT_HTML = """
<div class="event">
  <section class="market-group" data-marketid="m1" data-market-name="Win or Each Way">
    <h2 class="market-title">Win or Each Way</h2>
    <a class="odds luf white black-text" data-selectionid="s1" data-name="Miss Lady Grace"
       data-price="10/11" data-marketid="m1" data-isnr="False"></a>
    <a class="relative" data-selectionid="s1" data-name="Miss Lady Grace"
       data-price="10/11" data-marketid="m1"></a>
    <a class="odds" data-selectionid="s2" data-name="The Resdev Scholar"
       data-price="7/2" data-marketid="m1" data-isnr="False"></a>
    <a class="odds" data-selectionid="s3" data-name="Scratched"
       data-price="NR" data-marketid="m1" data-isnr="True"></a>
  </section>
</div>
"""

_FIXTURES = Path(__file__).parent / "fixtures"


def test_adversarial_html_fixture_scopes_to_primary_win_container():
    from scraper.boylesports import _parse_event

    html = (_FIXTURES / "boylesports_adversarial.html").read_text(encoding="utf-8")
    rows = _parse_event(
        html,
        {"race_id": "r-adv", "venue": "Ascot",
         "race_time": "2026-07-25T14:00:00+01:00"},
    )

    assert {row["horse_name"] for row in rows} == {"Alpha", "Bravo", "Charlie"}
    assert {row["market_id"] for row in rows} == {"primary-win"}
    assert {row["validation_status"] for row in rows} == {"VALID"}


def test_live_dom_contract_excludes_boosts_featured_duplicates_and_favourites():
    from scraper.boylesports import _parse_event

    html = (
        _FIXTURES / "boylesports_live_dom_adversarial.html"
    ).read_text(encoding="utf-8")
    rows = _parse_event(
        html,
        {"race_id": "r-live-dom", "venue": "Ascot",
         "race_time": "2026-07-25T14:00:00+01:00"},
    )

    assert [row["horse_name"] for row in rows] == ["Alpha", "Bravo", "Charlie"]
    assert {row["market_id"] for row in rows} == {"primary-win"}
    assert {row["market_name"] for row in rows} == {"win or e/w"}
    assert {row["validation_status"] for row in rows} == {"VALID"}


def test_boylesports_unlabelled_odds_buttons_fail_closed():
    from scraper.boylesports import _parse_event

    html = """
    <div><a class="odds" data-marketid="mystery" data-selectionid="s1"
    data-name="Alpha" data-price="2/1"></a></div>
    """
    rows = _parse_event(
        html, {"race_id": "r", "venue": "Ascot", "race_time": ""}
    )
    assert rows == []


def test_multiple_boylesports_primary_containers_fail_closed():
    from scraper.boylesports import _parse_event, _rows_to_df
    from datetime import datetime, timezone

    html = (_FIXTURES / "boylesports_adversarial.html").read_text(encoding="utf-8")
    second = html.replace(
        'data-marketid="primary-win"',
        'data-marketid="primary-win-2"',
    ).replace(
        'data-marketid="special-pair"',
        'data-marketid="ignored-pair"',
    )
    # Two full pages yield two labelled primary containers in one event document.
    combined = f"<html><body>{html}{second}</body></html>"
    rows = _parse_event(
        combined,
        {"race_id": "r-adv", "venue": "Ascot",
         "race_time": "2026-07-25T14:00:00+01:00"},
    )
    assert rows and {row["validation_status"] for row in rows} == {"INVALID"}
    assert _rows_to_df(rows, datetime.now(timezone.utc)).empty


def test_parse_index_event_meta():
    from scraper.boylesports import _parse_index
    events = _parse_index(_INDEX_HTML)
    assert len(events) == 2
    open_ev = next(e for e in events if not e["resulted"])
    assert open_ev["race_id"] == "45248940.10"
    assert open_ev["venue"] == "Chester"
    assert open_ev["region"] == "uk-ire-featured"
    assert open_ev["event_url"].endswith("/chester/23:50")
    assert open_ev["race_time"].startswith("2026-06-12T23:50")
    assert any(e["resulted"] for e in events)


def test_parse_index_excludes_virtuals(monkeypatch):
    import scraper.boylesports as bs
    monkeypatch.setattr(bs, "_EXCLUDED_REGIONS", {"virtuals"})
    html = _INDEX_HTML + """
    <li class="race-card-group race-time ">
      <a class="ui-raceCards" href="/sports/horse-racing/virtuals/120626/portman-park/23:55"
         data-eventid="99999999.10"><span class="time">23:55</span></a>
    </li>
    """
    events = bs._parse_index(html)
    regions = {e["region"] for e in events}
    assert "virtuals" not in regions
    assert "uk-ire-featured" in regions


def test_parse_event_rows_dedupe_and_nr():
    from scraper.boylesports import _parse_event
    meta = {"race_id": "45248940.10", "venue": "Chester",
            "race_time": "2026-06-12T23:50:00+01:00"}
    rows = _parse_event(_EVENT_HTML, meta)
    # NR dropped, duplicate selection collapsed -> 2 runners
    assert len(rows) == 2
    fav = next(r for r in rows if r["horse_name"] == "Miss Lady Grace")
    assert fav["odds_decimal"] == round(1 + 10 / 11, 6)
    assert fav["currency"] == "GBP"          # Chester = UK
    assert fav["market_type"] == "WIN"
    assert fav["ew_places"] is None          # not in cards; cross-sourced later


def test_parse_event_low_odds_flag():
    from scraper.boylesports import _parse_event
    meta = {"race_id": "r", "venue": "Chester", "race_time": ""}
    rows = _parse_event(_EVENT_HTML, meta)
    fav = next(r for r in rows if r["horse_name"] == "Miss Lady Grace")
    longshot = next(r for r in rows if r["horse_name"] == "The Resdev Scholar")
    assert fav["is_low_odds"] is True        # 1.909 < 2.0
    assert longshot["is_low_odds"] is False  # 4.5


# ---------------------------------------------------------------------------
# Cross-source each-way
# ---------------------------------------------------------------------------
def test_cross_source_ew_fills_terms_and_margin(tmp_path):
    import pandas as pd
    from scraper.boylesports import _cross_source_ew
    path = str(tmp_path / "live.parquet")
    # Another bookie's row for the same race (same venue + instant).
    other = pd.DataFrame([{
        "source": "livescorebet", "venue": "Chester",
        "race_time": pd.Timestamp("2026-06-12T22:50:00Z"),
        "ew_places": 3, "ew_reduction": 0.2,
    }])
    other.to_parquet(path, index=False)

    rows = [
        {"race_id": "r1", "venue": "Chester",
         "race_time": "2026-06-12T23:50:00+01:00", "odds_decimal": 1.9,
         "ew_places": None, "ew_reduction": None, "ew_margin": float("nan")},
        {"race_id": "r1", "venue": "Chester",
         "race_time": "2026-06-12T23:50:00+01:00", "odds_decimal": 4.5,
         "ew_places": None, "ew_reduction": None, "ew_margin": float("nan")},
    ]
    out = _cross_source_ew(rows, path)
    assert out[0]["ew_places"] == 3
    assert out[0]["ew_reduction"] == 0.2
    assert not _math.isnan(out[0]["ew_margin"])
    assert out[0]["ew_margin"] == out[1]["ew_margin"]  # per-race scalar


def test_cross_source_ew_no_parquet_is_noop(tmp_path):
    from scraper.boylesports import _cross_source_ew
    rows = [{"race_id": "r", "venue": "X", "race_time": "", "odds_decimal": 2.0,
             "ew_places": None, "ew_reduction": None, "ew_margin": float("nan")}]
    out = _cross_source_ew(rows, str(tmp_path / "missing.parquet"))
    assert out[0]["ew_places"] is None


# ---------------------------------------------------------------------------
# DataFrame / parquet
# ---------------------------------------------------------------------------
def test_rows_to_df_columns():
    from scraper.boylesports import _rows_to_df, _PARQUET_COLUMNS
    from datetime import datetime, timezone
    rows = [{
        "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
        "venue": "Chester", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": -0.5, "horse_name": "H",
        "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
        "currency": "GBP",
    }, {
        "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
        "venue": "Chester", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": -0.5, "horse_name": "I",
        "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
        "currency": "GBP",
    }]
    df = _rows_to_df(rows, datetime(2026, 6, 12, tzinfo=timezone.utc))
    assert list(df.columns) == _PARQUET_COLUMNS
    assert df.iloc[0]["source"] == "boylesports"


def test_merge_parquet_preserves_other_sources(tmp_path):
    import pandas as pd
    from scraper.boylesports import _merge_parquet
    path = str(tmp_path / "live_odds.parquet")
    pd.DataFrame([{"source": "livescorebet", "race_id": "x"}]).to_parquet(path, index=False)
    _merge_parquet(pd.DataFrame([{"source": "boylesports", "race_id": "y"}]), path)
    out = pd.read_parquet(path)
    assert set(out["source"]) == {"livescorebet", "boylesports"}


# ---------------------------------------------------------------------------
# Row collection + scrape orchestration
# ---------------------------------------------------------------------------
def test_collect_rows_index_then_events():
    from scraper.boylesports import _collect_rows, _RACE_CARD_URL
    pages = {_RACE_CARD_URL: _INDEX_HTML}

    def get_html(url):
        if url in pages:
            return pages[url]
        return _EVENT_HTML  # any event url returns the event fixture

    rows = _collect_rows(get_html)
    # only the open (non-resulted) event is fetched -> 2 runners
    assert len(rows) == 2
    assert {r["horse_name"] for r in rows} == {"Miss Lady Grace", "The Resdev Scholar"}


def test_scrape_returns_cached_when_fresh(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    cache_path = str(tmp_path / "bs.json")
    monkeypatch.setattr(bs, "_CACHE_PATH", cache_path)
    bs.CacheManager(path=cache_path).write({
        "fetched_at": bs.datetime.now(tz=bs.TZ).isoformat(),
        "rows": [{
            "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
            "venue": "Chester", "market_type": "WIN", "market_id": "m1",
            "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "H",
            "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
            "currency": "GBP",
        }, {
            "race_id": "r1", "race_time": "2026-06-12T15:30:00+01:00",
            "venue": "Chester", "market_type": "WIN", "market_id": "m1",
            "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "I",
            "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
            "currency": "GBP",
        }],
    })
    df = bs.scrape()
    assert len(df) == 2 and df.iloc[0]["horse_name"] == "H"


def test_scrape_live_success(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    monkeypatch.setattr(bs, "_CACHE_PATH", str(tmp_path / "bs.json"))
    monkeypatch.setattr(bs, "_PARQUET_PATH", str(tmp_path / "live.parquet"))
    monkeypatch.setattr(bs, "_run_live_tiers", lambda: [{
        "race_id": "r1", "race_time": "2026-06-12T23:50:00+01:00",
        "venue": "Chester", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": None,
        "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "Miss Lady Grace",
        "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False, "currency": "GBP",
    }, {
        "race_id": "r1", "race_time": "2026-06-12T23:50:00+01:00",
        "venue": "Chester", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": None,
        "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "Other Horse",
        "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False, "currency": "GBP",
    }])
    df = bs.scrape(force=True)
    assert len(df) == 2
    assert df.iloc[0]["currency"] == "GBP"
    assert (tmp_path / "live.parquet").exists()
    assert bool(df.iloc[0]["stale"]) is False


def test_scrape_falls_back_to_stale_cache(tmp_path, monkeypatch):
    import scraper.boylesports as bs
    cache_path = str(tmp_path / "bs.json")
    monkeypatch.setattr(bs, "_CACHE_PATH", cache_path)
    monkeypatch.setattr(bs, "_PARQUET_PATH", str(tmp_path / "live.parquet"))
    bs.CacheManager(path=cache_path).write({
        "fetched_at": "2000-01-01T00:00:00+00:00",
        "rows": [{
            "race_id": "old", "race_time": "2000-01-01T00:00:00+00:00",
            "venue": "Cork", "market_type": "WIN", "market_id": "m-old",
            "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "Z",
            "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
            "currency": "EUR",
        }, {
            "race_id": "old", "race_time": "2000-01-01T00:00:00+00:00",
            "venue": "Cork", "market_type": "WIN", "market_id": "m-old",
            "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": None,
            "ew_reduction": None, "ew_margin": float("nan"), "horse_name": "Y",
            "odds_decimal": 2.0, "sp": float("nan"), "is_low_odds": False,
            "currency": "EUR",
        }],
    })
    monkeypatch.setattr(bs, "_run_live_tiers", lambda: None)
    df = bs.scrape(force=True)
    assert df.iloc[0]["horse_name"] == "Z"
    assert bool(df.iloc[0]["stale"]) is True
