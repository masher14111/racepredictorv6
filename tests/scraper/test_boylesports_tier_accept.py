"""Tier acceptance in _run_live_tiers.

Cloudflare usually serves the race-card index and then 403s most event pages, so
a cheap tier "succeeds" with one race on the card. Returning that stopped the
chain before the anti-detect browser ran: on 2026-09-18 curl_cffi returned 14
rows from 1 race while Botasaurus could deliver 80 from 9.
"""

import pytest

import scraper.boylesports as bs


def _rows(n_races: int, per_race: int = 10) -> list:
    return [
        {"race_id": f"r{i}", "horse_name": f"H{i}-{j}", "odds_decimal": 3.0}
        for i in range(n_races)
        for j in range(per_race)
    ]


@pytest.fixture
def tiers(monkeypatch):
    """Silence every tier; individual tests re-enable the ones they need."""
    monkeypatch.setattr(bs, "_MIN_RACES", 3)
    monkeypatch.setattr(bs, "_STICKY_ENABLED", False)
    monkeypatch.setattr(bs, "_BOTA_ENABLED", False)
    calls: list[str] = []

    def _stub(name, result):
        def _fn(*a, **k):
            calls.append(name)
            if isinstance(result, Exception):
                raise result
            return result
        return _fn

    def _install(**kw):
        for tier, result in kw.items():
            attr = {
                "sticky": "_sticky_collect",
                "curl": "_curl_collect",
                "bota": "_botasaurus_collect",
                "playwright": "_playwright_collect",
                "selenium": "_selenium_collect",
                "firecrawl": "_firecrawl_collect",
            }[tier]
            monkeypatch.setattr(bs, attr, _stub(tier, result))
        # httpx tier goes through _collect_rows
        if "httpx" not in kw:
            monkeypatch.setattr(bs, "_collect_rows", _stub("httpx", []))
        monkeypatch.setattr(bs, "BoyleSportsClient", lambda *a, **k: object())

    return type("T", (), {"install": staticmethod(_install), "calls": calls})


def _fill(tiers, **kw):
    defaults = dict(sticky=[], curl=[], bota=[], playwright=[], selenium=[], firecrawl=[])
    defaults.update(kw)
    tiers.install(**defaults)


# ---------------------------------------------------------------------------
# The bug
# ---------------------------------------------------------------------------
def test_a_one_race_card_does_not_stop_the_chain(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(1, 14), bota=_rows(9, 9))
    out = bs._run_live_tiers()
    assert "bota" in tiers.calls, "Botasaurus must still run after a thin card"
    assert len({r["race_id"] for r in out}) == 9


def test_a_full_card_stops_the_chain(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(9, 9), bota=_rows(9, 9))
    bs._run_live_tiers()
    assert "bota" not in tiers.calls, "no need to open a browser for a good card"


def test_exactly_min_races_is_accepted(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(3), bota=_rows(9))
    bs._run_live_tiers()
    assert "bota" not in tiers.calls


# ---------------------------------------------------------------------------
# The fallback: a partial card still beats nothing
# ---------------------------------------------------------------------------
def test_best_partial_is_returned_when_nothing_reaches_the_bar(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(1, 5), bota=_rows(2, 10))
    out = bs._run_live_tiers()
    assert out is not None
    assert len(out) == 20, "the richest partial card wins"


def test_all_empty_returns_none(tiers):
    _fill(tiers)
    assert bs._run_live_tiers() is None


def test_a_failing_tier_does_not_lose_an_earlier_partial(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(1, 7), bota=RuntimeError("browser died"))
    out = bs._run_live_tiers()
    assert out is not None and len(out) == 7


def test_later_richer_partial_replaces_an_earlier_thinner_one(tiers, monkeypatch):
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=_rows(1, 3), bota=_rows(2, 6))
    assert len(bs._run_live_tiers()) == 12


def test_rows_without_race_ids_never_satisfy_the_bar(tiers, monkeypatch):
    """Defensive: unkeyed rows must not be mistaken for full coverage."""
    monkeypatch.setattr(bs, "_BOTA_ENABLED", True)
    _fill(tiers, curl=[{"horse_name": "X"}] * 50, bota=_rows(9))
    out = bs._run_live_tiers()
    assert "bota" in tiers.calls
    assert len({r["race_id"] for r in out}) == 9
