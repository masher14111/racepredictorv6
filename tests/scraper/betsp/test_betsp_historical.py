import json
from datetime import date
import pandas as pd
import scraper.betsp_historical as bh


def test_fetch_one_requests_the_next_days_file_for_the_target_local_date(monkeypatch):
    """Betfair names the file for local race-date D under D+1's filename
    (verified against the live file — see _BACKBONE_FILE_DAY_OFFSET). A short
    bounded-window backfill (scripts/fetch_results_window) depends on
    `_fetch_one`'s requested day matching the target date, not the raw
    filename day, or its enrichment join silently sees 0% position fill."""
    seen = {}

    def fake_file_url(region, market, year, month, day):
        seen["ymd"] = (year, month, day)
        return "https://example.invalid/x.csv"

    monkeypatch.setattr(bh.betfair_sp, "file_url", fake_file_url)
    monkeypatch.setattr(bh.betfair_sp, "fetch_csv", lambda url: None)

    bh._fetch_one((date(2026, 9, 18), "uk", "win"))

    assert seen["ymd"] == (2026, 9, 19)


SAMPLE_CSV = (
    "event_id,menu_hint,event_name,event_dt,selection_id,selection_name,win_lose,bsp,"
    "ppwap,morningwap,ppmax,ppmin,ipmax,ipmin,morningtradedvol,pptradedvol,iptradedvol\n"
    "258715928,Nottingham 31st May,1m2f Hcap,31-05-2026 16:55,96505154,Sharp Romance (IRE),1,2.51,"
    "2.49,2.99,3.0,2.46,2.74,1.01,3337.16,77465.12,38963.54\n"
)


def _next_data(payload: dict) -> str:
    blob = json.dumps(payload)
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{blob}</script></body></html>"
    )


# Sporting Life is a Next.js app: the index lists races, each race-detail page
# carries the full field as typed JSON. Mirror that two-stage shape here.
SL_INDEX = _next_data({"props": {"pageProps": {"meetings": [
    {"races": [{
        "country_short_name": "ENG",
        "course_name": "Nottingham",
        "race_summary_reference": {"id": 555111},
    }]}
]}}})

SL_DETAIL = _next_data({"props": {"pageProps": {"race": {
    "race_summary": {
        # SL time is UTC; the backbone event_dt 16:55 is local (BST on 2026-05-31).
        # 16:55 BST == 15:55 UTC, so the UTC time here must be 15:55 to join.
        "course_name": "Nottingham", "date": "2026-05-31",
        "time": "15:55", "going": "Good to Soft",
    },
    "rides": [{
        "finish_position": 1,
        "ride_status": "RUNNER",
        "horse": {"name": "Sharp Romance"},
        "jockey": {"person_reference": {"id": "oscar-456"}},
        "trainer": {"business_reference": {"id": "charlie-789"}},
    }],
}}}})


def test_fetch_end_to_end(tmp_path, monkeypatch):
    def fake_fetch_csv(url, max_retries=3):
        # Betfair's file for local race date 31 May is published under the
        # NEXT day's filename (01062026) — see _BACKBONE_FILE_DAY_OFFSET.
        return SAMPLE_CSV if "ukwin01062026" in url else None

    def fake_get_html(url):
        # Detail URLs carry the race id; the index URL does not.
        return SL_DETAIL if "555111" in url else SL_INDEX

    monkeypatch.setattr(bh.betfair_sp, "fetch_csv", fake_fetch_csv)
    monkeypatch.setattr(bh, "_results_get_html", fake_get_html)
    monkeypatch.setattr(bh, "_iter_days", lambda years: [date(2026, 5, 31)])

    out_path = str(tmp_path / "betsp.parquet")
    df = bh.fetch(years=[2026], force=True,
                  parquet_path=out_path, raw_root=str(tmp_path / "raw"))

    assert not df.empty
    assert df.loc[0, "horse_name"] == "Sharp Romance (IRE)"
    assert df.loc[0, "odds_finish"] == 2.51
    assert df.loc[0, "position"] == 1
    assert df.loc[0, "jockey_id"] == "oscar-456"
    back = pd.read_parquet(out_path)
    assert set(back["year"]) == {2026}
