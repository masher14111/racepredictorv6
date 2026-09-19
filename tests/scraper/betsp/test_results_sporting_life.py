import json
from datetime import date
from scraper.betsp.results.base import RawResult
from scraper.betsp.results.sporting_life import SportingLife


def _next_data(payload: dict) -> str:
    blob = json.dumps(payload)
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{blob}</script></body></html>"
    )


# A race-detail page's __NEXT_DATA__: race_summary carries course/date/time/going,
# rides[] carries the full field. The second runner is a non-finisher (pulled up)
# with a non-positive finish_position, which must map to position=None.
DETAIL = _next_data({"props": {"pageProps": {"race": {
    "race_summary": {
        # SL time is UTC; on 2026-05-31 (BST) it converts to local +1h.
        "course_name": "Nottingham", "date": "2026-05-31",
        "time": "15:55", "going": "Good to Soft",
    },
    "rides": [
        {
            "finish_position": 1,
            "ride_status": "RUNNER",
            "horse": {"name": "Sharp Romance"},
            "jockey": {"person_reference": {"id": "oscar-456"}},
            "trainer": {"business_reference": {"id": "charlie-789"}},
        },
        {
            "finish_position": 0,  # pulled up / non-finisher
            "ride_status": "PULLED_UP",
            "horse": {"name": "Penelope Valentine"},
            "jockey": {"person_reference": {"id": "tom-457"}},
            "trainer": {"business_reference": {"id": "sara-790"}},
        },
    ],
}}}})


def test_parse_extracts_rows():
    src = SportingLife()
    raw = RawResult(source="sporting_life", race_date=date(2026, 5, 31),
                    url="http://x", html=DETAIL, venue="Nottingham")
    rows = src.parse(raw)
    assert len(rows) == 2

    r0 = rows[0]
    assert r0.horse_name == "Sharp Romance"
    assert r0.position == 1
    assert r0.jockey_id == "oscar-456"
    assert r0.trainer_id == "charlie-789"
    assert r0.going == "Good to Soft"
    assert r0.venue == "Nottingham"
    assert r0.source == "sporting_life"
    # race_date must be an exact-minute ISO string in LOCAL time for the join key:
    # SL's 15:55 UTC -> 16:55 local (Europe/Dublin BST) on this date.
    assert r0.race_date == "2026-05-31T16:55"

    # Non-finisher keeps its row but carries a null position.
    assert rows[1].horse_name == "Penelope Valentine"
    assert rows[1].position is None


def test_parse_handles_missing_next_data():
    src = SportingLife()
    raw = RawResult(source="sporting_life", race_date=date(2026, 5, 31),
                    url="http://x", html="<html>no json here</html>")
    assert src.parse(raw) == []
