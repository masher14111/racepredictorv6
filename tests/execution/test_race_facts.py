"""Offline settlement ground truth: parsing, and the bridge into settlement.

The synthetic cases pin the contract; the archive-backed cases at the bottom
prove the contract holds on the real 2026 documents and skip cleanly when the
archive is absent (CI, a fresh clone).
"""
from __future__ import annotations

import gzip
import json
import os

import pytest

from execution import race_facts as rf
from execution.settlement import EachWayTerms, settle_ticket

ARCHIVE = os.path.join("data", "historical", "raw", "sporting_life")
needs_archive = pytest.mark.skipif(
    not os.path.isdir(ARCHIVE), reason="raw Sporting Life archive not present"
)


# ────────────────────────────── document builders ────────────────────────────
def _ride(name, *, status="RUNNER", position=None, casualty=None, odds="2/1", draw=None):
    return {
        "horse": {"name": name},
        "ride_status": status,
        # 0 is the archive's "did not complete" sentinel, and non-runners carry
        # it too — the parser must collapse both to None.
        "finish_position": 0 if position is None else position,
        "casualty": {"reason": casualty} if casualty else {},
        "betting": {"current_odds": odds},
        "draw_number": draw,
    }


def _doc(rides, *, deductions=None, places=2, course="Windsor", day="2026-07-20",
         clock="19:20"):
    return {
        "props": {
            "pageProps": {
                "race": {
                    "race_summary": {
                        "course_name": course,
                        "date": day,
                        "time": clock,
                        "off_time": f"{clock}:11",
                        "going": "Good",
                        "distance": "1m",
                        "race_class": "6",
                        "has_handicap": True,
                    },
                    "number_of_placed_rides": places,
                    "deductions": deductions or [],
                    "rides": rides,
                }
            }
        }
    }


# ──────────────────────────────── key + parsers ──────────────────────────────
def test_race_facts_key_normalises_venue_and_truncates_to_the_minute():
    assert rf.race_facts_key("Newmarket (July)", "2026-06-25T11:15:42+01:00") == (
        "newmarketjuly|2026-06-25T11:15"
    )


@pytest.mark.parametrize(
    "day, clock, expected",
    [
        # Winter: the archive's UTC clock and local racing time coincide.
        ("2026-01-14", "14:20", "2026-01-14T14:20"),
        # Summer: BST is UTC+1, so the local off is an hour later than the file.
        ("2026-07-20", "14:20", "2026-07-20T15:20"),
        # The DST boundary itself (BST began 2026-03-29 01:00 UTC).
        ("2026-03-28", "16:05", "2026-03-28T16:05"),
        ("2026-03-29", "16:05", "2026-03-29T17:05"),
    ],
)
def test_the_archive_clock_is_read_as_utc_and_keyed_locally(day, clock, expected):
    """The join defect that made settlement read the wrong race.

    Sporting Life stamps a naive clock in UTC; ``race_uid``, the Betfair panel and
    ``scraper.betsp.joiner`` are all in local racing time. Reading the archive
    clock as local does not merely lose the summer join — a 16:20 UTC race at one
    course usually has a sibling at 16:20 local an hour up the road, so roughly a
    quarter of races matched *something* and settled from a different race.
    """
    facts = rf.parse_race_document(
        _doc([_ride("A", position=1), _ride("B", position=2)], day=day, clock=clock)
    )
    assert facts.race_key == f"windsor|{expected}"
    # Both halves of the key must move together, or a late race would carry a
    # local time against the previous day's date.
    assert facts.race_date == expected[:10]


@pytest.mark.parametrize(
    "raw, expected",
    [("3/1", 4.0), ("10/3", 4.3333), ("evens", 2.0), ("1/1", 2.0), ("4.5", 4.5)],
)
def test_fractional_to_decimal(raw, expected):
    assert rf.fractional_to_decimal(raw) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("raw", [None, "", "SP", "0/0", "0.5", "not odds"])
def test_fractional_to_decimal_rejects_unusable(raw):
    assert rf.fractional_to_decimal(raw) is None


# ─────────────────────────────── status semantics ────────────────────────────
def test_non_finisher_is_a_loser_not_a_void():
    """The defect this module exists to avoid: refunding a losing bet."""
    facts = rf.parse_race_document(
        _doc([
            _ride("Winner", position=1),
            _ride("Faller", casualty="Fell"),
        ])
    )
    faller = facts.runner("Faller")
    assert faller.ran and not faller.completed
    assert faller.finish_position is None
    assert faller.casualty_reason == "Fell"
    assert facts.outcome_for("Faller") == rf.LOSE


def test_non_runner_is_void_and_excluded_from_the_runner_count():
    facts = rf.parse_race_document(
        _doc([
            _ride("Winner", position=1),
            _ride("Scratched", status="NONRUNNER"),
            _ride("Pulled", status="WITHDRAWN"),
        ])
    )
    assert facts.declared_field == 3
    assert facts.n_runners == 1
    assert facts.non_runners == {"scratched", "pulled"}
    assert facts.outcome_for("Scratched") == rf.NON_RUNNER
    assert facts.outcome_for("Pulled") == rf.NON_RUNNER


def test_a_horse_not_in_the_race_is_unknown_never_void():
    facts = rf.parse_race_document(_doc([_ride("Winner", position=1)]))
    assert facts.outcome_for("Some Other Horse") == rf.UNKNOWN


def test_places_paid_drives_the_place_outcome():
    facts = rf.parse_race_document(
        _doc([_ride(f"H{i}", position=i) for i in range(1, 5)], places=2)
    )
    assert facts.places_paid == 2
    assert [facts.outcome_for(f"H{i}") for i in range(1, 5)] == [
        rf.WIN, rf.PLACE, rf.LOSE, rf.LOSE
    ]


# ───────────────────────────────── dead heats ────────────────────────────────
def test_dead_heat_size_counts_only_real_finishing_positions():
    facts = rf.parse_race_document(
        _doc([
            _ride("A", position=1),
            _ride("B", position=1),
            _ride("C", position=3),
            _ride("Faller", casualty="PulledUp"),
            _ride("Another", casualty="Fell"),
        ])
    )
    assert facts.runner("A").dead_heat_size == 2
    assert facts.runner("C").dead_heat_size == 1
    # The two non-completers share the position-0 sentinel but are NOT a dead heat.
    assert facts.runner("Faller").dead_heat_size == 1
    assert facts.runner("Another").dead_heat_size == 1


# ────────────────────────────────── rule 4 ───────────────────────────────────
def test_rule_4_reads_the_published_deduction_in_pence():
    facts = rf.parse_race_document(
        _doc([_ride("A", position=1)], deductions=[{"type": "AllBets", "value": 25}])
    )
    assert facts.rule_4_deduction == 0.25
    assert facts.rule_4_type == "AllBets"


def test_rule_4_sums_within_a_type_and_maxes_across_types():
    facts = rf.parse_race_document(
        _doc(
            [_ride("A", position=1)],
            deductions=[
                {"type": "AllBets", "value": 10},
                {"type": "AllBets", "value": 15},
                {"type": "BoardPrices", "value": 20},
            ],
        )
    )
    # Two withdrawals sum to 25p within AllBets; BoardPrices' 20p describes the
    # same money, so it does not add on top.
    assert facts.rule_4_deduction == 0.25


def test_rule_4_is_capped_and_ignores_unknown_types():
    capped = rf.parse_race_document(
        _doc([_ride("A", position=1)], deductions=[{"type": "AllBets", "value": 250}])
    )
    assert capped.rule_4_deduction == 0.90
    unknown = rf.parse_race_document(
        _doc([_ride("A", position=1)], deductions=[{"type": "Mystery", "value": 40}])
    )
    assert unknown.rule_4_deduction == 0.0
    assert unknown.rule_4_type == ""


# ───────────────────────────── settlement bridge ─────────────────────────────
def test_bridge_settles_a_faller_as_a_loss():
    facts = rf.parse_race_document(
        _doc([_ride("Winner", position=1), _ride("Faller", casualty="Fell")])
    )
    result = settle_ticket(
        stake=10.0, decimal_odds=5.0, bet_type="win",
        horse_key="faller", race_result=facts.to_race_result(),
    )
    assert result.status == "LOSE"
    assert result.profit == pytest.approx(-10.0)


def test_bridge_leaves_an_unjoined_horse_unsettled():
    facts = rf.parse_race_document(
        _doc([_ride("Winner", position=1), _ride("Second", position=2)])
    )
    result = settle_ticket(
        stake=10.0, decimal_odds=5.0, bet_type="win",
        horse_key="never heard of it", race_result=facts.to_race_result(),
    )
    assert result.status == "NO_RESULT"
    assert result.returns == 0.0 and result.profit == 0.0


def test_bridge_prefers_the_published_rule_4_over_the_derived_table():
    facts = rf.parse_race_document(
        _doc(
            [
                _ride("Winner", position=1),
                _ride("Second", position=2),
                _ride("Gone", status="NONRUNNER", odds="1/2"),
            ],
            deductions=[{"type": "AllBets", "value": 25}],
        )
    )
    race_result = facts.to_race_result()
    assert race_result.rule_4_override == 0.25
    result = settle_ticket(
        stake=10.0, decimal_odds=5.0, bet_type="win",
        horse_key="winner", race_result=race_result,
    )
    # A 1/2 withdrawal derives 0.55 from the Tattersalls table; the published
    # 25p is the evidence and must win.
    assert result.rule_4_deduction == 0.25
    assert result.detail["rule_4_source"] == "published"
    assert result.returns == pytest.approx(10.0 * (1.0 + 4.0 * 0.75))


def test_bridge_halves_a_dead_heated_winner():
    facts = rf.parse_race_document(
        _doc([_ride("A", position=1), _ride("B", position=1), _ride("C", position=3)])
    )
    result = settle_ticket(
        stake=10.0, decimal_odds=5.0, bet_type="win",
        horse_key="a", race_result=facts.to_race_result(),
    )
    assert result.dead_heat_divisor == 2.0
    assert result.returns == pytest.approx(25.0)


def test_bridge_binds_each_way_places_down_to_what_the_race_paid():
    facts = rf.parse_race_document(
        _doc([_ride(f"H{i}", position=i) for i in range(1, 9)], places=2)
    )
    result = settle_ticket(
        stake=10.0, decimal_odds=9.0, bet_type="each_way", horse_key="h3",
        race_result=facts.to_race_result(),
        ew_terms=EachWayTerms(places=3, fraction=0.2),
    )
    assert result.detail["places_used"] == 2
    assert result.status == "LOSE"


def test_bridge_voids_a_walkover():
    facts = rf.parse_race_document(
        _doc([_ride("Alone", position=1), _ride("Gone", status="NONRUNNER")])
    )
    assert facts.n_runners == 1
    assert facts.to_race_result().void_race is True


# ─────────────────────────────── frame round-trip ────────────────────────────
def test_frame_round_trip_preserves_every_settlement_fact():
    facts = [
        rf.parse_race_document(
            _doc(
                [
                    _ride("A", position=1),
                    _ride("B", position=1),
                    _ride("Faller", casualty="PulledUp"),
                    _ride("Gone", status="NONRUNNER"),
                ],
                deductions=[{"type": "BoardPrices", "value": 30}],
            )
        )
    ]
    frame = rf.to_frame(facts)
    assert len(frame) == 4
    restored = rf.facts_from_frame(frame)[facts[0].race_key]
    assert restored.rule_4_deduction == 0.30
    assert restored.places_paid == 2
    assert restored.non_runners == {"gone"}
    assert restored.runner("A").dead_heat_size == 2
    assert restored.runner("Faller").finish_position is None
    assert restored.runner("Faller").casualty_reason == "PulledUp"
    assert restored.outcome_for("Faller") == rf.LOSE


def test_empty_frame_still_has_the_schema():
    frame = rf.to_frame([])
    assert len(frame) == 0
    assert "rule_4_deduction" in frame.columns and "casualty_reason" in frame.columns
    assert rf.facts_from_frame(frame) == {}


# ──────────────────────────── malformed documents ────────────────────────────
@pytest.mark.parametrize(
    "payload",
    [None, {}, {"props": {}}, {"props": {"pageProps": {}}}, "not a dict", []],
)
def test_non_race_documents_are_skipped_not_raised(payload):
    assert rf.parse_race_document(payload) is None


def test_document_without_a_course_or_time_is_skipped():
    doc = _doc([_ride("A", position=1)])
    doc["props"]["pageProps"]["race"]["race_summary"]["course_name"] = ""
    assert rf.parse_race_document(doc) is None


def test_unreadable_file_returns_none(tmp_path):
    path = tmp_path / "broken.html.gz"
    path.write_bytes(b"not gzip at all")
    assert rf.parse_raw_file(str(path)) is None


def test_parse_raw_file_reads_a_real_gzip(tmp_path):
    doc = _doc([_ride("A", position=1)])
    html = (
        '<html><script id="__NEXT_DATA__" type="application/json">'
        + json.dumps(doc)
        + "</script></html>"
    )
    path = tmp_path / "race.html.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(html)
    facts = rf.parse_raw_file(str(path))
    assert facts is not None
    # 19:20 in the document, 20:20 in the key: the archive clock is UTC and the
    # key is local racing time, and 2026-07-20 is inside BST. See
    # test_the_archive_clock_is_read_as_utc_and_keyed_locally.
    assert facts.race_key == "windsor|2026-07-20T20:20"
    assert facts.source_path == str(path)


def test_missing_day_yields_no_paths(tmp_path):
    assert rf.iter_day_paths("2026-07-20", root=str(tmp_path)) == []
    assert rf.extract_window("2026-07-20", "2026-07-22", root=str(tmp_path)) == []


# ─────────────────────────── the real archive ────────────────────────────────
@needs_archive
def test_archive_window_parses_and_reports_plausible_coverage():
    facts = rf.extract_window("2026-06-01", "2026-06-30")
    if not facts:
        pytest.skip("June 2026 not present in the archive")
    summary = rf.summarise(facts)
    assert summary["n_races"] > 500
    # Rates measured over the full archive; these bounds catch a parser
    # regression (e.g. re-reading position 0 as a real finish) without pinning
    # the data itself.
    assert 30.0 < summary["pct_races_with_non_runner"] < 75.0
    assert 0.5 < summary["pct_races_with_rule_4"] < 8.0
    assert 0.0 < summary["pct_races_with_dead_heat"] < 3.0
    assert 1.0 < summary["pct_runners_no_finish_position"] < 10.0


@needs_archive
def test_archive_non_finishers_all_carry_a_casualty_reason():
    facts = rf.extract_window("2026-06-20", "2026-06-30")
    if not facts:
        pytest.skip("late June 2026 not present in the archive")
    unexplained = [
        r.horse_name
        for f in facts
        for r in f.runners.values()
        if r.ran and not r.completed and not r.casualty_reason
    ]
    assert not unexplained, f"non-finishers without a reason: {unexplained[:5]}"
