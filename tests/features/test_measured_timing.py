"""Step 11: the measured-timing ingestion contract.

Covers the unit parsers, the INCREMENTAL margin convention, non-finisher and
missingness handling, publication-timestamp eligibility (a future-dated card
must contribute nothing) and the join key's agreement with
``execution.race_facts``.
"""
import gzip
import json
import os

import pandas as pd
import pytest

from execution import race_facts
from features import measured_timing as mt


# ───────────────────────────── unit parsers ─────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("2m 4f 56y", 2 * 1760 + 4 * 220 + 56),
    ("3m", 5280.0),
    ("5f", 1100.0),
    ("1m 30y", 1790.0),
    ("2m 7f 191y", 2 * 1760 + 7 * 220 + 191),
    ("", None),
    (None, None),
    ("nonsense", None),
])
def test_parse_distance_yards(text, expected):
    assert mt.parse_distance_yards(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("5m 14.84s", 314.84),
    ("58.31s", 58.31),
    ("1m 0.00s", 60.0),
    ("", None),
    (None, None),
    ("abc", None),
])
def test_parse_time_seconds(text, expected):
    got = mt.parse_time_seconds(text)
    assert got is None if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("text,expected", [
    ("5", 5.0),
    ("4 ¾", 4.75),
    ("½", 0.5),
    ("1 ¼", 1.25),
    ("1/2", 0.5),
    ("nk", 0.3),
    ("hd", 0.2),
    ("sh", 0.1),
    ("nse", 0.05),
    ("dh", 0.0),
    ("DH", 0.0),
    ("", None),
    (None, None),
    ("???", None),
])
def test_parse_margin_lengths(text, expected):
    got = mt.parse_margin_lengths(text)
    assert got is None if expected is None else got == pytest.approx(expected)


@pytest.mark.parametrize("name,expected", [
    ("Novices' Handicap Chase (GBB Race)", "chase"),
    ("Steeple Chase Cup", "chase"),
    ("Maiden Hurdle (GBB Race)", "hurdle"),
    ("Standard Open National Hunt Flat Race", "nhf"),
    ("Racing Post Bumper", "nhf"),
    ("Sprint Handicap", "flat"),
    ("", "flat"),
    (None, "flat"),
])
def test_classify_race_type(name, expected):
    """A 2m hurdle and a 2m flat race are run at different speeds, so the par
    must know which it is."""
    assert mt.classify_race_type(name) == expected


def test_chase_beats_hurdle_when_a_title_mentions_both():
    assert mt.classify_race_type("Handicap Chase (formerly a Hurdle)") == "chase"


def test_unknown_margin_token_is_missing_not_zero():
    """A token we cannot read must not silently become 'beaten by nothing'."""
    assert mt.parse_margin_lengths("unreadable") is None


# ───────────────────────────── document parsing ─────────────────────────────
def _doc(*, winning_time="1m 40.00s", distance="1m", rides=None, day="2024-03-01",
         clock="14:00", course="Kempton", surface="POLYTRACK"):
    return {"props": {"pageProps": {"race": {
        "race_summary": {
            "course_name": course, "date": day, "time": clock,
            "winning_time": winning_time, "distance": distance,
            "course_surface": {"surface": surface}, "going": "Standard",
            "race_class": "4", "has_handicap": True,
        },
        "rides": rides if rides is not None else [],
    }}}}


def _ride(name, position, finish_distance=None, status="RUNNER", casualty=None):
    ride = {"horse": {"name": name}, "finish_position": position,
            "ride_status": status, "casualty": casualty or {}}
    if finish_distance is not None:
        ride["finish_distance"] = finish_distance
    return ride


def test_margins_are_cumulated_because_the_feed_is_incremental():
    """``finish_distance`` is the gap to the horse IN FRONT, so beaten lengths
    behind the winner is the running sum down the finishing order."""
    doc = _doc(rides=[
        _ride("Alpha", 1),
        _ride("Bravo", 2, "2"),
        _ride("Charlie", 3, "1 ½"),
        _ride("Delta", 4, "nk"),
    ])
    timing = mt.parse_timing_document(doc)
    beaten = {r["horse_name"]: r["beaten_lengths"] for r in timing.runners}
    assert beaten["Alpha"] == 0.0
    assert beaten["Bravo"] == pytest.approx(2.0)
    assert beaten["Charlie"] == pytest.approx(3.5)
    assert beaten["Delta"] == pytest.approx(3.8)


def test_unreadable_margin_stops_the_cumulative_chain():
    """One unreadable gap makes every later runner's beaten distance unknown --
    it must go null rather than under-report the deficit."""
    doc = _doc(rides=[
        _ride("Alpha", 1),
        _ride("Bravo", 2, "???"),
        _ride("Charlie", 3, "1"),
    ])
    beaten = {r["horse_name"]: r["beaten_lengths"]
              for r in mt.parse_timing_document(doc).runners}
    assert beaten["Alpha"] == 0.0
    assert beaten["Bravo"] is None
    assert beaten["Charlie"] is None


def test_non_runner_is_excluded_and_non_finisher_has_null_time():
    doc = _doc(rides=[
        _ride("Alpha", 1),
        _ride("Bravo", 2, "3"),
        _ride("Ghost", 0, status="NONRUNNER"),
        _ride("Faller", 0, casualty={"reason": "Fell"}),
    ])
    timing = mt.parse_timing_document(doc)
    names = {r["horse_name"] for r in timing.runners}
    assert "Ghost" not in names, "a non-runner has no performance to measure"
    faller = next(r for r in timing.runners if r["horse_name"] == "Faller")
    assert faller["finished"] is False
    assert faller["beaten_lengths"] is None
    assert faller["casualty_reason"] == "Fell"

    frame = mt.to_runner_frame([timing])
    row = frame.loc[frame["horse_name"] == "Faller"].iloc[0]
    assert pd.isna(row["est_time_seconds"]) and pd.isna(row["est_speed_yps"])
    assert bool(row["finished"]) is False


def test_race_without_winning_time_is_unusable_and_emits_no_rows():
    timing = mt.parse_timing_document(_doc(winning_time="", rides=[_ride("Alpha", 1)]))
    assert timing.usable is False
    assert mt.to_runner_frame([timing]).empty


def test_future_card_contributes_no_timing():
    """Publication eligibility: a card that has not run carries no winning time
    and no finishing positions, so it can never leak a measured performance."""
    doc = _doc(winning_time=None, rides=[
        _ride("Alpha", 0), _ride("Bravo", 0),
    ])
    timing = mt.parse_timing_document(doc)
    assert timing.usable is False
    assert mt.to_runner_frame([timing]).empty


def test_implausible_speed_is_rejected():
    """A 1-mile race in 10 seconds is a parse fault, not a record."""
    timing = mt.parse_timing_document(_doc(winning_time="10.00s", distance="1m"))
    assert timing.usable is False


def test_available_from_is_the_local_off_time():
    timing = mt.parse_timing_document(_doc(day="2024-06-01", clock="13:00"))
    frame = mt.to_runner_frame([mt.parse_timing_document(
        _doc(day="2024-06-01", clock="13:00", rides=[_ride("Alpha", 1)]))])
    # June -> BST, so 13:00 UTC is 14:00 local.
    assert timing.off_time == "2024-06-01T14:00"
    assert frame["available_from"].iloc[0] == "2024-06-01T14:00"


def test_join_key_matches_race_facts():
    """The measured-timing key must be byte-identical to the settlement
    archive's, or the two views of the same race will never join."""
    doc = _doc(course="Newton Abbot", day="2024-01-05", clock="12:40")
    timing = mt.parse_timing_document(doc)
    assert timing.race_key == race_facts.race_facts_key("Newton Abbot", timing.off_time)


def test_estimated_time_uses_the_declared_lengths_per_second():
    doc = _doc(winning_time="100.00s", distance="1m",
               rides=[_ride("Alpha", 1), _ride("Bravo", 2, "5")])
    frame = mt.to_runner_frame([mt.parse_timing_document(doc)], lengths_per_second=5.0)
    bravo = frame.loc[frame["horse_name"] == "Bravo"].iloc[0]
    assert bravo["est_time_seconds"] == pytest.approx(101.0)
    assert bravo["lengths_per_second"] == 5.0
    slower = mt.to_runner_frame([mt.parse_timing_document(doc)], lengths_per_second=4.0)
    assert slower.loc[slower["horse_name"] == "Bravo", "est_time_seconds"].iloc[0] \
        == pytest.approx(101.25)


def test_winner_speed_is_exact_published_arithmetic():
    frame = mt.to_runner_frame([mt.parse_timing_document(
        _doc(winning_time="100.00s", distance="1m", rides=[_ride("Alpha", 1)]))])
    assert frame["winner_speed_yps"].iloc[0] == pytest.approx(17.6)


# ───────────────────────────── traversal ─────────────────────────────
def test_extract_window_reads_gzip_and_keeps_most_complete_duplicate(tmp_path):
    day_dir = tmp_path / mt.DEFAULT_SOURCE / "2024" / "2024-02-02"
    day_dir.mkdir(parents=True)

    def write(name, doc):
        html = ('<script id="__NEXT_DATA__" type="application/json">'
                + json.dumps(doc) + "</script>")
        with gzip.open(day_dir / name, "wt", encoding="utf-8") as fh:
            fh.write(html)

    sparse = _doc(day="2024-02-02", clock="15:00", rides=[_ride("Alpha", 1)])
    full = _doc(day="2024-02-02", clock="15:00",
                rides=[_ride("Alpha", 1), _ride("Bravo", 2, "2")])
    write("a.html.gz", sparse)
    write("b.html.gz", full)

    timings = mt.extract_timing_window("2024-02-02", "2024-02-02", root=str(tmp_path))
    assert len(timings) == 1, "same race, one record"
    assert len(timings[0].runners) == 2, "the more complete document wins"


def test_extract_window_on_missing_day_is_empty(tmp_path):
    assert mt.extract_timing_window("2024-02-02", "2024-02-03", root=str(tmp_path)) == []
