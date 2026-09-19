"""The live-market audit must evidence *why* races were rejected.

A report that only counts VALID/INVALID cannot be used to diagnose ingestion
contamination, so these tests pin the reason and example reporting.
"""
import json

import pandas as pd

from scripts.audit_live_markets import (
    distribution,
    load_rows,
    race_audit,
    rejected_races,
    rejection_reason_counts,
    source_summary,
    suspicious_selections,
)
from utils.market_validation import INVALID, VALID


def _row(**overrides):
    row = {
        "source": "livescorebet",
        "race_id": "r1",
        "race_time": "2026-07-25T13:00:00+01:00",
        "venue": "Ascot",
        "market_type": "WIN",
        "market_id": "m1",
        "market_name": "To win",
        "selection_id": "s1",
        "horse_name": "Alpha",
        "odds_decimal": 2.5,
        "validation_status": VALID,
        "validation_reasons": "[]",
    }
    row.update(overrides)
    return row


def _clean_race(source="livescorebet", race_id="r1"):
    return [
        _row(source=source, race_id=race_id, selection_id="s1",
             horse_name="Alpha", odds_decimal=3.0),
        _row(source=source, race_id=race_id, selection_id="s2",
             horse_name="Bravo", odds_decimal=3.2),
        _row(source=source, race_id=race_id, selection_id="s3",
             horse_name="Charlie", odds_decimal=3.4),
    ]


def _rejected_race(reasons, source="boylesports", race_id="bad1", **overrides):
    payload = json.dumps(reasons, separators=(",", ":"))
    return [
        _row(source=source, race_id=race_id, selection_id=f"x{i}",
             horse_name=name, validation_status=INVALID,
             validation_reasons=payload, **overrides)
        for i, name in enumerate(("Delta", "Echo"), start=1)
    ]


def test_audit_carries_rejection_reasons_per_race():
    rows = pd.DataFrame(
        _clean_race() + _rejected_race(["extreme_booksum:9.363000"])
    )
    audit = race_audit(rows)

    assert set(audit["source"]) == {"livescorebet", "boylesports"}
    good = audit[audit["source"].eq("livescorebet")].iloc[0]
    assert good["validation_status"] == VALID
    assert good["validation_reasons"] == ""

    bad = audit[audit["source"].eq("boylesports")].iloc[0]
    assert bad["validation_status"] == INVALID
    assert bad["validation_reasons"] == "extreme_booksum:9.363000"


def test_multiple_reasons_are_merged_and_deduplicated():
    rows = pd.DataFrame(
        _rejected_race([
            "duplicate_selection_id",
            "implausible_field_size:60",
            "duplicate_selection_id",
        ])
    )
    reasons = race_audit(rows).iloc[0]["validation_reasons"].split(";")

    assert reasons == ["duplicate_selection_id", "implausible_field_size:60"]


def test_rejected_races_excludes_valid_races():
    rows = pd.DataFrame(_clean_race() + _rejected_race(["extreme_booksum:none"]))
    bad = rejected_races(race_audit(rows))

    assert list(bad["race_id"]) == ["bad1"]
    assert "livescorebet" not in set(bad["source"])


def test_reason_counts_group_by_family_not_measured_value():
    """extreme_booksum:9.36 and extreme_booksum:none are one reason family."""
    rows = pd.DataFrame(
        _rejected_race(["extreme_booksum:9.363000"], race_id="bad1")
        + _rejected_race(["extreme_booksum:none"], race_id="bad2")
        + _rejected_race(["duplicate_selection_id"], race_id="bad3")
    )
    counts = rejection_reason_counts(race_audit(rows))
    by_reason = dict(zip(counts["reason"], counts["races"]))

    assert by_reason == {"extreme_booksum": 2, "duplicate_selection_id": 1}


def test_reason_counts_empty_when_nothing_rejected():
    counts = rejection_reason_counts(race_audit(pd.DataFrame(_clean_race())))

    assert counts.empty
    assert list(counts.columns) == ["source", "reason", "races"]


def test_suspicious_selection_examples_flag_specials():
    rows = pd.DataFrame(
        _clean_race()
        + [_row(race_id="r2", selection_id="p1",
                horse_name="Alpha & Bravo Both To Finish In The Top 3")]
        + [_row(race_id="r3", selection_id="p2",
                horse_name="Delta by 2 Lengths or more")]
    )
    suspect = suspicious_selections(rows)

    assert set(suspect["horse_name"]) == {
        "Alpha & Bravo Both To Finish In The Top 3",
        "Delta by 2 Lengths or more",
    }
    assert set(suspect["flag"]) == {"special_selection_pattern"}


def test_suspicious_selections_flag_non_win_market_rows():
    rows = pd.DataFrame(
        _clean_race()
        + [_row(race_id="r9", selection_id="w1", horse_name="Foxtrot",
                market_type="PLACE", market_name="To be placed")]
    )
    suspect = suspicious_selections(rows)

    assert list(suspect["flag"]) == ["non_primary_win_market_type"]
    assert list(suspect["horse_name"]) == ["Foxtrot"]


def test_suspicious_selections_are_capped_per_source():
    rows = pd.DataFrame([
        _row(source="livescorebet", race_id=f"r{i}", selection_id=f"s{i}",
             horse_name=f"Runner {i} Both To Finish In The Top 3")
        for i in range(8)
    ] + [
        _row(source="boylesports", race_id=f"b{i}", selection_id=f"c{i}",
             horse_name=f"Horse {i} Both To Finish In The Top 3")
        for i in range(8)
    ])
    suspect = suspicious_selections(rows, limit=3)

    assert suspect.groupby("source").size().to_dict() == {
        "boylesports": 3, "livescorebet": 3,
    }


def test_clean_rows_produce_no_suspicious_examples():
    suspect = suspicious_selections(pd.DataFrame(_clean_race()))

    assert suspect.empty


def test_distributions_and_summary_cover_valid_races_only():
    rows = pd.DataFrame(_clean_race() + _rejected_race(["extreme_booksum:none"]))
    audit = race_audit(rows)

    runners = distribution(audit, "runners")
    assert list(runners.index) == ["livescorebet"]
    assert runners.loc["livescorebet", "50%"] == 3
    for percentile in ("10%", "25%", "50%", "75%", "90%"):
        assert percentile in runners.columns

    booksum = distribution(audit, "booksum")
    assert list(booksum.index) == ["livescorebet"]
    assert source_summary(audit).index.tolist() == ["livescorebet"]


def test_empty_inputs_do_not_crash_the_report():
    empty = pd.DataFrame()
    audit = race_audit(empty)

    assert audit.empty
    assert rejected_races(audit).empty
    assert rejection_reason_counts(audit).empty
    assert suspicious_selections(empty).empty
    assert source_summary(audit).empty
    assert distribution(audit, "runners").empty


def test_paddy_cache_reasons_inherit_race_level_quarantine(tmp_path):
    """Paddy stores validation at race level; the audit must not lose it."""
    cache = {
        "fetched_at": "2026-07-25T12:00:00+01:00",
        "races": [{
            "race_id": "pp1",
            "race_time": "2026-07-25T12:58:00+01:00",
            "venue": "ENGHIEN",
            "validation_status": INVALID,
            "validation_reasons": '["extreme_booksum:none"]',
            "markets": [{
                "market_id": "m1",
                "market_name": "Win or Each Way",
                "market_type": "WIN",
                "selections": [
                    {"selection_id": "a", "horse_name": "Alpha",
                     "odds_decimal": None},
                    {"selection_id": "b", "horse_name": "Bravo",
                     "odds_decimal": None},
                ],
            }],
        }],
    }
    paddy = tmp_path / "paddy_power.json"
    paddy.write_text(json.dumps(cache), encoding="utf-8")

    audit = race_audit(load_rows(tmp_path / "missing.parquet", paddy))
    race = audit.iloc[0]

    assert race["source"] == "paddy_power"
    assert race["validation_status"] == INVALID
    assert race["validation_reasons"] == "extreme_booksum:none"
    assert rejection_reason_counts(audit)["reason"].tolist() == [
        "extreme_booksum"
    ]
