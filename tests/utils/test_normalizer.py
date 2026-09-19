import pandas as pd
import pytest

from utils.normalizer import (
    CANONICAL_COLUMNS, _canonical_id, _to_decimal, _to_dublin, _validate,
    _from_live_odds, _from_paddy_power, _from_betsp, _from_timeform,
    _dedupe, normalize,
)


def _valid_row():
    return {
        "race_date": "2026-06-13", "venue": "Sandown", "horse_name": "A Horse",
        "horse_id": "abc123", "market_type": "WIN", "odds_decimal": 6.5,
        "position": 1, "source": "betsp", "fetched_at": "2026-06-13T13:50:00+01:00",
    }


# --- _canonical_id ---

def test_canonical_id_deterministic_for_same_name():
    assert _canonical_id("Prince Of The Seas") == _canonical_id("Prince Of The Seas")


def test_canonical_id_differs_for_different_names():
    assert _canonical_id("Prince Of The Seas") != _canonical_id("Other Horse")


def test_canonical_id_null_for_empty_name():
    assert _canonical_id("") is None
    assert _canonical_id(None) is None


@pytest.mark.parametrize("text", ["nan", "NaN", " nan ", "None", "null", "n/a", "-"])
def test_canonical_id_null_for_a_string_that_spells_a_missing_value(text):
    """Step 09 audit F6: a float NaN that reached the store as the literal text
    'nan' hashed to an ordinary-looking id, pooling 501 distinct horses into one
    fake jockey and one fake trainer that then shared a trailing form history."""
    assert _canonical_id(text) is None


def test_canonical_id_still_resolves_a_real_name_that_merely_looks_odd():
    assert _canonical_id("Unknown Soldier") is not None
    assert _canonical_id("Nanette") is not None


# --- _to_decimal ---

def test_to_decimal_passes_through_decimal():
    assert _to_decimal(6.5) == 6.5


def test_to_decimal_parses_fractional():
    assert _to_decimal("5/2") == 3.5


def test_to_decimal_handles_evs_and_nr():
    assert _to_decimal("EVS") == 2.0
    assert _to_decimal("NR") is None
    assert _to_decimal(None) is None


# --- _to_dublin ---

def test_to_dublin_converts_utc_iso_to_dublin():
    # 12:50 UTC in summer = 13:50 Dublin (UTC+1)
    assert _to_dublin("2026-06-13T12:50:00+00:00").startswith("2026-06-13T13:50")


def test_to_dublin_handles_none():
    assert _to_dublin(None) is None


# --- _validate ---

def test_validate_keeps_valid_rows():
    df = pd.DataFrame([_valid_row()])
    out = _validate(df)
    assert len(out) == 1


def test_validate_drops_bad_odds():
    bad = _valid_row()
    bad["odds_decimal"] = 0.5  # must be > 1.0
    out = _validate(pd.DataFrame([bad]))
    assert len(out) == 0


def test_validate_drops_bad_position():
    bad = _valid_row()
    bad["position"] = 0  # must be >= 1
    out = _validate(pd.DataFrame([bad]))
    assert len(out) == 0


def test_validate_drops_bad_market_type():
    bad = _valid_row()
    bad["market_type"] = "EXOTIC"
    out = _validate(pd.DataFrame([bad]))
    assert len(out) == 0


def test_validate_allows_null_optionals():
    row = _valid_row()
    row["odds_decimal"] = None
    row["position"] = None
    out = _validate(pd.DataFrame([row]))
    assert len(out) == 1


def test_validate_preserves_trainer_and_jockey_names():
    """Task 06/07 regression: trainer_name/jockey_name must survive _validate.

    The pydantic contract is extra='ignore', so the vectorized validator must
    not silently drop these enriched columns from kept rows."""
    row = _valid_row()
    row["trainer_name"] = "T Cooper"
    row["jockey_name"] = "R Moore"
    out = _validate(pd.DataFrame([row]))
    assert len(out) == 1
    assert out.iloc[0]["trainer_name"] == "T Cooper"
    assert out.iloc[0]["jockey_name"] == "R Moore"


# --- adapters ---

def _assert_canonical(df):
    assert list(df.columns) == CANONICAL_COLUMNS


def test_from_live_odds_maps_to_canonical():
    native = pd.DataFrame([{
        "fetched_at": "2026-06-13T13:00:00+01:00", "source": "boylesports",
        "race_id": "r1", "race_time": "2026-06-13T12:50:00+00:00",
        "venue": "Sandown", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": 1.1, "horse_name": "A Horse",
        "odds_decimal": 2.0, "sp": 2.0, "is_low_odds": False, "currency": "GBP",
    }, {
        "fetched_at": "2026-06-13T13:00:00+01:00", "source": "boylesports",
        "race_id": "r1", "race_time": "2026-06-13T12:50:00+00:00",
        "venue": "Sandown", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": 1.1, "horse_name": "B Horse",
        "odds_decimal": 2.0, "sp": 2.0, "is_low_odds": False, "currency": "GBP",
    }])
    out = _from_live_odds(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["source"] == "boylesports"
    assert row["horse_id"] == _canonical_id("A Horse")
    assert row["odds_decimal"] == 2.0
    assert row["market_id"] == "m1"
    assert row["selection_id"] == "s1"
    assert row["validation_status"] == "VALID"
    # race_date derived from Dublin race_time (12:50 UTC -> 13:50 Dublin, same day)
    assert row["race_date"] == "2026-06-13"


def test_from_paddy_power_flattens_nested_cache():
    cache = {
        "fetched_at": "2026-06-13T13:00:00+01:00",
        "races": [{
            "race_id": "r1", "race_time": "2026-06-13T12:50:00+00:00",
            "venue": "Sandown",
            "markets": [{
                "market_id": "m1", "market_name": "Win or Each Way",
                "market_type": "WIN",
                "each_way_terms": {"places": 3, "reduction": 0.25},
                "selections": [
                    {"selection_id": "s1", "horse_name": "A Horse",
                     "sp": 2.0, "odds_decimal": 2.0},
                    {"selection_id": "s2", "horse_name": "B Horse",
                     "sp": 2.0, "odds_decimal": 2.0},
                ],
            }],
        }],
    }
    out = _from_paddy_power(cache)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["source"] == "paddy_power"
    assert row["market_type"] == "WIN"
    assert row["ew_places"] == 3
    assert row["src_horse_id"] == "s1"
    assert row["horse_id"] == _canonical_id("A Horse")


def test_from_betsp_maps_native_ids_to_src():
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_id": "bf99",
        "horse_name": "A Horse", "jockey_id": "j1", "trainer_id": "t1",
        "odds_finish": 6.0, "position": 1, "win_lose": 1, "going": "Good",
        "distance": "1m2f", "market_type": "WIN", "region": "uk",
        "morningwap": 6.2, "ppwap": 6.1, "result_source": "racing_post",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_betsp(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["source"] == "betsp"
    assert row["src_horse_id"] == "bf99"
    assert row["horse_id"] == _canonical_id("A Horse")
    assert row["position"] == 1


def test_from_betsp_recovers_off_time_into_race_time():
    """The Betfair backbone stores the off-time in race_date; _from_betsp must
    capture it into race_time before race_date is floored, so the feature pipeline
    can rebuild a true per-race key (audit C3)."""
    native = pd.DataFrame([{
        "race_date": "2026-06-13T14:45:00+01:00", "venue": "Sandown",
        "horse_id": "bf99", "horse_name": "A Horse", "odds_finish": 6.0,
        "position": 1, "win_lose": 1, "going": "Good", "distance": "1m2f",
        "market_type": "WIN", "region": "uk",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_betsp(native)
    row = out.iloc[0]
    assert row["race_date"] == "2026-06-13"            # floored to date string
    assert row["race_time"] is not None
    assert "14:45" in str(row["race_time"])            # off-time preserved


def test_from_betsp_carries_trainer_name_and_derives_trainer_id():
    """trainer_name enriched by results sources must survive into unified, and
    trainer_id must be derived from it so trailing trainer/jt-combo features live."""
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_id": "bf99",
        "horse_name": "A Horse", "jockey_id": "j1", "trainer_id": "92",
        "trainer_name": "T Cooper", "odds_finish": 6.0, "position": 1,
        "win_lose": 1, "going": "Good", "distance": "1m2f", "market_type": "WIN",
        "region": "uk", "result_source": "sporting_life",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_betsp(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["trainer_name"] == "T Cooper"
    assert row["src_trainer_id"] == "92"          # native id preserved
    assert row["trainer_id"] == _canonical_id("T Cooper")  # derived, feature key


def test_from_betsp_carries_jockey_name_and_derives_jockey_id():
    """jockey_name enriched by results sources must survive into unified, and
    jockey_id must be derived from it so trailing jockey/jt-combo features live."""
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_id": "bf99",
        "horse_name": "A Horse", "jockey_id": "55", "jockey_name": "R Moore",
        "trainer_id": "92", "trainer_name": "T Cooper", "odds_finish": 6.0,
        "position": 1, "win_lose": 1, "going": "Good", "distance": "1m2f",
        "market_type": "WIN", "region": "uk", "result_source": "sporting_life",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_betsp(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["jockey_name"] == "R Moore"
    assert row["src_jockey_id"] == "55"          # native id preserved
    assert row["jockey_id"] == _canonical_id("R Moore")  # derived, feature key


def test_from_timeform_derives_ids_from_names():
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown",
        "race_time": "2026-06-13T13:50:00+01:00", "horse_name": "A Horse",
        "horse_id": "tf1", "jockey_name": "Jane Jock", "jockey_id": "jx",
        "trainer_name": "Tom Train", "trainer_id": "tx", "position": 1,
        "timeform_rating": 120, "race_class": 3, "going": "Good",
        "distance": "1m2f", "market_type": "WIN", "region": "uk",
        "source": "timeform", "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_timeform(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["source"] == "timeform"
    assert row["jockey_id"] == _canonical_id("Jane Jock")
    assert row["src_jockey_id"] == "jx"


def test_from_timeform_carries_declared_card_fields_and_native_race_id():
    """Stage 05: draw/weight/age/OR/equipment/colour/sex/runner_status survive the
    adapter, and the native per-event id lands in src_race_id (not the dead
    canonical race_id column, per DECISIONS.md D21)."""
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "York",
        "race_time": "2026-06-13T13:50:00+01:00", "horse_name": "Prince Of The Seas",
        "horse_id": "000000614450", "jockey_name": "Miss Megan Jordan",
        "jockey_id": "000000018452", "trainer_name": "David O'Meara",
        "trainer_id": "000000045008", "position": None,
        "market_type": "WIN", "region": "uk", "source": "timeform",
        "fetched_at": "2026-06-13T13:00:00+01:00",
        "draw": 5, "weight_lbs": 149, "age": 4, "official_rating": 89,
        "equipment": "t", "colour": "b", "sex": "g", "runner_status": "RUNNER",
        "timeform_race_id": "2026-06-13-62-1",
    }])
    out = _from_timeform(native)
    _assert_canonical(out)
    row = out.iloc[0]
    assert row["src_race_id"] == "2026-06-13-62-1"
    assert row["draw"] == 5
    assert row["weight_lbs"] == 149
    assert row["age"] == 4
    assert row["official_rating"] == 89
    assert row["equipment"] == "t"
    assert row["colour"] == "b" and row["sex"] == "g"
    assert row["runner_status"] == "RUNNER"


def test_from_timeform_defaults_market_type_to_win():
    native = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_name": "A Horse",
        "position": 1, "source": "timeform",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_timeform(native)
    assert out.iloc[0]["market_type"] == "WIN"


def test_adapter_coerces_timestamp_race_date_to_string():
    native = pd.DataFrame([{
        "race_date": pd.Timestamp("2026-06-13", tz="UTC"), "venue": "Sandown",
        "horse_name": "A Horse", "position": 1,
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out = _from_timeform(native)
    assert out.iloc[0]["race_date"] == "2026-06-13"


# --- dedupe + orchestration ---

def test_dedupe_keeps_latest_fetched_at():
    base = _valid_row()
    early = {**base, "fetched_at": "2026-06-13T13:00:00+01:00", "odds_decimal": 6.0}
    late = {**base, "fetched_at": "2026-06-13T13:30:00+01:00", "odds_decimal": 7.0}
    out = _dedupe(pd.DataFrame([early, late]))
    assert len(out) == 1
    assert out.iloc[0]["odds_decimal"] == 7.0


def test_normalize_end_to_end(tmp_path):
    live = pd.DataFrame([{
        "fetched_at": "2026-06-13T13:00:00+01:00", "source": "boylesports",
        "race_id": "r1", "race_time": "2026-06-13T12:50:00+00:00",
        "venue": "Sandown", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s1", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": 1.1, "horse_name": "A Horse",
        "odds_decimal": 2.0, "sp": 2.0, "is_low_odds": False, "currency": "GBP",
    }, {
        "fetched_at": "2026-06-13T13:00:00+01:00", "source": "boylesports",
        "race_id": "r1", "race_time": "2026-06-13T12:50:00+00:00",
        "venue": "Sandown", "market_type": "WIN", "market_id": "m1",
        "market_name": "Win or Each Way", "selection_id": "s2", "ew_places": 3,
        "ew_reduction": 0.25, "ew_margin": 1.1, "horse_name": "B Horse",
        "odds_decimal": 2.0, "sp": 2.0, "is_low_odds": False, "currency": "GBP",
    }])
    betsp = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_id": "bf99",
        "horse_name": "A Horse", "jockey_id": "j1", "trainer_id": "t1",
        "odds_finish": 6.0, "position": 1, "win_lose": 1, "going": "Good",
        "distance": "1m2f", "market_type": "WIN", "region": "uk",
        "morningwap": 6.2, "ppwap": 6.1, "result_source": "racing_post",
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    out_path = tmp_path / "unified_races.parquet"
    df = normalize(sources={"live_odds": live, "betsp": betsp},
                   write=True, output_path=str(out_path))
    # two live runners plus one historical source row
    assert len(df) == 3
    assert set(df["source"]) == {"boylesports", "betsp"}
    assert list(df.columns) == CANONICAL_COLUMNS
    written = pd.read_parquet(out_path)
    assert len(written) == 3


def test_write_interrupted_leaves_prior_unified_dataset_untouched(tmp_path):
    """Requirement 6: a crash mid-write must never leave data/unified_races.parquet
    empty or half-written. _write used to rmtree the partition dir in place
    before rewriting it; a kill between those two steps erased everything,
    including years/sources untouched by this run."""
    import pandas as pd
    from unittest.mock import patch

    def pair(race_id, venue, fetched_at, names):
        return [
            {
                "fetched_at": fetched_at, "source": "boylesports",
                "race_id": race_id, "race_time": "2026-06-13T12:50:00+00:00",
                "venue": venue, "region": "UK", "market_type": "WIN", "market_id": "m1",
                "market_name": "Win or Each Way", "selection_id": f"s{i}",
                "ew_places": 3, "ew_reduction": 0.25, "ew_margin": 1.1,
                "horse_name": name, "odds_decimal": 2.0, "sp": 2.0,
                "is_low_odds": False, "currency": "GBP",
            }
            for i, name in enumerate(names, start=1)
        ]

    live1 = pd.DataFrame(
        pair("r1", "Sandown", "2026-06-13T13:00:00+01:00", ["A Horse", "B Horse"])
    )
    out_path = tmp_path / "unified_races.parquet"
    normalize(sources={"live_odds": live1}, write=True, output_path=str(out_path))
    assert len(pd.read_parquet(out_path)) == 2

    live2 = pd.DataFrame(
        pair("r2", "Ascot", "2026-06-13T14:00:00+01:00", ["C Horse", "D Horse"])
    )

    real_to_parquet = pd.DataFrame.to_parquet

    def _crash_after_tmp_write(self, target, *a, **kw):
        real_to_parquet(self, target, *a, **kw)
        raise OSError("simulated crash mid-write")

    with patch.object(pd.DataFrame, "to_parquet", _crash_after_tmp_write):
        try:
            normalize(sources={"live_odds": live2}, write=True, output_path=str(out_path))
        except OSError:
            pass

    # The prior snapshot must still be readable and non-empty.
    survived = pd.read_parquet(out_path)
    assert len(survived) == 2
    assert set(survived["horse_name"]) == {"A Horse", "B Horse"}
    import os
    assert not os.path.isdir(f"{out_path}.tmp")
    assert not os.path.isdir(f"{out_path}.bak")


def test_write_replaces_prior_live_source_snapshot(tmp_path):
    def snapshot(names, fetched_at):
        return pd.DataFrame([
            {
                "fetched_at": fetched_at, "source": "paddy_power",
                "race_id": "r1", "race_time": "2026-06-13T13:00:00+01:00",
                "venue": "Sandown", "region": "UK", "market_type": "WIN",
                "market_id": "m1", "market_name": "Win or Each Way",
                "selection_id": f"s{index}", "horse_name": name,
                "odds_decimal": 2.0, "validation_status": "VALID",
            }
            for index, name in enumerate(names, start=1)
        ])

    out_path = tmp_path / "unified_races.parquet"
    normalize(
        sources={"live_odds": snapshot(
            ["Old Alpha", "Old Bravo"], "2026-06-13T12:00:00+01:00"
        )},
        write=True,
        output_path=str(out_path),
    )
    normalize(
        sources={"live_odds": snapshot(
            ["New Alpha", "New Bravo"], "2026-06-13T12:05:00+01:00"
        )},
        write=True,
        output_path=str(out_path),
    )

    written = pd.read_parquet(out_path)
    live = written[written["source"].eq("paddy_power")]
    assert set(live["horse_name"]) == {"New Alpha", "New Bravo"}


def test_mixed_source_snapshot_fresh_stale_unavailable(tmp_path):
    """Requirement 10: one bookmaker fresh, one degraded-to-cache (stale), and
    one that didn't report this cycle at all (unavailable) must each be
    handled independently by the same write, not lumped into one blanket
    "live snapshot" decision."""
    now = pd.Timestamp.now(tz="UTC")

    def pair(source, venue, race_id, fetched_at, names, stale=False):
        return [
            {
                "fetched_at": fetched_at, "source": source, "stale": stale,
                "race_id": race_id, "race_time": "2026-06-13T13:00:00+01:00",
                "venue": venue, "region": "UK", "market_type": "WIN",
                "market_id": "m1", "market_name": "Win or Each Way",
                "selection_id": f"s{i}", "horse_name": name,
                "odds_decimal": 2.0, "sp": 2.0,
            }
            for i, name in enumerate(names, start=1)
        ]

    out_path = tmp_path / "unified_races.parquet"

    # Cycle 1: all three sources present and fresh.
    cycle1 = pd.DataFrame(
        pair("livescorebet", "Sandown", "r_lsb",
             (now - pd.Timedelta(minutes=1)).isoformat(), ["LSB Old A", "LSB Old B"])
        + pair("boylesports", "Ascot", "r_bs",
               (now - pd.Timedelta(minutes=1)).isoformat(), ["BS Old A", "BS Old B"])
        + pair("paddy_power", "Newbury", "r_pp",
               (now - pd.Timedelta(minutes=2)).isoformat(),
               ["PP Retained A", "PP Retained B"])
    )
    normalize(sources={"live_odds": cycle1}, write=True, output_path=str(out_path))

    # Cycle 2: livescorebet refreshes with new runners (fresh); boylesports
    # degrades to its cache and is flagged stale; paddy_power is entirely
    # absent this cycle (proxy/network failure -- "unavailable").
    cycle2 = pd.DataFrame(
        pair("livescorebet", "Sandown", "r_lsb", now.isoformat(),
             ["LSB New A", "LSB New B"])
        + pair("boylesports", "Ascot", "r_bs",
               (now - pd.Timedelta(minutes=1)).isoformat(),
               ["BS Old A", "BS Old B"], stale=True)
    )
    normalize(sources={"live_odds": cycle2}, write=True, output_path=str(out_path))

    written = pd.read_parquet(out_path)

    # 1. Fresh source: new snapshot fully replaced the old one.
    lsb = written[written["source"].eq("livescorebet")]
    assert set(lsb["horse_name"]) == {"LSB New A", "LSB New B"}

    # 2. Degraded-to-cache source: rows survive the write (assembly never
    #    silently drops them) but carry stale=True for downstream gates.
    bs = written[written["source"].eq("boylesports")]
    assert set(bs["horse_name"]) == {"BS Old A", "BS Old B"}
    assert bs["stale"].all()

    # 3. Unavailable source: absent from cycle 2 entirely, so its last known
    #    snapshot from cycle 1 is retained untouched rather than erased just
    #    because the source didn't report this cycle.
    pp = written[written["source"].eq("paddy_power")]
    assert set(pp["horse_name"]) == {"PP Retained A", "PP Retained B"}

    # Assembly correctness alone isn't the whole story: the downstream
    # staleness gate must still independently exclude the flagged-stale
    # boylesports rows from anything fused for prediction/EV.
    from features.fuse import fuse_sources

    fused = fuse_sources(written)
    fused_sources = set(fused["source"]) if len(fused) else set()
    assert "boylesports" not in fused_sources
    assert "livescorebet" in fused_sources
    assert "paddy_power" in fused_sources


def test_normalize_no_write_returns_frame(tmp_path):
    betsp = pd.DataFrame([{
        "race_date": "2026-06-13", "venue": "Sandown", "horse_id": "bf99",
        "horse_name": "A Horse", "market_type": "WIN", "position": 1,
        "fetched_at": "2026-06-13T13:00:00+01:00",
    }])
    df = normalize(sources={"betsp": betsp}, write=False)
    assert len(df) == 1
    assert not (tmp_path / "unified_races.parquet").exists()


# ── UK/IRE-only filter ─────────────────────────────────────────────────────────

def _live_row(venue, region=None, source="livescorebet", date="2026-06-19",
              horse="A Horse", selection_id="s1"):
    row = {
        "race_id": f"r_{venue}", "market_id": f"m_{venue}",
        "market_name": "To win", "selection_id": selection_id,
        "race_date": date, "race_time": f"{date}T14:00:00+01:00",
        "venue": venue, "horse_id": f"id_{venue}_{selection_id}",
        "horse_name": horse, "market_type": "WIN", "odds_decimal": 2.0,
        "source": source, "fetched_at": "2026-06-19T10:00:00+01:00",
    }
    if region is not None:
        row["region"] = region
    return row


def test_filter_drops_foreign_live_keeps_uk_ire():
    betsp = pd.DataFrame([{
        "race_date": "2024-05-01", "venue": "Sandown", "horse_id": "h1",
        "horse_name": "Hist", "market_type": "WIN", "position": 1,
        "region": "UK", "fetched_at": "2024-05-01T13:00:00+01:00",
    }])
    live = pd.DataFrame([
        _live_row("Churchill Downs"),          # foreign, no region -> drop
        _live_row("Churchill Downs", horse="B Horse", selection_id="s2"),
        _live_row("Ascot", region="UK"),       # region-tagged -> keep
        _live_row("Ascot", region="UK", horse="B Horse", selection_id="s2"),
        _live_row("sandown-park", source="paddy_power"),  # known UK course -> keep
        _live_row("sandown-park", source="paddy_power", horse="B Horse",
                  selection_id="s2"),
    ])
    df = normalize(sources={"betsp": betsp, "live_odds": live}, write=False)
    venues = set(df["venue"])
    assert "Churchill Downs" not in venues
    assert {"Ascot", "sandown-park", "Sandown"}.issubset(venues | {"Sandown"})
    assert "Ascot" in venues and "sandown-park" in venues


def test_filter_never_drops_historical_rows():
    betsp = pd.DataFrame([{
        "race_date": "2024-05-01", "venue": "Sandown", "horse_id": "h1",
        "horse_name": "Hist", "market_type": "WIN", "position": 1,
        "fetched_at": "2024-05-01T13:00:00+01:00",
    }])
    df = normalize(sources={"betsp": betsp}, write=False)
    assert len(df) == 1  # historical row kept even with no region tag
