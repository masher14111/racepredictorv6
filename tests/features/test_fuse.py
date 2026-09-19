import pandas as pd

from features.fuse import fuse_sources

_NOW = pd.Timestamp.now(tz="UTC")


def test_fuse_collapses_to_one_row_per_runner_race():
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "position": 1, "timeform_rating": None, "sp": 6.0},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "timeform", "position": None, "timeform_rating": 120, "sp": None},
    ])
    out = fuse_sources(df)
    assert len(out) == 1
    row = out.iloc[0]
    # union of signals: betsp position + sp, timeform rating
    assert row["position"] == 1
    assert row["timeform_rating"] == 120
    assert row["sp"] == 6.0


def test_fuse_source_priority_breaks_conflicts():
    # both sources supply going; timeform priority wins
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "going": "Good"},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "timeform", "going": "Soft"},
    ])
    out = fuse_sources(df)
    assert out.iloc[0]["going"] == "Soft"


def test_fuse_keeps_distinct_runners_separate():
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "position": 1},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h2",
         "source": "betsp", "position": 2},
    ])
    out = fuse_sources(df)
    assert len(out) == 2


def test_fuse_drops_entire_explicitly_invalid_live_race():
    df = pd.DataFrame([
        {
            "race_id": "r1", "race_date": "2026-06-13",
            "race_time": "2026-06-13T14:00:00+01:00", "venue": "Ascot",
            "horse_id": "h1", "source": "livescorebet",
            "market_type": "WIN", "validation_status": "INVALID",
        },
        {
            "race_id": "r1", "race_date": "2026-06-13",
            "race_time": "2026-06-13T14:00:00+01:00", "venue": "Ascot",
            "horse_id": "h2", "source": "livescorebet",
            "market_type": "WIN", "validation_status": "VALID",
        },
    ])
    assert fuse_sources(df).empty


def test_fuse_drops_live_row_flagged_explicitly_stale():
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "boylesports", "fetched_at": _NOW.isoformat(), "stale": True,
        },
    ])
    assert fuse_sources(df).empty


def test_fuse_drops_live_row_older_than_ttl():
    stale_time = _NOW - pd.Timedelta(seconds=901)  # max_age_seconds is 900
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "boylesports", "fetched_at": stale_time.isoformat(), "stale": False,
        },
    ])
    assert fuse_sources(df).empty


def test_fuse_keeps_fresh_live_row_within_ttl():
    fresh_time = _NOW - pd.Timedelta(seconds=60)
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "boylesports", "fetched_at": fresh_time.isoformat(), "stale": False,
        },
    ])
    out = fuse_sources(df)
    assert len(out) == 1


def test_fuse_keeps_fresh_live_row_when_batch_mixes_timestamp_precision():
    """Stage 05 regression: pandas' to_datetime infers ONE format from a
    column's first non-null value; mixing a whole-second fetched_at (e.g. a
    declared-card source written by Stage 05's timeform fetch) with a
    sub-second one (a live odds source) in the SAME fuse_sources() call used
    to make the differently-shaped row parse to NaT and get fail-closed as
    "unparseable -> too old", even though it was fresh."""
    fresh_subsecond = (_NOW - pd.Timedelta(seconds=5)).isoformat()
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "timeform", "fetched_at": _NOW.floor("s").isoformat(),
        },
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h2",
            "source": "boylesports", "fetched_at": fresh_subsecond, "stale": False,
        },
    ])
    out = fuse_sources(df)
    assert set(out["horse_id"]) == {"h1", "h2"}


def test_fuse_drops_live_row_with_unparseable_fetched_at():
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "boylesports", "fetched_at": "not-a-timestamp", "stale": False,
        },
    ])
    assert fuse_sources(df).empty


def test_fuse_does_not_gate_non_live_sources_on_staleness():
    stale_time = _NOW - pd.Timedelta(seconds=99999)
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
            "source": "timeform", "fetched_at": stale_time.isoformat(), "stale": False,
        },
    ])
    out = fuse_sources(df)
    assert len(out) == 1


# --- step 03: canonical WIN/PLACE market association -----------------------

def _win_place_rows():
    """One horse with a WIN row and a PLACE row for the same race, from two
    different sources — the normal shape (data/unified_races.parquet: 99.2% of
    runner-races carry both books)."""
    return [
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "WIN", "odds_decimal": 3.0,
         "sp": 3.2, "jockey_id": "j1", "position": 1},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "PLACE", "odds_decimal": 1.4,
         "sp": 1.5, "jockey_id": "j1", "position": 1},
    ]


def test_fuse_keeps_one_horse_in_win_and_place_as_two_rows():
    out = fuse_sources(pd.DataFrame(_win_place_rows()))
    assert len(out) == 2
    assert sorted(out["market_type"]) == ["PLACE", "WIN"]
    win = out[out["market_type"] == "WIN"].iloc[0]
    place = out[out["market_type"] == "PLACE"].iloc[0]
    assert win["odds_decimal"] == 3.0
    assert place["odds_decimal"] == 1.4
    # Result is a shared, market-independent fact — correctly on both rows.
    assert win["position"] == 1
    assert place["position"] == 1


def test_fuse_market_assignment_survives_source_row_reordering():
    rows = _win_place_rows()
    forward = fuse_sources(pd.DataFrame(rows))
    backward = fuse_sources(pd.DataFrame(list(reversed(rows))))
    shuffled = fuse_sources(pd.DataFrame([rows[1], rows[0]]))

    def _by_market(df):
        return {r["market_type"]: r["odds_decimal"] for _, r in df.iterrows()}

    expected = {"WIN": 3.0, "PLACE": 1.4}
    assert _by_market(forward) == expected
    assert _by_market(backward) == expected
    assert _by_market(shuffled) == expected


def test_fuse_place_price_never_lands_on_the_win_row():
    """No place price or terms may contaminate a win record, in either sort
    order — the WIN row's odds must always be the WIN book's own price."""
    rows = _win_place_rows()
    for ordering in (rows, list(reversed(rows))):
        out = fuse_sources(pd.DataFrame(ordering))
        win = out[out["market_type"] == "WIN"].iloc[0]
        place = out[out["market_type"] == "PLACE"].iloc[0]
        assert win["odds_decimal"] == 3.0
        assert win["sp"] == 3.2
        assert place["odds_decimal"] == 1.4
        assert place["sp"] == 1.5


def test_fuse_shared_attrs_backfill_across_markets_without_touching_price():
    """A runner's WIN row is missing jockey_id (that source pass didn't carry
    it); the sibling PLACE row has it. jockey_id (shared) backfills onto the
    WIN row; odds_decimal (market-specific) is never copied across."""
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "boylesports", "market_type": "WIN", "odds_decimal": 3.0,
         "jockey_id": None},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "PLACE", "odds_decimal": 1.4,
         "jockey_id": "j1"},
    ])
    out = fuse_sources(df)
    win = out[out["market_type"] == "WIN"].iloc[0]
    place = out[out["market_type"] == "PLACE"].iloc[0]
    assert win["jockey_id"] == "j1"       # backfilled from the PLACE row
    assert win["odds_decimal"] == 3.0     # own market price, not overwritten
    assert place["odds_decimal"] == 1.4   # PLACE price never took the WIN price


def test_fuse_conflicting_shared_attr_keeps_each_rows_own_value():
    """Two sources disagree on a nominally-shared field (e.g. a jockey change
    mid-card mis-recorded). Backfill only fills NULLS — it must never overwrite
    an already-populated value on either market row."""
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "boylesports", "market_type": "WIN", "jockey_id": "j_win"},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "PLACE", "jockey_id": "j_place"},
    ])
    out = fuse_sources(df)
    win = out[out["market_type"] == "WIN"].iloc[0]
    place = out[out["market_type"] == "PLACE"].iloc[0]
    assert win["jockey_id"] == "j_win"
    assert place["jockey_id"] == "j_place"


def test_fuse_partial_book_one_horse_win_only_another_both_markets():
    """A partial book: h1 has both markets, h2 only ever prices WIN. h2 must
    not gain a phantom PLACE row or lose its WIN row."""
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "WIN", "odds_decimal": 3.0},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h1",
         "source": "betsp", "market_type": "PLACE", "odds_decimal": 1.4},
        {"race_date": "2026-06-13", "venue": "Ascot", "horse_id": "h2",
         "source": "betsp", "market_type": "WIN", "odds_decimal": 8.0},
    ])
    out = fuse_sources(df)
    assert len(out) == 3
    h2 = out[out["horse_id"] == "h2"]
    assert len(h2) == 1
    assert h2.iloc[0]["market_type"] == "WIN"
    assert h2.iloc[0]["odds_decimal"] == 8.0


def test_fuse_repeated_races_and_venues_keep_markets_within_their_own_race():
    """Same venue, two races (different off-times) on the same date, each with
    its own WIN/PLACE pair, plus a same-named venue/date pairing at a second
    venue — no cross-race or cross-venue market bleed."""
    df = pd.DataFrame([
        {"race_date": "2026-06-13", "race_time": "2026-06-13T14:00:00+01:00",
         "venue": "Ascot", "horse_id": "h1", "source": "betsp",
         "market_type": "WIN", "odds_decimal": 2.0},
        {"race_date": "2026-06-13", "race_time": "2026-06-13T14:00:00+01:00",
         "venue": "Ascot", "horse_id": "h1", "source": "betsp",
         "market_type": "PLACE", "odds_decimal": 1.3},
        {"race_date": "2026-06-13", "race_time": "2026-06-13T15:00:00+01:00",
         "venue": "Ascot", "horse_id": "h1", "source": "betsp",
         "market_type": "WIN", "odds_decimal": 6.0},
        {"race_date": "2026-06-13", "race_time": "2026-06-13T15:00:00+01:00",
         "venue": "Ascot", "horse_id": "h1", "source": "betsp",
         "market_type": "PLACE", "odds_decimal": 2.1},
        {"race_date": "2026-06-13", "race_time": "2026-06-13T14:00:00+01:00",
         "venue": "Newbury", "horse_id": "h1", "source": "betsp",
         "market_type": "WIN", "odds_decimal": 4.5},
    ])
    out = fuse_sources(df)
    assert len(out) == 5
    ascot_1400_win = out[(out["venue"] == "Ascot") & (out["race_time"] == "2026-06-13T14:00:00+01:00")
                         & (out["market_type"] == "WIN")].iloc[0]
    ascot_1500_win = out[(out["venue"] == "Ascot") & (out["race_time"] == "2026-06-13T15:00:00+01:00")
                         & (out["market_type"] == "WIN")].iloc[0]
    newbury_win = out[out["venue"] == "Newbury"].iloc[0]
    assert ascot_1400_win["odds_decimal"] == 2.0
    assert ascot_1500_win["odds_decimal"] == 6.0
    assert newbury_win["odds_decimal"] == 4.5


def test_fuse_conflicting_race_id_does_not_merge_distinct_markets():
    """A stale/mismatched race_id (source drift) must not override the
    (date, venue, time, horse, market) identity fuse actually keys on."""
    df = pd.DataFrame([
        {"race_id": "shared_id", "race_date": "2026-06-13", "venue": "Ascot",
         "horse_id": "h1", "source": "betsp", "market_type": "WIN",
         "odds_decimal": 2.0},
        {"race_id": "shared_id", "race_date": "2026-06-13", "venue": "Ascot",
         "horse_id": "h1", "source": "betsp", "market_type": "PLACE",
         "odds_decimal": 1.3},
    ])
    out = fuse_sources(df)
    assert len(out) == 2
    win = out[out["market_type"] == "WIN"].iloc[0]
    place = out[out["market_type"] == "PLACE"].iloc[0]
    assert win["odds_decimal"] == 2.0
    assert place["odds_decimal"] == 1.3


def test_fuse_does_not_collapse_same_horse_across_two_race_times():
    df = pd.DataFrame([
        {
            "race_date": "2026-06-13", "race_time": "2026-06-13T14:00:00+01:00",
            "venue": "Ascot", "horse_id": "h1", "source": "betsp",
        },
        {
            "race_date": "2026-06-13", "race_time": "2026-06-13T15:00:00+01:00",
            "venue": "Ascot", "horse_id": "h1", "source": "betsp",
        },
    ])
    assert len(fuse_sources(df)) == 2
