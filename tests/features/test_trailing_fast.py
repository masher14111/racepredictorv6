import pandas as pd
import pytest

from features import _trailing_fast


# --- _race_event_key ------------------------------------------------------

def test_race_event_key_prefers_race_uid():
    df = pd.DataFrame({
        "race_uid": ["Ascot|2026-01-01T14:00", "Ascot|2026-01-01T14:00"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "venue": ["Ascot", "Ascot"],
    })
    key = _trailing_fast._race_event_key(df)
    assert key[0] == key[1] == "Ascot|2026-01-01T14:00"


def test_race_event_key_falls_back_to_date_venue():
    df = pd.DataFrame({
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True),
        "venue": ["Ascot", "Ascot"],
    })
    key = _trailing_fast._race_event_key(df)
    assert key[0] != key[1]


# --- _dedupe_group_ids ------------------------------------------------------

def test_dedupe_group_ids_collapses_market_duplicate_rows():
    # Horse A: race R1 (WIN+PLACE rows), race R2 (WIN+PLACE rows) -> 2 real events.
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A", "A"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
    })
    keep, gid = _trailing_fast._dedupe_group_ids(df, ["horse_id"])
    assert keep.sum() == 2
    assert gid[0] == gid[1]
    assert gid[2] == gid[3]
    assert gid[0] != gid[2]
    # group ids for the kept rows are exactly 0..k-1 in first-occurrence order
    assert list(gid[keep]) == [0, 1]


def test_dedupe_group_ids_no_duplicates_keeps_everything():
    df = pd.DataFrame({
        "horse_id": ["A", "A"],
        "race_uid": ["R1", "R2"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-02-01"], utc=True),
    })
    keep, gid = _trailing_fast._dedupe_group_ids(df, ["horse_id"])
    assert keep.all()
    assert list(gid) == [0, 1]


# --- windowed_rates: market-duplicate collapse (step 04 / D25-D26) --------

def test_windowed_rates_market_duplicate_rows_count_one_prior_race():
    # Horse A: race1 (won, WIN+PLACE rows), race2 (WIN+PLACE rows, today).
    # Both of race2's rows must see EXACTLY one prior race (race1), not two.
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A", "A"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    }).reset_index(drop=True)
    win, plc, runs = _trailing_fast.windowed_rates(
        df, "horse_id", runs=20, months=12, places=3, strict=False, dropna=True)
    # race2's WIN row and PLACE row must agree, and each see 1 prior run (not 2)
    assert runs[2] == runs[3] == 1
    assert win[2] == win[3] == 1.0


def test_windowed_rates_market_sibling_does_not_leak_into_itself():
    # Horse A's WIN and PLACE row of the SAME race (only one race total) must
    # NOT count each other as a prior result for their own race.
    df = pd.DataFrame({
        "horse_id": ["A", "A"],
        "race_uid": ["R1", "R1"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "market_type": ["WIN", "PLACE"],
        "position": [1, 1],
    }).reset_index(drop=True)
    win, plc, runs = _trailing_fast.windowed_rates(
        df, "horse_id", runs=20, months=12, places=3, strict=False, dropna=True)
    assert runs[0] == 0 and runs[1] == 0
    import numpy as np
    assert np.isnan(win[0]) and np.isnan(win[1])


# --- windowed_mean: market-duplicate collapse ------------------------------

def test_windowed_mean_market_duplicate_rows_share_identical_result():
    df = pd.DataFrame({
        "horse_id": ["A", "A", "A", "A"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "value": [80.0, 80.0, None, None],
    }).reset_index(drop=True)
    out = _trailing_fast.windowed_mean(
        df, ["horse_id"], "value", runs=20, months=12, strict=True, dropna=True)
    assert out[2] == out[3] == 80.0


# --- windowed_win_rate_days: trainer/jockey day-windowed rate --------------

def test_windowed_win_rate_days_market_duplicate_counts_one_prior():
    df = pd.DataFrame({
        "trainer_id": ["T", "T", "T", "T"],
        "race_uid": ["R1", "R1", "R2", "R2"],
        "race_date": pd.to_datetime(
            ["2026-01-01", "2026-01-01", "2026-01-05", "2026-01-05"], utc=True),
        "market_type": ["WIN", "PLACE", "WIN", "PLACE"],
        "position": [1, 1, 2, 2],
    }).reset_index(drop=True)
    rate, cnt = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=14, min_runners=1)
    assert cnt[2] == cnt[3] == 1
    assert rate[2] == rate[3] == 1.0


def test_windowed_win_rate_days_market_sibling_does_not_leak_into_itself():
    df = pd.DataFrame({
        "trainer_id": ["T", "T"],
        "race_uid": ["R1", "R1"],
        "race_date": pd.to_datetime(["2026-01-01", "2026-01-01"], utc=True),
        "market_type": ["WIN", "PLACE"],
        "position": [1, 1],
    }).reset_index(drop=True)
    rate, cnt = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=14, min_runners=1)
    assert cnt[0] == 0 and cnt[1] == 0
    import numpy as np
    assert np.isnan(rate[0]) and np.isnan(rate[1])


# --- step 09 audit A1 / D38: the collapse is market duplicates ONLY ---------
# A yard's several runners in ONE race are distinct results and must all count.


def _multi_runner_yard_frame():
    """Trainer T saddles 3 horses in race R1 (one wins), then has a runner in a
    later race R2. Every row is duplicated WIN/PLACE, exactly as features/fuse.py
    emits it."""
    rows = []
    for horse, pos in (("A", 5), ("B", 1), ("C", 4)):
        for market in ("WIN", "PLACE"):
            rows.append({"trainer_id": "T", "horse_id": horse, "race_uid": "R1",
                         "race_date": "2026-01-01", "market_type": market,
                         "position": pos})
    for market in ("WIN", "PLACE"):
        rows.append({"trainer_id": "T", "horse_id": "D", "race_uid": "R2",
                     "race_date": "2026-01-05", "market_type": market,
                     "position": 2})
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    return df.reset_index(drop=True)


def test_dedupe_keeps_distinct_runners_of_one_trainer_in_one_race():
    df = _multi_runner_yard_frame()
    keep, gid = _trailing_fast._dedupe_group_ids(df, ["trainer_id"])
    # 4 real runner-events (A, B, C in R1; D in R2) — not 2 races.
    assert keep.sum() == 4
    # the two market rows of the same runner-race still share one group
    assert gid[0] == gid[1] and gid[2] == gid[3] and gid[4] == gid[5]
    # different horses in the SAME race are different groups
    assert len({gid[0], gid[2], gid[4]}) == 3


def test_trainer_day_window_counts_every_runner_not_one_per_race():
    df = _multi_runner_yard_frame()
    rate, cnt = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=14, min_runners=1)
    # R2's rows look back at R1: 3 runners, 1 winner -> 1/3, not 1/1 or 0/1.
    assert cnt[6] == cnt[7] == 3
    assert rate[6] == rate[7] == pytest.approx(1.0 / 3.0)


def test_trainer_trailing_rate_counts_every_runner_not_one_per_race():
    df = _multi_runner_yard_frame()
    win, _plc, runs = _trailing_fast.windowed_rates(
        df, ["trainer_id"], runs=20, months=12, places=3, strict=False, dropna=True)
    assert runs[6] == runs[7] == 3
    assert win[6] == win[7] == pytest.approx(1.0 / 3.0)


def test_trainer_trailing_rate_is_invariant_to_row_order():
    """The pre-fix collapse kept whichever of a yard's runners appeared first,
    so merely reordering the rows changed the feature value."""
    df = _multi_runner_yard_frame()
    shuffled = df.iloc[[5, 1, 6, 3, 0, 7, 2, 4]].reset_index(drop=True)
    win_a, _, runs_a = _trailing_fast.windowed_rates(
        df, ["trainer_id"], runs=20, months=12, places=3, strict=False, dropna=True)
    win_b, _, runs_b = _trailing_fast.windowed_rates(
        shuffled, ["trainer_id"], runs=20, months=12, places=3,
        strict=False, dropna=True)
    # compare on the R2 rows, whichever positions they now occupy
    a = {(h, m): (win_a[i], runs_a[i]) for i, (h, m)
         in enumerate(zip(df["horse_id"], df["market_type"]))}
    b = {(h, m): (win_b[i], runs_b[i]) for i, (h, m)
         in enumerate(zip(shuffled["horse_id"], shuffled["market_type"]))}
    assert a[("D", "WIN")] == b[("D", "WIN")]
    assert a[("D", "PLACE")] == b[("D", "PLACE")]


def test_jt_combo_keeps_distinct_runners_of_one_pairing():
    """engine.add_combo_win_rate keys on (jockey_id, trainer_id) — the same
    over-collapse applied there before the runner component was added."""
    df = _multi_runner_yard_frame().assign(jockey_id="J")
    keep, _gid = _trailing_fast._dedupe_group_ids(df, ["jockey_id", "trainer_id"])
    assert keep.sum() == 4


def test_horse_keyed_dedupe_is_unchanged_by_the_runner_component():
    df = _multi_runner_yard_frame()
    keep, gid = _trailing_fast._dedupe_group_ids(df, ["horse_id"])
    assert keep.sum() == 4
    assert gid[0] == gid[1]


def test_null_runner_ids_are_never_merged_into_one_fake_runner():
    df = _multi_runner_yard_frame()
    df.loc[[0, 1, 2, 3], "horse_id"] = None
    keep, _gid = _trailing_fast._dedupe_group_ids(df, ["trainer_id"])
    # two null-id runners in R1 stay two separate runner-events (4 rows -> 4
    # groups, since a null identity cannot prove two rows are the same runner)
    assert keep.sum() == 6


# --- step 09 audit F1 / D39: the prior cut is the OFF-TIME, not the row order --


def _same_day_card(order):
    """Trainer T has a 13:30 runner and a 20:20 runner on ONE date. ``order``
    chooses which appears first in the frame — the feature must not care."""
    rows = {
        "early": {"trainer_id": "T", "horse_id": "A", "jockey_id": "J",
                  "race_uid": "Ascot|2026-03-01T13:30", "race_date": "2026-03-01",
                  "market_type": "WIN", "position": 4},
        "late": {"trainer_id": "T", "horse_id": "B", "jockey_id": "J",
                 "race_uid": "Ascot|2026-03-01T20:20", "race_date": "2026-03-01",
                 "market_type": "WIN", "position": 1},
    }
    df = pd.DataFrame([rows[k] for k in order])
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    return df.reset_index(drop=True)


@pytest.mark.parametrize("order", [("early", "late"), ("late", "early")])
def test_prior_cut_uses_off_time_not_row_position(order):
    df = _same_day_card(order)
    win, _plc, runs = _trailing_fast.windowed_rates(
        df, ["trainer_id"], runs=20, months=12, places=3, strict=False, dropna=True)
    pos = {name: i for i, name in enumerate(order)}
    # the 13:30 runner cannot see the 20:20 result — it had not been run yet
    assert runs[pos["early"]] == 0
    # the 20:20 runner legitimately sees the 13:30 result (1 run, 0 wins)
    assert runs[pos["late"]] == 1
    assert win[pos["late"]] == 0.0


@pytest.mark.parametrize("order", [("early", "late"), ("late", "early")])
def test_day_window_prior_cut_uses_off_time_not_row_position(order):
    df = _same_day_card(order)
    rate, cnt = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=14, min_runners=1)
    pos = {name: i for i, name in enumerate(order)}
    assert cnt[pos["early"]] == 0
    assert cnt[pos["late"]] == 1
    assert rate[pos["late"]] == 0.0


def test_event_without_an_off_time_sees_and_is_seen_by_no_same_day_prior():
    """An untimed event is unavailable information in both directions: it may
    have run after any same-day race, so neither side may count the other."""
    df = pd.DataFrame([
        {"trainer_id": "T", "horse_id": "A", "race_uid": "Ascot|2026-03-01",
         "race_date": "2026-03-01", "market_type": "WIN", "position": 1},
        {"trainer_id": "T", "horse_id": "B", "race_uid": "Ascot|2026-03-01T20:20",
         "race_date": "2026-03-01", "market_type": "WIN", "position": 1},
    ])
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    _win, _plc, runs = _trailing_fast.windowed_rates(
        df, ["trainer_id"], runs=20, months=12, places=3, strict=False, dropna=True)
    assert list(runs) == [0, 0]


def test_run_cap_selects_the_same_priors_under_a_row_shuffle():
    """The lookback_runs tail cap used to cut into a block of same-date rows
    whose internal order was arbitrary, so the value moved on a pure shuffle."""
    rows = []
    for hour in range(10, 22):                    # 12 races on one date
        rows.append({"trainer_id": "T", "horse_id": f"H{hour}",
                     "race_uid": f"Ascot|2026-03-01T{hour:02d}:00",
                     "race_date": "2026-03-01", "market_type": "WIN",
                     "position": 1 if hour % 2 else 6})
    rows.append({"trainer_id": "T", "horse_id": "Z",
                 "race_uid": "Ascot|2026-03-02T14:00", "race_date": "2026-03-02",
                 "market_type": "WIN", "position": 3})
    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True)
    df = df.reset_index(drop=True)
    shuffled = df.iloc[list(range(len(df) - 1, -1, -1))].reset_index(drop=True)
    kw = dict(runs=5, months=12, places=3, strict=False, dropna=True)
    win_a, _, runs_a = _trailing_fast.windowed_rates(df, ["trainer_id"], **kw)
    win_b, _, runs_b = _trailing_fast.windowed_rates(shuffled, ["trainer_id"], **kw)
    a = dict(zip(df["horse_id"], zip(win_a, runs_a)))
    b = dict(zip(shuffled["horse_id"], zip(win_b, runs_b)))
    assert a["Z"][1] == 5                                  # the cap really bites
    assert a["Z"][0] == pytest.approx(3.0 / 5.0)
    assert a.keys() == b.keys()
    for horse, (rate, n) in a.items():
        assert b[horse][1] == n
        assert b[horse][0] == pytest.approx(rate, nan_ok=True)


# --- step 09 audit F7: the day window must be in the frame's own time unit ----


def _paced_frame(unit):
    """Trainer T: a result 10 days back, another 20 days back, then today."""
    rows = [("2026-02-09", "H1", 1), ("2026-02-19", "H2", 1),
            ("2026-03-01", "H3", None)]
    df = pd.DataFrame([
        {"trainer_id": "T", "horse_id": h, "race_uid": f"Ascot|{d}T14:00",
         "race_date": d, "market_type": "WIN", "position": p}
        for d, h, p in rows])
    df["race_date"] = pd.to_datetime(df["race_date"], utc=True).dt.as_unit(unit)
    return df.reset_index(drop=True)


@pytest.mark.parametrize("unit", ["us", "ns", "ms", "s"])
def test_day_window_length_is_independent_of_datetime_resolution(unit):
    """A hard-coded nanoseconds-per-day constant made window_days 1,000x too
    long on a microsecond frame, so the 14-day "hot" window silently became a
    14,000-day all-history window."""
    df = _paced_frame(unit)
    _rate, cnt = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=14, min_runners=1)
    # only the 10-day-old result is inside a real 14-day window; the 20-day-old is not
    assert cnt[2] == 1
    _rate90, cnt90 = _trailing_fast.windowed_win_rate_days(
        df, "trainer_id", window_days=90, min_runners=1)
    assert cnt90[2] == 2
