"""Vectorized point-in-time trailing aggregates.

Replaces the old per-row nested scan (``for group: for row: re-slice & re-filter
all priors``) that was O(n^2) per entity and turned the full-history feature build
into a 6-hour job. For each row we need an aggregate over PRIOR rows of the same
entity, within a ``[date - months, date)`` window, capped to the most recent
``runs`` of them, counting only rows with a known finishing position.

Per entity group we sort by date once, build cumulative sums, and resolve every
row's window with two ``searchsorted`` lookups — O(m log m) per group, O(n log n)
overall. Semantics are preserved exactly, including:

  * ``strict`` date handling — engine stages exclude same-date priors
    (``date < cutoff``); the derive stage includes earlier same-date rows, but
    "earlier" now means an earlier real OFF-TIME, not an earlier row position
    (see below).
  * the run cap is applied to the windowed priors *before* the known-position
    filter, so non-finishers still consume one of the ``runs`` slots.

Off-time ordering (step 09 audit F1 / DECISIONS D39): ``race_date`` is a
date-only value, so the non-strict prior cut used to be POSITIONAL
(``end = arange(len(d))``) — every earlier-*positioned* row sharing the date
counted as a prior, and within a day the frame's row order is not the off-time
order. A 13:30 runner was therefore scored on its yard's 20:20 result, and the
value moved when the frame was merely reordered. ``_order_instants`` recovers
the real off-time from the race-event key (``race_uid`` = venue + off-time to
the minute) and the cut is made on that instant, so a prior must have actually
RUN before the race being scored. Ordering additionally breaks ties on the
event and runner identity (``_chrono_rank``) so the ``lookback_runs`` tail cap
selects the same priors regardless of row order. An event whose key carries no
off-time is treated as unavailable information in BOTH directions: it sees no
same-day priors, and no same-day row sees it.

Market-duplicate collapse (step 04 / DECISIONS D25-D26): ``features/fuse.py``
gives a runner one row per market it raced under (WIN + PLACE), so a single
real race can contribute TWO rows here. Every trailing/window aggregate must
count that race once, not once per market row, or every window here
double-counts run counts and, worse, can let a runner's own market-sibling row
(the SAME race) leak in as a spurious "prior" result. ``_dedupe_group_ids``
collapses rows onto one representative per (entity keys..., RUNNER, real race)
before any window is computed, then callers broadcast the deduplicated result
back onto every original row. The runner component is what keeps the collapse
to market duplicates only: without it a trainer-keyed window also deleted every
runner but one from a yard's multi-runner race (step 09 audit A1 / D38).
"""
import re

import numpy as np
import pandas as pd

_NAT = np.iinfo(np.int64).min
# race_uid's off-time slot is utils.text_norm.minute_key's "%Y-%m-%dT%H:%M".
_OFFTIME_RE = re.compile(r"T(\d{2}):(\d{2})")
# ``asi8`` returns the frame's OWN datetime resolution, which pandas 3 resolves
# to microseconds for a parsed date column — not nanoseconds. Any integer
# duration compared against it must therefore be built from the same unit
# (step 09 audit F7: a hard-coded ns-per-day made the trainer/jockey day
# windows 1,000x too long on a microsecond frame).
_TICKS_PER_DAY = {"s": 86_400, "ms": 86_400_000,
                  "us": 86_400_000_000, "ns": 86_400_000_000_000}


def _asi8(s) -> np.ndarray:
    """tz-aware datetime Series -> int64 ticks (UTC) in the series' own
    resolution. NaT -> iNaT (sorts first)."""
    return pd.DatetimeIndex(s).asi8


def _ticks_per_day(s) -> int:
    """Integer ticks in one day for the series' datetime resolution."""
    unit = getattr(pd.DatetimeIndex(s), "unit", "ns")
    return _TICKS_PER_DAY.get(unit, 86_400_000_000_000)


def _race_event_key(df: pd.DataFrame) -> np.ndarray:
    """Per-row key identifying the REAL race behind a row, collapsing a runner's
    WIN/PLACE market-duplicate rows (features/fuse.py) onto the same key.

    Prefers ``race_uid`` (venue + off-time, features/derive.py::add_race_key);
    falls back to ``race_date + venue`` when race_uid is absent, matching
    features/derive.py::_race_cols' own fallback."""
    n = len(df)
    if "race_uid" in df.columns:
        return df["race_uid"].astype(str).to_numpy()
    venue = df["venue"].astype(str).to_numpy() if "venue" in df.columns \
        else np.full(n, "", dtype=object)
    rd = df["race_date"].astype(str).to_numpy() if "race_date" in df.columns \
        else np.full(n, "", dtype=object)
    return np.array([f"{r}|{v}" for r, v in zip(rd, venue)], dtype=object)


def _event_time_of_day(event: np.ndarray, day_ticks: int) -> np.ndarray:
    """Off-time carried by each row's race-event key, in ticks since midnight.

    ``race_uid`` is ``venue|<minute_key>`` (features/derive.py::add_race_key),
    where the minute key is ``%Y-%m-%dT%H:%M`` whenever an off-time was
    recovered and a bare ``%Y-%m-%d`` when it was not. Returns ``-1`` for a key
    with no time component so callers can treat that race's position in the day
    as unknown rather than guessing midnight.

    The time is read from the EVENT key, not from a per-row ``race_time``, so
    every row of one race necessarily gets the same instant (market duplicates
    and a yard's several runners must never order against each other)."""
    sec_ticks = day_ticks // 86_400
    codes, uniques = pd.factorize(pd.Series(event, dtype=object), sort=False)
    tod = np.full(len(uniques), -1, dtype="int64")
    for i, key in enumerate(uniques):
        m = _OFFTIME_RE.search(str(key))
        if m:
            tod[i] = (int(m.group(1)) * 3600 + int(m.group(2)) * 60) * sec_ticks
    safe = np.maximum(codes, 0)
    return np.where(codes >= 0, tod[safe], -1).astype("int64")


def _order_instants(df: pd.DataFrame, event: np.ndarray):
    """(date, self, prior, ticks_per_day) — the real chronological instants used
    for the point-in-time cut (step 09 audit F1), in the frame's own resolution.

    ``self`` is when the row being scored runs; ``prior`` is when the row
    becomes an observable result for someone else. They differ only when the
    event key carries no off-time, where the two safe assumptions point in
    opposite directions: such a race must be treated as the EARLIEST possible
    slot when deciding what IT may see (so it counts no same-day priors) and as
    the LATEST possible slot when it is offered as a prior to another row (so
    nobody counts a race that may not have run yet). A row is therefore never
    its own prior, since ``self <= prior`` always."""
    day_ticks = _ticks_per_day(df["race_date"])
    d = _asi8(df["race_date"])
    tod = _event_time_of_day(event, day_ticks)
    known = tod >= 0
    nat = d == _NAT
    base = np.where(nat, 0, d)
    self_ns = np.where(nat, _NAT, base + np.where(known, tod, 0))
    prior_ns = np.where(nat, _NAT, base + np.where(known, tod, day_ticks - 1))
    return d, self_ns.astype("int64"), prior_ns.astype("int64"), day_ticks


def _chrono_rank(df: pd.DataFrame, event: np.ndarray,
                 prior_ns: np.ndarray) -> np.ndarray:
    """A deterministic total order over the frame: off-time instant first, then
    the event key, then the runner identity.

    Sorting on the instant alone leaves same-instant rows (a yard's several
    runners in one race, or two meetings off at the same minute) in whatever
    order the frame happened to arrive in, and the ``lookback_runs`` tail cap
    can cut inside that block — which made the feature depend on row order.
    Both tiebreak keys are content-derived (``factorize(sort=True)``), so the
    resulting order is a property of the data, not of the row order."""
    ev = pd.factorize(pd.Index(np.asarray(event, dtype=object)).astype(str),
                      sort=True)[0]
    col = "horse_id" if "horse_id" in df.columns else (
        "horse_name" if "horse_name" in df.columns else None)
    if col is None:
        rk = np.zeros(len(df), dtype="int64")
    else:
        rk = pd.factorize(pd.Index(df[col]).astype(str), sort=True)[0]
    order = np.lexsort((rk, ev, prior_ns))
    rank = np.empty(len(order), dtype="int64")
    rank[order] = np.arange(len(order), dtype="int64")
    return rank


def _runner_key(df: pd.DataFrame, keys) -> np.ndarray | None:
    """Per-row RUNNER identity used to keep the market-duplicate collapse from
    also swallowing an entity's genuinely distinct runners.

    The thing being collapsed is "one runner, one real race, two market rows".
    Keying the collapse on ``(entity keys..., race event)`` alone is only
    equivalent to that when the entity IS the runner (``keys`` contains
    ``horse_id``). For a TRAINER- or JOCKEY-keyed window it is not: a yard
    routinely saddles several different horses in one race, and collapsing on
    ``(trainer_id, race)`` deleted all but one of them from the yard's own
    trailing form — measured at 6.50% of real trainer runner-events on
    ``data/audit/06/training_refreshed.parquet`` (step 09 audit A1), and it made
    ``trainer_win_rate`` depend on the frame's row order, because the surviving
    representative is simply whichever row came first.

    Returns ``None`` when the frame carries no runner column at all (the caller
    then keeps the plain ``(keys..., event)`` behaviour). A null runner id is
    given a row-unique token rather than being merged with every other null —
    an unknown identity must never make two unrelated rows look like the same
    runner (same rule as ``models/split_utils.py``'s unknown-race handling).
    """
    col = "horse_id" if "horse_id" in df.columns else (
        "horse_name" if "horse_name" in df.columns else None)
    if col is None or col in keys:
        return None
    runner = df[col].astype(object).to_numpy().copy()
    na = pd.isna(runner)
    if na.any():
        pos = np.where(na)[0]
        runner[pos] = [f"__unknown_runner__{p}" for p in pos]
    return runner


def _dedupe_group_ids(df: pd.DataFrame, keys):
    """Row -> (keep_mask, group_id) collapsing a runner's same-real-race market
    duplicates onto one group id per (keys..., runner, race event).

    ``keep_mask`` selects one representative row per group, in FIRST-OCCURRENCE
    order, so ``group_id[keep_mask]`` is exactly ``0..k-1``. A caller that
    computes a per-group result aligned to ``df.loc[keep_mask]`` can broadcast
    it back onto every original row via ``result[group_id]``.
    """
    event = _race_event_key(df)
    cols = [df[k].to_numpy() for k in keys]
    runner = _runner_key(df, keys)
    if runner is not None:
        cols.append(runner)
    cols.append(event)
    combo = pd.MultiIndex.from_arrays(cols)
    gid, _ = pd.factorize(combo, sort=False)
    keep = ~pd.Series(gid).duplicated(keep="first").to_numpy()
    return keep, gid


def _bounds(d: np.ndarray, lower: np.ndarray, self_ns: np.ndarray,
            prior_ns: np.ndarray, runs: int, strict: bool):
    """For a chronologically sorted group, return the [start, end) window per row.

    end   = count of qualifying priors — strict: an earlier DATE; otherwise an
            earlier real OFF-TIME instant (step 09 audit F1: this used to be the
            row's POSITION, which counted races that had not yet run).
    start = max(first-in-time-window, end - runs)  -> the run cap as a suffix.
    """
    lo = np.searchsorted(d, lower, side="left")           # first date >= cutoff-window
    if strict:
        end = np.searchsorted(d, d, side="left")          # priors strictly earlier in date
    else:
        end = np.searchsorted(prior_ns, self_ns, side="left")   # priors that actually ran first
    start = np.maximum(lo, end - runs) if runs else lo
    start = np.minimum(start, end)
    return start, end


def _cum(x: np.ndarray) -> np.ndarray:
    """Prefix-sum with a leading 0 so range-sum is cum[end] - cum[start]."""
    out = np.zeros(len(x) + 1, dtype="float64")
    np.cumsum(x, dtype="float64", out=out[1:])
    return out


def _groups(df: pd.DataFrame, keys, dropna: bool):
    return df.groupby(keys, sort=False, dropna=dropna).indices


def _windowed_rates_core(df, keys, *, runs, months, places, strict, dropna):
    """windowed_rates' original algorithm, run on an already-deduplicated frame
    (one row per entity per real race)."""
    n = len(df)
    win = np.full(n, np.nan)
    plc = np.full(n, np.nan)
    runs_out = np.zeros(n, dtype="int64")
    if n == 0:
        return win, plc, runs_out

    event = _race_event_key(df)
    d_all, self_ns, prior_ns, _ = _order_instants(df, event)
    rank = _chrono_rank(df, event, prior_ns)
    lower_all = _asi8(df["race_date"] - pd.DateOffset(months=months))
    pos = pd.to_numeric(df["position"], errors="coerce").to_numpy(dtype="float64")
    contrib = ~np.isnan(pos)
    is_win = contrib & (pos == 1)
    is_plc = contrib & (pos <= places)

    for idx in _groups(df, keys, dropna).values():
        order = idx[np.argsort(rank[idx], kind="stable")]
        start, end = _bounds(d_all[order], lower_all[order], self_ns[order],
                             prior_ns[order], runs, strict)
        cc = _cum(contrib[order])
        cw = _cum(is_win[order])
        cp = _cum(is_plc[order])
        nn = cc[end] - cc[start]
        nz = nn > 0
        denom = np.where(nz, nn, 1.0)
        win[order] = np.where(nz, (cw[end] - cw[start]) / denom, np.nan)
        plc[order] = np.where(nz, (cp[end] - cp[start]) / denom, np.nan)
        runs_out[order] = nn.astype("int64")
    return win, plc, runs_out


def windowed_rates(df, keys, *, runs, months, places, strict, dropna):
    """Trailing win-rate, place-rate and run-count over the point-in-time window.

    Returns three numpy arrays (win_rate, place_rate, n_runs) aligned to df's
    positional order (df must carry a default RangeIndex). Rates are NaN where no
    qualifying prior runs exist; n_runs counts only known-position priors.

    A runner's WIN and PLACE row of the SAME race are collapsed to one prior
    event before windows are computed (see module docstring), then the result is
    broadcast back to both rows — they must see an identical set of strictly
    prior real races.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    n = len(df)
    win = np.full(n, np.nan)
    plc = np.full(n, np.nan)
    runs_out = np.zeros(n, dtype="int64")
    if n == 0 or "position" not in df.columns or "race_date" not in df.columns:
        return win, plc, runs_out

    keep, gid = _dedupe_group_ids(df, keys)
    if keep.all():
        return _windowed_rates_core(
            df, keys, runs=runs, months=months, places=places, strict=strict, dropna=dropna)

    sub = df.loc[keep].reset_index(drop=True)
    sub_win, sub_plc, sub_runs = _windowed_rates_core(
        sub, keys, runs=runs, months=months, places=places, strict=strict, dropna=dropna)
    return sub_win[gid], sub_plc[gid], sub_runs[gid]


def _windowed_mean_core(df, keys, value_col, *, runs, months, strict, dropna):
    """windowed_mean's original algorithm, run on an already-deduplicated frame."""
    n = len(df)
    out = np.full(n, np.nan)
    if n == 0:
        return out

    event = _race_event_key(df)
    d_all, self_ns, prior_ns, _ = _order_instants(df, event)
    rank = _chrono_rank(df, event, prior_ns)
    lower_all = _asi8(df["race_date"] - pd.DateOffset(months=months))
    val = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype="float64")
    has = ~np.isnan(val)
    val0 = np.where(has, val, 0.0)

    for idx in _groups(df, keys, dropna).values():
        order = idx[np.argsort(rank[idx], kind="stable")]
        start, end = _bounds(d_all[order], lower_all[order], self_ns[order],
                             prior_ns[order], runs, strict)
        csum = _cum(val0[order])
        ccnt = _cum(has[order])
        c = ccnt[end] - ccnt[start]
        nz = c > 0
        out[order] = np.where(nz, (csum[end] - csum[start]) / np.where(nz, c, 1.0), np.nan)
    return out


def windowed_mean(df, keys, value_col, *, runs, months, strict, dropna):
    """Trailing mean of ``value_col`` over the point-in-time window (NaN-skipping).

    Used for the leak-safe speed proxy. Returns one numpy array aligned to df's
    positional order; NaN where no non-null values fall in the window.

    Same market-duplicate collapse as ``windowed_rates``: a runner's WIN/PLACE
    row of the SAME race counts as one prior event, and both rows of that race
    get the identical trailing value.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    n = len(df)
    out = np.full(n, np.nan)
    if n == 0 or value_col not in df.columns or "race_date" not in df.columns:
        return out

    keep, gid = _dedupe_group_ids(df, keys)
    if keep.all():
        return _windowed_mean_core(
            df, keys, value_col, runs=runs, months=months, strict=strict, dropna=dropna)

    sub = df.loc[keep].reset_index(drop=True)
    sub_out = _windowed_mean_core(
        sub, keys, value_col, runs=runs, months=months, strict=strict, dropna=dropna)
    return sub_out[gid]


def _windowed_win_rate_days_core(df, key_col, window_days, min_runners):
    """Trailing win-rate over a strictly-prior CALENDAR-DAY window (day windows,
    not the month/run windows the rest of this module uses), on an
    already-deduplicated frame. Shared core for the trainer/jockey form-cycle
    features (features/_trainer_form.py, features/_jockey_form.py)."""
    n = len(df)
    rate = np.full(n, np.nan)
    cnt = np.zeros(n, dtype="int64")
    if n == 0:
        return rate, cnt

    event = _race_event_key(df)
    d_all, self_all, prior_all, day_ticks = _order_instants(df, event)
    rank = _chrono_rank(df, event, prior_all)
    pos = pd.to_numeric(df.get("position"), errors="coerce").to_numpy(dtype="float64")
    contrib = ~np.isnan(pos)
    is_win = contrib & (pos == 1)
    window_ns = np.int64(window_days) * np.int64(day_ticks)

    for idx in _groups(df, [key_col], dropna=True).values():
        order = idx[np.argsort(rank[idx], kind="stable")]
        d = d_all[order]
        cc = _cum(contrib[order])
        cw = _cum(is_win[order])
        # priors that had actually RUN by this row's own off-time (audit F1);
        # the calendar-day window bound stays on the date, as before.
        end = np.searchsorted(prior_all[order], self_all[order], side="left")
        lo = np.searchsorted(d, np.where(d == _NAT, _NAT, d - window_ns), side="left")
        lo = np.minimum(lo, end)
        n_in_window = cc[end] - cc[lo]
        cnt[order] = n_in_window.astype("int64")
        ok = n_in_window >= min_runners
        wins = cw[end] - cw[lo]
        rate[order] = np.where(ok, wins / np.where(n_in_window > 0, n_in_window, 1.0), np.nan)
    return rate, cnt


def windowed_win_rate_days(df, key_col, *, window_days, min_runners):
    """Trailing win-rate over a strictly-prior calendar-day window, keyed on a
    single entity column (e.g. trainer_id / jockey_id).

    Returns (rate, n) numpy arrays aligned to df's positional order. ``rate`` is
    NaN unless at least ``min_runners`` known-position priors fall in the
    window. Same market-duplicate collapse as ``windowed_rates`` — a runner's
    WIN+PLACE row of one real race must count as a single prior result for
    another entity's window, not two, and must never leak into its own
    same-race sibling's rate.
    """
    n = len(df)
    rate = np.full(n, np.nan)
    cnt = np.zeros(n, dtype="int64")
    if n == 0 or key_col not in df.columns or "race_date" not in df.columns:
        return rate, cnt

    keep, gid = _dedupe_group_ids(df, [key_col])
    if keep.all():
        return _windowed_win_rate_days_core(df, key_col, window_days, min_runners)

    sub = df.loc[keep].reset_index(drop=True)
    sub_rate, sub_cnt = _windowed_win_rate_days_core(sub, key_col, window_days, min_runners)
    return sub_rate[gid], sub_cnt[gid]
