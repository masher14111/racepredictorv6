"""Fuse per-source unified rows into one row per runner-race-market."""
import pandas as pd

from utils.market_validation import LIVE_SOURCES, VALID

_RUNNER_KEY_BASE = ["race_date", "venue"]

# Higher rank wins a column conflict. Unknown sources rank lowest.
_SOURCE_PRIORITY = {
    "timeform": 3,
    "betsp": 2,
    "boylesports": 1, "livescorebet": 1, "paddy_power": 1,
}

# WIN and PLACE are different price books (DESIGN.md): a column in this set
# describes the specific market/quote a row came from, not the runner or race,
# and must never be inherited from a runner's OTHER market row. Everything
# else (identity, ratings, form, finishing position) is genuinely shared by a
# runner across both its WIN and PLACE rows and is safe to cross-fill.
_MARKET_SPECIFIC_COLS = frozenset({
    "market_id", "market_name", "market_type", "odds_decimal", "sp",
    "ew_places", "ew_reduction", "ew_margin", "is_low_odds", "currency",
    "morningwap", "ppwap", "odds_finish", "win_lose",
    "market_field_size", "market_booksum",
    "source", "fetched_at", "stale", "validation_status", "validation_reasons",
})


def _priority(source) -> int:
    return _SOURCE_PRIORITY.get(source, 0)


def fuse_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per runner, actual race AND market.

    For each column, take the first non-null value, preferring higher-priority
    sources (timeform > betsp > live) on conflicts. Explicitly invalid live races
    are removed before any cross-source values can be fused into a runner.

    A runner racing under both a WIN and a PLACE book (the normal case — see
    DESIGN.md) yields two output rows, one per market, so market-specific price/
    terms/provenance never overwrite each other. Non-market-specific attributes
    (horse/race identity, ratings, form, finishing position) are then backfilled
    across a runner's own market rows so neither book is missing data the other
    happened to carry."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = _drop_invalid_live_races(df.copy())
    out = _drop_stale_live_rows(out)
    if out.empty:
        return out
    out["_prio"] = out["source"].map(_priority)
    # Highest priority first so the first non-null per group is the preferred source.
    out = out.sort_values("_prio", ascending=False, kind="stable")

    runner_key = _runner_key(out)
    value_cols = [c for c in out.columns if c not in runner_key + ["_prio"]]
    agg = {c: _first_valid for c in value_cols}
    fused = out.groupby(runner_key, sort=False, as_index=False, dropna=False).agg(agg)
    fused = _backfill_shared_attrs(fused, _runner_key(out, include_market=False))
    return fused


def _runner_key(df: pd.DataFrame, include_market: bool = True) -> list[str]:
    key = list(_RUNNER_KEY_BASE)
    if "race_time" in df.columns and df["race_time"].notna().any():
        key.append("race_time")
    key.append("horse_id")
    if include_market and "market_type" in df.columns:
        key.append("market_type")
    return key


def _backfill_shared_attrs(df: pd.DataFrame, base_key: list[str]) -> pd.DataFrame:
    """Fill a runner's missing non-market-specific attributes from its OTHER
    market row(s) in the same race (e.g. a PLACE row missing jockey_id because
    only the WIN-market source reported it). Only fills nulls — never overwrites
    an existing value — and never touches ``_MARKET_SPECIFIC_COLS``, so a price,
    term or provenance field can never cross from one market's row to the
    other's. No-op when there is no market_type to fan out across."""
    if "market_type" not in df.columns or df.empty:
        return df
    shared_cols = [c for c in df.columns
                   if c not in _MARKET_SPECIFIC_COLS and c not in base_key]
    if not shared_cols:
        return df
    out = df.copy()
    # One column at a time: grouping the whole shared-column block together and
    # transforming in one call crashes pyarrow-backed pandas (mixed extension
    # dtypes, e.g. the null[pyarrow] columns, inside a single multi-column
    # groupby.transform). Per-column transform is the documented-safe shape.
    # An entirely-null column (e.g. pandas' all-null "null[pyarrow]" dtype, seen
    # on race_id/selection_id/market_id/validation_status before any source
    # populates them) has nothing to backfill, and Series.where() on that dtype
    # crashes the pyarrow C++ layer outright — skip it, a true no-op either way.
    #
    # Use DataFrameGroupBy.ffill()/bfill(), NOT .transform(lambda s: ...) — the
    # lambda form calls Python once per group (~326k runner-race groups on the
    # live matrix), which is minutes slower than the vectorized cython ffill/
    # bfill implementation for the exact same result (groups are almost always
    # size <=2, so direction doesn't matter: either fill direction reaches the
    # other market row's value).
    keys = [out[k] for k in base_key]
    for c in shared_cols:
        if not out[c].notna().any():
            continue
        fwd = out.groupby(keys, sort=False)[c].ffill()
        filled = fwd.groupby(keys, sort=False).bfill()
        out[c] = out[c].where(out[c].notna(), filled)
    return out


def _drop_invalid_live_races(df: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on live races carrying an explicit non-VALID status."""
    if (
        "source" not in df.columns
        or "validation_status" not in df.columns
        or not df["source"].isin(LIVE_SOURCES).any()
    ):
        return df

    live = df["source"].isin(LIVE_SOURCES)
    bad = live & ~df["validation_status"].astype("string").eq(VALID)
    if "market_type" in df.columns:
        bad |= live & ~df["market_type"].astype("string").eq("WIN")
    if not bad.any():
        return df

    group_cols = ["source"]
    if "race_id" in df.columns and df["race_id"].notna().any():
        group_cols.append("race_id")
    else:
        group_cols.extend(c for c in ("venue", "race_time") if c in df.columns)
    bad_keys = df.loc[bad, group_cols].drop_duplicates()
    marked = df.merge(bad_keys.assign(_invalid_race=True), on=group_cols, how="left")
    return marked[marked["_invalid_race"].isna()].drop(columns="_invalid_race")


def _drop_stale_live_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Fail closed on live rows older than the configured staleness TTL.

    Age is measured from ``fetched_at`` (provenance, never reset on a cache
    hit), not wall-clock scrape time — this is what stops a degraded
    stale-cache-fallback scrape from reaching the value layer disguised as a
    fresh quote. Rows explicitly flagged ``stale`` (a source served its cache
    after every live fetch tier failed) are dropped unconditionally, on top of
    the age-based TTL that catches sources with no such flag but a simply
    outdated fetch.
    """
    if "source" not in df.columns or "fetched_at" not in df.columns:
        return df
    live = df["source"].isin(LIVE_SOURCES)
    if not live.any():
        return df

    from utils.config_loader import get_config
    max_age = float(get_config().get("staleness", {}).get("max_age_seconds", 900))

    # format="mixed": a batch mixing whole-second (e.g. a declared-card source)
    # and sub-second (e.g. a live odds source) fetched_at strings makes pandas'
    # single-format inference silently emit NaT for every row shaped
    # differently from the first non-null value — errors="coerce" masks it, so
    # a genuinely fresh row gets fail-closed as "unparseable -> too old" instead
    # of surviving on its real age (found tracing Stage 05's provenance chain).
    fetched = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce", format="mixed")
    now = pd.Timestamp.now(tz="UTC")
    age_seconds = (now - fetched).dt.total_seconds()
    flagged_stale = (
        df["stale"].fillna(False).astype(bool)
        if "stale" in df.columns
        else pd.Series(False, index=df.index)
    )
    # Missing/unparseable fetched_at on a live row is itself untrustworthy —
    # fail closed rather than treat it as fresh.
    too_old = live & (flagged_stale | age_seconds.isna() | (age_seconds > max_age))
    if not too_old.any():
        return df
    return df.loc[~too_old].reset_index(drop=True)


def _first_valid(series: pd.Series):
    for v in series:
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return series.iloc[0] if len(series) else None
