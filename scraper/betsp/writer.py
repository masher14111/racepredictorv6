"""Year-partitioned parquet writer with read-merge-write dedupe."""
import os

import pandas as pd

from utils.logger import get_logger
from utils.storage import parquet_store

logger = get_logger(__name__)

_DEFAULT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical", "betsp.parquet")
)

FINAL_COLUMNS = [
    "race_date", "venue", "horse_id", "horse_name", "jockey_id", "jockey_name",
    "trainer_id", "trainer_name",
    "odds_finish", "position", "win_lose", "going", "distance", "market_type",
    "region", "morningwap", "ppwap", "result_source", "fetched_at", "year",
]
_DEDUPE_KEY = ["race_date", "venue", "horse_id", "market_type"]


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["race_date"], utc=True).dt.year.astype(int)
    return df


def _completeness_order(df: pd.DataFrame) -> pd.DataFrame:
    """Order rows so a resolved result always survives ``drop_duplicates(keep="last")``.

    A resumable backfill/catch-up re-request for an already-covered day (e.g. a
    partial results-site enrichment miss, or Betfair's daily SP file boundary
    landing one calendar day off the requested date — see
    ``scripts/fetch_results_window``) can legitimately return ``position=None``
    for a (race_date, venue, horse_id, market_type) key that the store already
    has a real finishing position for. Plain "last write wins" would silently
    regress that row from resolved back to unknown. Sorting by completeness
    first (then by ``fetched_at``) means the most-complete row is always the one
    ``keep="last"`` keeps; among equally-complete rows the freshest fetch wins,
    same as before.
    """
    completeness = df["position"].notna().astype(int)
    fetched_at = pd.to_datetime(df.get("fetched_at"), utc=True, errors="coerce")
    order = pd.DataFrame(
        {"c": completeness, "f": fetched_at}, index=df.index
    ).sort_values(["c", "f"], kind="stable").index
    return df.loc[order]


def write(df: pd.DataFrame, path: str = _DEFAULT_PATH) -> None:
    """Merge df into the partitioned dataset at `path`, dedupe, write per year."""
    if df.empty:
        logger.debug("writer: empty df, nothing to write")
        return
    df = _add_year(df)
    existing = parquet_store.read_parquet(path)
    merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    merged = _completeness_order(merged)
    parquet_store.write_parquet(
        merged, path, partition_by="year", dedupe_key=_DEDUPE_KEY, columns=FINAL_COLUMNS)
    logger.debug("writer: wrote %d rows across %d years",
                 len(df), df["year"].nunique())
