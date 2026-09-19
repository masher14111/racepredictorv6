"""Year-partitioned parquet writer for the Timeform dataset (read-merge-write)."""
import pandas as pd

from utils.logger import get_logger
from utils.storage import parquet_store

logger = get_logger(__name__)

FINAL_COLUMNS = [
    "race_date", "venue", "race_time", "horse_name", "horse_id",
    "jockey_name", "jockey_id", "trainer_name", "trainer_id", "position",
    "timeform_rating", "pace_rating", "race_class", "going", "going_speed",
    "distance", "class_change", "recent_form", "historical_win_rate",
    "historical_place_rate", "jockey_win_rate", "runs_in_window",
    "draw", "weight_lbs", "age", "official_rating", "equipment",
    "colour", "sex", "runner_status", "timeform_race_id",
    "region", "source", "fetched_at", "year",
]
_DEDUPE_KEY = ["race_date", "venue", "race_time", "horse_name"]


def _add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["race_date"], utc=True).dt.year.astype(int)
    return df


def write(df: pd.DataFrame, path: str) -> None:
    if df is None or df.empty:
        logger.debug("timeform writer: empty df, nothing to write")
        return
    df = _add_year(df)
    parquet_store.append_parquet(
        df, path, partition_by="year", dedupe_key=_DEDUPE_KEY, columns=FINAL_COLUMNS)
    logger.debug("timeform writer: wrote %d rows", len(df))
