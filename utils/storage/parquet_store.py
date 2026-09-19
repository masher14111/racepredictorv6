"""Centralized Parquet read-merge-write + year-partition helpers."""
import os
import shutil

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


def read_parquet(path: str, filters=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if os.path.isdir(path) and not os.listdir(path):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path, engine="pyarrow", filters=filters)
    except Exception as exc:  # noqa: BLE001
        logger.warning("read_parquet failed for %s (%s)", path, exc)
        return pd.DataFrame()


def _atomic_replace_dir(tmp_path: str, path: str) -> None:
    """Swap a fully-written ``tmp_path`` into ``path`` via two directory renames.

    ``os.rename`` on a directory is a single filesystem metadata operation, so
    each swap step is effectively instantaneous — a world apart from
    ``rmtree(path)`` followed by a multi-second ``to_parquet`` write, which
    leaves ``path`` empty/missing for the entire write. If the process dies
    between the two renames here, the previous good dataset is still on disk
    at ``{path}.bak`` and the new one at ``tmp_path`` — never silently lost,
    never partially mixed.
    """
    backup = f"{path}.bak"
    if os.path.isdir(backup):
        shutil.rmtree(backup, ignore_errors=True)
    had_existing = os.path.isdir(path)
    if had_existing:
        os.rename(path, backup)
    try:
        os.rename(tmp_path, path)
    except Exception:
        if had_existing and os.path.isdir(backup) and not os.path.isdir(path):
            os.rename(backup, path)
        raise
    if had_existing:
        shutil.rmtree(backup, ignore_errors=True)


def write_partitioned_parquet(df: pd.DataFrame, path: str, partition_cols) -> None:
    """Crash-safe replacement for ``df.to_parquet(path, partition_cols=...)``.

    Pyarrow's ``partition_cols`` writer appends part-files rather than
    overwriting, so a clean write needs a fresh directory — but clearing
    ``path`` in place first (``rmtree`` then write) leaves it empty or
    partially written for the whole write duration if the process is
    interrupted. Instead, write the full partitioned dataset to a sibling
    temp directory and swap it in atomically once it is complete.
    """
    cols = [partition_cols] if isinstance(partition_cols, str) else list(partition_cols)
    tmp_path = f"{path}.tmp"
    if os.path.isdir(tmp_path):
        shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False, partition_cols=cols)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        _atomic_replace_dir(tmp_path, path)
    except Exception:
        if os.path.isdir(tmp_path):
            shutil.rmtree(tmp_path, ignore_errors=True)
        raise


def write_parquet(df: pd.DataFrame, path: str, partition_by=None,
                  dedupe_key=None, columns=None) -> None:
    if df is None or df.empty:
        logger.debug("write_parquet: empty df, skipping %s", path)
        return
    df = df.copy()
    if dedupe_key:
        df = df.drop_duplicates(subset=dedupe_key, keep="last")
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[columns]
    if partition_by:
        write_partitioned_parquet(df, path, partition_by)
    else:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Atomic write: a scrape interrupted mid-write (crash, kill) must never
        # leave a truncated/corrupt file mixed with the previous snapshot.
        tmp = f"{path}.tmp"
        try:
            df.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, path)
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            raise


def append_parquet(df: pd.DataFrame, path: str, partition_by=None,
                   dedupe_key=None, columns=None) -> None:
    """Read-merge-write: concat existing + new, dedupe, write back."""
    if df is None or df.empty:
        return
    existing = read_parquet(path)
    merged = pd.concat([existing, df], ignore_index=True) if not existing.empty else df
    write_parquet(merged, path, partition_by=partition_by,
                  dedupe_key=dedupe_key, columns=columns)
