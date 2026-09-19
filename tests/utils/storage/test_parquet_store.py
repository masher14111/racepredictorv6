import os
from unittest.mock import patch

import pandas as pd
import pytest

from utils.storage.parquet_store import (
    append_parquet,
    read_parquet,
    write_parquet,
    write_partitioned_parquet,
)


def test_write_then_read_roundtrip(tmp_path):
    path = str(tmp_path / "d.parquet")
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    write_parquet(df, path)
    out = read_parquet(path)
    assert set(out["a"]) == {1, 2}


def test_append_dedupes_on_key(tmp_path):
    path = str(tmp_path / "d.parquet")
    write_parquet(pd.DataFrame({"k": [1], "v": ["old"]}), path)
    append_parquet(pd.DataFrame({"k": [1], "v": ["new"]}), path, dedupe_key=["k"])
    out = read_parquet(path)
    assert len(out) == 1 and out.iloc[0]["v"] == "new"


def test_year_partition_write(tmp_path):
    path = str(tmp_path / "part")
    df = pd.DataFrame({"race_date": ["2025-01-01", "2026-01-01"], "v": [1, 2]})
    df["year"] = [2025, 2026]
    write_parquet(df, path, partition_by="year")
    out = read_parquet(path)
    assert set(out["year"]) == {2025, 2026}


def test_read_missing_returns_empty(tmp_path):
    out = read_parquet(str(tmp_path / "nope.parquet"))
    assert out.empty


def test_interrupted_write_leaves_previous_snapshot_untouched(tmp_path):
    path = str(tmp_path / "d.parquet")
    write_parquet(pd.DataFrame({"k": [1], "v": ["old"]}), path)

    real_to_parquet = pd.DataFrame.to_parquet

    def _crash_after_tmp_write(self, target, *a, **kw):
        # Let the tmp file get written (proving it exists mid-write), then
        # blow up before os.replace() runs — simulating a killed process.
        real_to_parquet(self, target, *a, **kw)
        raise OSError("simulated crash mid-write")

    with patch.object(pd.DataFrame, "to_parquet", _crash_after_tmp_write):
        with pytest.raises(OSError):
            write_parquet(pd.DataFrame({"k": [2], "v": ["new"]}), path)

    # Previous snapshot must be intact — never a partial/corrupt mix.
    out = read_parquet(path)
    assert len(out) == 1 and out.iloc[0]["v"] == "old"
    # The tmp file must be cleaned up, not left dangling next to the snapshot.
    assert not os.path.exists(f"{path}.tmp")


def test_interrupted_write_on_fresh_path_leaves_no_partial_file(tmp_path):
    path = str(tmp_path / "fresh.parquet")

    real_to_parquet = pd.DataFrame.to_parquet

    def _crash_after_tmp_write(self, target, *a, **kw):
        real_to_parquet(self, target, *a, **kw)
        raise OSError("simulated crash mid-write")

    with patch.object(pd.DataFrame, "to_parquet", _crash_after_tmp_write):
        with pytest.raises(OSError):
            write_parquet(pd.DataFrame({"k": [1], "v": ["x"]}), path)

    assert not os.path.exists(path)
    assert not os.path.exists(f"{path}.tmp")


def test_partitioned_write_no_longer_clears_dir_in_place(tmp_path):
    """Requirement 6: the old rmtree(path)-then-write left `path` empty for the
    full write duration; the swap-based writer must never remove `path` until
    the replacement dataset is fully written and ready beside it."""
    path = str(tmp_path / "part")
    df = pd.DataFrame({"race_date": ["2025-01-01"], "v": [1], "year": [2025]})
    write_parquet(df, path, partition_by="year")

    real_to_parquet = pd.DataFrame.to_parquet
    path_existed_mid_write = []

    def _observe_mid_write(self, target, *a, **kw):
        path_existed_mid_write.append(os.path.isdir(path))
        return real_to_parquet(self, target, *a, **kw)

    with patch.object(pd.DataFrame, "to_parquet", _observe_mid_write):
        write_parquet(
            pd.DataFrame({"race_date": ["2026-01-01"], "v": [2], "year": [2026]}),
            path, partition_by="year",
        )

    assert path_existed_mid_write == [True]


def test_partitioned_write_interrupted_leaves_previous_snapshot_untouched(tmp_path):
    path = str(tmp_path / "part")
    write_parquet(
        pd.DataFrame({"race_date": ["2025-01-01"], "v": [1], "year": [2025]}),
        path, partition_by="year",
    )

    real_to_parquet = pd.DataFrame.to_parquet

    def _crash_after_tmp_write(self, target, *a, **kw):
        # Let the temp partitioned dir get fully written (proving it exists),
        # then blow up before the atomic rename-swap runs.
        real_to_parquet(self, target, *a, **kw)
        raise OSError("simulated crash mid-write")

    with patch.object(pd.DataFrame, "to_parquet", _crash_after_tmp_write):
        with pytest.raises(OSError):
            write_partitioned_parquet(
                pd.DataFrame({"race_date": ["2026-01-01"], "v": [2], "year": [2026]}),
                path, "year",
            )

    # Previous snapshot survives intact — never emptied, never a partial mix.
    out = read_parquet(path)
    assert set(out["year"]) == {2025}
    # No dangling temp/backup directories left beside the real path.
    assert not os.path.isdir(f"{path}.tmp")
    assert not os.path.isdir(f"{path}.bak")


def test_partitioned_write_interrupted_on_fresh_path_leaves_no_partial_dir(tmp_path):
    path = str(tmp_path / "part")

    real_to_parquet = pd.DataFrame.to_parquet

    def _crash_after_tmp_write(self, target, *a, **kw):
        real_to_parquet(self, target, *a, **kw)
        raise OSError("simulated crash mid-write")

    with patch.object(pd.DataFrame, "to_parquet", _crash_after_tmp_write):
        with pytest.raises(OSError):
            write_partitioned_parquet(
                pd.DataFrame({"race_date": ["2026-01-01"], "v": [1], "year": [2026]}),
                path, "year",
            )

    assert not os.path.isdir(path)
    assert not os.path.isdir(f"{path}.tmp")
    assert not os.path.isdir(f"{path}.bak")


def test_partitioned_write_leaves_no_backup_dir_after_success(tmp_path):
    path = str(tmp_path / "part")
    write_parquet(
        pd.DataFrame({"race_date": ["2025-01-01"], "v": [1], "year": [2025]}),
        path, partition_by="year",
    )
    write_parquet(
        pd.DataFrame({"race_date": ["2026-01-01"], "v": [2], "year": [2026]}),
        path, partition_by="year",
    )
    assert not os.path.isdir(f"{path}.bak")
    assert not os.path.isdir(f"{path}.tmp")
    out = read_parquet(path)
    assert set(out["year"]) == {2026}  # partitioned writer replaces, not appends
