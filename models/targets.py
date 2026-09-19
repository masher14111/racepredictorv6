"""Derive binary target columns from finishing position.

add_targets never overwrites the existing `placed` (top-3) column from
features/labels.py — it only adds won, placed_2, and showed.
"""
import pandas as pd


def add_targets(df: pd.DataFrame, show_positions: int = 3) -> pd.DataFrame:
    """Add won, placed_2, showed columns derived from position.

    All three are Int64 (nullable). Null position -> null label.
    Existing columns (including `placed`) are never modified.
    """
    out = df.copy()
    pos = pd.to_numeric(out["position"], errors="coerce")

    def _binary(mask_series) -> pd.array:
        result = pd.array([pd.NA] * len(out), dtype="Int64")
        known = pos.notna()
        result[known] = mask_series[known].astype("Int64")
        return result

    out["won"] = _binary(pos == 1)
    out["placed_2"] = _binary(pos <= 2)
    out["showed"] = _binary(pos <= show_positions)
    return out
