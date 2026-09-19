"""Training labels derived from finishing position."""
import pandas as pd


def add_labels(df: pd.DataFrame, place_positions: int) -> pd.DataFrame:
    """won = position==1; placed = position<=place_positions. Null position -> null."""
    out = df.copy()
    pos = pd.to_numeric(out["position"], errors="coerce")
    out["won"] = pos.where(pos.isna(), (pos == 1).astype("Int64"))
    out["placed"] = pos.where(pos.isna(), (pos <= place_positions).astype("Int64"))
    out["won"] = out["won"].astype("Int64")
    out["placed"] = out["placed"].astype("Int64")
    return out
