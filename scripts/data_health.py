"""Data-health snapshot: row/column counts, date range, and key-column fill
rates for the three core datasets, plus a count of "upcoming" rows.

Run: python scripts/data_health.py
"""
import os
import sys

import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from utils.timezone import now

DATASETS = {
    "unified_races.parquet": os.path.join(_BASE, "data", "unified_races.parquet"),
    "data/features/training.parquet": os.path.join(_BASE, "data", "features", "training.parquet"),
    "data/features.parquet": os.path.join(_BASE, "data", "features.parquet"),
}

KEY_COLS = ["position", "odds_decimal", "implied_prob", "sp", "jockey_name", "trainer_name"]


def _date_range(df: pd.DataFrame) -> str:
    if "race_date" not in df.columns or df["race_date"].notna().sum() == 0:
        return "n/a (no race_date)"
    d = pd.to_datetime(df["race_date"], utc=True, errors="coerce")
    return f"{d.min().date()} -> {d.max().date()}"


def _upcoming_count(df: pd.DataFrame) -> str:
    if "position" not in df.columns or "race_date" not in df.columns:
        return "n/a (missing position/race_date)"
    today = pd.Timestamp(now().date(), tz="UTC")
    d = pd.to_datetime(df["race_date"], utc=True, errors="coerce")
    mask = df["position"].isna() & (d >= today)
    return str(int(mask.sum()))


def report(name: str, path: str) -> None:
    print(f"\n=== {name} ===")
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return
    df = pd.read_parquet(path)
    n = len(df)
    print(f"  rows:    {n:,}")
    print(f"  columns: {df.shape[1]}")
    print(f"  dates:   {_date_range(df)}")
    print("  fill rates:")
    for col in KEY_COLS:
        if col not in df.columns:
            print(f"    {col:<14} absent")
            continue
        nn = int(df[col].notna().sum())
        pct = (nn / n * 100) if n else 0.0
        print(f"    {col:<14} {nn:>10,} / {n:,}  ({pct:6.2f}%)")
    print(f"  upcoming rows (position null AND race_date >= today): {_upcoming_count(df)}")


def main() -> None:
    print(f"Data health snapshot  (today = {now().date()}, Europe/Dublin)")
    for name, path in DATASETS.items():
        report(name, path)


if __name__ == "__main__":
    main()
