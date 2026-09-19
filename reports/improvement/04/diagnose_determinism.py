"""Diagnostic: is build_training_matrix deterministic on the IDENTICAL input,
called twice in the same process? Isolates whether step 04's append-invariance
"failure" is actually a pre-existing non-determinism bug unrelated to appending
any new rows (the previous run's "later" sample turned out to be EMPTY, i.e.
appended_raw == raw exactly, yet race_complexity_v2 still differed)."""
import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _BASE)

from features.builder import build_training_matrix  # noqa: E402


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    raw_full = pd.read_parquet(os.path.join(_BASE, "data", "unified_races.parquet"))
    sample_races = (
        raw_full[["race_date", "venue", "race_time"]]
        .drop_duplicates()
        .sample(n=8000, random_state=42)
    )
    raw = raw_full.merge(sample_races, on=["race_date", "venue", "race_time"], how="inner")
    _log(f"sample: {len(raw)} raw rows")

    t0 = time.time()
    a = build_training_matrix(unified=raw.copy(), write=False)
    _log(f"build 1: {time.time()-t0:.1f}s, {len(a)} rows")
    t0 = time.time()
    b = build_training_matrix(unified=raw.copy(), write=False)
    _log(f"build 2: {time.time()-t0:.1f}s, {len(b)} rows")

    common_keys = ["race_uid", "horse_id", "market_type"]
    merged = a.merge(b, on=common_keys, suffixes=("_a", "_b"), how="inner")
    _log(f"join: {len(merged)} / {len(a)} matched")
    for col in ["field_size", "race_complexity_v2", "overround_norm_prob", "implied_prob"]:
        if f"{col}_a" not in merged.columns:
            continue
        x = pd.to_numeric(merged[f"{col}_a"], errors="coerce")
        y = pd.to_numeric(merged[f"{col}_b"], errors="coerce")
        bad = ~((x.isna() & y.isna()) | np.isclose(x, y, equal_nan=True))
        _log(f"{col}: {int(bad.sum())} differ / {len(merged)}")
        if bad.sum():
            sample = merged.loc[bad, common_keys + [f"{col}_a", f"{col}_b"]].head(5)
            _log(f"sample:\n{sample}")


if __name__ == "__main__":
    main()
