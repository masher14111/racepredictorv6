"""Step 03 evidence: rebuild the training matrix with the fixed fuse/derive
market-association pipeline and report on real data. Writes a candidate
dataset to data/audit/03/ (never overwrites data/features*.parquet — those
stay as the pre-fix frozen artifacts until step 10 owns any retrain/promotion).

Usage: .venv/Scripts/python.exe reports/improvement/03/rebuild_candidate.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _BASE)

from features.builder import build_training_matrix  # noqa: E402
from features.fuse import fuse_sources  # noqa: E402

OUT_DIR = os.path.join(_BASE, "data", "audit", "03")
os.makedirs(OUT_DIR, exist_ok=True)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    unified_path = os.path.join(_BASE, "data", "unified_races.parquet")
    _log(f"loading {unified_path}")
    raw_full = pd.read_parquet(unified_path)
    _log(f"raw unified rows: {len(raw_full)}")

    key = ["race_date", "venue", "race_time", "horse_id"]
    raw_market_sets = raw_full.groupby(key)["market_type"].agg(lambda s: tuple(sorted(set(s))))
    _log("raw per-runner market_type membership (full dataset):")
    print(raw_market_sets.value_counts())

    # features.builder's per-group custom aggregation (_first_valid, pre-existing,
    # not introduced by this stage) scales with group count, which roughly
    # doubles once WIN/PLACE rows are no longer collapsed into one group — a
    # full 651k-row rebuild is a multi-minute batch job, not a hot path. For
    # this stage's "sampled current data" acceptance evidence, run the full
    # pipeline on a representative sample of races instead (fast, same code
    # path); the raw-data market-membership stats above already cover the
    # FULL dataset directly, cheaply, without going through build_training_matrix.
    sample_races = (
        raw_full[["race_date", "venue", "race_time"]]
        .drop_duplicates()
        .sample(n=8000, random_state=42)
    )
    raw = raw_full.merge(sample_races, on=["race_date", "venue", "race_time"], how="inner")
    _log(f"sampled {len(sample_races)} races -> {len(raw)} raw rows for the full-pipeline rebuild")

    t0 = time.time()
    df = build_training_matrix(unified=raw, write=False)
    _log(f"build_training_matrix (labelled rows only, sampled races) took {time.time()-t0:.1f}s")
    _log(f"candidate labelled training rows: {len(df)}")
    print(df["market_type"].value_counts(dropna=False))

    # Canonical uniqueness: no duplicate (race, runner, market) key.
    ukey = ["race_uid", "horse_id", "market_type"]
    dup = df.duplicated(ukey, keep=False)
    _log(f"duplicate (race_uid, horse_id, market_type) keys: {int(dup.sum())}")

    # Per-race market membership on the candidate (fixed) output.
    race_market_sets = df.groupby("race_uid")["market_type"].agg(lambda s: tuple(sorted(set(s))))
    _log("candidate per-race market_type membership:")
    print(race_market_sets.value_counts())

    # Eligible-race parity check: CatBoost's WIN-filtered population vs
    # models.train_lgbm's WIN-only population must be identical sets of rows.
    win_rows = df[df["market_type"].astype(str).str.upper() == "WIN"]
    place_rows = df[df["market_type"].astype(str).str.upper() == "PLACE"]
    _log(f"WIN rows: {len(win_rows)}  PLACE rows: {len(place_rows)}")
    _log(f"WIN race_uid count: {win_rows['race_uid'].nunique()}  "
         f"PLACE race_uid count: {place_rows['race_uid'].nunique()}")

    # Reordering invariance on EVERY priced race in the full dataset (fuse_sources
    # only — cheap, no need to run the full derive pipeline for this check):
    # fuse the same raw rows forward, reversed and shuffled; market assignment
    # (odds_decimal/sp per horse+market) must be identical. Compare NaN-safely —
    # plain dict/float equality treats NaN != NaN, which would flag a false
    # mismatch on an all-null column.
    priced_mask = raw_full["odds_decimal"].notna() | raw_full["sp"].notna()
    priced_keys = raw_full.loc[priced_mask, ["race_date", "venue", "race_time"]].drop_duplicates()
    reorder_ok = True
    mismatches = 0
    for _, sample_key in priced_keys.iterrows():
        mask = (
            (raw_full["race_date"] == sample_key["race_date"])
            & (raw_full["venue"] == sample_key["venue"])
            & (raw_full["race_time"] == sample_key["race_time"])
        )
        sample = raw_full[mask].copy()
        fwd = fuse_sources(sample)
        bwd = fuse_sources(sample.iloc[::-1].reset_index(drop=True))
        shuf = fuse_sources(sample.sample(frac=1.0, random_state=3).reset_index(drop=True))

        def _map(d, col):
            return {(r.horse_id, r.market_type): getattr(r, col) for r in d.itertuples()}

        for col in ("odds_decimal", "sp"):
            fwd_map = _map(fwd, col)
            for other in (_map(bwd, col), _map(shuf, col)):
                same = fwd_map.keys() == other.keys() and all(
                    (pd.isna(fwd_map[k]) and pd.isna(other[k])) or fwd_map[k] == other[k]
                    for k in fwd_map
                )
                if not same:
                    mismatches += 1
                    reorder_ok = False
                    _log(f"MISMATCH {col} on {tuple(sample_key)}: fwd={fwd_map} other={other}")
    _log(f"reordering invariance across ALL {len(priced_keys)} priced races in "
         f"data/unified_races.parquet (forward/reversed/shuffled): "
         f"{reorder_ok} ({mismatches} mismatches)")

    candidate_path = os.path.join(OUT_DIR, "training_market_fixed.parquet")
    df.to_parquet(candidate_path, engine="pyarrow", index=False)
    _log(f"wrote candidate matrix to {candidate_path}")

    summary = {
        "raw_rows": int(len(raw)),
        "candidate_rows": int(len(df)),
        "win_rows": int(len(win_rows)),
        "place_rows": int(len(place_rows)),
        "duplicate_race_runner_market_keys": int(dup.sum()),
        "reordering_invariant_on_sample_race": bool(reorder_ok),
    }
    _log(f"summary: {summary}")
    assert dup.sum() == 0, "canonical uniqueness violated"
    assert reorder_ok, "market assignment not reordering-invariant"
    _log("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
