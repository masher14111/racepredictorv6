"""Step 04 evidence: rebuild a sampled candidate matrix with the fixed
trailing-history / race_complexity_v2 pipeline, fit+freeze the
race_complexity_v2 scale artifact, and verify append-future invariance on real
data. Never overwrites data/features*.parquet (frozen, pre-step-04 artifacts)
or the champion race_complexity behaviour — see features/engine.py.

Usage: .venv/Scripts/python.exe reports/improvement/04/build_candidate_and_scale.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _BASE)

from features import _feature_scale, engine  # noqa: E402
from features.builder import build_training_matrix  # noqa: E402

OUT_DIR = os.path.join(_BASE, "data", "audit", "04")
os.makedirs(OUT_DIR, exist_ok=True)


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    unified_path = os.path.join(_BASE, "data", "unified_races.parquet")
    _log(f"loading {unified_path}")
    raw_full = pd.read_parquet(unified_path)
    _log(f"raw unified rows: {len(raw_full)}")

    # Same sample selection as stage 03's rebuild_candidate.py (same
    # random_state) so the two candidate matrices cover the same races.
    sample_races = (
        raw_full[["race_date", "venue", "race_time"]]
        .drop_duplicates()
        .sample(n=8000, random_state=42)
    )
    raw = raw_full.merge(sample_races, on=["race_date", "venue", "race_time"], how="inner")
    _log(f"sampled {len(sample_races)} races -> {len(raw)} raw rows")

    t0 = time.time()
    df = build_training_matrix(unified=raw, write=False)
    _log(f"build_training_matrix took {time.time()-t0:.1f}s; {len(df)} labelled rows")

    # ---- fit + freeze the scale from THIS build's labelled rows first (fitting
    # reads the raw field_size/race_class/... columns directly, not the
    # already-computed race_complexity_v2 column, so it does not need a scale
    # artifact to already exist). Then recompute race_complexity_v2 in-place
    # using the now-frozen scale, so the candidate parquet actually carries a
    # populated column instead of the "no artifact yet" all-null default. ----
    scale = engine.fit_race_complexity_v2_scale(df)
    _feature_scale.save_scale(scale, version="v1")
    _log(f"fitted+saved scale: {scale}")
    df = engine.add_race_complexity_v2(df, scale=scale)

    cand_path = os.path.join(OUT_DIR, "training_pit_fixed.parquet")
    df.to_parquet(cand_path, engine="pyarrow", index=False)
    _log(f"wrote candidate matrix: {cand_path}")

    # ---- 1. trailing-history market-duplicate fix: both market rows of the
    # same real race must show the IDENTICAL days_since_last_run/career_runs. ----
    both = df[df.duplicated(subset=["race_uid", "horse_id"], keep=False)]
    _log(f"runner-races with both WIN+PLACE rows in this sample: "
         f"{both['race_uid'].nunique()} races / {len(both)} rows")
    mism_dsr = 0
    mism_car = 0
    for _, g in both.groupby(["race_uid", "horse_id"]):
        if g["days_since_last_run"].nunique(dropna=False) > 1:
            mism_dsr += 1
        if g["horse_career_runs"].nunique(dropna=False) > 1:
            mism_car += 1
    _log(f"days_since_last_run WIN/PLACE-sibling mismatches: {mism_dsr}")
    _log(f"horse_career_runs WIN/PLACE-sibling mismatches: {mism_car}")

    # ---- 2. race_complexity_v2 reads no market column (verified structurally by
    # tests/features/test_engine.py::test_race_complexity_v2_has_no_market_input,
    # which mutates overround_norm_prob and checks byte-identical output). A
    # non-zero correlation printed below is EXPECTED and is not evidence of a
    # market dependency: race_complexity_v2 is dominated by field_size on today's
    # data (the other 3 components are all-null — see Limitations), and
    # market_rank/implied_prob's per-row distribution is mechanically bounded by
    # field_size (a bigger field has runners at higher ranks and a smaller mean
    # implied_prob) independent of any formula reading the market. ----
    market_cols = ["implied_prob", "overround_norm_prob", "log_odds", "market_rank"]
    present = [c for c in market_cols if c in df.columns]
    _log(f"race_complexity_v2 non-null: {df['race_complexity_v2'].notna().sum()} / {len(df)}")
    _log(f"race_market_entropy non-null: {df['race_market_entropy'].notna().sum()} / {len(df)}")
    corr = df[["race_complexity_v2"] + present].corr().iloc[0, 1:]
    _log(f"race_complexity_v2 vs market columns correlation (informational only "
         f"— see comment above; NOT evidence of a market dependency):\n{corr}")

    # ---- 3. append-future invariance: recompute race_complexity_v2 and the
    # trailing features on the sample PLUS extra later rows; earlier admissible
    # rows must be BYTE-IDENTICAL. The frozen scale saved in step 3 (already on
    # disk by this point) is what both builds below load — that is the whole
    # point being verified: neither build recomputes it from its own frame. ----
    base_full = df
    # Strictly LATER than every race in the base sample, so any diff on a base
    # row can only be a real invariance violation, never a legitimate "a new
    # earlier prior appeared" update.
    later = raw_full[raw_full["race_date"] > raw["race_date"].max()]
    extra_races = (
        later[["race_date", "venue", "race_time"]]
        .drop_duplicates()
        .sample(n=min(500, later[["race_date", "venue", "race_time"]].drop_duplicates().shape[0]),
                random_state=7)
    )
    appended_raw = pd.concat([raw, later.merge(
        extra_races, on=["race_date", "venue", "race_time"], how="inner")])
    appended_full = build_training_matrix(unified=appended_raw, write=False)
    # Diagnostic: which raw race_complexity_v2 component's non-null population
    # differs between the two builds (isolates why a value changed).
    for name, series in engine._race_complexity_v2_raw_components(appended_full).items():
        base_series = engine._race_complexity_v2_raw_components(base_full)[name]
        _log(f"component {name}: base non-null {base_series.notna().sum()}/{len(base_full)}, "
             f"appended non-null {series.notna().sum()}/{len(appended_full)}")

    common_keys = ["race_uid", "horse_id", "market_type"]
    merged = base_full.merge(
        appended_full, on=common_keys, suffixes=("_base", "_appended"), how="inner")
    _log(f"append-invariance join: {len(merged)} / {len(base_full)} base rows matched")
    check_cols = ["race_complexity_v2", "days_since_last_run", "horse_career_runs",
                  "historical_win_rate", "trainer_hot_strike_rate"]
    total_diff = 0
    for col in check_cols:
        a = pd.to_numeric(merged[f"{col}_base"], errors="coerce")
        b = pd.to_numeric(merged[f"{col}_appended"], errors="coerce")
        bad = ~((a.isna() & b.isna()) | np.isclose(a, b, equal_nan=True))
        n_bad = int(bad.sum())
        total_diff += n_bad
        _log(f"append-invariance {col}: {n_bad} changed / {len(merged)}")
        if n_bad and col == "race_complexity_v2":
            sample = merged.loc[bad, ["race_uid", "horse_id", "market_type",
                                       f"{col}_base", f"{col}_appended"]].head(5)
            _log(f"sample changed rows:\n{sample}")

    print("CANDIDATE_OK" if (mism_dsr == 0 and mism_car == 0 and total_diff == 0)
          else "CANDIDATE_FAIL", flush=True)


if __name__ == "__main__":
    main()
