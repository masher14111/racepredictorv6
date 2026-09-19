"""Stage 06 candidate: rebuild the labelled training matrix over the FULL current
``data/unified_races.parquet`` through the current (steps 03-06 fixed) pipeline,
without touching the frozen ``data/features/training.parquet`` baseline.

Measures the accessible historical window this stage can add (races with a
result after the frozen baseline's 2026-07-25 cutoff) and the going_speed vs
going_speed_v2 coverage repair, side by side, on real data.

Run: .venv/Scripts/python.exe reports/improvement/06/build_candidate.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

from features import builder
from tools.coverage_report import DEFAULT_WATCH_COLS, build_coverage_report

t0 = time.monotonic()
uni = pd.read_parquet("data/unified_races.parquet")
print(f"[{time.monotonic()-t0:.1f}s] loaded unified: {len(uni)} rows", flush=True)

df = builder.build_training_matrix(unified=uni, write=False)
print(f"[{time.monotonic()-t0:.1f}s] built candidate training matrix: {len(df)} rows, "
      f"{df.shape[1]} cols", flush=True)

out_path = "data/audit/06/training_refreshed.parquet"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_parquet(out_path, engine="pyarrow", index=False)
print(f"[{time.monotonic()-t0:.1f}s] wrote {out_path}", flush=True)

watch = DEFAULT_WATCH_COLS + ["going_is_all_weather", "race_complexity_v2", "race_market_entropy"]
report = build_coverage_report(df, "training_refreshed_06", watch_cols=watch)
with open("reports/improvement/06/coverage_training_refreshed.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print(f"[{time.monotonic()-t0:.1f}s] wrote coverage report", flush=True)
print("date range:", report["date_min"], "->", report["date_max"])

frozen = pd.read_parquet("data/features/training.parquet")
frozen_report = build_coverage_report(frozen, "training_frozen_baseline", watch_cols=watch)
with open("reports/improvement/06/coverage_training_frozen_baseline.json", "w", encoding="utf-8") as fh:
    json.dump(frozen_report, fh, indent=2, default=str)

print("\n=== before (frozen baseline, data/features/training.parquet) ===")
print("rows:", frozen_report["rows"], "dates:", frozen_report["date_min"], "->", frozen_report["date_max"])
for col in ("going_speed", "going_speed_v2", "race_class", "timeform_rating"):
    print(f"  {col}: {frozen_report['watched_fill'].get(col)}")

print("\n=== after (candidate, data/audit/06/training_refreshed.parquet) ===")
print("rows:", report["rows"], "dates:", report["date_min"], "->", report["date_max"])
for col in ("going_speed", "going_speed_v2", "race_class", "timeform_rating"):
    print(f"  {col}: {report['watched_fill'].get(col)}")

new_rows = df[pd.to_datetime(df["race_date"], utc=True) > pd.Timestamp("2026-07-25", tz="UTC")]
print(f"\nrows strictly after the frozen baseline's 2026-07-25 cutoff: {len(new_rows)}")
print("BUILD_CANDIDATE_06_COMPLETE", flush=True)
