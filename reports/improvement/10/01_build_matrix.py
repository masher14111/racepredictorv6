"""Step 10: rebuild the labelled training matrix through the D38/D39/D40/D41-
repaired pipeline. Does NOT touch the frozen ``data/features/training.parquet``
baseline (D9 — champion/baseline artifacts stay untouched; new experiments use
distinct directories).

Run: .venv/Scripts/python.exe reports/improvement/10/01_build_matrix.py
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
print(f"[{time.monotonic()-t0:.1f}s] built rebuilt training matrix: {len(df)} rows, "
      f"{df.shape[1]} cols", flush=True)

out_path = "data/audit/10/training_rebuilt.parquet"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_parquet(out_path, engine="pyarrow", index=False)
print(f"[{time.monotonic()-t0:.1f}s] wrote {out_path}", flush=True)

watch = DEFAULT_WATCH_COLS + [
    "going_is_all_weather", "race_complexity_v2", "race_market_entropy",
    "trainer_form_zscore", "jockey_form_zscore", "trainer_hot_strike_rate",
]
report = build_coverage_report(df, "training_rebuilt_10", watch_cols=watch)
with open("reports/improvement/10/coverage_training_rebuilt.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, default=str)
print(f"[{time.monotonic()-t0:.1f}s] wrote coverage report", flush=True)
print("date range:", report["date_min"], "->", report["date_max"])
print("rows:", report["rows"])
print("market_type counts:", df["market_type"].value_counts(dropna=False).to_dict()
      if "market_type" in df.columns else "NA")
print("BUILD_MATRIX_10_COMPLETE", flush=True)
