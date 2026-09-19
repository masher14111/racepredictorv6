"""Full historical backfill: Betfair SP backbone (from checkpoint) + SL results.

Reuses the existing backbone checkpoint (force=False) and runs SL enrichment over
the full configured year range, then joins + writes data/historical/betsp.parquet.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper.betsp_historical as bh

t0 = time.monotonic()
print("=== backfill start ===", flush=True)
df = bh.fetch()  # force=False -> backbone from checkpoint, enrich + join + write
dt = time.monotonic() - t0
filled = int(df["position"].notna().sum()) if "position" in df.columns else 0
print(f"=== backfill done in {dt/60:.1f} min ===", flush=True)
print(f"rows: {len(df)}  position-filled: {filled} "
      f"({filled/max(1,len(df)):.1%})", flush=True)
print("BACKFILL_COMPLETE", flush=True)
