"""Re-parse saved SL raw pages with the current parser, then re-join + write.

Used after the timezone fix to regenerate data/historical/betsp.parquet WITHOUT
re-fetching: every detail page is already archived gzip under data/historical/raw.
"""
import glob
import gzip
import os
import sys
import time
from datetime import date

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper.betsp_historical as bh
from scraper.betsp import joiner, writer
from scraper.betsp.results.base import RawResult
from scraper.betsp.results.sporting_life import SportingLife
from utils.timezone import now as _now

RAW_ROOT = "data/historical/raw/sporting_life"
cfg = bh._load_cfg()
src = SportingLife()

t0 = time.monotonic()
files = glob.glob(os.path.join(RAW_ROOT, "**", "*.html.gz"), recursive=True)
print(f"re-parsing {len(files)} raw detail pages...", flush=True)

rows = []
bad = 0
for i, path in enumerate(files):
    day_str = os.path.basename(os.path.dirname(path))  # .../{date}/{hash}.html.gz
    try:
        d = date.fromisoformat(day_str)
    except ValueError:
        d = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            html = f.read()
        raw = RawResult(source="sporting_life", race_date=d, url=path, html=html)
        rows.extend(src.parse(raw))
    except Exception:  # noqa: BLE001
        bad += 1
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(files)} files, {len(rows)} rows", flush=True)

print(f"parsed {len(rows)} result rows ({bad} unreadable files)", flush=True)

backbone = pd.read_parquet("data/historical/betsp_backbone.parquet")
priority = cfg.get("source_priority", ["sporting_life"])
joined = joiner.join(backbone, rows, priority=priority)
joined["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
writer.write(joined, path=cfg.get("parquet_path", "data/historical/betsp.parquet"))

filled = int(joined["position"].notna().sum())
print(f"=== reparse done in {(time.monotonic()-t0)/60:.1f} min ===", flush=True)
print(f"rows: {len(joined)}  position-filled: {filled} "
      f"({filled/max(1,len(joined)):.1%})", flush=True)
print("REPARSE_COMPLETE", flush=True)
