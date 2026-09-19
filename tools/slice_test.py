"""One-off slice test: prove the SL extractor fix yields a non-empty training matrix.

Runs the REAL pipeline on a 30-day window against the existing backbone checkpoint:
  backbone slice -> SL enrichment -> joiner -> normalizer -> feature builder.
Reports position fill, unified rows, training rows, and label distribution.
"""
import os
import sys
import tempfile
from datetime import date, timedelta

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scraper.betsp_historical as bh
from scraper.betsp import joiner
from utils import normalizer
from features import builder
from utils.timezone import now as _now

START = date(2024, 1, 1)
NDAYS = 30
days = [START + timedelta(days=i) for i in range(NDAYS)]
end = days[-1]

cfg = bh._load_cfg()

print(f"=== Slice test: {START} .. {end} ({NDAYS} days) ===", flush=True)

# Cache the joined slice so builder iterations don't re-fetch (5 min of requests).
CACHE = "data/_slice_joined.parquet"
if os.path.exists(CACHE):
    joined = pd.read_parquet(CACHE)
    print(f"loaded cached join: {len(joined)} rows", flush=True)
else:
    # 1. Backbone slice from the existing checkpoint (no re-fetch).
    bb = pd.read_parquet("data/historical/betsp_backbone.parquet")
    bb["_d"] = pd.to_datetime(bb["race_date"], utc=True, errors="coerce").dt.date
    mask = (bb["_d"] >= START) & (bb["_d"] <= end)
    bb_slice = bb[mask].drop(columns="_d").reset_index(drop=True)
    print(f"backbone slice rows: {len(bb_slice)}", flush=True)

    # 2. SL enrichment for those days (real fetch through curl_cffi + proxy).
    with tempfile.TemporaryDirectory() as raw_root:
        rows = bh._fetch_enrichment(cfg, days, raw_root, bh._results_get_html)
    print(f"result rows fetched: {len(rows)} (with position: "
          f"{sum(1 for r in rows if r.position is not None)})", flush=True)

    # 3. Join enrichment onto the backbone slice.
    priority = cfg.get("source_priority", ["sporting_life"])
    joined = joiner.join(bb_slice, rows, priority=priority)
    joined["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
    joined.to_parquet(CACHE, index=False)

fill = joined["position"].notna().sum()
print(f"join: {fill}/{len(joined)} rows position-filled "
      f"({fill/max(1,len(joined)):.1%})", flush=True)

# 4. Normalize betsp -> unified canonical schema.
unified = normalizer.normalize(sources={"betsp": joined}, write=False)
print(f"unified rows (post-validate/dedupe): {len(unified)}", flush=True)
print(f"unified position non-null: {unified['position'].notna().sum()}", flush=True)

# 5. Build the labelled training matrix.
train = builder.build_training_matrix(unified=unified, write=False)
print(f"=== TRAINING MATRIX: {len(train)} rows, {train.shape[1]} cols ===", flush=True)

# 6. Prove the matrix is trainable: derive targets + fit CatBoost (no Optuna).
from models.targets import add_targets
from models import train as trainer

labelled = add_targets(train, 3)
for label in ("won", "placed_2", "showed"):
    if label in labelled.columns:
        vc = labelled[label].value_counts(dropna=False).to_dict()
        print(f"  label {label}: {vc}", flush=True)

with tempfile.TemporaryDirectory() as mdir:
    meta = trainer.train(df=train, no_tune=True, model_dir=mdir)
if meta is None:
    print("TRAIN FAILED (empty matrix)", flush=True)
else:
    for tgt, info in meta["targets"].items():
        print(f"  model {tgt}: test_auc={info['test_auc']:.4f}", flush=True)
    print(f"  features used: {len(meta['feature_cols'])}", flush=True)
print("DONE", flush=True)
