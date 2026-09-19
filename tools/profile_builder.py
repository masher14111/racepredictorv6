"""Time each stage of the feature-derive pipeline on a sample to find hot spots."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from features import derive, engine
from features.fuse import fuse_sources
from features.builder import _load_cfg

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50000

cfg = _load_cfg()
uni = pd.read_parquet("data/unified_races.parquet")
# Sample by whole entities is ideal, but a head slice keeps date order intact and
# is enough to expose the per-stage cost ratio.
df = uni.head(N).copy()
print(f"profiling on {len(df)} rows", flush=True)


def timed(name, fn, frame):
    t = time.monotonic()
    out = fn(frame)
    dt = time.monotonic() - t
    print(f"  {dt:8.2f}s  {name}", flush=True)
    return out


out = timed("fuse_sources", fuse_sources, df)
out["race_date"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
out = timed("add_going_speed", lambda f: derive.add_going_speed(f, cfg["going_speed_map"]), out)
out = timed("add_class_change", derive.add_class_change, out)
out = timed("add_distance_furlongs", derive.add_distance_furlongs, out)
out = timed("add_recent_form", derive.add_recent_form, out)
out = timed("add_odds_features", derive.add_odds_features, out)
out = timed("add_rating_rank", derive.add_rating_rank, out)
out = timed("add_all_trailing_rates  <<<", lambda f: derive.add_all_trailing_rates(f, cfg), out)
out = timed("add_speed_figures  <<<", lambda f: engine.add_speed_figures(f, cfg), out)
out = timed("add_combo_win_rate  <<<", lambda f: engine.add_combo_win_rate(f, cfg), out)
out = timed("add_going_preference  <<<", lambda f: engine.add_going_preference(f, cfg), out)
out = timed("add_pace_bias", engine.add_pace_bias, out)
out = timed("add_ew_value_index", engine.add_ew_value_index, out)
out = timed("add_odds_delta", engine.add_odds_delta, out)
out = timed("add_race_complexity", engine.add_race_complexity, out)
print("PROFILE_COMPLETE", flush=True)
