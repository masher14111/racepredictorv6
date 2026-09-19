"""Scratch: regenerate the full (model-10) training matrix from the betSP backbone.

Writes data/features/training_full.parquet so the feature-selection analysis can
reload it without re-deriving (the derive pipeline is the slow part).
"""
import time

import pandas as pd

from features.builder import build_training_matrix
from utils.normalizer import normalize

OUT = "data/features/training_full.parquet"


def main():
    t0 = time.time()
    betsp = pd.read_parquet("data/historical/betsp.parquet")
    print(f"betsp rows={len(betsp)}  ({time.time()-t0:.1f}s)")

    t1 = time.time()
    unified = normalize(sources={"betsp": betsp}, write=False)
    print(f"unified rows={len(unified)}  has race_time={'race_time' in unified.columns}  "
          f"({time.time()-t1:.1f}s)")

    t2 = time.time()
    df = build_training_matrix(unified=unified, write=False)
    print(f"training rows={len(df)}  cols={len(df.columns)}  ({time.time()-t2:.1f}s)")

    from models.features import FEATURE_COLS
    present = [c for c in FEATURE_COLS if c in df.columns]
    print(f"FEATURE_COLS present {len(present)}/{len(FEATURE_COLS)}")
    print("missing:", [c for c in FEATURE_COLS if c not in df.columns])

    df.to_parquet(OUT, engine="pyarrow", index=False)
    print(f"wrote {OUT}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
