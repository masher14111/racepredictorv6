"""One-off: retrain v3 (priced) and v3nf (price-free) on the C2-fixed matrix.

Loads the already-regenerated training.parquet from disk (so the ~6-min derive
runs zero times here) and calls models.train.train with reuse_params=True, which
holds each model's tuned hyperparameters constant and only refits + recalibrates.
That isolates the leakage fix as the single changed variable vs the model-09
baseline. v3 consumes the now-honest market features; v3nf is price-free but its
odds-inverse sample weights also de-leak, so we refit it too for a clean pair.
"""
import os

import pandas as pd

from models.features import FEATURE_COLS, PRICE_FREE_FEATURE_COLS
from models.train import train

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN = os.path.join(_BASE, "data", "features", "training.parquet")

df = pd.read_parquet(_TRAIN)
print(f"loaded {len(df)} training rows from {_TRAIN}", flush=True)

print("=== retrain v3 (priced, honest morningwap market features) ===", flush=True)
train(df=df, feature_cols=FEATURE_COLS, version_tag="v3", reuse_params=True)

print("=== retrain v3nf (price-free) ===", flush=True)
train(df=df, feature_cols=PRICE_FREE_FEATURE_COLS, version_tag="v3nf", reuse_params=True)

print("=== retrain complete ===", flush=True)
