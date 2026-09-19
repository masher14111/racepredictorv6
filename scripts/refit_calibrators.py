"""Refit v3 (priced) and v3nf (price-free) with calibrators, reusing the stored
Optuna params. Loads the prebuilt training matrix to skip the feature rebuild.

The model is refit on the core split with a time-ordered calibration slice held
out; the calibrator (method per config: auto → isotonic-vs-sigmoid) is fit on
that slice and persisted as catboost_<target>_<tag>_calib.pkl. This is the fix
for audit C1 (v3 served uncalibrated).

    python -m scripts.refit_calibrators
"""
import pandas as pd

from models.features import PRICE_FREE_FEATURE_COLS
from models.train import train

_TRAINING = "data/features/training.parquet"


def main() -> None:
    df = pd.read_parquet(_TRAINING)
    print(f"loaded {len(df):,} training rows")

    print("\n=== v3 (priced) ===")
    train(df=df, reuse_params=True, version_tag="v3")

    print("\n=== v3nf (price-free) ===")
    train(df=df, reuse_params=True, version_tag="v3nf",
          feature_cols=PRICE_FREE_FEATURE_COLS)

    print("\ndone")


if __name__ == "__main__":
    main()
