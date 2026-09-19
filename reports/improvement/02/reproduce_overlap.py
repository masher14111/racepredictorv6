"""Step 02 evidence: reproduce the stage-01 overlap defect and show it is now
zero on the real training matrix, using the exact boundary chain models.train
now uses (whole-race chronological split -> calibration carve -> early-stop
carve), for every configured target.

Run:  .venv/Scripts/python.exe reports/improvement/02/reproduce_overlap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.targets import add_targets  # noqa: E402
from models.train import _group_carve, _sample_weights, _time_split  # noqa: E402
from models.split_utils import race_group_overlap  # noqa: E402
from utils.config_loader import get_config  # noqa: E402

TRAIN_PARQUET = Path(__file__).resolve().parents[3] / "data" / "features" / "training.parquet"


def main() -> None:
    cfg = get_config().get("model", {})
    test_size = float(cfg.get("test_size", 0.2))
    calib_frac = float(cfg.get("calibration_size", 0.15))
    targets = list(cfg.get("targets", ["won", "placed_2", "showed"]))

    df = pd.read_parquet(TRAIN_PARQUET)
    df = add_targets(df, int(cfg.get("show_positions", 3)))
    df["_sample_weight"] = _sample_weights(df, float(cfg.get("max_sample_weight", 20.0)))

    print(f"rows={len(df)} unique race_uid={df['race_uid'].nunique()} "
          f"test_size={test_size} calibration_size={calib_frac}")

    train_df, test_df = _time_split(df, test_size)
    overlap = race_group_overlap(
        (np.arange(len(train_df)), train_df["race_uid"].to_numpy()),
        (np.arange(len(test_df)), test_df["race_uid"].to_numpy()),
    )
    print(f"\n[whole-df] train={len(train_df)} test={len(test_df)}  "
          f"train/test shared race_uid = {overlap}")
    assert train_df["race_date"].max() <= test_df["race_date"].min()

    any_overlap = False
    for target in targets:
        tr_sub = train_df.loc[train_df[target].notna()].reset_index(drop=True)
        te_sub = test_df.loc[test_df[target].notna()].reset_index(drop=True)
        core_df, cal_df = _group_carve(tr_sub, calib_frac)
        fit_df, es_df = _group_carve(core_df, 0.10)

        parts = {
            "test": te_sub["race_uid"].to_numpy(),
            "calib": cal_df["race_uid"].to_numpy(),
            "fit": fit_df["race_uid"].to_numpy(),
            "early_stop": es_df["race_uid"].to_numpy(),
        }
        print(f"\n[target={target}] rows: fit={len(fit_df)} early_stop={len(es_df)} "
              f"calib={len(cal_df)} test={len(te_sub)}")
        names = list(parts)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                shared = len(set(parts[a].tolist()) & set(parts[b].tolist()))
                flag = "" if shared == 0 else "  <-- LEAK"
                print(f"  shared race_uid {a}/{b}: {shared}{flag}")
                any_overlap = any_overlap or shared > 0

    print("\nRESULT:", "LEAK DETECTED" if any_overlap else "zero overlap on every boundary, every target")


if __name__ == "__main__":
    main()
