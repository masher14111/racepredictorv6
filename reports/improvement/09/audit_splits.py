"""Step 09 audit B — independently re-derive step 02's split claims.

Does NOT reuse reports/improvement/02/reproduce_overlap.py. Differences that
matter for an independent check:
  * applies models/train.py's real WIN-market restriction BEFORE splitting
    (step 02's script did not; on a market-un-collapsed matrix that changes the
    populations being split);
  * also checks the TUNER's grouped CV folds (step 02's script never did);
  * checks chronological admissibility (fit <= early_stop <= calib <= test)
    at group granularity, not just disjointness;
  * checks determinism under a real row shuffle of the live matrix;
  * checks backtest/splitter.py's load-bearing premise (a race_uid never spans
    two race_date values, so a day-level cut cannot split a race) on real data;
  * runs on the frozen baseline AND the step-06 candidate matrix.

Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from models.split_utils import (  # noqa: E402
    chronological_group_split, group_time_series_split, race_group_overlap)
from models.targets import add_targets  # noqa: E402
from models.train import _group_carve, _time_split  # noqa: E402
from utils.config_loader import get_config  # noqa: E402

MATRICES = {
    "frozen  data/features/training.parquet": Path("data/features/training.parquet"),
    "cand-06 data/audit/06/training_refreshed.parquet":
        Path("data/audit/06/training_refreshed.parquet"),
}


def check_matrix(label: str, path: Path, cfg: dict) -> list[str]:
    fails: list[str] = []
    df = pd.read_parquet(path)
    print(f"\n=== {label} ===")
    print(f"rows={len(df):,}  race_uid={df['race_uid'].nunique():,}  "
          f"dates={df['race_date'].min().date()}..{df['race_date'].max().date()}")

    # --- premise check: a race_uid never spans two race_date / venue values ---
    spans = df.groupby("race_uid", sort=False).agg(
        d=("race_date", "nunique"), v=("venue", "nunique"))
    bad_d, bad_v = int((spans["d"] > 1).sum()), int((spans["v"] > 1).sum())
    print(f"race_uid spanning >1 race_date: {bad_d}   >1 venue: {bad_v}")
    if bad_d:
        fails.append(f"{label}: {bad_d} race_uid span >1 race_date "
                     f"(backtest/splitter.py's day-cut premise broken)")
    if bad_v:
        fails.append(f"{label}: {bad_v} race_uid span >1 venue")

    df = add_targets(df, int(cfg.get("show_positions", 3)))
    if "market_type" in df.columns:               # train.py does this FIRST
        n0 = len(df)
        df = df[df["market_type"].astype("string").str.upper() == "WIN"].copy()
        print(f"WIN-market restriction: {n0:,} -> {len(df):,} rows")
    df = df.reset_index(drop=True)
    if len(df) == 0:
        fails.append(f"{label}: no WIN rows after restriction")
        return fails

    test_size = float(cfg.get("test_size", 0.2))
    calib_frac = float(cfg.get("calibration_size", 0.15))
    train_df, test_df = _time_split(df, test_size)
    ov = race_group_overlap(
        (np.arange(len(train_df)), train_df["race_uid"].to_numpy()),
        (np.arange(len(test_df)), test_df["race_uid"].to_numpy()))
    print(f"train={len(train_df):,} test={len(test_df):,}  shared race_uid={ov}")
    if ov:
        fails.append(f"{label}: train/test shares {ov} race_uid")
    if not train_df["race_date"].max() <= test_df["race_date"].min():
        fails.append(f"{label}: train max date > test min date")

    for target in list(cfg.get("targets", ["won", "placed_2", "showed"])):
        tr = train_df.loc[train_df[target].notna()].reset_index(drop=True)
        te = test_df.loc[test_df[target].notna()].reset_index(drop=True)
        core, cal = _group_carve(tr, calib_frac)
        fit, es = _group_carve(core, 0.10)
        parts = {"fit": fit, "early_stop": es, "calib": cal, "test": te}
        names = list(parts)
        shared_total = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                s = race_group_overlap(
                    (np.arange(len(parts[names[i]])), parts[names[i]]["race_uid"].to_numpy()),
                    (np.arange(len(parts[names[j]])), parts[names[j]]["race_uid"].to_numpy()))
                shared_total += s
                if s:
                    fails.append(f"{label}/{target}: {names[i]}~{names[j]} shares {s} race_uid")
        order_ok = (fit["race_date"].max() <= es["race_date"].min()
                    and es["race_date"].max() <= cal["race_date"].min()
                    and cal["race_date"].max() <= te["race_date"].min())
        if not order_ok:
            fails.append(f"{label}/{target}: chronological order violated "
                         f"(fit<=early_stop<=calib<=test)")
        # tuner CV folds, on core only (what models/train.py passes to run_study)
        folds = list(group_time_series_split(
            core["race_uid"].to_numpy(), core["race_date"].to_numpy(),
            int(cfg.get("tuning", {}).get("cv_folds", 3) or 3)))
        fold_shared = 0
        for k, (tr_i, va_i) in enumerate(folds):
            fold_shared += race_group_overlap(
                (tr_i, core["race_uid"].to_numpy()), (va_i, core["race_uid"].to_numpy()))
            if core["race_date"].to_numpy()[tr_i].max() > core["race_date"].to_numpy()[va_i].min():
                fails.append(f"{label}/{target}: CV fold {k} train date > val min date")
        # folds must also never touch calib/test rows
        cv_vs_cal = race_group_overlap(
            (np.arange(len(core)), core["race_uid"].to_numpy()),
            (np.arange(len(cal)), cal["race_uid"].to_numpy()))
        cv_vs_test = race_group_overlap(
            (np.arange(len(core)), core["race_uid"].to_numpy()),
            (np.arange(len(te)), te["race_uid"].to_numpy()))
        if cv_vs_cal or cv_vs_test:
            fails.append(f"{label}/{target}: tuning core touches calib({cv_vs_cal})"
                         f"/test({cv_vs_test})")
        print(f"[{target}] fit={len(fit):,} es={len(es):,} calib={len(cal):,} "
              f"test={len(te):,} | pairwise shared={shared_total} "
              f"chrono_ok={order_ok} cv_folds={len(folds)} cv_shared={fold_shared} "
              f"core~calib={cv_vs_cal} core~test={cv_vs_test}")

    # determinism under a real row shuffle of the live matrix
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(df))
    sh = df.iloc[perm].reset_index(drop=True)
    a_l, a_r = chronological_group_split(
        df["race_uid"].to_numpy(), df["race_date"].to_numpy(), test_size)
    b_l, b_r = chronological_group_split(
        sh["race_uid"].to_numpy(), sh["race_date"].to_numpy(), test_size)
    same = (set(df["race_uid"].to_numpy()[a_r].tolist())
            == set(sh["race_uid"].to_numpy()[b_r].tolist()))
    print(f"shuffle-invariant test-group membership: {same}")
    if not same:
        fails.append(f"{label}: split not invariant to row order")
    return fails


def main() -> int:
    cfg = get_config().get("model", {})
    all_fails: list[str] = []
    for label, path in MATRICES.items():
        if not path.exists():
            print(f"\n=== {label} === MISSING, skipped")
            continue
        all_fails += check_matrix(label, path, cfg)
    print("\n" + "=" * 60)
    if all_fails:
        print("FAIL:")
        for f in all_fails:
            print("  -", f)
        return 1
    print("PASS: every split/calibration/tuning boundary is race-disjoint, "
          "chronologically admissible and row-order invariant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
