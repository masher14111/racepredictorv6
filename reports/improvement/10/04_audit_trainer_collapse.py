"""Step 09 audit A1 — does step 04's market-duplicate collapse also swallow a
trainer's DISTINCT runners in the same race?

`features/_trailing_fast._dedupe_group_ids(df, keys)` collapses rows onto one
representative per (keys..., race event). For a horse-keyed window
(`keys=["horse_id"]`) that is exactly the WIN/PLACE market-duplicate collapse
step 04 intended. For a TRAINER-keyed window (`keys=["trainer_id"]`, used by
derive.add_all_trailing_rates -> trainer_win_rate and by
features/_trainer_form.py -> trainer_hot_strike_rate/trainer_form_zscore) the
key has no horse in it, so two DIFFERENT horses the same trainer saddles in the
same race collapse to one row too — the yard's other runners (and their
results) disappear from its own trailing form.

Read-only. Measures the real magnitude on the step-06 candidate matrix.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features import _trailing_fast  # noqa: E402

SRC = Path("data/audit/10/training_rebuilt.parquet")  # step10 regression gate
COLS = ["race_uid", "race_date", "venue", "market_type", "horse_id",
        "jockey_id", "trainer_id", "position", "trainer_win_rate",
        "trainer_hot_strike_rate"]


def main() -> int:
    df = pd.read_parquet(SRC, columns=COLS).reset_index(drop=True)
    print(f"source: {SRC}  rows={len(df):,}")

    for key in ("trainer_id", "jockey_id", "horse_id"):
        sub = df[df[key].notna()]
        # rows the current dedupe keeps
        keep_cur, _ = _trailing_fast._dedupe_group_ids(sub.reset_index(drop=True), [key])
        # rows a horse-aware dedupe (market duplicates only) would keep
        keep_ok, _ = _trailing_fast._dedupe_group_ids(
            sub.reset_index(drop=True), [key, "horse_id"])
        n_cur, n_ok = int(keep_cur.sum()), int(keep_ok.sum())
        print(f"\n[{key}] rows={len(sub):,}  kept_by_current_dedupe={n_cur:,}  "
              f"kept_by_market_only_dedupe={n_ok:,}  "
              f"runner-events_lost={n_ok - n_cur:,} "
              f"({(n_ok - n_cur) / max(n_ok, 1):.2%} of real runner-events)")
        # how many (entity, race) groups carry >1 distinct horse
        g = sub.groupby([key, "race_uid"], sort=False)["horse_id"].nunique()
        print(f"       (entity,race) groups={len(g):,}  with>1 distinct horse="
              f"{int((g > 1).sum()):,} ({(g > 1).mean():.2%})  max_horses={int(g.max())}")

    # Does the loss change the actual feature value? Recompute trainer_win_rate
    # the current way vs. a horse-aware way on a date-contiguous slice.
    print("\n--- effect on trainer_win_rate (recomputed both ways) ---")
    recent = df[df["race_date"] >= (df["race_date"].max() - pd.Timedelta(days=200))]
    recent = recent.reset_index(drop=True)
    print(f"slice rows={len(recent):,}  {recent['race_date'].min()} .. {recent['race_date'].max()}")

    cur_w, _, cur_n = _trailing_fast.windowed_rates(
        recent, ["trainer_id"], runs=20, months=12, places=3,
        strict=False, dropna=True)

    # horse-aware variant: temporarily widen the dedupe key by monkeypatching the
    # race-event key to include horse_id, which is exactly "collapse market
    # duplicates only".
    orig = _trailing_fast._race_event_key
    try:
        _trailing_fast._race_event_key = lambda d: (
            d["race_uid"].astype(str) + "|" + d["horse_id"].astype(str)).to_numpy()
        ok_w, _, ok_n = _trailing_fast.windowed_rates(
            recent, ["trainer_id"], runs=20, months=12, places=3,
            strict=False, dropna=True)
    finally:
        _trailing_fast._race_event_key = orig

    both = ~np.isnan(cur_w) & ~np.isnan(ok_w)
    diff = np.abs(cur_w[both] - ok_w[both])
    print(f"rows with both defined: {both.sum():,}")
    print(f"trainer_win_rate differs on {int((diff > 1e-12).sum()):,} rows "
          f"({(diff > 1e-12).mean():.2%} of them)")
    if diff.size:
        print(f"  mean|delta|={diff.mean():.4f}  p95|delta|={np.percentile(diff, 95):.4f}  "
              f"max|delta|={diff.max():.4f}")
    nd = cur_n[both] - ok_n[both]
    print(f"runs_in_window undercount: mean={nd.mean():.2f}  min={nd.min()}  max={nd.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
