"""Step 09 audit A2 — do the jockey/trainer trailing rates read SAME-DAY results?

`features/derive.add_all_trailing_rates` calls `_trailing_fast.windowed_rates`
with ``strict=False``. In `_bounds`, strict=False means ``end = arange(len(d))``
— a purely POSITIONAL "prior" cut, so a row counts every earlier-positioned row
of the same entity that shares its ``race_date`` (date-only, per
docs/improvement/CONTRACTS.md). Within one day the order is the frame's own row
order, which is not the off-time order, so a 13:00 runner's trainer rate can
include that yard's 16:05 winner from later the same afternoon.

Also checks whether a placeholder/canonical-unknown jockey/trainer id pools many
distinct runners into one fake mega-entity.

Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features import _trailing_fast  # noqa: E402

SRC = Path("data/audit/06/training_refreshed.parquet")
COLS = ["race_uid", "race_date", "race_time", "venue", "market_type", "horse_id",
        "jockey_id", "jockey_name", "trainer_id", "trainer_name", "position"]


def same_day_prior_counts(df, key):
    """Per row: how many of its counted priors share its race_date (= lookahead
    risk under the positional strict=False cut)."""
    keep, gid = _trailing_fast._dedupe_group_ids(df, [key])
    sub = df.loc[keep].reset_index(drop=True)
    event = _trailing_fast._race_event_key(sub)
    d, self_ns, prior_ns, _ = _trailing_fast._order_instants(sub, event)
    rank = _trailing_fast._chrono_rank(sub, event, prior_ns)
    lower = _trailing_fast._asi8(sub["race_date"] - pd.DateOffset(months=12))
    n_same = np.zeros(len(sub), dtype="int64")
    for idx in _trailing_fast._groups(sub, [key], dropna=True).values():
        order = idx[np.argsort(rank[idx], kind="stable")]
        dd = d[order]
        start, end = _trailing_fast._bounds(dd, lower[order], self_ns[order],
                                            prior_ns[order], 20, strict=False)
        # strictly-earlier-in-DATE count
        strict_end = np.searchsorted(dd, dd, side="left")
        counted_same_day = np.maximum(end - np.maximum(start, strict_end), 0)
        n_same[order] = counted_same_day
    return n_same[gid]


def main() -> int:
    df = pd.read_parquet(SRC, columns=COLS).reset_index(drop=True)
    print(f"source: {SRC}  rows={len(df):,}")
    print("row order is time-ordered within a day? ", end="")
    d0 = df[df["race_date"] == df["race_date"].max()]
    print(f"(sample day {d0['race_date'].max().date()}, "
          f"race_time monotonic across rows: "
          f"{bool(pd.Series(d0['race_time'].astype(str)).is_monotonic_increasing)})")

    for key in ("trainer_id", "jockey_id", "horse_id"):
        n_same = same_day_prior_counts(df, key)
        rows = df[key].notna().to_numpy()
        print(f"\n[{key}] rows with >=1 SAME-DAY prior counted: "
              f"{int((n_same[rows] > 0).sum()):,} / {int(rows.sum()):,} "
              f"({(n_same[rows] > 0).mean():.2%});  mean same-day priors="
              f"{n_same[rows].mean():.2f}  max={int(n_same.max())}")

    print("\n--- placeholder / pooled entity ids ---")
    for key, name in (("jockey_id", "jockey_name"), ("trainer_id", "trainer_name")):
        vc = df[key].value_counts(dropna=True)
        print(f"\n[{key}] distinct={len(vc):,}  null_rows={int(df[key].isna().sum()):,}")
        top = vc.head(5)
        for v, c in top.items():
            nm = df.loc[df[key] == v, name].dropna().unique()[:3]
            horses = df.loc[df[key] == v, "horse_id"].nunique()
            print(f"   id={v!r:>18}  rows={c:,}  distinct_horses={horses:,}  names={list(nm)}")
        # biggest single-race pile-up
        g = df[df[key].notna()].groupby([key, "race_uid"], sort=False)["horse_id"].nunique()
        worst = g.sort_values(ascending=False).head(3)
        print(f"   worst (entity,race) horse pile-ups: {list(worst.items())[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
