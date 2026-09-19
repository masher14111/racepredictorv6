"""Step 09 audit A3 — is there a placeholder canonical id pooling unrelated
runners into one fake jockey/trainer entity, and do the positional same-day
priors really include LATER off-times?

Read-only.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from utils.normalizer import _canonical_id  # noqa: E402

SRC = Path("data/audit/06/training_refreshed.parquet")
PLACEHOLDER = "e3f0bf10b8cc4488"


def main() -> int:
    df = pd.read_parquet(SRC, columns=[
        "race_uid", "race_date", "race_time", "venue", "market_type", "horse_id",
        "jockey_id", "jockey_name", "trainer_id", "trainer_name", "position"])
    print(f"source: {SRC}  rows={len(df):,}")

    for probe in ("", " ", "unknown", "None", "nan"):
        print(f"  _canonical_id({probe!r}) = {_canonical_id(probe)}")

    for key, name in (("jockey_id", "jockey_name"), ("trainer_id", "trainer_name")):
        m = df[key].astype(str) == PLACEHOLDER
        names = df.loc[m, name].astype(str).value_counts(dropna=False).head(5)
        print(f"\n[{key} == {PLACEHOLDER}] rows={int(m.sum()):,} "
              f"({m.mean():.2%} of all rows)  distinct_horses={df.loc[m,'horse_id'].nunique():,}"
              f"  distinct_races={df.loc[m,'race_uid'].nunique():,}")
        print(f"   {name} values: {dict(names)}")
        print(f"   known-position rows in this pool: "
              f"{int(df.loc[m, 'position'].notna().sum()):,}")

    # Do positional same-day priors really include LATER off-times? Take the
    # busiest trainer-day and show the frame order vs. the off-time order.
    print("\n--- same-day ordering evidence ---")
    real = df[(df["trainer_id"].astype(str) != PLACEHOLDER) & df["trainer_id"].notna()]
    grp = real.groupby(["trainer_id", "race_date"], sort=False)
    sizes = grp["race_uid"].nunique().sort_values(ascending=False)
    tid, day = sizes.index[0]
    block = real[(real["trainer_id"] == tid) & (real["race_date"] == day)]
    block = block.drop_duplicates("race_uid")
    print(f"trainer={tid} day={day.date()} races={len(block)}")
    order = list(zip(block["race_uid"], block["race_time"].astype(str),
                     block["position"]))
    for i, (uid, t, pos) in enumerate(order):
        print(f"   frame_pos={i}  {uid}  race_time={t}  position={pos}")
    times = [str(t) for _, t, _ in order]
    print(f"frame order == off-time order? {times == sorted(times)}")
    inversions = sum(1 for i in range(len(times)) for j in range(i)
                     if times[j] > times[i])
    print(f"inversions (a counted 'prior' that ran LATER): {inversions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
