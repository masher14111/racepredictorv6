"""Step 09 audit A2 (magnitude) — of the same-day priors the trainer/jockey
trailing rates actually count, how many ran LATER than the row being scored?

`features/derive.add_all_trailing_rates` uses ``strict=False``, whose "prior"
cut is positional within a group sorted by ``race_date`` (date-only). Rows that
share a date keep the frame's own row order, which is not the off-time order,
so a counted "prior" can be a race that had not yet been run at the scored
row's prediction cutoff. ``race_uid`` carries the real off-time, so the true
ordering is recoverable and the violation is measurable exactly.

Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features import _trailing_fast as tf  # noqa: E402

SRC = Path("data/audit/06/training_refreshed.parquet")
RUNS, MONTHS = 20, 12


def main() -> int:
    df = pd.read_parquet(SRC, columns=[
        "race_uid", "race_date", "market_type", "horse_id", "jockey_id",
        "trainer_id", "position"]).reset_index(drop=True)
    # off-time (minute) parsed out of race_uid -> a true chronological key
    off = pd.to_datetime(
        df["race_uid"].astype(str).str.split("|").str[-1], errors="coerce", utc=True)
    print(f"source: {SRC}  rows={len(df):,}  race_uid off-times parsed: "
          f"{off.notna().mean():.2%}")

    for key in ("trainer_id", "jockey_id"):
        keep, gid = tf._dedupe_group_ids(df, [key])
        sub = df.loc[keep].reset_index(drop=True)
        sub_off = off[keep].reset_index(drop=True).to_numpy()
        event = tf._race_event_key(sub)
        d, self_ns, prior_ns, _ = tf._order_instants(sub, event)
        rank = tf._chrono_rank(sub, event, prior_ns)
        lower = tf._asi8(sub["race_date"] - pd.DateOffset(months=MONTHS))
        counted = later = 0
        rows_with_later = 0
        for idx in tf._groups(sub, [key], dropna=True).values():
            order = idx[np.argsort(rank[idx], kind="stable")]
            dd = d[order]
            start, end = tf._bounds(dd, lower[order], self_ns[order],
                                    prior_ns[order], RUNS, strict=False)
            strict_end = np.searchsorted(dd, dd, side="left")
            o = sub_off[order]
            for i in range(len(order)):
                lo = max(start[i], strict_end[i])
                if end[i] <= lo:
                    continue
                window = o[lo:end[i]]                # counted SAME-DAY priors
                counted += len(window)
                n_later = int(np.sum(window > o[i]))
                later += n_later
                if n_later:
                    rows_with_later += 1
        print(f"\n[{key}] counted same-day priors={counted:,}; of those, "
              f"{later:,} ({later / max(counted, 1):.2%}) ran LATER than the "
              f"row being scored")
        print(f"       rows scored with >=1 not-yet-run 'prior': "
              f"{rows_with_later:,} of {len(sub):,} runner-events "
              f"({rows_with_later / max(len(sub), 1):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
