"""Step 09 audit F1 — how much did repairing the prior cut actually move the
features? Step 10 needs this to know the corrected baseline is not the old one.

Isolates F1 only: both arms use the CURRENT (D38, runner-aware) market-duplicate
collapse; the reference arm restores the pre-repair POSITIONAL prior cut inside a
date-only sort, which is exactly what `_bounds(strict=False)` used to do.

Read-only. Usage: python reports/improvement/09/audit_f1_feature_delta.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features import _trailing_fast as tf  # noqa: E402

SRC = Path("data/audit/06/training_refreshed.parquet")
DAYS = 200
RUNS, MONTHS, PLACES = 20, 12, 3


def _old_windowed_rates(df, keys):
    """The pre-repair algorithm: sort each entity group by date-only and cut the
    priors at the row's POSITION within that group."""
    keep, gid = tf._dedupe_group_ids(df, keys)
    sub = df.loc[keep].reset_index(drop=True)
    n = len(sub)
    win = np.full(n, np.nan)
    runs_out = np.zeros(n, dtype="int64")
    d_all = tf._asi8(sub["race_date"])
    lower_all = tf._asi8(sub["race_date"] - pd.DateOffset(months=MONTHS))
    pos = pd.to_numeric(sub["position"], errors="coerce").to_numpy(dtype="float64")
    contrib = ~np.isnan(pos)
    is_win = contrib & (pos == 1)
    for idx in tf._groups(sub, keys, dropna=True).values():
        order = idx[np.argsort(d_all[idx], kind="stable")]
        d = d_all[order]
        lo = np.searchsorted(d, lower_all[order], side="left")
        end = np.arange(len(d))
        start = np.minimum(np.maximum(lo, end - RUNS), end)
        cc, cw = tf._cum(contrib[order]), tf._cum(is_win[order])
        nn = cc[end] - cc[start]
        nz = nn > 0
        win[order] = np.where(nz, (cw[end] - cw[start]) / np.where(nz, nn, 1.0), np.nan)
        runs_out[order] = nn.astype("int64")
    return win[gid], runs_out[gid]


def main() -> int:
    df = pd.read_parquet(SRC, columns=[
        "race_uid", "race_date", "market_type", "horse_id", "jockey_id",
        "trainer_id", "position"])
    cutoff = df["race_date"].max() - pd.Timedelta(days=DAYS)
    df = df[df["race_date"] >= cutoff].reset_index(drop=True)
    print(f"source: {SRC}  slice rows={len(df):,}  last {DAYS} days")

    for key in ("trainer_id", "jockey_id", "horse_id"):
        new_w, new_n = tf.windowed_rates(df, [key], runs=RUNS, months=MONTHS,
                                         places=PLACES, strict=False, dropna=True)[::2]
        old_w, old_n = _old_windowed_rates(df, [key])
        both = ~np.isnan(new_w) & ~np.isnan(old_w)
        delta = np.abs(new_w[both] - old_w[both])
        moved = int((delta > 1e-12).sum())
        dn = (new_n.astype("int64") - old_n.astype("int64"))
        print(f"\n[{key}] comparable rows={int(both.sum()):,}")
        print(f"  rate moved on {moved:,} ({moved / max(int(both.sum()), 1):.2%})  "
              f"mean|delta|={delta.mean():.4f}  p95={np.quantile(delta, 0.95):.4f}  "
              f"max={delta.max():.4f}")
        print(f"  n_runs delta: mean={dn.mean():+.3f}  min={dn.min()}  max={dn.max()}  "
              f"rows with fewer priors now={int((dn < 0).sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
