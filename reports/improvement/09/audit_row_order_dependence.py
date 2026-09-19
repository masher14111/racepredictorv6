"""Step 09 audit — is a trailing feature a function of the DATA or of the frame's
ROW ORDER?

A point-in-time aggregate must be reproducible: shuffling the rows of the input
matrix must not move a single value. Before the F1 repair the non-strict prior
cut was positional, so it did (21.94% of rows), and even a date-strict variant
moved 6.78% because the ``lookback_runs`` tail cap cut into a block of same-date
rows whose internal order was arbitrary.

Read-only. Usage: python reports/improvement/09/audit_row_order_dependence.py
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
SEED = 20260919


def _slice() -> pd.DataFrame:
    df = pd.read_parquet(SRC, columns=[
        "race_uid", "race_date", "market_type", "horse_id", "jockey_id",
        "trainer_id", "position"])
    cutoff = df["race_date"].max() - pd.Timedelta(days=DAYS)
    return df[df["race_date"] >= cutoff].reset_index(drop=True)


def _key(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [df["race_uid"].astype(str), df["horse_id"].astype(str),
         df["market_type"].astype(str)])


def _compare(name, base, shuf, vals_base, vals_shuf) -> None:
    a = pd.Series(vals_base, index=_key(base))
    b = pd.Series(vals_shuf, index=_key(shuf))
    a = a[~a.index.duplicated()]
    b = b[~b.index.duplicated()].reindex(a.index)
    both = a.notna() & b.notna()
    delta = (a[both] - b[both]).abs()
    n_diff = int((delta > 1e-12).sum())
    only_one = int((a.notna() ^ b.notna()).sum())
    print(f"{name}: rows differing after ROW SHUFFLE: {n_diff} / {int(both.sum())} "
          f"({n_diff / max(int(both.sum()), 1):.2%})  max|delta|="
          f"{(delta.max() if len(delta) else 0.0):.4f}  null-mismatch={only_one}")


def main() -> int:
    base = _slice()
    rng = np.random.default_rng(SEED)
    shuf = base.iloc[rng.permutation(len(base))].reset_index(drop=True)
    print(f"source: {SRC}  slice rows={len(base):,}  last {DAYS} days")

    for key, strict in (("trainer_id", False), ("jockey_id", False),
                        ("horse_id", True)):
        wb, _, rb = tf.windowed_rates(base, [key], runs=RUNS, months=MONTHS,
                                      places=PLACES, strict=strict, dropna=True)
        ws, _, rs = tf.windowed_rates(shuf, [key], runs=RUNS, months=MONTHS,
                                      places=PLACES, strict=strict, dropna=True)
        _compare(f"windowed_rates[{key}, strict={strict}] rate", base, shuf, wb, ws)
        _compare(f"windowed_rates[{key}, strict={strict}] n_runs", base, shuf,
                 rb.astype(float), rs.astype(float))

    for key in ("trainer_id", "jockey_id"):
        rb, cb = tf.windowed_win_rate_days(base, key, window_days=14, min_runners=5)
        rs_, cs = tf.windowed_win_rate_days(shuf, key, window_days=14, min_runners=5)
        _compare(f"windowed_win_rate_days[{key}] rate", base, shuf, rb, rs_)
        _compare(f"windowed_win_rate_days[{key}] n", base, shuf,
                 cb.astype(float), cs.astype(float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
