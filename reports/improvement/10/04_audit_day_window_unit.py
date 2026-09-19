"""Step 09 audit F7 — the trainer/jockey "form cycle" day windows were 1,000x
too long on this environment's data.

``features/_trailing_fast._windowed_win_rate_days_core`` compared a hard-coded
NANOSECONDS-per-day constant against ``race_date.asi8``. pandas 3 parses this
project's date column at MICROSECOND resolution, so ``window_days=14`` actually
subtracted 14,000 days. Effect: ``trainer_hot_strike_rate`` (nominally 14 days)
and its 90-day baseline were the SAME all-history rate, so
``trainer_form_zscore`` (short minus baseline) collapsed to ~0 and the feature
carried no form-cycle signal at all.

This script reports the resolution actually in use, and contrasts the repaired
14/90-day windows with the old effective 14,000/90,000-day ones.

Read-only. Usage: python reports/improvement/09/audit_day_window_unit.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from features import _trailing_fast as tf  # noqa: E402
from features._trainer_form import add_trainer_form  # noqa: E402

SRC = Path("data/audit/10/training_rebuilt.parquet")  # step10 regression gate
DAYS = 200
OLD_SCALE = 1000  # ns-per-day / us-per-day


def main() -> int:
    df = pd.read_parquet(SRC, columns=[
        "race_uid", "race_date", "market_type", "horse_id", "jockey_id",
        "trainer_id", "position"])
    cutoff = df["race_date"].max() - pd.Timedelta(days=DAYS)
    df = df[df["race_date"] >= cutoff].reset_index(drop=True)
    unit = pd.DatetimeIndex(df["race_date"]).unit
    print(f"source: {SRC}  slice rows={len(df):,}  race_date resolution={unit!r}  "
          f"ticks/day={tf._ticks_per_day(df['race_date']):,}")
    if unit != "us":
        print("NOTE: resolution is not microseconds here; the 1,000x factor below "
              "is the measured effect for a microsecond frame.")

    for key in ("trainer_id", "jockey_id"):
        r14, n14 = tf.windowed_win_rate_days(df, key, window_days=14, min_runners=5)
        r90, n90 = tf.windowed_win_rate_days(df, key, window_days=90, min_runners=5)
        ro14, no14 = tf.windowed_win_rate_days(
            df, key, window_days=14 * OLD_SCALE, min_runners=5)
        ro90, _ = tf.windowed_win_rate_days(
            df, key, window_days=90 * OLD_SCALE, min_runners=5)
        both_new = ~np.isnan(r14) & ~np.isnan(r90)
        both_old = ~np.isnan(ro14) & ~np.isnan(ro90)
        print(f"\n[{key}]")
        print(f"  repaired  : mean n(14d)={n14.mean():.2f}  mean n(90d)={n90.mean():.2f}  "
              f"short==long on {np.isclose(r14[both_new], r90[both_new]).mean():.2%} "
              f"of comparable rows")
        print(f"  pre-repair: mean n(14d)={no14.mean():.2f}  "
              f"short==long on {np.isclose(ro14[both_old], ro90[both_old]).mean():.2%} "
              f"of comparable rows  (window was {14 * OLD_SCALE:,} days)")

    tf_out = add_trainer_form(df)
    z = pd.to_numeric(tf_out["trainer_form_zscore"], errors="coerce")
    print(f"\nrepaired trainer_form_zscore: non-null {z.notna().mean():.2%}  "
          f"exactly-zero among non-null {float((z[z.notna()] == 0).mean()):.2%}  "
          f"std={z.std():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
