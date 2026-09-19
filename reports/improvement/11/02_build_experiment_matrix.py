"""Step 11 / 2: attach measured performance to step 10's frozen matrix.

Reads (never writes) ``data/audit/10/training_rebuilt.parquet`` and this stage's
timing extracts, then writes ONLY into this stage's own experiment directory:
  * ``data/audit/11/training_measured.parquet``
  * ``reports/improvement/11/coverage_features.json``

Also runs, on the real data rather than on fixtures, the two invariance checks
the acceptance criteria name:
  * appending later races leaves every historical feature value unchanged
  * frame order does not change a feature value
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from features import measured_speed as ms  # noqa: E402
from models.features import MEASURED_SPEED_FEATURE_COLS  # noqa: E402
from utils.text_norm import minute_key, norm_horse, norm_venue  # noqa: E402

MATRIX_IN = os.path.join("data", "audit", "10", "training_rebuilt.parquet")
RACES_IN = os.path.join("data", "audit", "11", "measured_timing_races.parquet")
RUNNERS_IN = os.path.join("data", "audit", "11", "measured_timing_runners.parquet")
MATRIX_OUT = os.path.join("data", "audit", "11", "training_measured.parquet")
REPORT_OUT = os.path.join("reports", "improvement", "11", "coverage_features.json")

# Frozen by step 10's run_manifest.json. Reproduced here only to report coverage
# per window; this script never fits or scores anything.
DEV_CORE = ("2024-01-01", "2026-08-07")
DEV_OOS = ("2026-08-08", "2026-08-27")
FINAL_HOLDOUT = ("2026-08-28", "2026-09-17")


def race_key_of(race_uid: pd.Series) -> pd.Series:
    venue = race_uid.str.split("|").str[0].map(norm_venue)
    stamp = race_uid.str.split("|").str[1].map(minute_key)
    return venue + "|" + stamp


def build(matrix: pd.DataFrame, races: pd.DataFrame, runners: pd.DataFrame) -> pd.DataFrame:
    figures = ms.add_speed_figures(runners, races.loc[races["usable"]])
    keep = ["race_key", "horse_key", "measured_speed_pct", "measured_speed_z",
            "beaten_lengths", "winner_speed_yps", "est_speed_yps", "finished",
            "par_level", "par_n", "distance_yards", "winning_time_seconds"]
    figures = figures[keep].drop_duplicates(subset=["race_key", "horse_key"], keep="first")

    out = matrix.copy()
    out["race_key"] = race_key_of(out["race_uid"])
    out["horse_key"] = out["horse_name"].map(norm_horse)
    out = out.merge(figures, on=["race_key", "horse_key"], how="left",
                    suffixes=("", "_measured"))
    return ms.add_measured_pre_race_features(out)


def invariance_checks(matrix: pd.DataFrame, races: pd.DataFrame,
                      runners: pd.DataFrame) -> dict:
    """Rebuild history-only and compare against the full build, on real data."""
    # Reconstruct the world as it stood at the cutoff: every record whose race
    # had already been run, and nothing else. Filtering the timing extracts by
    # their OWN date (not by which races happen to appear in the matrix) is what
    # makes this a genuine "append the future" test -- restricting them to the
    # matrix's races would also silently drop archive races the matrix never
    # carried, and those legitimately inform earlier pars.
    cutoff = DEV_OOS[0]
    early_matrix = matrix.loc[
        matrix["race_date"].astype(str).str.slice(0, 10) < cutoff].reset_index(drop=True)
    early_races = races.loc[races["race_date"].astype(str).str.slice(0, 10) < cutoff]
    early_runners = runners.loc[runners["race_date"].astype(str).str.slice(0, 10) < cutoff]

    full = build(matrix, races, runners)
    partial = build(early_matrix, early_races, early_runners)

    idx = ["race_uid", "market_type", "horse_id"]
    a = full.set_index(idx).loc[partial.set_index(idx).index, MEASURED_SPEED_FEATURE_COLS]
    b = partial.set_index(idx)[MEASURED_SPEED_FEATURE_COLS]
    append_diffs = {c: int((~np.isclose(a[c].astype(float), b[c].astype(float),
                                        equal_nan=True)).sum())
                    for c in MEASURED_SPEED_FEATURE_COLS}

    shuffled_matrix = matrix.sample(frac=1.0, random_state=11).reset_index(drop=True)
    shuffled = build(shuffled_matrix, races.sample(frac=1.0, random_state=12),
                     runners.sample(frac=1.0, random_state=13))
    c = shuffled.set_index(idx).loc[full.set_index(idx).index, MEASURED_SPEED_FEATURE_COLS]
    d = full.set_index(idx)[MEASURED_SPEED_FEATURE_COLS]
    order_diffs = {col: int((~np.isclose(c[col].astype(float), d[col].astype(float),
                                         equal_nan=True)).sum())
                   for col in MEASURED_SPEED_FEATURE_COLS}

    return {
        "rows_compared_append": int(len(b)),
        "append_invariance_diffs": append_diffs,
        "append_invariant": all(v == 0 for v in append_diffs.values()),
        "rows_compared_order": int(len(d)),
        "row_order_diffs": order_diffs,
        "row_order_invariant": all(v == 0 for v in order_diffs.values()),
    }


def coverage(frame: pd.DataFrame) -> dict:
    win = frame.loc[frame["market_type"] == "WIN"].copy()
    win["day"] = win["race_date"].astype(str).str.slice(0, 10)

    def window(name, lo, hi):
        sel = win.loc[(win["day"] >= lo) & (win["day"] <= hi)]
        if sel.empty:
            return {"rows": 0}
        out = {"rows": int(len(sel)), "races": int(sel["race_uid"].nunique()),
               "measured_figure_pct": round(100 * float(sel["measured_speed_z"].notna().mean()), 2)}
        for col in MEASURED_SPEED_FEATURE_COLS:
            out[f"{col}_fill_pct"] = round(100 * float(sel[col].notna().mean()), 2)
        return out

    by_type = {}
    if "race_type" in win.columns:
        for key, grp in win.groupby(win["race_type"].fillna("(na)")):
            by_type[str(key)] = round(100 * float(grp["msf_mean3"].notna().mean()), 2)

    by_month = {}
    for month, grp in win.groupby(win["day"].str.slice(0, 7)):
        by_month[month] = {
            "rows": int(len(grp)),
            "figure_pct": round(100 * float(grp["measured_speed_z"].notna().mean()), 2),
            "msf_mean3_pct": round(100 * float(grp["msf_mean3"].notna().mean()), 2),
        }

    par_levels = (win["par_level"].value_counts(dropna=False).sort_index()
                  .rename(lambda k: str(k)).to_dict())
    return {
        "win_rows": int(len(win)),
        "win_races": int(win["race_uid"].nunique()),
        "measured_figure_pct_overall": round(
            100 * float(win["measured_speed_z"].notna().mean()), 2),
        "feature_fill_pct": {c: round(100 * float(win[c].notna().mean()), 2)
                             for c in MEASURED_SPEED_FEATURE_COLS},
        "par_level_counts": {k: int(v) for k, v in par_levels.items()},
        "by_window": {"dev_core": window("dev_core", *DEV_CORE),
                      "dev_oos": window("dev_oos", *DEV_OOS),
                      "final_holdout_RESERVED": window("holdout", *FINAL_HOLDOUT)},
        "msf_mean3_fill_by_race_type": by_type,
        "by_month": by_month,
    }


def main() -> int:
    matrix = pd.read_parquet(MATRIX_IN)
    races = pd.read_parquet(RACES_IN)
    runners = pd.read_parquet(RUNNERS_IN)
    print(f"matrix {matrix.shape}; usable races {int(races['usable'].sum())}; "
          f"runner timings {len(runners)}", flush=True)

    built = build(matrix, races, runners)
    # Attach race_type for the coverage split (history only, never a feature).
    built = built.merge(races[["race_key", "race_type"]].drop_duplicates("race_key"),
                        on="race_key", how="left")
    built.to_parquet(MATRIX_OUT, index=False)
    print(f"wrote {MATRIX_OUT} {built.shape}", flush=True)

    report = coverage(built)
    print("running invariance checks on real data...", flush=True)
    report["invariance"] = invariance_checks(matrix, races, runners)

    # The race's own figure must never be usable as a feature of that race.
    report["own_race_figure_excluded_from_features"] = bool(
        "measured_speed_z" not in MEASURED_SPEED_FEATURE_COLS
        and "measured_speed_pct" not in MEASURED_SPEED_FEATURE_COLS)

    with open(REPORT_OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "by_month"},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
