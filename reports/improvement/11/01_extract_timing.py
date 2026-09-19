"""Step 11 / 1: extract measured timing from the whole raw archive.

Read-only over ``data/historical/raw/sporting_life``; writes only into this
stage's own experiment directories:
  * ``data/audit/11/measured_timing_races.parquet``  (one row per race)
  * ``data/audit/11/measured_timing_runners.parquet`` (one row per runner)
  * ``reports/improvement/11/coverage_timing.json``   (the coverage report)

Coverage is reported by date, source and race type, with the unit checks and
the timestamp-eligibility check the acceptance criteria require.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from features import measured_timing as mt  # noqa: E402

OUT_DATA = os.path.join("data", "audit", "11")
OUT_REPORT = os.path.join("reports", "improvement", "11")
RAW_ROOT = mt.DEFAULT_RAW_ROOT
SOURCE = mt.DEFAULT_SOURCE


def all_paths() -> list[str]:
    root = os.path.join(RAW_ROOT, SOURCE)
    paths = []
    for year in sorted(os.listdir(root)):
        ydir = os.path.join(root, year)
        if not os.path.isdir(ydir):
            continue
        for day in sorted(os.listdir(ydir)):
            ddir = os.path.join(ydir, day)
            paths += [os.path.join(ddir, n) for n in sorted(os.listdir(ddir))
                      if n.endswith(".html") or n.endswith(".html.gz")]
    return paths


def main() -> int:
    os.makedirs(OUT_DATA, exist_ok=True)
    os.makedirs(OUT_REPORT, exist_ok=True)

    paths = all_paths()
    print(f"archive documents: {len(paths)}", flush=True)

    best: dict[str, mt.RaceTiming] = {}
    n_docs = n_races = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for timing in pool.map(mt.parse_timing_file, paths):
            n_docs += 1
            if n_docs % 5000 == 0:
                print(f"  parsed {n_docs}/{len(paths)}", flush=True)
            if timing is None:
                continue
            n_races += 1
            prior = best.get(timing.race_key)
            if prior is None or mt._completeness(timing) > mt._completeness(prior):
                best[timing.race_key] = timing

    timings = list(best.values())
    usable = [t for t in timings if t.usable]
    print(f"race documents {n_races}; distinct races {len(timings)}; "
          f"usable (time+distance+sane speed) {len(usable)}", flush=True)

    races = pd.DataFrame([{
        "race_key": t.race_key, "race_date": t.race_date, "off_time": t.off_time,
        "available_from": t.off_time, "venue": t.venue, "surface": t.surface,
        "going": t.going, "race_class": t.race_class, "handicap": t.handicap,
        "race_type": t.race_type, "race_name": t.race_name,
        "distance_yards": t.distance_yards,
        "winning_time_seconds": t.winning_time_seconds,
        "usable": t.usable, "n_runner_rows": len(t.runners),
        "n_finished": sum(1 for r in t.runners if r["finished"]),
    } for t in timings])
    runners = mt.to_runner_frame(usable)

    races.to_parquet(os.path.join(OUT_DATA, "measured_timing_races.parquet"), index=False)
    runners.to_parquet(os.path.join(OUT_DATA, "measured_timing_runners.parquet"), index=False)

    report = build_coverage(races, runners, n_docs=len(paths), n_race_docs=n_races)
    with open(os.path.join(OUT_REPORT, "coverage_timing.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("by_month", "by_venue_top", "by_going_top")},
                     indent=2, default=str))
    return 0


def build_coverage(races: pd.DataFrame, runners: pd.DataFrame, *,
                   n_docs: int, n_race_docs: int) -> dict:
    races = races.copy()
    races["month"] = races["race_date"].str.slice(0, 7)
    usable = races.loc[races["usable"]]

    # ---- unit checks on the published figures -------------------------------
    speed = runners["winner_speed_yps"].dropna() if not runners.empty else pd.Series(dtype=float)
    dist = usable["distance_yards"]
    secs = usable["winning_time_seconds"]
    beaten = runners["beaten_lengths"].dropna() if not runners.empty else pd.Series(dtype=float)
    unit_checks = {
        "winner_speed_yps_min": float(speed.min()) if len(speed) else None,
        "winner_speed_yps_median": float(speed.median()) if len(speed) else None,
        "winner_speed_yps_max": float(speed.max()) if len(speed) else None,
        "winner_speed_within_envelope": bool(
            len(speed) and speed.between(mt.MIN_SPEED_YPS, mt.MAX_SPEED_YPS).all()),
        "distance_yards_min": float(dist.min()) if len(dist) else None,
        "distance_yards_max": float(dist.max()) if len(dist) else None,
        "distance_ge_5f_le_4p5m": bool(len(dist) and dist.between(1000, 8200).all()),
        "winning_time_seconds_min": float(secs.min()) if len(secs) else None,
        "winning_time_seconds_max": float(secs.max()) if len(secs) else None,
        "beaten_lengths_nonnegative": bool(len(beaten) and (beaten >= 0).all()),
        "beaten_lengths_max": float(beaten.max()) if len(beaten) else None,
        "winner_beaten_lengths_all_zero": bool(
            not runners.empty
            and (runners.loc[runners["finish_position"] == 1, "beaten_lengths"]
                 .fillna(-1) == 0).all()),
    }

    # ---- timestamp eligibility ---------------------------------------------
    today = pd.Timestamp.utcnow().tz_localize(None).strftime("%Y-%m-%d")
    future = races.loc[races["race_date"] > today]
    eligibility = {
        "today": today,
        "future_dated_race_documents": int(len(future)),
        "future_dated_usable": int(future["usable"].sum()) if len(future) else 0,
        "future_dated_leaks_no_timing": bool(len(future) == 0 or future["usable"].sum() == 0),
        "available_from_equals_off_time": bool(
            runners.empty or (runners["available_from"] == runners["off_time"]).all()),
        "min_race_date": races["race_date"].min(),
        "max_usable_race_date": usable["race_date"].max() if len(usable) else None,
    }

    # ---- non-finisher handling ---------------------------------------------
    nf = runners.loc[~runners["finished"]] if not runners.empty else runners
    non_finishers = {
        "runner_rows": int(len(runners)),
        "finished_rows": int(runners["finished"].sum()) if not runners.empty else 0,
        "non_finisher_rows": int(len(nf)),
        "non_finisher_share": round(float(len(nf) / len(runners)), 6) if len(runners) else None,
        "non_finishers_all_null_time": bool(nf.empty or nf["est_time_seconds"].isna().all()),
        "finished_rows_missing_beaten_lengths": int(
            runners.loc[runners["finished"], "beaten_lengths"].isna().sum())
        if not runners.empty else 0,
        "casualty_reasons": dict(Counter(nf["casualty_reason"].dropna()).most_common(10))
        if not nf.empty else {},
    }

    by_month = {}
    for month, grp in races.groupby("month"):
        by_month[month] = {"races": int(len(grp)), "usable": int(grp["usable"].sum()),
                           "usable_pct": round(100 * float(grp["usable"].mean()), 2)}

    def share(col):
        if not len(races):
            return {}
        out = {}
        for key, grp in races.groupby(races[col].fillna("").replace("", "(blank)")):
            out[str(key)] = {"races": int(len(grp)), "usable": int(grp["usable"].sum()),
                             "usable_pct": round(100 * float(grp["usable"].mean()), 2)}
        return out

    return {
        "documents_scanned": n_docs,
        "race_documents_parsed": n_race_docs,
        "distinct_races": int(len(races)),
        "usable_races": int(len(usable)),
        "usable_race_pct": round(100 * float(races["usable"].mean()), 2) if len(races) else None,
        "date_range": [races["race_date"].min(), races["race_date"].max()],
        "sources": {SOURCE: int(len(races))},
        "unit_checks": unit_checks,
        "timestamp_eligibility": eligibility,
        "non_finisher_handling": non_finishers,
        "by_surface": share("surface"),
        "by_race_type": share("race_type"),
        "by_race_class": share("race_class"),
        "by_handicap": share("handicap"),
        "by_month": by_month,
        "by_going_top": dict(sorted(share("going").items(),
                                    key=lambda kv: -kv[1]["races"])[:15]),
        "by_venue_top": dict(sorted(share("venue").items(),
                                    key=lambda kv: -kv[1]["races"])[:25]),
    }


if __name__ == "__main__":
    raise SystemExit(main())
