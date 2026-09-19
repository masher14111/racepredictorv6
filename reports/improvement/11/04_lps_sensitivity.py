"""Step 11 / 4: is the family driven by the one declared constant?

Only the WINNER's time is published. A beaten runner's time is reconstructed as
``winning_time + beaten_lengths / lengths_per_second``, and
``lengths_per_second`` is declared (:data:`features.measured_timing.DEFAULT_LENGTHS_PER_SECOND`),
not measured -- there is no within-race time spread to fit it against.

So the honest question is whether the feature family's *values* depend on that
choice. This rebuilds the figures at 4.0 / 5.0 / 6.0 lengths per second and
reports the rank correlation of the resulting pre-race features against the
default. A family that reorders under a plausible alternative constant would not
be safe to retain on the default's evidence alone.

Read-only apart from reports/improvement/11/lps_sensitivity.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from features import measured_speed as ms, measured_timing as mt  # noqa: E402
from models.features import MEASURED_SPEED_FEATURE_COLS  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module  # noqa: E402

build_mod = import_module("02_build_experiment_matrix") if False else None

RACES_IN = "data/audit/11/measured_timing_races.parquet"
RUNNERS_IN = "data/audit/11/measured_timing_runners.parquet"
MATRIX_IN = "data/audit/10/training_rebuilt.parquet"
OUT = "reports/improvement/11/lps_sensitivity.json"

LPS_VALUES = (4.0, 5.0, 6.0)


def rebuild(matrix, races, runners, lps):
    """Recompute est_* at a different lengths-per-second, then rebuild the family."""
    runners = runners.copy()
    beaten = pd.to_numeric(runners["beaten_lengths"], errors="coerce")
    base = pd.to_numeric(runners["winning_time_seconds"], errors="coerce")
    est_time = base + beaten / lps
    runners["est_time_seconds"] = est_time
    runners["est_speed_yps"] = pd.to_numeric(runners["distance_yards"],
                                             errors="coerce") / est_time
    runners["lengths_per_second"] = lps

    figures = ms.add_speed_figures(runners, races.loc[races["usable"]])
    keep = ["race_key", "horse_key", "measured_speed_z"]
    figures = figures[keep].drop_duplicates(subset=["race_key", "horse_key"])

    from utils.text_norm import minute_key, norm_horse, norm_venue
    out = matrix.copy()
    out["race_key"] = (out["race_uid"].str.split("|").str[0].map(norm_venue) + "|"
                       + out["race_uid"].str.split("|").str[1].map(minute_key))
    out["horse_key"] = out["horse_name"].map(norm_horse)
    out = out.merge(figures, on=["race_key", "horse_key"], how="left")
    return ms.add_measured_pre_race_features(out)


def main() -> int:
    matrix = pd.read_parquet(MATRIX_IN)
    matrix = matrix[matrix["market_type"].astype(str).str.upper() == "WIN"] \
        .reset_index(drop=True)
    races = pd.read_parquet(RACES_IN)
    runners = pd.read_parquet(RUNNERS_IN)

    built = {}
    for lps in LPS_VALUES:
        print(f"rebuilding at lengths_per_second={lps}", flush=True)
        built[lps] = rebuild(matrix, races, runners, lps)

    default = mt.DEFAULT_LENGTHS_PER_SECOND
    report = {"default_lengths_per_second": default, "values_tested": list(LPS_VALUES),
              "spearman_vs_default": {}, "pearson_vs_default": {}, "fill_pct": {}}
    ref = built[default]
    for lps in LPS_VALUES:
        frame = built[lps]
        sp, pe, fill = {}, {}, {}
        for col in MEASURED_SPEED_FEATURE_COLS:
            a, b = ref[col].astype(float), frame[col].astype(float)
            both = a.notna() & b.notna()
            sp[col] = round(float(a[both].corr(b[both], method="spearman")), 6) \
                if both.sum() > 2 else None
            pe[col] = round(float(a[both].corr(b[both])), 6) if both.sum() > 2 else None
            fill[col] = round(100 * float(frame[col].notna().mean()), 2)
        report["spearman_vs_default"][str(lps)] = sp
        report["pearson_vs_default"][str(lps)] = pe
        report["fill_pct"][str(lps)] = fill

    worst = min(v for table in report["spearman_vs_default"].values()
                for v in table.values() if v is not None)
    report["min_spearman_across_all"] = worst
    report["family_is_robust_to_the_constant"] = bool(worst >= 0.99)

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
