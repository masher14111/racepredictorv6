"""Step 16 / 0: how much of step 10's frozen matrix has ANY archived text?

Read-only. Never writes to data/audit/10 or reports/improvement/10. Answers
the question that decides this stage's status before any model is touched:
does the text_archive corpus have timestamped history reaching into dev_oos
or final_holdout, or only into dev_core?
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pandas as pd  # noqa: E402

from features.text_features_v1 import _archive_key  # noqa: E402
from llm.text_archive import get_text_archive  # noqa: E402

MANIFEST_PATH = "reports/improvement/10/run_manifest.json"
MATRIX_PATH = "data/audit/10/training_rebuilt.parquet"
OUT_PATH = "reports/improvement/16/coverage_report.json"


def main() -> int:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    windows = manifest["windows"]

    archive = get_text_archive()
    archive_coverage = archive.coverage()
    known = archive.known_race_uids()

    df = pd.read_parquet(MATRIX_PATH, columns=["race_uid", "race_date", "market_type"])
    win = df[df["market_type"].astype(str).str.upper() == "WIN"].copy()
    win["_day"] = pd.to_datetime(win["race_date"], utc=True, errors="coerce") \
        .dt.tz_localize(None).dt.normalize()
    win["_archive_key"] = win["race_uid"].astype(str).map(_archive_key)
    win["_has_text"] = win["_archive_key"].isin(known)

    def window_stats(lo, hi_exclusive):
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi_exclusive)
        sel = win[(win["_day"] >= lo_ts) & (win["_day"] < hi_ts)]
        with_text = sel[sel["_has_text"]]
        return {
            "rows": int(len(sel)),
            "races": int(sel["race_uid"].nunique()),
            "rows_with_archived_text": int(len(with_text)),
            "races_with_archived_text": int(with_text["race_uid"].nunique()),
        }

    report = {
        "manifest": MANIFEST_PATH,
        "matrix": MATRIX_PATH,
        "text_archive_coverage": archive_coverage,
        "by_window": {
            "dev_core": window_stats(windows["dev_core"]["start"], windows["dev_core"]["end_exclusive"]),
            "dev_oos": window_stats(windows["dev_oos"]["start"], windows["dev_oos"]["end_exclusive"]),
            "final_holdout_RESERVED": window_stats(
                windows["final_holdout_RESERVED_UNTOUCHED"]["start"],
                pd.Timestamp(windows["final_holdout_RESERVED_UNTOUCHED"]["end"]) + pd.Timedelta(days=1),
            ),
        },
        "conclusion": None,
    }
    dev_oos_races = report["by_window"]["dev_oos"]["races_with_archived_text"]
    holdout_races = report["by_window"]["final_holdout_RESERVED"]["races_with_archived_text"]
    dev_core_races = report["by_window"]["dev_core"]["races_with_archived_text"]
    if dev_oos_races == 0 and holdout_races == 0:
        report["conclusion"] = (
            f"ZERO archived-text races in dev_oos or final_holdout; only "
            f"{dev_core_races} races (a single archived day) inside dev_core. "
            "A pre-registered chronological comparison across development "
            "windows cannot be executed on any genuine out-of-sample fold — "
            "DEFERRED_DATA, not a NO-GO model finding."
        )
    else:
        report["conclusion"] = "Coverage exists outside dev_core — a real ablation may be possible."

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
