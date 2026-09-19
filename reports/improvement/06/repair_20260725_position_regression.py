"""Stage 06: restore `position` for 2026-07-25 rows in the raw betsp store.

Found while validating stage 06's candidate rebuild against the frozen baseline
(memory/improvement/stages/06.md): every 2026-07-25 row in
``data/historical/betsp.parquet`` (1,010 rows, both WIN and PLACE markets) carries
``position = NaN`` with a single ``fetched_at`` of 2026-09-18T22:29:20Z — one
blanket overwrite, predating this session, most likely an uncorrected
``scripts.fetch_results_window`` re-run (before D31/D32 landed) whose blocked/
misaligned enrichment wiped an already-resolved day (exactly the D32 failure mode,
just not caught for this specific day at the time).

The frozen baseline (``data/features/training.parquet``, D9-protected, untouched
here) still carries the correct WIN-market positions for that day (its own
build predates the corruption) — 501 rows, keyed by
(venue, race_time-minute, src_horse_id, market_type). This script restores
ONLY ``position`` for those exact matched WIN rows in the raw store, via the
same completeness-aware write path as D32 (so a future incomplete re-fetch
cannot regress it again).

Deliberately market_type-EXACT, not cross-market: raw betsp WIN/PLACE rows for
the same horse+race do carry an identical position when BOTH happen to be
resolved (verified directly on 2026-07-20's raw rows: 195/245 matched pairs,
100% equal) — but the frozen baseline itself never has a race_uid+horse_id
resolved on both markets simultaneously (0 overlap across the whole file,
148,860 WIN vs 111,630 PLACE keys) for reasons this stage did not chase down.
Copying a WIN position onto a same-race PLACE row on 2026-07-25 would be an
inference consistent with the raw data but NOT verified by the one ground
truth this stage actually has — so PLACE rows for this day stay null,
genuinely unresolved rather than fabricated.

Read-only against data/features/training.parquet; writes only
data/historical/betsp.parquet (via the existing dedupe-safe writer path).

Run: .venv/Scripts/python.exe reports/improvement/06/repair_20260725_position_regression.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import pandas as pd

from scraper.betsp.writer import _DEDUPE_KEY, FINAL_COLUMNS, _completeness_order, _DEFAULT_PATH
from utils.storage import parquet_store

BAD_FETCHED_AT = pd.Timestamp("2026-09-18 22:29:20.845386", tz="UTC")
TARGET_DATE = pd.Timestamp("2026-07-25").date()

bet = pd.read_parquet(_DEFAULT_PATH)
day_mask = (bet["race_date"].dt.tz_convert("UTC").dt.date == TARGET_DATE)
bad_mask = day_mask & bet["position"].isna() & (bet["fetched_at"] == BAD_FETCHED_AT)
print(f"rows on {TARGET_DATE} with the bad fetched_at and null position: {int(bad_mask.sum())}")

frozen = pd.read_parquet("data/features/training.parquet")
truth = frozen[(frozen["race_date"] == pd.Timestamp("2026-07-25", tz="UTC"))
               & (frozen["market_type"] == "WIN") & (frozen["source"] == "betsp")].copy()
print(f"frozen-baseline WIN ground-truth rows for that day: {len(truth)}")

def _minute_key(ts) -> str:
    t = pd.Timestamp(ts)
    return t.strftime("%H:%M")

truth["_venue"] = truth["venue"].astype(str).str.strip().str.lower()
truth["_minute"] = truth["race_time"].map(_minute_key)
truth["_src_id"] = truth["src_horse_id"].astype(str)
truth["_market"] = truth["market_type"].astype(str)
truth_lookup = truth.set_index(["_venue", "_minute", "_src_id", "_market"])["position"]
assert truth_lookup.index.is_unique, "ground-truth key must be unique per venue/time/horse/market"

bet["_venue"] = bet["venue"].astype(str).str.strip().str.lower()
bet["_minute"] = bet["race_date"].map(_minute_key)
bet["_src_id"] = bet["horse_id"].astype(str)
bet["_market"] = bet["market_type"].astype(str)
bet_key = pd.MultiIndex.from_arrays([bet["_venue"], bet["_minute"], bet["_src_id"], bet["_market"]])

restored = truth_lookup.reindex(bet_key)
restored.index = bet.index
apply_mask = bad_mask & restored.notna().reindex(bet.index).fillna(False)
print(f"rows matched to frozen-baseline ground truth and restorable: {int(apply_mask.sum())}")

before_fill = int((day_mask & bet["position"].notna()).sum())
bet.loc[apply_mask, "position"] = restored.loc[apply_mask].astype("float64")
after_fill = int((day_mask & bet["position"].notna()).sum())
print(f"position fill on {TARGET_DATE}: {before_fill} -> {after_fill} (of {int(day_mask.sum())} rows)")

still_null_win = bet[day_mask & (bet["market_type"] == "WIN") & bet["position"].isna()]
print(f"WIN rows on {TARGET_DATE} still null after restore (no frozen match found): {len(still_null_win)}")

bet = bet.drop(columns=["_venue", "_minute", "_src_id", "_market"])
ordered = _completeness_order(bet)
parquet_store.write_parquet(ordered, _DEFAULT_PATH, partition_by="year",
                            dedupe_key=_DEDUPE_KEY, columns=FINAL_COLUMNS)
print("REPAIR_20260725_COMPLETE")
