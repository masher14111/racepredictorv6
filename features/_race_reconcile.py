"""Reconcile source off-time disagreements for one physical race (Stage 20 / B5).

``features.fuse.fuse_sources`` groups a runner-race-market row by the EXACT
``race_time`` value each source reported. Two odds sources scraping the same
physical race sometimes disagree on the off-time by a minute (reproduced live:
Dundalk 19:30 vs 19:31) -- left unreconciled, that fragments one race into two
downstream rows, which ``features.derive.add_race_key`` then turns into two
DIFFERENT ``race_uid`` values, and the live loop issues two ticket sets for the
same runners (GAP-C in ``reports/improvement/17/06_paper_replay.py``).

This module is deliberately NOT wired into ``features.builder.build_training_matrix``
-- the frozen dev/final-holdout matrices (D42) must never be silently rebuilt
with different race membership. It is wired only into
``build_inference_matrix``'s LIVE rows (Stage 20), so it only affects today's
predictions/tickets, never a historical evaluation window.

Resolution never blindly rounds two nearby off-times together: a candidate
merge additionally requires genuine runner-name overlap between the two
slots. A close-but-unconvincing pair (no shared runner name, or no runner
evidence on one side at all) is QUARANTINED -- excluded from the live matrix
with a recorded reason -- rather than guessed either way, per the Stage 20
work order ("ambiguous mappings must be quarantined with a reason").
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from features.fuse import _SOURCE_PRIORITY
from utils.logger import get_logger
from utils.text_norm import norm_horse, norm_venue

logger = get_logger(__name__)

# Measured on data/unified_races.parquet (Stage 20): the minimum real gap
# between two DISTINCT races at the same venue on the same day is 600s (10
# minutes). Kept well below that floor so a genuine adjacent race can never be
# merged, however runner-overlap evidence turns out.
TOLERANCE_SECONDS = 180.0

QUARANTINE_AMBIGUOUS_OFF_TIME = "ambiguous_close_off_times"


def reconcile_live_race_times(
    df: pd.DataFrame, *, tolerance_seconds: float = TOLERANCE_SECONDS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snap near-duplicate off-times at the SAME venue to one canonical time.

    Returns ``(resolved, quarantined)``. ``resolved`` is ``df`` with
    ``race_time`` reassigned to the canonical value for any row whose off-time
    is recognisably the same physical race as a nearby one at the same venue
    (a tight time tolerance AND genuine runner-name overlap, both required).
    ``race_date``/every other column is left untouched -- only the identity
    component that ``features.derive.add_race_key`` reads is corrected.
    Rows in an ambiguous close-time cluster are excluded from ``resolved`` and
    returned in ``quarantined`` with ``_quarantine_reason`` set, never merged
    or split by a guess. A different venue, or two off-times genuinely more
    than ``tolerance_seconds`` apart, is never touched.
    """
    empty_quarantine = (df.iloc[0:0].copy() if df is not None else pd.DataFrame())
    if df is None or df.empty or "venue" not in df.columns or "race_time" not in df.columns:
        return df, empty_quarantine

    out = df.copy()
    venue_n = out["venue"].astype(str).map(norm_venue)
    race_time = pd.to_datetime(out["race_time"], utc=True, errors="coerce")
    horse = (
        out["horse_name"].astype(str).map(norm_horse)
        if "horse_name" in out.columns
        else pd.Series("", index=out.index)
    )
    source = out["source"].astype(str) if "source" in out.columns else pd.Series("", index=out.index)

    quarantine_mask = pd.Series(False, index=out.index)
    canonical_time = race_time.copy()

    for v in venue_n.dropna().unique():
        if not v:
            continue
        v_mask = (venue_n == v).to_numpy()
        distinct = sorted(t for t in race_time[v_mask].dropna().unique())
        if len(distinct) < 2:
            continue

        slots: list[dict[str, Any]] = []
        for t in distinct:
            slot_mask = v_mask & (race_time == t).to_numpy()
            runners = set(horse[slot_mask]) - {""}
            priority = max((_SOURCE_PRIORITY.get(s, 0) for s in source[slot_mask]), default=0)
            slots.append({"time": t, "mask": slot_mask, "runners": runners, "priority": priority})

        clusters: list[list[dict[str, Any]]] = []
        for slot in slots:
            if clusters and (slot["time"] - clusters[-1][-1]["time"]).total_seconds() <= tolerance_seconds:
                clusters[-1].append(slot)
            else:
                clusters.append([slot])

        for cluster in clusters:
            if len(cluster) == 1:
                continue
            overlap_ok = True
            for i in range(len(cluster) - 1):
                a, b = cluster[i]["runners"], cluster[i + 1]["runners"]
                if not a or not b or not (a & b):
                    overlap_ok = False
                    break
            if not overlap_ok:
                for c in cluster:
                    quarantine_mask |= c["mask"]
                logger.warning(
                    "features._race_reconcile: quarantined %d close off-time row(s) "
                    "for venue %r (%s) -- no confirmed runner overlap",
                    sum(int(c["mask"].sum()) for c in cluster),
                    v,
                    [str(c["time"]) for c in cluster],
                )
                continue
            best = max(cluster, key=lambda c: (c["priority"], -c["time"].value))
            for c in cluster:
                canonical_time.loc[c["mask"]] = best["time"]

    resolved_mask = ~quarantine_mask
    resolved = out[resolved_mask].copy()
    resolved["race_time"] = canonical_time[resolved_mask].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    quarantined = out[quarantine_mask].copy()
    if not quarantined.empty:
        quarantined["_quarantine_reason"] = QUARANTINE_AMBIGUOUS_OFF_TIME

    return resolved.reset_index(drop=True), quarantined.reset_index(drop=True)
