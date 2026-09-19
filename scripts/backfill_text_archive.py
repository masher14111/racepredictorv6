"""One-off/idempotent import of ``data/spotlight.parquet`` into ``llm.text_archive``.

Stage 14 introduced the append-only text archive; this backfills the ONE
pre-existing text source (the single-day, 243-row Spotlight sample already on
disk) so real coverage numbers exist rather than starting from zero for no
reason. It is idempotent — re-running just re-inserts the same content-hash
keyed rows, which is a no-op under the archive's own dedupe.

Preserves the source file untouched; this only reads it. ``published_at`` is
left unknown for every row (Sporting Life gives no comment publish time) —
only the real, already-recorded ``fetched_at`` is used.

    python -m scripts.backfill_text_archive [--parquet data/spotlight.parquet]
"""
from __future__ import annotations

import argparse

import pandas as pd

from execution.race_facts import race_facts_key
from llm.text_archive import get_text_archive
from utils.logger import get_logger

logger = get_logger(__name__)


def backfill(parquet_path: str = "data/spotlight.parquet") -> int:
    df = pd.read_parquet(parquet_path)
    archive = get_text_archive()
    n = 0
    for row in df.itertuples(index=False):
        if not str(getattr(row, "venue", "") or "").strip() or not getattr(row, "commentary", None):
            continue
        race_uid = race_facts_key(row.venue, row.race_date)
        if not race_uid:
            continue
        try:
            archive.record(
                text=row.commentary,
                source="sporting_life_spotlight",
                race_uid=race_uid,
                horse_name=row.horse_name,
                horse_id=row.horse_id,
                text_kind="spotlight",
                fetched_at=row.fetched_at,
                published_at=None,
                metadata={"cloth_number": getattr(row, "cloth_number", None),
                          "race_verdict": getattr(row, "race_verdict", None),
                          "backfilled_from": parquet_path},
            )
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("backfill_text_archive: skipped %s/%s: %s", row.venue, row.horse_name, exc)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/spotlight.parquet")
    args = ap.parse_args()
    n = backfill(args.parquet)
    archive = get_text_archive()
    cov = archive.coverage()
    print(f"backfill_text_archive: archived {n} rows from {args.parquet}")
    print(f"text_archive coverage: {cov}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
