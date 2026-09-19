"""Incremental betSP results catch-up for an explicit date window.

The full ``scraper.betsp_historical.fetch()`` iterates whole *years* and reuses a
stale backbone checkpoint, so it cannot cheaply catch the dataset up after a gap
(results lag the calendar — see README "Status & known limits"). This CLI drives
the same building blocks for just the missing days:

    Betfair SP backbone (win/place, uk+ire)  ->  results-site enrichment
    ->  joiner  ->  canonical append-dedupe writer (data/historical/betsp.parquet)

Nothing here can clobber existing history: the writer read-merge-writes with the
same dedupe key the nightly path uses, through the Stage-2 atomic partitioned
parquet store. The stale ``betsp_backbone.parquet`` checkpoint is deliberately
left untouched (it belongs to the full-year fetch path).

Usage:
    python -m scripts.fetch_results_window --start 2026-06-13 --end 2026-07-26
    python -m scripts.fetch_results_window --start ... --end ... --skip-enrichment
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

import pandas as pd

from scraper import betsp_historical as bh
from scraper.betsp import joiner, writer
from utils.logger import get_logger

logger = get_logger(__name__)


def _parse_day(raw: str) -> date:
    return date.fromisoformat(str(raw).strip())


def day_range(start: date, end: date) -> list[date]:
    """Inclusive list of days from start to end; empty when start > end."""
    if start > end:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def fetch_window(start: date, end: date, *, skip_enrichment: bool = False,
                 parquet_path: str | None = None) -> pd.DataFrame:
    """Fetch backbone (+ enrichment) for [start, end] and merge into the store.

    Returns the joined frame for the window (possibly empty). ``parquet_path``
    overrides the canonical store — used by tests; production callers leave it
    None so rows land in ``data/historical/betsp.parquet``.
    """
    cfg = bh._load_cfg()
    days = day_range(start, end)
    if not days:
        logger.warning("fetch_results_window: empty day range %s..%s", start, end)
        return pd.DataFrame()

    logger.info("fetch_results_window: %d day(s) %s..%s", len(days), start, end)
    backbone = bh._fetch_backbone(cfg, days)
    if backbone.empty:
        logger.warning("fetch_results_window: no Betfair SP rows for window")
        return backbone
    logger.info("fetch_results_window: backbone %d rows", len(backbone))

    result_rows: list = []
    if not skip_enrichment:
        raw_root = cfg.get("raw_path", "data/historical/raw")
        result_rows = bh._fetch_enrichment(cfg, days, raw_root, bh._results_get_html)
        logger.info("fetch_results_window: enrichment %d result rows", len(result_rows))

    priority = cfg.get("source_priority",
                       ["racing_post", "sporting_life", "at_the_races"])
    joined = joiner.join(backbone, result_rows, priority=priority)
    joined["fetched_at"] = pd.Timestamp.now(tz="UTC")

    if parquet_path is None:
        writer.write(joined)
    else:
        writer.write(joined, path=parquet_path)

    pos_fill = float(joined["position"].notna().mean()) if len(joined) else 0.0
    logger.info(
        "fetch_results_window: merged %d rows (%s..%s), position fill %.1f%%",
        len(joined), joined["race_date"].min(), joined["race_date"].max(),
        100.0 * pos_fill,
    )
    return joined


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="scripts.fetch_results_window",
        description="Catch the betSP historical store up over an explicit date window")
    p.add_argument("--start", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (inclusive)")
    p.add_argument("--skip-enrichment", action="store_true",
                   help="Betfair SP backbone only (no results-site scrape); "
                        "position stays null for the new rows")
    args = p.parse_args(argv)

    joined = fetch_window(_parse_day(args.start), _parse_day(args.end),
                          skip_enrichment=args.skip_enrichment)
    if joined.empty:
        print("NO ROWS FETCHED — window may predate publication or network failed")
        return 1

    win = joined[joined["market_type"] == "WIN"]
    print(f"window rows: {len(joined)} (WIN {len(win)}), "
          f"days {joined['race_date'].min()} .. {joined['race_date'].max()}")
    print(f"position fill: {100.0 * joined['position'].notna().mean():.1f}%  "
          f"win_lose fill: {100.0 * joined['win_lose'].notna().mean():.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
