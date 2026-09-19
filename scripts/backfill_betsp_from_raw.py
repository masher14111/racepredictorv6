"""Offline backfill: rebuild data/historical/betsp.parquet from STORED raw results.

Tasks 06/07 added trainer_name/jockey_name to the results parsers + joiner + writer,
but the on-disk betsp.parquet predates that change (it carries only *_id, no names).
Re-scraping 3 years of results over the network is infeasible (Cloudflare), so this
script replays the already-captured raw HTML in data/historical/raw/<source>/<year>/
<date>/*.html.gz through the *current* parser, re-joins onto the Betfair SP backbone
checkpoint, and rewrites betsp.parquet with the name columns populated.

    python -m scripts.backfill_betsp_from_raw            # all stored raw
    python -m scripts.backfill_betsp_from_raw --limit-days 5   # smoke (first N days)

After this, run `python -m scripts.refresh --no-scrape` (re-normalize) so the names
flow into unified_races.parquet, then rebuild the training matrix.
"""
import argparse
import gzip
import os
import sys
from datetime import date

import pandas as pd

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

from scraper.betsp import joiner, writer
from scraper.betsp.extractor import StubExtractor
from scraper.betsp.results.base import RawResult
from scraper.betsp.results.at_the_races import AtTheRaces
from scraper.betsp.results.racing_post import RacingPost
from scraper.betsp.results.sporting_life import SportingLife
from utils.config_loader import get_config
from utils.logger import get_logger
from utils.timezone import now as _now

logger = get_logger(__name__)

_SOURCES = {
    "sporting_life": SportingLife,
    "racing_post": RacingPost,
    "at_the_races": AtTheRaces,
}


def _iter_raw_files(raw_root: str, source: str):
    """Yield (date, path) for every stored .html.gz under a source, oldest first."""
    src_root = os.path.join(raw_root, source)
    if not os.path.isdir(src_root):
        return
    days = []
    for year in sorted(os.listdir(src_root)):
        ydir = os.path.join(src_root, year)
        if not os.path.isdir(ydir):
            continue
        for day in sorted(os.listdir(ydir)):
            ddir = os.path.join(ydir, day)
            if os.path.isdir(ddir):
                days.append((day, ddir))
    for day, ddir in days:
        try:
            d = date.fromisoformat(day)
        except ValueError:
            continue
        for fn in sorted(os.listdir(ddir)):
            if fn.endswith(".html.gz"):
                yield d, os.path.join(ddir, fn)


def _replay_source(source: str, raw_root: str, limit_days: int) -> list:
    """Parse every stored raw doc for one source into ResultRows."""
    src = _SOURCES[source]()
    extractor = StubExtractor()
    rows = []
    seen_days: set = set()
    files = 0
    for d, path in _iter_raw_files(raw_root, source):
        if limit_days and d.isoformat() not in seen_days and len(seen_days) >= limit_days:
            break
        seen_days.add(d.isoformat())
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                html = fh.read()
        except OSError as exc:
            logger.warning("backfill: unreadable %s (%s)", path, exc)
            continue
        raw = RawResult(source=source, race_date=d, url=path, html=html)
        rows.extend(extractor.extract(raw, src))
        files += 1
        if files % 2000 == 0:
            logger.info("backfill %s: %d files, %d rows", source, files, len(rows))
    logger.info("backfill %s: replayed %d files across %d day(s) -> %d rows",
                source, files, len(seen_days), len(rows))
    print(f"  {source:<14} {files:>6} files / {len(seen_days):>4} days -> {len(rows):,} rows")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-days", type=int, default=0,
                    help="replay only the first N stored days per source (smoke test)")
    args = ap.parse_args()

    cfg = get_config().get("betsp_historical", {})
    parquet_path = os.path.join(_BASE, cfg.get("parquet_path",
                                               "data/historical/betsp.parquet"))
    raw_root = os.path.join(_BASE, cfg.get("raw_path", "data/historical/raw"))
    backbone_path = os.path.join(os.path.dirname(parquet_path), "betsp_backbone.parquet")
    priority = cfg.get("source_priority",
                       ["racing_post", "sporting_life", "at_the_races"])

    if not os.path.exists(backbone_path):
        print(f"ERROR: backbone checkpoint missing at {backbone_path}")
        return 1

    print("betsp backfill from stored raw")
    print(f"[1/4] backbone — loading {backbone_path}")
    backbone = pd.read_parquet(backbone_path)
    print(f"  backbone rows: {len(backbone):,}")

    print("[2/4] replay   — parsing stored raw results")
    result_rows = []
    for source in _SOURCES:
        result_rows.extend(_replay_source(source, raw_root, args.limit_days))
    if not result_rows:
        print("ERROR: no result rows parsed from stored raw — nothing to backfill.")
        return 1
    print(f"  total result rows: {len(result_rows):,}")

    print("[3/4] join     — enriching backbone with parsed results")
    joined = joiner.join(backbone, result_rows, priority=priority)
    joined["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
    nn = lambda c: int(joined[c].notna().sum()) if c in joined else 0
    print(f"  joined rows: {len(joined):,} | trainer_name {nn('trainer_name'):,} | "
          f"jockey_name {nn('jockey_name'):,} | position {nn('position'):,}")

    print(f"[4/4] write    — replacing {parquet_path}")
    import shutil
    if os.path.isdir(parquet_path):
        shutil.rmtree(parquet_path, ignore_errors=True)
    writer.write(joined, path=parquet_path)
    print("  done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
