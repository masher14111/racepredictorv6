"""Persist captured raw results documents for later (LLM) re-extraction.

Stored gzip-compressed (.html.gz): results pages are large (~300 KB each) and a
full historical backfill captures tens of thousands of them, so raw storage would
otherwise reach ~9 GB. gzip cuts that ~10x. Read back with gzip.open(path, "rt").
"""
import gzip
import hashlib
import os

from scraper.betsp.results.base import RawResult
from utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "historical", "raw")
)


def write(raw: RawResult, root: str = _DEFAULT_ROOT) -> str:
    """Store raw.html gzipped at {root}/{source}/{year}/{date}/{hash}.html.gz. Returns path."""
    year = f"{raw.race_date.year:04d}"
    day = raw.race_date.isoformat()
    digest = hashlib.sha1(raw.url.encode("utf-8")).hexdigest()[:12]
    folder = os.path.join(root, raw.source, year, day)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{digest}.html.gz")
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(raw.html or "")
    logger.debug("raw_store wrote %s", path)
    return path
