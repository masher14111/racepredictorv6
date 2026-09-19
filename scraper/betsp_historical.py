"""betSP historical fetcher — PUBLIC orchestrator.

Betfair SP daily price files form the spine; results sites enrich
jockey/trainer/position/going; output is a year-partitioned parquet.
"""
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Callable, Optional

import pandas as pd

from scraper.betsp import betfair_sp, joiner, raw_store, writer
from scraper.betsp.extractor import StubExtractor
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


def _load_cfg() -> dict:
    return get_config().get("betsp_historical", {})


def _iter_days(years: list) -> list:
    days = []
    for y in years:
        d = date(y, 1, 1)
        end = date(y, 12, 31)
        while d <= end:
            days.append(d)
            d += timedelta(days=1)
    return days


def _resolve_years(cfg: dict, years: Optional[list]) -> list:
    if years:
        return years
    span = int(cfg.get("rolling_years", 3))
    this_year = _now().year
    return list(range(this_year - span + 1, this_year + 1))


def _results_get_html(url: str) -> str:
    """Default live HTML fetcher for results pages.

    Results sites (Racing Post, Sporting Life, At The Races) sit behind Cloudflare,
    which TLS-fingerprints plain httpx and returns 406/403 block pages regardless of
    headers. curl_cffi impersonates a real Chrome TLS handshake and gets through.

    Throttled by the central rate limiter (per-domain rps from config.yaml) and
    routed through the proxy manager. Raises on failure; callers' fetch_raw()
    catch it per-URL.
    """
    from curl_cffi import requests as creq
    from utils.proxy_manager import get_proxy_manager
    from utils.rate_limiter import get_rate_limiter

    rotator = get_proxy_manager()
    proxy = rotator.next()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    with get_rate_limiter().acquire_for_url(url):
        try:
            resp = creq.get(url, impersonate="chrome", proxies=proxies,
                            timeout=30, allow_redirects=True)
        except Exception:
            # Network-level failure (connection refused, timeout, SSL etc.) —
            # penalise the proxy so the rotator can route around it.
            rotator.report_failure(proxy)
            raise
    if resp.status_code >= 400:
        # HTTP 4xx/5xx: the proxy connected fine — don't penalise it.
        rotator.report_success(proxy)
        resp.raise_for_status()
    rotator.report_success(proxy)
    return resp.text


# Betfair's daily SP file is published one calendar day AFTER the local race
# date it actually contains: the file at .../dwbfpricesukwin18092026.csv has
# every `event_dt` dated 17-09-2026 (confirmed against the live file, stage 06,
# 2026-09-19; reproduced on 4 consecutive days). Add a day when building the
# request so `_fetch_backbone(cfg, days)` returns rows whose ACTUAL race_date
# matches the caller's `days`, matching every caller's assumption (docstrings,
# `fetch()`'s year loop, `scripts/fetch_results_window`'s explicit date
# window). Uncorrected, a short/bounded window request returns backbone rows
# for `days` shifted back by one, which no longer lines up with
# `_fetch_enrichment`'s results-site rows (fetched for the literal `days`) —
# the join key requires the same calendar date on both sides, so a single-day
# backfill request silently joined 0% of positions.
_BACKBONE_FILE_DAY_OFFSET = timedelta(days=1)


def _fetch_one(args):
    d, region, market = args
    file_day = d + _BACKBONE_FILE_DAY_OFFSET
    url = betfair_sp.file_url(region, market, file_day.year, file_day.month, file_day.day)
    text = betfair_sp.fetch_csv(url)
    if not text:
        return None
    try:
        return betfair_sp.parse_csv(text, region, market)
    except Exception as exc:  # noqa: BLE001
        logger.warning("backbone parse failed %s: %s", url, exc)
        return None


def _fetch_backbone(cfg: dict, days: list) -> pd.DataFrame:
    regions = cfg.get("regions", ["uk", "ire"])
    markets = cfg.get("markets", ["win", "place"])
    workers = cfg.get("backbone_workers", 10)
    tasks = [(d, r, m) for d in days for r in regions for m in markets]
    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_one, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                logger.info("betsp backbone: %d/%d files fetched", done, len(tasks))
            result = fut.result()
            if result is not None:
                frames.append(result)
    if not frames:
        return pd.DataFrame(columns=betfair_sp.BACKBONE_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _enrich_one_day(args):
    src, d, get_html, raw_root, extractor = args
    rows = []
    for raw in src.fetch_raw(d, get_html):
        try:
            raw_store.write(raw, root=raw_root)
        except Exception as exc:  # noqa: BLE001
            logger.warning("raw_store failed: %s", exc)
        rows.extend(extractor.extract(raw, src))
    return rows


def _fetch_enrichment(cfg: dict, days: list, raw_root: str,
                      get_html: Callable[[str], str]) -> list:
    source_names = cfg.get("results_sources", list(_SOURCES))
    workers = cfg.get("enrichment_workers", 8)
    extractor = StubExtractor()

    # Build one flat task list across ALL sources so the three domains scrape
    # concurrently instead of sequentially. Each domain still honours its own
    # rps/max_concurrent cap inside the rate limiter, so they never contend with
    # each other — total wall-clock ≈ slowest single source, not their sum.
    tasks = []
    for name in source_names:
        src_cls = _SOURCES.get(name)
        if not src_cls:
            logger.warning("unknown results source: %s", name)
            continue
        src = src_cls()
        tasks.extend((src, d, get_html, raw_root, extractor) for d in days)

    if not tasks:
        return []

    # One thread per (source × worker) lane; the per-domain limiter parks any
    # excess so no site is hit harder than its config allows.
    pool_size = workers * max(1, len(source_names))
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        futures = {pool.submit(_enrich_one_day, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            if done % 200 == 0:
                logger.info("betsp enrichment: %d/%d source-days done", done, len(tasks))
            result = fut.result()
            if result:
                rows.extend(result)
    return rows


def fetch(years: Optional[list] = None, force: bool = False,
          parquet_path: Optional[str] = None,
          raw_root: Optional[str] = None) -> pd.DataFrame:
    """Build/refresh the betSP historical dataset; returns the joined DataFrame."""
    cfg = _load_cfg()
    parquet_path = parquet_path or cfg.get("parquet_path",
                                           "data/historical/betsp.parquet")
    raw_root = raw_root or cfg.get("raw_path", "data/historical/raw")
    priority = cfg.get("source_priority",
                       ["racing_post", "sporting_life", "at_the_races"])

    years = _resolve_years(cfg, years)
    days = _iter_days(years)
    max_days = int(cfg.get("max_days_per_run", 0) or 0)
    if max_days:
        days = days[:max_days]

    backbone_checkpoint = os.path.join(os.path.dirname(parquet_path), "betsp_backbone.parquet")

    logger.info("betsp_historical: %d day(s) across years %s", len(days), years)
    if not force and os.path.exists(backbone_checkpoint):
        logger.info("betsp_historical: loading backbone from checkpoint %s", backbone_checkpoint)
        backbone = pd.read_parquet(backbone_checkpoint)
    else:
        backbone = _fetch_backbone(cfg, days)
        if backbone.empty:
            logger.warning("betsp_historical: no Betfair SP data fetched")
            return backbone
        backbone.to_parquet(backbone_checkpoint, index=False)
        logger.info("betsp_historical: backbone checkpoint saved (%d rows)", len(backbone))

    result_rows = _fetch_enrichment(cfg, days, raw_root, _results_get_html)
    joined = joiner.join(backbone, result_rows, priority=priority)
    joined["fetched_at"] = pd.Timestamp(_now()).tz_convert("UTC")
    writer.write(joined, path=parquet_path)
    return joined
