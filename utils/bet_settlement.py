"""Auto-settle paper bets from scraped race results.

Results flow in from the betsp historical scrapers (``data/historical/betsp.parquet``):
each finished horse-run carries a finishing ``position`` and the Betfair starting
price ``odds_finish`` — the closing line used for CLV. This module matches pending
paper bets to those results and settles them (won / lost / placed / void),
updating the virtual bankroll via :class:`utils.bet_tracker.BetTracker`.

A bet is left **pending** until its race actually appears in the results feed, so
un-scraped races are never mis-settled. Once the race is present:

  * finishing position 1                     → win
  * within the each-way places (each-way bet) → place
  * any other finishing position             → lose
  * horse ran but has no finishing position,
    or the race ran without our horse         → void (stake refunded)
"""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from utils.bet_tracker import BetTracker
from utils.logger import get_logger
from utils.text_norm import norm_horse, norm_venue

logger = get_logger(__name__)


def _day(value) -> str:
    """ISO day key (YYYY-MM-DD) from a date/timestamp string or value."""
    return str(value or "")[:10]


def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and math.isnan(v):
        return False
    return str(v).strip() != ""


def _as_position(v):
    """Coerce a finishing position to int, or None for non-finishers/blanks."""
    if not _has_value(v):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _outcome_for(bet: dict, position: int | None, ew_places: int) -> str:
    """Map a finishing position to a settlement outcome for this bet."""
    if position is None or position <= 0:
        return "void"
    if position == 1:
        return "win"
    if bet.get("bet_type") == "each_way" and position <= ew_places:
        return "place"
    return "lose"


# Venue aliases for settlement matching only.
#
# The results feed files the June meeting under "Royal Ascot" while the odds
# feeds (and therefore every placed bet) say "Ascot" — the same racecourse under
# the meeting's name. Without this, a Royal Ascot bet can never match its own
# result: it is not mis-settled, it just stays pending forever. Eleven bets from
# 19 Jun 2026 sat open for three months on exactly this.
#
# Deliberately an explicit map, NOT a "strip the Royal prefix" rule: the archive
# also carries "Down Royal", which is a different course in Co. Down and must
# not collapse into anything.
_VENUE_ALIASES = {"royalascot": "ascot"}


def _venue_key(value) -> str:
    """Normalised venue, with meeting-name aliases folded onto the course."""
    key = norm_venue(value)
    return _VENUE_ALIASES.get(key, key)


def _index_results(results: Iterable[dict]):
    """Build lookup indexes from a results feed.

    Returns (by_id, by_name, raced) where ``raced`` is the set of
    (venue_norm, day) races that appeared at all — used to void bets whose horse
    is absent from a race that has clearly run.
    """
    by_id: dict[str, dict] = {}
    by_name: dict[tuple, dict] = {}
    raced: set[tuple] = set()
    for r in results:
        venue_n = _venue_key(r.get("venue"))
        day = _day(r.get("race_date"))
        raced.add((venue_n, day))
        if _has_value(r.get("horse_id")):
            by_id[str(r["horse_id"])] = r
        by_name[(venue_n, norm_horse(r.get("horse_name")), day)] = r
    return by_id, by_name, raced


def settle_from_results(tracker: BetTracker, results: Iterable[dict]) -> list[dict]:
    """Settle every pending bet that the results feed can resolve.

    ``results``: iterable of dicts with at least ``venue``, ``horse_name``,
    ``race_date`` and ``position``; ``horse_id`` and ``odds_finish`` (the closing
    SP, for CLV) are used when present.

    Returns the list of settled bet rows. Bets whose race has not yet appeared in
    the feed are skipped and remain pending.
    """
    results = list(results)
    by_id, by_name, raced = _index_results(results)
    ew_places = tracker.ew_places
    settled: list[dict] = []

    for bet in tracker.pending_bets():
        day = _day(bet.get("race_time"))
        venue_n = _venue_key(bet.get("venue"))

        res = None
        if _has_value(bet.get("horse_id")):
            res = by_id.get(str(bet["horse_id"]))
        if res is None:
            res = by_name.get((venue_n, norm_horse(bet.get("horse_name")), day))

        if res is None:
            # No row for this horse. Only settle (as void) if the race itself has
            # run; otherwise results aren't in yet — leave it pending.
            if (venue_n, day) in raced:
                row = tracker.settle_bet(bet["id"], "void")
                settled.append(row)
                logger.info("settlement: bet #%s voided (horse absent from result)",
                            bet["id"])
            continue

        position = _as_position(res.get("position"))
        outcome = _outcome_for(bet, position, ew_places)
        closing = res.get("odds_finish")
        closing = float(closing) if _has_value(closing) else None
        row = tracker.settle_bet(bet["id"], outcome, closing_odds=closing)
        settled.append(row)
        logger.info("settlement: bet #%s settled %s (pos=%s, SP=%s)",
                    bet["id"], outcome, position, closing)

    return settled


def settle_from_storage(tracker: BetTracker, storage=None) -> list[dict]:
    """Convenience wrapper: load scraped results from the betsp parquet dataset
    and settle pending bets against them. Returns the settled rows (empty if the
    dataset is missing or unreadable)."""
    if storage is None:
        from utils.storage import get_storage
        storage = get_storage()
    try:
        df = storage.read_parquet("betsp")
    except Exception as exc:  # noqa: BLE001 — missing/empty dataset is non-fatal
        logger.warning("settlement: could not read results dataset: %s", exc)
        return []
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        return []
    cols = [c for c in ("venue", "horse_name", "horse_id", "race_date",
                        "position", "odds_finish") if c in df.columns]
    return settle_from_results(tracker, df[cols].to_dict("records"))
