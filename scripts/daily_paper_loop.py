"""Stage 6: one idempotent command for the daily paper-betting loop.

    python -m scripts.daily_paper_loop            # full run
    python -m scripts.daily_paper_loop --no-scrape # reuse today's already-captured board

Each run: capture today's board into the immutable snapshot store, re-verify the
forward-validation window (resetting it if a frozen input changed), evaluate
today's card through the PASS-by-default gate and issue paper tickets for every
candidate, settle yesterday's (and any other still-open) tickets against results
now available, record today's gap-ledger entry, recompute the forward-gate
verdict, and write the dated reports.

Re-running this on the same day must be a no-op on top of what already ran:
ticket issuance and settlement are both keyed so a repeat raises a caught,
non-fatal "already exists" error rather than duplicating anything, and the
gap-ledger entry is sticky once a day is marked captured. Capture failure is
caught and recorded, never allowed to abort the rest of the loop -- and nothing
in this module ever writes a snapshot or a settlement price itself; both come
only from execution.snapshots' own real-time paths, so no step here can ever
backdate a fetched_at or invent a closing price.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta as _timedelta, timezone
from typing import Any, Optional

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import pandas as pd

from execution import gap_ledger, window
from execution.config import ExecutionConfig, load_execution_config
from execution.forward_gate import evaluate_forward_gate
from execution.gates import _parse_ts, _s
from execution.model_gate import load_model_verdict
from execution.report import forward_validation_report, write_json, write_report
from execution.snapshots import SnapshotStore
from execution.snapshots import horse_key as _mk_horse_key
from execution.tickets import RealMoneyTicketRefused, Ticket, TicketStore, build_ticket
from utils.logger import get_logger
from utils.text_norm import norm_horse, norm_venue

logger = get_logger(__name__)

DEFAULT_PREDICTIONS_PATH = os.path.join("data", "predictions.json")


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _day_start(stamp: datetime) -> datetime:
    """Midnight UTC of ``stamp``'s date -- the stable per-day dedupe anchor.

    ``build_ticket``'s ``ticket_id`` is a hash of ``(race_uid, horse_key,
    bet_type, dedupe_key)``. This value is that ``dedupe_key`` (via
    ``.date().isoformat()``), so a same-day re-run mints identical ticket_ids
    without ever needing the ticket's own disclosed ``issued_at`` to be a fake
    day-truncated stand-in (Stage 20 / B6) -- ``issued_at`` is instead the real
    ``evaluated_at`` wall-clock instant, passed separately.
    """
    d = stamp.astimezone(timezone.utc)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


# Predictions.json older than this relative to evaluated_at, or dated on a
# different Dublin calendar day, is refused outright rather than re-ticketed
# with today's stamp (Stage 20 / B6, GAP-B). Generous enough to cover a normal
# once-daily refresh cadence; a stale card past this is a genuine capture gap,
# not something to paper over.
_MAX_PREDICTIONS_AGE_SECONDS = 20 * 3600
# Tolerance for the predictions file claiming to be generated after the real
# decision instant -- small clock skew between processes, not a real future.
_CLOCK_SKEW_TOLERANCE_SECONDS = 120


def _predictions_freshness_reason(payload, *, evaluated_at: datetime) -> Optional[str]:
    """``None`` when ``payload["generated_at"]`` is trustworthy for issuing
    tickets *today*; otherwise a short machine-readable rejection reason.

    Three independent fail-closed checks, all Stage 20 / B6 requirements:
    unreadable/missing (never treated as fresh), timestamped in the future
    beyond ordinary clock skew, and dated on a different Dublin calendar day
    than the real decision instant -- computed via ``utils.timezone.to_local``
    so a Dublin DST transition or a card generated right around local midnight
    is judged on the real wall clock, never a naive UTC date comparison.
    """
    from utils.timezone import to_local

    raw = payload.get("generated_at") if isinstance(payload, dict) else None
    generated = _parse_ts(raw)
    if generated is None:
        return f"predictions_generated_at_unreadable:{raw!r}"
    skew = (generated - evaluated_at).total_seconds()
    if skew > _CLOCK_SKEW_TOLERANCE_SECONDS:
        return f"predictions_generated_at_future:{skew:.0f}s"
    age = (evaluated_at - generated).total_seconds()
    if age > _MAX_PREDICTIONS_AGE_SECONDS:
        return f"predictions_stale:{age:.0f}s"
    gen_day = to_local(generated).date()
    eval_day = to_local(evaluated_at).date()
    if gen_day != eval_day:
        return f"predictions_prior_day:{gen_day.isoformat()}!={eval_day.isoformat()}"
    return None


# ── capture ──────────────────────────────────────────────────────────────────
def _capture() -> bool:
    """Run the live scrape (which itself appends this poll to the snapshot
    store). Returns False if the scrape step raised outright; a per-source
    scraper failure or a capture failure inside it is already tolerated by
    ``scripts.refresh._scrape`` and does not surface here."""
    from scripts.refresh import _scrape

    try:
        _scrape()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("daily_paper_loop: capture step raised (%s)", exc)
        print(f"[capture] FAILED ({exc})")
        return False


def _refresh_results_archive(ticket_store: TicketStore, *, no_scrape: bool) -> Optional[dict]:
    """Best-effort network refresh of the raw Sporting-Life results archive
    (Stage 21 / B3), covering the date range of every still-open ticket.

    ``execution.race_facts`` only ever *reads* that archive (never fetches);
    without something running the fetch, a race's Rule 4/dead-heat/non-runner
    facts would never arrive and every settlement would silently fall back to
    the coarser betSP join forever, even once a bookmaker's own results page
    has long since published them. This reuses ``scripts.fetch_results_window``
    -- the same tested backbone+enrichment+raw-doc-write path stage 06 already
    exercises -- rather than a second fetcher.

    Deliberately never requests today or a future day: Betfair's own SP file
    for a given local race date is not published until the following day
    (D31), so a same-day request cannot return anything and would only cost a
    wasted round trip every cycle. Best-effort like ``_capture()`` -- a
    network failure here must not stop issuance or settlement from running
    against whatever is already on disk; skipped entirely under
    ``--no-scrape``, matching every other network step in this loop.
    """
    if no_scrape:
        return None
    try:
        open_tickets = ticket_store.open_tickets()
    except Exception as exc:  # noqa: BLE001
        logger.warning("daily_paper_loop: could not read open tickets for results refresh (%s)", exc)
        return None
    dates = sorted({d for d in (_day_key(t.race_time) for t in open_tickets) if d})
    if not dates:
        return None
    start = date.fromisoformat(dates[0])
    end = min(date.fromisoformat(dates[-1]), _now_utc().date() - _timedelta(days=1))
    if start > end:
        return None
    try:
        from scripts.fetch_results_window import fetch_window

        joined = fetch_window(start, end)
        result = {"start": start.isoformat(), "end": end.isoformat(), "rows": int(len(joined))}
        print(f"  results archive -> {result['rows']} row(s) {result['start']}..{result['end']}")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("daily_paper_loop: results archive refresh failed (%s)", exc)
        print(f"  results archive -> FAILED ({exc})")
        return {"error": str(exc)}


def _today_has_races(predictions_path: str) -> bool:
    if not os.path.exists(predictions_path):
        return False
    try:
        with open(predictions_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    races = payload.get("races") if isinstance(payload, dict) else payload
    return isinstance(races, list) and len(races) > 0


def _record_gap_ledger(*, cfg: ExecutionConfig, day: date, predictions_path: str,
                        gap_ledger_path: str, now: datetime) -> dict:
    """A day counts as captured only if the snapshot store actually has a row for
    it -- not merely because the scrape step ran without raising. A day with no
    races is recorded distinctly from a day races existed but nothing was
    captured, so a genuinely dead scraper is never misfiled as a quiet no-bet
    day."""
    captured_dates = gap_ledger._captured_dates_from_snapshots(cfg.snapshots.db_path)
    captured = day.isoformat() in captured_dates
    if captured:
        cause = None
    elif not _today_has_races(predictions_path):
        cause = gap_ledger.CAUSE_NO_RACING
    else:
        cause = gap_ledger.CAUSE_SCRAPER_OUTAGE
    return gap_ledger.record_day(
        gap_ledger_path, day=day, captured=captured, cause=cause, now=now
    )


# ── today's card -> tickets ──────────────────────────────────────────────────
def _today_tickets(*, cfg: ExecutionConfig, verdict, gate, predictions_path: str,
                    now: datetime, evaluated_at: datetime,
                    snap_store: Optional[SnapshotStore] = None,
                    config_hash: Optional[str] = None,
                    ticket_store: Optional[TicketStore] = None,
                    bankroll: float = 1000.0) -> tuple[list[Ticket], Optional[str]]:
    """Mirrors ``scripts.paper_betting.today_candidates`` but returns the
    ``Ticket`` objects themselves (candidates and passes both) instead of
    discarding them as dicts -- this loop needs the object to issue it.

    Returns ``(tickets, reject_reason)``. ``reject_reason`` is set (and
    ``tickets`` is always ``[]``) when the whole predictions file failed the
    freshness/provenance check (Stage 20 / B6) -- an unreadable, future-dated
    or prior-day ``generated_at`` means nothing on the file is issued this
    cycle, rather than being silently re-ticketed under today's stamp.

    ``now`` is the stable per-day dedupe anchor (see ``_day_start``), passed to
    ``build_ticket`` as ``dedupe_key`` only -- it is no longer stamped onto the
    ticket as ``issued_at`` (Stage 20 / B6: that used to force every disclosed
    issue time to a fake day-truncated midnight). ``evaluated_at`` (the actual
    wall-clock instant this loop is running) is both the gate's decision
    instant AND the ticket's real ``issued_at``.

    ``snap_store`` is threaded through to ``evaluate_race`` as a per-runner
    quote lookup against the immutable, tested ``odds_snapshots`` store
    (``execution.snapshots.SnapshotStore.latest_quote``), read ``as_of
    evaluated_at``, WIN market only (live inference only ever serves WIN
    rows), additionally narrowed by the venue-qualified ``race_key`` (Stage 20
    / B5) so a same-off-time different-venue race can never satisfy the
    lookup. Without a store the gate falls back to whatever price is cached on
    the runner dict in ``predictions.json`` -- a path that carries no
    store-level staleness verdict and is only meant for surfaces with no
    snapshot store wired in (``execution.gates._quote_view``).

    ``ticket_store``, when supplied, seeds the day's stake-exposure tracking
    (Stage 21 / B2) with every CANDIDATE already issued today, so a resumed
    partial run cannot exceed the race/day exposure ceilings by re-planning
    from an empty ledger. ``bankroll`` is the sizing base for
    :class:`execution.staking.StakePlanner`, which also refuses to stake a
    price outside the frozen selection lock's own tested odds band (Stage 21
    / B10, :func:`execution.staking.selection_lock_odds_band`) before any
    ceiling is even consulted. A PASS never carries a stake regardless of what
    the planner would allow (``build_ticket`` already forces ``stake=None``
    for anything that is not a CANDIDATE).
    """
    from execution.gates import (
        CANDIDATE as _CANDIDATE_DECISION,
        evaluate_race,
        horse_key as _horse_key,
        race_key as _race_key_of,
        race_uid as _race_uid,
    )
    from execution.staking import ExposureState, StakePlanner, selection_lock_odds_band

    if not os.path.exists(predictions_path):
        logger.info("daily_paper_loop: no %s; nothing to gate today", predictions_path)
        return [], None
    try:
        with open(predictions_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("daily_paper_loop: could not read %s (%s)", predictions_path, exc)
        return [], f"predictions_unreadable:{exc}"

    reject_reason = _predictions_freshness_reason(payload, evaluated_at=evaluated_at)
    if reject_reason is not None:
        logger.warning(
            "daily_paper_loop: predictions.json rejected (%s) -- issuing nothing "
            "this cycle rather than re-ticketing a stale/unreadable card",
            reject_reason,
        )
        return [], reject_reason

    races = payload.get("races") if isinstance(payload, dict) else payload
    if not isinstance(races, list):
        return [], None

    cycle_provenance = dict(payload.get("provenance") or {}) if isinstance(payload, dict) else {}
    if config_hash is not None:
        cycle_provenance["config_hash"] = config_hash

    try:
        from utils.source_health import get_health
        health = get_health() or {}
    except Exception:  # noqa: BLE001
        health = {}

    # Stage 21 / B2: size every CANDIDATE against real bankroll/exposure
    # ceilings instead of leaving it permanently unstaked. `exposure` is
    # threaded (not rebuilt) across every runner/race below so a second
    # candidate today -- in the same race or merely the same day -- sees the
    # first one's stake already consuming its share of the ceiling; seeding
    # it from `ticket_store` additionally covers a resumed partial run.
    planner = StakePlanner(cfg, bankroll)
    odds_band = selection_lock_odds_band(cfg)
    seed_tickets: list[dict] = []
    if ticket_store is not None:
        try:
            seed_tickets = [t.to_dict() for t in ticket_store.all_tickets(decision=_CANDIDATE_DECISION)]
        except Exception as exc:  # noqa: BLE001 - a seeding failure must not block issuance
            logger.warning("daily_paper_loop: could not seed stake exposure (%s)", exc)
    exposure = ExposureState.from_tickets(seed_tickets)

    tickets: list[Ticket] = []
    for race in races:
        if not isinstance(race, dict):
            continue

        quotes = None
        if snap_store is not None:
            r_uid = _race_uid(race)
            r_key = _race_key_of(race)

            def _quote_for(runner, _r_uid=r_uid, _r_key=r_key):
                try:
                    return snap_store.latest_quote(
                        _r_uid, _horse_key(runner), evaluated_at, market_type="WIN",
                        race_key=_r_key or None,
                    )
                except Exception as exc:  # noqa: BLE001 - a lookup failure is not a fill price
                    logger.warning(
                        "daily_paper_loop: snapshot quote lookup failed for %s (%s)",
                        _r_uid, exc,
                    )
                    return None

            quotes = _quote_for

        try:
            results = evaluate_race(
                race, cfg=cfg, model_verdict=verdict, source_health=health,
                now=evaluated_at, quotes=quotes,
            )
        except Exception as exc:  # pragma: no cover - a malformed race is a PASS
            logger.warning("daily_paper_loop: gate could not evaluate a race (%s)", exc)
            continue
        runners = race.get("runners") or race.get("selections") or []
        for runner, result in zip(runners, results):
            stake_decision = None
            if result.decision == _CANDIDATE_DECISION:
                detail = result.detail or {}
                # Exposure is tracked on the venue-qualified `race_key`
                # (Stage 20 / D53), not the bare `race_uid` -- `race_uid`
                # drops venue once an off-time parses, so two genuinely
                # different races sharing an exact off-time (a real,
                # reproduced event on this project's own store) would
                # otherwise share one race's stake ceiling between them.
                # `race_uid` is only the fallback when a race_key cannot be
                # formed at all (a degraded key must still separate races by
                # *something*, never merge them onto "").
                exposure_key = (
                    _s(detail.get("race_key")) or _race_key_of(race)
                    or _s(detail.get("race_uid")) or _race_uid(race)
                )
                stake_decision = planner.plan(
                    prob=detail.get("model_prob"),
                    decimal_odds=detail.get("executable_odds"),
                    race_uid=exposure_key,
                    race_date=race.get("race_time"),
                    exposure=exposure,
                    odds_band=odds_band,
                )
                exposure = exposure.with_stake(
                    race_uid=exposure_key, race_date=race.get("race_time"),
                    stake=stake_decision.stake,
                )
            ticket = build_ticket(
                runner, race, result, cfg=cfg, model_verdict=verdict,
                forward_gate_state=gate.state_label, now=evaluated_at,
                dedupe_key=now.date().isoformat(), provenance=cycle_provenance,
                stake_decision=stake_decision,
            )
            tickets.append(ticket)
    return tickets, None


def _issue_tickets(store: TicketStore, tickets: list[Ticket]) -> dict:
    issued = duplicates = 0
    for ticket in tickets:
        try:
            store.issue(ticket)
            issued += 1
        except RealMoneyTicketRefused:
            raise
        except ValueError as exc:
            if "already issued" in str(exc):
                duplicates += 1
                continue
            raise
    return {"issued": issued, "duplicates": duplicates, "seen": len(tickets)}


# ── settlement ───────────────────────────────────────────────────────────────
def _has_value(v) -> bool:
    if v is None:
        return False
    if isinstance(v, float) and pd.isna(v):
        return False
    return str(v).strip() != ""


def _as_position(v) -> Optional[int]:
    if not _has_value(v):
        return None
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return None


def _day_key(value) -> str:
    return str(value or "")[:10]


def _load_results(storage=None) -> list[dict]:
    if storage is None:
        from utils.storage import get_storage
        storage = get_storage()
    try:
        frame = storage.read_parquet("betsp")
    except Exception as exc:  # noqa: BLE001 - missing/empty dataset is non-fatal
        logger.warning("daily_paper_loop: could not read results dataset (%s)", exc)
        return []
    if frame is None or (isinstance(frame, pd.DataFrame) and frame.empty):
        return []
    cols = [c for c in ("venue", "horse_name", "horse_id", "race_date", "position")
            if c in frame.columns]
    return frame[cols].to_dict("records")


def _identity_key(
    *, venue, race_time, horse_key_value: Optional[str] = None,
    horse_name=None, horse_id=None,
) -> Optional[tuple]:
    """Canonical ``(venue, off-time-to-the-minute, horse)`` settlement key.

    Deliberately independent of ``execution.snapshots.race_uid`` -- that
    function drops venue entirely once an off-time parses (it exists to join
    the same race across bookmakers, not to disambiguate races), and two
    meetings sharing an exact post time is a real, reproduced event on this
    project's own results store (Leopardstown and York both went off at
    16:15 on 2024-05-17). A settlement join must never let a same-time race
    at a *different* venue satisfy it, so venue is always an explicit,
    separate key component here, never folded into a single off-time string.
    """
    dt = _parse_ts(race_time)
    if dt is None:
        return None
    venue_n = norm_venue(venue)
    hkey = horse_key_value or _mk_horse_key(horse_name, horse_id)
    if not venue_n or not hkey:
        return None
    return (venue_n, dt.strftime("%Y-%m-%dT%H:%M"), hkey)


def _index_results(results: list[dict]):
    """Canonical race+horse identity first; every fallback stays day/venue-scoped.

    The previous version indexed ``by_id`` on ``horse_id`` **alone**, with no
    venue or date restriction at all: any open ticket for that horse would
    match whichever result row for it happened to be read last -- including a
    completely different race, on a different day, at a different course.
    Reproducible on the real store: the same ``horse_id`` recurs across
    multiple dates (a horse running more than once a season is the normal
    case, not an edge case), so the old lookup could settle a ticket against
    another race's result with no warning.

    ``by_key`` is the canonical (venue, off-time-to-the-minute, horse) index
    (see :func:`_identity_key`). ``by_fallback_id``/``by_fallback_name`` exist
    only for rows whose ``race_date`` carried no usable time-of-day (a
    day-only value) -- both are scoped to ``(venue, day)``, never global. A
    key that collects two rows disagreeing on the finishing position is
    dropped from its index entirely rather than resolved by "last write
    wins": settlement must fail closed on an ambiguous match, never silently
    select one of two conflicting results. A duplicate fetch of the *same*
    result (identical or still-missing position) is not ambiguity, and a
    later fetch that fills in a previously-missing position is preferred over
    the earlier placeholder -- this is how a late result is picked up without
    ever letting an incomplete re-fetch regress a known one.
    """
    by_key: dict[tuple, dict] = {}
    by_fallback_id: dict[tuple, dict] = {}
    by_fallback_name: dict[tuple, dict] = {}
    bad_key: set[tuple] = set()
    bad_fallback_id: set[tuple] = set()
    bad_fallback_name: set[tuple] = set()
    raced_days: set[tuple] = set()
    raced_races: set[tuple] = set()

    def _merge(index: dict, bad: set, k: tuple, row: dict) -> None:
        prev = index.get(k)
        if prev is None:
            index[k] = row
            return
        pa, pb = _as_position(prev.get("position")), _as_position(row.get("position"))
        if pa is not None and pb is not None and pa != pb:
            bad.add(k)
            return
        if pa is None and pb is not None:
            index[k] = row

    for r in results:
        venue_n = norm_venue(r.get("venue"))
        day = _day_key(r.get("race_date"))
        raced_days.add((venue_n, day))

        key = _identity_key(
            venue=r.get("venue"), race_time=r.get("race_date"),
            horse_name=r.get("horse_name"), horse_id=r.get("horse_id"),
        )
        if key is not None:
            raced_races.add((key[0], key[1]))
            _merge(by_key, bad_key, key, r)

        if _has_value(r.get("horse_id")):
            _merge(by_fallback_id, bad_fallback_id, (venue_n, day, str(r["horse_id"])), r)
        _merge(by_fallback_name, bad_fallback_name, (venue_n, norm_horse(r.get("horse_name")), day), r)

    for k in bad_key:
        by_key.pop(k, None)
    for k in bad_fallback_id:
        by_fallback_id.pop(k, None)
    for k in bad_fallback_name:
        by_fallback_name.pop(k, None)

    ambiguous = (
        {("key", k) for k in bad_key}
        | {("id", k) for k in bad_fallback_id}
        | {("name", k) for k in bad_fallback_name}
    )
    return by_key, by_fallback_id, by_fallback_name, raced_days, raced_races, ambiguous


def _race_facts_key_for(venue, race_time) -> Optional[str]:
    """``execution.race_facts.race_facts_key`` for a ticket/race.

    ``execution.race_facts`` builds its key from the race's LOCAL (Dublin)
    off time (``_off_datetime`` explicitly converts the archive's UTC clock),
    while ``execution.snapshots.race_key``/``execution.gates.race_key``
    convert to UTC instead -- the two disagree by the DST offset for roughly
    half the year (BST/IST, late March to late October). Reusing either of
    those here would silently fail to join a ticket to its own settlement
    facts through the summer. This helper reproduces race_facts' own
    convention directly: parse whatever timestamp the ticket carries, convert
    to local racing time, then build the key the identical way
    ``race_facts_key`` does.
    """
    from execution.race_facts import race_facts_key
    from utils.timezone import to_local

    dt = _parse_ts(race_time)
    if dt is None or not venue:
        return None
    return race_facts_key(venue, to_local(dt).strftime("%Y-%m-%dT%H:%M"))


def _load_race_facts_for(race_times) -> dict:
    """Read-only parse of whatever raw Sporting-Life documents already sit on
    disk for the exact calendar days ``race_times`` fall on (Stage 21 / B3).

    Parses each distinct day individually (:func:`execution.race_facts.extract_day`)
    rather than the whole ``[min, max]`` span -- two open tickets weeks apart
    must not force a scan of every day in between, most of which would be
    unrelated real archive content. Never writes the shared
    ``data/execution/race_facts.parquet`` cache and never fetches over the
    network -- populating the raw archive for a new day is
    :func:`_refresh_results_archive`'s job, run separately and only from
    :func:`run`. A day with no archived documents simply contributes no facts
    (a data gap the caller falls back on), never an exception.
    """
    dates = sorted({d for d in (_day_key(t) for t in race_times) if d})
    if not dates:
        return {}
    try:
        from datetime import date as _date

        from execution.race_facts import extract_day

        facts: dict = {}
        for d in dates:
            for fact in extract_day(_date.fromisoformat(d)):
                facts[fact.race_key] = fact
        return facts
    except Exception as exc:  # noqa: BLE001 - an unreadable archive is a data gap
        logger.warning("daily_paper_loop: race_facts archive unavailable (%s)", exc)
        return {}


def _settle_via_race_facts(ticket: Ticket, facts_by_key: dict, *, cfg: ExecutionConfig) -> Optional[dict]:
    """Settle one ticket through the tested :mod:`execution.settlement` engine
    using real Rule 4 / dead-heat / non-runner / DNF facts (Stage 21 / B3).

    Returns a payload for :meth:`TicketStore.settle`, or ``None`` when this
    race or runner is not covered by the archive -- the caller must then fall
    back to the coarser betsp-position join, never guess here.
    """
    from execution.settlement import EachWayTerms, STATUS_NO_RESULT, settle_ticket

    if not facts_by_key or ticket.offered_odds is None:
        return None
    key = _race_facts_key_for(ticket.venue, ticket.race_time)
    fact = facts_by_key.get(key) if key else None
    if fact is None:
        return None
    runner_fact = fact.runner(ticket.horse_name) if ticket.horse_name else None
    if runner_fact is None:
        return None

    bet_type = (ticket.bet_type or "win").strip().lower()
    ew_terms = None
    if bet_type not in ("win", ""):
        if not (ticket.ew_places and ticket.ew_reduction):
            # Each-way ticket with no terms captured at bet time -- settlement
            # refuses to assume house terms; leave it for the legacy path
            # (which is also win-only) or a future cycle, never guessed here.
            return None
        ew_terms = EachWayTerms(places=int(ticket.ew_places), fraction=float(ticket.ew_reduction))

    settlement = settle_ticket(
        stake=ticket.stake or 0.0,
        decimal_odds=ticket.offered_odds,
        bet_type=ticket.bet_type or "win",
        horse_key=runner_fact.horse_key,
        race_result=fact.to_race_result(),
        ew_terms=ew_terms,
        commission_rate=cfg.frictions.commission_for(ticket.bookmaker),
        cfg=cfg,
    )
    if settlement.status == STATUS_NO_RESULT:
        return None
    outcome = {"WIN": "won", "PLACE": "placed", "LOSE": "lost", "VOID": "void"}.get(
        settlement.status
    )
    if outcome is None:  # pragma: no cover - exhaustive over settle_ticket's vocabulary
        return None
    raw = settlement.to_dict()
    nested = raw.pop("detail", {})
    return {
        "outcome": outcome,
        "returns": settlement.returns,
        "profit": settlement.profit,
        # Flattened one level: settlement.to_dict()'s own top-level fields
        # (rule_4_deduction, dead_heat_divisor, ...) plus its nested `detail`
        # (rule_4_source, withdrawals, ew_terms, ...) -- both surfaces matter
        # for audit and neither key-collides with the other.
        "detail": {"settlement_source": "race_facts", **raw, **nested},
    }


def _settle_open_tickets(store: TicketStore, snap_store: SnapshotStore) -> dict:
    """Settle every open ticket whose race has results available.

    :func:`_settle_via_race_facts` is tried first for every open ticket: when
    the raw Sporting-Life archive already covers a ticket's race, settlement
    goes through the real, tested :mod:`execution.settlement` engine (Rule 4,
    dead heats, explicit non-runner void, DNF-as-loss, each-way legs -- Stage
    21 / B3), not a parallel win/lose calculator.

    Every ticket the archive does not cover falls back to the coarser betSP
    join below: canonical-identity-first (venue + off-time + horse, via
    :func:`_identity_key`), with day/venue-scoped fallbacks only -- see
    :func:`_index_results`. That fallback carries no non-runner/DNF evidence
    (the betSP schema has no such column -- confirmed against
    ``data/historical/betsp.parquet``'s own columns), so it settles only what
    it can prove: a recorded position of 1 is a win, any other recorded
    position is a loss (WIN-only; the fallback never handles each-way). A
    horse present in the archive's results but with a **null** position is
    treated as a loss (it was priced and is not a winner -- the DNF reading),
    but a horse **absent** from the results entirely stays open/pending
    rather than assumed void: only :func:`_settle_via_race_facts`'s explicit
    ``ride_status`` evidence may ever void a ticket for a non-runner or an
    abandoned race (requirement 3 -- "missing/ambiguous facts... stay pending
    with a reason", tracked in ``pending_reasons`` below).

    No result for a different race can settle a ticket, and an ambiguous
    match (two disagreeing results under one key) is refused rather than
    guessed, leaving the ticket open.

    CLV is read only from the live snapshot store's own closing quote, never
    from the historical archive's SP column -- requirement 7 is explicit that an
    archive proxy is not an acceptable substitute. Where no closing snapshot was
    captured, ``closing_odds`` stays ``None`` and ``TicketStore.settle`` records
    an unmeasurable (``None``) ``clv_pct`` rather than inventing one.
    """
    open_tickets = store.open_tickets()
    if not open_tickets:
        return {
            "settled": 0, "already_settled": 0, "still_open": 0, "unmeasurable_clv": 0,
            "settled_via_race_facts": 0, "pending_reasons": {},
        }

    facts_by_key = _load_race_facts_for(t.race_time for t in open_tickets)
    results = _load_results()
    by_key, by_fallback_id, by_fallback_name, _raced_days, _raced_races, ambiguous = (
        _index_results(results)
    )

    settled = already_settled = still_open = unmeasurable_clv = settled_via_race_facts = 0
    pending_reasons: dict[str, int] = {}

    def _pend(reason: str) -> None:
        nonlocal still_open
        still_open += 1
        pending_reasons[reason] = pending_reasons.get(reason, 0) + 1

    for ticket in open_tickets:
        settlement_payload = _settle_via_race_facts(ticket, facts_by_key, cfg=store.cfg)
        outcome: Any = None
        if settlement_payload is not None:
            outcome = settlement_payload
        else:
            day = _day_key(ticket.race_time)
            venue_n = norm_venue(ticket.venue)
            fid_key = (venue_n, day, str(ticket.horse_id)) if _has_value(ticket.horse_id) else None
            fname_key = (venue_n, norm_horse(ticket.horse_name), day)

            key = _identity_key(
                venue=ticket.venue, race_time=ticket.race_time,
                horse_key_value=ticket.horse_key,
            )
            is_ambiguous = (
                (key is not None and ("key", key) in ambiguous)
                or (fid_key is not None and ("id", fid_key) in ambiguous)
                or ("name", fname_key) in ambiguous
            )

            res = by_key.get(key) if key is not None else None
            if res is None and fid_key is not None:
                res = by_fallback_id.get(fid_key)
            if res is None:
                res = by_fallback_name.get(fname_key)

            if res is None:
                # An ambiguous match (two disagreeing results under the same
                # identity) must never be resolved by guessing.
                if is_ambiguous:
                    _pend("ambiguous_result")
                    continue
                # Absent from the archive entirely: could be a genuine
                # non-runner or simply an unjoined row -- the betSP schema
                # carries no ride-status column to tell the two apart
                # (requirement 3), so this stays pending rather than the
                # previous behaviour of assuming void.
                _pend("no_result_recorded")
                continue
            position = _as_position(res.get("position"))
            if position == 1:
                outcome = "won"
            elif position is None:
                # Present in the results (so it was priced -- betSP would not
                # carry a row for a horse that never had a market) but no
                # finishing position: the DNF reading, a loss, never a void
                # refund (requirement 3).
                outcome = "lost"
            else:
                outcome = "lost"

        quote = None
        try:
            quote = snap_store.closing_quote(
                ticket.race_uid, ticket.horse_key,
                market_type=ticket.market_type, bookmaker=ticket.bookmaker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("daily_paper_loop: closing_quote lookup failed for %s (%s)",
                            ticket.ticket_id, exc)
        closing_odds = quote.odds_decimal if quote is not None else None
        if closing_odds is None:
            unmeasurable_clv += 1

        try:
            store.settle(ticket.ticket_id, outcome, closing_odds=closing_odds)
            settled += 1
            if settlement_payload is not None:
                settled_via_race_facts += 1
        except ValueError as exc:
            if "already settled" in str(exc):
                already_settled += 1
                continue
            raise

    return {
        "settled": settled, "already_settled": already_settled,
        "still_open": still_open, "unmeasurable_clv": unmeasurable_clv,
        "settled_via_race_facts": settled_via_race_facts,
        "pending_reasons": pending_reasons,
    }


# ── PASS / shadow observation (Stage 21 / B4) ───────────────────────────────
def _observe_pass_tickets(store: TicketStore, snap_store: SnapshotStore) -> dict:
    """Reconcile results and closing prices for PASS tickets -- a SEPARATE
    lane from :func:`_settle_open_tickets`.

    A PASS is never a bet: it carries no stake, and this function never
    changes that (``TicketStore.settle`` computes profit/returns from the
    ticket's own recorded stake, which for a PASS is always ``None`` -> 0.0).
    What a PASS *does* carry, once observed, is exactly the evidence that
    tells us whether passing was the right call: what actually won, and what
    the closing price ended up being versus what was offered at decision
    time. Reusing :meth:`TicketStore.settle` (rather than a second ledger)
    means this evidence lives in the same ``paper_tickets`` row the candidate
    lane uses, distinguished only by ``decision == 'PASS'`` -- so it can never
    be mistaken for, or summed into, a qualified bet, bankroll, ROI or
    drawdown figure (:func:`_forward_ledger` already filters
    ``decision == 'CANDIDATE'`` before any of those are computed).

    Uses the same identity/result machinery as :func:`_settle_open_tickets`
    (race_facts first, betSP fallback second) so a PASS and a CANDIDATE for
    the same race always agree on what happened.
    """
    open_passes = store.unobserved_pass_tickets()
    if not open_passes:
        return {"observed": 0, "already_observed": 0, "still_open": 0, "unmeasurable_clv": 0}

    facts_by_key = _load_race_facts_for(t.race_time for t in open_passes)
    results = _load_results()
    by_key, by_fallback_id, by_fallback_name, _raced_days, _raced_races, ambiguous = (
        _index_results(results)
    )

    observed = already_observed = still_open = unmeasurable_clv = 0
    for ticket in open_passes:
        settlement_payload = _settle_via_race_facts(ticket, facts_by_key, cfg=store.cfg)
        outcome: Any = None
        if settlement_payload is not None:
            outcome = settlement_payload
        else:
            day = _day_key(ticket.race_time)
            venue_n = norm_venue(ticket.venue)
            fid_key = (venue_n, day, str(ticket.horse_id)) if _has_value(ticket.horse_id) else None
            fname_key = (venue_n, norm_horse(ticket.horse_name), day)
            key = _identity_key(
                venue=ticket.venue, race_time=ticket.race_time,
                horse_key_value=ticket.horse_key,
            )
            is_ambiguous = (
                (key is not None and ("key", key) in ambiguous)
                or (fid_key is not None and ("id", fid_key) in ambiguous)
                or ("name", fname_key) in ambiguous
            )
            res = by_key.get(key) if key is not None else None
            if res is None and fid_key is not None:
                res = by_fallback_id.get(fid_key)
            if res is None:
                res = by_fallback_name.get(fname_key)
            if res is None or is_ambiguous:
                still_open += 1
                continue
            position = _as_position(res.get("position"))
            outcome = "won" if position == 1 else "lost"

        quote = None
        try:
            quote = snap_store.closing_quote(
                ticket.race_uid, ticket.horse_key,
                market_type=ticket.market_type, bookmaker=ticket.bookmaker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("daily_paper_loop: closing_quote lookup failed for %s (%s)",
                            ticket.ticket_id, exc)
        closing_odds = quote.odds_decimal if quote is not None else None
        if closing_odds is None:
            unmeasurable_clv += 1

        try:
            store.settle(ticket.ticket_id, outcome, closing_odds=closing_odds)
            observed += 1
        except ValueError as exc:
            if "already settled" in str(exc):
                already_observed += 1
                continue
            raise

    return {
        "observed": observed, "already_observed": already_observed,
        "still_open": still_open, "unmeasurable_clv": unmeasurable_clv,
    }


# ── ledger for the forward gate / strategy metrics ──────────────────────────
def _forward_ledger(store: TicketStore) -> pd.DataFrame:
    """Every CANDIDATE ticket, in the column shape both
    ``execution.forward_gate.evaluate_forward_gate`` and
    ``execution.evaluation.evaluate_ledger`` expect.

    ``qualified`` marks settled candidates WITH COMPLETE PROVENANCE only -- an
    open (unsettled) ticket is real evidence of a decision but not yet a
    resolved bet, and a ticket missing its model/feature/config/cycle
    provenance cannot be traced back to what actually produced it (Stage 20 /
    B7: "missing or inconsistent provenance must exclude formal eligibility").
    Without this column ``forward_gate._qualified`` would treat every CANDIDATE
    row (including still-open or unprovenanced ones) as a qualifying bet.
    """
    frame = store.to_frame(decision="CANDIDATE")
    if frame is None or len(frame) == 0:
        return pd.DataFrame()
    frame = frame.copy()

    settled_mask = (
        frame["settled_at"].notna() if "settled_at" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    provenance_mask = (
        frame["provenance_complete"].fillna(False).astype(bool)
        if "provenance_complete" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    frame["qualified"] = settled_mask & provenance_mask

    if "offered_odds" in frame.columns:
        frame["decimal_odds"] = pd.to_numeric(frame["offered_odds"], errors="coerce")
    if "race_time" in frame.columns:
        frame["race_date"] = pd.to_datetime(frame["race_time"], errors="coerce")

    if "outcome" in frame.columns:
        outcome = frame["outcome"]
        # "placed" (an each-way ticket whose place leg won, win leg did not)
        # scores as a WIN-probability miss, same as "lost" -- the win did not
        # happen, even though the ticket returned money.
        frame["won"] = outcome.map(
            {"won": 1.0, "placed": 0.0, "lost": 0.0, "void": float("nan")}
        )
        frame["voided"] = outcome.eq("void")
    else:
        frame["won"] = float("nan")
        frame["voided"] = False

    return frame


def _paper_summary(store: TicketStore) -> dict:
    """Decisions logged vs. wagers -- kept visibly separate.

    ``n_tickets`` is every disclosed decision, PASS included -- on a NO-GO
    day that is nearly the whole count (this run's own gate forces PASS
    regardless of the runner). Reporting that count as "open tickets" (the
    previous behaviour: ``n_open``/``n_settled`` summed over every decision)
    reads as hundreds of live exposures that never existed -- a PASS carries
    no stake and was never a bet to open or settle. ``n_open``/``n_settled``
    are therefore scoped to CANDIDATE decisions only.

    ``n_pass_priced``/``n_pass_observed`` (Stage 21 / B4) are the separate
    PASS/shadow observation lane: how many PASS tickets ever carried an
    executable price to observe, and how many of those have since had a
    result + closing price reconciled via :func:`_observe_pass_tickets`. Both
    stay zero-stake by construction and are never folded into
    ``n_candidates``/``n_settled`` or any bankroll/ROI/drawdown figure.
    """
    frame = store.to_frame()
    empty = {
        "n_tickets": 0, "n_pass": 0, "n_candidates": 0,
        "n_open": 0, "n_settled": 0,
        "n_pass_priced": 0, "n_pass_observed": 0,
        "first_ticket": None, "last_ticket": None,
    }
    if frame is None or len(frame) == 0:
        return empty
    is_candidate = (
        frame["decision"].eq("CANDIDATE") if "decision" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    candidates = frame[is_candidate]
    settled = (
        candidates["outcome"].notna() if "outcome" in candidates.columns
        else pd.Series(False, index=candidates.index)
    )
    passes = frame[~is_candidate]
    pass_priced = (
        passes["offered_odds"].notna() if "offered_odds" in passes.columns
        else pd.Series(False, index=passes.index)
    )
    pass_observed = (
        passes["outcome"].notna() if "outcome" in passes.columns
        else pd.Series(False, index=passes.index)
    )
    issued = frame["issued_at"] if "issued_at" in frame.columns else pd.Series(dtype=object)
    return {
        "n_tickets": int(len(frame)),
        "n_pass": int((~is_candidate).sum()),
        "n_candidates": int(is_candidate.sum()),
        "n_settled": int(settled.sum()),
        "n_open": int((~settled).sum()),
        "n_pass_priced": int(pass_priced.sum()),
        "n_pass_observed": int(pass_observed.sum()),
        "first_ticket": str(issued.min()) if len(issued) else None,
        "last_ticket": str(issued.max()) if len(issued) else None,
    }


# ── the run ──────────────────────────────────────────────────────────────────
def run(
    *,
    cfg: Optional[ExecutionConfig] = None,
    no_scrape: bool = False,
    predictions_path: str = DEFAULT_PREDICTIONS_PATH,
    gap_ledger_path: str = gap_ledger.DEFAULT_GAP_LEDGER_PATH,
    window_path: str = window.DEFAULT_WINDOW_PATH,
    report_dir: Optional[str] = None,
    bankroll: float = 1000.0,
    now: Optional[datetime] = None,
) -> dict:
    cfg = cfg or load_execution_config()
    report_dir = report_dir or cfg.reports.dir
    stamp = now or _now_utc()
    day = stamp.date()
    day_start = _day_start(stamp)

    # 1. Capture. Best-effort; a failure here must not stop the rest of the loop.
    capture_ran = True if no_scrape else _capture()

    days = _record_gap_ledger(
        cfg=cfg, day=day, predictions_path=predictions_path,
        gap_ledger_path=gap_ledger_path, now=stamp,
    )
    capture_active = gap_ledger.is_qualifying_day(days, day)

    # 2. Window: re-verify every run. Never start it here -- that stays a
    #    separate, deliberate action (execution.window's own docstring).
    hashes = window.compute_state_hashes()
    win_state, win_changes = window.verify_window(window_path, hashes=hashes, now=stamp)
    if win_changes:
        print(f"[window] RESET: {'; '.join(win_changes)}")

    # 3. Gap reconciliation for any past day the loop never ran on, once a
    #    window is actually running.
    if win_state.window_start:
        days = gap_ledger.reconcile_gaps(
            gap_ledger_path, window_start=date.fromisoformat(win_state.window_start),
            db_path=cfg.snapshots.db_path, today=day,
        )

    # 4. Model gate.
    verdict = load_model_verdict()

    ticket_store = TicketStore(cfg=cfg)
    snap_store = SnapshotStore(cfg=cfg)
    try:
        weeks_elapsed = gap_ledger.qualifying_weeks(
            days, window_start=date.fromisoformat(win_state.window_start)
            if win_state.window_start else None,
        )

        # 5. Gate today's card against the *pre-issuance* ledger's gate state,
        #    then issue every candidate. `now=day_start` is only the dedupe
        #    anchor (idempotent issue on a same-day re-run); each ticket's real
        #    issued_at/decision instant is `evaluated_at=stamp` (Stage 20 / B6).
        pre_ledger = _forward_ledger(ticket_store)
        pre_gate = evaluate_forward_gate(
            pre_ledger, cfg=cfg, model_verdict=verdict,
            metrics={"weeks_elapsed": weeks_elapsed}, evidence_kind="forward", now=stamp,
        )
        from execution.config import config_fingerprint

        today_tickets, predictions_reject_reason = _today_tickets(
            cfg=cfg, verdict=verdict, gate=pre_gate,
            predictions_path=predictions_path, now=day_start,
            evaluated_at=stamp, snap_store=snap_store,
            config_hash=config_fingerprint(cfg),
            ticket_store=ticket_store, bankroll=bankroll,
        )
        issue_stats = _issue_tickets(ticket_store, today_tickets)
        issue_stats["predictions_reject_reason"] = predictions_reject_reason

        # 5b. Best-effort: pull any newly-published results (Rule 4/dead-heat/
        #     non-runner facts included) for whatever is still open, so step 6
        #     can settle through the real settlement engine rather than the
        #     coarser betSP-position fallback. Never blocks the rest of the run.
        results_refresh = _refresh_results_archive(ticket_store, no_scrape=no_scrape)

        # 6. Settle whatever is now resolvable -- yesterday's tickets and any
        #    other still-open one, never limited to "yesterday" specifically,
        #    since a lagging result can leave a ticket open for more than a day.
        settle_stats = _settle_open_tickets(ticket_store, snap_store)

        # 6b. Separate lane: reconcile results/closing prices for PASS tickets
        #     (Stage 21 / B4) -- disclosure evidence only, never a wager.
        observe_stats = _observe_pass_tickets(ticket_store, snap_store)

        # 7. Recompute forward metrics against the now-updated ledger.
        post_ledger = _forward_ledger(ticket_store)
        post_gate = evaluate_forward_gate(
            post_ledger, cfg=cfg, model_verdict=verdict,
            metrics={"weeks_elapsed": weeks_elapsed}, evidence_kind="forward", now=stamp,
        )
        paper_stats = _paper_summary(ticket_store)
        settled_ledger = (
            post_ledger[post_ledger["qualified"]] if len(post_ledger) else post_ledger
        )
        paper_metrics = {}
        if len(settled_ledger):
            from execution import evaluation
            paper_metrics = {"paper": evaluation.evaluate_ledger(
                settled_ledger, name="paper", initial_bankroll=bankroll,
                n_boot=int(cfg.selection.bootstrap_resamples), seed=int(cfg.selection.seed),
            )}
    finally:
        ticket_store.close()
        snap_store.close()

    releasable = bool(verdict.go) and bool(post_gate.passed)
    deployment = "CANDIDATE-ELIGIBLE" if (releasable and not cfg.paper_only) else "PAPER-ONLY"

    # 8. Reports.
    fv_text = forward_validation_report(
        model_verdict=verdict,
        forward_gate=post_gate,
        paper_metrics=paper_metrics or None,
        paper_summary=paper_stats,
        shadow_observation=observe_stats,
        deployment_state=deployment,
        window_state=win_state.to_dict(),
        gap_days=days,
        weeks_elapsed_qualifying=weeks_elapsed,
        now=stamp,
    )
    fv_path = write_report(fv_text, name="forward_validation", report_dir=report_dir, now=stamp)

    payload = {
        "generated_at": stamp.isoformat(timespec="seconds"),
        "day": day.isoformat(),
        "model_verdict": verdict.to_dict(),
        "forward_gate": post_gate.to_dict(),
        "window": win_state.to_dict(),
        "window_reset_reasons": win_changes,
        "capture_active_today": capture_active,
        "gap_entries": gap_ledger.gap_entries(days),
        "weeks_elapsed_qualifying": weeks_elapsed,
        "deployment": deployment,
        "paper_only": bool(cfg.paper_only),
        "issue": issue_stats,
        "results_refresh": results_refresh,
        "settle": settle_stats,
        "observe": observe_stats,
        "paper": paper_stats,
        "reports": {"forward_validation": fv_path},
    }
    write_json(payload, path=os.path.join(cfg.reports.artifact_dir, "daily_loop_run.json"))

    print(
        f"MODEL {verdict.verdict_label} · {post_gate.state_label} · "
        f"CAPTURE {'ACTIVE' if capture_active else 'INACTIVE'} · "
        f"WINDOW {win_state.status.upper()}"
        + (f" (day {win_state.days_elapsed(now=stamp)} of 56)" if win_state.running else "")
        + f" · DEPLOYMENT {deployment}"
    )
    print(f"  issued {issue_stats['issued']} ({issue_stats['duplicates']} duplicate), "
          f"settled {settle_stats['settled']} ({settle_stats['settled_via_race_facts']} via race_facts, "
          f"{settle_stats['already_settled']} already settled, "
          f"{settle_stats['still_open']} still open, {settle_stats['unmeasurable_clv']} unmeasurable CLV)")
    print(f"  PASS/shadow observed {observe_stats['observed']} "
          f"({observe_stats['already_observed']} already, {observe_stats['still_open']} pending, "
          f"{observe_stats['unmeasurable_clv']} unmeasurable CLV)")
    print(f"  report: {fv_path}")
    return payload


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-scrape", action="store_true",
                     help="skip the live scrape/capture step (reuse today's board)")
    ap.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--bankroll", type=float, default=1000.0)
    ap.add_argument("--now", default=None, help="ISO timestamp override, for tests")
    args = ap.parse_args(argv)

    now = datetime.fromisoformat(args.now) if args.now else None
    if now is not None and now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    run(
        no_scrape=args.no_scrape,
        predictions_path=args.predictions_path,
        report_dir=args.report_dir,
        bankroll=args.bankroll,
        now=now,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
