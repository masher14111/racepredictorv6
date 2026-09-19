"""Offline settlement ground truth mined from the raw Sporting Life archive.

Requirement 2 needs non-runners, Rule 4 deductions, dead heats, voids and
places-paid to be **real**, not assumed. All of it is already on disk: the
historical backfill stored 31k raw ``__NEXT_DATA__`` documents under
``data/historical/raw/sporting_life/{year}/{date}/{sha1}.html.gz`` and the
per-race payload carries every field. This module reads them — **no network, at
any point** — and emits one authoritative record per race.

What the archive gives us, measured over the full window:

===========================  ==========================================
``ride_status``              ``RUNNER`` / ``NONRUNNER`` / ``WITHDRAWN``.
                             48.5% of races have >= 1 non-runner.
``race.deductions``          ``[{'type': 'AllBets'|'BoardPrices',
                             'value': <pence per £>}]``. Present on 2.05%
                             of races; values 5-70p. Never more than one
                             entry per race in the sampled days.
``number_of_placed_rides``   Places actually paid, 1-4.
``finish_position``          ``None`` for a horse that ran but did not
                             complete — see the settlement note below.
``casualty``                 ``{}`` or ``{'reason': 'PulledUp'|'Fell'|...}``
                             Dead heats                   Shared ``finish_position``. 0.15% of races.
===========================  ==========================================

**The settlement note.** ``utils/bet_settlement._outcome_for`` maps a missing
finishing position to ``void``, which refunds a losing bet on a horse that
pulled up or fell (~5% of real runners). This module never conflates the two:

* ``ride_status != RUNNER``            -> :data:`NON_RUNNER` (genuinely void)
* ran, no finishing position           -> :data:`LOSE` (it lost; it just did
  not complete)
* horse absent from the race document  -> :data:`UNKNOWN` (a *join* miss —
  settlement must leave the ticket open, never refund it)

Dead heats are *not* readable from the Betfair side of the warehouse: both
dead-heaters are marked ``win_lose=1`` at full BSP, so the half-stake reduction
must be modelled from ``places_won`` here rather than trusted downstream.

Each-way *fraction* has no coverage anywhere in the archive; it stays a
parameterised assumption in :mod:`execution.settlement`, captured at bet time.
"""
from __future__ import annotations

import gzip
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Iterable, Iterator, Optional
from zoneinfo import ZoneInfo

from utils.logger import get_logger
from utils.text_norm import minute_key, norm_horse, norm_venue

logger = get_logger(__name__)

DEFAULT_RAW_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "historical", "raw")
)
DEFAULT_SOURCE = "sporting_life"
DEFAULT_CACHE_PATH = os.path.join("data", "execution", "race_facts.parquet")

# Britain and Ireland keep identical DST rules, so one zone converts both cards.
_RACING_TZ = ZoneInfo("Europe/Dublin")

# Outcome vocabulary. Deliberately distinct from utils.bet_settlement's, which
# folds LOSE-without-position into VOID.
WIN = "win"
PLACE = "place"
LOSE = "lose"
NON_RUNNER = "non_runner"
UNKNOWN = "unknown"

# Finishing position stamped on a horse that ran but did not complete. Large
# enough to be outside any paid place, finite so it settles as a definite loser
# rather than an unknown.
NON_FINISHER_POSITION = 999

_NON_RUNNER_STATUSES = frozenset({"NONRUNNER", "NON_RUNNER", "WITHDRAWN", "WITHDRAWAL"})
_RUNNER_STATUSES = frozenset({"RUNNER"})
_DEDUCTION_TYPES = ("AllBets", "BoardPrices")
# Rule 4 caps at 90p in the £; anything above is a parse error, not a deduction.
_MAX_RULE_4 = 0.90

_FRACTION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$")


# ─────────────────────────── keys and small parsers ──────────────────────────
def race_facts_key(venue: Any, race_time: Any) -> str:
    """The cross-source race key: ``norm_venue(venue)|YYYY-MM-DDTHH:MM``.

    Mirrors ``features.derive.add_race_key`` / ``models.predictor._race_uid``
    but normalises the venue, because bookmakers spell courses differently and
    ``race_id`` is minted per book (see
    ``memory/per-source-race-id-fragments-oddsmap.md``).
    """
    return f"{norm_venue(venue)}|{minute_key(race_time)}"


def fractional_to_decimal(text: Any) -> Optional[float]:
    """``'3/1'`` -> 4.0, ``'evens'`` -> 2.0. Returns ``None`` when unparseable."""
    if text is None:
        return None
    raw = str(text).strip().lower()
    if not raw:
        return None
    if raw in {"evens", "evs", "even", "1/1"}:
        return 2.0
    match = _FRACTION_RE.match(raw)
    if match:
        num, den = float(match.group(1)), float(match.group(2))
        if den <= 0:
            return None
        return round(num / den + 1.0, 4)
    try:  # already decimal?
        value = float(raw)
    except ValueError:
        return None
    return value if value > 1.0 else None


def _int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ────────────────────────────────── records ─────────────────────────────────
@dataclass(frozen=True)
class RunnerFact:
    """One declared runner's settlement-relevant truth."""

    horse_key: str
    horse_name: str = ""
    ride_status: str = ""
    finish_position: Optional[int] = None
    casualty_reason: Optional[str] = None
    industry_sp: Optional[float] = None
    cloth_number: Optional[int] = None
    draw_number: Optional[int] = None
    dead_heat_size: int = 1

    @property
    def is_non_runner(self) -> bool:
        return self.ride_status.upper() in _NON_RUNNER_STATUSES

    @property
    def ran(self) -> bool:
        """Came under starter's orders — a bet on it stands, win or lose."""
        return not self.is_non_runner

    @property
    def completed(self) -> bool:
        """Finished the race with a recorded position."""
        return self.ran and self.finish_position is not None

    @property
    def dead_heat(self) -> bool:
        return self.dead_heat_size > 1


@dataclass(frozen=True)
class RaceFacts:
    """Everything settlement needs about one race, read from the raw document."""

    race_key: str
    race_date: str
    off_time: str = ""
    venue: str = ""
    declared_field: int = 0
    n_runners: int = 0
    places_paid: Optional[int] = None
    rule_4_deduction: float = 0.0
    rule_4_type: str = ""
    going: str = ""
    distance: str = ""
    race_class: str = ""
    handicap: bool = False
    runners: dict[str, RunnerFact] = field(default_factory=dict)
    source_path: str = ""

    # ── lookups ────────────────────────────────────────────────────────────
    @property
    def non_runners(self) -> frozenset[str]:
        return frozenset(k for k, r in self.runners.items() if r.is_non_runner)

    @property
    def winners(self) -> tuple[str, ...]:
        return tuple(
            k for k, r in self.runners.items() if r.finish_position == 1
        )

    def runner(self, horse: Any) -> Optional[RunnerFact]:
        """Look a runner up by raw or normalised name."""
        key = norm_horse(horse)
        return self.runners.get(key)

    def outcome_for(self, horse: Any, *, places: Optional[int] = None) -> str:
        """Settlement outcome for a horse. ``UNKNOWN`` when it is not in the race.

        ``UNKNOWN`` is a *join* failure and must never be settled as a void
        refund; callers leave those tickets open.
        """
        runner = self.runner(horse)
        if runner is None:
            return UNKNOWN
        if runner.is_non_runner:
            return NON_RUNNER
        if runner.finish_position is None:
            # Ran and did not complete: pulled up, fell, unseated, refused.
            # This is a LOSS, not a refund.
            return LOSE
        if runner.finish_position == 1:
            return WIN
        cutoff = places if places is not None else (self.places_paid or 0)
        if cutoff and runner.finish_position <= cutoff:
            return PLACE
        return LOSE

    def to_race_result(self):
        """Bridge to :class:`execution.settlement.RaceResult`.

        The one translation that matters: a horse that ran and did not complete
        is mapped to :data:`NON_FINISHER_POSITION`, an unreachable finishing
        position, so it settles as a **loser**. ``RaceResult.positions`` treats
        an absent key as "unknown" and leaves such tickets unsettled, which is
        right for a join miss and wrong for a faller — the exact conflation that
        makes ``utils.bet_settlement._outcome_for`` refund losing bets.
        """
        from execution.settlement import RaceResult

        positions: dict[str, Optional[int]] = {}
        dead_heats: dict[str, int] = {}
        withdrawn: dict[str, float] = {}
        for key, runner in self.runners.items():
            if runner.is_non_runner:
                if runner.industry_sp is not None:
                    withdrawn[key] = runner.industry_sp
                continue
            positions[key] = (
                runner.finish_position
                if runner.finish_position is not None
                else NON_FINISHER_POSITION
            )
            if runner.dead_heat:
                dead_heats[key] = runner.dead_heat_size
        return RaceResult(
            race_uid=self.race_key,
            positions=positions,
            non_runners=self.non_runners,
            withdrawn_odds=withdrawn,
            # The published deduction, not one derived from withdrawal prices.
            rule_4_override=self.rule_4_deduction if self.rule_4_deduction > 0 else None,
            dead_heat_counts=dead_heats,
            places_paid=self.places_paid,
            void_race=self.n_runners <= 1,
        )

    def to_rows(self) -> list[dict]:
        """Flatten to one row per declared runner (the parquet cache schema)."""
        return [
            {
                "race_key": self.race_key,
                "race_date": self.race_date,
                "off_time": self.off_time,
                "venue": self.venue,
                "declared_field": self.declared_field,
                "n_runners": self.n_runners,
                "places_paid": self.places_paid,
                "rule_4_deduction": self.rule_4_deduction,
                "rule_4_type": self.rule_4_type,
                "going": self.going,
                "distance": self.distance,
                "race_class": self.race_class,
                "handicap": self.handicap,
                "horse_key": r.horse_key,
                "horse_name": r.horse_name,
                "ride_status": r.ride_status,
                "finish_position": r.finish_position,
                "casualty_reason": r.casualty_reason,
                "industry_sp": r.industry_sp,
                "cloth_number": r.cloth_number,
                "draw_number": r.draw_number,
                "dead_heat_size": r.dead_heat_size,
                "source_path": self.source_path,
            }
            for r in self.runners.values()
        ]


# ────────────────────────────────── parsing ─────────────────────────────────
def _rule_4(deductions: Any) -> tuple[float, str]:
    """Rule 4 fraction of net winnings, plus the deduction type it came from.

    Entries are ``{'type': 'AllBets'|'BoardPrices', 'value': <pence per £>}``.
    Multiple withdrawals in one race produce multiple entries of the *same*
    type, which sum; the two types describe overlapping bet populations and so
    are combined with ``max``, never added. We take an early fixed price, so
    both categories can reach us. In the sampled days the two never co-occur,
    making the choice moot in practice but explicit in code.
    """
    if not isinstance(deductions, (list, tuple)):
        return 0.0, ""
    by_type: dict[str, float] = {}
    for entry in deductions:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("type") or "")
        try:
            pence = float(entry.get("value"))
        except (TypeError, ValueError):
            continue
        if pence <= 0:
            continue
        by_type[kind] = by_type.get(kind, 0.0) + pence
    if not by_type:
        return 0.0, ""
    applicable = {k: v for k, v in by_type.items() if k in _DEDUCTION_TYPES}
    if not applicable:  # unknown type — record it but do not silently apply
        logger.debug("race_facts: ignoring deduction types %s", sorted(by_type))
        return 0.0, ""
    kind, pence = max(applicable.items(), key=lambda kv: kv[1])
    fraction = min(pence / 100.0, _MAX_RULE_4)
    return round(fraction, 4), kind


def _off_datetime(summary: dict) -> str:
    """``race_summary`` -> local ``YYYY-MM-DDTHH:MM`` (scheduled off, not actual).

    Sporting Life stores a naive clock in **UTC**; every other race key in this
    repo (``race_uid``, the Betfair panel, ``scraper.betsp.joiner``) is stamped
    in local racing time. Verified by matching whole fields horse-for-horse
    against the panel: a January window lines up at +0 (GMT), an April window at
    +60 (BST), 100% of races matched in both. Ireland and Britain share the same
    DST rule, so one zone covers the whole card.

    Skipping this conversion does not merely lose the summer join — it silently
    makes the *wrong* one, since a 16:20 UTC race at a course usually has a
    sibling at 16:20 local an hour away. Settlement would then come from a
    different race at the same track.
    """
    day = str(summary.get("date") or "").strip()
    clock = str(summary.get("time") or "").strip()
    if not clock:
        clock = str(summary.get("off_time") or "").strip()[:5]
    if not day:
        return ""
    if not clock:
        return day
    try:
        utc = datetime.strptime(f"{day}T{clock[:5]}", "%Y-%m-%dT%H:%M")
    except ValueError:
        return f"{day}T{clock[:5]}"
    local = utc.replace(tzinfo=timezone.utc).astimezone(_RACING_TZ)
    return local.strftime("%Y-%m-%dT%H:%M")


def parse_race_document(payload: Any, *, source_path: str = "") -> Optional[RaceFacts]:
    """Turn a decoded ``__NEXT_DATA__`` document into :class:`RaceFacts`.

    Returns ``None`` for any document that is not a race page or that lacks the
    course/time needed to build a join key — silently skipping is correct here,
    the archive contains index pages too.
    """
    if not isinstance(payload, dict):
        return None
    race = (((payload.get("props") or {}).get("pageProps") or {}).get("race")) or {}
    if not isinstance(race, dict) or not race:
        return None
    summary = race.get("race_summary") or {}
    if not isinstance(summary, dict):
        return None

    venue = str(summary.get("course_name") or "").strip()
    off = _off_datetime(summary)
    if not venue or not off:
        return None

    rides = race.get("rides") or []
    if not isinstance(rides, list):
        rides = []

    runners: dict[str, RunnerFact] = {}
    positions: Counter[int] = Counter()
    for ride in rides:
        if not isinstance(ride, dict):
            continue
        horse = (ride.get("horse") or {}) if isinstance(ride.get("horse"), dict) else {}
        name = str(horse.get("name") or ride.get("horse_name") or "").strip()
        key = norm_horse(name)
        if not key:
            continue
        status = str(ride.get("ride_status") or "").strip().upper()
        # Sporting Life encodes "did not complete" as finish_position 0, not
        # null — verified: every RUNNER at position 0 carries a casualty reason
        # (PulledUp/Fell/UnseatedRider/BroughtDown/RanOut/RefusedToRace), and
        # non-runners are stamped 0 too. Collapsing 0 to None here is what keeps
        # dead-heat detection from grouping all the fallers together.
        position = _int(ride.get("finish_position"))
        if position is not None and position <= 0:
            position = None
        casualty = ride.get("casualty") if isinstance(ride.get("casualty"), dict) else {}
        reason = str((casualty or {}).get("reason") or "").strip() or None
        betting = ride.get("betting") if isinstance(ride.get("betting"), dict) else {}
        if status in _NON_RUNNER_STATUSES:
            position = None  # a non-runner cannot hold a finishing position
        if position is not None and status in _RUNNER_STATUSES:
            positions[position] += 1
        runners[key] = RunnerFact(
            horse_key=key,
            horse_name=name,
            ride_status=status,
            finish_position=position,
            casualty_reason=reason,
            industry_sp=fractional_to_decimal((betting or {}).get("current_odds")),
            cloth_number=_int(ride.get("cloth_number") or horse.get("cloth_number")),
            draw_number=_int(ride.get("draw_number")),
        )

    # Second pass: stamp dead-heat sizes now that every position is known.
    if positions:
        runners = {
            key: (
                runner
                if runner.finish_position is None
                or positions[runner.finish_position] <= 1
                else replace(runner, dead_heat_size=positions[runner.finish_position])
            )
            for key, runner in runners.items()
        }

    deduction, kind = _rule_4(race.get("deductions"))
    n_runners = sum(1 for r in runners.values() if r.ran)

    return RaceFacts(
        race_key=race_facts_key(venue, off),
        # Local date, so it agrees with the local off time in the key. UK/IRE
        # cards never run late enough for UTC->local to roll the day over, so
        # this equals summary["date"] in practice; taking it from ``off`` keeps
        # the two halves of the key from ever drifting apart.
        race_date=off[:10] or str(summary.get("date") or ""),
        off_time=off,
        venue=venue,
        declared_field=len(runners),
        n_runners=n_runners,
        places_paid=_int(race.get("number_of_placed_rides")),
        rule_4_deduction=deduction,
        rule_4_type=kind,
        going=str(summary.get("going") or ""),
        distance=str(summary.get("distance") or ""),
        race_class=str(summary.get("race_class") or ""),
        handicap=bool(summary.get("has_handicap")),
        runners=runners,
        source_path=source_path,
    )


def parse_raw_file(path: str) -> Optional[RaceFacts]:
    """Decompress, extract ``__NEXT_DATA__`` and parse. Never raises."""
    from scraper.betsp.results.sporting_life import _next_data

    try:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            html = fh.read()
    except (OSError, EOFError, UnicodeDecodeError) as exc:
        logger.debug("race_facts: unreadable %s (%s)", path, exc)
        return None
    payload = _next_data(html)
    if payload is None:
        return None
    try:
        return parse_race_document(payload, source_path=path)
    except (TypeError, ValueError, AttributeError) as exc:  # pragma: no cover
        logger.debug("race_facts: unparseable %s (%s)", path, exc)
        return None


# ───────────────────────────────── traversal ────────────────────────────────
def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def day_dir(day: Any, *, root: str = DEFAULT_RAW_ROOT, source: str = DEFAULT_SOURCE) -> str:
    day = _as_date(day)
    return os.path.join(root, source, f"{day.year:04d}", day.isoformat())


def iter_day_paths(
    day: Any, *, root: str = DEFAULT_RAW_ROOT, source: str = DEFAULT_SOURCE
) -> list[str]:
    folder = day_dir(day, root=root, source=source)
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".html.gz")
    )


def extract_day(
    day: Any, *, root: str = DEFAULT_RAW_ROOT, source: str = DEFAULT_SOURCE
) -> list[RaceFacts]:
    """Parse every archived document for one race day."""
    facts = [parse_raw_file(p) for p in iter_day_paths(day, root=root, source=source)]
    return [f for f in facts if f is not None]


def extract_window(
    start: Any,
    end: Any,
    *,
    root: str = DEFAULT_RAW_ROOT,
    source: str = DEFAULT_SOURCE,
    max_workers: int = 8,
) -> list[RaceFacts]:
    """Parse an inclusive date window. Deduplicates on ``race_key``.

    Duplicate documents do occur (the same race re-scraped). The last-written
    file wins, which is the most complete one — an early capture can precede
    the result being published.
    """
    first, last = _as_date(start), _as_date(end)
    if last < first:
        first, last = last, first
    paths: list[str] = []
    day = first
    while day <= last:
        paths.extend(iter_day_paths(day, root=root, source=source))
        day += timedelta(days=1)
    if not paths:
        logger.warning(
            "race_facts: no archived documents for %s..%s under %s",
            first, last, os.path.join(root, source),
        )
        return []

    logger.info("race_facts: parsing %d documents (%s..%s)", len(paths), first, last)
    if max_workers and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            parsed: Iterable[Optional[RaceFacts]] = list(pool.map(parse_raw_file, paths))
    else:
        parsed = [parse_raw_file(p) for p in paths]

    merged: dict[str, RaceFacts] = {}
    for fact in parsed:
        if fact is None:
            continue
        existing = merged.get(fact.race_key)
        # Prefer the document that actually carries results.
        if existing is not None and _completeness(existing) > _completeness(fact):
            continue
        merged[fact.race_key] = fact
    logger.info(
        "race_facts: %d races from %d documents (%d unparseable)",
        len(merged), len(paths), sum(1 for f in parsed if f is None),
    )
    return [merged[k] for k in sorted(merged)]


def _completeness(fact: RaceFacts) -> int:
    """How much settlement truth a document carries — higher wins on duplicates."""
    return sum(1 for r in fact.runners.values() if r.completed)


# ───────────────────────────────── the cache ────────────────────────────────
def to_frame(facts: Iterable[RaceFacts]):
    """One row per declared runner."""
    import pandas as pd

    rows: list[dict] = []
    for fact in facts:
        rows.extend(fact.to_rows())
    if not rows:
        return pd.DataFrame(
            columns=[
                "race_key", "race_date", "off_time", "venue", "declared_field",
                "n_runners", "places_paid", "rule_4_deduction", "rule_4_type",
                "going", "distance", "race_class", "handicap", "horse_key",
                "horse_name", "ride_status", "finish_position", "casualty_reason",
                "industry_sp", "cloth_number", "draw_number", "dead_heat_size",
                "source_path",
            ]
        )
    return pd.DataFrame(rows)


def facts_from_frame(df) -> dict[str, RaceFacts]:
    """Rebuild :class:`RaceFacts` objects from the flat cache."""
    import pandas as pd

    out: dict[str, RaceFacts] = {}
    if df is None or len(df) == 0:
        return out
    for race_key, group in df.groupby("race_key", sort=True):
        head = group.iloc[0]
        runners = {}
        for row in group.itertuples(index=False):
            pos = getattr(row, "finish_position", None)
            runners[row.horse_key] = RunnerFact(
                horse_key=str(row.horse_key),
                horse_name=str(getattr(row, "horse_name", "") or ""),
                ride_status=str(getattr(row, "ride_status", "") or ""),
                finish_position=None if pd.isna(pos) else int(pos),
                casualty_reason=(
                    None
                    if pd.isna(getattr(row, "casualty_reason", None))
                    else str(row.casualty_reason)
                ),
                industry_sp=(
                    None
                    if pd.isna(getattr(row, "industry_sp", None))
                    else float(row.industry_sp)
                ),
                cloth_number=(
                    None
                    if pd.isna(getattr(row, "cloth_number", None))
                    else int(row.cloth_number)
                ),
                draw_number=(
                    None
                    if pd.isna(getattr(row, "draw_number", None))
                    else int(row.draw_number)
                ),
                dead_heat_size=int(getattr(row, "dead_heat_size", 1) or 1),
            )
        places = head.get("places_paid")
        out[str(race_key)] = RaceFacts(
            race_key=str(race_key),
            race_date=str(head.get("race_date") or ""),
            off_time=str(head.get("off_time") or ""),
            venue=str(head.get("venue") or ""),
            declared_field=int(head.get("declared_field") or len(runners)),
            n_runners=int(head.get("n_runners") or 0),
            places_paid=None if pd.isna(places) else int(places),
            rule_4_deduction=float(head.get("rule_4_deduction") or 0.0),
            rule_4_type=str(head.get("rule_4_type") or ""),
            going=str(head.get("going") or ""),
            distance=str(head.get("distance") or ""),
            race_class=str(head.get("race_class") or ""),
            handicap=bool(head.get("handicap")),
            runners=runners,
            source_path=str(head.get("source_path") or ""),
        )
    return out


def build_cache(
    start: Any,
    end: Any,
    *,
    path: str = DEFAULT_CACHE_PATH,
    root: str = DEFAULT_RAW_ROOT,
    source: str = DEFAULT_SOURCE,
    max_workers: int = 8,
):
    """Parse the window and write the parquet cache. Returns the frame."""
    facts = extract_window(start, end, root=root, source=source, max_workers=max_workers)
    frame = to_frame(facts)
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("race_facts: wrote %d runner rows to %s", len(frame), path)
    return frame


def load_race_facts(
    start: Any,
    end: Any,
    *,
    path: str = DEFAULT_CACHE_PATH,
    root: str = DEFAULT_RAW_ROOT,
    source: str = DEFAULT_SOURCE,
    rebuild: bool = False,
    max_workers: int = 8,
):
    """Load the cached frame, rebuilding when it is missing or short of the window."""
    import pandas as pd

    if not rebuild and os.path.exists(path):
        try:
            cached = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            logger.warning("race_facts: cache unreadable (%s) — rebuilding", exc)
        else:
            if len(cached):
                have_lo = str(cached["race_date"].min())
                have_hi = str(cached["race_date"].max())
                want_lo = _as_date(start).isoformat()
                want_hi = _as_date(end).isoformat()
                if have_lo <= want_lo and have_hi >= want_hi:
                    mask = cached["race_date"].between(want_lo, want_hi)
                    return cached.loc[mask].reset_index(drop=True)
                logger.info(
                    "race_facts: cache covers %s..%s, need %s..%s — rebuilding",
                    have_lo, have_hi, want_lo, want_hi,
                )
    return build_cache(
        start, end, path=path, root=root, source=source, max_workers=max_workers
    )


def summarise(facts: Iterable[RaceFacts]) -> dict:
    """Coverage counters for the forward-validation report."""
    facts = list(facts)
    total = len(facts)
    if not total:
        return {"n_races": 0}
    with_nr = sum(1 for f in facts if f.non_runners)
    with_r4 = sum(1 for f in facts if f.rule_4_deduction > 0)
    dead_heats = sum(
        1 for f in facts if any(r.dead_heat for r in f.runners.values())
    )
    runners = [r for f in facts for r in f.runners.values()]
    ran = [r for r in runners if r.ran]
    return {
        "n_races": total,
        "n_declared": len(runners),
        "n_runners": len(ran),
        "n_non_runners": len(runners) - len(ran),
        "pct_races_with_non_runner": round(100.0 * with_nr / total, 3),
        "pct_races_with_rule_4": round(100.0 * with_r4 / total, 3),
        "pct_races_with_dead_heat": round(100.0 * dead_heats / total, 3),
        "pct_runners_no_finish_position": round(
            100.0 * sum(1 for r in ran if not r.completed) / max(len(ran), 1), 3
        ),
        "mean_rule_4_when_applied": round(
            sum(f.rule_4_deduction for f in facts if f.rule_4_deduction > 0)
            / max(with_r4, 1),
            4,
        ),
        "date_range": [
            min(f.race_date for f in facts),
            max(f.race_date for f in facts),
        ],
    }


def _iter_all_days(root: str, source: str) -> Iterator[str]:  # pragma: no cover
    base = os.path.join(root, source)
    if not os.path.isdir(base):
        return
    for year in sorted(os.listdir(base)):
        year_dir = os.path.join(base, year)
        if os.path.isdir(year_dir):
            yield from sorted(os.listdir(year_dir))
