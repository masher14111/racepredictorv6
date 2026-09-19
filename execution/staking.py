"""Provisional fractional-Kelly staking under hard exposure ceilings (req 6).

Why the numbers are this small
------------------------------
Stage 4 (``reports/calibration_audit_20260727.md``) issued **MODEL NO-GO**: on an
untouched 904-race window the price-free line *loses* to the de-vigged pre-off
market by 0.186 race log-loss (95% CI [-0.225, -0.146]) and closing-line value is
-13.4%. Kelly sizing is only meaningful when the probability fed to it is honest;
ours is measurably not. So tenth-Kelly here is **not** a Kelly-optimal choice — it
is a damper on a stake size we have no evidence to justify at all, and the real
controls are the exposure ceilings below, which bind irrespective of how large an
edge the model claims.

The rules, in the order they decide
-----------------------------------
1. **No edge, no bet.** A non-positive (or unreadable) Kelly fraction returns 0.0
   regardless of what the ceilings would permit. There is no "small punt anyway"
   path.
2. **Every ceiling is applied and the smallest wins**: per bet
   (``max_stake_pct_bankroll``, further clamped by the bookmaker liability limit
   ``frictions.max_stake_per_bet``), remaining capacity in this race
   (``max_race_exposure_pct``), remaining capacity today
   (``max_daily_exposure_pct``). An exhausted capacity yields exactly ``0.0``,
   never a negative number that could be mistaken for headroom.
3. **Rounding is always down**, then a stake under ``min_stake`` is dropped
   rather than rounded up.

Sizing uses the *current* bankroll, so a shrinking bankroll shrinks stakes
automatically. (:mod:`execution.safeguards` anchors its daily-exposure ceiling to
the day's *starting* bankroll instead, so a losing day cannot re-expand its own
budget.)

Accumulators and same-race (correlated) bets are **structurally unavailable**, not
merely discouraged: :func:`reject_accumulator` and :func:`reject_correlated` raise,
and :meth:`execution.config.ExecutionConfig.from_config` clamps
``allow_accumulators`` / ``allow_correlated_bets`` back to ``False`` even when the
YAML asks for them, so there is no configuration under which a multiple can be
constructed. Multiples multiply an unvalidated edge by itself; correlated legs
break the independence every exposure ceiling here assumes.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_FLOOR, Decimal, InvalidOperation

from backtest import metrics
from utils.logger import get_logger

logger = get_logger(__name__)

# The complete vocabulary of ``StakeDecision.binding_constraint``. Reports and the
# UI may switch on these; nothing else is ever emitted.
BINDING_CONSTRAINTS: tuple[str, ...] = (
    "kelly",  # fractional Kelly itself was the smallest number
    "per_bet_cap",  # max_stake_pct_bankroll (or the book's liability limit)
    "race_cap",  # remaining max_race_exposure_pct capacity in this race
    "daily_cap",  # remaining max_daily_exposure_pct capacity today
    "min_stake",  # a positive stake rounded below min_stake -> dropped
    "no_edge",  # Kelly <= 0 / unreadable price -> never a bet
    "outside_selection_lock_band",  # price outside the frozen, tested odds band
)

# Ticket ``status``/``outcome`` values that mean "this ticket is finished". Anything
# else (including a missing status) counts as OPEN — unknown state must not shrink
# the open-ticket count that the circuit breaker in execution.safeguards reads.
_SETTLED_STATES = frozenset(
    {"settled", "won", "win", "lost", "lose", "void", "voided", "cancelled",
     "canceled", "rejected", "expired", "closed"}
)


# ── small numeric helpers ────────────────────────────────────────────────────
def _finite(value, default: float | None = None) -> float | None:
    """``value`` as a finite float, else ``default`` (NaN and None both fail)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _floor_to_step(value: float, step: float) -> float:
    """Round ``value`` DOWN to a multiple of ``step``.

    Decimal, not ``math.floor(value / step)``: 0.1 is not representable in binary,
    so the float division of an exact multiple lands a hair below the integer and
    a naive floor would silently shave a whole increment off every stake.
    """
    v = _finite(value, 0.0) or 0.0
    if v <= 0:
        return 0.0
    s = _finite(step, 0.0) or 0.0
    if s <= 0:  # rounding disabled
        return v
    try:
        # round() first so a value that is an exact multiple bar float noise
        # (4.999999999999999) does not floor down an entire increment.
        units = (Decimal(str(round(v, 10))) / Decimal(str(s))).to_integral_value(
            rounding=ROUND_FLOOR
        )
        return float(units * Decimal(str(s)))
    except (InvalidOperation, ValueError, ZeroDivisionError):  # pragma: no cover
        return v


def _day_key(value) -> str:
    """Normalise a race date/time to a ``YYYY-MM-DD`` exposure-bucket key."""
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())[:10]
        except (TypeError, ValueError):
            pass
    return str(value).strip()[:10]


def selection_lock_odds_band(cfg) -> Optional[tuple[float, float]]:
    """The frozen selection lock's own tested price band, if it has one.

    ``execution.selection.write_lock`` records the ONE pre-registered strategy
    a chronological train/validation/test selection actually scored on the
    held-out window -- ``data/execution/selection_lock.json``'s
    ``selected.params.odds_min``/``odds_max`` today, e.g.
    ``short_lt_4`` == ``[1.0, 4.0]``. A candidate priced outside that range was
    never part of what was validated, so :meth:`StakePlanner.plan` refuses to
    stake it regardless of edge (Stage 21, B10). Returns ``None`` (no
    restriction) when the lock is absent/unreadable or carries no numeric
    band -- an unlocked/legacy deployment must not be treated as if it had
    passed a band it never recorded.
    """
    from execution.selection import load_lock

    lock_path = getattr(getattr(cfg, "selection", None), "lock_file", None)
    if not lock_path:
        return None
    payload = load_lock(lock_path)
    if not isinstance(payload, Mapping):
        return None
    params = (payload.get("selected") or {}).get("params") if isinstance(payload.get("selected"), Mapping) else None
    if not isinstance(params, Mapping):
        return None
    lo = _finite(params.get("odds_min"))
    hi = _finite(params.get("odds_max"))
    if lo is None or hi is None or lo > hi:
        return None
    return (lo, hi)


# ── ticket shape helpers (shared with execution.safeguards) ──────────────────
def ticket_race_uid(ticket: Mapping) -> str:
    """The race identity of a ticket/candidate mapping, for EXPOSURE grouping.

    Prefers ``race_key`` (Stage 20's venue-qualified, minute-precision
    physical-race identity, ``execution.snapshots.race_key``) over the bare
    ``race_uid`` -- ``race_uid`` drops venue once an off-time parses, so two
    different venues sharing an exact off-time (a real, reproduced event on
    this project's own store, D37/D53) would otherwise share one exposure
    bucket and let the race-exposure ceiling under-count real capacity used.
    ``race_uid`` remains the fallback for a ticket minted before Stage 20 (no
    ``race_key`` column) or one with no venue at all. Falls further back to
    ``models.predictor._race_uid`` (race_time first, race_id last) rather
    than reimplementing that choice — each bookmaker mints its own ``race_id``
    for the same physical race, so a race_id key fragments the exposure map
    and would let the same race be backed twice. Returns ``""`` when the race
    cannot be identified; callers must treat that as "cannot prove this is a
    new race", not as "no race".
    """
    if not isinstance(ticket, Mapping):
        return ""
    rkey = ticket.get("race_key")
    if rkey is not None and str(rkey).strip():
        return str(rkey).strip()
    uid = ticket.get("race_uid")
    if uid is not None and str(uid).strip():
        return str(uid).strip()
    race_id = ticket.get("race_id")
    race_time = ticket.get("race_time")
    if race_id is None and race_time is None:
        return ""
    try:  # imported lazily: models.predictor pulls in the whole scoring stack
        from models.predictor import _race_uid as _predictor_race_uid
    except Exception:  # pragma: no cover - only if the scoring stack is unusable
        logger.debug("staking: models.predictor unavailable for race_uid fallback")
        for candidate in (race_time, race_id):
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
        return ""
    return str(_predictor_race_uid(race_id, race_time) or "")


def ticket_is_open(ticket: Mapping) -> bool:
    """True when a ticket is still live. Unknown state counts as OPEN."""
    if not isinstance(ticket, Mapping):
        return False
    for key in ("settled", "is_settled"):
        if key in ticket and bool(ticket[key]):
            return False
    if ticket.get("settled_at"):
        return False
    for key in ("status", "outcome", "state"):
        value = ticket.get(key)
        if value is not None and str(value).strip().lower() in _SETTLED_STATES:
            return False
    return True


def ticket_stake(ticket: Mapping) -> float:
    """Stake recorded on a ticket, in currency units.

    An absent ``stake`` key is 0.0 (a PASSed candidate never staked). A key that
    is *present but unreadable or negative* raises: exposure computed from corrupt
    money is worse than no exposure figure at all.
    """
    if not isinstance(ticket, Mapping) or "stake" not in ticket:
        return 0.0
    raw = ticket["stake"]
    if raw is None:
        return 0.0
    stake = _finite(raw)
    if stake is None or stake < 0:
        raise ValueError(f"ticket has an unreadable stake: {raw!r}")
    return stake


# ── exposure ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ExposureState:
    """What has already been staked, per race and per day.

    Both maps hold *all* stakes (settled or not) because turnover is turnover;
    ``open_tickets`` counts only unsettled tickets, which is what the issuance
    circuit breaker cares about.
    """

    race_staked: Mapping[str, float] = field(default_factory=dict)
    day_staked: Mapping[str, float] = field(default_factory=dict)
    open_tickets: int = 0

    @classmethod
    def from_tickets(cls, tickets: Iterable[Mapping]) -> "ExposureState":
        race: dict[str, float] = {}
        day: dict[str, float] = {}
        open_count = 0
        for ticket in tickets or ():
            if not isinstance(ticket, Mapping):
                raise ValueError(f"ticket must be a mapping, got {type(ticket)!r}")
            stake = ticket_stake(ticket)
            uid = ticket_race_uid(ticket)
            if uid:
                race[uid] = race.get(uid, 0.0) + stake
            elif stake > 0:
                # A staked ticket we cannot attribute to a race would silently
                # free up race capacity; say so loudly rather than lose it.
                logger.warning(
                    "staking: ticket with stake %.2f has no identifiable race_uid; "
                    "its race exposure cannot be enforced",
                    stake,
                )
            key = _day_key(
                ticket.get("race_date")
                or ticket.get("race_time")
                or ticket.get("placed_at")
            )
            if key:
                day[key] = day.get(key, 0.0) + stake
            if ticket_is_open(ticket):
                open_count += 1
        return cls(race_staked=race, day_staked=day, open_tickets=open_count)

    def staked_in_race(self, race_uid: str) -> float:
        return float(self.race_staked.get(str(race_uid or ""), 0.0) or 0.0)

    def staked_on_day(self, race_date) -> float:
        return float(self.day_staked.get(_day_key(race_date), 0.0) or 0.0)

    def with_stake(self, *, race_uid: str, race_date, stake: float) -> "ExposureState":
        """A new state reflecting one more staked bet, without a round trip
        through :meth:`from_tickets`.

        The daily loop must track exposure across every runner/race it is
        deciding on *within one run*, before any of them exist as a stored
        ticket a fresh ``from_tickets`` read could see — this is what lets a
        second candidate in the same race (or a second race on the same day)
        see the first one's stake already consuming its share of the race/day
        ceiling, using exactly the same day-key convention ``from_tickets``
        uses so the two never drift apart.
        """
        amount = float(stake or 0.0)
        if amount <= 0:
            return self
        race = dict(self.race_staked)
        uid = str(race_uid or "")
        if uid:
            race[uid] = race.get(uid, 0.0) + amount
        day = dict(self.day_staked)
        key = _day_key(race_date)
        if key:
            day[key] = day.get(key, 0.0) + amount
        return ExposureState(race_staked=race, day_staked=day, open_tickets=self.open_tickets + 1)

    def to_dict(self) -> dict:
        return {
            "race_staked": {k: round(float(v), 2) for k, v in self.race_staked.items()},
            "day_staked": {k: round(float(v), 2) for k, v in self.day_staked.items()},
            "open_tickets": int(self.open_tickets),
        }


_EMPTY_EXPOSURE = ExposureState()


# ── the decision ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class StakeDecision:
    """One sizing decision, with the full arithmetic that produced it.

    ``reasons`` records every ceiling that was evaluated and what it permitted, so
    a ticket can show *why* the maximum stake is what it is instead of asserting a
    number. ``stake == 0.0`` is a perfectly normal outcome and the default one.
    """

    stake: float
    max_stake: float
    kelly_stake: float
    full_kelly_stake: float
    binding_constraint: str
    reasons: tuple[str, ...]
    detail: dict

    @property
    def is_bet(self) -> bool:
        return self.stake > 0.0

    def to_dict(self) -> dict:
        return {
            "stake": round(float(self.stake), 2),
            "max_stake": round(float(self.max_stake), 4),
            "kelly_stake": round(float(self.kelly_stake), 4),
            "full_kelly_stake": round(float(self.full_kelly_stake), 4),
            "binding_constraint": self.binding_constraint,
            "reasons": list(self.reasons),
            "detail": dict(self.detail),
        }


class AccumulatorBlocked(RuntimeError):
    """Raised on any attempt to build a multiple. Never caught to "just skip a leg"."""


class CorrelatedBetBlocked(RuntimeError):
    """Raised on a second bet in a race already backed (or on an unidentifiable race)."""


class StakePlanner:
    """Sizes a single back bet against every provisional ceiling at once."""

    def __init__(self, cfg, bankroll: float) -> None:
        self.cfg = cfg
        self.staking = cfg.staking
        self.bankroll = _finite(bankroll, 0.0) or 0.0
        # The book will not lay more than this on one bet; sizing above it just
        # manufactures an unmatched remainder (execution.frictions models the fill).
        liability = _finite(
            getattr(getattr(cfg, "frictions", None), "max_stake_per_bet", None)
        )
        self.book_liability_cap = liability if liability and liability > 0 else None

    # ── capacities ───────────────────────────────────────────────────────────
    def remaining_race_capacity(self, race_uid: str, exposure: ExposureState | None = None) -> float:
        """Unstaked share of this race's ceiling. Never negative."""
        exposure = exposure or _EMPTY_EXPOSURE
        total = max(0.0, self.staking.max_race_exposure_pct) * max(0.0, self.bankroll)
        return max(0.0, total - exposure.staked_in_race(race_uid))

    def remaining_daily_capacity(self, race_date, exposure: ExposureState | None = None) -> float:
        """Unstaked share of today's ceiling. Never negative."""
        exposure = exposure or _EMPTY_EXPOSURE
        total = max(0.0, self.staking.max_daily_exposure_pct) * max(0.0, self.bankroll)
        return max(0.0, total - exposure.staked_on_day(race_date))

    def per_bet_cap(self) -> float:
        """Single-bet ceiling: the bankroll fraction, clamped by book liability."""
        cap = max(0.0, self.staking.max_stake_pct_bankroll) * max(0.0, self.bankroll)
        if self.book_liability_cap is not None:
            cap = min(cap, self.book_liability_cap)
        return max(0.0, cap)

    # ── the plan ─────────────────────────────────────────────────────────────
    def plan(
        self,
        *,
        prob,
        decimal_odds,
        race_uid: str,
        race_date,
        exposure: ExposureState | None = None,
        odds_band: tuple[float, float] | None = None,
    ) -> StakeDecision:
        """``odds_band`` (Stage 21), when supplied, is ``(odds_min, odds_max)``
        from the frozen, bias-corrected selection lock
        (:func:`selection_lock_odds_band`) -- the price range the ONE strategy
        actually tested on the held-out window. A candidate priced outside it
        was never validated at that price and is refused before Kelly or any
        exposure ceiling is even consulted, exactly like ``no_edge``.
        """
        exposure = exposure or _EMPTY_EXPOSURE
        st = self.staking
        bankroll = self.bankroll

        full_kelly = _finite(metrics.kelly_fraction(prob, decimal_odds), 0.0) or 0.0
        full_kelly_stake = max(0.0, full_kelly * max(0.0, bankroll))
        kelly_stake = max(0.0, full_kelly * max(0.0, st.kelly_fraction) * max(0.0, bankroll))

        if odds_band is not None:
            lo, hi = odds_band
            d = _finite(decimal_odds)
            if d is None or d < lo or d > hi:
                reason = (
                    f"binding: outside_selection_lock_band — price "
                    f"{d if d is not None else 'unreadable'} not in the tested "
                    f"[{lo:g}, {hi:g}] band — never staked regardless of edge or "
                    "exposure headroom"
                )
                return StakeDecision(
                    stake=0.0, max_stake=0.0, kelly_stake=kelly_stake,
                    full_kelly_stake=full_kelly_stake,
                    binding_constraint="outside_selection_lock_band",
                    reasons=(reason,),
                    detail={
                        "bankroll": round(bankroll, 2), "prob": _finite(prob),
                        "decimal_odds": d, "race_uid": str(race_uid or ""),
                        "race_date": _day_key(race_date),
                        "odds_band": [lo, hi],
                    },
                )

        cap_bet = self.per_bet_cap()
        race_staked = exposure.staked_in_race(race_uid)
        day_staked = exposure.staked_on_day(race_date)
        cap_race = self.remaining_race_capacity(race_uid, exposure)
        cap_day = self.remaining_daily_capacity(race_date, exposure)

        reasons: list[str] = [
            f"kelly: full {full_kelly:.4f} of bankroll = {full_kelly_stake:.2f}; "
            f"x{st.kelly_fraction:g} fractional = {kelly_stake:.2f}",
            f"per_bet_cap: {st.max_stake_pct_bankroll:.4%} of bankroll {bankroll:.2f}"
            + (
                f", clamped by book liability {self.book_liability_cap:.2f}"
                if self.book_liability_cap is not None
                else ""
            )
            + f" = {cap_bet:.2f}",
            f"race_cap: {st.max_race_exposure_pct:.4%} of bankroll less "
            f"{race_staked:.2f} already staked in race {race_uid or '<unknown>'} "
            f"= {cap_race:.2f} remaining",
            f"daily_cap: {st.max_daily_exposure_pct:.4%} of bankroll less "
            f"{day_staked:.2f} already staked on {_day_key(race_date) or '<unknown>'} "
            f"= {cap_day:.2f} remaining",
        ]

        detail = {
            "bankroll": round(bankroll, 2),
            "prob": _finite(prob),
            "decimal_odds": _finite(decimal_odds),
            "race_uid": str(race_uid or ""),
            "race_date": _day_key(race_date),
            "edge": _finite(metrics.edge(prob, decimal_odds)),
            "expected_value": _finite(metrics.expected_value(prob, decimal_odds)),
            "full_kelly_fraction": round(full_kelly, 6),
            "kelly_fraction_applied": float(st.kelly_fraction),
            "full_kelly_stake": round(full_kelly_stake, 4),
            "kelly_stake": round(kelly_stake, 4),
            "per_bet_cap": round(cap_bet, 4),
            "book_liability_cap": self.book_liability_cap,
            "race_staked": round(race_staked, 2),
            "race_capacity": round(cap_race, 4),
            "day_staked": round(day_staked, 2),
            "daily_capacity": round(cap_day, 4),
            "min_stake": float(st.min_stake),
            "stake_rounding": float(st.stake_rounding),
        }

        # 1. No edge is decided before any ceiling: a non-positive Kelly is never
        #    a bet, however much room the exposure caps happen to leave.
        if full_kelly <= 0.0:
            reasons.append(
                "binding: no_edge — kelly fraction is not positive, so no ceiling is "
                "consulted (a non-positive edge is never staked)"
            )
            return StakeDecision(
                stake=0.0,
                max_stake=0.0,
                kelly_stake=kelly_stake,
                full_kelly_stake=full_kelly_stake,
                binding_constraint="no_edge",
                reasons=tuple(reasons),
                detail=detail,
            )

        if bankroll <= 0.0:
            reasons.append("binding: per_bet_cap — bankroll is not positive, nothing to stake")
            return StakeDecision(
                stake=0.0,
                max_stake=0.0,
                kelly_stake=kelly_stake,
                full_kelly_stake=full_kelly_stake,
                binding_constraint="per_bet_cap",
                reasons=tuple(reasons),
                detail=detail,
            )

        # 2. Every ceiling applies; the smallest wins. Ties resolve to the earlier
        #    (more fundamental) constraint so the label is stable across configs
        #    where, say, the per-bet and per-race percentages are equal.
        candidates = (
            ("kelly", kelly_stake),
            ("per_bet_cap", cap_bet),
            ("race_cap", cap_race),
            ("daily_cap", cap_day),
        )
        max_stake = min(value for _, value in candidates)
        binding = next(name for name, value in candidates if value <= max_stake)
        reasons.append(f"binding: {binding} at {max_stake:.4f} before rounding")

        # 3. Round DOWN, then drop anything under the minimum ticket size.
        stake = _floor_to_step(max_stake, st.stake_rounding)
        if stake > 0:
            reasons.append(
                f"rounded down to {stake:.2f} (increment {st.stake_rounding:g})"
            )
        if max_stake <= 0.0:
            # Capacity was already exhausted; keep the ceiling's own label rather
            # than blaming min_stake for a bet that had no room in the first place.
            stake = 0.0
            reasons.append("no capacity remaining — stake 0.00 (no bet)")
        elif stake < st.min_stake:
            reasons.append(
                f"min_stake: {stake:.2f} is below {st.min_stake:.2f} — dropped, not rounded up"
            )
            stake = 0.0
            binding = "min_stake"

        detail["binding_constraint"] = binding
        detail["max_stake"] = round(max_stake, 4)
        detail["stake"] = round(stake, 2)
        return StakeDecision(
            stake=stake,
            max_stake=max_stake,
            kelly_stake=kelly_stake,
            full_kelly_stake=full_kelly_stake,
            binding_constraint=binding,
            reasons=tuple(reasons),
            detail=detail,
        )


# ── structurally unavailable bet shapes ──────────────────────────────────────
def reject_accumulator(legs: Sequence, *, cfg) -> None:
    """Raise :class:`AccumulatorBlocked` for anything with more than one leg.

    Multiples are unavailable by construction, not merely discouraged: a double
    multiplies an unvalidated edge by itself and its variance by more than that,
    and Stage 4 says the edge is negative to begin with. ``allow_accumulators`` is
    clamped to ``False`` by :meth:`ExecutionConfig.from_config`, so the flag is
    only ever read here to *report* that the config asked and was refused — it can
    never permit a multiple.
    """
    n = len(legs) if legs is not None else 0
    if n <= 1:
        return
    allowed = bool(getattr(getattr(cfg, "staking", None), "allow_accumulators", False))
    raise AccumulatorBlocked(
        f"accumulators are structurally unavailable: {n} legs requested "
        f"(config allow_accumulators={allowed}; multiples are never permitted "
        f"while the deployment is provisional)"
    )


def reject_correlated(new_bet: Mapping, existing: Iterable[Mapping], *, cfg) -> None:
    """Raise :class:`CorrelatedBetBlocked` for a second bet in the same race.

    Two runners in one race are mutually exclusive, so same-race legs break the
    independence that every exposure ceiling in this module assumes. A bet whose
    race cannot be identified also raises: we cannot prove it is *not* correlated,
    and unprovable means blocked.
    """
    if bool(getattr(getattr(cfg, "staking", None), "allow_correlated_bets", False)):
        # Unreachable through ExecutionConfig (it clamps the flag to False), but a
        # hand-built config object must not silently disable the check.
        logger.warning("staking: allow_correlated_bets is set; same-race check still enforced")
    uid = ticket_race_uid(new_bet) if isinstance(new_bet, Mapping) else ""
    if not uid:
        raise CorrelatedBetBlocked(
            "cannot identify the race of the new bet (no race_uid/race_time/race_id); "
            "correlation with an existing bet cannot be ruled out"
        )
    for other in existing or ():
        if not isinstance(other, Mapping):
            continue
        if ticket_race_uid(other) == uid:
            raise CorrelatedBetBlocked(
                f"race {uid} already carries a bet; same-race (correlated) bets are "
                "never permitted — at most one runner per race"
            )
