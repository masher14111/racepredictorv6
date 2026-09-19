"""Settlement of a struck ticket: non-runners, Rule 4, dead heats, voids, each-way.

WHY this module exists
----------------------
The Stage-4 audit (``reports/calibration_audit_20260727.md``) found the model's
measured edge to be smaller than the market's margin — 0.186 race log-loss the
wrong way, CLV -13.4%. At that scale the settlement details are not rounding
error: a Tattersalls Rule 4(c) deduction removes up to 75% of a bet's winnings, a
two-horse dead heat halves them, and each-way terms that are re-read at
settlement instead of captured at bet time quietly hand the simulation a price it
never had. A simulator that skips these reports profit the deployment could not
have made.

Every rule here is therefore written to fail *against* the bet:

* Our horse withdrawn => VOID, stake back, no profit — never a guessed result.
* Another horse withdrawn with a recorded withdrawal price => Rule 4 comes off
  the **net winnings** (never the returned stake), summed over withdrawals and
  capped at 0.90. A withdrawal price we cannot read is treated as the shortest
  band (0.75), because the alternative — assuming no deduction — flatters P&L.
* Dead heat => only ``stake / divisor`` was ever on the winner; the rest loses.
* Each-way terms come from :class:`EachWayTerms` captured when the bet was
  struck. If the race ends up paying *fewer* places we honour the smaller
  number, but a race that pays *more* places never improves a settled ticket.
* A runner with no recorded finishing position settles as ``NO_RESULT``. The
  caller leaves it unsettled; this module never invents an outcome.

The price maths goes through :func:`backtest.metrics.settle` so the win/lose and
commission conventions cannot drift from the rest of the repo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from backtest.metrics import settle as _metrics_settle
from execution.config import FrictionConfig
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_VALID_ODDS = 1.0 + 1e-9

# Standard Tattersalls Rule 4(c) bands, keyed on the **withdrawn** horse's
# decimal odds at the moment of withdrawal: ``(upper_bound_inclusive, deduction)``
# in ascending order of price. Anything longer than 11.00 attracts no deduction.
RULE_4_TABLE: tuple[tuple[float, float], ...] = (
    (1.30, 0.75),
    (1.40, 0.70),
    (1.53, 0.65),
    (1.62, 0.60),
    (1.80, 0.55),
    (1.95, 0.50),
    (2.20, 0.45),
    (2.50, 0.40),
    (2.75, 0.35),
    (3.25, 0.30),
    (4.00, 0.25),
    (4.50, 0.20),
    (6.00, 0.15),
    (8.00, 0.10),
    (11.00, 0.05),
)

# Total deduction across multiple withdrawals is capped by the rule itself.
MAX_COMBINED_RULE_4 = 0.90

STATUS_WIN = "WIN"
STATUS_PLACE = "PLACE"
STATUS_LOSE = "LOSE"
STATUS_VOID = "VOID"
STATUS_NO_RESULT = "NO_RESULT"

_WIN_TYPES = ("win",)
_EW_TYPES = ("each_way", "each-way", "eachway", "ew", "e/w")


@dataclass(frozen=True)
class EachWayTerms:
    """The place terms as offered **when the bet was struck**.

    Books change each-way terms as the field firms up (extra places, then fewer
    after non-runners). Re-reading them at settlement would credit the ticket
    with a contract it never had, so the terms are frozen into the ticket here
    and carried through settlement unchanged.
    """

    places: int
    fraction: float

    @classmethod
    def from_quote(cls, quote: Any, *, cfg: Any = None) -> Optional["EachWayTerms"]:
        """Capture terms off a snapshot :class:`~execution.snapshots.Quote`.

        Returns ``None`` when the snapshot does not carry usable terms. We never
        fall back to a house default (1/5, 3 places): an assumed place fraction
        is an assumed price, and the caller must decline the each-way bet rather
        than settle one on invented terms.

        ``cfg`` is accepted so callers can pass an ``ExecutionConfig`` without
        special-casing; no config value may currently supply missing terms.
        """
        places = _as_int(getattr(quote, "ew_places", None))
        fraction = _parse_fraction(getattr(quote, "ew_reduction", None))
        if places is None or places < 1 or fraction is None:
            logger.debug(
                "each-way terms unusable on quote (places=%r reduction=%r) — no EW bet",
                getattr(quote, "ew_places", None),
                getattr(quote, "ew_reduction", None),
            )
            return None
        return cls(places=places, fraction=fraction)

    def place_odds(self, decimal_odds: float) -> float:
        """Place-leg decimal price, mirroring ``utils.bet_tracker._calc_gross_return``."""
        return (float(decimal_odds) - 1.0) * self.fraction + 1.0

    def to_dict(self) -> dict:
        return {"places": int(self.places), "fraction": float(self.fraction)}


@dataclass(frozen=True)
class RaceResult:
    """The settled facts of one race. Absent facts stay absent — never defaulted."""

    race_uid: str
    # horse_key -> finishing position. ``None`` (or an absent key) means "we do
    # not know", which settles as NO_RESULT, not as a loser.
    positions: dict[str, Optional[int]]
    non_runners: frozenset[str] = frozenset()
    # Withdrawn horse -> its decimal odds at withdrawal. This is the Rule 4
    # evidence record: a non-runner absent from here attracts no deduction
    # (it was withdrawn before the bet was struck, or before a market formed).
    withdrawn_odds: dict[str, float] = field(default_factory=dict)
    # The deduction the racecourse actually published, as a fraction of net
    # winnings. When present this *replaces* the Tattersalls table derived from
    # ``withdrawn_odds``: a recorded deduction is evidence, a derived one is an
    # estimate. ``execution.race_facts`` supplies it from the raw archive.
    rule_4_override: Optional[float] = None
    # horse_key -> how many horses shared that finishing position.
    dead_heat_counts: dict[str, int] = field(default_factory=dict)
    # Places the race actually paid, when known. Used only to *reduce* the
    # bet-time terms, never to extend them.
    places_paid: Optional[int] = None
    void_race: bool = False


@dataclass(frozen=True)
class SettlementResult:
    """What the ticket returned, and every deduction that got it there."""

    status: str
    stake: float
    returns: float
    profit: float
    # Gross leg returns *before* ticket-level commission, so the invariant
    # ``returns == win_leg_return + place_leg_return - commission_paid`` holds.
    win_leg_return: float
    place_leg_return: float
    rule_4_deduction: float
    dead_heat_divisor: float
    commission_paid: float
    detail: dict

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "stake": round(float(self.stake), 4),
            "returns": round(float(self.returns), 4),
            "profit": round(float(self.profit), 4),
            "win_leg_return": round(float(self.win_leg_return), 4),
            "place_leg_return": round(float(self.place_leg_return), 4),
            "rule_4_deduction": round(float(self.rule_4_deduction), 4),
            "dead_heat_divisor": round(float(self.dead_heat_divisor), 4),
            "commission_paid": round(float(self.commission_paid), 4),
            "detail": dict(self.detail),
        }


def rule_4_deduction(withdrawn_decimal_odds: Any) -> float:
    """Rule 4(c) deduction for a single withdrawal, as a fraction of net winnings.

    An unreadable or invalid withdrawal price returns the **largest** band
    (0.75). Assuming no deduction would inflate the simulated return of exactly
    the races where the market moved most, which is the direction this project
    must never err in.
    """
    d = _as_odds(withdrawn_decimal_odds)
    if d is None:
        logger.warning(
            "rule 4: unusable withdrawal price %r — applying the maximum band (0.75)",
            withdrawn_decimal_odds,
        )
        return RULE_4_TABLE[0][1]
    for upper, deduction in RULE_4_TABLE:
        if d <= upper:
            return deduction
    return 0.0


def combined_rule_4(withdrawn_odds: Iterable[Any]) -> float:
    """Total deduction across every withdrawal, capped at 0.90 as the rule requires."""
    total = sum(rule_4_deduction(d) for d in withdrawn_odds)
    return min(float(total), MAX_COMBINED_RULE_4)


def settle_ticket(
    *,
    stake: float,
    decimal_odds: float,
    bet_type: str,
    horse_key: str,
    race_result: RaceResult,
    ew_terms: Optional[EachWayTerms] = None,
    commission_rate: float = 0.0,
    cfg: Any = None,
) -> SettlementResult:
    """Settle one struck ticket against ``race_result``.

    ``cfg`` is an :class:`~execution.config.ExecutionConfig` (or a bare
    :class:`~execution.config.FrictionConfig`). ``None`` uses the dataclass
    defaults, which are the same rules ``config.yaml`` ships: Rule 4 on, dead
    heats on, non-runners voided.
    """
    fr = _frictions(cfg)
    stake = float(stake)
    if stake < 0:
        raise ValueError(f"stake must be non-negative, got {stake!r}")
    kind = _bet_kind(bet_type)

    d = _as_odds(decimal_odds)
    if d is None:
        raise ValueError(
            f"decimal_odds={decimal_odds!r} is not an executable price; a ticket "
            "cannot be settled at a price it could not have been struck at"
        )
    if kind == "each_way" and ew_terms is None:
        raise ValueError(
            "each-way settlement requires the EachWayTerms captured at bet time; "
            "refusing to assume house terms"
        )

    stake_win = stake / 2.0 if kind == "each_way" else stake
    stake_place = stake / 2.0 if kind == "each_way" else 0.0

    # ── voids come first: nothing else about the race matters ────────────────
    if race_result.void_race:
        return _void(stake, stake_win, stake_place, kind, "race void/abandoned")
    if horse_key in race_result.non_runners:
        if fr.void_non_runners:
            return _void(stake, stake_win, stake_place, kind, "our runner was withdrawn")
        # config.yaml never enables this; if it is ever forced off, the harsher
        # reading applies — a stake on a horse that did not run is lost.
        return SettlementResult(
            status=STATUS_LOSE,
            stake=stake,
            returns=0.0,
            profit=-stake,
            win_leg_return=0.0,
            place_leg_return=0.0,
            rule_4_deduction=0.0,
            dead_heat_divisor=1.0,
            commission_paid=0.0,
            detail={
                "bet_type": kind,
                "reason": "our runner was withdrawn and void_non_runners is disabled",
                "non_runner": True,
            },
        )

    position = race_result.positions.get(horse_key)
    position = _as_int(position)
    if position is None or position < 1:
        return SettlementResult(
            status=STATUS_NO_RESULT,
            stake=stake,
            returns=0.0,
            profit=0.0,
            win_leg_return=0.0,
            place_leg_return=0.0,
            rule_4_deduction=0.0,
            dead_heat_divisor=1.0,
            commission_paid=0.0,
            detail={
                "bet_type": kind,
                "reason": "no finishing position recorded — ticket left unsettled",
                "race_uid": race_result.race_uid,
                "horse_key": horse_key,
            },
        )

    # ── Rule 4 on the other withdrawals ──────────────────────────────────────
    withdrawals = {
        h: odds for h, odds in race_result.withdrawn_odds.items() if h != horse_key
    }
    r4_source = "none"
    if not fr.rule_4_enabled:
        r4 = 0.0
    elif race_result.rule_4_override is not None:
        # A published deduction beats one derived from withdrawal prices.
        r4 = min(max(float(race_result.rule_4_override), 0.0), MAX_COMBINED_RULE_4)
        r4_source = "published"
    elif withdrawals:
        r4 = combined_rule_4(withdrawals.values())
        r4_source = "derived"
    else:
        r4 = 0.0

    # ── dead heat ────────────────────────────────────────────────────────────
    divisor = 1.0
    if fr.dead_heat_enabled:
        divisor = max(1.0, float(_as_int(race_result.dead_heat_counts.get(horse_key)) or 1))

    # ── which legs won ───────────────────────────────────────────────────────
    win_leg_won = position == 1
    places = None
    place_leg_won = False
    if kind == "each_way":
        places = int(ew_terms.places)
        if race_result.places_paid is not None:
            # Fewer places actually paid binds; more never does.
            places = min(places, max(0, int(race_result.places_paid)))
        place_leg_won = position <= places

    win_gross = _leg_gross(stake_win, d, r4, divisor) if win_leg_won else 0.0
    place_gross = (
        _leg_gross(stake_place, ew_terms.place_odds(d), r4, divisor)
        if place_leg_won
        else 0.0
    )

    gross = win_gross + place_gross
    net = gross - stake
    # Exchange commission is charged on net winnings only, after Rule 4 and the
    # dead-heat division — never on turnover and never on a losing ticket.
    commission = net * float(commission_rate) if net > 0 else 0.0
    returns = gross - commission
    profit = returns - stake

    if win_leg_won:
        status = STATUS_WIN
    elif place_leg_won:
        status = STATUS_PLACE
    else:
        status = STATUS_LOSE

    return SettlementResult(
        status=status,
        stake=stake,
        returns=returns,
        profit=profit,
        win_leg_return=win_gross,
        place_leg_return=place_gross,
        rule_4_deduction=r4,
        dead_heat_divisor=divisor,
        commission_paid=commission,
        detail={
            "bet_type": kind,
            "race_uid": race_result.race_uid,
            "horse_key": horse_key,
            "position": position,
            "decimal_odds": d,
            "win_leg_won": win_leg_won,
            "place_leg_won": place_leg_won,
            "places_used": places,
            "ew_terms": ew_terms.to_dict() if ew_terms is not None else None,
            "withdrawals": {h: _as_odds(o) for h, o in withdrawals.items()},
            "rule_4_enabled": bool(fr.rule_4_enabled),
            "rule_4_source": r4_source,
            "dead_heat_enabled": bool(fr.dead_heat_enabled),
            "commission_rate": float(commission_rate),
        },
    )


# ── internals ────────────────────────────────────────────────────────────────
def _leg_gross(stake_leg: float, decimal_odds: float, r4: float, divisor: float) -> float:
    """Gross return of a winning leg after Rule 4 and dead-heat division.

    Rule 4 shrinks the *winnings* (``d - 1``), never the stake. A dead heat means
    only ``stake / divisor`` was ever on the winner; the remainder is a loser and
    returns nothing. The price maths itself is delegated to
    :func:`backtest.metrics.settle` so it cannot drift from the rest of the repo.
    """
    if stake_leg <= 0:
        return 0.0
    effective_odds = 1.0 + (float(decimal_odds) - 1.0) * (1.0 - float(r4))
    matched = stake_leg / float(divisor)
    if effective_odds <= _MIN_VALID_ODDS:
        # A 90% deduction on a very short price can leave nothing to win; the
        # matched portion still comes back, the dead-heat remainder does not.
        return matched
    winnings = float(_metrics_settle(matched, effective_odds, 1.0, 0.0))
    return matched + winnings


def _void(
    stake: float, stake_win: float, stake_place: float, kind: str, reason: str
) -> SettlementResult:
    return SettlementResult(
        status=STATUS_VOID,
        stake=stake,
        returns=stake,
        profit=0.0,
        win_leg_return=stake_win,
        place_leg_return=stake_place,
        rule_4_deduction=0.0,
        dead_heat_divisor=1.0,
        commission_paid=0.0,
        detail={"bet_type": kind, "reason": reason},
    )


def _frictions(cfg: Any) -> FrictionConfig:
    if cfg is None:
        return FrictionConfig()
    return getattr(cfg, "frictions", cfg)


def _bet_kind(bet_type: Any) -> str:
    kind = str(bet_type or "").strip().lower().replace(" ", "_")
    if kind in _WIN_TYPES:
        return "win"
    if kind in _EW_TYPES:
        return "each_way"
    raise ValueError(f"unsupported bet_type {bet_type!r}; expected 'win' or 'each_way'")


def _as_odds(raw: Any) -> Optional[float]:
    try:
        d = float(raw)
    except (TypeError, ValueError):
        return None
    if not (d > _MIN_VALID_ODDS):
        return None
    return d


def _as_int(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _parse_fraction(raw: Any) -> Optional[float]:
    """Coerce a place-terms reduction to a fraction in ``(0, 1]``.

    Feeds express the same terms three ways: ``"1/5"``, the denominator ``5``,
    or the fraction ``0.2``. Anything else yields ``None`` so the caller declines
    the bet rather than guessing.
    """
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        if "/" in text:
            num, _, den = text.partition("/")
            try:
                value = float(num) / float(den)
            except (TypeError, ValueError, ZeroDivisionError):
                return None
            return value if 0.0 < value <= 1.0 else None
        try:
            raw = float(text)
        except ValueError:
            return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    # A value above 1 is the denominator ("1/5" written as 5).
    if value > 1.0:
        value = 1.0 / value
    return value if 0.0 < value <= 1.0 else None
