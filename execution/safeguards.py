"""Bankroll and candidate-issuance safeguards (requirement 9).

Stage 4 (``reports/calibration_audit_20260727.md``) issued **MODEL NO-GO** — the
price-free line loses to the de-vigged pre-off market by 0.186 race log-loss and
CLV is -13.4%. A model that is behind the market does not fail gracefully: it
fails by issuing *more* tickets as its probabilities drift. Everything here is the
circuit breaker for that failure mode, so every check is written to answer
"block" when it cannot affirmatively prove "allow":

* a missing race time blocks (an unknown off-time is not a future off-time),
* a source health lookup that raises blocks (an unreadable health file is not a
  healthy one),
* a candidate whose race or runner cannot be identified blocks (an unidentifiable
  bet cannot be shown not to be a duplicate of one already open),
* an unsettled ticket of unknown status counts as open.

:meth:`Safeguards.check` deliberately evaluates *every* rule and reports all the
codes that fired, rather than short-circuiting on the first. A report that says
"blocked: race_started" when four things were wrong teaches the wrong lesson.

``paper_only`` is not one of the block codes. Paper tickets are the entire point
of the deployment, so the flag rides along in ``detail['paper_only']`` as a
disclosure. The only thing that raises is
:meth:`Safeguards.assert_paper_only`, which fires when something tries to treat
this output as real-money-eligible while
:func:`execution.config.real_money_enabled` says it is not.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone

from execution.config import real_money_enabled
from execution.staking import ticket_is_open, ticket_race_uid
from utils.logger import get_logger

logger = get_logger(__name__)

# The complete block vocabulary. Reports and the UI switch on these strings, so
# they are part of the module's contract — add to the tuple, never rename.
BLOCK_CODES: tuple[str, ...] = (
    "bankroll_stop_loss",
    "daily_loss_limit",
    "daily_exposure_limit",
    "race_started",
    "duplicate_ticket",
    "correlated_bet",
    "stale_source",
    "unhealthy_source",
    "max_open_tickets",
)

BLOCK_DESCRIPTIONS: dict[str, str] = {
    "bankroll_stop_loss": "bankroll has fallen to or below its stop-loss floor",
    "daily_loss_limit": "today's loss has reached the daily limit",
    "daily_exposure_limit": "this stake would exceed today's exposure ceiling",
    "race_started": "the race is at, past, or inside the buffer before its off time",
    "duplicate_ticket": "an unsettled ticket already covers this race/runner/bet type",
    "correlated_bet": "an unsettled ticket already covers another runner in this race",
    "stale_source": "a contributing price source has not succeeded recently enough",
    "unhealthy_source": "a contributing price source is unhealthy, unknown or unreadable",
    "max_open_tickets": "the open-ticket circuit breaker has tripped",
}

# The code raised when real-money eligibility is claimed without the evidence. It
# is never a `check()` block — see the module docstring.
PAPER_ONLY_CODE = "paper_only_override"

_CANDIDATE_SOURCE_KEYS = (
    "sources",
    "price_sources",
    "contributing_sources",
    "source",
    "bookmaker",
    "book",
    "reference_source",
)


class SafeguardViolation(RuntimeError):
    """A safeguard that was violated by an *action*, not merely by a candidate."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SafeguardState:
    """Bankroll/ledger facts the safeguards judge a candidate against.

    ``initial_bankroll`` is the anchor for both loss limits and the daily exposure
    ceiling: anchoring to the *current* bankroll would let a losing day quietly
    re-expand its own budget as it shrank.
    """

    bankroll: float
    initial_bankroll: float
    day_profit: float = 0.0
    day_staked: float = 0.0
    open_tickets: int = 0
    trading_day: str | None = None

    def to_dict(self) -> dict:
        return {
            "bankroll": _round(self.bankroll),
            "initial_bankroll": _round(self.initial_bankroll),
            "day_profit": _round(self.day_profit),
            "day_staked": _round(self.day_staked),
            "open_tickets": int(self.open_tickets),
            "trading_day": self.trading_day,
        }


@dataclass(frozen=True)
class SafeguardCheck:
    """The verdict for one candidate. ``allowed`` is only ever True when empty."""

    allowed: bool
    blocks: tuple[str, ...] = ()
    messages: tuple[str, ...] = ()
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "allowed": bool(self.allowed),
            "blocks": list(self.blocks),
            "messages": list(self.messages),
            "detail": dict(self.detail),
        }


class Safeguards:
    """Issue-time circuit breakers around a single candidate ticket."""

    def __init__(self, cfg, *, source_health_fn=None, now_fn=None) -> None:
        self.cfg = cfg
        self.safeguards = cfg.safeguards
        self.staking = cfg.staking
        self._source_health_fn = source_health_fn
        self._now_fn = now_fn or (lambda: datetime.now(tz=timezone.utc))
        for flag, why in (
            ("block_started_races", "bets could be issued into a race already run"),
            ("block_duplicates", "the same runner could be backed twice"),
            ("block_stale_sources", "prices from a dead scraper would be treated as live"),
        ):
            if not bool(getattr(self.safeguards, flag, True)):
                logger.warning("safeguards: %s is disabled in config — %s", flag, why)

    # ── health lookup ────────────────────────────────────────────────────────
    def _health(self, source: str) -> Mapping | None:
        """Health record for ``source``; ``None`` means "could not be read"."""
        fn = self._source_health_fn
        if fn is None:
            # Lazy import so a caller that injects its own lookup never touches
            # the on-disk telemetry file (and tests never do).
            from utils.source_health import get_health

            fn = get_health
        try:
            record = fn(source)
        except Exception as exc:
            logger.warning("safeguards: source health lookup failed for %s: %s", source, exc)
            return None
        return record if isinstance(record, Mapping) else None

    # ── the check ────────────────────────────────────────────────────────────
    def check(
        self,
        candidate: Mapping,
        state: SafeguardState,
        *,
        existing_tickets: Iterable[Mapping] = (),
        now: datetime | None = None,
    ) -> SafeguardCheck:
        cfg_s = self.safeguards
        now = now or self._now_fn()
        candidate = candidate if isinstance(candidate, Mapping) else {}
        tickets = [t for t in (existing_tickets or ()) if isinstance(t, Mapping)]
        open_tickets = [t for t in tickets if ticket_is_open(t)]

        blocks: list[str] = []
        messages: list[str] = []
        detail: dict = {
            "checked_at": _iso(now),
            "state": state.to_dict(),
            "codes_evaluated": list(BLOCK_CODES),
            # Disclosure, never a block: every ticket this module clears is paper.
            "paper_only": {
                "code": PAPER_ONLY_CODE,
                "paper_only": self.paper_only(),
                "note": (
                    "output is PAPER-ONLY; real-money eligibility additionally "
                    "requires a model GO and a passing forward gate"
                ),
            },
        }

        def block(code: str, message: str) -> None:
            if code not in blocks:
                blocks.append(code)
                messages.append(message)

        initial = _num(state.initial_bankroll)
        bankroll = _num(state.bankroll)

        # 1. bankroll stop-loss — an unreadable bankroll is not a safe bankroll.
        if initial is None or initial <= 0 or bankroll is None:
            block(
                "bankroll_stop_loss",
                f"bankroll state is unusable (bankroll={state.bankroll!r}, "
                f"initial={state.initial_bankroll!r}); issuance is halted.",
            )
            detail["bankroll_floor"] = None
        else:
            floor = initial * (1.0 - _num(cfg_s.bankroll_stop_loss_pct, 0.0))
            detail["bankroll_floor"] = _round(floor)
            if bankroll <= floor:
                block(
                    "bankroll_stop_loss",
                    f"Bankroll {bankroll:.2f} is at or below the stop-loss floor "
                    f"{floor:.2f} ({cfg_s.bankroll_stop_loss_pct:.1%} below the "
                    f"{initial:.2f} starting bankroll); all issuance is halted.",
                )

        # 2. daily loss limit
        day_profit = _num(state.day_profit, 0.0)
        if initial is not None and initial > 0:
            loss_floor = -initial * _num(cfg_s.daily_loss_limit_pct, 0.0)
            detail["daily_loss_floor"] = _round(loss_floor)
            if day_profit <= loss_floor:
                block(
                    "daily_loss_limit",
                    f"Today's P&L {day_profit:.2f} has reached the daily loss limit "
                    f"{loss_floor:.2f} ({cfg_s.daily_loss_limit_pct:.1%} of the "
                    f"{initial:.2f} starting bankroll); no further tickets today.",
                )

        # 3. daily exposure ceiling. An absent stake means "not sized yet" and adds
        #    nothing; a stake that is present but unreadable blocks outright.
        stake = 0.0
        stake_readable = True
        if "stake" in candidate and candidate.get("stake") is not None:
            parsed = _num(candidate.get("stake"))
            if parsed is None or parsed < 0:
                stake_readable = False
            else:
                stake = parsed
        detail["candidate_stake"] = stake if stake_readable else None
        if not stake_readable:
            block(
                "daily_exposure_limit",
                f"Candidate stake {candidate.get('stake')!r} is unreadable, so its "
                "contribution to today's exposure cannot be checked.",
            )
        elif initial is not None and initial > 0:
            cap = initial * _num(self.staking.max_daily_exposure_pct, 0.0)
            day_staked = _num(state.day_staked, 0.0)
            detail["daily_exposure_cap"] = _round(cap)
            detail["daily_exposure_used"] = _round(day_staked)
            if day_staked + stake > cap:
                block(
                    "daily_exposure_limit",
                    f"Staking {stake:.2f} on top of {day_staked:.2f} already staked "
                    f"today would exceed the {cap:.2f} daily exposure ceiling "
                    f"({self.staking.max_daily_exposure_pct:.2%} of the starting bankroll).",
                )

        # 4. started race — a missing or unparsable off-time blocks.
        buffer_s = _num(cfg_s.started_race_buffer_seconds, 0.0)
        race_time = _to_utc(candidate.get("race_time"))
        seconds_to_off = (
            (race_time - _as_utc(now)).total_seconds() if race_time is not None else None
        )
        detail["seconds_to_off"] = (
            round(seconds_to_off, 1) if seconds_to_off is not None else None
        )
        detail["started_race_buffer_seconds"] = buffer_s
        if bool(getattr(cfg_s, "block_started_races", True)):
            if seconds_to_off is None:
                block(
                    "race_started",
                    f"Race time {candidate.get('race_time')!r} is missing or unreadable; "
                    "an unknown off-time cannot be shown to be in the future.",
                )
            elif seconds_to_off <= buffer_s:
                block(
                    "race_started",
                    f"Race is {seconds_to_off:.0f}s away, inside the {buffer_s:.0f}s "
                    "pre-off buffer; the race counts as started.",
                )

        # 5/6. duplicate and correlated tickets, judged against UNSETTLED tickets.
        race_uid = ticket_race_uid(candidate)
        horse_key = _horse_key(candidate)
        bet_type = _bet_type(candidate)
        detail["race_uid"] = race_uid
        detail["horse_key"] = horse_key
        detail["bet_type"] = bet_type
        detail["open_tickets_supplied"] = len(open_tickets)

        if bool(getattr(cfg_s, "block_duplicates", True)):
            if not race_uid or not horse_key:
                block(
                    "duplicate_ticket",
                    "Candidate cannot be identified (race_uid="
                    f"{race_uid or 'missing'}, horse={horse_key or 'missing'}), so it "
                    "cannot be shown not to duplicate an open ticket.",
                )
            else:
                for other in open_tickets:
                    if ticket_race_uid(other) != race_uid:
                        continue
                    if _horse_key(other) != horse_key:
                        continue
                    other_type = _bet_type(other)
                    # A missing bet type on either side matches everything: the
                    # stricter reading is the safe one.
                    if bet_type and other_type and other_type != bet_type:
                        continue
                    block(
                        "duplicate_ticket",
                        f"An unsettled {other_type or 'unspecified'} ticket already "
                        f"covers {horse_key} in race {race_uid}.",
                    )
                    break

        if not bool(getattr(self.staking, "allow_correlated_bets", False)):
            if not race_uid:
                block(
                    "correlated_bet",
                    "Candidate has no identifiable race, so correlation with an open "
                    "ticket in the same race cannot be ruled out.",
                )
            else:
                for other in open_tickets:
                    if ticket_race_uid(other) == race_uid:
                        block(
                            "correlated_bet",
                            f"Race {race_uid} already carries an unsettled ticket "
                            f"({_horse_key(other) or 'unnamed runner'}); at most one "
                            "runner per race.",
                        )
                        break

        # 7/8. contributing source health.
        sources = _sources(candidate)
        detail["sources"] = sources
        source_detail: dict[str, dict] = {}
        detail["source_health"] = source_detail
        if bool(getattr(cfg_s, "block_stale_sources", True)):
            max_age = _num(cfg_s.max_source_age_seconds, 0.0)
            detail["max_source_age_seconds"] = max_age
            if not sources:
                block(
                    "unhealthy_source",
                    "No contributing price source is recorded on the candidate, so "
                    "source health cannot be verified.",
                )
            for source in sources:
                record = self._health(source)
                if record is None or not _source_ok(record):
                    why = (
                        "its health record could not be read"
                        if record is None
                        else f"status={record.get('status')!r}"
                    )
                    source_detail[source] = {
                        "ok": False,
                        "age_seconds": None if record is None else _num(record.get("age_seconds")),
                        "status": None if record is None else record.get("status"),
                    }
                    block(
                        "unhealthy_source",
                        f"Source '{source}' is not healthy ({why}).",
                    )
                    continue
                age = _num(record.get("age_seconds"))
                source_detail[source] = {
                    "ok": True,
                    "age_seconds": None if age is None else round(age, 1),
                    "status": record.get("status"),
                }
                if age is None:
                    block(
                        "stale_source",
                        f"Source '{source}' reports no age for its last success; "
                        "freshness cannot be proven.",
                    )
                elif age > max_age:
                    block(
                        "stale_source",
                        f"Source '{source}' last succeeded {age:.0f}s ago, beyond the "
                        f"{max_age:.0f}s freshness limit.",
                    )

        # 9. open-ticket circuit breaker. Trust whichever count is larger.
        limit = int(_num(cfg_s.max_open_tickets, 0) or 0)
        count = max(int(_num(state.open_tickets, 0) or 0), len(open_tickets))
        detail["open_tickets"] = count
        detail["max_open_tickets"] = limit
        if count >= limit:
            block(
                "max_open_tickets",
                f"{count} tickets are already open, at or above the {limit} ceiling; "
                "issuance is paused until some settle.",
            )

        allowed = not blocks
        detail["allowed"] = allowed
        return SafeguardCheck(
            allowed=allowed,
            blocks=tuple(blocks),
            messages=tuple(messages),
            detail=detail,
        )

    # ── paper-only ───────────────────────────────────────────────────────────
    def paper_only(self) -> bool:
        """The permanent paper-only override from config (default/fail-closed True)."""
        return bool(getattr(self.cfg, "paper_only", True))

    def assert_paper_only(
        self, *, model_go: bool = False, forward_gate_passed: bool = False
    ) -> None:
        """Guard the boundary where output would stop being paper.

        Call this from anything that is about to present a ticket as real-money
        eligible. It raises unless :func:`execution.config.real_money_enabled`
        affirmatively permits it — which, under Stage 4's MODEL NO-GO, it does not.
        """
        if real_money_enabled(
            self.cfg, model_go=model_go, forward_gate_passed=forward_gate_passed
        ):
            return
        raise SafeguardViolation(
            PAPER_ONLY_CODE,
            "output is PAPER-ONLY and may not be treated as real-money eligible "
            f"(paper_only={self.paper_only()}, model_go={bool(model_go)}, "
            f"forward_gate_passed={bool(forward_gate_passed)}); real money requires "
            "paper_only:false AND a model GO AND a passing forward gate.",
        )


# ── helpers ──────────────────────────────────────────────────────────────────
def _num(value, default: float | None = None) -> float | None:
    """Finite float or ``default``. NaN and infinity are *not* numbers here."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _round(value, nd: int = 2):
    out = _num(value)
    return round(out, nd) if out is not None else None


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if isinstance(value, datetime) else None


def _as_utc(value: datetime) -> datetime:
    """A datetime in UTC; naive input is read as Europe/Dublin local time."""
    if value.tzinfo is None:
        from utils.timezone import to_utc

        return to_utc(value)
    return value.astimezone(timezone.utc)


def _to_utc(value) -> datetime | None:
    """Parse a race time to UTC. ``None`` for anything unreadable — never 'now'."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return _as_utc(value)
    text = str(value).strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return _as_utc(datetime.fromisoformat(text))
    except (TypeError, ValueError):
        return None


def _horse_key(record: Mapping) -> str:
    for key in ("horse_key", "horse_id", "horse", "runner", "selection"):
        value = record.get(key) if isinstance(record, Mapping) else None
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _bet_type(record: Mapping) -> str:
    value = record.get("bet_type") if isinstance(record, Mapping) else None
    return str(value).strip().lower() if value is not None and str(value).strip() else ""


def _sources(candidate: Mapping) -> list[str]:
    """Every source that contributed to the candidate, de-duplicated, order kept."""
    out: list[str] = []
    for key in _CANDIDATE_SOURCE_KEYS:
        value = candidate.get(key)
        if value is None:
            continue
        items = value if isinstance(value, (list, tuple, set, frozenset)) else [value]
        for item in items:
            name = str(item).strip().lower()
            if name and name not in out:
                out.append(name)
    return out


def _source_ok(record: Mapping) -> bool:
    """Healthy only on affirmative evidence: unknown or empty is not healthy."""
    if not isinstance(record, Mapping) or not record:
        return False
    if "ok" in record and not bool(record.get("ok")):
        return False
    status = record.get("status")
    if status is not None:
        return str(status).strip().lower() == "ok"
    return bool(record.get("ok"))
