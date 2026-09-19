"""Execution frictions: what price we could *actually* have had, and when.

WHY this module exists
----------------------
Stage 4 (``reports/calibration_audit_20260727.md``) issued **MODEL NO-GO**: the
independent price-free line loses to the de-vigged pre-off market by 0.186 race
log-loss (95% CI [-0.225, -0.146]) with CLV at -13.4%. A backtest that fills
every candidate instantly, at the price the model saw, is therefore not merely
optimistic — it is measuring a market that never existed, and it would hide the
negative CLV that is the whole finding.

Four frictions are modelled here, each of which can only ever make a simulated
result *worse* than the naive one:

1. **Latency.** A candidate emitted at ``t`` cannot be struck until
   ``t + decision_latency + placement_latency``. The executable price is the
   newest snapshot at or before that later moment.
2. **No-fill, not last-known-price.** If no snapshot exists in the window the bet
   simply does not happen (``NO_QUOTE``). Reaching back to the last price we
   happen to have — or forward to the closing price — is the exact look-ahead the
   Stage-4 audit was written to stop, so :func:`attempt_fill` raises
   ``FuturePriceError`` if a store ever hands it a quote from after the fill.
3. **Rejections and suspensions.** Books bounce requests and markets suspend.
   Both are drawn *deterministically* from a hash of the bet's identity, so a
   re-run — or the same bets evaluated in a different order — reproduces exactly
   the same fills. An order-dependent RNG would let a strategy's result move when
   an unrelated strategy is added to the grid.
4. **Bookmaker liability limits, exchange commission and BOG.** Stake above the
   book's limit is unmatched, not matched at a worse price. Commission is charged
   on exchange sources only. Best-odds-guaranteed is *never* assumed: it needs
   both the config flag and per-bet recorded evidence.

Nothing in this module may improve a price. ``adverse_move_only`` exists to make
that explicit: when set, a price that drifted in our favour is ignored and the
originally quoted price is used, while a price that drifted against us is always
taken at the worse available number.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

from utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from execution.snapshots import Quote, SnapshotStore

try:  # pragma: no cover - execution.snapshots is written alongside this module
    from execution.snapshots import FuturePriceError
except ImportError:  # pragma: no cover - fallback keeps the guard usable standalone
    class FuturePriceError(RuntimeError):
        """A quote dated after the moment being simulated was read.

        Mirrors ``execution.snapshots.FuturePriceError``; defined locally only so
        the look-ahead guard still fires if the snapshot store is unavailable.
        """

logger = get_logger(__name__)

# Mirrors ``backtest.metrics``: the feeds emit exactly 1.0 for an absent price,
# so anything at or below 1 is "no executable price", not "a price of evens-on".
_MIN_VALID_ODDS = 1.0 + 1e-9

# Field separator for the deterministic hash. Chosen because no bookmaker,
# race_uid or horse_key can contain it, so keys cannot alias each other.
_KEY_SEP = "\x1f"

STATUS_FILLED = "FILLED"
STATUS_PARTIAL = "PARTIAL"
STATUS_REJECTED = "REJECTED"
STATUS_SUSPENDED = "SUSPENDED"
STATUS_NO_QUOTE = "NO_QUOTE"
STATUS_STALE_QUOTE = "STALE_QUOTE"


@dataclass(frozen=True)
class FillRequest:
    """A bet the model wants to strike, as of the moment it was emitted."""

    race_uid: str
    horse_key: str
    stake: float
    # When the model emitted the candidate — NOT when it is struck. The fill
    # happens ``total_latency_seconds`` later.
    requested_at: datetime
    # The price the model saw at ``requested_at``. Never assumed obtainable.
    quoted_odds: float
    bookmaker: Optional[str] = None
    market_type: str = "WIN"
    bet_type: str = "win"
    # Only ever True when the book's best-odds-guaranteed promise is evidenced
    # for this individual bet. A blanket assumption of BOG is free money that
    # does not exist.
    bog_recorded: bool = False


@dataclass(frozen=True)
class FillResult:
    """The outcome of trying to strike :class:`FillRequest` after latency."""

    filled: bool
    status: str
    reason: str
    matched_stake: float
    unmatched_stake: float
    fill_odds: Optional[float]
    fill_at: Optional[datetime]
    quote_fetched_at: Optional[datetime]
    quote_age_seconds: Optional[float]
    price_move: Optional[float]
    price_move_pct: Optional[float]
    bookmaker: Optional[str]
    commission_rate: float
    bog_applied: bool

    def to_dict(self) -> dict:
        return {
            "filled": bool(self.filled),
            "status": self.status,
            "reason": self.reason,
            "matched_stake": round(float(self.matched_stake), 4),
            "unmatched_stake": round(float(self.unmatched_stake), 4),
            "fill_odds": _round_opt(self.fill_odds, 4),
            "fill_at": _iso(self.fill_at),
            "quote_fetched_at": _iso(self.quote_fetched_at),
            "quote_age_seconds": _round_opt(self.quote_age_seconds, 3),
            "price_move": _round_opt(self.price_move, 4),
            "price_move_pct": _round_opt(self.price_move_pct, 6),
            "bookmaker": self.bookmaker,
            "commission_rate": round(float(self.commission_rate), 6),
            "bog_applied": bool(self.bog_applied),
        }


def deterministic_uniform(seed: int, *keys: Any) -> float:
    """A reproducible uniform draw in ``[0, 1)`` keyed on ``seed`` and ``keys``.

    Deliberately stateless: the draw for a bet depends only on that bet's
    identity, never on how many draws preceded it. That is what makes rejection
    and suspension order-independent, so adding a strategy to a grid cannot
    silently change the fills of the strategies already in it.

    ``None`` keys stringify to the empty string, so callers should normalise
    optional keys (e.g. lower-case a bookmaker) before passing them.
    """
    payload = _KEY_SEP.join("" if k is None else str(k) for k in keys)
    material = f"{int(seed)}{_KEY_SEP}{payload}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 18446744073709551616.0  # 2**64


def attempt_fill(
    req: FillRequest,
    *,
    store: "SnapshotStore",
    cfg,
    seed: Optional[int] = None,
) -> FillResult:
    """Try to strike ``req`` against the snapshot store, with all frictions applied.

    ``store`` only needs the ``latest_quote(race_uid, horse_key, as_of, *,
    bookmaker=None, market_type='WIN', max_age_seconds=None)`` half of the
    :class:`~execution.snapshots.SnapshotStore` contract. ``cfg`` is an
    :class:`~execution.config.ExecutionConfig`.

    Every failure mode returns a no-fill result rather than a degraded fill: this
    function has no code path that invents a price.
    """
    fr = cfg.frictions
    if seed is None:
        seed = cfg.selection.seed

    fill_at = req.requested_at + timedelta(seconds=fr.total_latency_seconds)

    quote = store.latest_quote(
        req.race_uid,
        req.horse_key,
        fill_at,
        bookmaker=req.bookmaker,
        market_type=req.market_type,
        # Ask for the raw newest quote and decide staleness here. The store's own
        # limit is the snapshot-level one (``snapshots.max_age_seconds``, 900s)
        # and omitting this argument would apply it, collapsing "too old to bet
        # on" into "no quote at all" and making ``allow_stale_fill`` unreachable.
        # The execution limit is the tighter, separate
        # ``frictions.max_price_age_seconds``.
        max_age_seconds=float("inf"),
    )
    if quote is None:
        return _no_fill(
            req,
            STATUS_NO_QUOTE,
            "no snapshot at or before the fill time — no bet, not a fill at the "
            "last known price",
            fill_at,
        )

    _assert_not_future(quote, fill_at)
    age = _age_seconds(quote, fill_at)
    resolved_book = getattr(quote, "bookmaker", None) or req.bookmaker
    book_key = _normalise_book(resolved_book)

    if age > fr.max_price_age_seconds and not fr.allow_stale_fill:
        return _no_fill(
            req,
            STATUS_STALE_QUOTE,
            f"quote is {age:.0f}s old at fill time (limit {fr.max_price_age_seconds:.0f}s)",
            fill_at,
            quote=quote,
            age=age,
            bookmaker=resolved_book,
        )

    available = _valid_odds(getattr(quote, "odds_decimal", None))
    if available is None:
        return _no_fill(
            req,
            STATUS_NO_QUOTE,
            "snapshot carries no executable decimal price",
            fill_at,
            quote=quote,
            age=age,
            bookmaker=resolved_book,
        )

    # Suspension first: a suspended market is gone, so the book never gets to
    # reject the request at all.
    if deterministic_uniform(seed, req.race_uid, req.horse_key, book_key, "suspend") < fr.suspension_rate:
        return _no_fill(
            req, STATUS_SUSPENDED, "market suspended between decision and placement",
            fill_at, quote=quote, age=age, bookmaker=resolved_book,
        )
    if deterministic_uniform(seed, req.race_uid, req.horse_key, book_key, "reject") < fr.rejection_rate:
        return _no_fill(
            req, STATUS_REJECTED, "bookmaker rejected the request (price changed/unavailable)",
            fill_at, quote=quote, age=age, bookmaker=resolved_book,
        )

    stake = float(req.stake)
    if not (stake > 0):
        return _no_fill(
            req, STATUS_REJECTED, "requested stake is not positive",
            fill_at, quote=quote, age=age, bookmaker=resolved_book,
        )

    limit = float(fr.max_stake_per_bet)
    matched = min(stake, limit) if limit > 0 else 0.0
    if matched <= 0:
        return _no_fill(
            req,
            STATUS_REJECTED,
            f"bookmaker liability limit ({limit:g}) leaves the request fully unmatched",
            fill_at,
            quote=quote,
            age=age,
            bookmaker=resolved_book,
        )
    unmatched = max(0.0, stake - matched)

    quoted = _valid_odds(req.quoted_odds)
    fill_odds = available
    if fr.adverse_move_only and quoted is not None and available > quoted:
        # The price drifted in our favour. Assume we did not get the benefit;
        # an adverse drift is always taken at the worse number below.
        fill_odds = quoted

    price_move = (fill_odds - quoted) if quoted is not None else None
    price_move_pct = (price_move / quoted) if (price_move is not None and quoted) else None

    # BOG is recorded, never *applied* to the price here: the SP that would make
    # it pay is not known at fill time, and reading it would be look-ahead.
    bog_applied = bool(req.bog_recorded and fr.best_odds_guaranteed)

    status = STATUS_PARTIAL if unmatched > 0 else STATUS_FILLED
    reason = (
        f"matched {matched:g} of {stake:g} at the book's per-bet limit"
        if unmatched > 0
        else "matched in full at the price available after latency"
    )
    return FillResult(
        filled=True,
        status=status,
        reason=reason,
        matched_stake=matched,
        unmatched_stake=unmatched,
        fill_odds=fill_odds,
        fill_at=fill_at,
        quote_fetched_at=getattr(quote, "fetched_at", None),
        quote_age_seconds=age,
        price_move=price_move,
        price_move_pct=price_move_pct,
        bookmaker=resolved_book,
        commission_rate=float(fr.commission_for(resolved_book)),
        bog_applied=bog_applied,
    )


# ── internals ────────────────────────────────────────────────────────────────
def _assert_not_future(quote: "Quote", fill_at: datetime) -> None:
    """Hard guard against reading a price from after the moment being simulated.

    This is the single defect that would make every downstream number
    meaningless, so it raises rather than warns.
    """
    fetched_at = getattr(quote, "fetched_at", None)
    if fetched_at is None:
        raise FuturePriceError("snapshot has no fetched_at; cannot prove it predates the fill")
    try:
        ahead = fetched_at > fill_at
    except TypeError as exc:  # naive vs aware datetimes
        raise ValueError(
            "quote.fetched_at and FillRequest.requested_at must both be "
            "timezone-aware or both naive"
        ) from exc
    if ahead:
        raise FuturePriceError(
            f"snapshot fetched_at={fetched_at.isoformat()} is after fill_at="
            f"{fill_at.isoformat()} — a future price can never be the executable price"
        )
    as_of = getattr(quote, "as_of", None)
    if as_of is not None and as_of > fill_at:
        raise FuturePriceError(
            f"snapshot as_of={as_of.isoformat()} is after fill_at={fill_at.isoformat()}"
        )


def _age_seconds(quote: "Quote", fill_at: datetime) -> float:
    """Seconds between the snapshot and the fill, taking the *older* reading.

    We recompute from ``fetched_at`` rather than trusting ``quote.age_seconds``,
    which the store may have derived against a different ``as_of``; where the two
    disagree the larger (more likely stale) value wins.
    """
    ours = (fill_at - quote.fetched_at).total_seconds()
    theirs = getattr(quote, "age_seconds", None)
    try:
        theirs = float(theirs) if theirs is not None else None
    except (TypeError, ValueError):
        theirs = None
    if theirs is not None and theirs == theirs:  # not NaN
        return max(ours, theirs)
    return ours


def _no_fill(
    req: FillRequest,
    status: str,
    reason: str,
    fill_at: datetime,
    *,
    quote: Optional["Quote"] = None,
    age: Optional[float] = None,
    bookmaker: Optional[str] = None,
) -> FillResult:
    """A bet that did not happen. Zero stake, no price, no commission."""
    logger.debug(
        "frictions: no fill %s %s/%s — %s", status, req.race_uid, req.horse_key, reason
    )
    return FillResult(
        filled=False,
        status=status,
        reason=reason,
        matched_stake=0.0,
        unmatched_stake=max(0.0, float(req.stake)),
        fill_odds=None,
        fill_at=fill_at,
        quote_fetched_at=getattr(quote, "fetched_at", None) if quote is not None else None,
        quote_age_seconds=age,
        price_move=None,
        price_move_pct=None,
        bookmaker=bookmaker if bookmaker is not None else req.bookmaker,
        commission_rate=0.0,
        bog_applied=False,
    )


def _valid_odds(raw: Any) -> Optional[float]:
    try:
        d = float(raw)
    except (TypeError, ValueError):
        return None
    if not (d > _MIN_VALID_ODDS):
        return None
    return d


def _normalise_book(book: Optional[str]) -> str:
    """Stable hash key for a bookmaker, so casing cannot change a draw."""
    return str(book).strip().lower() if book else ""


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt is not None else None


def _round_opt(v: Optional[float], nd: int) -> Optional[float]:
    return round(float(v), nd) if v is not None else None
