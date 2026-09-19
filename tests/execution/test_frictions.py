"""Execution-friction tests.

The load-bearing assertion in this file is the look-ahead one: a fill must use
the newest snapshot at or before ``requested_at + latency`` and must never reach
the closing price. Stage 4 measured CLV at -13.4%, so a simulator that quietly
settled at the close would invert the single number the audit turned on.

The snapshot store is faked here (``FakeStore``) so these tests pin
``execution.frictions`` against the documented ``SnapshotStore`` contract rather
than against another module's implementation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Optional

import pytest

from execution.config import ExecutionConfig, FrictionConfig
from execution.frictions import (
    FillRequest,
    FuturePriceError,
    attempt_fill,
    deterministic_uniform,
)

T0 = datetime(2026, 7, 20, 14, 0, 0)


# ── fakes ────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FakeQuote:
    """Mirrors the fields of ``execution.snapshots.Quote``."""

    race_uid: str
    horse_key: str
    bookmaker: str
    odds_decimal: float
    fetched_at: datetime
    market_type: str = "WIN"
    as_of: Optional[datetime] = None
    age_seconds: Optional[float] = None
    is_stale: bool = False
    sp: Optional[float] = None
    ew_places: Optional[int] = None
    ew_reduction: Optional[float] = None
    horse_name: Optional[str] = None


class FakeStore:
    """Minimal honest ``SnapshotStore``, mirroring the real semantics exactly.

    Two details matter and both are copied deliberately from
    ``execution.snapshots.SnapshotStore``: a quote after ``as_of`` is never
    returned, and ``max_age_seconds=None`` means *the store's own* 900s snapshot
    limit (not "no limit"), with a stale quote refused outright. The friction
    layer must therefore ask for an unfiltered quote and judge staleness itself,
    or ``STALE_QUOTE`` and ``allow_stale_fill`` become unreachable.
    """

    STORE_MAX_AGE = 900.0

    def __init__(self, quotes, *, leak_future: bool = False):
        self.quotes = sorted(quotes, key=lambda q: q.fetched_at)
        self.leak_future = leak_future
        self.calls: list[dict] = []

    def latest_quote(
        self,
        race_uid,
        horse_key,
        as_of,
        *,
        bookmaker=None,
        market_type="WIN",
        max_age_seconds=None,
    ):
        self.calls.append(
            {
                "race_uid": race_uid,
                "horse_key": horse_key,
                "as_of": as_of,
                "bookmaker": bookmaker,
                "market_type": market_type,
                "max_age_seconds": max_age_seconds,
            }
        )
        cands = [
            q
            for q in self.quotes
            if q.race_uid == race_uid
            and q.horse_key == horse_key
            and q.market_type == market_type
            and (bookmaker is None or q.bookmaker == bookmaker)
            and (self.leak_future or q.fetched_at <= as_of)
        ]
        if not cands:
            return None
        q = cands[-1]
        age = (as_of - q.fetched_at).total_seconds()
        limit = self.STORE_MAX_AGE if max_age_seconds is None else float(max_age_seconds)
        is_stale = age > limit
        if is_stale:
            return None  # the real store refuses stale quotes outright
        return replace(q, as_of=as_of, age_seconds=age, is_stale=is_stale)

    def best_quote(self, race_uid, horse_key, as_of, **kw):
        return self.latest_quote(race_uid, horse_key, as_of, **kw)


def make_cfg(**friction_overrides) -> ExecutionConfig:
    """An ExecutionConfig with only the friction knobs under test changed."""
    base = FrictionConfig(rejection_rate=0.0, suspension_rate=0.0)
    return replace(ExecutionConfig(), frictions=replace(base, **friction_overrides))


def make_req(**kw) -> FillRequest:
    defaults = dict(
        race_uid="2026-07-20T14:10:00+navan",
        horse_key="gallant_ruler",
        stake=10.0,
        requested_at=T0,
        quoted_odds=5.0,
        bookmaker="paddy_power",
    )
    defaults.update(kw)
    return FillRequest(**defaults)


def one_quote(odds=5.0, offset_seconds=-60, book="paddy_power") -> FakeQuote:
    return FakeQuote(
        race_uid="2026-07-20T14:10:00+navan",
        horse_key="gallant_ruler",
        bookmaker=book,
        odds_decimal=odds,
        fetched_at=T0 + timedelta(seconds=offset_seconds),
    )


# ── latency ──────────────────────────────────────────────────────────────────
def test_latency_shifts_the_fill_time_by_decision_plus_placement():
    cfg = make_cfg(decision_latency_seconds=30.0, placement_latency_seconds=15.0)
    store = FakeStore([one_quote()])

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.fill_at == T0 + timedelta(seconds=45)
    # The store must be queried AS OF the fill time, not the request time.
    assert store.calls[0]["as_of"] == T0 + timedelta(seconds=45)


def test_zero_latency_still_fills_at_the_request_moment():
    cfg = make_cfg(decision_latency_seconds=0.0, placement_latency_seconds=0.0)
    store = FakeStore([one_quote()])

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.fill_at == T0
    assert res.filled is True


# ── the closing price is never the executable price ──────────────────────────
def test_fill_uses_newest_quote_at_or_before_fill_time_never_the_close():
    """A far better closing price exists; it must be invisible to the fill."""
    cfg = make_cfg()
    store = FakeStore(
        [
            one_quote(odds=4.0, offset_seconds=-600),
            one_quote(odds=5.0, offset_seconds=-30),  # newest at-or-before fill_at
            one_quote(odds=12.0, offset_seconds=+600),  # the close: must not be used
        ]
    )

    res = attempt_fill(make_req(quoted_odds=5.0), store=store, cfg=cfg)

    assert res.filled is True
    assert res.fill_odds == 5.0
    assert res.quote_fetched_at == T0 - timedelta(seconds=30)


def test_a_store_that_leaks_a_future_quote_raises_rather_than_settling():
    cfg = make_cfg()
    store = FakeStore([one_quote(odds=12.0, offset_seconds=+600)], leak_future=True)

    with pytest.raises(FuturePriceError):
        attempt_fill(make_req(), store=store, cfg=cfg)


def test_a_quote_arriving_inside_the_latency_window_is_used():
    """Newest at-or-before fill_at means a price struck during the latency counts."""
    cfg = make_cfg(decision_latency_seconds=30.0, placement_latency_seconds=15.0)
    store = FakeStore(
        [one_quote(odds=5.0, offset_seconds=-30), one_quote(odds=4.2, offset_seconds=+20)]
    )

    res = attempt_fill(make_req(quoted_odds=5.0), store=store, cfg=cfg)

    assert res.fill_odds == 4.2  # the adverse move inside the window is taken
    assert res.price_move == pytest.approx(-0.8)
    assert res.price_move_pct == pytest.approx(-0.16)


# ── no quote / stale quote ───────────────────────────────────────────────────
def test_no_quote_is_a_no_fill_not_a_fill_at_the_last_known_price():
    cfg = make_cfg()
    # The only price predates nothing — it is after the fill time entirely.
    store = FakeStore([one_quote(odds=6.0, offset_seconds=+3600)])

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.status == "NO_QUOTE"
    assert res.filled is False
    assert res.matched_stake == 0.0
    assert res.fill_odds is None
    assert res.unmatched_stake == 10.0


def test_empty_store_yields_no_quote():
    res = attempt_fill(make_req(), store=FakeStore([]), cfg=make_cfg())
    assert res.status == "NO_QUOTE"
    assert res.filled is False


def test_quote_older_than_max_price_age_is_blocked():
    cfg = make_cfg(max_price_age_seconds=300.0, allow_stale_fill=False)
    store = FakeStore([one_quote(odds=5.0, offset_seconds=-600)])

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.status == "STALE_QUOTE"
    assert res.filled is False
    assert res.fill_odds is None
    assert res.quote_age_seconds == pytest.approx(645.0)  # 600 + 45s latency


def test_staleness_is_judged_by_the_friction_limit_not_the_stores_own():
    """The store's 900s snapshot limit must not pre-empt the execution decision.

    A quote older than the store's limit must still surface as STALE_QUOTE (and
    still be fillable under ``allow_stale_fill``) rather than being swallowed
    into NO_QUOTE by the store's default filter.
    """
    store = FakeStore([one_quote(odds=5.0, offset_seconds=-1800)])  # > 900s old

    blocked = attempt_fill(make_req(), store=store, cfg=make_cfg(allow_stale_fill=False))
    allowed = attempt_fill(make_req(), store=store, cfg=make_cfg(allow_stale_fill=True))

    assert blocked.status == "STALE_QUOTE"
    assert allowed.status == "FILLED"
    assert allowed.fill_odds == 5.0
    # The store must have been asked for an unfiltered quote.
    assert store.calls[0]["max_age_seconds"] == float("inf")


def test_stale_quote_fills_only_when_allow_stale_fill_is_set():
    cfg = make_cfg(max_price_age_seconds=300.0, allow_stale_fill=True)
    store = FakeStore([one_quote(odds=5.0, offset_seconds=-600)])

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.status == "FILLED"
    assert res.fill_odds == 5.0


def test_quote_without_an_executable_price_is_a_no_quote():
    cfg = make_cfg()
    store = FakeStore([one_quote(odds=1.0)])  # the feeds' "missing price" sentinel

    res = attempt_fill(make_req(), store=store, cfg=cfg)

    assert res.status == "NO_QUOTE"
    assert res.filled is False


# ── deterministic rejection / suspension ─────────────────────────────────────
def test_deterministic_uniform_is_stable_and_in_range():
    a = deterministic_uniform(20260727, "race", "horse", "book", "reject")
    b = deterministic_uniform(20260727, "race", "horse", "book", "reject")
    assert a == b
    assert 0.0 <= a < 1.0
    # Different key => different draw; different seed => different draw.
    assert a != deterministic_uniform(20260727, "race", "horse", "book", "suspend")
    assert a != deterministic_uniform(1, "race", "horse", "book", "reject")


def test_rate_one_always_rejects_and_rate_zero_never_does():
    store = FakeStore([one_quote()])
    always = attempt_fill(
        make_req(), store=store, cfg=make_cfg(rejection_rate=1.0, suspension_rate=0.0)
    )
    never = attempt_fill(
        make_req(), store=store, cfg=make_cfg(rejection_rate=0.0, suspension_rate=0.0)
    )
    assert always.status == "REJECTED"
    assert always.filled is False and always.matched_stake == 0.0
    assert never.status == "FILLED"


def test_suspension_is_checked_before_rejection():
    store = FakeStore([one_quote()])
    res = attempt_fill(
        make_req(), store=store, cfg=make_cfg(rejection_rate=1.0, suspension_rate=1.0)
    )
    assert res.status == "SUSPENDED"


def _statuses_for(horse_keys, cfg, order):
    """Fill one request per horse, in ``order``, returning {horse_key: status}."""
    out = {}
    for key in order:
        quote = FakeQuote(
            race_uid="R1",
            horse_key=key,
            bookmaker="paddy_power",
            odds_decimal=5.0,
            fetched_at=T0 - timedelta(seconds=30),
        )
        store = FakeStore([quote])
        req = make_req(race_uid="R1", horse_key=key)
        out[key] = attempt_fill(req, store=store, cfg=cfg).status
    return out


def test_rejection_and_suspension_are_reproducible_and_order_independent():
    cfg = make_cfg(rejection_rate=0.35, suspension_rate=0.15)
    keys = [f"horse_{i:03d}" for i in range(60)]

    first = _statuses_for(keys, cfg, keys)
    again = _statuses_for(keys, cfg, keys)
    shuffled_order = list(keys)
    random.Random(7).shuffle(shuffled_order)
    shuffled = _statuses_for(keys, cfg, shuffled_order)

    assert first == again, "same inputs must reproduce the same fills"
    assert first == shuffled, "fills must not depend on evaluation order"
    # Sanity: the draws actually bite, in roughly the configured proportions.
    assert {"FILLED", "REJECTED", "SUSPENDED"} == set(first.values())


def test_seed_argument_overrides_the_config_seed():
    store = FakeStore([one_quote()])
    cfg = make_cfg(rejection_rate=0.5, suspension_rate=0.0)
    a = attempt_fill(make_req(), store=store, cfg=cfg, seed=1).status
    b = attempt_fill(make_req(), store=store, cfg=cfg, seed=1).status
    assert a == b
    statuses = {
        attempt_fill(make_req(), store=store, cfg=cfg, seed=s).status for s in range(30)
    }
    assert statuses == {"FILLED", "REJECTED"}


# ── bookmaker limits ─────────────────────────────────────────────────────────
def test_stake_above_the_book_limit_is_partially_matched():
    cfg = make_cfg(max_stake_per_bet=50.0)
    store = FakeStore([one_quote()])

    res = attempt_fill(make_req(stake=125.0), store=store, cfg=cfg)

    assert res.status == "PARTIAL"
    assert res.filled is True
    assert res.matched_stake == 50.0
    assert res.unmatched_stake == 75.0


def test_stake_at_the_limit_is_a_full_fill():
    cfg = make_cfg(max_stake_per_bet=50.0)
    res = attempt_fill(make_req(stake=50.0), store=FakeStore([one_quote()]), cfg=cfg)
    assert res.status == "FILLED"
    assert res.matched_stake == 50.0
    assert res.unmatched_stake == 0.0


def test_zero_book_limit_is_a_rejected_no_fill():
    cfg = make_cfg(max_stake_per_bet=0.0)
    res = attempt_fill(make_req(stake=10.0), store=FakeStore([one_quote()]), cfg=cfg)
    assert res.status == "REJECTED"
    assert res.filled is False
    assert res.matched_stake == 0.0


def test_non_positive_stake_is_rejected():
    res = attempt_fill(make_req(stake=0.0), store=FakeStore([one_quote()]), cfg=make_cfg())
    assert res.status == "REJECTED"
    assert res.filled is False


# ── commission ───────────────────────────────────────────────────────────────
def test_exchange_commission_applies_only_to_exchange_sources():
    cfg = make_cfg(exchange_commission=0.02)
    exch = attempt_fill(
        make_req(bookmaker="betfair"),
        store=FakeStore([one_quote(book="betfair")]),
        cfg=cfg,
    )
    book = attempt_fill(
        make_req(bookmaker="paddy_power"),
        store=FakeStore([one_quote(book="paddy_power")]),
        cfg=cfg,
    )
    assert exch.commission_rate == pytest.approx(0.02)
    assert book.commission_rate == 0.0


def test_no_fill_carries_no_commission():
    res = attempt_fill(
        make_req(bookmaker="betfair"), store=FakeStore([]), cfg=make_cfg()
    )
    assert res.commission_rate == 0.0


# ── best odds guaranteed ─────────────────────────────────────────────────────
def test_bog_is_never_applied_by_default_even_when_recorded():
    cfg = make_cfg()  # config default: best_odds_guaranteed=False
    assert cfg.frictions.best_odds_guaranteed is False
    res = attempt_fill(
        make_req(bog_recorded=True), store=FakeStore([one_quote()]), cfg=cfg
    )
    assert res.bog_applied is False


def test_bog_requires_both_the_config_flag_and_recorded_evidence():
    store = FakeStore([one_quote()])
    flag_only = attempt_fill(
        make_req(bog_recorded=False),
        store=store,
        cfg=make_cfg(best_odds_guaranteed=True),
    )
    both = attempt_fill(
        make_req(bog_recorded=True),
        store=store,
        cfg=make_cfg(best_odds_guaranteed=True),
    )
    assert flag_only.bog_applied is False
    assert both.bog_applied is True


def test_bog_does_not_improve_the_fill_price():
    """BOG is a settlement promise; it may never move the price we struck."""
    store = FakeStore([one_quote(odds=4.5)])
    res = attempt_fill(
        make_req(quoted_odds=5.0, bog_recorded=True),
        store=store,
        cfg=make_cfg(best_odds_guaranteed=True),
    )
    assert res.bog_applied is True
    assert res.fill_odds == 4.5


# ── adverse move only ────────────────────────────────────────────────────────
def test_adverse_move_only_ignores_a_favourable_drift():
    store = FakeStore([one_quote(odds=7.0)])
    res = attempt_fill(
        make_req(quoted_odds=5.0), store=store, cfg=make_cfg(adverse_move_only=True)
    )
    assert res.fill_odds == 5.0
    assert res.price_move == pytest.approx(0.0)


def test_adverse_move_only_still_takes_the_worse_price():
    store = FakeStore([one_quote(odds=4.0)])
    res = attempt_fill(
        make_req(quoted_odds=5.0), store=store, cfg=make_cfg(adverse_move_only=True)
    )
    assert res.fill_odds == 4.0
    assert res.price_move == pytest.approx(-1.0)


def test_without_adverse_move_only_a_favourable_drift_is_taken():
    store = FakeStore([one_quote(odds=7.0)])
    res = attempt_fill(
        make_req(quoted_odds=5.0), store=store, cfg=make_cfg(adverse_move_only=False)
    )
    assert res.fill_odds == 7.0
    assert res.price_move == pytest.approx(2.0)


# ── serialisation ────────────────────────────────────────────────────────────
def test_to_dict_is_json_friendly():
    res = attempt_fill(make_req(), store=FakeStore([one_quote()]), cfg=make_cfg())
    d = res.to_dict()
    assert d["status"] == "FILLED"
    assert d["fill_at"] == (T0 + timedelta(seconds=45)).isoformat()
    assert d["quote_fetched_at"] == (T0 - timedelta(seconds=60)).isoformat()
    assert isinstance(d["bog_applied"], bool)
