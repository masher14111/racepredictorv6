"""Walk-forward execution simulator (Stage-5 requirement 2).

``execution.baselines`` answers "which runner would the strategy pick?" in a
frictionless world. This module answers the only question that matters for
money: **what would actually have been struck, and what would it have
returned?** Between the two sit every friction the requirement lists —

* **odds age and movement** — the price is read from
  :class:`~execution.snapshots.SnapshotStore` with ``fetched_at <= as_of``
  enforced in SQL, so a decision can only ever see prices that existed when it
  was made;
* **request and bet latency** — the decision is taken at
  ``off - decision_latency``, the bet lands at ``+ placement_latency``, and the
  price is re-read at the later instant;
* **rejected and suspended markets** — a configurable share of attempts return
  no fill at all, chosen deterministically so a run is reproducible;
* **non-runners, Rule 4, dead heats, voids** — read from the raw archive by
  :mod:`execution.race_facts`, not assumed;
* **bookmaker limits** — ``frictions.max_stake_per_bet`` caps the matched stake
  and the remainder is recorded as unmatched, never silently filled;
* **exchange commission** — charged on net winnings only, for exchange sources;
* **each-way terms captured at bet time** — carried on the ticket, never
  re-derived at settlement;
* **no assumed BOG** — best-odds-guaranteed applies only where the quote itself
  records it.

Two rules govern everything here
--------------------------------
1. **The closing price is never a fill price.** It is read through
   :meth:`SnapshotStore.closing_quote`, used for CLV, and for nothing else.
2. **A friction that cannot be evaluated blocks the bet.** No quote, a stale
   quote, an unknown result — each ends the ticket rather than degrading into an
   optimistic assumption.

The simulator is *deterministic*: rejections and suspensions come from
:func:`execution.frictions.deterministic_uniform`, a blake2b hash of the seed and
the bet's identity, so results do not depend on iteration order and a rerun
reproduces the ledger exactly.

**What this cannot tell you.** A backtest over 2025-2026 measures the past under
assumed frictions. It is evidence about execution mechanics, not evidence that a
strategy is profitable, and it can never satisfy the forward-release gate — see
:mod:`execution.forward_gate`, which refuses backtest evidence by construction.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from execution import baselines as bl
from execution import race_facts as rfacts
from execution.config import ExecutionConfig, load_execution_config
from execution.frictions import FillRequest, FillResult, attempt_fill
from execution.safeguards import SafeguardState, Safeguards
from execution.settlement import EachWayTerms, RaceResult, settle_ticket
from execution.snapshots import Quote, SnapshotStore, horse_key as snap_horse_key
from execution.staking import ExposureState, StakePlanner
from utils.logger import get_logger
from utils.text_norm import minute_key, norm_horse, norm_venue

logger = get_logger(__name__)

DEFAULT_SCORED_PATH = os.path.join("data", "audit", "stage4", "wf_scored.parquet")
DEFAULT_PANEL_PATH = os.path.join("data", "audit", "stage4", "panel.parquet")
DEFAULT_SNAPSHOT_DB = os.path.join("data", "execution", "backtest_snapshots.db")

DEFAULT_SEED = 20260727
# The historical warehouse holds ONE pre-off observation per runner (ppwap,
# falling back to morningwap). We stamp it at this lead so a decision taken at
# off - decision_latency can see it. It is a *reconstruction*, not a captured
# series: the backtest therefore cannot measure price movement between
# observations, only the frictions that act on a single quote. Live capture
# through execution.snapshots is what produces a real series.
PREOFF_LEAD_SECONDS = 30 * 60
# How far before the off the bettor acts. It must clear the started-race buffer
# with room to spare — the safeguard exists to stop bets going in as the tapes
# rise, and a simulator that decided inside the buffer would be modelling a bet
# the live system refuses to place. Defaulting it to the seeding lead means the
# quote read at decision time is the one actually on the book at that moment.
DECISION_LEAD_SECONDS = PREOFF_LEAD_SECONDS

# The panel's race_uid minute stamp is local wall-clock (Europe/Dublin), matching
# the rest of the repo; Britain and Ireland share DST rules.
_RACING_TZ = ZoneInfo("Europe/Dublin")

SETTLEMENT_ARCHIVE = "archive"
SETTLEMENT_PANEL = "panel"

LEDGER_COLUMNS: tuple[str, ...] = (
    # the columns execution.evaluation and execution.forward_gate consume
    "race_uid", "race_date", "horse_key", "stake", "decimal_odds", "model_prob",
    "won", "returns", "profit", "closing_odds", "voided",
    # the execution record
    "strategy", "requested_odds", "fill_status", "fill_reason", "fill_at",
    "quote_fetched_at", "quote_age_seconds", "price_move_pct", "unmatched_stake",
    "bookmaker", "commission_rate", "commission_paid", "bog_applied",
    "rule_4_deduction", "dead_heat_divisor", "settlement_status",
    "settlement_source", "binding_constraint", "blocked_by", "bankroll_after",
)


# ─────────────────────────────── result records ──────────────────────────────
@dataclass(frozen=True)
class SimulationResult:
    """One walk-forward run: what was struck, what was not, and why."""

    strategy: str
    ledger: pd.DataFrame
    attempts: pd.DataFrame
    summary: dict = field(default_factory=dict)

    @property
    def n_struck(self) -> int:
        return int(len(self.ledger))

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "n_struck": self.n_struck, **self.summary}


# ─────────────────────────────── key helpers ─────────────────────────────────
def facts_key_for(race_uid: str) -> str:
    """Map a panel ``race_uid`` (``Venue|YYYY-MM-DDTHH:MM``) to a race-facts key.

    The panel keeps the venue verbatim; :mod:`execution.race_facts` normalises
    it, because course spellings differ across sources. Normalising here is the
    single place the two vocabularies meet.
    """
    text = str(race_uid or "")
    venue, _, stamp = text.partition("|")
    return f"{norm_venue(venue)}|{minute_key(stamp)}"


def _race_off(race_uid: str, race_time: Any = None) -> Optional[datetime]:
    """The scheduled off in UTC.

    ``race_time`` from the panel is the authority when present — it is offset-aware
    (``2026-05-01T16:15:00+01:00``), so it converts exactly. The ``race_uid`` minute
    stamp is *local* wall-clock with no offset attached, so the fallback reads it in
    the racing timezone rather than pretending it is already UTC.
    """
    stamp = _as_utc(race_time)
    if stamp is not None:
        return stamp
    _, _, text = str(race_uid or "").partition("|")
    try:
        dt = datetime.strptime(minute_key(text), "%Y-%m-%dT%H:%M")
    except (TypeError, ValueError):
        return None
    return dt.replace(tzinfo=_RACING_TZ).astimezone(timezone.utc)


def _as_utc(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if ts is pd.NaT:
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _f(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ───────────────────────────── loading the panel ─────────────────────────────
def load_scored_panel(
    *,
    scored_path: str = DEFAULT_SCORED_PATH,
    panel_path: str = DEFAULT_PANEL_PATH,
    start: Any = None,
    end: Any = None,
    prob_col: str = "norm_ind",
) -> pd.DataFrame:
    """Load the Stage-4 walk-forward scored frame, enriched for execution.

    ``prob_col`` defaults to ``norm_ind`` — the **within-race normalised
    independent** probability. That is the honest headline number: ``p_ind``
    saturates out of sample and ``p_adj`` was measured at 0.925 correlation with
    ``1/price``, i.e. mostly a price echo (Stage 4).

    Adds ``horse_name`` / ``horse_key`` / ``race_time`` from the audit panel,
    which the scored frame drops.
    """
    scored = pd.read_parquet(scored_path)
    if prob_col not in scored.columns:
        raise KeyError(
            f"{prob_col!r} not in {scored_path} (have: {sorted(scored.columns)})"
        )

    cols = ["race_uid", "horse_id", "horse_name", "race_time", "ew_places",
            "ew_reduction"]
    panel = pd.read_parquet(panel_path, columns=cols)
    panel = panel.drop_duplicates(subset=["race_uid", "horse_id"], keep="first")
    out = scored.merge(panel, on=["race_uid", "horse_id"], how="left")

    out["race_date"] = pd.to_datetime(out["race_date"], utc=True, errors="coerce")
    if start is not None:
        out = out[out["race_date"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        out = out[out["race_date"] <= pd.Timestamp(end, tz="UTC")]

    out["horse_key"] = [
        norm_horse(n) if isinstance(n, str) and n.strip() else snap_horse_key(horse_id=h)
        for n, h in zip(out.get("horse_name"), out["horse_id"])
    ]
    out["model_prob"] = pd.to_numeric(out[prob_col], errors="coerce")
    out["decimal_odds"] = pd.to_numeric(out["bet_price"], errors="coerce")
    out["closing_odds"] = pd.to_numeric(out["close_price"], errors="coerce")
    # The reference book for the de-vigged baseline is the same pre-off market —
    # the ONLY price we have a full book for historically. Stated, not hidden:
    # the model-vs-market comparison is model against its own execution price.
    out["reference_odds"] = out["decimal_odds"]
    out["prob_source"] = prob_col

    logger.info(
        "simulator: loaded %d rows / %d races (%s..%s) using prob=%s",
        len(out), out["race_uid"].nunique(),
        out["race_date"].min(), out["race_date"].max(), prob_col,
    )
    return out.reset_index(drop=True)


# ───────────────────────────── seeding snapshots ─────────────────────────────
def seed_snapshots(
    panel: pd.DataFrame,
    store: SnapshotStore,
    *,
    lead_seconds: float = PREOFF_LEAD_SECONDS,
    bookmaker: str = "betfair",
    include_closing: bool = True,
) -> dict:
    """Reconstruct a point-in-time series from the historical warehouse.

    Two rows per runner, and the gap between their timestamps is the whole
    point:

    * the **pre-off** quote at ``off - lead_seconds``, which a decision taken
      before the off can read;
    * the **closing** quote (Betfair SP) stamped at the off itself, which
      ``fetched_at <= as_of`` makes *unreadable* to any pre-off decision and
      which only :meth:`SnapshotStore.closing_quote` can reach.

    So the look-ahead the store forbids is not merely untested here — the data
    is laid out so that taking it would require breaking the store's contract.
    """
    rows: list[dict] = []
    for r in panel.itertuples(index=False):
        off = _race_off(r.race_uid, getattr(r, "race_time", None))
        if off is None:
            continue
        odds = _f(getattr(r, "decimal_odds", None))
        if odds is not None and odds > 1.0:
            rows.append({
                "race_uid": r.race_uid,
                "horse_key": r.horse_key,
                "bookmaker": bookmaker,
                "market_type": "WIN",
                "odds_decimal": odds,
                "fetched_at": off - timedelta(seconds=float(lead_seconds)),
                "race_time": off,
                "horse_name": getattr(r, "horse_name", None),
                "ew_places": getattr(r, "ew_places", None),
                "ew_reduction": getattr(r, "ew_reduction", None),
            })
        close = _f(getattr(r, "closing_odds", None))
        if include_closing and close is not None and close > 1.0:
            rows.append({
                "race_uid": r.race_uid,
                "horse_key": r.horse_key,
                "bookmaker": bookmaker,
                "market_type": "WIN",
                "odds_decimal": close,
                "fetched_at": off,
                "race_time": off,
                "horse_name": getattr(r, "horse_name", None),
                "sp": close,
            })
    result = store.record(rows)
    logger.info("simulator: seeded snapshots %s", result.to_dict())
    return result.to_dict()


# ───────────────────────────────── selection ─────────────────────────────────
def _backtest_source_health(age_seconds: float):
    """Health lookup for a *reconstructed* historical feed.

    ``utils.source_health`` telemetry describes scrapers running now; asking it
    about a race in May 2026 answers a different question and would block the
    whole backtest on today's scraper state. What is knowable is the age of the
    quote the walk is actually reading: the reconstruction stamps it at
    ``off - lead`` and the decision happens ``decision_latency`` later, so that
    latency *is* the feed age at decision time. Reporting it means the staleness
    limit is applied to a real number rather than waived.

    This still leaves live scraper-outage blocking unexercised by the backtest —
    :func:`_summary` records that under ``not_exercised``.
    """
    age = float(age_seconds)

    def lookup(source: str) -> dict:
        return {
            "source": source,
            "status": "ok",
            "ok": True,
            "age_seconds": age,
            "note": "reconstructed historical snapshot, aged by decision latency",
        }

    return lookup


def _strategy_picks(
    panel: pd.DataFrame, *, strategy: str, cfg: ExecutionConfig,
    eligible: Optional[set[str]] = None, odds_band: Optional[tuple] = None,
) -> pd.DataFrame:
    """Frictionless selections for one strategy, as a race-keyed frame.

    ``odds_band`` reaches ``model_only`` alone. The band is part of the strategy
    under test, so handing it to the controls as well would tell us how the
    market and the favourite fare *inside the model's chosen niche* rather than
    what they do with the whole book. Leaving the baselines unrestricted can only
    make the model's comparison harder, which is the direction this file always
    errs in.
    """
    fn = bl._STRATEGIES.get(strategy)
    if fn is None:
        raise KeyError(f"unknown strategy {strategy!r} (have {sorted(bl._STRATEGIES)})")
    if eligible is None:
        eligible = bl.eligible_race_ids(panel, cfg=cfg)
    frame = panel[panel["race_uid"].isin(eligible)]
    if odds_band and strategy == "model_only":
        return fn(frame, cfg=cfg, odds_band=odds_band)
    return fn(frame, cfg=cfg)


# ──────────────────────────────── the simulator ──────────────────────────────
def simulate(
    panel: pd.DataFrame,
    *,
    cfg: Optional[ExecutionConfig] = None,
    store: SnapshotStore,
    strategy: str = "model_only",
    race_facts: Optional[Mapping[str, "rfacts.RaceFacts"]] = None,
    bankroll: float = 1000.0,
    seed: int = DEFAULT_SEED,
    eligible: Optional[set[str]] = None,
    apply_safeguards: bool = True,
    bet_type: str = "win",
    ew_terms: Optional[EachWayTerms] = None,
    decision_lead_seconds: float = DECISION_LEAD_SECONDS,
    odds_band: Optional[tuple] = None,
) -> SimulationResult:
    """Walk one strategy forward through the panel, race by race, in time order.

    Nothing looks beyond ``decision_at``. Bankroll, per-race and per-day exposure
    and the safeguard state all evolve as the walk proceeds, so a bet late in the
    window is sized by a bankroll that already absorbed every earlier result —
    the ordering that makes drawdown and stop-loss meaningful rather than
    decorative.

    ``race_facts`` supplies real non-runner / Rule 4 / dead-heat / places-paid
    truth. Races missing from it settle from the panel's ``won`` label under
    ``settlement_source == 'panel'``, and the summary reports how many did, so
    the coverage is never implicit.
    """
    cfg = cfg or load_execution_config()
    if ew_terms is None and bet_type != "win":
        raise ValueError(
            "each-way terms have no historical coverage in this warehouse — pass "
            "ew_terms explicitly to state the assumption, or simulate win-only"
        )

    picks = _strategy_picks(panel, strategy=strategy, cfg=cfg, eligible=eligible,
                            odds_band=odds_band)
    if picks.empty:
        logger.warning("simulator: strategy %s selected nothing", strategy)
        return SimulationResult(
            strategy=strategy,
            ledger=pd.DataFrame(columns=list(LEDGER_COLUMNS)),
            attempts=pd.DataFrame(columns=list(LEDGER_COLUMNS)),
            summary=_summary([], [], strategy=strategy, cfg=cfg, bankroll=bankroll,
                             initial=bankroll, race_facts=race_facts),
        )

    # Race order is the walk order. Everything downstream depends on it. The off
    # comes from the panel rather than the picks frame, so it stays correct no
    # matter which columns a strategy chooses to carry through.
    offs = {
        str(uid): _race_off(uid, grp["race_time"].iloc[0] if "race_time" in grp else None)
        for uid, grp in panel.groupby("race_uid", sort=False)
    }
    picks = picks.copy()
    picks["_off"] = [offs.get(str(u)) for u in picks["race_uid"]]
    picks = picks[picks["_off"].notna()].sort_values(["_off", "race_uid", "horse_key"])

    fr = cfg.frictions
    guards = (
        Safeguards(
            cfg,
            source_health_fn=_backtest_source_health(fr.decision_latency_seconds),
        )
        if apply_safeguards
        else None
    )

    initial_bankroll = float(bankroll)
    current = float(bankroll)
    day_staked = 0.0
    day_profit = 0.0
    current_day: Optional[str] = None
    struck: list[dict] = []
    attempts: list[dict] = []
    open_tickets = 0

    for race_uid, group in picks.groupby("race_uid", sort=False):
        off: datetime = group["_off"].iloc[0]
        # We read the book at ``off - lead``, spend decision_latency deciding and
        # placement_latency getting the bet down. Latency therefore *ages the
        # quote* between reading and striking, which is what it costs in
        # practice; it is not the lead itself.
        # ``decision_at`` is the moment we read the book. ``attempt_fill`` ages the
        # request by ``total_latency_seconds`` itself to reach the strike time, so
        # nothing here adds latency on top — that would charge it twice.
        decision_at = off - timedelta(seconds=float(decision_lead_seconds))
        day = off.date().isoformat()
        if day != current_day:
            current_day, day_staked, day_profit = day, 0.0, 0.0

        facts = None
        if race_facts is not None:
            facts = race_facts.get(facts_key_for(race_uid))
        race_staked = 0.0

        for pick in group.itertuples(index=False):
            horse_key = str(pick.horse_key)
            prob = _f(getattr(pick, "model_prob", None))
            quoted = _f(getattr(pick, "decimal_odds", None))
            row: dict = {
                "race_uid": race_uid,
                "race_date": off.date().isoformat(),
                "horse_key": horse_key,
                "strategy": strategy,
                "model_prob": prob,
                "requested_odds": quoted,
                "stake": 0.0,
                "decimal_odds": None,
                "won": None,
                "returns": 0.0,
                "profit": 0.0,
                "closing_odds": None,
                "voided": False,
                "fill_status": None,
                "fill_reason": None,
                "fill_at": None,
                "quote_fetched_at": None,
                "quote_age_seconds": None,
                "price_move_pct": None,
                "unmatched_stake": 0.0,
                "bookmaker": None,
                "commission_rate": 0.0,
                "commission_paid": 0.0,
                "bog_applied": False,
                "rule_4_deduction": 0.0,
                "dead_heat_divisor": 1.0,
                "settlement_status": None,
                "settlement_source": None,
                "binding_constraint": None,
                "blocked_by": None,
                "bankroll_after": round(current, 4),
            }

            if prob is None or quoted is None or quoted <= 1.0:
                row["blocked_by"] = "unusable_probability_or_price"
                attempts.append(row)
                continue

            # ── stake, under every ceiling ────────────────────────────────
            planner = StakePlanner(cfg, bankroll=current)
            decision = planner.plan(
                prob=prob,
                decimal_odds=quoted,
                race_uid=race_uid,
                race_date=off,
                # ExposureState is keyed per race and per day so a live ticket
                # store can hand it a whole book. Here the walk is strictly
                # chronological and both counters reset on their own boundary,
                # so a one-entry map for the race and day in hand is exact.
                exposure=ExposureState(
                    race_staked={race_uid: race_staked},
                    day_staked={day: day_staked},
                    open_tickets=open_tickets,
                ),
            )
            row["binding_constraint"] = decision.binding_constraint
            if decision.stake <= 0.0:
                row["blocked_by"] = decision.binding_constraint or "zero_stake"
                attempts.append(row)
                continue

            # ── safeguards ────────────────────────────────────────────────
            if guards is not None:
                check = guards.check(
                    {
                        "race_uid": race_uid,
                        "horse_key": horse_key,
                        "race_time": off,
                        "stake": decision.stake,
                        "bet_type": bet_type,
                        "sources": ["betfair"],
                        # A reconstructed pre-off quote is by construction as
                        # fresh as the seeding lead; stale-source blocking is
                        # exercised live, not here.
                        "source_health_ok": True,
                    },
                    SafeguardState(
                        bankroll=current,
                        initial_bankroll=initial_bankroll,
                        day_profit=day_profit,
                        day_staked=day_staked,
                        open_tickets=open_tickets,
                        trading_day=day,
                    ),
                    existing_tickets=[],
                    now=decision_at,
                )
                if not check.allowed:
                    row["blocked_by"] = ",".join(check.blocks) or "safeguard"
                    attempts.append(row)
                    continue

            # ── strike it: latency, staleness, movement, rejection, limits ─
            fill: FillResult = attempt_fill(
                FillRequest(
                    race_uid=race_uid,
                    horse_key=horse_key,
                    stake=decision.stake,
                    requested_at=decision_at,
                    quoted_odds=quoted,
                    bookmaker=None,
                    market_type="WIN",
                    bet_type=bet_type,
                    bog_recorded=False,
                ),
                store=store,
                cfg=cfg,
                seed=seed,
            )
            row.update({
                "fill_status": fill.status,
                "fill_reason": fill.reason,
                "fill_at": fill.fill_at.isoformat() if fill.fill_at else None,
                "quote_fetched_at": fill.quote_fetched_at,
                "quote_age_seconds": fill.quote_age_seconds,
                "price_move_pct": fill.price_move_pct,
                "unmatched_stake": fill.unmatched_stake,
                "bookmaker": fill.bookmaker,
                "commission_rate": fill.commission_rate,
                "bog_applied": fill.bog_applied,
            })
            if not fill.filled or not fill.matched_stake:
                row["blocked_by"] = fill.status
                attempts.append(row)
                continue

            stake = float(fill.matched_stake)
            odds = float(fill.fill_odds)
            row["stake"] = stake
            row["decimal_odds"] = odds

            # ── settle ────────────────────────────────────────────────────
            settled = _settle(
                facts=facts,
                pick=pick,
                race_uid=race_uid,
                horse_key=horse_key,
                stake=stake,
                odds=odds,
                bet_type=bet_type,
                ew_terms=ew_terms,
                commission_rate=fill.commission_rate,
                cfg=cfg,
            )
            if settled is None:
                row["blocked_by"] = "no_result"
                row["stake"] = 0.0
                row["decimal_odds"] = None
                attempts.append(row)
                continue
            result, source = settled
            row.update({
                "won": 1 if result.status == "WIN" else 0,
                "returns": result.returns,
                "profit": result.profit,
                "voided": result.status == "VOID",
                "commission_paid": result.commission_paid,
                "rule_4_deduction": result.rule_4_deduction,
                "dead_heat_divisor": result.dead_heat_divisor,
                "settlement_status": result.status,
                "settlement_source": source,
            })

            # ── closing price, for CLV only ───────────────────────────────
            closing = store.closing_quote(race_uid, horse_key, market_type="WIN")
            row["closing_odds"] = closing.odds_decimal if closing else _f(
                getattr(pick, "closing_odds", None)
            )

            race_staked += stake
            day_staked += stake
            day_profit += result.profit
            current += result.profit
            row["bankroll_after"] = round(current, 4)
            struck.append(row)
            attempts.append(row)

    ledger = pd.DataFrame(struck, columns=list(LEDGER_COLUMNS))
    all_attempts = pd.DataFrame(attempts, columns=list(LEDGER_COLUMNS))
    summary = _summary(
        struck, attempts, strategy=strategy, cfg=cfg, bankroll=current,
        initial=initial_bankroll, race_facts=race_facts,
    )
    logger.info(
        "simulator: %s struck %d of %d attempts, bankroll %.2f -> %.2f",
        strategy, len(struck), len(attempts), initial_bankroll, current,
    )
    return SimulationResult(
        strategy=strategy, ledger=ledger, attempts=all_attempts, summary=summary
    )


def _settle(
    *, facts, pick, race_uid: str, horse_key: str, stake: float, odds: float,
    bet_type: str, ew_terms, commission_rate: float, cfg,
):
    """Settle from the archive when we have it, from the panel label otherwise.

    Returns ``None`` when neither source can settle the ticket — a genuinely
    unknown outcome, which must leave the bet out of the ledger rather than
    entering it as a loss (that would understate returns) or a void (that would
    launder it away).
    """
    if facts is not None:
        outcome = facts.outcome_for(horse_key)
        if outcome != rfacts.UNKNOWN:
            result = settle_ticket(
                stake=stake, decimal_odds=odds, bet_type=bet_type,
                horse_key=horse_key, race_result=facts.to_race_result(),
                ew_terms=ew_terms, commission_rate=commission_rate, cfg=cfg,
            )
            if result.status != "NO_RESULT":
                return result, SETTLEMENT_ARCHIVE

    won = getattr(pick, "won", None)
    try:
        won = int(won)
    except (TypeError, ValueError):
        return None
    if won not in (0, 1):
        return None
    # Panel fallback: a plain win/lose with no deductions available. Recorded as
    # such so the report can state how much of the ledger lacks archive truth.
    result = settle_ticket(
        stake=stake, decimal_odds=odds, bet_type="win", horse_key=horse_key,
        race_result=RaceResult(race_uid=race_uid, positions={horse_key: 1 if won else 99}),
        commission_rate=commission_rate, cfg=cfg,
    )
    return result, SETTLEMENT_PANEL


def _summary(struck, attempts, *, strategy, cfg, bankroll, initial, race_facts) -> dict:
    """Execution-mechanics counters — deliberately not a performance claim."""
    blocked: dict[str, int] = {}
    for row in attempts:
        reason = row.get("blocked_by")
        if reason:
            blocked[reason] = blocked.get(reason, 0) + 1
    archive = sum(1 for r in struck if r.get("settlement_source") == SETTLEMENT_ARCHIVE)
    turnover = sum(float(r.get("stake") or 0.0) for r in struck)
    return {
        "strategy": strategy,
        "n_attempts": len(attempts),
        "n_struck": len(struck),
        "strike_rate": round(len(struck) / len(attempts), 4) if attempts else 0.0,
        "blocked_by": dict(sorted(blocked.items(), key=lambda kv: -kv[1])),
        "turnover": round(turnover, 2),
        "initial_bankroll": round(float(initial), 2),
        "final_bankroll": round(float(bankroll), 2),
        "settled_from_archive": archive,
        "settled_from_panel": len(struck) - archive,
        "archive_coverage": round(archive / len(struck), 4) if struck else 0.0,
        "race_facts_supplied": race_facts is not None,
        "paper_only": bool(cfg.paper_only),
        "evidence_kind": "backtest",
        "not_exercised": [
            # Say what these numbers cannot speak to, next to the numbers.
            "live source-health staleness (scraper telemetry describes today, not "
            "the backtest window; quote age is enforced instead)",
            "price movement between observations (the warehouse holds one pre-off "
            "quote per runner, so movement is modelled, not measured)",
            "each-way terms and best-odds-guaranteed (neither is recorded "
            "historically; win-only unless terms are passed explicitly)",
        ],
        "caveat": (
            "Backtest under assumed frictions over a fixed historical window. "
            "This is evidence about execution mechanics, NOT evidence that the "
            "strategy is profitable, and it cannot satisfy the forward-release "
            "gate."
        ),
    }


# ─────────────────────────── the full comparison run ─────────────────────────
def run_walk_forward(
    panel: pd.DataFrame,
    *,
    cfg: Optional[ExecutionConfig] = None,
    store: SnapshotStore,
    race_facts: Optional[Mapping[str, "rfacts.RaceFacts"]] = None,
    strategies: Sequence[str] = tuple(bl.BASELINES),
    bankroll: float = 1000.0,
    seed: int = DEFAULT_SEED,
    odds_band: Optional[tuple] = None,
) -> dict[str, SimulationResult]:
    """Run every strategy over the **same** eligible race set.

    The shared ``eligible`` set is computed once and handed to each run, so the
    three ROI/CLV numbers are comparable. Which races the set contains is itself
    a model-driven choice (the gates use the model's EV), so ``favourite``'s
    number here is "the favourite, in races the model liked" — never a
    standalone claim about backing favourites.

    Eligibility is unaffected by ``cfg.gates.min_edge`` / ``min_expected_value``
    by design (see :func:`baselines.eligible_race_ids`), so handing this the
    selected strategy's thresholds narrows what ``model_only`` *bets*, never
    which races the comparison runs over.
    """
    cfg = cfg or load_execution_config()
    eligible = bl.eligible_race_ids(panel, cfg=cfg)
    logger.info(
        "simulator: %d eligible races of %d for %s",
        len(eligible), panel["race_uid"].nunique(), list(strategies),
    )
    return {
        name: simulate(
            panel, cfg=cfg, store=store, strategy=name, race_facts=race_facts,
            bankroll=bankroll, seed=seed, eligible=eligible, odds_band=odds_band,
        )
        for name in strategies
    }
