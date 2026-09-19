"""Live closing-line-value test — the one test a backtest cannot do.

Backtests only ever see a single historical price (the SP), so they cannot
measure whether the price *we could actually take* (the best board price across
books, at decision time) beats the eventual closing line. This harness does,
forward in time, by paper-betting the system's value picks at the best board
price and settling them against the Betfair SP once results land:

    # daily, after `python -m scripts.refresh` has written predictions.json:
    python -m scripts.clv_live log       # paper-bet today's value picks @ best board price
    python -m scripts.clv_live settle    # settle matured picks vs SP, store CLV
    python -m scripts.clv_live report     # realised CLV / ROI / beat-close so far

    python -m scripts.clv_live run        # log + settle + report in one go

It reuses the existing paper-betting machinery end to end: ``record_bet`` already
stores the price taken so CLV can be computed at settlement, ``place_paper_bet``
guards against double-logging (idempotent per race/horse/market) and races that
have already started, ``settle_from_storage`` matches the betsp SP results, and
``settle_bet`` computes ``clv_pct = price_taken / SP - 1``. The picks are tagged
``notes="live-clv"`` so the report scopes to exactly this experiment.

Why this and not the holdout: the holdout settles at the pre-off ppwap and showed
CLV ~-4.7% in the [2.0,4.0] band — but ppwap is not the price you take, and those
models trained on the window (leakage unverified). This measures the *real*
edge: best board price vs the close, going forward, with no leakage possible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.bet_tracker import (
    BetTracker,
    DuplicateBetError,
    RaceStartedError,
    StopLossError,
    Strategy,
)
from utils.bet_settlement import settle_from_storage
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("clv_live")

_PREDICTIONS = ROOT / "data" / "predictions.json"
_TAG = "live-clv"  # notes marker isolating this experiment's bets


def _tracker() -> BetTracker:
    cfg = get_config().get("bet_tracker", {}) or {}
    return BetTracker(
        initial_bankroll=float(cfg.get("initial_bankroll", 1000.0)),
        flat_stake=float(cfg.get("flat_stake", 10.0)),
        stop_loss_pct=float(cfg.get("stop_loss_pct", 0.20)),
    )


def _value_picks(predictions_path: Path) -> list[dict]:
    """The system's value selections with a usable best board price.

    Uses the ``value_bet`` flag the predictor already sets — which encodes the
    config gates (EV >= ``value.min_expected_value``, odds in
    [``value.min_odds``, ``value.max_odds``], support required). The price taken
    is ``best_odds`` (best across books); we fall back to ``decimal_odds`` only
    if no board price is present."""
    try:
        data = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("clv_live: could not read %s (%s)", predictions_path, exc)
        return []

    picks: list[dict] = []
    for race in data.get("races", []):
        venue = race.get("venue")
        rt = race.get("race_time")
        # Complete field first (audit req 2): a mid-field value pick must reach
        # the CLV log too; display lists only for legacy caches.
        field = race.get("runners") or (
            race.get("selections", []) + race.get("excluded_low_odds", []))
        for r in field:
            if not r.get("value_bet"):
                continue
            price = r.get("best_odds") or r.get("decimal_odds")
            if not price or price <= 1.0:
                continue
            picks.append({
                "race_id": r.get("race_uid") or race.get("race_uid"),
                "horse_id": r.get("horse_id"),
                "horse_name": r.get("horse_name"),
                "venue": venue,
                "race_time": r.get("race_time") or rt,
                "odds": float(price),
                "book": r.get("best_book"),
                "won_prob": r.get("value_win_prob") or r.get("won_prob"),
                "value_edge": r.get("value_edge"),
            })
    return picks


def cmd_log(args) -> int:
    """Paper-bet today's value picks at the best board price (idempotent)."""
    tracker = _tracker()
    stake = float((get_config().get("bet_tracker", {}) or {}).get("flat_stake", 10.0))
    picks = _value_picks(Path(args.predictions))
    placed = skipped = 0
    for p in picks:
        notes = f"{_TAG}|{p.get('book') or '?'}"
        try:
            bet_id = tracker.place_paper_bet(
                horse_name=p["horse_name"], odds_decimal=p["odds"],
                stake=stake, bet_type=args.bet_type,
                race_id=p["race_id"], horse_id=p["horse_id"], venue=p["venue"],
                race_time=p["race_time"], won_prob=p.get("won_prob"),
                value_edge=p.get("value_edge"), strategy=Strategy.FLAT, notes=notes,
            )
            placed += 1
            logger.info("clv_live: logged #%d %s @ %.2f (%s)",
                        bet_id, p["horse_name"], p["odds"], p.get("book"))
        except DuplicateBetError:
            skipped += 1  # already logged on an earlier run today — idempotent
        except RaceStartedError:
            skipped += 1  # off already; too late to take the board price honestly
        except StopLossError as exc:
            logger.warning("clv_live: stop-loss active, halting (%s)", exc)
            break
    print(f"clv_live log: {placed} placed, {skipped} skipped "
          f"(of {len(picks)} value picks), bet_type={args.bet_type}")
    return 0


def cmd_settle(args) -> int:
    """Settle matured picks against the betsp SP and store CLV."""
    tracker = _tracker()
    settled = settle_from_storage(tracker)
    print(f"clv_live settle: {len(settled)} bet(s) settled against SP results")
    return 0


def _live_rows(tracker: BetTracker) -> list[dict]:
    return [b for b in tracker.all_bets(settled_only=True)
            if str(b.get("notes") or "").startswith(_TAG)]


def cmd_report(args) -> int:
    """Realised live CLV / ROI / beat-close for the logged value picks."""
    tracker = _tracker()
    rows = _live_rows(tracker)
    clvs = [b["clv_pct"] for b in rows if b.get("clv_pct") is not None]
    n_settled = len(rows)
    print("\n=== Live CLV test — best board price vs eventual SP ===")
    if not rows:
        pend = sum(1 for b in tracker.all_bets()
                   if str(b.get("notes") or "").startswith(_TAG))
        print(f"  No settled live-clv picks yet ({pend} pending — results lag a few days).")
        return 0
    wins = sum(1 for b in rows if b.get("outcome") == "win")
    staked = sum(b.get("stake", 0.0) for b in rows)
    pnl = sum(b.get("profit", 0.0) or 0.0 for b in rows)
    mean_clv = sum(clvs) / len(clvs) if clvs else None
    beat = sum(1 for c in clvs if c > 0) / len(clvs) * 100 if clvs else None
    print(f"  settled picks : {n_settled}   (CLV measured on {len(clvs)})")
    print(f"  win rate      : {wins / n_settled * 100:.1f}%")
    print(f"  ROI           : {pnl / staked * 100:+.2f}%   (P&L {pnl:+.2f})")
    print(f"  mean CLV      : {mean_clv:+.2f}%" if mean_clv is not None else "  mean CLV      : n/a")
    print(f"  beat-close    : {beat:.1f}%" if beat is not None else "  beat-close    : n/a")
    if mean_clv is not None:
        verdict = ("POSITIVE — the edge survives the close" if mean_clv > 0
                   else "still negative — paper-only, edge not proven")
        print(f"  VERDICT       : {verdict}")
    return 0


def cmd_run(args) -> int:
    cmd_log(args)
    cmd_settle(args)
    return cmd_report(args)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="scripts.clv_live",
                                description="Live CLV test: best board price vs SP")
    p.add_argument("--predictions", default=str(_PREDICTIONS))
    p.add_argument("--bet-type", default="win", choices=["win", "each_way"])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("log", "settle", "report", "run"):
        sub.add_parser(name)
    args = p.parse_args(argv)
    return {"log": cmd_log, "settle": cmd_settle,
            "report": cmd_report, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
