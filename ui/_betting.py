"""Streamlit glue for paper betting — tracker accessor + reusable widgets.

Keeps all Streamlit-specific paper-betting code in one place so the race-detail
page and the dedicated Paper Betting page share the same tracker instance, the
same "place bet" form, and the same guard handling. The design system styles the
native widgets (button / number_input / selectbox), so the form needs no custom
CSS — only the standing "Paper / Practice — no real money" reminder is bespoke.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st

from utils.bet_tracker import (
    BetTracker,
    DuplicateBetError,
    RaceStartedError,
    StopLossError,
    Strategy,
)
from utils.config_loader import get_config
from ui import _components as C

PAPER_NOTICE = (
    '<div class="rp-paper-note">'
    '<span class="dot"></span>'
    '<b>Paper / Practice — no real money.</b> '
    'Bets are simulated against a virtual bankroll to test the model.'
    '</div>'
)


@st.cache_resource
def get_tracker() -> BetTracker:
    """Process-wide tracker bound to the live ``races.db`` (cached across reruns).

    Reads its bankroll/staking defaults from the ``bet_tracker`` config section so
    the UI and the CLI agree on the starting bankroll and stop-loss.
    """
    cfg = get_config().get("bet_tracker") or {}
    return BetTracker(
        initial_bankroll=float(cfg.get("initial_bankroll", 1000.0)),
        flat_stake=float(cfg.get("flat_stake", 10.0)),
        kelly_fraction=float(cfg.get("kelly_fraction", 0.25)),
        stop_loss_pct=float(cfg.get("stop_loss_pct", 0.20)),
    )


def _current_strategy() -> str:
    s = st.session_state.get("strategy", "flat")
    return s if s in (e.value for e in Strategy) else "flat"


def place_bet_widget(sel: dict, race: dict, ccy: str) -> None:
    """Render a compact "place paper bet" form for one runner.

    Prefills market, best price, and a strategy-based recommended stake; on
    submit it routes through ``BetTracker.place_paper_bet`` so the started-race,
    double-bet, and stop-loss guards all apply. Surfaces each guard as an inline
    message rather than a stack trace.
    """
    tracker = get_tracker()
    horse_id = sel.get("horse_id")
    race_id = C.race_slug(race)
    horse_name = sel.get("horse_name") or "—"
    venue = race.get("venue")
    race_time = race.get("race_time")
    best = sel.get("best_odds") or sel.get("decimal_odds") or 0.0

    started = BetTracker.race_has_started(race_time)
    key = f"{race_id}:{horse_id}"

    markets = ["Win"] + (["Each-way"] if race.get("each_way_available") else [])

    with st.expander(f"＋ Paper bet — {horse_name}", expanded=False):
        if started:
            st.caption("Race has started — betting closed.")
            return
        with st.form(key=f"betform-{key}", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            market = c1.selectbox("Market", markets, key=f"mkt-{key}")
            bet_type = "each_way" if market == "Each-way" else "win"
            odds = c2.number_input(
                "Odds taken", min_value=1.01, value=float(best) if best else 1.01,
                step=0.10, format="%.2f", key=f"odds-{key}",
            )
            rec = tracker.recommend_stake(odds, sel.get("won_prob") or 0.0,
                                          _current_strategy())
            stake = c3.number_input(
                "Stake", min_value=0.01, value=float(round(rec, 2)) if rec else 10.0,
                step=1.0, format="%.2f", key=f"stk-{key}",
            )
            edge = sel.get("value_edge")
            edge_txt = f" · model edge {edge * 100:+.0f}pp" if edge is not None else ""
            st.caption(f"Bankroll {ccy}{tracker.bankroll:,.2f}{edge_txt}")
            submitted = st.form_submit_button(f"Place {market.lower()} bet")

        if submitted:
            try:
                bet_id = tracker.place_paper_bet(
                    horse_name, odds, stake, bet_type=bet_type,
                    race_id=race_id, horse_id=str(horse_id) if horse_id else None,
                    venue=venue, race_time=race_time,
                    composite_score=sel.get("composite_score"),
                    won_prob=sel.get("won_prob"), value_edge=edge,
                    strategy=_current_strategy(),
                )
            except DuplicateBetError:
                st.warning(f"You already have a {bet_type.replace('_', '-')} bet on "
                           f"{horse_name} in this race.")
            except RaceStartedError:
                st.warning("Race has started — betting closed.")
            except StopLossError as exc:
                st.error(str(exc))
            else:
                st.success(f"Paper bet #{bet_id} placed: {ccy}{stake:,.2f} on "
                           f"{horse_name} @ {odds:.2f}.")
                st.rerun()


def place_suggestion_widget(s: dict, ccy: str) -> None:
    """One-click "place paper bet" for a suggestion, pre-filled with the engine's
    suggested fractional-Kelly stake and the offered price.

    Unlike :func:`place_bet_widget` (which re-derives a stake from the sidebar
    strategy), this honours the suggestion's own value-sized stake — the brief's
    "place paper bet using the suggested stake". Still routes through
    ``place_paper_bet`` so the started-race / double-bet / stop-loss guards apply.
    """
    tracker = get_tracker()
    horse_id = s.get("horse_id")
    horse_name = s.get("horse_name") or "—"
    venue = s.get("venue")
    race_time = s.get("race_time")
    race_id = s.get("race_slug") or ""
    offered = s.get("offered_odds") or 1.01
    rec_stake = s.get("suggested_stake")
    key = f"sg:{race_id}:{horse_id}"

    markets = ["Win"] + (["Each-way"] if s.get("each_way_available") else [])
    started = BetTracker.race_has_started(race_time)

    with st.expander(f"＋ Place paper bet — {horse_name}", expanded=False):
        if started:
            st.caption("Race has started — betting closed.")
            return
        with st.form(key=f"sgform-{key}", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            market = c1.selectbox("Market", markets, key=f"sgmkt-{key}")
            bet_type = "each_way" if market == "Each-way" else "win"
            odds = c2.number_input(
                "Odds taken", min_value=1.01,
                value=float(offered) if offered else 1.01,
                step=0.10, format="%.2f", key=f"sgodds-{key}",
            )
            stake = c3.number_input(
                "Stake (suggested)", min_value=0.01,
                value=float(round(rec_stake, 2)) if rec_stake else 10.0,
                step=1.0, format="%.2f", key=f"sgstk-{key}",
                help="Pre-filled with the engine's fractional-Kelly suggested stake.",
            )
            edge = s.get("edge")
            tier = s.get("tier") or "—"
            st.caption(f"{tier} · bankroll {ccy}{tracker.bankroll:,.2f}"
                       + (f" · model edge {edge * 100:+.0f}pp" if edge is not None else ""))
            submitted = st.form_submit_button(f"Place {market.lower()} bet")

        if submitted:
            try:
                bet_id = tracker.place_paper_bet(
                    horse_name, odds, stake, bet_type=bet_type,
                    race_id=race_id, horse_id=str(horse_id) if horse_id else None,
                    venue=venue, race_time=race_time,
                    won_prob=s.get("model_prob"), value_edge=edge,
                    strategy=_current_strategy(),
                    notes=f"suggestion:{tier}",
                )
            except DuplicateBetError:
                st.warning(f"You already have a {bet_type.replace('_', '-')} bet on "
                           f"{horse_name} in this race.")
            except RaceStartedError:
                st.warning("Race has started — betting closed.")
            except StopLossError as exc:
                st.error(str(exc))
            else:
                st.success(f"Paper bet #{bet_id} placed: {ccy}{stake:,.2f} on "
                           f"{horse_name} @ {odds:.2f}.")
                st.rerun()


def settle_outcomes() -> tuple[Optional[str], ...]:
    return ("win", "place", "lose", "void")
