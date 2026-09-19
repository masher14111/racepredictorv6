"""Paper Betting — the virtual-bankroll practice ledger.

A clearly-labelled **paper / practice** surface (no real money): set a virtual
bankroll, review open paper bets, auto-settle them from scraped results, and read
the running P&L / ROI / CLV. Bets are placed from the race-detail page's per-runner
"Place paper bet" action; this page is the bankroll + settlement cockpit.

Prompt 21 visualises this ledger (equity curve, CLV distribution); the tracker
already exposes ``pl_series`` / ``bankroll_history`` / ``breakdown`` for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now  # noqa: E402
from ui import _components as C  # noqa: E402
from ui._betting import PAPER_NOTICE, get_tracker, settle_outcomes  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Paper Betting · MASHR’s Predictor v6",
                   page_icon="🏇", layout="wide", initial_sidebar_state="auto")

inject_design()



def _fmt_money(v: float) -> str:
    return f"{v:,.2f}"


def _bankroll_controls(tracker) -> None:
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)
        st.markdown("### Virtual bankroll")
        with st.form("set-bankroll", clear_on_submit=False):
            amount = st.number_input(
                "Set bankroll", min_value=1.0, value=float(round(tracker.bankroll, 2)),
                step=50.0, format="%.2f", label_visibility="collapsed",
            )
            if st.form_submit_button("Set bankroll"):
                tracker.set_bankroll(float(amount))
                st.success(f"Bankroll set to {_fmt_money(amount)}.")
                st.rerun()
        st.caption("Resets the baseline before your first bet; afterwards logs an "
                   "adjustment so the equity curve stays continuous.")
        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")


def _kpi_rail(summary: dict) -> None:
    pl = summary["total_profit"]
    cards = [
        C.kpi("Bankroll", f"{_fmt_money(summary['bankroll'])}",
              f"start {_fmt_money(summary['initial_bankroll'])}"),
        C.kpi("Profit / Loss", f"{pl:+,.2f}", "settled bets",
              tone="value" if pl > 0 else ""),
        C.kpi("ROI", f"{summary['roi_pct']:+.1f}%", "return on stake",
              tone="value" if summary["roi_pct"] > 0 else ""),
        C.kpi("Open bets", str(summary["pending_bets"]), "awaiting result"),
    ]
    clv = summary.get("avg_clv_pct")
    if clv is not None:
        cards.append(C.kpi("Avg CLV", f"{clv:+.1f}%", "vs closing price",
                           tone="value" if clv > 0 else ""))
    st.markdown(C.kpi_rail(cards), unsafe_allow_html=True)


def _open_bets(tracker) -> None:
    pending = tracker.pending_bets()
    st.markdown(C.section("Open bets", f"{len(pending)} awaiting settlement"),
                unsafe_allow_html=True)
    if not pending:
        st.markdown(C.empty_state("No open paper bets",
                                  "Place bets from a race's "
                                  '<a href="Race_Detail" target="_self">runner card</a>.'),
                    unsafe_allow_html=True)
        return
    for bet in pending:
        label = (f"{bet['horse_name']} · {bet.get('venue') or '—'} · "
                 f"{bet['bet_type'].replace('_', '-')} @ {bet['odds_decimal']:.2f} "
                 f"· stake {_fmt_money(bet['stake'])}")
        with st.expander(label, expanded=False):
            with st.form(f"settle-{bet['id']}", clear_on_submit=False):
                c1, c2, c3 = st.columns([1, 1, 1])
                outcome = c1.selectbox("Outcome", settle_outcomes(),
                                       key=f"out-{bet['id']}")
                closing = c2.number_input(
                    "Closing odds (SP)", min_value=0.0,
                    value=float(bet["odds_decimal"]), step=0.10, format="%.2f",
                    key=f"close-{bet['id']}",
                    help="Used to record CLV (price taken vs closing price).",
                )
                if c3.form_submit_button("Settle manually"):
                    tracker.settle_bet(bet["id"], outcome,
                                       closing_odds=closing or None)
                    st.success(f"Bet #{bet['id']} settled as {outcome}.")
                    st.rerun()


def _settled_history(tracker) -> None:
    settled = tracker.all_bets(settled_only=True)
    st.markdown(C.section("Settled history", f"{len(settled)} resolved"),
                unsafe_allow_html=True)
    if not settled:
        st.caption("No settled bets yet.")
        return
    df = pd.DataFrame(settled)
    cols = [c for c in ("settled_at", "horse_name", "venue", "bet_type",
                        "odds_decimal", "closing_odds", "clv_pct", "stake",
                        "outcome", "profit") if c in df.columns]
    view = df[cols].sort_values("settled_at", ascending=False).reset_index(drop=True)
    st.dataframe(view, width="stretch", hide_index=True)


def main() -> None:
    tracker = get_tracker()
    _bankroll_controls(tracker)

    st.markdown(C.app_bar("Paper betting", now().strftime("%A, %d %B %Y"), ""),
                unsafe_allow_html=True)
    st.markdown(PAPER_NOTICE, unsafe_allow_html=True)

    summary = tracker.summary()
    if summary["stop_loss_active"]:
        st.error("Stop-loss active — bankroll is below the floor; new bets are blocked.")
    _kpi_rail(summary)

    left, right = st.columns([1, 1])
    with left:
        if st.button("Settle from results", help="Match open bets to scraped "
                     "race results and settle won/lost/placed/void automatically."):
            from utils.bet_settlement import settle_from_storage
            settled = settle_from_storage(tracker)
            if settled:
                st.success(f"Settled {len(settled)} bet(s) from results.")
            else:
                st.info("No open bets could be settled yet — results may not be "
                        "scraped for those races.")
            st.rerun()
    with right:
        if st.button("Export bets to CSV", help="Write the full bet ledger to "
                     "data/bets.csv."):
            out = tracker.export_csv()
            st.success(f"Exported to {out}.")

    _open_bets(tracker)
    _settled_history(tracker)


main()
