"""Yesterday's Bet Predictor — did the model make money on a completed day?

An honest one-day backtest: pick a settled racing day, score every runner **blind to the
result** with the same models the live UI uses, place the model's bets (the live value
picks, or its top win pick per race), stake a fixed amount, then settle each bet against
the real finishing position. The page reports the profit you'd have made.

It is deliberately honest about its data: finishing positions arrive a few days after the
races run, so "yesterday" is the most recent day that actually has results; and settled
rows only carry the starting price (SP), so bets are settled at SP (you'd usually take a
bigger board price, making these figures, if anything, conservative). See
:mod:`models.yesterday`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.yesterday import YesterdayConfig, run_yesterday  # noqa: E402
from ui import _components as C  # noqa: E402
from ui._design import inject_design  # noqa: E402
from ui._winrate import realized_rate_sub, realized_rate_value  # noqa: E402

st.set_page_config(page_title="Yesterday's Bet Predictor · MASHR’s Predictor v6",
                   page_icon="🏇", layout="wide", initial_sidebar_state="auto")

inject_design()


_MATRIX_PATH = _ROOT / "data" / "features" / "training.parquet"


@st.cache_data(ttl=600, show_spinner=False)
def _settled_days() -> list[str]:
    """Days that carry finishing positions — cheap read of just two columns."""
    if not _MATRIX_PATH.exists():
        return []
    try:
        df = pd.read_parquet(_MATRIX_PATH, columns=["race_date", "position"])
    except Exception:
        return []
    d = pd.to_datetime(df["race_date"], utc=True, errors="coerce")
    return sorted({str(x) for x in d[df["position"].notna()].dt.date.dropna().unique()})


@st.cache_data(ttl=600, show_spinner=False)
def _run(strategy: str, bet_type: str, stake: float, target_date: str,
         bankroll: float) -> dict:
    """Run the settlement and return a plain dict (cache-friendly)."""
    res = run_yesterday(YesterdayConfig(
        strategy=strategy, bet_type=bet_type, stake=stake,
        target_date=target_date or None, bankroll=bankroll,
    ))
    return {
        "date": res.date, "available_dates": res.available_dates,
        "bets": res.bets, "summary": res.summary, "n_races": res.n_races,
        "n_value_races": res.n_value_races, "has_results": res.has_results,
        "message": res.message,
    }


def _sidebar() -> dict:
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)
        st.markdown("### Yesterday's Bet Predictor")
        st.caption("Replays the model's bets on a completed day and settles them against "
                   "the real results — an honest, out-of-sample profit check.")

        strat_label = st.radio(
            "What to bet", ["Value picks only", "Top win pick / race"], index=0,
            help="Value picks = exactly what Today's Suggestions would have shown "
                 "(price-free model edge over the de-vigged market). Top pick = the "
                 "model's highest win-probability runner in every race.",
        )
        strategy = "value" if strat_label.startswith("Value") else "top_pick"

        bt_label = st.radio("Bet type", ["Win", "Each-way"], index=0)
        bet_type = "win" if bt_label == "Win" else "each_way"

        stake = float(st.number_input("Stake per bet (€)", min_value=1.0, max_value=1000.0,
                                      value=10.0, step=1.0))

        days = _settled_days()
        if days:
            date = st.selectbox("Settled day", options=list(reversed(days)), index=0,
                                help="Most recent day with results is selected by default. "
                                     "Earlier days are available because results feeds lag "
                                     "the racing calendar by a few days.")
        else:
            date = ""

        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")
        st.page_link("pages/11_Todays_Suggestions.py", label="Today's suggestions →")
        st.page_link("pages/10_Dashboard.py", label="Performance dashboard →")

    return {"strategy": strategy, "bet_type": bet_type, "stake": stake, "date": date}


def _kpi_rail(summary: dict) -> None:
    profit = float(summary.get("total_profit", 0.0))
    roi = float(summary.get("roi_pct", 0.0))
    tone = "value" if profit > 0 else ""
    cards = [
        C.kpi("Bets placed", str(summary.get("n_bets", 0)),
              f"{summary.get('n_winners', 0)} winners"),
        C.kpi("Win rate (settled)",
              realized_rate_value(int(summary.get("n_winners", 0)),
                                  int(summary.get("n_bets", 0))),
              realized_rate_sub(int(summary.get("n_winners", 0)),
                                int(summary.get("n_bets", 0)),
                                f"place {summary.get('place_rate', 0.0):.1f}%")),
        C.kpi("Total staked", f"€{summary.get('total_staked', 0.0):,.2f}", "flat per bet"),
        C.kpi("Profit / loss", f"€{profit:+,.2f}",
              f"bankroll → €{summary.get('final_bankroll', 0.0):,.2f}", tone=tone),
        C.kpi("ROI", f"{roi:+.1f}%", "return on stake", tone=tone),
    ]
    st.markdown(C.kpi_rail(cards), unsafe_allow_html=True)


def _bets_table(bets: list[dict]) -> None:
    df = pd.DataFrame(bets)
    cols = {
        "venue": "Venue", "horse_name": "Horse", "decimal_odds": "SP",
        "model_win_prob": "Model win%", "expected_value": "EV",
        "position": "Finish", "outcome": "Result", "stake": "Stake",
        "profit": "Profit",
    }
    view = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    if "Model win%" in view.columns:
        view["Model win%"] = (pd.to_numeric(view["Model win%"], errors="coerce") * 100).round(1)
    if "EV" in view.columns:
        view["EV"] = pd.to_numeric(view["EV"], errors="coerce").round(3)
    view = view.sort_values("Profit", ascending=False).reset_index(drop=True)
    st.dataframe(
        view, width="stretch", hide_index=True,
        column_config={
            "SP": st.column_config.NumberColumn(format="%.2f"),
            "Stake": st.column_config.NumberColumn(format="€%.2f"),
            "Profit": st.column_config.NumberColumn(format="€%.2f"),
        },
    )


def main() -> None:
    ctrl = _sidebar()

    st.markdown(
        C.app_bar("Yesterday's Bet Predictor", "Out-of-sample profit check on a settled day",
                  C.trust_badge("Settled at", "starting price (SP)", "info")),
        unsafe_allow_html=True,
    )

    if not ctrl["date"]:
        st.markdown(C.empty_state(
            "No settled results yet",
            "Finishing positions haven't been scraped for any day, so there's nothing to "
            "settle against. Run the results scrape, then come back.",
        ), unsafe_allow_html=True)
        return

    _loading = st.empty()
    _loading.markdown(C.loading_skeleton("Scoring the day blind and settling bets…", n_cards=2),
                      unsafe_allow_html=True)
    res = _run(ctrl["strategy"], ctrl["bet_type"], ctrl["stake"], ctrl["date"], 1000.0)
    _loading.empty()

    if not res["has_results"]:
        st.markdown(C.empty_state("Nothing to settle", res["message"] or "No bets."),
                    unsafe_allow_html=True)
        return

    strat_word = "value picks" if ctrl["strategy"] == "value" else "top win picks"
    st.markdown(
        C.app_bar(f"Results for {res['date']}",
                  f"{res['n_races']} races scanned · "
                  + (f"{res['n_value_races']} with a value bet" if ctrl["strategy"] == "value"
                     else "one pick per race"),
                  ""),
        unsafe_allow_html=True,
    )

    st.info(
        "**How to read this:** the model scored every runner without seeing the result, "
        f"then bet the {strat_word} ({'each-way' if ctrl['bet_type']=='each_way' else 'win'}) "
        f"at €{ctrl['stake']:.0f} a time, settled at the starting price. Results feeds lag "
        "the calendar by a few days, so the latest settleable day is "
        f"**{res['available_dates'][-1]}**, not literally yesterday. A single day is a tiny "
        "sample — judge the edge over many days on the Performance dashboard.",
        icon="🧪",
    )

    _kpi_rail(res["summary"])

    bets = res["bets"]
    if not bets:
        scanned = res["n_races"]
        body = (
            f"Scanned {scanned} race{'s' if scanned != 1 else ''} on {res['date']} and the "
            "model found no bet that clears the value gates — no qualifying edge over the "
            "de-vigged market. No value, no bet. Switch to “Top win pick / race” in the "
            "sidebar to force a pick in every race instead."
        )
        st.markdown(C.empty_state("No qualifying bets that day", body), unsafe_allow_html=True)
        return

    st.markdown(C.section("Settled bets", f"{len(bets)} bet{'s' if len(bets) != 1 else ''} "
                          "· sorted by profit"), unsafe_allow_html=True)
    _bets_table(bets)


main()
