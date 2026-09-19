"""Today's suggestions — the curated "what should I bet on today?" view.

The dashboard's action surface: instead of making you read every racecard, this
page runs the value layer over the day's scored races, tiers and ranks the
handful of genuinely value picks (Strong before Lean), explains each in plain
English from the model's own SHAP drivers, and lets you place a one-click paper
bet at the engine's suggested fractional-Kelly stake.

It curates honestly — a race with no edge contributes nothing, and when the whole
day has no value the page says so rather than inventing a pick. All gating
(de-vig, longshot/odds-on band, EV floor, support) comes from the backtester-
validated ``value:`` config via :func:`models.suggestions.suggest_from_cache`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now  # noqa: E402
from utils.currency import currency_for_venue  # noqa: E402
from models.suggestions import suggest_from_cache  # noqa: E402
from ui import _components as C  # noqa: E402
from ui._betting import PAPER_NOTICE, get_tracker, place_suggestion_widget  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Today's Suggestions · MASHR’s Predictor v6",
                   page_icon="🏇", layout="wide", initial_sidebar_state="auto")

inject_design()



@st.cache_data(ttl=60, show_spinner=False)
def _book(bankroll: float) -> dict:
    """Curated suggestion book as a plain dict (cache-friendly). Keyed on the
    live bankroll so stakes track the tracker's current balance."""
    return suggest_from_cache(bankroll=bankroll).to_dict()


def _ccy(venue: str) -> str:
    return "£" if currency_for_venue(venue) == "GBP" else "€"


def _kpi_rail(book: dict) -> None:
    n = len(book.get("suggestions") or [])
    cards = [
        C.kpi("Suggestions", str(n), "actionable today",
              tone="value" if n else ""),
        C.kpi("Strong", str(book.get("n_strong", 0)), "all bars cleared",
              tone="value" if book.get("n_strong") else ""),
        C.kpi("Lean", str(book.get("n_lean", 0)), "value, lower conviction"),
        C.kpi("Races with value", f"{book.get('n_races_with_value', 0)}",
              f"of {book.get('n_races', 0)} scanned"),
    ]
    st.markdown(C.kpi_rail(cards), unsafe_allow_html=True)


def _why_html(s: dict) -> str:
    """Render the suggestion's positive SHAP drivers as "why" bars (or "" when the
    feature store had no row for the runner — the rationale text still stands)."""
    drivers = s.get("rationale_drivers") or []
    if not drivers:
        return ""
    return C.why_drivers({"top_positive": drivers, "top_negative": []})


def _sidebar(tracker) -> None:
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)
        st.markdown("### Today's suggestions")
        st.caption("Curated value picks from the price-free calibrated model. "
                   "Strong clears every bar; Lean is value with lower conviction.")
        st.markdown(f"**Bankroll** {tracker.bankroll:,.2f}")
        st.caption("Suggested stakes are fractional-Kelly against this bankroll. "
                   "Adjust it on the Paper Betting page.")
        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")
        st.page_link("pages/9_Paper_Betting.py", label="Paper betting ledger →")


def main() -> None:
    tracker = get_tracker()
    _sidebar(tracker)

    st.markdown(
        C.app_bar("Today's suggestions", now().strftime("%A, %d %B %Y"),
                  C.trust_badge("Model", "v3nf price-free", "ok")),
        unsafe_allow_html=True,
    )
    st.markdown(PAPER_NOTICE, unsafe_allow_html=True)

    if tracker.summary().get("stop_loss_active"):
        st.error("Stop-loss active — bankroll is below the floor; new bets are "
                 "blocked until you reset it on the Paper Betting page.")

    # The value scan over the day's racecards is the slow path on a cold cache;
    # paint a skeleton into a placeholder so the wait reads as working, not blank.
    _loading = st.empty()
    _loading.markdown(C.loading_skeleton("Scanning today’s races for value…", n_cards=2),
                      unsafe_allow_html=True)
    book = _book(round(float(tracker.bankroll), 2))
    _loading.empty()
    _kpi_rail(book)

    suggestions = book.get("suggestions") or []
    if not suggestions:
        scanned = book.get("n_races", 0)
        body = (
            f"Scanned {scanned} race{'s' if scanned != 1 else ''} and found no bet "
            "that clears the value gates today — no qualifying edge over the "
            "de-vigged market inside the trusted odds band. That's the honest "
            "answer: no value means no bet. Browse the full "
            '<a href="app.py" target="_self">racecards</a> if you want to look anyway.'
        )
        st.markdown(C.empty_state("No value bets today", body), unsafe_allow_html=True)
        return

    st.markdown(
        C.section("Ranked picks",
                  f"{len(suggestions)} suggestion{'s' if len(suggestions) != 1 else ''} "
                  "· Strong before Lean, then by edge"),
        unsafe_allow_html=True,
    )
    for s in suggestions:
        ccy = _ccy(s.get("venue") or "")
        st.markdown(C.suggestion_card(s, ccy, why_html=_why_html(s)),
                    unsafe_allow_html=True)
        place_suggestion_widget(s, ccy)


main()
