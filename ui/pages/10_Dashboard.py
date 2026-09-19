"""Performance Dashboard — does the model actually work? (Prompt 21)

The project's honest scoreboard. The two **heroes** are the bankroll equity curve
and the A/E calibration-by-probability-bucket chart; CLV (did we beat the close?)
is the deciding signal. Supporting metrics — cumulative profit, ROI/yield,
hit-rate, max drawdown — sit below, deliberately *not* as a wall of SaaS
hero-number tiles.

Filter the paper-trading ledger by date range, market, odds band, and "model
picks only vs all bets", and compare it side by side with a saved walk-forward
backtest run. All chart/data logic lives in ``ui.dashboard`` (unit tested); this
file is the Streamlit assembly only. Themed to the Prompt-19 design system.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now  # noqa: E402
from ui import _components as C  # noqa: E402
from ui import dashboard as D  # noqa: E402
from ui._betting import get_tracker  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Performance · MASHR’s Predictor v6",
                   page_icon="📈", layout="wide", initial_sidebar_state="auto")

inject_design()


_CCY = "€"
_PLOTLY_CFG = {"displaylogo": False,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def _fmt(v, suffix="", prefix="", dp=2, na="—"):
    if v is None:
        return na
    return f"{prefix}{v:,.{dp}f}{suffix}"


# ── sidebar: filters + backtest selector ──────────────────────────────────────

def _sidebar(df_all):
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)

        st.markdown("### Date range")
        if not df_all.empty and df_all["placed_date"].notna().any():
            lo_default = df_all["placed_date"].dropna().min()
            hi_default = df_all["placed_date"].dropna().max()
        else:
            hi_default = date.today()
            lo_default = hi_default - timedelta(days=30)
        date_from = st.date_input("From", value=lo_default, key="dash_from",
                                  label_visibility="collapsed")
        date_to = st.date_input("To", value=hi_default, key="dash_to",
                                label_visibility="collapsed")
        if date_from > date_to:
            date_from, date_to = date_to, date_from

        st.markdown("### Market")
        markets = st.multiselect(
            "Market", options=[m[0] for m in D.MARKETS],
            default=[m[0] for m in D.MARKETS],
            format_func=lambda k: dict(D.MARKETS)[k],
            label_visibility="collapsed",
        )

        st.markdown("### Odds band")
        bands = st.multiselect(
            "Odds band", options=[b[0] for b in D.ODDS_BANDS],
            default=[b[0] for b in D.ODDS_BANDS], label_visibility="collapsed",
        )

        st.markdown("### Selection")
        picks_only = st.toggle("Model picks only", value=False,
                               help="Only bets the model flagged as positive value "
                                    "(edge > 0), vs every bet placed.")

        st.markdown("### Backtest")
        runs = D.list_backtest_runs()
        run_id = st.selectbox("Run", options=["(none)"] + runs,
                              index=1 if runs else 0, label_visibility="collapsed")
        run_id = None if run_id == "(none)" else run_id

        summary = D.load_backtest_summary(run_id) if run_id else None
        strategy = None
        if summary:
            strats = list((summary.get("strategies") or {}).keys())
            if strats:
                strategy = st.selectbox("Strategy", options=strats,
                                        label_visibility="collapsed")
            overlay = st.toggle("Overlay backtest on calibration", value=True)
        else:
            overlay = False

        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")
        st.page_link("pages/9_Paper_Betting.py", label="Paper betting ledger")

    return {
        "date_range": (date_from, date_to), "markets": markets, "bands": bands,
        "picks_only": picks_only, "run_id": run_id, "summary": summary,
        "strategy": strategy, "overlay": overlay,
    }


def _trust_badges(summary):
    """AUC + calibration ECE from the backtest model_metrics (the honest OOS read)."""
    if not summary:
        return ""
    m = summary.get("model_metrics") or {}
    badges = []
    if m.get("auc") is not None:
        badges.append(C.trust_badge("OOS AUC", f"{m['auc']:.3f}", tone="info"))
    if m.get("ece") is not None:
        tone = "ok" if m["ece"] <= 0.02 else "warn"
        badges.append(C.trust_badge("ECE", f"{m['ece']:.3f}", tone=tone))
    return "".join(badges)


def _kpi_rail(pm: dict, pending: int):
    """Compact metric rail — supporting cast, below the heroes (not the headline)."""
    roi = pm["roi_pct"]
    clv = pm["clv_pct_mean"]
    cards = [
        C.kpi("Net P&L", _fmt(pm["profit"], prefix=_CCY) if pm["n_bets"] else "—",
              f"{pm['n_bets']} settled · {pending} open",
              tone="value" if (pm["profit"] or 0) > 0 else ""),
        C.kpi("ROI / yield", _fmt(roi, suffix="%", dp=1) if roi is not None else "—",
              "profit ÷ staked", tone="value" if (roi or 0) > 0 else ""),
        C.kpi("Hit rate",
              _fmt(pm["hit_rate"] * 100, suffix="%", dp=0) if pm["hit_rate"] is not None else "—",
              "wins ÷ decided bets"),
        C.kpi("Avg CLV", _fmt(clv, suffix="%", dp=1) if clv is not None else "—",
              "vs closing price", tone="value" if (clv or 0) > 0 else ""),
        C.kpi("Beat close",
              _fmt(pm["beat_close_rate"] * 100, suffix="%", dp=0)
              if pm["beat_close_rate"] is not None else "—", "of settled bets"),
        C.kpi("Max drawdown", _fmt(pm["max_drawdown_pct"], suffix="%", dp=1),
              "peak-to-trough"),
    ]
    st.markdown(C.kpi_rail(cards), unsafe_allow_html=True)


def main():
    tracker = get_tracker()
    df_all = D.load_paper_bets(tracker)
    f = _sidebar(df_all)

    st.markdown(
        C.app_bar(
            'Performance <span class="v">— does it work?</span>',
            now().strftime("%A, %d %B %Y"),
            _trust_badges(f["summary"]),
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rp-paper-note"><span class="dot"></span>'
        '<b>Paper / Practice — no real money.</b> Equity & calibration below judge '
        'the model on simulated bets and the leak-free backtest.</div>',
        unsafe_allow_html=True,
    )

    # Latest GO / NO-GO verdict, so the bankroll view is read in context.
    from ui import model_honesty as MH  # noqa: E402
    MH.render_verdict_strip()

    df = D.apply_filters(df_all, date_range=f["date_range"], markets=f["markets"],
                         odds_bands=f["bands"], picks_only=f["picks_only"])
    settled = D.settled_only(df)
    initial = float(tracker.initial_bankroll)
    pm = D.paper_metrics(settled, initial)
    pending = int((df["outcome"].isna()).sum()) if not df.empty else 0

    # ── HERO 1 — equity curve ────────────────────────────────────────────────
    st.markdown(C.section("Bankroll equity curve",
                          f"{pm['n_bets']} settled bets · {_fmt(pm['profit'], prefix=_CCY)} net"),
                unsafe_allow_html=True)
    st.plotly_chart(D.equity_curve_fig(settled, initial, _CCY),
                    width="stretch", config=_PLOTLY_CFG, key="hero_equity")

    # ── HERO 2 — calibration A/E ─────────────────────────────────────────────
    bt_ae = (f["summary"] or {}).get("ae_by_prob") or [] if f["overlay"] else []
    paper_ae = D.paper_ae_rows(settled)
    st.markdown(C.section("Calibration — A/E by probability bucket",
                          "are the model's win-probabilities honest?"),
                unsafe_allow_html=True)
    st.plotly_chart(D.calibration_ae_fig(paper_ae, bt_ae),
                    width="stretch", config=_PLOTLY_CFG, key="hero_calib")
    st.caption("A/E ≈ 1 is perfectly calibrated. Below 1 = over-confident "
               "(the model claims more winners than it gets); above 1 = under-confident. "
               "Backtest bars come from the full out-of-sample run; paper bars from your "
               "settled bets.")

    # ── supporting metric rail (below the heroes) ────────────────────────────
    _kpi_rail(pm, pending)

    # ── cumulative profit + CLV ──────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(C.section("Cumulative profit"), unsafe_allow_html=True)
        st.plotly_chart(D.cumulative_profit_fig(settled, _CCY),
                        width="stretch", config=_PLOTLY_CFG, key="cum_pl")
    with c2:
        st.markdown(C.section("CLV distribution", "beat the closing price?"),
                    unsafe_allow_html=True)
        st.plotly_chart(D.clv_fig(settled), width="stretch",
                        config=_PLOTLY_CFG, key="clv")

    # ── backtest comparison ──────────────────────────────────────────────────
    if f["summary"]:
        st.markdown(C.section("Paper vs backtest",
                              f"run {f['run_id']}"), unsafe_allow_html=True)
        bt_ledger = D.load_backtest_ledger(f["run_id"], f["strategy"]) \
            if f["strategy"] else pd.DataFrame()
        st.plotly_chart(
            D.normalized_equity_fig(settled, initial, bt_ledger,
                                    bt_label=f"Backtest · {f['strategy'] or ''}"),
            width="stretch", config=_PLOTLY_CFG, key="norm_equity")

        cmp = D.backtest_compare_table(f["summary"])
        if not cmp.empty:
            st.caption("Every walk-forward strategy in this run, sorted by yield. "
                       "Per [[model-14-backtest]], judge on CLV & A/E, not raw ROI.")
            st.dataframe(cmp, width="stretch", hide_index=True)
    else:
        st.markdown(C.section("Paper vs backtest"), unsafe_allow_html=True)
        st.markdown(
            C.empty_state("No backtest selected",
                          "Run one with <code>python -m backtest</code>, then pick it "
                          "in the sidebar to compare its equity and calibration."),
            unsafe_allow_html=True)

    m = (f["summary"] or {}).get("model_metrics") or {}
    st.markdown(
        C.footer(
            f'{C.status_dot("ok")}Paper bankroll <span class="mono">{_CCY}'
            f'{pm["final_bankroll"]:,.2f}</span>',
            f'OOS AUC {m.get("auc", "—")} · ECE {m.get("ece", "—")}',
        ),
        unsafe_allow_html=True,
    )


main()
