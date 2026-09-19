"""ui/performance_dashboard.py — Historical performance dashboard.

Interactive P&L over time, ROI, win rate, each-way success rate.
Filters: date range, bet type, odds range, venue.
Export: PDF summary via ReportLab.

Launch:
    streamlit run ui/performance_dashboard.py
"""
from __future__ import annotations

import io
import re
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from utils import config_loader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import local_day, local_month, now, to_local
from utils.bet_tracker import BetTracker
from ui._winrate import realized_rate_sub, realized_rate_value

st.set_page_config(
    page_title="Performance Dashboard — MASHR’s Predictor v6",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── design tokens ─────────────────────────────────────────────────────────────
# Read from the shared design system rather than re-declared here. This module
# used to carry its own nine-hex copy of the palette, which had already drifted
# from the locked semantics (its `lose` was #dc2626, not --danger). The page is
# dark now, so it takes the dark binding.
from ui import _components as C  # noqa: E402
from ui._design import (palette as _palette, plotly_layout,
                        ordinal as _ordinal, tint as _tint)

_P          = _palette("dark")
_ACCENT     = _P["info"]
_SUCCESS    = _P["value"]
_WARN       = _P["amber"]
_DANGER     = _P["danger"]
_MUTED      = _P["muted"]
_FG         = _P["ink"]
_FG2        = _P["ink_2"]
_BORDER     = _P["border"]
_SURFACE    = _P["surface"]
_BG         = _P["bg"]

_OUTCOME_COLOURS = {
    "win":     _SUCCESS,
    "place":   _ACCENT,
    "lose":    _DANGER,
    "pending": _MUTED,
}


# ── design system ─────────────────────────────────────────────
# Was injected from inside main(), after page content; it belongs beside set_page_config like every other page.
# The rp-* classes this page emits are mapped onto the tokens in ui/_design.py
# (see its "legacy page class map" section).
from ui._design import inject_design  # noqa: E402

inject_design()

# ── chart layout defaults ─────────────────────────────────────────────────────

def _chart_layout(**kwargs) -> dict:
    """This page's charts, themed from the shared system. `density="compact"`
    carries the tight margins and small type these in-card charts need; the
    colours come from the light token binding."""
    return plotly_layout(mode="dark", density="compact", **kwargs)


def _plotly_cfg() -> dict:
    return {
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "select2d", "lasso2d", "autoScale2d", "toggleSpikelines",
        ],
        "displaylogo": False,
    }


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _get_tracker() -> BetTracker:
    bt = config_loader.get("bet_tracker", {}) or {}
    return BetTracker(
        initial_bankroll=bt.get("initial_bankroll", 1000.0),
        flat_stake=bt.get("flat_stake", 10.0),
    )


@st.cache_data(ttl=30)
def _load_bets() -> pd.DataFrame:
    """Load all bets into a DataFrame with parsed dates and derived columns."""
    tracker = _get_tracker()
    bets = tracker.all_bets()
    if not bets:
        return pd.DataFrame()

    df = pd.DataFrame(bets)
    for col in ("placed_at", "settled_at", "race_time"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in ("stake", "odds_decimal", "profit", "gross_return",
                "composite_score", "won_prob"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "placed_at" in df.columns:
        # Bucket by Dublin time, not UTC — see utils.timezone.local_day.
        df["placed_date"] = local_day(df["placed_at"])
        df["placed_month"] = local_month(df["placed_at"]).astype(str)

    return df


# ── formatters ────────────────────────────────────────────────────────────────

def _fmt_dt(ts, fmt: str = "%d %b %H:%M") -> str:
    if ts is None or (hasattr(ts, "isnull") and ts.isnull()):
        return "—"
    try:
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        if hasattr(ts, "tzinfo") and ts.tzinfo:
            ts = to_local(ts)
        return ts.strftime(fmt)
    except Exception:
        return str(ts)


def _profit_colour(val: float) -> str:
    return _SUCCESS if val >= 0 else _DANGER


# ── filtering ─────────────────────────────────────────────────────────────────

def _apply_filters(
    df: pd.DataFrame,
    date_range: tuple[date, date],
    bet_types: list[str],
    odds_range: tuple[float, float],
    venues: list[str],
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "placed_date" in out.columns:
        out = out[
            (out["placed_date"] >= date_range[0]) &
            (out["placed_date"] <= date_range[1])
        ]
    if bet_types and "bet_type" in out.columns:
        out = out[out["bet_type"].isin(bet_types)]
    if "odds_decimal" in out.columns:
        valid_odds = out["odds_decimal"].notna()
        out = out[
            (~valid_odds) |
            (
                (out["odds_decimal"] >= odds_range[0]) &
                (out["odds_decimal"] <= odds_range[1])
            )
        ]
    if venues and "venue" in out.columns:
        out = out[out["venue"].isin(venues)]
    return out.reset_index(drop=True)


# ── sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(tracker: BetTracker, df_all: pd.DataFrame) -> dict:
    with st.sidebar:
        st.markdown(
            C.brand_lockup(),
            unsafe_allow_html=True,
        )

        summ = tracker.summary()
        br, init = summ["bankroll"], summ["initial_bankroll"]
        delta_pct = (br - init) / init * 100 if init else 0.0
        col = _profit_colour(br - init)

        st.markdown("### Bankroll")
        st.markdown(
            f'<div style="font-size:26px;font-weight:700;color:{col};'
            f'font-variant-numeric:tabular-nums">€{br:,.2f}</div>'
            f'<div style="font-size:12px;color:var(--muted);margin-top:2px">'
            f'{delta_pct:+.1f}% vs €{init:,.2f}</div>',
            unsafe_allow_html=True,
        )
        if summ["stop_loss_active"]:
            st.markdown(
                f'<div style="background:{_tint(_DANGER, .15)};border:1px solid {_DANGER};'
                'border-radius:6px;padding:8px 10px;margin-top:8px;font-size:12px;'
                'color:var(--danger);font-weight:500">⛔ Stop-loss active</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("### Filters")

        # Date range
        st.markdown("**Date Range**")
        min_date = date.today() - timedelta(days=365)
        max_date = date.today()
        if not df_all.empty and "placed_date" in df_all.columns:
            valid_dates = df_all["placed_date"].dropna()
            if len(valid_dates):
                min_date = valid_dates.min()
                max_date = valid_dates.max()

        date_from = st.date_input("From", value=min_date, min_value=min_date,
                                   max_value=max_date, key="date_from",
                                   label_visibility="collapsed")
        date_to = st.date_input("To", value=max_date, min_value=min_date,
                                 max_value=max_date, key="date_to",
                                 label_visibility="collapsed")
        if date_from > date_to:
            date_from, date_to = date_to, date_from

        # Bet type (win / each_way)
        st.markdown("**Bet Type**")
        available_types = (
            sorted(df_all["bet_type"].dropna().unique().tolist())
            if not df_all.empty and "bet_type" in df_all.columns
            else ["win", "each_way"]
        )
        bet_types = st.multiselect(
            "Bet type",
            options=available_types,
            default=available_types,
            label_visibility="collapsed",
        )

        # Odds range
        st.markdown("**Odds Range**")
        min_odds, max_odds = 1.01, 100.0
        if not df_all.empty and "odds_decimal" in df_all.columns:
            valid_odds = df_all["odds_decimal"].dropna()
            if len(valid_odds):
                min_odds = float(valid_odds.min())
                max_odds = float(valid_odds.max())
        min_odds = max(1.01, min_odds)
        max_odds = max(min_odds + 0.01, max_odds)

        odds_lo, odds_hi = st.slider(
            "Decimal odds range",
            min_value=round(min_odds, 2),
            max_value=round(max_odds, 2),
            value=(round(min_odds, 2), round(max_odds, 2)),
            step=0.25,
            format="%.2f",
            label_visibility="collapsed",
        )

        # Venue
        st.markdown("**Venue**")
        all_venues: list[str] = []
        if not df_all.empty and "venue" in df_all.columns:
            all_venues = sorted(df_all["venue"].dropna().unique().tolist())
        venues = st.multiselect(
            "Venue", options=all_venues, default=all_venues,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### Export")
        export_pdf = st.button("Export PDF Summary")

    return {
        "date_range": (date_from, date_to),
        "bet_types": bet_types,
        "odds_range": (odds_lo, odds_hi),
        "venues": venues,
        "export_pdf": export_pdf,
    }


# ── KPI strip ─────────────────────────────────────────────────────────────────

def _compute_kpis(df: pd.DataFrame, tracker: BetTracker) -> dict:
    """Compute KPIs from the filtered bets DataFrame."""
    settled = df[df["outcome"].notna()].copy() if not df.empty else pd.DataFrame()
    pending = df[df["outcome"].isna()].copy() if not df.empty else pd.DataFrame()
    ew = settled[settled["bet_type"] == "each_way"] if not settled.empty else pd.DataFrame()

    total_bets     = len(settled)
    total_staked   = float(settled["stake"].sum()) if not settled.empty else 0.0
    total_profit   = float(settled["profit"].sum()) if not settled.empty else 0.0
    roi_pct        = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    wins           = int((settled["outcome"] == "win").sum()) if not settled.empty else 0
    places         = int((settled["outcome"] == "place").sum()) if not settled.empty else 0
    win_rate       = wins / total_bets if total_bets else 0.0
    place_rate     = (wins + places) / total_bets if total_bets else 0.0

    ew_total       = len(ew)
    ew_success     = int(ew["outcome"].isin(["win", "place"]).sum()) if not ew.empty else 0
    ew_rate        = ew_success / ew_total if ew_total else 0.0

    avg_odds       = float(settled["odds_decimal"].mean()) if not settled.empty else 0.0
    pending_count  = len(pending)

    # max drawdown from bankroll_history
    max_dd_pct = tracker.summary().get("max_drawdown_pct", 0.0)

    return {
        "total_bets":    total_bets,
        "pending":       pending_count,
        "total_staked":  total_staked,
        "total_profit":  total_profit,
        "roi_pct":       roi_pct,
        "wins":          wins,
        "win_rate":      win_rate,
        "place_rate":    place_rate,
        "ew_rate":       ew_rate,
        "ew_total":      ew_total,
        "avg_odds":      avg_odds,
        "max_dd_pct":    max_dd_pct,
    }


def _render_kpis(kpis: dict) -> None:
    profit = kpis["total_profit"]
    roi    = kpis["roi_pct"]

    def _kpi_html(label: str, value: str, sub: str = "", cls: str = "") -> str:
        sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
        return (
            f'<div class="kpi">'
            f'<div class="kpi-lbl">{label}</div>'
            f'<div class="kpi-val {cls}">{value}</div>'
            f'{sub_html}'
            f'</div>'
        )

    tiles = [
        _kpi_html("Total P&L",
                  f"{'+'if profit>=0 else ''}€{profit:,.2f}",
                  f"ROI {roi:+.1f}%",
                  "kpi-pos" if profit >= 0 else "kpi-neg"),
        _kpi_html("Win Rate (settled)",
                  realized_rate_value(kpis.get("wins", 0), kpis["total_bets"]),
                  realized_rate_sub(kpis.get("wins", 0), kpis["total_bets"],
                                    f"place rate {kpis['place_rate']*100:.0f}%")),
        _kpi_html("E/W Success",
                  f"{kpis['ew_rate']*100:.0f}%",
                  f"{kpis['ew_success'] if 'ew_success' in kpis else kpis['ew_total']} E/W bets"
                  if kpis["ew_total"] else "No E/W bets",
                  "kpi-neu" if kpis["ew_total"] == 0 else ""),
        _kpi_html("Bets Settled",
                  str(kpis["total_bets"]),
                  f"{kpis['pending']} pending"),
        _kpi_html("Total Staked",
                  f"€{kpis['total_staked']:,.2f}",
                  f"Avg odds {kpis['avg_odds']:.2f}" if kpis["avg_odds"] else ""),
        _kpi_html("Max Drawdown",
                  f"{kpis['max_dd_pct']:.1f}%",
                  "",
                  "kpi-neg" if kpis["max_dd_pct"] > 15 else "kpi-neu"),
    ]
    # fix ew_success access since we compute it above
    tiles[2] = _kpi_html(
        "E/W Success",
        f"{kpis['ew_rate']*100:.0f}%",
        (f"{int(kpis['ew_rate']*kpis['ew_total'])}/{kpis['ew_total']} E/W"
         if kpis["ew_total"] else "No E/W bets"),
    )

    st.markdown(
        f'<div class="kpi-wrap">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


# ── P&L over time chart ───────────────────────────────────────────────────────

def _pl_over_time(df: pd.DataFrame) -> go.Figure:
    settled = (
        df[df["outcome"].notna()].copy()
        if not df.empty else pd.DataFrame()
    )
    if settled.empty or "settled_at" not in settled.columns:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=300))
        fig.add_annotation(text="No settled bets yet", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=_MUTED, size=13))
        return fig

    s = settled.dropna(subset=["settled_at"]).sort_values("settled_at").reset_index(drop=True)
    s["bet_num"] = range(1, len(s) + 1)
    s["cum_profit"] = s["profit"].cumsum()
    s["outcome_str"] = s["outcome"].fillna("pending")
    s["hover_dt"] = s["settled_at"].apply(lambda x: _fmt_dt(x, "%d %b %Y %H:%M"))
    s["horse_name"] = s["horse_name"].fillna("—")
    s["venue"] = s["venue"].fillna("—")
    s["odds_decimal"] = s["odds_decimal"].fillna(0)

    fig = go.Figure()

    # shaded area under the cumulative P&L line
    fig.add_trace(go.Scatter(
        x=s["bet_num"],
        y=s["cum_profit"],
        mode="none",
        fill="tozeroy",
        fillcolor=_tint(_ACCENT, 0.08),
        showlegend=False,
        hoverinfo="skip",
    ))

    # cumulative P&L line
    fig.add_trace(go.Scatter(
        x=s["bet_num"],
        y=s["cum_profit"],
        mode="lines",
        line=dict(color=_ACCENT, width=2.5),
        name="Cumulative P&L",
        hovertemplate=(
            "<b>Bet #%{x}</b><br>"
            "%{customdata[0]}<br>"
            "%{customdata[1]} @ %{customdata[2]:.2f}<br>"
            "Running P&L: <b>%{y:+.2f}€</b>"
            "<extra></extra>"
        ),
        customdata=list(zip(s["hover_dt"], s["horse_name"],
                            s["odds_decimal"])),
    ))

    # per-bet outcome markers
    for outcome, colour in _OUTCOME_COLOURS.items():
        mask = s["outcome_str"] == outcome
        if not mask.any():
            continue
        sub = s[mask]
        fig.add_trace(go.Scatter(
            x=sub["bet_num"],
            y=sub["cum_profit"],
            mode="markers",
            name=outcome.title(),
            marker=dict(color=colour, size=7, symbol="circle",
                        line=dict(color=_SURFACE, width=1.5)),
            hovertemplate=(
                f"<b>{outcome.title()}</b>: %{{customdata[0]}}<br>"
                "P&L: <b>%{customdata[1]:+.2f}€</b>"
                "<extra></extra>"
            ),
            customdata=list(zip(sub["horse_name"],
                                sub["profit"].fillna(0))),
        ))

    # zero reference line
    fig.add_hline(y=0, line_dash="dot", line_color=_MUTED, line_width=1)

    fig.update_layout(
        **_chart_layout(
            height=300,
            xaxis_title="Bet #",
            yaxis_title="Cumulative P&L (€)",
            yaxis_tickprefix="€",
            yaxis_tickformat="+,.0f",
        )
    )
    return fig


# ── monthly P&L bar chart ─────────────────────────────────────────────────────

def _monthly_pl(df: pd.DataFrame) -> go.Figure:
    settled = (
        df[df["outcome"].notna()].copy()
        if not df.empty else pd.DataFrame()
    )
    if settled.empty or "placed_month" not in settled.columns:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=260))
        fig.add_annotation(text="No settled bets", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=_MUTED, size=13))
        return fig

    monthly = (
        settled.groupby("placed_month")
        .agg(profit=("profit", "sum"), bets=("id", "count"),
             staked=("stake", "sum"))
        .reset_index()
    )
    monthly["roi"] = (monthly["profit"] / monthly["staked"] * 100).round(1)
    monthly["colour"] = monthly["profit"].apply(
        lambda x: _SUCCESS if x >= 0 else _DANGER
    )

    fig = go.Figure(go.Bar(
        x=monthly["placed_month"],
        y=monthly["profit"],
        marker_color=monthly["colour"],
        marker_line_width=0,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "P&L: <b>%{y:+.2f}€</b><br>"
            "Bets: %{customdata[0]}<br>"
            "ROI: %{customdata[1]:+.1f}%"
            "<extra></extra>"
        ),
        customdata=list(zip(monthly["bets"], monthly["roi"])),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=_MUTED, line_width=1)
    fig.update_layout(
        **_chart_layout(
            height=260,
            bargap=0.3,
            xaxis_title=None,
            yaxis_title="P&L (€)",
            yaxis_tickprefix="€",
            yaxis_tickformat="+,.0f",
        )
    )
    return fig


# ── win rate by odds band ─────────────────────────────────────────────────────

def _win_rate_by_odds(df: pd.DataFrame) -> go.Figure:
    settled = (
        df[df["outcome"].notna()].copy()
        if not df.empty else pd.DataFrame()
    )
    if settled.empty or "odds_decimal" not in settled.columns:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=260))
        fig.add_annotation(text="No settled bets", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=_MUTED, size=13))
        return fig

    bands = [
        ("Fav (< 3.0)",   lambda r: r["odds_decimal"] < 3.0),
        ("Mid (3–8)",      lambda r: (r["odds_decimal"] >= 3.0) & (r["odds_decimal"] < 8.0)),
        ("Each-Way (8–20)",lambda r: (r["odds_decimal"] >= 8.0) & (r["odds_decimal"] < 20.0)),
        ("Longshot (20+)", lambda r: r["odds_decimal"] >= 20.0),
    ]
    rows = []
    for label, mask_fn in bands:
        sub = settled[mask_fn(settled)]
        if len(sub) == 0:
            continue
        bets = len(sub)
        wins = int((sub["outcome"] == "win").sum())
        places = int((sub["outcome"].isin(["win", "place"])).sum())
        profit = float(sub["profit"].sum())
        roi = profit / float(sub["stake"].sum()) * 100 if sub["stake"].sum() > 0 else 0.0
        rows.append(dict(band=label, bets=bets, wins=wins,
                         win_rate=wins/bets*100, place_rate=places/bets*100,
                         roi=roi, profit=profit))

    if not rows:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=260))
        fig.add_annotation(text="No data", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=_MUTED, size=13))
        return fig

    band_df = pd.DataFrame(rows)

    # Odds bands are ORDINAL (favourites → longshots), so the x-axis already
    # carries the order and colour must not re-encode it. Win% and the
    # place-ONLY remainder stack — an honest decomposition, since
    # win + place-only = win+place — replacing the old overlay, whose 25%-alpha
    # second bar was both hard to read and below contrast.
    ramp = _ordinal("dark")
    place_only = (band_df["place_rate"] - band_df["win_rate"]).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=band_df["band"], y=band_df["win_rate"], name="Win",
        marker_color=ramp[2], marker_line_width=0,
        text=[f"{v:.0f}%" for v in band_df["win_rate"]],
        textposition="inside", insidetextanchor="middle",
        textfont=dict(color=_SURFACE, size=10),
        hovertemplate="<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=band_df["band"], y=place_only, name="Place only",
        marker_color=ramp[0], marker_line_width=0,
        customdata=band_df["place_rate"],
        hovertemplate="<b>%{x}</b><br>Win+Place: %{customdata:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        **_chart_layout(
            height=260,
            barmode="stack",
            bargap=0.6,           # bars stay thin; the band keeps its air
            xaxis_title=None,
            yaxis_title="Rate (%)",
            yaxis_ticksuffix="%",
        )
    )
    # a 2px gap in the surface colour separates the two stacked segments
    fig.update_traces(marker_line_color=_SURFACE, marker_line_width=2,
                      selector=dict(type="bar"))
    return fig


# ── ROI trend (rolling 10-bet) ────────────────────────────────────────────────

def _roi_trend(df: pd.DataFrame) -> go.Figure:
    settled = (
        df[df["outcome"].notna()].copy()
        if not df.empty else pd.DataFrame()
    )
    if len(settled) < 3:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=260))
        fig.add_annotation(text="Need ≥3 settled bets for trend",
                           showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=0.5, font=dict(color=_MUTED, size=13))
        return fig

    s = settled.sort_values("settled_at").reset_index(drop=True)
    s["bet_num"] = range(1, len(s) + 1)
    window = min(10, max(3, len(s) // 5))
    s["roll_profit"] = s["profit"].rolling(window, min_periods=1).sum()
    s["roll_staked"] = s["stake"].rolling(window, min_periods=1).sum()
    s["roll_roi"] = (s["roll_profit"] / s["roll_staked"] * 100).where(
        s["roll_staked"] > 0, 0
    )

    colours = s["roll_roi"].apply(
        lambda x: _tint(_SUCCESS, 0.85) if x >= 0 else _tint(_DANGER, 0.85)
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s["bet_num"],
        y=s["roll_roi"],
        mode="lines+markers",
        line=dict(color=_ACCENT, width=2),
        marker=dict(size=5, color=colours),
        name=f"ROI (rolling {window})",
        hovertemplate=(
            "Bet #%{x}<br>"
            f"Rolling {window}-bet ROI: <b>%{{y:+.1f}}%</b>"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=_MUTED, line_width=1)
    fig.update_layout(
        **_chart_layout(
            height=260,
            xaxis_title="Bet #",
            yaxis_title=f"Rolling {window}-bet ROI (%)",
            yaxis_ticksuffix="%",
            yaxis_tickformat="+.0f",
        )
    )
    return fig


# ── outcome donut ─────────────────────────────────────────────────────────────

def _outcome_donut(df: pd.DataFrame) -> go.Figure:
    settled = (
        df[df["outcome"].notna()].copy()
        if not df.empty else pd.DataFrame()
    )
    if settled.empty:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(height=240))
        fig.add_annotation(text="No settled bets", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           font=dict(color=_MUTED, size=13))
        return fig

    counts = settled["outcome"].value_counts()
    labels = counts.index.tolist()
    values = counts.values.tolist()
    colours = [_OUTCOME_COLOURS.get(lbl, _MUTED) for lbl in labels]

    fig = go.Figure(go.Pie(
        labels=[l.title() for l in labels],
        values=values,
        marker_colors=colours,
        hole=0.6,
        textinfo="percent+label",
        textfont=dict(size=11),
        hovertemplate="%{label}: <b>%{value}</b> (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        **_chart_layout(height=240, showlegend=False),
        annotations=[dict(
            text=f"{sum(values)}<br>bets",
            x=0.5, y=0.5, font=dict(size=13, color=_FG2),
            showarrow=False,
        )],
    )
    return fig


# ── chart card wrapper ────────────────────────────────────────────────────────

def _chart_card(title: str, fig: go.Figure) -> None:
    st.markdown(
        f'<div class="chart-card"><div class="chart-title">{escape(title)}</div></div>',
        unsafe_allow_html=True,
    )
    # Key off the title so charts that come out identical on empty/sparse data
    # (e.g. before any bets exist) don't collide on Streamlit's auto-generated ID.
    key = "chart_" + re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    st.plotly_chart(fig, width="stretch", config=_plotly_cfg(), key=key)


# ── settle pending bets ───────────────────────────────────────────────────────

def _render_settle(tracker: BetTracker) -> None:
    pending = tracker.pending_bets()
    if not pending:
        st.info("No pending bets to settle.", icon="✅")
        return

    for bet in pending:
        bid = bet["id"]
        name = bet.get("horse_name") or "?"
        venue = bet.get("venue") or "?"
        stake = float(bet.get("stake") or 0)
        odds = float(bet.get("odds_decimal") or 0)
        bt_lbl = "E/W" if bet.get("bet_type") == "each_way" else "Win"

        with st.expander(
            f"#{bid} — {name}  ·  {venue}  ·  €{stake:.2f} {bt_lbl} @ {odds:.2f}"
        ):
            col_info, col_form = st.columns([2, 1])
            with col_info:
                race_time_str = ""
                if bet.get("race_time"):
                    ts = bet["race_time"]
                    if hasattr(ts, "strftime"):
                        ts = _fmt_dt(ts)
                    race_time_str = f" · Race: <b>{escape(str(ts))}</b>"
                st.markdown(
                    f'<div class="pb-meta">'
                    f'Placed: <b>{_fmt_dt(bet.get("placed_at"))}</b><br>'
                    f'Venue: <b>{escape(venue)}</b>{race_time_str}<br>'
                    f'Stake: <b>€{stake:.2f}</b> · Odds: <b>{odds:.2f}</b>'
                    f' · Type: <b>{bt_lbl}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if bet.get("notes"):
                    st.caption(f"Notes: {escape(str(bet['notes']))}")
            with col_form:
                with st.form(key=f"settle_{bid}"):
                    outcome = st.radio(
                        "Outcome", ["win", "place", "lose"],
                        horizontal=True, key=f"out_{bid}",
                    )
                    if st.form_submit_button("✓ Settle", type="primary",
                                             width="stretch"):
                        try:
                            result = tracker.settle_bet(bid, outcome)
                            profit = float(result.get("profit") or 0)
                            _load_bets.clear()
                            st.success(
                                f"Settled #{bid}: **{outcome}** · "
                                f"{'+'if profit >= 0 else ''}€{profit:.2f}"
                            )
                            st.rerun()
                        except Exception as exc:
                            st.error(str(exc))


# ── bet history table ─────────────────────────────────────────────────────────

def _render_history(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No bets match the current filters.")
        return

    show = [
        "id", "placed_at", "horse_name", "venue", "bet_type",
        "stake", "odds_decimal", "composite_score",
        "strategy", "outcome", "profit", "notes",
    ]
    present = [c for c in show if c in df.columns]
    disp = df[present].sort_values("placed_at", ascending=False).reset_index(drop=True)

    fmts = {
        "placed_at":       lambda v: _fmt_dt(v),
        "odds_decimal":    C.price_cell,
        "composite_score": lambda x: f"{x:.3f}" if pd.notna(x) else "—",
        "profit":          lambda x: f"{'+'if x>=0 else ''}€{x:.2f}" if pd.notna(x) else "—",
        "stake":           lambda x: f"€{x:.2f}" if pd.notna(x) else "—",
    }
    for col, fn in fmts.items():
        if col in disp.columns:
            disp[col] = disp[col].apply(fn)

    disp.columns = [{"odds_decimal": "Best Price"}.get(c, c.replace("_", " ").title())
                    for c in disp.columns]
    st.dataframe(disp, width="stretch", hide_index=True)

    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "Download filtered CSV",
        data=csv_bytes,
        file_name="bet_history_filtered.csv",
        mime="text/csv",
    )


# ── PDF export ────────────────────────────────────────────────────────────────

def _build_pdf(
    kpis: dict,
    df: pd.DataFrame,
    date_range: tuple,
    filters_summary: str,
) -> bytes:
    try:
        from reportlab.lib import colors as rl_colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
    except ImportError as exc:
        raise ImportError(
            "reportlab is required for PDF export. Run: pip install reportlab"
        ) from exc

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.0 * cm, rightMargin=2.0 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"],
                         fontSize=18, textColor=rl_colors.HexColor("#172033"),
                         spaceAfter=4)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         fontSize=13, textColor=rl_colors.HexColor("#2563eb"),
                         spaceBefore=14, spaceAfter=6)
    muted = ParagraphStyle("Muted", parent=styles["Normal"],
                            fontSize=9, textColor=rl_colors.HexColor("#6b7689"),
                            spaceAfter=2)

    story = []

    # Title
    story.append(Paragraph("🏇 MASHR’s Predictor v6 — Performance Summary", h1))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %b %Y %H:%M')} IST", muted
    ))
    story.append(Paragraph(f"Period: {date_range[0]} → {date_range[1]}", muted))
    story.append(Paragraph(f"Filters: {filters_summary}", muted))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=rl_colors.HexColor("#d8dee8"),
                             spaceAfter=12))

    # KPI summary
    story.append(Paragraph("Performance Metrics", h2))
    kpi_data = [
        ["Metric", "Value"],
        ["Total P&L", f"{'+'if kpis['total_profit']>=0 else ''}€{kpis['total_profit']:,.2f}"],
        ["ROI", f"{kpis['roi_pct']:+.1f}%"],
        ["Bets Settled", str(kpis["total_bets"])],
        ["Pending Bets", str(kpis["pending"])],
        ["Total Staked", f"€{kpis['total_staked']:,.2f}"],
        ["Win Rate (settled)",
         realized_rate_value(kpis.get("wins", 0), kpis["total_bets"])],
        ["Win + Place Rate", f"{kpis['place_rate']*100:.0f}%"],
        ["Each-Way Success Rate",
         f"{kpis['ew_rate']*100:.0f}% ({kpis['ew_total']} E/W bets)"],
        ["Average Odds", f"{kpis['avg_odds']:.2f}"],
        ["Max Drawdown", f"{kpis['max_dd_pct']:.1f}%"],
    ]
    kpi_table = Table(kpi_data, colWidths=[8 * cm, 7 * cm])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#2563eb")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#f6f7f9"), rl_colors.white]),
        ("FONTSIZE",   (0, 1), (-1, -1), 9),
        ("TEXTCOLOR",  (0, 1), (-1, -1), rl_colors.HexColor("#3b4658")),
        ("GRID",       (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#d8dee8")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(kpi_table)

    # Breakdown by venue
    settled = df[df["outcome"].notna()].copy() if not df.empty else pd.DataFrame()
    if not settled.empty and "venue" in settled.columns:
        story.append(Paragraph("Breakdown by Venue", h2))
        g = (
            settled.groupby("venue")
            .agg(bets=("id", "count"), wins=("outcome", lambda x: (x == "win").sum()),
                 staked=("stake", "sum"), profit=("profit", "sum"))
            .reset_index()
            .sort_values("profit", ascending=False)
        )
        g["win_rate"] = (g["wins"] / g["bets"] * 100).round(0)
        g["roi"] = (g["profit"] / g["staked"] * 100).round(1).where(g["staked"] > 0, 0)

        vd_data = [["Venue", "Bets", "Win Rate", "Staked", "Profit", "ROI"]]
        for _, row in g.iterrows():
            vd_data.append([
                str(row["venue"]),
                str(int(row["bets"])),
                f"{row['win_rate']:.0f}%",
                f"€{row['staked']:,.2f}",
                f"{'+'if row['profit']>=0 else ''}€{abs(row['profit']):,.2f}",
                f"{row['roi']:+.1f}%",
            ])
        vd_table = Table(
            vd_data,
            colWidths=[4.5 * cm, 2 * cm, 2.5 * cm, 3 * cm, 3 * cm, 2.5 * cm],
        )
        vd_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#172033")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#f6f7f9"), rl_colors.white]),
            ("FONTSIZE",   (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",  (0, 1), (-1, -1), rl_colors.HexColor("#3b4658")),
            ("GRID",       (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#d8dee8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ]))
        story.append(vd_table)

    # Recent bets list (last 20)
    if not settled.empty:
        story.append(Paragraph("Recent Settled Bets (last 20)", h2))
        recent = settled.sort_values("settled_at", ascending=False).head(20)
        rb_data = [["Date", "Horse", "Venue", "Odds", "Stake", "Outcome", "P&L"]]
        for _, row in recent.iterrows():
            rb_data.append([
                _fmt_dt(row.get("placed_at"), "%d %b"),
                str(row.get("horse_name") or "—")[:18],
                str(row.get("venue") or "—")[:12],
                f"{row.get('odds_decimal', 0):.2f}",
                f"€{row.get('stake', 0):.2f}",
                str(row.get("outcome") or "—").title(),
                f"{'+'if (row.get('profit') or 0) >= 0 else ''}€{abs(row.get('profit') or 0):.2f}",
            ])
        rb_table = Table(
            rb_data,
            colWidths=[2.2 * cm, 4 * cm, 3 * cm, 2 * cm, 2.3 * cm, 2.3 * cm, 2.2 * cm],
        )
        rb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#172033")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [rl_colors.HexColor("#f6f7f9"), rl_colors.white]),
            ("FONTSIZE",   (0, 1), (-1, -1), 7),
            ("TEXTCOLOR",  (0, 1), (-1, -1), rl_colors.HexColor("#3b4658")),
            ("GRID",       (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#d8dee8")),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ]))
        story.append(rb_table)

    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Generated by MASHR’s Predictor v6 — confidential", muted
    ))

    doc.build(story)
    return buf.getvalue()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    tracker = _get_tracker()
    df_all = _load_bets()

    filters = _render_sidebar(tracker, df_all)
    df = _apply_filters(
        df_all,
        date_range=filters["date_range"],
        bet_types=filters["bet_types"],
        odds_range=filters["odds_range"],
        venues=filters["venues"],
    )

    local_now = now()
    st.markdown(
        f'<div class="pg-hdr"><h1>📈 Performance Dashboard</h1>'
        f'<div class="sub">'
        f'{local_now.strftime("%A, %d %B %Y")} · {local_now.strftime("%H:%M")} IST'
        f' · {len(df[df["outcome"].notna()]) if not df.empty and "outcome" in df.columns else 0}'
        f' settled bets in view'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Latest GO / NO-GO verdict, so the bankroll view is read in context.
    from ui import model_honesty as MH
    MH.render_verdict_strip()

    kpis = _compute_kpis(df, tracker)
    _render_kpis(kpis)

    # ── PDF export (triggered from sidebar button) ────────────────────────────
    if filters["export_pdf"]:
        try:
            filters_summary = (
                f"Bet types: {', '.join(filters['bet_types']) or 'all'} · "
                f"Odds: {filters['odds_range'][0]:.2f}–{filters['odds_range'][1]:.2f} · "
                f"Venues: {', '.join(filters['venues'][:3]) or 'all'}"
                + (" …" if len(filters["venues"]) > 3 else "")
            )
            pdf_bytes = _build_pdf(kpis, df, filters["date_range"], filters_summary)
            fname = (
                f"race_predictor_performance_"
                f"{filters['date_range'][0]}_{filters['date_range'][1]}.pdf"
            )
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=fname,
                mime="application/pdf",
                key="pdf_dl",
            )
            st.success("PDF ready — click Download PDF above.")
        except ImportError:
            st.error("Install reportlab to enable PDF export: `pip install reportlab`")
        except Exception as exc:
            st.error(f"PDF generation failed: {exc}")

    # ── main chart row ────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">P&amp;L Over Time</div>',
                unsafe_allow_html=True)
    _chart_card("Cumulative P&L — bet by bet", _pl_over_time(df))

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-hdr">Monthly P&amp;L</div>',
                    unsafe_allow_html=True)
        _chart_card("Monthly profit / loss", _monthly_pl(df))
    with col_r:
        st.markdown('<div class="section-hdr">Rolling ROI Trend</div>',
                    unsafe_allow_html=True)
        _chart_card("Rolling ROI (10-bet window)", _roi_trend(df))

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-hdr">Win Rate by Odds Band</div>',
                    unsafe_allow_html=True)
        _chart_card("Win & place rate by odds band", _win_rate_by_odds(df))
    with col_b:
        st.markdown('<div class="section-hdr">Outcome Breakdown</div>',
                    unsafe_allow_html=True)
        _chart_card("Win / Place / Lose split", _outcome_donut(df))

    # ── settle pending bets ───────────────────────────────────────────────────
    pending_n = kpis["pending"]
    st.markdown("---")
    badge = f" ({pending_n})" if pending_n else ""
    st.markdown(
        f'<div class="section-hdr">Settle Pending Bets{badge}</div>',
        unsafe_allow_html=True,
    )
    _render_settle(tracker)

    # ── bet history ───────────────────────────────────────────────────────────
    st.markdown("---")
    total_shown = len(df) if not df.empty else 0
    st.markdown(
        f'<div class="section-hdr">Bet History — {total_shown} bets</div>',
        unsafe_allow_html=True,
    )
    _render_history(df)

    # ── footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="rp-footer">'
        f'Bankroll: €{tracker.bankroll:,.2f}'
        f' &nbsp;·&nbsp; Settled: {kpis["total_bets"]}'
        f' &nbsp;·&nbsp; Pending: {kpis["pending"]}'
        f' &nbsp;·&nbsp; ROI: {kpis["roi_pct"]:+.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )


main()
