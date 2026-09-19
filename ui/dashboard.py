"""Performance dashboard — does the model actually work? (Prompt 21)

The honest scoreboard for the whole project. Two charts are the heroes — the
**bankroll equity curve** (are we making money?) and the **A/E calibration by
probability bucket** (are the model's probabilities true?) — with CLV (did we beat
the closing price?) the deciding tie-breaker, exactly as [[reference-horse-racing-mvp-vault]]
and [[model-14-backtest]] argue. Supporting metrics (cumulative profit, ROI/yield,
hit-rate, max drawdown) sit *below* the heroes, not as a wall of SaaS hero-number
tiles.

Two data sources, shown side by side:

* **Paper trading** — the live virtual-bankroll ledger (``utils.bet_tracker`` /
  ``data/races.db``; see [[ui-20-paper-betting]]). Filterable by date, market,
  odds band, and "model picks only vs all bets".
* **Backtest** — the leak-free walk-forward runs saved under
  ``data/backtests/<run_id>/`` (see [[model-14-backtest]]). Lets us compare what
  the model *did* historically against what paper trading is doing now.

This module keeps Streamlit out of the data/figure builders so they stay unit
testable (``tests/ui/test_dashboard.py``); ``main()`` is the only Streamlit-bound
entry point. Themed to the Prompt-19 design system via ``ui._design``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from backtest.metrics import ae_table, clv_pct, max_drawdown
from ui._design import PALETTE, plotly_layout, tint
from utils.timezone import local_day

_ROOT = Path(__file__).resolve().parent.parent
_BACKTEST_DIR = _ROOT / "data" / "backtests"

# Odds bands for the filter + the by-band views (favourites → longshots). Kept in
# sync with backtest.metrics._ODDS_BANDS in spirit; labels are UI-friendly.
ODDS_BANDS: list[tuple[str, float, float]] = [
    ("Odds-on (<2.0)", 1.0, 2.0),
    ("Fav (2–4)", 2.0, 4.0),
    ("Mid (4–8)", 4.0, 8.0),
    ("Each-way (8–16)", 8.0, 16.0),
    ("Longshot (16+)", 16.0, math.inf),
]

MARKETS: list[tuple[str, str]] = [("win", "Win"), ("each_way", "Each-way")]


# ── paper-bet loading + filtering ─────────────────────────────────────────────

def load_paper_bets(tracker) -> pd.DataFrame:
    """All paper bets as a typed DataFrame (empty frame with the right columns if
    none exist). Dates are parsed to UTC; numerics coerced; a ``placed_date`` and
    a boolean ``is_pick`` (model flagged positive value) are derived."""
    cols = [
        "id", "placed_at", "settled_at", "race_time", "horse_name", "venue",
        "bet_type", "stake", "odds_decimal", "closing_odds", "clv_pct",
        "won_prob", "value_edge", "composite_score", "strategy", "outcome",
        "profit", "gross_return",
    ]
    bets = tracker.all_bets()
    if not bets:
        return pd.DataFrame(columns=cols + ["placed_date", "is_pick"])

    df = pd.DataFrame(bets)
    for c in ("placed_at", "settled_at", "race_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    for c in ("stake", "odds_decimal", "closing_odds", "clv_pct", "won_prob",
              "value_edge", "composite_score", "profit", "gross_return"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Local (Dublin) day, not UTC — see utils.timezone.local_day.
    df["placed_date"] = local_day(df["placed_at"]) if "placed_at" in df.columns else None
    # A "model pick" is a bet the model itself flagged as value (positive edge).
    edge = df["value_edge"] if "value_edge" in df.columns else pd.Series(dtype=float)
    df["is_pick"] = edge.fillna(0) > 0
    return df


def _odds_mask(odds: pd.Series, band_labels: list[str]) -> pd.Series:
    """Boolean mask: rows whose decimal odds fall in any selected band. An empty
    selection (or all bands) keeps everything, including rows with no price."""
    selected = [b for b in ODDS_BANDS if b[0] in band_labels]
    if not selected or len(selected) == len(ODDS_BANDS):
        return pd.Series(True, index=odds.index)
    mask = pd.Series(False, index=odds.index)
    for _, lo, hi in selected:
        mask |= (odds >= lo) & (odds < hi)
    # rows with no usable price never match a band → excluded only when filtering
    return mask.fillna(False)


def apply_filters(
    df: pd.DataFrame,
    *,
    date_range: Optional[tuple] = None,
    markets: Optional[list[str]] = None,
    odds_bands: Optional[list[str]] = None,
    picks_only: bool = False,
) -> pd.DataFrame:
    """Filter the paper-bet frame by date range, market, odds band and the
    model-picks-only toggle. Missing/empty filters are treated as "no filter"."""
    if df.empty:
        return df
    out = df
    if date_range and "placed_date" in out.columns:
        lo, hi = date_range
        out = out[(out["placed_date"] >= lo) & (out["placed_date"] <= hi)]
    if markets and "bet_type" in out.columns and len(markets) < len(MARKETS):
        out = out[out["bet_type"].isin(markets)]
    if odds_bands and "odds_decimal" in out.columns:
        out = out[_odds_mask(out["odds_decimal"], odds_bands)]
    if picks_only and "is_pick" in out.columns:
        out = out[out["is_pick"]]
    return out.reset_index(drop=True)


def settled_only(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a resolved outcome (win/place/lose/void), chronological by settle."""
    if df.empty or "outcome" not in df.columns:
        return df.iloc[0:0] if not df.empty else df
    s = df[df["outcome"].notna()].copy()
    if "settled_at" in s.columns:
        s = s.sort_values("settled_at").reset_index(drop=True)
    return s


# ── paper metrics (computed from the *filtered* frame so filters bite) ─────────

def paper_metrics(df_settled: pd.DataFrame, initial_bankroll: float) -> dict:
    """Headline metrics for a settled-bet selection. Computed from the frame (not
    the tracker) so the active filters actually change the numbers."""
    n = len(df_settled)
    if n == 0:
        return {
            "n_bets": 0, "staked": 0.0, "profit": 0.0, "roi_pct": None,
            "hit_rate": None, "max_drawdown_pct": 0.0, "clv_pct_mean": None,
            "beat_close_rate": None, "avg_odds": None, "final_bankroll": initial_bankroll,
            "n_void": 0,
        }
    stake = df_settled["stake"].fillna(0.0)
    profit = df_settled["profit"].fillna(0.0)
    staked = float(stake.sum())
    total_profit = float(profit.sum())
    # void bets refund the stake; they're not "decisions" for hit-rate.
    decided = df_settled[df_settled["outcome"] != "void"]
    wins = int((decided["outcome"] == "win").sum())
    n_decided = len(decided)

    equity = initial_bankroll + profit.cumsum()
    equity = pd.concat([pd.Series([initial_bankroll]), equity], ignore_index=True)

    clv = df_settled["clv_pct"].dropna() if "clv_pct" in df_settled.columns else pd.Series(dtype=float)
    return {
        "n_bets": n,
        "n_void": int((df_settled["outcome"] == "void").sum()),
        "staked": round(staked, 2),
        "profit": round(total_profit, 2),
        "roi_pct": round(total_profit / staked * 100, 2) if staked > 0 else None,
        "hit_rate": round(wins / n_decided, 4) if n_decided else None,
        "max_drawdown_pct": round(max_drawdown(equity.to_numpy()) * 100, 2),
        "clv_pct_mean": round(float(clv.mean()), 2) if not clv.empty else None,
        "beat_close_rate": round(float((clv > 0).mean()), 4) if not clv.empty else None,
        "avg_odds": round(float(df_settled["odds_decimal"].dropna().mean()), 2)
        if df_settled["odds_decimal"].notna().any() else None,
        "final_bankroll": round(float(equity.iloc[-1]), 2),
    }


def paper_ae_rows(df_settled: pd.DataFrame) -> list[dict]:
    """A/E calibration rows by predicted-probability bucket for paper bets, using
    the same bucketing as the backtester. Void bets are excluded (no decision)."""
    if df_settled.empty:
        return []
    d = df_settled[df_settled["outcome"] != "void"]
    if d.empty or "won_prob" not in d.columns:
        return []
    prob = d["won_prob"]
    won = (d["outcome"] == "win").astype(float)
    valid = prob.notna()
    if not valid.any():
        return []
    return ae_table(prob[valid].to_numpy(), won[valid].to_numpy())


# ── backtest loading ──────────────────────────────────────────────────────────

def list_backtest_runs(base: Optional[Path] = None) -> list[str]:
    """Run ids (newest first) for every saved backtest with a ``summary.json``."""
    base = base or _BACKTEST_DIR
    if not base.exists():
        return []
    runs = [p.name for p in base.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(runs, reverse=True)


def load_backtest_summary(run_id: str, base: Optional[Path] = None) -> Optional[dict]:
    """The ``summary.json`` for a run (config, model_metrics, ae tables, per-strategy)."""
    base = base or _BACKTEST_DIR
    path = base / run_id / "summary.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def load_backtest_ledger(run_id: str, strategy: str,
                         base: Optional[Path] = None) -> pd.DataFrame:
    """Per-bet ledger parquet for one strategy of a run (empty frame if absent)."""
    base = base or _BACKTEST_DIR
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in strategy)
    path = base / run_id / f"bets_{safe}.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except (OSError, ValueError):
        return pd.DataFrame()


# ── figure builders (pure: DataFrame/dict in, go.Figure out) ──────────────────

def _empty_fig(message: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**plotly_layout(height=height))
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(color=PALETTE["muted"], size=13))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def equity_curve_fig(df_settled: pd.DataFrame, initial_bankroll: float,
                     ccy: str = "€", height: int = 360) -> go.Figure:
    """HERO. Bankroll over time for the selected paper bets — reconstructed from
    the filtered ledger (initial + running profit) so it honours the filters. A
    dashed baseline marks the starting bankroll: above the line = in profit."""
    if df_settled.empty or "settled_at" not in df_settled.columns:
        return _empty_fig("No settled paper bets in view — place and settle some, "
                          "or widen the filters.", height=height)
    d = df_settled.dropna(subset=["settled_at"]).sort_values("settled_at")
    if d.empty:
        return _empty_fig("No settled paper bets with a timestamp yet.", height=height)

    balance = initial_bankroll + d["profit"].fillna(0.0).cumsum()
    x = d["settled_at"]
    # seed the curve at the starting bankroll just before the first settle
    x0 = x.iloc[0] - pd.Timedelta(minutes=1)
    xs = pd.concat([pd.Series([x0]), x], ignore_index=True)
    ys = pd.concat([pd.Series([initial_bankroll]), balance], ignore_index=True)

    # Money is the one place where green-up / red-down is honest, so the line
    # keeps the semantic pair. The fill takes the LINE's own hue at ~10% — it
    # used to be a fixed blue regardless, which read as a third, meaningless
    # colour on a two-colour chart.
    end_up = float(balance.iloc[-1]) >= initial_bankroll
    line_col = PALETTE["value"] if end_up else PALETTE["danger"]

    fig = go.Figure()
    # An invisible baseline trace at the starting bankroll, so `tonexty` fills
    # the gap between the curve and the START. `tozeroy` filled from 0, which is
    # off-scale when a bankroll opens at 1,000 — it flooded the whole plot.
    fig.add_trace(go.Scatter(
        x=xs, y=[initial_bankroll] * len(xs), mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=line_col, width=2),
        fill="tonexty", fillcolor=tint(line_col, 0.10),
        name="Paper bankroll",
        hovertemplate=f"%{{x|%d %b %H:%M}}<br><b>{ccy}%{{y:,.2f}}</b><extra></extra>",
    ))
    # end marker: >=8px, with a 2px ring in the surface colour so it stays
    # legible where it crosses the line — plus the one direct label the chart
    # gets (label selectively, never a number on every point)
    fig.add_trace(go.Scatter(
        x=[xs.iloc[-1]], y=[ys.iloc[-1]], mode="markers+text",
        marker=dict(color=line_col, size=9,
                    line=dict(color=PALETTE["surface"], width=2)),
        text=[f"  {ccy}{float(ys.iloc[-1]):,.0f}"], textposition="middle left",
        textfont=dict(color=PALETTE["ink"], size=12),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_hline(y=initial_bankroll, line_dash="dash",
                  line_color=PALETTE["muted"], line_width=1,
                  annotation_text=f"start {ccy}{initial_bankroll:,.0f}",
                  annotation_position="bottom right",
                  annotation_font=dict(color=PALETTE["muted"], size=11))
    fig.update_layout(**plotly_layout(
        height=height, showlegend=False, hovermode="x unified",
        yaxis_title=f"Bankroll ({ccy})", xaxis_title=None,
    ))
    fig.update_yaxes(tickprefix=ccy, scaleanchor=None)
    return fig


def calibration_ae_fig(paper_rows: list[dict], backtest_rows: list[dict],
                       height: int = 360) -> go.Figure:
    """HERO. A/E (actual ÷ expected) per predicted-probability bucket. A/E≈1 means
    honest probabilities; <1 over-confident, >1 under-confident. Paper and backtest
    bars sit side by side against the dashed A/E=1 reference — the live calibration
    check the whole project rests on."""
    if not paper_rows and not backtest_rows:
        return _empty_fig("No probability data yet — needs settled paper bets or a "
                          "backtest run.", height=height)

    buckets: list[str] = []
    for rows in (backtest_rows, paper_rows):
        for r in rows:
            if r["bucket"] not in buckets:
                buckets.append(r["bucket"])

    def _series(rows: list[dict]) -> tuple[list, list, list]:
        """Bar LENGTHS measured from the 1.0 baseline, plus the true A/E and the
        sample count for the tooltip."""
        by = {r["bucket"]: r for r in rows}
        lengths, ae, text = [], [], []
        for b in buckets:
            r = by.get(b)
            if r and r.get("ae") is not None:
                lengths.append(float(r["ae"]) - 1.0)
                ae.append(float(r["ae"]))
                text.append(f"n={r['n']:,}")
            else:
                lengths.append(None)
                ae.append(None)
                text.append("")
        return lengths, ae, text

    # Bars are anchored at the 1.0 baseline and grow up or down from it, so the
    # DIRECTION of the miss is the shape of the chart rather than something to
    # be read off an axis. Colour comes from the validated colorway in trace
    # order (Backtest = slot 1, Paper = slot 2) — never from a semantic token.
    fig = go.Figure()
    if backtest_rows:
        lengths, ae, text = _series(backtest_rows)
        fig.add_trace(go.Bar(
            x=buckets, y=lengths, base=1.0, name="Backtest",
            marker_color=PALETTE["series_1"],
            customdata=list(zip(ae, text)),
            hovertemplate="Backtest %{x}<br>A/E <b>%{customdata[0]:.2f}</b>"
                          "<br>%{customdata[1]}<extra></extra>",
        ))
    if paper_rows:
        lengths, ae, text = _series(paper_rows)
        fig.add_trace(go.Bar(
            x=buckets, y=lengths, base=1.0, name="Paper",
            marker_color=PALETTE["series_2"],
            customdata=list(zip(ae, text)),
            # sparse direct labels: the sample count rides the live series only
            text=text, textposition="outside",
            textfont=dict(color=PALETTE["muted"], size=10),
            hovertemplate="Paper %{x}<br>A/E <b>%{customdata[0]:.2f}</b>"
                          "<br>%{customdata[1]}<extra></extra>",
        ))
    # NEUTRAL ink, not the value green: A/E = 1 is an honesty claim, not a
    # positive-EV signal, and BOTH directions of miscalibration are bad.
    fig.add_hline(y=1.0, line_dash="dash", line_color=PALETTE["border_strong"],
                  line_width=1.5,
                  annotation_text="perfect (A/E = 1)", annotation_position="top left",
                  annotation_font=dict(color=PALETTE["muted"], size=11))
    fig.update_layout(**plotly_layout(
        height=height, barmode="group", bargap=0.58, bargroupgap=0.12,
        yaxis_title="Actual / Expected", xaxis_title="Predicted win-probability bucket",
    ))
    return fig


def cumulative_profit_fig(df_settled: pd.DataFrame, ccy: str = "€") -> go.Figure:
    """Cumulative profit, bet by bet, with per-bet outcome markers."""
    if df_settled.empty:
        return _empty_fig("No settled bets in view.", height=280)
    d = df_settled.copy().reset_index(drop=True)
    d["bet_num"] = range(1, len(d) + 1)
    d["cum"] = d["profit"].fillna(0.0).cumsum()
    if "horse_name" not in d.columns:
        d["horse_name"] = "—"

    colours = {"win": PALETTE["value"], "place": PALETTE["info"],
               "lose": PALETTE["danger"], "void": PALETTE["muted"]}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["bet_num"], y=d["cum"], mode="lines",
        line=dict(color=PALETTE["info"], width=2),
        fill="tozeroy", fillcolor=tint(PALETTE["info"], 0.08),
        showlegend=False,
        hovertemplate=f"Bet #%{{x}}<br>Running P&L <b>{ccy}%{{y:+,.2f}}</b><extra></extra>",
    ))
    for outcome, col in colours.items():
        sub = d[d["outcome"] == outcome]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["bet_num"], y=sub["cum"], mode="markers", name=outcome.title(),
            marker=dict(color=col, size=7, line=dict(color=PALETTE["surface"], width=1)),
            customdata=sub[["horse_name", "profit"]].fillna(0).to_numpy(),
            hovertemplate=("%{customdata[0]}<br>"
                           f"{outcome.title()}: {ccy}%{{customdata[1]:+,.2f}}<extra></extra>"),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=PALETTE["muted"], line_width=1)
    fig.update_layout(**plotly_layout(
        height=280, xaxis_title="Bet #", yaxis_title=f"Cumulative P&L ({ccy})",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=PALETTE["ink_2"])),
    ))
    fig.update_yaxes(tickprefix=ccy)
    return fig


def clv_fig(df_settled: pd.DataFrame) -> go.Figure:
    """CLV distribution: each settled bet's price improvement vs the closing (SP)
    price. Bars right of zero (green) beat the close — the strongest +EV signal."""
    if df_settled.empty or "clv_pct" not in df_settled.columns:
        return _empty_fig("No CLV data — settle bets with a closing price.", height=280)
    clv = df_settled["clv_pct"].dropna()
    if clv.empty:
        return _empty_fig("No closing prices captured yet — CLV needs the SP at "
                          "settlement.", height=280)
    mean_clv = float(clv.mean())
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clv, nbinsx=max(6, min(30, len(clv))),
        marker_color=PALETTE["info"], marker_line_color=PALETTE["surface"],
        marker_line_width=1,
        hovertemplate="CLV %{x:.1f}%<br>%{y} bets<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dot", line_color=PALETTE["muted"], line_width=1)
    fig.add_vline(x=mean_clv, line_dash="dash",
                  line_color=PALETTE["value"] if mean_clv >= 0 else PALETTE["danger"],
                  line_width=1.5,
                  annotation_text=f"mean {mean_clv:+.1f}%", annotation_position="top",
                  annotation_font=dict(
                      color=PALETTE["value"] if mean_clv >= 0 else PALETTE["danger"],
                      size=11))
    fig.update_layout(**plotly_layout(
        height=280, showlegend=False, bargap=0.05,
        xaxis_title="Closing-line value (%)", yaxis_title="Bets",
    ))
    fig.update_xaxes(ticksuffix="%")
    return fig


def normalized_equity_fig(paper_settled: pd.DataFrame, initial_bankroll: float,
                          bt_ledger: pd.DataFrame, bt_label: str = "Backtest") -> go.Figure:
    """Paper vs backtest, directly comparable: both bankrolls re-based to 100 at the
    start and plotted against bet number, so the *shape* of growth lines up even
    though they cover different periods."""
    fig = go.Figure()
    plotted = False

    if not bt_ledger.empty and "bankroll_after" in bt_ledger.columns:
        bt = bt_ledger.reset_index(drop=True)
        start = float(bt["bankroll_after"].iloc[0]) - float(bt["profit"].iloc[0]) \
            if "profit" in bt.columns else initial_bankroll
        start = start or initial_bankroll
        idx = range(1, len(bt) + 1)
        y = bt["bankroll_after"] / start * 100.0
        fig.add_trace(go.Scatter(
            x=list(idx), y=y, mode="lines", name=bt_label,
            line=dict(color=PALETTE["muted"], width=2, dash="dot"),
            hovertemplate=f"{bt_label} bet %{{x}}<br>%{{y:.1f}} of start<extra></extra>",
        ))
        plotted = True

    if not paper_settled.empty:
        p = paper_settled.reset_index(drop=True)
        bal = initial_bankroll + p["profit"].fillna(0.0).cumsum()
        y = bal / initial_bankroll * 100.0
        fig.add_trace(go.Scatter(
            x=list(range(1, len(p) + 1)), y=y, mode="lines+markers", name="Paper",
            line=dict(color=PALETTE["value"], width=2.5),
            marker=dict(size=5, color=PALETTE["value"]),
            hovertemplate="Paper bet %{x}<br>%{y:.1f} of start<extra></extra>",
        ))
        plotted = True

    if not plotted:
        return _empty_fig("No paper or backtest data to compare.", height=300)

    fig.add_hline(y=100, line_dash="dash", line_color=PALETTE["muted"], line_width=1)
    fig.update_layout(**plotly_layout(
        height=300, xaxis_title="Bet # (sequence)",
        yaxis_title="Bankroll (% of start)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color=PALETTE["ink_2"])),
    ))
    return fig


def backtest_compare_table(summary: dict) -> pd.DataFrame:
    """Per-strategy comparison rows for a backtest run, sorted by yield."""
    strategies = (summary or {}).get("strategies") or {}
    if not strategies:
        return pd.DataFrame()
    rows = list(strategies.values())
    df = pd.DataFrame(rows)
    cols = [c for c in ("strategy", "n_bets", "yield_pct", "hit_rate",
                        "clv_pct_mean", "beat_close_rate", "max_drawdown_pct",
                        "bankroll_growth_pct", "avg_odds") if c in df.columns]
    df = df[cols]
    if "yield_pct" in df.columns:
        df = df.sort_values("yield_pct", ascending=False, na_position="last")
    return df.reset_index(drop=True)
