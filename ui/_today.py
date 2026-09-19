"""Today-facing charts for the landing page — the pre-race half of the desk.

``ui/dashboard.py`` answers *did the model work?* from the settled ledger. This
module answers the two questions you have **before** a race runs:

* **Where is the value?** — expected value by odds band, across today's card.
* **Does the model disagree with the market?** — model win-probability against
  the market's implied probability, runner by runner.
* **Who is pricing best?** — how often each book holds the best price.

Data comes from ``data/predictions.json`` (the shape ``models/predictor`` writes),
never from the network. Row builders are pure ``list[dict] -> list[dict]`` so
they unit-test headless; the figure builders are the only part that needs Plotly.

Colour follows ``DESIGN.md``: chart series take the validated categorical slots,
and the ONE place a semantic token is legal in a chart is the value/no-value
split — there the green genuinely means positive expected value, which is what
the token is reserved for.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

import plotly.graph_objects as go

from ui._components import headline_win_prob
from ui._design import PALETTE, plotly_layout, tint
from utils.odds_format import book_label
from utils.text_norm import minute_key, norm_horse, norm_venue

# Favourites → longshots. An ORDERED sequence, so charts over it use the ordinal
# ramp and let the axis carry the order — never a categorical hue per band.
ODDS_BANDS: list[tuple[str, float, float]] = [
    ("Odds-on", 1.0, 2.0),
    ("Fav 2–4", 2.0, 4.0),
    ("Mid 4–8", 4.0, 8.0),
    ("E/W 8–16", 8.0, 16.0),
    ("16/1+", 16.0, math.inf),
]


def field(race: dict) -> list[dict]:
    """The complete validated field for one race.

    Prefers ``runners`` over the display ``selections`` for the same reason the
    landing page's KPI rail does (audit req 2/7): ``selections`` is the top-3
    display list, so a mid-field value bet would be invisible to anything that
    counts over it — while the suggestion engine, which reads ``runners``, would
    still surface it. Older caches without ``runners`` fall back to the display
    lists.
    """
    full = race.get("runners")
    if full:
        return full
    return (race.get("selections") or []) + (race.get("excluded_low_odds") or [])


def _runners(races: list[dict]) -> list[tuple[dict, dict]]:
    """Every (race, runner) pair across the card, over the full field."""
    return [(r, s) for r in (races or []) for s in field(r)]


def _price(sel: dict) -> Optional[float]:
    """The price we would actually take: best available, falling back to the
    quoted decimal."""
    p = sel.get("best_odds") or sel.get("decimal_odds")
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    return p if p > 1.0 else None


# ── where is the value? ───────────────────────────────────────────────────────

def odds_band_rows(races: list[dict]) -> list[dict]:
    """Mean expected value per odds band for today's card.

    EV is already market-relative (model probability vs the price on offer), so
    a negative mean is the honest and usual answer: it says the book is ahead of
    us in that band today.
    """
    out: list[dict] = []
    for label, lo, hi in ODDS_BANDS:
        evs, n_value = [], 0
        for _race, sel in _runners(races):
            price = _price(sel)
            if price is None or not (lo <= price < hi):
                continue
            ev = sel.get("expected_value")
            if ev is not None:
                evs.append(float(ev))
            if sel.get("value_bet"):
                n_value += 1
        if not evs:
            continue
        out.append({
            "band": label,
            "n": len(evs),
            "mean_ev": sum(evs) / len(evs),
            "n_value": n_value,
        })
    return out


def ev_by_odds_band_fig(rows: list[dict], height: int = 260) -> go.Figure:
    """Mean EV per odds band, as bars growing from the zero line.

    Zero is the meaningful baseline (break-even against the market), so the bars
    are anchored there and the SIGN carries the read — value above, overround
    below. Polarity, not identity, so this is the diverging pair rather than a
    categorical hue per band.
    """
    if not rows:
        return _empty("No priced runners on today's card.", height)

    pos, neg = PALETTE["div_pos"], PALETTE["div_neg"]
    ys = [r["mean_ev"] * 100 for r in rows]
    fig = go.Figure(go.Bar(
        x=[r["band"] for r in rows], y=ys,
        marker_color=[pos if y >= 0 else neg for y in ys],
        marker_line_color=PALETTE["surface"], marker_line_width=2,
        customdata=[(r["n"], r["n_value"]) for r in rows],
        hovertemplate="<b>%{x}</b><br>Mean EV %{y:.1f}%"
                      "<br>%{customdata[0]} runners · %{customdata[1]} value"
                      "<extra></extra>",
    ))
    fig.add_hline(y=0, line_width=1, line_color=PALETTE["border_strong"])
    fig.update_layout(**plotly_layout(
        height=height, density="compact", showlegend=False, bargap=0.55,
        yaxis_title="Mean EV", yaxis_ticksuffix="%", xaxis_title=None,
    ))
    return fig


# ── does the model disagree with the market? ──────────────────────────────────

def model_vs_market_rows(races: list[dict]) -> list[dict]:
    """One row per priced runner: the market's implied probability against the
    model's, plus whether the value layer flagged it."""
    out: list[dict] = []
    for race, sel in _runners(races):
        market = sel.get("implied_prob") or sel.get("market_prob")
        model = headline_win_prob(sel)
        if market is None or model is None:
            continue
        out.append({
            "horse": sel.get("horse_name") or "—",
            "venue": race.get("venue") or "—",
            "market": float(market),
            "model": float(model),
            "is_value": bool(sel.get("value_bet")),
            "ev": sel.get("expected_value"),
        })
    return out


def model_vs_market_fig(rows: list[dict], height: int = 260) -> go.Figure:
    """Model win-probability against the market's implied probability.

    The diagonal is agreement. A point ABOVE it means the model rates the runner
    higher than the price does — the shape of today's disagreement, and where
    any edge would have to come from.

    This is the one chart where a semantic token is legal as series colour: the
    split is value / no-value, and ``--value`` is reserved for exactly that. The
    marker shape and the legend carry it too, never the colour alone.
    """
    if not rows:
        return _empty("No priced runners with a model probability yet.", height)

    hi = max([r["market"] for r in rows] + [r["model"] for r in rows] + [0.05])
    hi = min(1.0, hi * 1.12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, hi], y=[0, hi], mode="lines",
        line=dict(color=PALETTE["border_strong"], width=1, dash="dash"),
        hoverinfo="skip", showlegend=False,
    ))
    for is_value, name, colour, symbol in (
        (False, "Not flagged", PALETTE["muted"], "circle"),
        (True, "Value flagged", PALETTE["value"], "diamond"),
    ):
        pts = [r for r in rows if r["is_value"] is is_value]
        if not pts:
            continue
        fig.add_trace(go.Scatter(
            x=[p["market"] for p in pts], y=[p["model"] for p in pts],
            mode="markers", name=name,
            marker=dict(color=colour, size=9, symbol=symbol,
                        line=dict(color=PALETTE["surface"], width=2)),
            customdata=[(p["horse"], p["venue"]) for p in pts],
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}"
                          "<br>Market %{x:.0%} · Model %{y:.0%}<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        height=height, density="compact",
        xaxis_title="Market implied", yaxis_title="Model",
        xaxis=dict(range=[0, hi], tickformat=".0%", automargin=True,
                   gridcolor=PALETTE["chart_grid"], zerolinecolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["chart_axis"], size=10)),
        yaxis=dict(range=[0, hi], tickformat=".0%", automargin=True,
                   gridcolor=PALETTE["chart_grid"], zerolinecolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["chart_axis"], size=10)),
    ))
    return fig


# ── who is pricing best? ──────────────────────────────────────────────────────

def best_book_rows(races: list[dict]) -> list[dict]:
    """How often each bookmaker holds the best price on today's card."""
    counts = Counter(
        sel.get("best_book") for _r, sel in _runners(races) if sel.get("best_book")
    )
    total = sum(counts.values())
    return [
        {"book": book_label(b), "n": n, "share": n / total}
        for b, n in counts.most_common()
    ] if total else []


def best_book_fig(rows: list[dict], height: int = 260) -> go.Figure:
    """Best-price count per book. Nominal identity between peers, and bar length
    already carries the value — so every bar takes the SAME slot-1 hue rather
    than spending the identity channel on re-encoding the count."""
    if not rows:
        return _empty("No per-book prices on today's card.", height)
    fig = go.Figure(go.Bar(
        x=[r["n"] for r in rows], y=[r["book"] for r in rows], orientation="h",
        marker_color=PALETTE["series_1"],
        marker_line_color=PALETTE["surface"], marker_line_width=2,
        text=[f"{r['share']:.0%}" for r in rows], textposition="outside",
        textfont=dict(color=PALETTE["muted"], size=10),
        hovertemplate="<b>%{y}</b><br>Best price on %{x} runners<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        height=height, density="compact", showlegend=False, bargap=0.55,
        xaxis_title="Runners priced best", yaxis_title=None,
        # headroom so the outside share label is never clipped at the plot edge
        xaxis=dict(range=[0, max(r["n"] for r in rows) * 1.28], automargin=True,
                   gridcolor=PALETTE["chart_grid"], zerolinecolor=PALETTE["border"],
                   tickfont=dict(color=PALETTE["chart_axis"], size=10)),
        yaxis=dict(autorange="reversed", automargin=True,
                   tickfont=dict(color=PALETTE["chart_axis"], size=10)),
    ))
    return fig


# ── shared empty state ────────────────────────────────────────────────────────

def _empty(message: str, height: int) -> go.Figure:
    """A calm, honest empty state — never a blank box or a crash."""
    fig = go.Figure()
    fig.update_layout(**plotly_layout(height=height, density="compact",
                                      showlegend=False))
    fig.add_annotation(text=message, showarrow=False, xref="paper", yref="paper",
                       x=0.5, y=0.5, font=dict(color=PALETTE["muted"], size=12))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ── today's winners: did the picks land? ──────────────────────────────────────

def winners_rows(results, races: list[dict], top_n: int = 3) -> list[dict]:
    """Join our ranked picks to where the runners ACTUALLY finished.

    ``results`` is any frame carrying ``venue``, ``race_time``, ``horse_name``,
    ``position`` and (optionally) ``sp`` — the shape of
    ``data/unified_races.parquet``. Matching is on normalised venue + horse and
    a minute-resolution race time, because the odds feeds and the results feed
    disagree on punctuation and on seconds.

    Returns ``[]`` when the day has no finishing positions yet. That is the
    normal state before a card has run, and it must read as "not yet", never as
    "nothing won" — the caller is responsible for saying which.
    """
    if results is None or getattr(results, "empty", True) or not races:
        return []

    cols = {"venue", "race_time", "horse_name", "position"}
    if not cols.issubset(set(results.columns)):
        return []

    finished = results[results["position"].notna()]
    if finished.empty:
        return []

    by_runner: dict[tuple, dict] = {}
    for r in finished.itertuples(index=False):
        key = (norm_venue(str(r.venue or "")), minute_key(r.race_time),
               norm_horse(str(r.horse_name or "")))
        by_runner[key] = {
            "position": int(r.position),
            "sp": getattr(r, "sp", None),
        }

    out: list[dict] = []
    for race in races:
        venue, rtime = race.get("venue") or "", race.get("race_time")
        picks = sorted(
            [s for s in field(race) if s.get("rank")],
            key=lambda s: s.get("rank") or 99,
        )[:top_n]
        for sel in picks:
            hit = by_runner.get((norm_venue(str(venue)), minute_key(rtime),
                                 norm_horse(str(sel.get("horse_name") or ""))))
            if hit is None:
                continue
            pos = hit["position"]
            out.append({
                "venue": venue,
                "race_time": rtime,
                "horse": sel.get("horse_name") or "—",
                "rank": int(sel.get("rank")),
                "position": pos,
                "sp": hit["sp"],
                "price": _price(sel),
                "won": pos == 1,
                "placed": pos <= 3,
                "each_way": bool(sel.get("each_way_value")),
            })
    out.sort(key=lambda r: (str(r["race_time"]), r["rank"]))
    return out


def winners_summary(rows: list[dict]) -> dict:
    """Headline counts for the day. A rate never travels without its
    denominator, so the caller gets both."""
    top = [r for r in rows if r["rank"] == 1]
    return {
        "races": len({(r["venue"], str(r["race_time"])) for r in rows}),
        "picks": len(top),
        "wins": sum(1 for r in top if r["won"]),
        "places": sum(1 for r in top if r["placed"]),
    }


def last_results_day(results):
    """The most recent day that carries any finishing position, or ``None``.

    The empty state uses this to say *why* there are no winners yet: a card
    that has not run reads very differently from a results feed that stopped
    eight weeks ago, and the project's honesty rule says never to blur the two.
    """
    if results is None or getattr(results, "empty", True):
        return None
    if "position" not in results.columns or "race_time" not in results.columns:
        return None
    done = results[results["position"].notna()]
    if done.empty:
        return None
    import pandas as pd
    rt = pd.to_datetime(done["race_time"], utc=True, errors="coerce").dropna()
    if rt.empty:
        return None
    return rt.dt.tz_convert("Europe/Dublin").dt.date.max()
