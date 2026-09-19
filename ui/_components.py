"""MASHR’s Predictor v6 — reusable UI components (Prompt 19).

Pure HTML-string builders over the ``rp-*`` design classes defined in
:mod:`ui._design`. Keeping them Streamlit-free (they return strings; the caller
does the single ``st.markdown(..., unsafe_allow_html=True)``) means the markup
is unit-testable headless and a screen is assembled from named primitives
instead of one 200-line f-string.

Roles are fixed by the design system and must not drift:
  • brand    → identity, rank-1, primary action (indigo/violet)
  • green    → value / positive expected-value ONLY
  • blue     → probability bars, links, info
  • amber    → each-way / caution
Colour never carries meaning alone — every signal pairs with a word or glyph.
"""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional

from utils.odds_format import to_fraction
from utils.timezone import to_local
from ui import _icons as _I

# ── inline-SVG brand marks (crafted, not the heavy raster-in-svg source) ──────

def mark_finishline(size: int = 20, stroke: str = "currentColor") -> str:
    """Checkered finish flag on a post — the brand mark. Line-art, currentColor
    so it tints to whatever ``color`` the parent sets."""
    s = size
    return (
        f'<svg class="mk" width="{s}" height="{s}" viewBox="0 0 24 24" fill="none" '
        f'aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
        f'<path d="M5 3v18" stroke="{stroke}" stroke-width="1.6" stroke-linecap="round"/>'
        # flag body
        f'<path d="M5 4h13v8H5z" stroke="{stroke}" stroke-width="1.4" '
        f'stroke-linejoin="round"/>'
        # checker squares (filled diagonally)
        f'<path d="M5 4h3.25v2H5zM11.5 4h3.25v2H11.5zM8.25 6h3.25v2H8.25z'
        f'M14.75 6h3.25v2h-3.25zM5 8h3.25v2H5zM11.5 8h3.25v2H11.5z" '
        f'fill="{stroke}" opacity="0.92"/>'
        f'</svg>'
    )


def mark_loader(size: int = 56) -> str:
    """Racetrack-oval with an orbiting running dot. The dot animates via the
    ``.rp-loader .dot`` keyframes in the design CSS (stilled under reduced-motion)."""
    return (
        f'<span class="rp-loader" style="width:{size}px;height:{size}px">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 56 56" fill="none" '
        f'aria-hidden="true">'
        f'<ellipse cx="28" cy="28" rx="22" ry="13" stroke="var(--border, #31363d)" '
        f'stroke-width="2.5"/>'
        f'<g class="dot">'
        f'<circle cx="50" cy="28" r="4" fill="var(--brand, #6d5cf0)"/>'
        f'</g>'
        f'</svg></span>'
    )


# ── formatting helpers ────────────────────────────────────────────────────────

def fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = to_local(dt)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return iso


def fmt_odds(val: Optional[float]) -> str:
    return "—" if val is None else f"{val:.2f}"


def fmt_price(val: Optional[float]) -> str:
    """A price as a board fraction (7/1). Odds are a ratio — never money, so this
    carries no currency symbol and is never FX-converted."""
    return to_fraction(val)


def price_cell(val) -> str:
    """A price for a plain-text table cell: the board fraction with the decimal
    alongside (``11/4 · 3.75``). Same contract as :func:`price_html` — no
    currency symbol, no FX — but for a ``st.dataframe``, which takes no markup.

    Guards NaN explicitly: a pandas NaN is truthy and fails ``<= 1.0``, so it
    would slip past :func:`to_fraction`'s own guard.
    """
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if v != v:                      # NaN
        return "—"
    frac = to_fraction(v)
    return frac if frac == "—" else f"{frac} · {v:.2f}"


def price_html(val: Optional[float], cls: str = "rp-odds") -> str:
    """Headline price: the fraction, with the decimal kept alongside it small."""
    if not val:
        return '<span class="rp-sub">—</span>'
    return (
        f'<span class="{cls}">{escape(to_fraction(val))}</span>'
        f'<span class="rp-sub">&nbsp;{float(val):.2f}</span>'
    )


def fmt_pct(prob: Optional[float], dp: int = 0) -> str:
    if prob is None:
        return "—"
    return f"{max(0.0, min(1.0, float(prob))) * 100:.{dp}f}%"


def headline_win_prob(sel: dict) -> Optional[float]:
    """The win probability to PRESENT as the headline "Win %".

    Prefers ``won_prob_normalized`` — the within-race-normalized prob where each
    race's field sums to ~1 (exactly one winner). It is well-calibrated
    out-of-sample (AUC 0.78 / ECE 0.03 on the last settled week). The raw
    per-runner calibrated marginal ``won_prob`` saturates badly OOS — its v3
    isotonic curve, fit on a narrow historical slice, piles the live field near a
    0.73 ceiling (mean 0.33 vs a true 0.11 win rate, ECE 0.22) — so it is kept
    only as a debug / EV-reference field, never the headline. Falls back to
    ``won_prob`` when the normalized value is absent (single-runner rows or a
    pre-normalization cache). See calib-fl-01.
    """
    norm = sel.get("won_prob_normalized")
    return norm if norm is not None else sel.get("won_prob")


def race_slug(race: dict) -> str:
    """Stable id for a race, used as the ``?race=`` query param on the detail
    page. Races have no explicit id, so we key on venue + ISO post-time."""
    venue = (race.get("venue") or "race").strip().lower()
    venue = "".join(c if c.isalnum() else "-" for c in venue).strip("-")
    t = (race.get("race_time") or "").replace(":", "").replace("-", "")[:13]
    return f"{venue}-{t}" if t else venue


def find_race(races: list[dict], slug: str) -> Optional[dict]:
    return next((r for r in (races or []) if race_slug(r) == slug), None)


# ── small primitives ──────────────────────────────────────────────────────────

def pill(text: str, kind: str = "field") -> str:
    return f'<span class="pill pill-{kind}">{escape(str(text))}</span>'


def rank_badge(rank: Optional[int]) -> str:
    """Rank-1 is filled with the brand; every other rank is a neutral outline —
    no rainbow of rank colours competing with the value signal."""
    if rank is None:
        return '<span class="rk rk-n">—</span>'
    cls = "rk1" if rank == 1 else "rk-n"
    return f'<span class="rk {cls}">{rank}</span>'


def prob_bar(prob: Optional[float], kind: str = "win") -> str:
    """Probability bar (kind: win|place|show). Blue ramp — distinct from the
    green that means value. Visible numeric label, never colour-only."""
    if prob is None:
        return '<div class="pb"><span class="pb-val">—</span></div>'
    pct = max(0.0, min(100.0, float(prob) * 100.0))
    # width rides on `--pbw` so the fill can grow from 0 on mount (keyframe in
    # _design.py) and still render at the final width under reduced motion.
    return (
        f'<div class="pb"><div class="pb-track">'
        f'<div class="pb-fill {kind}" style="--pbw:{pct:.0f}%"></div></div>'
        f'<span class="pb-val">{pct:.0f}%</span></div>'
    )


def value_badge(edge: Optional[float] = None) -> str:
    """Positive-EV badge. The word VALUE carries the signal; the green and the
    edge figure reinforce it."""
    tail = ""
    if edge is not None:
        tail = f' <span class="tnum">{edge * 100:+.0f}pp</span>'
    return f'<span class="val-badge"><span class="vd"></span>VALUE{tail}</span>'


def ew_badge() -> str:
    return '<span class="ew-badge">E/W</span>'


def ev_text(ev: Optional[float], is_value: bool) -> str:
    """Expected-value figure as visible text (not a tooltip): green + weighted
    when it's a flagged value bet, muted otherwise."""
    if ev is None:
        return '<span class="ev ev-flat">—</span>'
    cls = "ev-pos" if is_value else "ev-flat"
    return f'<span class="ev {cls}">{ev * 100:+.0f}%</span>'


def confidence_for(sel: dict) -> Optional[str]:
    """Confidence level (high|med|low) for a runner.

    Uses the predictor's own ``confidence`` field when the cache carries it
    (Prompt-17 inference contract). Older caches predate it; rather than show
    nothing we fall back to a transparent proxy from data completeness +
    probability margin, so the chip is always meaningful and never fabricated
    out of thin air."""
    conf = sel.get("confidence")
    if isinstance(conf, str) and conf.lower() in {"high", "med", "medium", "low"}:
        return "med" if conf.lower() == "medium" else conf.lower()

    completeness = sel.get("data_completeness")
    won = sel.get("won_prob")
    if completeness is None and won is None:
        return None
    score = 0
    if completeness is not None:
        score += 2 if completeness >= 0.8 else (1 if completeness >= 0.5 else 0)
    if won is not None:
        score += 1 if won >= 0.18 else 0
    if sel.get("first_time_runner"):
        score -= 1
    return "high" if score >= 3 else ("med" if score >= 1 else "low")


def confidence_chip(level: Optional[str]) -> str:
    if not level:
        return ""
    label = {"high": "High", "med": "Medium", "low": "Low"}.get(level, level.title())
    dots = "".join("<i></i>" for _ in range(3))
    return (
        f'<span class="conf {level}" title="Model confidence: {label}">'
        f'<span class="cdots">{dots}</span>{label}</span>'
    )


_BOOK_LABELS = {"livescorebet": "LSB", "paddy_power": "Paddy", "boylesports": "Boyle"}


def book_chips(
    books: Optional[dict], best_book: Optional[str], ccy: Optional[str] = None
) -> str:
    """Per-book price row; the best (highest) price is marked ★ and weighted —
    the star carries the signal, the green only reinforces it.

    Prices render as traditional fractions (Paddy 7/1), never as currency: odds
    are a ratio, not an amount. ``ccy`` is accepted and ignored so older call
    sites keep working.
    """
    if not books:
        return ""
    items = sorted(books.items(), key=lambda kv: (kv[1] or 0), reverse=True)
    chips = []
    for src, price in items:
        if price is None:
            continue
        label = _BOOK_LABELS.get(src, src[:5].title())
        is_best = src == best_book
        cls = "bk bk-best" if is_best else "bk"
        star = " ★" if is_best else ""
        chips.append(
            f'<span class="{cls}" title="{escape(label)} {price:.2f} decimal">'
            f'<span class="bkn">{escape(label)}</span> '
            f"{escape(to_fraction(price))}{star}</span>"
        )
    return f'<div class="bk-row">{"".join(chips)}</div>' if chips else ""


# ── brand / chrome ────────────────────────────────────────────────────────────

def brand_lockup() -> str:
    """Sidebar brand lockup: finish-flag mark + wordmark + tagline."""
    return (
        '<div class="rp-brand">'
        f'<span class="mk" style="color:var(--brand-text)">{mark_finishline(22)}</span>'
        '<span><span class="wm">MASHR&rsquo;s <em>Predictor</em></span>'
        '<div class="tag">Form &amp; value desk</div></span>'
        '</div>'
    )


def trust_badge(label: str, value: str, tone: str = "ok") -> str:
    """Model-trust badge (app-bar right): a status dot + label + mono value, e.g.
    calibration ECE or held-out AUC. Reachable from the dashboard header per the
    Prompt-18 brief (surface model trust)."""
    dot = {"ok": "var(--value)", "warn": "var(--amber)", "err": "var(--danger)",
           "info": "var(--info)"}.get(tone, "var(--muted)")
    return (
        f'<span class="rp-trust"><span class="dot" style="background:{dot}"></span>'
        f'<span class="lab">{escape(label)}</span>'
        f'<span class="val">{escape(value)}</span></span>'
    )


def app_bar(title_html: str, subtitle: str, right_html: str = "") -> str:
    """Top app bar: serif title (with a quiet version tag), tabular subtitle, and
    a right cluster (trust badges). The mesh + diagonal rule lines come from CSS."""
    return (
        '<div class="rp-appbar"><div class="rp-appbar-row">'
        f'<div><h1>{title_html}</h1><div class="sub">{escape(subtitle)}</div></div>'
        f'<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">{right_html}</div>'
        '</div></div>'
    )


def kpi(label: str, value: str, sub: str = "", tone: str = "", small: bool = False,
        icon: str = "", icon_tone: str = "") -> str:
    """A KPI tile. Pass ``icon`` (a key from :mod:`ui._icons`) to lead the label
    with a tinted icon plate; ``icon_tone`` (info|value|amber|brand) sets the
    plate colour — keep it on the locked role for the metric (value→value, etc.).
    """
    vcls = "k-val"
    if small:
        vcls += " sm"
    if tone == "value":
        vcls += " value"
    sub_html = f'<div class="k-sub">{escape(sub)}</div>' if sub else ""
    lab = f'<div class="k-lab">{escape(label)}</div>'
    if icon:
        plate = (f'<span class="rp-ico-plate {escape(icon_tone)}">'
                 f'{_I.icon(icon, 16)}</span>')
        head = f'<div class="k-head">{plate}{lab}</div>'
    else:
        head = lab
    # the 2px accent on the tile's top edge, so a rail of tiles is scannable by
    # colour before it is readable by label (DESIGN.md §4.1)
    accent = icon_tone or tone
    tile_cls = f" t-{escape(accent)}" if accent else ""
    return (
        f'<div class="rp-kpi{tile_cls}">{head}'
        f'<div class="{vcls}">{escape(value)}</div>{sub_html}</div>'
    )


def kpi_rail(cards: list[str]) -> str:
    return f'<div class="rp-kpis">{"".join(cards)}</div>'


def section(label: str, note: str = "") -> str:
    note_html = f'<span class="n">{escape(note)}</span>' if note else ""
    return f'<div class="rp-sec">{escape(label)}{note_html}</div>'


def empty_state(title: str, body_html: str, hero: bool = False,
                illustration: str = "") -> str:
    """An empty / hero state.

    • ``illustration`` (pre-rendered SVG from :mod:`ui._icons`, e.g.
      ``horse_illustration()`` / ``finishline_scene()``) → the in-house CSS/SVG
      hero: the track-rail motif background (``hero-motif``) with the illustration
      centred. Preferred over a stock photo.
    • ``hero=True`` (no illustration) → the legacy photographic hero.
    • otherwise → a compact state led by the finish-flag mark.
    """
    if illustration:
        return (f'<div class="rp-empty hero-motif">{illustration}'
                f'<h3>{escape(title)}</h3><p>{body_html}</p></div>')
    cls = "rp-empty hero" if hero else "rp-empty"
    mk = (f'<div class="mk" style="color:var(--brand-text)">{mark_finishline(40)}</div>'
          if not hero else "")
    return f'<div class="{cls}">{mk}<h3>{escape(title)}</h3><p>{body_html}</p></div>'


def skeleton_rail(n: int = 4) -> str:
    """Shimmer placeholder for the KPI rail — the same grid shape as ``kpi_rail``
    so the real metrics drop into place without a layout shift."""
    n = max(1, min(6, int(n)))
    cards = "".join(
        '<div class="rp-skel-card kpi"><div class="sk sm w40" style="margin-bottom:12px"></div>'
        '<div class="sk lg"></div></div>'
        for _ in range(n)
    )
    return f'<div class="rp-skel-rail">{cards}</div>'


def skeleton_card() -> str:
    """One shimmer race/suggestion card (header strip + a few runner rows)."""
    rows = "".join(
        '<div style="display:flex;align-items:center;gap:12px">'
        '<div class="sk pill chip" style="width:22px;height:22px;flex:none"></div>'
        '<div class="sk w60" style="flex:1"></div>'
        '<div class="sk pill" style="width:88px;flex:none"></div>'
        '<div class="sk pill chip" style="flex:none"></div></div>'
        for _ in range(4)
    )
    return (
        '<div class="rp-skel-card">'
        '<div class="rp-skel-hdr"><div class="sk w25" style="height:16px"></div>'
        '<div class="sk pill chip"></div></div>'
        f'<div class="rp-skel-body">{rows}</div></div>'
    )


def loading_skeleton(label: str = "Working…", n_cards: int = 2,
                     rail: int = 4) -> str:
    """Full loading state: a captioned spinner-mark, a KPI-rail skeleton, and a
    couple of card skeletons. Rendered into an ``st.empty()`` placeholder while a
    slow path (predictor inference, value scan) runs, then cleared."""
    cards = "".join(skeleton_card() for _ in range(max(1, n_cards)))
    return (
        f'<div class="rp-skel-cap">{mark_loader(22)}<span>{escape(label)}</span></div>'
        f'{skeleton_rail(rail)}{cards}'
    )


def footer(left_html: str, right_html: str) -> str:
    return (
        f'<div class="rp-footer"><span>{left_html}</span>'
        f'<span class="mono">{right_html}</span></div>'
    )


def status_dot(tone: str = "ok") -> str:
    return f'<span class="sdot sdot-{tone}"></span>'


# ── bet-suggestion card (Today's suggestions view) ───────────────────────────

def tier_badge(tier: Optional[str]) -> str:
    """Confidence-tier chip. Both tiers are *value* bets, so they stay on the
    green value channel — Strong is filled/bold, Lean is a quiet outline. The
    word (STRONG / LEAN) carries the meaning; colour only reinforces it."""
    t = (tier or "").strip().lower()
    if t == "strong":
        return '<span class="tier tier-strong"><span class="td"></span>STRONG</span>'
    if t == "lean":
        return '<span class="tier tier-lean"><span class="td"></span>LEAN</span>'
    return f'<span class="tier tier-pass">{escape(tier or "—")}</span>'


def _stat(label: str, value_html: str, tone: str = "") -> str:
    cls = "sg-stat" + (f" {tone}" if tone else "")
    return (f'<div class="{cls}"><div class="sg-lab">{escape(label)}</div>'
            f'<div class="sg-val">{value_html}</div></div>')


def suggestion_card(s: dict, ccy: str, why_html: str = "") -> str:
    """Full "today's suggestion" card for one pick.

    Pure markup over the ``sg-*`` / ``rp-*`` classes — the caller renders the
    one-click paper-bet form (a Streamlit widget) directly beneath it. ``why_html``
    is pre-rendered SHAP driver bars (``why_drivers``); pass "" to omit them.
    """
    horse = escape(s.get("horse_name") or "—")
    venue = escape(s.get("venue") or "—")
    rtime = fmt_time(s.get("race_time"))
    conn = " · ".join(x for x in (
        f"J: {escape(s.get('jockey'))}" if s.get("jockey") else "",
        f"T: {escape(s.get('trainer'))}" if s.get("trainer") else "",
    ) if x) or "Connections unavailable"

    offered = s.get("offered_odds")
    best_book = s.get("best_book")
    star = " ★" if best_book else ""
    book_lab = (f' <span class="rp-sub">{escape(_BOOK_LABELS.get(best_book, str(best_book)[:5].title()))}</span>'
                if best_book else "")
    offered_html = (f'<span class="rp-odds">{escape(to_fraction(offered))}{star}</span>{book_lab}'
                    if offered else '<span class="rp-sub">no price</span>')
    fair_html = escape(to_fraction(s.get("fair_odds")))

    edge_pct = s.get("edge_pct")
    edge_html = (f'<span class="ev-pos">{edge_pct * 100:+.0f}%</span>'
                 if edge_pct is not None else "—")
    ev = s.get("expected_value")
    ev_html = ev_text(ev, True)
    stake = s.get("suggested_stake")
    stake_html = f'{ccy}{stake:,.2f}' if stake is not None else "—"
    win_html = fmt_pct(s.get("model_prob"))

    market = escape(s.get("market") or "Win")
    market_pill = pill(market, "field")
    ew_pill = pill("E/W avail.", "ew") if s.get("each_way_available") else ""

    reasons = s.get("tier_reasons") or []
    tier_title = escape(" · ".join(reasons)) if reasons else ""

    open_link = (
        f'<a class="rp-open" href="Race_Detail?race={escape(s.get("race_slug") or "")}" '
        f'target="_self">Card &amp; drivers →</a>'
    )
    header = (
        '<div class="rp-card-hdr">'
        f'{rank_badge(s.get("rank"))}'
        f'<span class="venue" style="font-size:var(--t-body)">{horse}</span>'
        f'<span title="{tier_title}">{tier_badge(s.get("tier"))}</span>'
        f'{market_pill}{ew_pill}'
        f'<span class="spacer"></span>{open_link}'
        '</div>'
    )

    sub = (f'<div class="sg-sub"><span class="venue">{venue}</span>'
           f'<span class="rtime">{rtime}</span><span class="rp-sub">{conn}</span></div>')

    stats = (
        '<div class="sg-stats">'
        f'{_stat("Offered", offered_html)}'
        f'{_stat("Fair", fair_html)}'
        f'{_stat("Edge", edge_html, "value")}'
        f'{_stat("EV", ev_html, "value")}'
        f'{_stat("Win %", win_html)}'
        f'{_stat("Stake", stake_html)}'
        f'{_stat("Confidence", confidence_chip(_conf_level(s.get("data_confidence"))) or "—")}'
        '</div>'
    )

    rationale = escape(s.get("rationale") or "")
    rat_html = (f'<div class="sg-rationale">{rationale}</div>' if rationale else "")
    why_block = (f'<div class="sg-why">{why_html}</div>' if why_html else "")

    return (f'<div class="rp-card sg-card">{header}'
            f'<div class="sg-body">{sub}{stats}{rat_html}{why_block}</div></div>')


def _conf_level(conf: Optional[str]) -> Optional[str]:
    """Normalise a predictor confidence string to the chip's high|med|low."""
    if not isinstance(conf, str) or not conf:
        return None
    c = conf.lower()
    return "med" if c in ("med", "medium") else (c if c in ("high", "low") else None)


# ── SHAP "why" drivers (detail view) ──────────────────────────────────────────

def driver_source_note(source: Optional[str]) -> str:
    """Honest provenance line for the SHAP drivers: whether they reflect today's
    race (live feature store) or the horse's most recent run profile (history)."""
    if source == "live":
        return ('<p class="rp-sub" style="margin:6px 0 0">Drivers computed for '
                'today&rsquo;s race.</p>')
    if source == "history":
        return ('<p class="rp-sub" style="margin:6px 0 0">Drivers from this '
                'horse&rsquo;s most recent run profile (no live feature row yet).</p>')
    return ""


def why_drivers(explanation: Optional[dict], max_rows: int = 5) -> str:
    """Diverging SHAP bars: green pushes win-chance up, danger-red pulls it down.
    Bars are normalised to the largest absolute contribution in the set so the
    longest bar fills the half-track. Falls back to an honest message when no
    explanation is available (synthetic feature matrix — out of scope to fix)."""
    if not explanation:
        return (
            '<p class="rp-sub" style="margin:0">Driver breakdown unavailable for '
            'this runner — the feature store has no row for it yet.</p>'
        )
    pos = explanation.get("top_positive") or []
    neg = explanation.get("top_negative") or []
    merged = []
    for d in pos:
        merged.append((d, 1))
    for d in neg:
        merged.append((d, -1))
    if not merged:
        return '<p class="rp-sub" style="margin:0">No strong drivers for this runner.</p>'
    merged.sort(key=lambda t: abs(t[0].get("contribution", 0.0)), reverse=True)
    merged = merged[:max_rows]
    peak = max((abs(d.get("contribution", 0.0)) for d, _ in merged), default=1.0) or 1.0

    rows = []
    for d, sign in merged:
        c = float(d.get("contribution", 0.0))
        w = abs(c) / peak * 50.0  # half-track is 50%
        bar_cls = "pos" if c > 0 else "neg"
        val = d.get("value")
        val_html = f' <span class="wv">{val:g}</span>' if isinstance(val, (int, float)) else ""
        label = escape(str(d.get("label") or d.get("feature") or "—"))
        rows.append(
            f'<div class="why-row"><div class="why-lab">{label}{val_html}</div>'
            f'<div class="why-bar"><i class="{bar_cls}" style="--bw:{w:.0f}%"></i></div>'
            f'</div>'
        )
    return f'<div class="why">{"".join(rows)}</div>'


# ── recent-form table (horse detail) ──────────────────────────────────────────

def winners_table(rows: list[dict]) -> str:
    """Today's ranked picks against where they actually finished.

    The finishing position is the point, so it leads the row after the pick.
    A price is a ratio here too: both our price and the SP show as board
    fractions, never with a currency symbol.
    """
    if not rows:
        return ""
    body = []
    for r in rows:
        pos = r.get("position")
        won, placed = r.get("won"), r.get("placed")
        row_cls = " class=\"is-value\"" if won else ""
        kind = "E/W" if r.get("each_way") else "Win"
        body.append(
            f'<tr{row_cls}>'
            f'<td data-label="#">{rank_badge(r.get("rank"))}</td>'
            f'<td data-label="Runner"><div class="rp-horse">{escape(str(r.get("horse") or "—"))}</div>'
            f'<span class="rp-sub">{escape(str(r.get("venue") or ""))} '
            f'{fmt_time(r.get("race_time"))}</span></td>'
            f'<td data-label="Our price">{price_html(r.get("price"))}</td>'
            f'<td data-label="SP">{price_html(r.get("sp")) if r.get("sp") else "<span class=\"rp-sub\">—</span>"}</td>'
            f'<td data-label="Bet"><span class="rp-sub">{kind}</span></td>'
            f'<td class="num" data-label="Finished">{_pos_badge(pos, won, placed)}</td>'
            f'</tr>'
        )
    return (
        '<div class="rp-card"><div class="rp-body">'
        '<table class="rp-tbl"><thead><tr>'
        '<th>#</th><th>Runner</th><th>Our price</th><th>SP</th><th>Bet</th>'
        '<th class="num">Finished</th></tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div></div>'
    )


def _pos_badge(pos: Optional[int], won: Optional[bool], placed: Optional[bool]) -> str:
    """Finishing-position chip. 1st is the brand (a win — never the value green),
    a place is amber, an also-ran is a neutral outline. Number carries it."""
    if pos is None:
        return '<span class="fp fp-n">–</span>'
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(pos if pos < 4 else 0, "th")
    if pos == 1 or won:
        cls = "fp1"
    elif placed or pos <= 3:
        cls = "fpp"
    else:
        cls = "fp-n"
    return f'<span class="fp {cls}">{pos}<sup>{suffix}</sup></span>'


def fmt_speed(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.0f}"


def form_table(runs: list[dict], ccy: str = "") -> str:
    """Recent-form table: one row per past run (most recent first).

    Each run: finishing position (as a chip, with field size), date, course,
    going, class, distance, speed figure and starting price — the canonical form
    line a punter scans. Empty / partial fields render as ``—`` rather than being
    fabricated. ``runs`` is the output of :func:`ui._form.horse_history_from`.
    """
    if not runs:
        return ('<p class="rp-sub" style="margin:0">No prior runs on record for this '
                'horse — it resolves to the model&rsquo;s base rate (a debutant or an '
                'unmatched runner).</p>')
    rows = []
    for r in runs:
        pos = r.get("position")
        fld = r.get("field_size")
        pos_cell = _pos_badge(pos, r.get("won"), r.get("placed"))
        of_fld = f'<span class="rp-sub"> /{fld}</span>' if fld else ""
        sp = r.get("sp")
        sp_html = f"{ccy}{sp:.2f}" if isinstance(sp, (int, float)) else "—"
        rows.append(
            "<tr>"
            f'<td>{pos_cell}{of_fld}</td>'
            f'<td class="mono">{escape(r.get("date") or "—")}</td>'
            f'<td>{escape(r.get("venue") or "—")}</td>'
            f'<td>{escape(r.get("going") or "—")}</td>'
            f'<td>{escape(r.get("race_class") or "—")}</td>'
            f'<td>{escape(r.get("distance") or "—")}</td>'
            f'<td class="num mono">{fmt_speed(r.get("speed"))}</td>'
            f'<td class="num mono">{sp_html}</td>'
            "</tr>"
        )
    return (
        '<div class="rp-body"><table class="rp-tbl rp-form"><thead><tr>'
        '<th>Finish</th><th>Date</th><th>Course</th><th>Going</th><th>Class</th>'
        '<th>Dist</th><th class="num">Speed</th><th class="num">SP</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
    )


# ── stat grid (connections / aggregate form) ──────────────────────────────────

def stat_grid(stats: list[tuple[str, str]]) -> str:
    """Compact label/value grid (reuses the suggestion-card ``sg-stat`` cells).

    ``stats`` is a list of ``(label, value_html)`` pairs already formatted by the
    caller; pairs whose value is falsy are dropped so empty data doesn't show."""
    cells = [_stat(lab, val) for lab, val in stats if val not in (None, "", "—")]
    if not cells:
        return ('<p class="rp-sub" style="margin:0">No connection or form '
                'aggregates available.</p>')
    return f'<div class="sg-stats" style="border:none;padding:0">{"".join(cells)}</div>'


def fmt_rate(v: Optional[float]) -> str:
    """A win/place RATE stored as a 0–1 fraction → percent, or ``—``."""
    if v is None or not isinstance(v, (int, float)):
        return "—"
    return f"{max(0.0, min(1.0, float(v))) * 100:.0f}%"


# ── model's view (race summary) ───────────────────────────────────────────────

def model_view(summary: dict) -> str:
    """Plain-language "model's view" panel for a race.

    ``summary`` carries pre-derived strings/values; this only renders. Keeps the
    headline read (top pick + how clear-cut the race is) above the per-runner
    detail, per the brief's "clear model's-view summary"."""
    top = escape(summary.get("top_name") or "—")
    top_pct = summary.get("top_win_pct")
    top_pct_html = (f'<span class="mv-num">{top_pct}</span>'
                    if top_pct else "")
    shape = escape(summary.get("shape") or "")
    n_value = summary.get("n_value") or 0
    value_line = ""
    if n_value:
        plural = "s" if n_value != 1 else ""
        value_line = (f'<div class="mv-row"><span class="vd"></span>'
                      f'<b>{n_value}</b> value bet{plural} vs the market in this race</div>')
    note = escape(summary.get("note") or "")
    note_html = f'<div class="mv-note">{note}</div>' if note else ""
    return (
        '<div class="rp-modelview">'
        '<div class="mv-lab">Model&rsquo;s view</div>'
        f'<div class="mv-lead">Forecast: <b>{top}</b> {top_pct_html}'
        f'<span class="mv-shape"> · {shape}</span></div>'
        f'{value_line}{note_html}'
        '</div>'
    )


def context_chips(items: list[tuple[str, str]]) -> str:
    """Row of labelled context chips (e.g. Going / Distance / Pace projection).

    ``items`` is ``(label, value)``; falsy values are skipped."""
    chips = []
    for lab, val in items:
        if not val:
            continue
        chips.append(
            f'<span class="ctx"><span class="ctx-l">{escape(lab)}</span>'
            f'<span class="ctx-v">{escape(val)}</span></span>'
        )
    return f'<div class="ctx-row">{"".join(chips)}</div>' if chips else ""
