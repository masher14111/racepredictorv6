"""Streamlit UI — MASHR’s Predictor v6 · Today / race-list landing.

The canonical prediction surface (Prompt 19 redesign). Dark "betting-desk"
theme via the shared design system in :mod:`ui._design`; markup assembled from
the named primitives in :mod:`ui._components`. Decision logic (date default,
empty-state selection) lives in :mod:`ui.logic` so it stays headless-testable.

Launch:
    streamlit run ui/app.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, date, timedelta
from html import escape
from pathlib import Path
from typing import Optional

import streamlit as st

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now, to_local
from utils.currency import GBP_EUR_RATE, currency_for_venue, from_eur
from utils.stake_advice import recommend_stake
from ui.logic import (
    DATE_LABELS,
    default_date_index,
    empty_state,
    parse_race_date as _race_date,
)
from ui import _components as C
from ui import _icons as I
from ui._design import inject_design
from ui import _today as TD

# ── page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="MASHR’s Predictor v6",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="auto",
)

_PLOTLY_CFG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}

_CACHE_PATH = _ROOT / "data" / "predictions.json"
_META_PATH = _ROOT / "models" / "catboost_v3nf_meta.json"

inject_design(width="wide")



# ── data helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_predictions() -> Optional[dict]:
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_data(ttl=300)
def _model_trust() -> dict:
    """Held-out trust metrics for the app-bar badges (best-effort; {} if absent).

    AUC + calibration flag come from the price-free model's ``*_meta.json``
    (nested under ``targets.won``); the calibrated ECE comes from the calibration
    report CSV (``docs/calibration/metrics.csv``), which is where it's recorded."""
    out: dict = {}
    if _META_PATH.exists():
        try:
            with open(_META_PATH, encoding="utf-8") as f:
                won = (json.load(f).get("targets") or {}).get("won") or {}
            if won.get("test_auc") is not None:
                out["auc"] = float(won["test_auc"])
            out["calibrated"] = bool(won.get("calibrated"))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            pass
    csv_path = _ROOT / "docs" / "calibration" / "metrics.csv"
    if csv_path.exists():
        try:
            import csv
            with open(csv_path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if (r.get("model") == "v3nf" and r.get("target") == "won"
                            and r.get("variant") == "calibrated" and r.get("ece")):
                        out["ece"] = float(r["ece"])
                        break
        except (OSError, ValueError):
            pass
    return out


@st.cache_data(ttl=300)
def _load_results():
    """Finishing positions, from the SAME store settlement uses.

    ``scripts/daily_paper_loop._load_results`` reads ``storage.read_parquet("betsp")``,
    so this does too: the derived ``unified_races.parquet`` is rebuilt on its own
    schedule and lagged the results archive by eight weeks, which would have made
    this panel disagree with the ledger sitting beside it.

    Only WIN rows — the archive carries a WIN and a PLACE row per runner, and
    counting both would double every race.
    """
    import pandas as pd
    try:
        from utils.storage import get_storage
        frame = get_storage().read_parquet("betsp")
    except Exception:  # noqa: BLE001 - a missing archive is non-fatal here
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    cols = [c for c in ("venue", "race_date", "horse_name", "position",
                        "odds_finish", "market_type") if c in frame.columns]
    out = frame[cols]
    if "market_type" in out.columns:
        out = out[out["market_type"].astype(str).str.upper() == "WIN"]
    return out.rename(columns={"race_date": "race_time", "odds_finish": "sp"})


def _run_predictor() -> dict:
    """Run the predictor and report its outcome.

    Returns ``{"status", "data", "message"}`` where ``status`` is
    ``ok`` | ``no_models`` | ``empty`` | ``error``. The caller paints a loading
    skeleton while this runs, so no inline spinner is needed here."""
    try:
        from models.predictor import Predictor
        p = Predictor()
        if not p.load():
            return {"status": "no_models", "data": None,
                    "message": "No trained models found. "
                               "Run `python -m models.train` first."}
        races = p.predict()
        _load_predictions.clear()
        data = _load_predictions()
        if not races:
            return {"status": "empty", "data": data,
                    "message": "Predictor ran but found 0 upcoming races — "
                               "scrape today's racecards first."}
        return {"status": "ok", "data": data,
                "message": f"Predictor ran — {len(races)} upcoming "
                           f"race{'s' if len(races) != 1 else ''} found."}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "data": None,
                "message": f"Predictor error: {exc}"}


def _fmt_generated_at(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is not None:
            dt = to_local(dt)
        return dt.strftime("%d %b %Y %H:%M IST")
    except (ValueError, TypeError):
        return iso or "—"


def _chart_card(title: str, note: str, fig, key: str) -> None:
    """A titled chart in a bordered card, matching the rp-card treatment.

    The title lives in the card header rather than inside the plot: it reads
    better, and it frees the figure's top margin on a phone (DESIGN.md §10.3).
    """
    with st.container(border=True):
        st.markdown(
            f'<div class="rp-chart-hd"><span class="t">{escape(title)}</span>'
            f'<span class="n">{escape(note)}</span></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, width="stretch", config=_PLOTLY_CFG, key=key)


# ── race card ─────────────────────────────────────────────────────────────────

def _render_race_card(race: dict, min_score: float, sort_by: str) -> str:
    """Build one race card as markup and **return** it, so the caller can join a
    venue's cards into a single ``rp-card-grid`` and flush them in one
    ``st.markdown``. That keeps the grid pure CSS — gridding Streamlit's internal
    ``stVerticalBlock`` wrappers would tie the layout to testids that are not a
    public API."""
    venue = race.get("venue") or "Unknown"
    field_size = race.get("field_size") or 0
    ew_available = race.get("each_way_available", False)
    selections = race.get("selections") or []

    ccy = currency_for_venue(venue)
    ccy_symbol = "£" if ccy == "GBP" else "€"

    visible = [s for s in selections if (s.get("composite_score") or 0.0) >= min_score]
    if sort_by == "Value edge":
        visible = sorted(
            visible, key=lambda s: (s.get("value_edge") is None, -(s.get("value_edge") or 0))
        )

    has_value = any(s.get("expected_value") is not None for s in visible)
    n_value_bets = sum(1 for s in visible if s.get("value_bet"))

    # ── header ──
    hdr_pills = (
        f'{C.pill(f"{field_size} runners", "field")}'
        f'{C.pill(ccy_symbol, "ccy")}'
        f'{(C.pill("E/W", "ew") if ew_available else "")}'
        f'{(C.pill(f"{n_value_bets} value", "value") if n_value_bets else "")}'
    )
    open_link = (
        f'<a class="rp-open" href="Race_Detail?race={C.race_slug(race)}" target="_self">'
        f'Card &amp; drivers →</a>'
    )
    header = (
        '<div class="rp-card-hdr">'
        f'<span class="venue">{escape(venue)}</span>'
        f'<span class="rtime">{C.fmt_time(race.get("race_time"))}</span>'
        f'{hdr_pills}<span class="spacer"></span>{open_link}'
        '</div>'
    )

    # ── runner rows ──
    rows = []
    for sel in visible:
        horse = escape(sel.get("horse_name") or "—")
        jockey = sel.get("jockey") or ""
        jockey_html = (f'<span class="rp-sub">{escape(jockey)}</span>'
                       if jockey else '<span class="rp-sub">—</span>')

        badges = ""
        if sel.get("value_bet"):
            badges += " " + C.value_badge(sel.get("value_edge"))
        if sel.get("each_way_value"):
            badges += " " + C.ew_badge()

        odds_val = sel.get("best_odds") or sel.get("decimal_odds")
        # A price is a ratio, so it shows as a board fraction with no currency
        # symbol and no FX. Money (the stake below) is what gets converted.
        odds_cell = C.price_html(odds_val) + C.book_chips(
            sel.get("odds_by_book"), sel.get("best_book"))

        conf = C.confidence_chip(C.confidence_for(sel))
        ev_cell = (f'<td class="num" data-label="EV">'
                   f'{C.ev_text(sel.get("expected_value"), sel.get("value_bet"))}</td>'
                   if has_value else "")

        # Recommended PAPER stake — fractional Kelly on the value layer's
        # market-aware probability, vetoed by its own EV so the stake can never
        # contradict the EV shown in the next column.
        advice = recommend_stake(
            sel.get("value_win_prob") or C.headline_win_prob(sel),
            odds_val,
            market_ev=sel.get("expected_value"),
        )
        if advice.is_bet:
            # The stake is budgeted in EUR and odds are a ratio, so the EUR
            # return is simply stake x price — no FX enters. At a GBP venue the
            # book still debits and pays in pounds, so show that too.
            ret_eur = advice.stake * float(odds_val)
            tip = (
                f"{advice.reason}. Returns about €{ret_eur:,.2f} if it wins."
            )
            if ccy == "GBP":
                tip += (
                    f" At this UK venue the book takes about "
                    f"£{from_eur(advice.stake, ccy):,.2f} and pays "
                    f"£{from_eur(ret_eur, ccy):,.2f} "
                    f"(rate {GBP_EUR_RATE:.2f})."
                )
            # The sub-label is the bet TYPE the model recommends, not the Kelly
            # size band — "Token" read as a bet type to anyone who had not seen
            # the sizing ladder. The band still travels, in the tooltip.
            bet_kind = "E/W" if sel.get("each_way_value") else "Win"
            stake_cell = (
                f'<td class="num" data-label="Paper stake"><span class="rp-stake" '
                f'title="{escape(advice.band)} stake. {escape(tip)} Paper only.">'
                f'€{advice.stake:,.2f}'
                f'</span><div class="rp-sub">{bet_kind}</div></td>'
            )
        else:
            stake_cell = (
                f'<td class="num" data-label="Paper stake"><span class="rp-sub" '
                f'title="{escape(advice.reason)}">—</span></td>'
            )

        row_cls = ' class="is-value"' if sel.get("value_bet") else ""
        # data-label drives the stacked mobile layout: below 420px the table
        # collapses to per-runner blocks and `td::before { content: attr(...) }`
        # supplies the inline label. See DESIGN.md §9.
        rows.append(
            f'<tr{row_cls}>'
            f'<td data-label="#">{C.rank_badge(sel.get("rank"))}</td>'
            f'<td data-label="Runner"><div class="rp-horse">{horse}{badges}</div>'
            f'{jockey_html}</td>'
            f'<td data-label="Best price">{odds_cell}</td>'
            f'<td data-label="Win %">{C.prob_bar(C.headline_win_prob(sel), "win")}</td>'
            f'<td data-label="Conf.">{conf}</td>'
            f'{ev_cell}'
            f'{stake_cell}'
            f'</tr>'
        )

    n_cols = 7 if has_value else 6
    if not rows:
        rows.append(
            f'<tr><td colspan="{n_cols}" style="text-align:center;color:var(--muted);'
            'padding:16px">No runners above the score threshold</td></tr>'
        )

    ev_head = '<th class="num">EV</th>' if has_value else ""
    stake_head = (
        '<th class="num" title="Fractional-Kelly paper stake on the calibrated '
        'win probability. Paper only — never a real-money recommendation.">'
        'Paper stake</th>'
    )
    body = (
        '<div class="rp-body"><table class="rp-tbl"><thead><tr>'
        '<th>#</th><th>Runner</th><th>Best price</th><th>Win&nbsp;%</th>'
        f'<th>Conf.</th>{ev_head}{stake_head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )

    return f'<div class="rp-card">{header}{body}</div>'


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if "predictions" not in st.session_state:
        st.session_state.predictions = _load_predictions()

    preds: Optional[dict] = st.session_state.predictions
    races_all: list[dict] = (preds or {}).get("races") or []
    today_dublin = now().date()

    # ── sidebar ──
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)

        st.markdown("### Date")
        date_choice = st.selectbox(
            "Show races for", DATE_LABELS,
            index=default_date_index(races_all, today_dublin),
            label_visibility="collapsed",
        )
        filter_date: Optional[date] = None
        if date_choice == "Today":
            filter_date = today_dublin
        elif date_choice == "Tomorrow":
            filter_date = today_dublin + timedelta(days=1)

        st.markdown("### Venue")
        venues_available = sorted({r.get("venue") for r in races_all if r.get("venue")})
        selected_venues = st.multiselect(
            "Venues", options=venues_available, default=venues_available,
            label_visibility="collapsed",
        )

        st.markdown("### Runners")
        sort_by = st.selectbox(
            "Sort runners by", ["Rank", "Value edge"], index=0,
            help="Order runners within each race by model rank, or by value edge "
                 "(model probability minus the market-implied probability).",
        )
        min_score = st.slider(
            "Min composite score", 0.0, 1.0, 0.0, 0.01,
            help="Composite = 0.5×win + 0.3×placed_2 + 0.2×showed. "
                 "Runners below this value are hidden.",
        )
        strategy = st.selectbox(
            "Stake strategy", ["flat", "kelly", "fractional_kelly"], index=0,
            help="Stored in session for BetTracker integration.",
        )
        st.session_state["strategy"] = strategy

        st.markdown("---")
        if st.button("Refresh predictions"):
            # defer the work to a dedicated loading frame so the main pane can
            # paint a skeleton while the ~1.7s inference runs (see main()).
            st.session_state["_refreshing"] = True
            st.rerun()

        gen_at = (preds or {}).get("generated_at")
        if gen_at:
            try:
                dt = datetime.fromisoformat(gen_at)
                if dt.tzinfo:
                    dt = to_local(dt)
                st.caption(f"Updated {dt.strftime('%d %b %H:%M')} IST")
            except (ValueError, TypeError):
                pass
        elif not preds:
            st.caption("No prediction cache found.")

    # ── apply filters ──
    races = list(races_all)
    if filter_date:
        races = [r for r in races if _race_date(r.get("race_time")) == filter_date]
    if venues_available:
        races = [r for r in races if r.get("venue") in selected_venues]

    # ── app bar (brand title + model-trust badges) ──
    trust = _model_trust()
    badges = []
    if trust.get("auc") is not None:
        badges.append(C.trust_badge("Win AUC", f"{trust['auc']:.3f}", "info"))
    if trust.get("ece") is not None:
        ece = trust["ece"]
        tone = "ok" if ece <= 0.03 else ("warn" if ece <= 0.08 else "err")
        badges.append(C.trust_badge("Calibration ECE", f"{ece:.3f}", tone))
    if not badges:
        targets = ", ".join((preds or {}).get("model_targets") or []) or "—"
        badges.append(C.trust_badge("Models", targets, "ok" if preds else "warn"))

    local_now = now()
    title = ('<span class="wm-a">MASHR&rsquo;s</span> '
             '<span class="wm-b">Predictor</span>'
             '<span class="v">v6</span>')
    subtitle = f'{local_now.strftime("%A, %d %B %Y")} · {local_now.strftime("%H:%M")} IST'
    st.markdown(C.app_bar(title, subtitle, "".join(badges)), unsafe_allow_html=True)

    # ── inference loading state ──
    # Drawn into a placeholder that's flushed to the browser *before* the blocking
    # predictor call, so the wait reads as a working skeleton, not a frozen pane.
    if st.session_state.pop("_refreshing", False):
        ph = st.empty()
        ph.markdown(
            C.loading_skeleton("Running the predictor over today’s racecards…"),
            unsafe_allow_html=True,
        )
        outcome = _run_predictor()
        st.session_state["refresh_outcome"] = outcome
        if outcome["data"] is not None:
            st.session_state.predictions = outcome["data"]
        ph.empty()
        st.rerun()

    # ── refresh-outcome banner ──
    outcome = st.session_state.pop("refresh_outcome", None)
    if outcome:
        {"ok": st.success, "empty": st.warning}.get(
            outcome["status"], st.error)(outcome["message"])

    # ── staleness banner ──
    gen_date = _race_date((preds or {}).get("generated_at"))
    if preds and gen_date is not None and gen_date < today_dublin:
        st.warning(
            f"Showing a stale cache generated "
            f"**{_fmt_generated_at(preds.get('generated_at'))}** "
            f"({(today_dublin - gen_date).days} day(s) ago). "
            f"Refresh after scraping today's racecards."
        )

    # ── KPI rail ──
    def _all_runners(r: dict) -> list[dict]:
        # The complete validated field (audit req 2/7): value counts must cover
        # every runner, not just the display top-3 + low-odds tail — a mid-field
        # value bet would otherwise be invisible here while the suggestion
        # engine (which consumes `runners`) still surfaces it. Older caches
        # without `runners` fall back to the display lists.
        full = r.get("runners")
        if full:
            return full
        return (r.get("selections") or []) + (r.get("excluded_low_odds") or [])

    total_selections = sum(len(r.get("selections") or []) for r in races)
    ew_count = sum(1 for r in races if r.get("each_way_available"))
    has_value_data = any(
        s.get("expected_value") is not None for r in races for s in _all_runners(r))
    value_count = sum(1 for r in races for s in _all_runners(r) if s.get("value_bet"))

    cards = [C.kpi("Races", str(len(races)), "upcoming",
                   icon="race", icon_tone="info")]
    if has_value_data:
        cards.append(C.kpi("Value bets", str(value_count),
                           "positive EV vs market", tone="value",
                           icon="value", icon_tone="value"))
    cards.append(C.kpi("Selections", str(total_selections), "top picks per race",
                       icon="trophy", icon_tone="brand"))
    cards.append(C.kpi("Each-way", str(ew_count), "races flagged",
                       icon="selection", icon_tone="amber"))
    st.markdown(C.kpi_rail(cards), unsafe_allow_html=True)

    # ── today's shape ──
    # The pre-race half of the desk: where the value sits, where the model
    # disagrees with the market, and who is pricing best. Charts read the FULL
    # field (see ui/_today.field), not the top-3 display list.
    if races:
        st.markdown(C.section("Today’s shape",
                              "the card before it runs"), unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            _chart_card("Value by odds band",
                        "mean EV vs the price on offer",
                        TD.ev_by_odds_band_fig(TD.odds_band_rows(races), height=208),
                        "today-ev-band")
        with c2:
            _chart_card("Model vs market",
                        "above the line = we rate it higher",
                        TD.model_vs_market_fig(TD.model_vs_market_rows(races), height=208),
                        "today-model-market")
        with c3:
            _chart_card("Best price held by",
                        "across every runner on the card",
                        TD.best_book_fig(TD.best_book_rows(races), height=208),
                        "today-best-book")

    # ── race cards (venue-grouped) ──
    if not races:
        title_, reason = empty_state(preds, races_all, races)
        # In-house CSS/SVG hero (track-rail motif + horse line-art) — no stock
        # photo. The horse leads the cold-start splash; once a cache exists but a
        # filter empties the list, the lighter finish-line scene reads as "run".
        illus = I.horse_illustration(150) if not preds else I.finishline_scene(160)
        st.markdown(C.empty_state(title_, reason, illustration=illus),
                    unsafe_allow_html=True)
    else:
        by_venue: dict[str, list] = {}
        for race in sorted(races, key=lambda r: (r.get("venue") or "",
                                                 r.get("race_time") or "")):
            by_venue.setdefault(race.get("venue") or "Unknown", []).append(race)

        for vi, (venue, vraces) in enumerate(by_venue.items()):
            n = len(vraces)
            span = (f" · {C.fmt_time(vraces[0].get('race_time'))}"
                    f"–{C.fmt_time(vraces[-1].get('race_time'))}" if vraces else "")
            n_value = sum(1 for r in vraces for s in _all_runners(r) if s.get("value_bet"))
            value_tag = f" · {n_value} value" if n_value else ""
            with st.expander(f"{venue} — {n} race{'s' if n != 1 else ''}{span}{value_tag}",
                             expanded=(vi == 0)):
                cards = [_render_race_card(race, min_score=min_score,
                                           sort_by=sort_by)
                         for race in vraces]
                st.markdown(f'<div class="rp-card-grid">{"".join(cards)}</div>',
                            unsafe_allow_html=True)

    # ── today's winners ──
    # Our ranked picks against where the runners actually finished. An empty
    # result here is ambiguous — a card that has not run looks identical to a
    # results feed that has stopped — so the empty state always says which.
    _results = _load_results()
    _win_rows = TD.winners_rows(_results, races)
    _sum = TD.winners_summary(_win_rows)
    _hdr_note = (f'{_sum["wins"]} won · {_sum["places"]} placed '
                 f'from {_sum["picks"]} top picks' if _win_rows
                 else "where the picks actually finished")
    st.markdown(C.section("Today’s winners", _hdr_note), unsafe_allow_html=True)
    if _win_rows:
        st.markdown(C.winners_table(_win_rows), unsafe_allow_html=True)
    else:
        _last = TD.last_results_day(_results)
        _today_d = now().date()
        _gap = None if _last is None else (_today_d - _last).days
        if _last is None:
            _why = "No finishing positions have been recorded yet."
        elif _gap <= 1:
            # yesterday's results are in; today's card simply has not run
            _why = ("Today’s races have not been settled yet — results through "
                    f"{_last:%d %b} are in. Check back after the card runs.")
        else:
            _why = (f"The results feed last recorded a finish on "
                    f"{_last:%d %b %Y} — {_gap} days ago. Nothing here is "
                    f"waiting on today’s racing; the feed itself is stale.")
        st.markdown(C.empty_state("No results for today", _why), unsafe_allow_html=True)

    # ── is the model honest? ──
    # The settled half of the desk, pulled forward from the Dashboard page so
    # the question that should govern every bet is on the page you open.
    # Guarded: a missing ledger must degrade to a calm empty chart, never take
    # the racecards down with it.
    st.markdown(C.section("Is the model honest?",
                          "settled paper bets · full history"),
                unsafe_allow_html=True)
    h1, h2 = st.columns(2)
    try:
        from ui._betting import get_tracker
        from ui import dashboard as DB
        _tracker = get_tracker()
        _settled = DB.settled_only(DB.load_paper_bets(_tracker))
        _initial = float(_tracker.initial_bankroll)
        with h1:
            _chart_card("Bankroll equity curve", "paper only · never real money",
                        DB.equity_curve_fig(_settled, _initial, height=250), "home-equity")
        with h2:
            _chart_card("A/E calibration", "1.0 = the probabilities are honest",
                        DB.calibration_ae_fig(DB.paper_ae_rows(_settled), [], height=250),
                        "home-ae")
    except Exception as exc:  # noqa: BLE001 - the ledger is optional here
        with h1:
            st.caption(f"Performance charts unavailable: {exc}")

    # ── footer ──
    if preds:
        model_targets = (preds or {}).get("model_targets") or []
        tone = "ok" if model_targets else "warn"
        left = (f'{C.status_dot(tone)}Model targets: '
                f'{", ".join(model_targets) or "—"}')
        right = (f'Cache: {preds.get("total_races", 0)} races · '
                 f'Generated {_fmt_generated_at(preds.get("generated_at"))}')
        st.markdown(C.footer(left, right), unsafe_allow_html=True)


main()
