"""Horse detail — one runner's form, connections, and "why" breakdown.

Reached from a race card / race-detail runner via
``?horse=<horse_id>&race=<slug>``. Brings together the three things a punter
wants on a single horse:

* **Recent form** — its last runs (finishing position, field size, going, class,
  distance, speed figure, SP), from the labelled training matrix (:mod:`ui._form`).
* **Connections & aggregate form** — jockey / trainer / combo win rates, career
  runs, course & distance records, going preference.
* **The SHAP "why" breakdown** for the current race — the same per-prediction
  drivers the race page shows, here with the full context around them.

All form data is keyed on the same hashed ``horse_id`` the predictor emits, so a
runner with prior runs resolves to its real history. A debutant (or an unmatched
runner) honestly shows "no prior runs" rather than fabricated form.
"""
from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now  # noqa: E402
from utils.currency import currency_for_venue  # noqa: E402
from ui import _components as C  # noqa: E402
from ui import _form  # noqa: E402
from ui._betting import PAPER_NOTICE, place_bet_widget  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Horse Detail · MASHR’s Predictor v6",
                   page_icon="🏇", layout="wide", initial_sidebar_state="auto")

_CACHE_PATH = _ROOT / "data" / "predictions.json"

inject_design()



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
def _matrix() -> pd.DataFrame:
    return _form.load_matrix()


@st.cache_data(ttl=60)
def _inference() -> pd.DataFrame:
    return _form.load_inference()


@st.cache_resource
def _explainer():
    try:
        from models.explain import get_explainer
        return get_explainer(target="won", version_tag="v3nf")
    except Exception:  # noqa: BLE001
        return None


def _find_runner(race: Optional[dict], horse_id: str) -> Optional[dict]:
    """The runner dict for ``horse_id`` within ``race``.

    Prefers the complete ``runners`` field (audit req 2) so a mid-field runner —
    outside both the top-3 selections and the low-odds tail — still resolves;
    older caches without it fall back to the display lists."""
    if not race:
        return None
    pool = race.get("runners") or (
        (race.get("selections") or []) + (race.get("excluded_low_odds") or []))
    return next((s for s in pool if str(s.get("horse_id")) == str(horse_id)), None)


def _horse_name(sel: Optional[dict], matrix: pd.DataFrame, horse_id: str) -> str:
    if sel and sel.get("horse_name"):
        return sel["horse_name"]
    if not matrix.empty and "horse_name" in matrix.columns and horse_id:
        hit = matrix[matrix["horse_id"].astype(str) == str(horse_id)]
        if not hit.empty:
            nm = hit.iloc[0].get("horse_name")
            if isinstance(nm, str) and nm:
                return nm
    return "Unknown horse"


def main() -> None:
    horse_id = st.query_params.get("horse")
    race_slug = st.query_params.get("race")

    preds = _load_predictions()
    races = (preds or {}).get("races") or []
    race = C.find_race(races, race_slug) if race_slug else None

    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)
        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")

    if not horse_id:
        st.markdown(C.app_bar("Horse detail", now().strftime("%A, %d %B %Y"), ""),
                    unsafe_allow_html=True)
        body = ('Open a horse from a '
                '<a href="app.py" target="_self">racecard</a> — click a runner&rsquo;s '
                'name on the race detail page.')
        st.markdown(C.empty_state("No horse selected", body), unsafe_allow_html=True)
        return

    matrix = _matrix()
    sel = _find_runner(race, horse_id)
    history = _form.horse_history_from(matrix, horse_id, limit=10)
    stats = _form.connection_stats_from(matrix, horse_id)
    name = _horse_name(sel, matrix, horse_id)

    venue = (race or {}).get("venue") or (sel or {}).get("venue")
    ccy = "£" if (venue and currency_for_venue(venue) == "GBP") else "€"

    # ── header ──
    if race_slug:
        st.markdown(
            f'<a class="rp-back" href="Race_Detail?race={escape(race_slug)}" '
            f'target="_self">← Back to race</a>', unsafe_allow_html=True)
    n_runs = stats.get("runs") or len(history)
    badges = [C.trust_badge("Runs on record", str(n_runs), "info" if n_runs else "warn")]
    if sel and sel.get("value_bet"):
        badges.append(C.trust_badge("Value bet", "today", "ok"))
    title = escape(name)
    sub_bits = []
    if venue:
        sub_bits.append(escape(venue))
    if race:
        sub_bits.append(f'{C.fmt_time(race.get("race_time"))} today')
    sub = " · ".join(sub_bits) or "Form & connections"
    st.markdown(C.app_bar(title, sub, "".join(badges)), unsafe_allow_html=True)

    # ── today's line (if this horse runs in a loaded race) ──
    if sel:
        odds_val = sel.get("best_odds") or sel.get("decimal_odds")
        # Board fraction, no currency — a price is a ratio, not an amount.
        odds_html = (f'{escape(C.fmt_price(odds_val))} ({float(odds_val):.2f})'
                     if odds_val else "no price")
        conn = " · ".join(x for x in (
            f"J: {escape(sel.get('jockey'))}" if sel.get("jockey") else "",
            f"T: {escape(sel.get('trainer'))}" if sel.get("trainer") else "") if x
        ) or "Connections unavailable"
        st.markdown(C.section("Today's race"), unsafe_allow_html=True)
        st.markdown(
            '<div class="rp-card"><div class="rp-detail-grid" style="grid-template-columns:1fr 1fr">'
            f'<div><div class="rp-sub" style="margin-bottom:8px">{conn}</div>'
            f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:8px">'
            f'<span class="rp-odds" style="font-size:var(--t-md)">{odds_html}</span>'
            f'{C.value_badge(sel.get("value_edge")) if sel.get("value_bet") else ""}'
            f'{C.ew_badge() if sel.get("each_way_value") else ""}</div>'
            '<div style="display:grid;gap:8px">'
            f'<div class="why-row"><div class="why-lab">Win</div>{C.prob_bar(C.headline_win_prob(sel), "win")}</div>'
            f'<div class="why-row"><div class="why-lab">Place</div>{C.prob_bar(sel.get("placed_2_prob"), "place")}</div>'
            f'<div class="why-row"><div class="why-lab">Show</div>{C.prob_bar(sel.get("showed_prob"), "show")}</div>'
            '</div></div>'
            f'<div><div class="k-lab" style="margin-bottom:6px">Expected value</div>'
            f'<div style="font-size:var(--t-md);margin-bottom:10px">'
            f'{C.ev_text(sel.get("expected_value"), sel.get("value_bet"))}</div>'
            f'<div class="k-lab" style="margin-bottom:6px">Confidence</div>'
            f'{C.confidence_chip(C.confidence_for(sel)) or "—"}</div>'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # ── context chips: trend / career / going preference ──
    # The feature row (live store, else latest matrix row) carries speed_trend,
    # which is also what the SHAP breakdown below explains — resolve it once.
    resolved = _form.feature_row_from(matrix, _inference(), str(horse_id))
    ctx = []
    trend = _form.trend_label(resolved[0].get("speed_trend")) if resolved else None
    if trend:
        ctx.append(("Form trend", trend))
    if stats.get("horse_career_runs") is not None:
        ctx.append(("Career runs", f"{int(stats['horse_career_runs'])}"))
    if stats.get("days_since_last_run") is not None:
        ctx.append(("Days since run", f"{int(stats['days_since_last_run'])}"))
    if stats.get("going_pref_win_rate") is not None:
        ctx.append(("Going-pref win%", C.fmt_rate(stats["going_pref_win_rate"])))
    if ctx:
        st.markdown(C.context_chips(ctx), unsafe_allow_html=True)

    # ── recent form ──
    st.markdown(C.section("Recent form", f"last {len(history)} runs" if history else ""),
                unsafe_allow_html=True)
    st.markdown(f'<div class="rp-card">{C.form_table(history, ccy)}</div>',
                unsafe_allow_html=True)

    # ── connections & aggregate form ──
    st.markdown(C.section("Connections & record"), unsafe_allow_html=True)
    stat_pairs = [
        ("Jockey", escape(stats.get("jockey_name") or "—")),
        ("Trainer", escape(stats.get("trainer_name") or "—")),
        ("Jockey win%", C.fmt_rate(stats.get("jockey_win_rate"))),
        ("Trainer win%", C.fmt_rate(stats.get("trainer_win_rate"))),
        ("J+T combo win%", C.fmt_rate(stats.get("jt_combo_win_rate"))),
        ("Horse win%", C.fmt_rate(stats.get("historical_win_rate"))),
        ("Horse place%", C.fmt_rate(stats.get("historical_place_rate"))),
        ("Course win%", C.fmt_rate(stats.get("course_win_rate"))),
        ("Distance win%", C.fmt_rate(stats.get("distance_win_rate"))),
    ]
    st.markdown(f'<div class="rp-card" style="padding:var(--s5)">{C.stat_grid(stat_pairs)}</div>',
                unsafe_allow_html=True)

    # ── why the model rates it (SHAP) ──
    st.markdown(C.section("Why the model rates this runner"), unsafe_allow_html=True)
    why_html = None
    source = None
    if resolved is not None and _explainer() is not None:
        try:
            why_html = _explainer().explain(resolved[0], top_k=6)
            source = resolved[1]
        except Exception:  # noqa: BLE001
            why_html = None
    st.markdown(
        f'<div class="rp-card" style="padding:var(--s5)">'
        f'{C.why_drivers(why_html)}{C.driver_source_note(source)}</div>',
        unsafe_allow_html=True,
    )

    # ── paper bet (only when this horse runs in a loaded race) ──
    if sel and race:
        st.markdown(PAPER_NOTICE, unsafe_allow_html=True)
        place_bet_widget(sel, race, ccy)


main()
