"""Race detail — one racecard in full, with per-runner "why" drivers.

Reached from the landing's "Card & drivers →" link, which sets ``?race=<slug>``.
Surfaces the depth the landing compresses: a plain-language "model's view"
summary, then per-runner calibrated win / place / show bars, value edge + EV,
confidence, form trend, and the SHAP drivers (``models.explain``) that explain
the rating. Each runner links on to its full Horse detail page.

SHAP drivers need a feature row per runner. They are resolved by
:mod:`ui._form` from the live inference store when the predictor has written one
for today's races, else the horse's most recent row in the training matrix — so
a runner with prior form gets real drivers instead of the old silent
"unavailable" (which was caused by a synthetic 5-row ``features.parquet`` whose
ids never matched the live runners).
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

st.set_page_config(page_title="Race Detail · MASHR’s Predictor v6",
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
    """Cached SHAP explainer for the price-free win model, or None if unloadable."""
    try:
        from models.explain import get_explainer
        return get_explainer(target="won", version_tag="v3nf")
    except Exception:  # noqa: BLE001
        return None


def _explain(sel: dict) -> tuple[Optional[dict], Optional[str]]:
    """(SHAP drivers, source) for one runner, or (None, None) when no row exists."""
    resolved = _form.feature_row_from(_matrix(), _inference(), str(sel.get("horse_id") or ""))
    if resolved is None:
        return None, None
    row, source = resolved
    ex = _explainer()
    if ex is None:
        return None, None
    try:
        return ex.explain(row, top_k=6), source
    except Exception:  # noqa: BLE001
        return None, None


def _runner_block(sel: dict, race: dict, ccy: str, race_slug: str) -> None:
    horse = escape(sel.get("horse_name") or "—")
    horse_id = sel.get("horse_id")
    jockey = sel.get("jockey") or ""
    trainer = sel.get("trainer") or ""
    meta_bits = " · ".join(
        x for x in (f"J: {escape(jockey)}" if jockey else "",
                    f"T: {escape(trainer)}" if trainer else "") if x
    ) or "Connections unavailable"

    badges = ""
    if sel.get("value_bet"):
        badges += " " + C.value_badge(sel.get("value_edge"))
    if sel.get("each_way_value"):
        badges += " " + C.ew_badge()

    # Horse name links to the full Horse detail page (form + connections + why).
    horse_link = (
        f'<a class="venue rp-hlink" style="font-size:var(--t-body)" '
        f'href="Horse_Detail?horse={escape(str(horse_id or ""))}&race={escape(race_slug)}" '
        f'target="_self">{horse}</a>'
    )

    odds_val = sel.get("best_odds") or sel.get("decimal_odds")
    # Board fraction, no currency — a price is a ratio, not an amount.
    odds_html = (f'<span class="rp-odds" style="font-size:var(--t-md)">'
                 f'{escape(C.fmt_price(odds_val))}</span>'
                 f'<span class="rp-sub">&nbsp;{float(odds_val):.2f}</span>' if odds_val
                 else '<span class="rp-sub">no price</span>')
    books = C.book_chips(sel.get("odds_by_book"), sel.get("best_book"))

    bars = (
        '<div style="display:grid;gap:8px;margin-top:4px">'
        f'<div class="why-row"><div class="why-lab">Win</div>{C.prob_bar(C.headline_win_prob(sel), "win")}</div>'
        f'<div class="why-row"><div class="why-lab">Place</div>{C.prob_bar(sel.get("placed_2_prob"), "place")}</div>'
        f'<div class="why-row"><div class="why-lab">Show</div>{C.prob_bar(sel.get("showed_prob"), "show")}</div>'
        '</div>'
    )

    ev = sel.get("expected_value")
    ev_html = (f'<div class="k-lab">Expected value</div>'
               f'<div style="font-size:var(--t-md)">{C.ev_text(ev, sel.get("value_bet"))}</div>'
               if ev is not None else "")
    conf = C.confidence_chip(C.confidence_for(sel))

    why_data, source = _explain(sel)
    why = C.why_drivers(why_data) + C.driver_source_note(source)

    full_form_link = (
        f'<a class="rp-open" href="Horse_Detail?horse={escape(str(horse_id or ""))}'
        f'&race={escape(race_slug)}" target="_self">Full form &amp; stats →</a>'
    )

    st.markdown(
        '<div class="rp-card" style="margin-bottom:16px">'
        '<div class="rp-card-hdr">'
        f'{C.rank_badge(sel.get("rank"))}'
        f'{horse_link}{badges}'
        f'<span class="spacer"></span>{conf}'
        '</div>'
        '<div class="rp-detail-grid">'
        # left column: price + probabilities
        f'<div><div class="rp-sub" style="margin-bottom:8px">{meta_bits}</div>'
        f'<div style="display:flex;align-items:baseline;gap:10px;margin-bottom:6px">'
        f'{odds_html}{books}</div>'
        f'{bars}{("<div style=margin-top:12px>" + ev_html + "</div>") if ev_html else ""}'
        f'<div style="margin-top:12px">{full_form_link}</div></div>'
        # right column: why drivers
        '<div><div class="k-lab" style="margin-bottom:10px">Why the model rates this runner</div>'
        f'{why}</div>'
        '</div></div>',
        unsafe_allow_html=True,
    )
    place_bet_widget(sel, race, ccy)


def main() -> None:
    preds = _load_predictions()
    races = (preds or {}).get("races") or []
    slug = st.query_params.get("race")

    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)
        st.markdown("### Navigate")
        st.page_link("app.py", label="← All racecards")

    race = C.find_race(races, slug) if slug else None

    if race is None:
        st.markdown(
            C.app_bar('Race detail', now().strftime("%A, %d %B %Y"), ""),
            unsafe_allow_html=True,
        )
        body = ('Pick a race from the '
                '<a href="app.py" target="_self">racecards list</a> — '
                'each card links here with its full runner breakdown.')
        st.markdown(C.empty_state("No race selected", body), unsafe_allow_html=True)
        return

    venue = race.get("venue") or "Unknown"
    ccy = "£" if currency_for_venue(venue) == "GBP" else "€"
    selections = race.get("selections") or []
    # Race-level counts + the model's-view summary read the COMPLETE validated
    # field (audit req 2/7) — a mid-field value bet must count here exactly as it
    # does in the value/suggestion engines. Display stays the curated top-3.
    full_field = race.get("runners") or (
        selections + (race.get("excluded_low_odds") or []))
    field_size = race.get("field_size") or len(selections)
    race_slug = C.race_slug(race)

    st.markdown('<a class="rp-back" href="app.py" target="_self">← All racecards</a>',
                unsafe_allow_html=True)

    n_value = sum(1 for s in full_field if s.get("value_bet"))
    badges = [C.trust_badge("Field", str(field_size), "info")]
    if n_value:
        badges.append(C.trust_badge("Value bets", str(n_value), "ok"))
    title = f'{escape(venue)} <span class="v">{C.fmt_time(race.get("race_time"))}</span>'
    sub = f'{field_size} runners · price-free calibrated model'
    if race.get("each_way_available"):
        sub += " · each-way available"
    st.markdown(C.app_bar(title, sub, "".join(badges)), unsafe_allow_html=True)

    if not selections:
        st.markdown(C.empty_state("No runners", "This race has no scored runners "
                                  "in the prediction cache."), unsafe_allow_html=True)
        return

    # ── model's view + race context (over the complete field, not the top-3) ──
    st.markdown(C.model_view(_form.race_shape(full_field)), unsafe_allow_html=True)
    ctx = [
        ("Field", f"{field_size} runners"),
        ("Each-way", "Available" if race.get("each_way_available") else "No"),
        ("Selections", str(len(selections))),
    ]
    if n_value:
        ctx.append(("Value bets", str(n_value)))
    # Race-level EV gate (audit req 9): a PASS race says so, with its reasons —
    # EV/value fields are withheld rather than served from a stale/partial card.
    if race.get("ev_eligible") is False:
        reasons = (race.get("ev_gate") or {}).get("reasons") or ["ineligible"]
        ctx.append(("EV", "PASS — " + "; ".join(str(r) for r in reasons)))
    st.markdown(C.context_chips(ctx), unsafe_allow_html=True)

    st.markdown(PAPER_NOTICE, unsafe_allow_html=True)
    st.markdown(C.section("Runners", f"{len(selections)} scored · sorted by model rank"),
                unsafe_allow_html=True)
    for sel in sorted(selections, key=lambda s: (s.get("rank") or 999)):
        _runner_block(sel, race, ccy, race_slug)


main()
