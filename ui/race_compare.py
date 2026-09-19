"""ui/race_compare.py — Head-to-head race selection comparison.

Pick two horses (from any race in the current predictions cache) and get a
full side-by-side breakdown: model probabilities, raw feature vectors grouped
by category, odds/market analysis, and an auto-detected key-differences strip.

Launch:
    streamlit run ui/race_compare.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import to_local
from ui._components import headline_win_prob

st.set_page_config(
    page_title="Compare — MASHR’s Predictor v6",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="auto",
)

_CACHE_PATH = _ROOT / "data" / "predictions.json"
_FEATURES_PATH = _ROOT / "data" / "features.parquet"

# ── feature metadata ───────────────────────────────────────────────────────────

_FEATURE_GROUPS: dict[str, list[str]] = {
    "Market":        ["implied_prob", "overround_norm_prob", "log_odds", "market_rank", "field_size"],
    "Race Context":  ["going_speed", "class_change", "distance_furlongs"],
    "Form":          ["recent_form_avg", "recent_form_wins", "recent_form_runs"],
    "Ratings":       ["timeform_rating", "rating_rank", "pace_bias"],
    "Track Record":  ["historical_win_rate", "historical_place_rate"],
    "Connections":   ["jockey_win_rate", "trainer_win_rate", "jt_combo_win_rate", "jt_combo_runs"],
    "Going Pref":    ["going_pref_win_rate", "going_pref_place_rate"],
    "Value Signals": ["ew_value_index", "odds_drift", "odds_value_delta"],
    "Speed":         ["horse_speed", "horse_speed_rank", "race_complexity"],
}

# For each feature: True → higher value = better for this horse
_HIGHER_IS_BETTER: set[str] = {
    "implied_prob", "overround_norm_prob", "timeform_rating",
    "historical_win_rate", "historical_place_rate",
    "jockey_win_rate", "trainer_win_rate", "jt_combo_win_rate", "jt_combo_runs",
    "going_pref_win_rate", "going_pref_place_rate",
    "ew_value_index", "horse_speed", "recent_form_wins", "jt_combo_runs",
    # recent_form_avg: lower form number in racing = better, but stored as avg figure
    # We treat as neutral — a raw figure depends on convention. Excluded from advantage logic.
}
_LOWER_IS_BETTER: set[str] = {
    "market_rank", "horse_speed_rank", "rating_rank", "log_odds",
}
# Everything else is considered neutral for advantage colouring

# Rate columns (shown as percentages)
_RATE_COLS: set[str] = {
    "implied_prob", "overround_norm_prob", "historical_win_rate",
    "historical_place_rate", "jockey_win_rate", "trainer_win_rate",
    "jt_combo_win_rate", "going_pref_win_rate", "going_pref_place_rate",
}
# Integer columns
_INT_COLS: set[str] = {
    "recent_form_wins", "recent_form_runs", "jt_combo_runs",
    "market_rank", "rating_rank", "horse_speed_rank", "field_size",
}



# ── design system ─────────────────────────────────────────────
# Head-to-head compare. Was injected TWICE (module level and again inside the render) - the second was dead.
# The rp-* classes this page emits are mapped onto the tokens in ui/_design.py
# (see its "legacy page class map" section).
from ui import _components as C  # noqa: E402
from ui._design import (inject_design, palette, plotly_layout,  # noqa: E402
                        tint as _tint)

_P = palette("dark")  # Plotly needs resolved hex, not CSS vars

inject_design()


# ── data loaders ──────────────────────────────────────────────────────────────

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
def _load_features() -> Optional[pd.DataFrame]:
    if not _FEATURES_PATH.exists():
        return None
    try:
        df = pd.read_parquet(_FEATURES_PATH)
        return df if "horse_id" in df.columns else None
    except Exception:
        return None


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo:
            dt = to_local(dt)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return str(iso)


def _race_label(race: dict) -> str:
    venue = escape(race.get("venue") or "Unknown")
    t = _fmt_time(race.get("race_time"))
    n = race.get("field_size", 0)
    return f"{venue} {t} ({n} runners)"


def _all_runners(race: dict) -> list[dict]:
    # Complete validated field when present (audit req 2); display lists only as
    # a legacy-cache fallback — they omit mid-field runners.
    full = race.get("runners")
    if full:
        return list(full)
    runners = list(race.get("selections") or [])
    runners += list(race.get("excluded_low_odds") or [])
    return runners


def _runner_label(r: dict) -> str:
    name = escape(r.get("horse_name") or r.get("horse_id") or "Unknown")
    odds = r.get("decimal_odds")
    # board fraction, like every other surface — a price is a ratio
    return f"{name} ({C.fmt_price(odds)})"


def _risk_tier(composite: Optional[float]) -> str:
    if composite is None:
        return "speculative"
    if composite >= 0.35:
        return "strong"
    if composite >= 0.25:
        return "good"
    if composite >= 0.15:
        return "moderate"
    return "speculative"


def _tier_label(tier: str) -> str:
    return {"strong": "Strong", "good": "Good", "moderate": "Moderate", "speculative": "Speculative"}[tier]


def _fmt_val(col: str, val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return str(val)
    if col in _RATE_COLS:
        return f"{fval * 100:.1f}%"
    if col in _INT_COLS:
        return str(int(round(fval)))
    return f"{fval:.3f}"


def _advantage(col: str, a: Optional[float], b: Optional[float]) -> str:
    """Return 'A', 'B', or '' based on which value is better for this feature."""
    if a is None or b is None:
        return ""
    if abs(a - b) < 1e-9:
        return ""
    if col in _HIGHER_IS_BETTER:
        return "A" if a > b else "B"
    if col in _LOWER_IS_BETTER:
        return "A" if a < b else "B"
    return ""


def _get_feature_val(feat_row: Optional[pd.Series], col: str) -> Optional[float]:
    if feat_row is None:
        return None
    val = feat_row.get(col)
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (TypeError, ValueError):
        return None


def _feature_row(feat_df: Optional[pd.DataFrame], horse_id: str) -> Optional[pd.Series]:
    if feat_df is None or horse_id is None:
        return None
    matches = feat_df[feat_df["horse_id"] == horse_id]
    if matches.empty:
        return None
    return matches.iloc[-1]


# ── summary card HTML ─────────────────────────────────────────────────────────

def _prob_row_html(label: str, val: Optional[float], color: str) -> str:
    v = float(val) if val is not None else 0.0
    w = min(100.0, v * 100)
    val_str = f"{v * 100:.1f}%" if val is not None else "—"
    return (
        f'<div class="prob-row">'
        f'<span class="prob-row-lbl">{escape(label)}</span>'
        f'<div class="prob-row-bar"><div class="prob-row-fill" '
        f'style="width:{w:.0f}%;background:{color}"></div></div>'
        f'<span class="prob-row-val">{val_str}</span>'
        f'</div>'
    )


def _horse_card_html(runner: dict, race: dict, side: str) -> str:
    name = escape(runner.get("horse_name") or runner.get("horse_id") or "Unknown")
    jockey = escape(runner.get("jockey") or "—")
    trainer = escape(runner.get("trainer") or "—")
    venue = escape(race.get("venue") or "—")
    rtime = _fmt_time(race.get("race_time"))
    odds = runner.get("decimal_odds")
    # board fraction with the decimal alongside — a price is a ratio
    odds_html = C.price_html(odds)
    impl = runner.get("implied_prob")
    impl_str = f"{float(impl) * 100:.1f}%" if impl is not None else "—"
    composite = runner.get("composite_score")
    comp_str = f"{float(composite):.3f}" if composite is not None else "—"
    tier = _risk_tier(composite)
    color_a, color_b = _P["series_1"], _P["series_2"]
    fill_color = color_a if side == "a" else color_b

    won_p = runner.get("won_prob")          # raw marginal — drives the edge badge
    head_wp = headline_win_prob(runner)     # field-coherent headline win prob
    p2_p  = runner.get("placed_2_prob")
    sw_p  = runner.get("showed_prob")

    ew_pill = '<span class="pill pill-ew" style="margin-right:4px">E/W</span>' if runner.get("each_way_value") else ""
    tier_pill = f'<span class="pill pill-tier-{tier}">{_tier_label(tier)}</span>'
    val_badge = ""
    if impl is not None and won_p is not None and float(won_p) > float(impl) + 0.05:
        val_badge = ' <span class="pill pill-val" style="margin-left:4px">Value Edge</span>'

    comp_color = {"strong": "var(--value)", "good": "var(--info)",
                  "moderate": "var(--amber)", "speculative": "var(--muted)"}[tier]

    return f"""
<div class="hc side-{side}">
  <div class="hc-label">Horse {side.upper()}</div>
  <div class="hc-name">{name}</div>
  <div class="hc-meta">{venue} &bull; {rtime} &bull; J: {jockey} &bull; T: {trainer}</div>
  <div class="hc-odds">{odds_html}</div>
  <div class="hc-odds-lbl">Best price &bull; Market: {impl_str} &nbsp;
    {ew_pill}{tier_pill}{val_badge}
  </div>
  {_prob_row_html("Win", head_wp, fill_color)}
  {_prob_row_html("Place", p2_p, fill_color)}
  {_prob_row_html("Show", sw_p, fill_color)}
  <div class="composite-row">
    <span class="composite-score" style="color:{comp_color}">{comp_str}</span>
    <span class="composite-lbl">Composite Score</span>
  </div>
</div>
"""


# ── key differences ───────────────────────────────────────────────────────────

def _key_differences(
    row_a: Optional[pd.Series],
    row_b: Optional[pd.Series],
    runner_a: dict,
    runner_b: dict,
    n: int = 5,
) -> list[dict]:
    """Return the top-n features with the largest normalised absolute delta."""
    all_cols = [c for grp in _FEATURE_GROUPS.values() for c in grp]
    diffs = []
    for col in all_cols:
        va = _get_feature_val(row_a, col) if row_a is not None else None
        vb = _get_feature_val(row_b, col) if row_b is not None else None
        # supplement from runner dict for market features
        if va is None and col == "implied_prob":
            va = runner_a.get("implied_prob")
            if va is not None:
                va = float(va)
        if vb is None and col == "implied_prob":
            vb = runner_b.get("implied_prob")
            if vb is not None:
                vb = float(vb)
        if va is None or vb is None:
            continue
        delta = abs(va - vb)
        # normalise by a rough scale
        scale = max(abs(va), abs(vb), 1e-9)
        norm = delta / scale
        adv = _advantage(col, va, vb)
        diffs.append({
            "col": col, "va": va, "vb": vb, "delta": delta,
            "norm": norm, "adv": adv,
        })
    diffs.sort(key=lambda x: x["norm"], reverse=True)
    return diffs[:n]


def _key_diff_strip_html(diffs: list[dict]) -> str:
    if not diffs:
        return ""
    cards = []
    for d in diffs:
        col = d["col"]
        va_str = _fmt_val(col, d["va"])
        vb_str = _fmt_val(col, d["vb"])
        adv = d["adv"]
        if adv == "A":
            adv_html = '<div class="kdiff-adv adv-a">↑ Horse A advantage</div>'
        elif adv == "B":
            adv_html = '<div class="kdiff-adv adv-b">↑ Horse B advantage</div>'
        else:
            adv_html = '<div class="kdiff-adv adv-neutral">Neutral</div>'
        feat_label = col.replace("_", " ").title()
        cards.append(
            f'<div class="kdiff-card">'
            f'<div class="kdiff-feat">{escape(feat_label)}</div>'
            f'<div class="kdiff-vals">{va_str} vs {vb_str}</div>'
            f'{adv_html}'
            f'</div>'
        )
    return '<div class="kdiff-strip">' + "".join(cards) + "</div>"


# ── comparison table ──────────────────────────────────────────────────────────

def _delta_bar_html(va: Optional[float], vb: Optional[float], col: str) -> str:
    if va is None or vb is None:
        return '<span style="color:var(--muted)">—</span>'
    delta = va - vb
    total = abs(va) + abs(vb)
    if total < 1e-9:
        frac = 0.0
    else:
        frac = min(abs(delta) / total, 1.0)
    w = int(frac * 100)
    side_class = "delta-fill-a" if delta > 0 else "delta-fill-b"
    pct_str = f"{frac * 100:.0f}%"
    return (
        f'<div class="delta-wrap">'
        f'<div class="delta-bar"><div class="{side_class}" style="width:{w}%"></div></div>'
        f'<span class="delta-pct">{pct_str}</span>'
        f'</div>'
    )


def _comparison_table_html(
    row_a: Optional[pd.Series],
    row_b: Optional[pd.Series],
    runner_a: dict,
    runner_b: dict,
) -> str:
    def _v(row, col, runner):
        v = _get_feature_val(row, col) if row is not None else None
        if v is None:
            # supplement from runner dict for a handful of market cols
            rv = runner.get(col)
            if rv is not None:
                try:
                    fv = float(rv)
                    return None if pd.isna(fv) else fv
                except (TypeError, ValueError):
                    pass
        return v

    rows_html = [
        '<table class="cmp-tbl">'
        '<thead><tr>'
        '<th>Feature</th>'
        '<th class="num" style="color:var(--series-1)">Horse A</th>'
        '<th style="text-align:center">Delta</th>'
        '<th class="num" style="color:var(--series-2)">Horse B</th>'
        '<th>Edge</th>'
        '</tr></thead>'
        '<tbody>'
    ]

    for grp_name, cols in _FEATURE_GROUPS.items():
        rows_html.append(
            f'<tr class="cmp-grp-hdr"><td colspan="5">{escape(grp_name)}</td></tr>'
        )
        for col in cols:
            va = _v(row_a, col, runner_a)
            vb = _v(row_b, col, runner_b)
            adv = _advantage(col, va, vb)
            cls_a = "adv-win-a" if adv == "A" else "adv-neutral-cell"
            cls_b = "adv-win-b" if adv == "B" else "adv-neutral-cell"
            feat_label = col.replace("_", " ").title()
            delta_html = _delta_bar_html(va, vb, col)
            va_str = _fmt_val(col, va)
            vb_str = _fmt_val(col, vb)
            if adv == "A":
                edge = '<span style="color:var(--series-1);font-size:11px;font-weight:600">A ↑</span>'
            elif adv == "B":
                edge = '<span style="color:var(--series-2);font-size:11px;font-weight:600">B ↑</span>'
            else:
                edge = '<span style="color:var(--muted);font-size:11px">—</span>'
            rows_html.append(
                f'<tr>'
                f'<td class="cmp-feat">{escape(feat_label)}</td>'
                f'<td class="cmp-val {cls_a}">{va_str}</td>'
                f'<td class="delta-cell">{delta_html}</td>'
                f'<td class="cmp-val {cls_b}">{vb_str}</td>'
                f'<td>{edge}</td>'
                f'</tr>'
            )

    rows_html.append("</tbody></table>")
    return "".join(rows_html)


# ── radar chart ───────────────────────────────────────────────────────────────

def _radar_chart(runner_a: dict, runner_b: dict, feat_a: Optional[pd.Series], feat_b: Optional[pd.Series]):
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None

    metrics = [
        # Headline (field-coherent) win prob, not the saturating raw marginal.
        # "Model win %" (not bare "Win %"): this is a PREDICTED probability,
        # never a realized win rate — realized rates always carry wins/bets
        # denominators (see ui/_winrate.py, Stage-4 requirement 9).
        ("Model win %", "won_prob_normalized",   None, True),
        ("Place %",     "placed_2_prob",          None, True),
        ("Show %",      "showed_prob",            None, True),
        ("Composite",   "composite_score",        None, True),
        ("Jockey WR",   "jockey_win_rate",        True, True),
        ("Trainer WR",  "trainer_win_rate",       True, True),
        ("History WR",  "historical_win_rate",    True, True),
        ("Going Pref",  "going_pref_win_rate",    True, True),
        ("Market Pos",  "market_rank",            True, False),  # inverted (lower=better)
    ]

    def _get(runner, feat_row, col, from_feat):
        if from_feat and feat_row is not None:
            v = _get_feature_val(feat_row, col)
            if v is not None:
                return v
        v = runner.get(col)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
        return None

    # collect raw values
    raw_a, raw_b = [], []
    labels = []
    for lbl, col, from_feat, higher in metrics:
        va = _get(runner_a, feat_a, col, from_feat)
        vb = _get(runner_b, feat_b, col, from_feat)
        raw_a.append(va)
        raw_b.append(vb)
        labels.append(lbl)

    # normalise each metric to [0,1] across both horses
    norm_a, norm_b = [], []
    for i, (lbl, col, from_feat, higher) in enumerate(metrics):
        va, vb = raw_a[i], raw_b[i]
        if va is None and vb is None:
            norm_a.append(0.0)
            norm_b.append(0.0)
            continue
        va = va if va is not None else 0.0
        vb = vb if vb is not None else 0.0
        mx = max(abs(va), abs(vb), 1e-9)
        if higher:
            norm_a.append(va / mx)
            norm_b.append(vb / mx)
        else:
            # lower is better: invert
            mn_val = min(va, vb)
            span = max(va, vb) - mn_val
            if span < 1e-9:
                norm_a.append(0.5)
                norm_b.append(0.5)
            else:
                norm_a.append(1.0 - (va - mn_val) / span)
                norm_b.append(1.0 - (vb - mn_val) / span)

    name_a = escape(runner_a.get("horse_name") or "Horse A")
    name_b = escape(runner_b.get("horse_name") or "Horse B")

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_a + [norm_a[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=name_a,
        line=dict(color=_P["series_1"]),
        fillcolor=_tint(_P["series_1"], 0.15),
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_b + [norm_b[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=name_b,
        line=dict(color=_P["series_2"]),
        fillcolor=_tint(_P["series_2"], 0.15),
    ))
    fig.update_layout(**plotly_layout(
        mode="dark", density="compact",
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1.05]),
            angularaxis=dict(gridcolor=_P["chart_grid"], linecolor=_P["border"],
                             tickfont=dict(color=_P["chart_axis"], size=10)),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(x=0.5, y=-0.1, xanchor="center", orientation="h",
                    font=dict(color=_P["chart_ink"], size=10)),
        margin=dict(l=30, r=30, t=30, b=50),
        height=340,
    ))
    return fig


# ── sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(races: list[dict]) -> tuple[Optional[dict], Optional[dict], Optional[dict], Optional[dict]]:
    with st.sidebar:
        st.markdown(C.brand_lockup(), unsafe_allow_html=True)

        st.markdown("### Horse A")
        race_labels = [_race_label(r) for r in races]
        idx_a = st.selectbox("Race A", range(len(races)), format_func=lambda i: race_labels[i], key="race_a")
        race_a = races[idx_a]
        runners_a = _all_runners(race_a)
        runner_labels_a = [_runner_label(r) for r in runners_a]
        idx_ra = st.selectbox("Horse A", range(len(runners_a)), format_func=lambda i: runner_labels_a[i], key="runner_a")
        runner_a = runners_a[idx_ra]

        st.markdown("---")
        st.markdown("### Horse B")
        idx_b = st.selectbox("Race B", range(len(races)), format_func=lambda i: race_labels[i], key="race_b", index=min(1, len(races) - 1))
        race_b = races[idx_b]
        runners_b = _all_runners(race_b)
        runner_labels_b = [_runner_label(r) for r in runners_b]
        # default to second runner if same race as A to avoid identical comparison
        default_rb = 1 if race_b is race_a and len(runners_b) > 1 else 0
        idx_rb = st.selectbox("Horse B", range(len(runners_b)), format_func=lambda i: runner_labels_b[i], key="runner_b", index=default_rb)
        runner_b = runners_b[idx_rb]

        st.markdown("---")
        if st.button("Refresh Predictions"):
            _load_predictions.clear()
            _load_features.clear()
            st.rerun()

    return race_a, runner_a, race_b, runner_b


# ── main ──────────────────────────────────────────────────────────────────────

def _main() -> None:
    pred_data = _load_predictions()
    feat_df = _load_features()

    if pred_data is None:
        st.markdown(
            '<div class="rp-empty"><div class="ei">⚖️</div>'
            '<h3>No predictions cached</h3>'
            '<p>Run <code>python -m models.predictor</code> to generate predictions, '
            'or click Refresh after the models are trained.</p></div>',
            unsafe_allow_html=True,
        )
        return

    races = pred_data.get("races") or []
    races = [r for r in races if _all_runners(r)]

    if not races:
        st.markdown(
            '<div class="rp-empty"><div class="ei">🏇</div>'
            '<h3>No races available</h3>'
            '<p>No races with runners found in the predictions cache.</p></div>',
            unsafe_allow_html=True,
        )
        return

    race_a, runner_a, race_b, runner_b = _sidebar(races)

    # page header
    gen_at = pred_data.get("generated_at", "")
    gen_str = _fmt_time(gen_at) if gen_at else "—"
    st.markdown(
        f'<div class="pg-hdr">'
        f'<h1>Head-to-Head Comparison</h1>'
        f'<div class="sub">Compare two selections across model scores, market signals, '
        f'and raw feature vectors &bull; Cache: {escape(gen_str)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── summary cards ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(_horse_card_html(runner_a, race_a, "a"), unsafe_allow_html=True)
    with col2:
        st.markdown(_horse_card_html(runner_b, race_b, "b"), unsafe_allow_html=True)

    # ── radar chart ────────────────────────────────────────────────────────────
    feat_row_a = _feature_row(feat_df, runner_a.get("horse_id") or "")
    feat_row_b = _feature_row(feat_df, runner_b.get("horse_id") or "")

    fig = _radar_chart(runner_a, runner_b, feat_row_a, feat_row_b)
    if fig is not None:
        st.markdown("#### Performance Radar", unsafe_allow_html=False)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    # ── odds head-to-head ──────────────────────────────────────────────────────
    odds_a = runner_a.get("decimal_odds")
    odds_b = runner_b.get("decimal_odds")
    impl_a = runner_a.get("implied_prob")
    impl_b = runner_b.get("implied_prob")

    if odds_a is not None and odds_b is not None:
        delta_odds = float(odds_a) - float(odds_b)
        delta_impl = (float(impl_a or 0) - float(impl_b or 0)) * 100
        st.markdown("#### Odds Head-to-Head", unsafe_allow_html=False)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Horse A Odds", f"{float(odds_a):.2f}", help="Decimal odds for Horse A")
        m2.metric("Horse B Odds", f"{float(odds_b):.2f}", help="Decimal odds for Horse B")
        m3.metric("Odds Delta", f"{abs(delta_odds):.2f}",
                  delta=f"A {'shorter' if delta_odds < 0 else 'longer'} by {abs(delta_odds):.2f}",
                  delta_color="off")
        m4.metric("Implied Prob Δ",
                  f"{abs(delta_impl):.1f}pp",
                  delta=f"{'A' if delta_impl > 0 else 'B'} favoured by {abs(delta_impl):.1f}pp",
                  delta_color="off")

    # ── key differences ────────────────────────────────────────────────────────
    diffs = _key_differences(feat_row_a, feat_row_b, runner_a, runner_b)
    if diffs:
        st.markdown("#### Key Differences", unsafe_allow_html=False)
        st.markdown(_key_diff_strip_html(diffs), unsafe_allow_html=True)

    # ── feature comparison table ───────────────────────────────────────────────
    st.markdown("#### Feature Breakdown", unsafe_allow_html=False)
    if feat_df is None:
        st.info(
            "Feature store not found (`data/features.parquet`). "
            "Run the feature pipeline to unlock per-runner feature breakdown."
        )
    tbl_html = _comparison_table_html(feat_row_a, feat_row_b, runner_a, runner_b)
    st.markdown(tbl_html, unsafe_allow_html=True)

    # ── footer ────────────────────────────────────────────────────────────────
    targets = pred_data.get("model_targets") or []
    tgt_str = ", ".join(targets) if targets else "—"
    st.markdown(
        f'<div class="rp-footer">'
        f'<span>MASHR&rsquo;s Predictor v6 &bull; Compare</span>'
        f'<span>Model targets: {escape(tgt_str)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


_main()
