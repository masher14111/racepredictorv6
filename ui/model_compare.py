"""ui/model_compare.py — CatBoost vs LightGBM vs the de-vigged market, head to head.

The companion to :mod:`ui.model_honesty`. Where the honesty panel judges a single
model line against the fair market, this page puts the two model families *and*
the market side by side over one leak-free holdout window:

* a **scorecard** — log-loss / Brier / ECE / EV-ROI / CLV for each model and the
  de-vigged market, with the best cell in each column marked so the eye lands on
  the winner (and, per [[v3-lgbm-holdout-verdict]], a model beating log-loss while
  its CLV is negative is shown as paper-only, never a green light);
* a **per-odds-band calibration chart** — predicted win-probability vs actual win
  rate for each model and the market on one reliability diagram, the diagonal
  being perfect; and
* the **per-band detail table** for whichever run you select, reusing the honesty
  panel's table so the two pages read identically.

A control picks which holdout run (folder) to view in detail; the scorecard and
chart compare the latest run of *each* model family, so the page is honest about
what exists — when only one model has a holdout it degrades gracefully to that
model vs the market, with a calm note that the other has no run yet.

The data/formatter layer is Streamlit-free so it stays unit-testable headless
(``tests/ui/test_model_compare.py``); :func:`render` is the only Streamlit-bound
entry point, called by ``ui/pages/15_Model_Compare.py``.
"""
from __future__ import annotations

import math
from html import escape
from pathlib import Path
from typing import Optional

import altair as alt
import pandas as pd
import streamlit as st

from ui import _components as C
from ui import model_honesty as MH

_ROOT = Path(__file__).resolve().parent.parent
_BACKTEST_DIR = _ROOT / "data" / "backtests"

# Model family key → human label. Order is the column / row order on the page.
_MODEL_LABELS: dict[str, str] = {
    "catboost": "CatBoost — v3nf",
    "lgbm": "LightGBM — v3 softmax",
}
_MODEL_ORDER = ["catboost", "lgbm"]
_MARKET_LABEL = "De-vigged market"

# Aggregate rows in the band CSV that are not a single odds band.
_AGG_BANDS = {"ALL", "10/1+ (>=11.0)"}

# Series → line colour on the reliability chart. Read from the shared design
# system, not re-declared: this page renders on the DARK theme, and the old
# hardcoded light values (#2563eb ≈ 3.0:1, labels #6b7689 ≈ 3.3:1 on the dark
# ground) failed WCAG AA. The two models take validated categorical slots; the
# market is a deliberately de-emphasised reference series, so it takes --muted.
from ui._design import altair_theme as _altair_theme

_THEME = _altair_theme("dark")
_SERIES_COLOUR = {
    _MODEL_LABELS["catboost"]: _THEME["series"]["catboost"],
    _MODEL_LABELS["lgbm"]: _THEME["series"]["lgbm"],
    _MARKET_LABEL: _THEME["series"]["market"],
}


# ── small helpers ─────────────────────────────────────────────────────────────

def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def _num(x) -> Optional[float]:
    """Coerce to float, treating None / NaN as missing."""
    if x is None or _isnan(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _clv_pct(mean_clv_log) -> Optional[float]:
    """mean(log(bet/close)) → percentage closing-line value."""
    v = _num(mean_clv_log)
    return None if v is None else (math.exp(v) - 1.0) * 100.0


# ── discovery / model identification ──────────────────────────────────────────

def model_kind(summary: Optional[dict], dir_name: str = "") -> Optional[str]:
    """Which model family produced a holdout run: ``"catboost"`` / ``"lgbm"`` /
    ``None``. Prefers the summary's ``model.model_type`` and falls back to the
    run-dir name so a run still classifies if the field is absent."""
    mt = str(((summary or {}).get("model") or {}).get("model_type") or "").lower()
    name = (dir_name or "").lower()
    for hay in (mt, name):
        if "catboost" in hay:
            return "catboost"
        if "lgbm" in hay or "lightgbm" in hay:
            return "lgbm"
    return None


def discover_holdouts(backtests_dir: Path = _BACKTEST_DIR) -> list[dict]:
    """Every ``holdout_*`` run that carries a head-to-head block, newest first.

    Only runs with a ``head_to_head`` section (the model-vs-market format) qualify
    — older backtest dirs without it can't populate the scorecard and are skipped.
    Each entry: ``dir``, ``name``, ``summary``, ``kind``, ``label``, ``window``."""
    backtests_dir = Path(backtests_dir)
    if not backtests_dir.exists():
        return []
    runs: list[dict] = []
    for p in backtests_dir.glob("holdout_*"):
        if not p.is_dir():
            continue
        summary = MH.load_summary(p)
        if not summary or not summary.get("head_to_head"):
            continue
        kind = model_kind(summary, p.name)
        runs.append({
            "dir": p,
            "name": p.name,
            "summary": summary,
            "kind": kind,
            "label": _MODEL_LABELS.get(kind, kind or p.name),
            "window": summary.get("window") or {},
            "_sort": MH._holdout_sort_key(p),
        })
    runs.sort(key=lambda r: r["_sort"], reverse=True)
    return runs


def latest_by_kind(runs: list[dict]) -> dict[str, dict]:
    """The newest run per model family. Assumes ``runs`` is newest-first."""
    out: dict[str, dict] = {}
    for r in runs:
        k = r.get("kind")
        if k and k not in out:
            out[k] = r
    return out


def comparison_set(runs: list[dict],
                   selected_name: Optional[str] = None) -> tuple[Optional[dict], dict[str, dict]]:
    """Resolve the run to view in detail plus the per-family set to compare.

    The selected run (default: the newest) is what the detail table and the
    market line are read from; the compare set is the latest run of each family,
    with the selected run overriding its own family so an explicit pick is honoured.
    """
    selected = next((r for r in runs if r["name"] == selected_name), None)
    if selected is None and runs:
        selected = runs[0]
    by_kind = latest_by_kind(runs)
    if selected and selected.get("kind"):
        by_kind[selected["kind"]] = selected
    return selected, by_kind


# ── scorecard rows ────────────────────────────────────────────────────────────

def model_metrics(run: dict) -> dict:
    """One scorecard row for a model run (its model line vs the market benchmark)."""
    s = run["summary"]
    h2h = s.get("head_to_head") or {}
    m = h2h.get("model") or {}
    bet = s.get("betting") or {}
    return {
        "label": run["label"],
        "kind": run.get("kind"),
        "is_market": False,
        "log_loss": _num(m.get("log_loss")),
        "brier": _num(m.get("brier_runner_level")),
        "ece": _num(m.get("ece")),
        "ev_roi": _num(bet.get("roi")),
        "clv_pct": _clv_pct(bet.get("mean_clv_log")),
        "beats_market": h2h.get("model_beats_market_logloss"),
        "window": run.get("window") or {},
        "source": run["name"],
    }


def market_metrics(run: dict) -> dict:
    """The de-vigged-market scorecard row read from a run's head-to-head block.
    The market is the benchmark, so it carries no EV-ROI / CLV of its own."""
    s = run["summary"]
    mk = (s.get("head_to_head") or {}).get("market") or {}
    return {
        "label": _MARKET_LABEL,
        "kind": "market",
        "is_market": True,
        "log_loss": _num(mk.get("log_loss")),
        "brier": _num(mk.get("brier_runner_level")),
        "ece": _num(mk.get("ece")),
        "ev_roi": None,
        "clv_pct": None,
        "window": run.get("window") or {},
        "source": run["name"],
    }


def compare_rows(selected: Optional[dict], by_kind: dict[str, dict]) -> list[dict]:
    """Scorecard rows in display order: each available model, then the market
    (read from the selected run so the benchmark matches the detail below)."""
    rows = [model_metrics(by_kind[k]) for k in _MODEL_ORDER if k in by_kind]
    if selected is not None:
        rows.append(market_metrics(selected))
    return rows


# columns: (key, header, kind) — kind drives best-cell selection.
#   "low"  → lower is better, compared across every row (model + market)
#   "high" → higher is better, compared across model rows only
_COLS = [
    ("log_loss", "Log-loss", "low"),
    ("brier", "Brier", "low"),
    ("ece", "ECE", "low"),
    ("ev_roi", "EV-ROI", "high"),
    ("clv_pct", "CLV", "high"),
]


def best_cells(rows: list[dict]) -> dict[str, int]:
    """For each metric, the index of the winning row (or absent if undecidable)."""
    best: dict[str, int] = {}
    for key, _hdr, kind in _COLS:
        candidates = []
        for i, r in enumerate(rows):
            if kind == "high" and r.get("is_market"):
                continue  # market has no EV-ROI / CLV
            v = r.get(key)
            if v is not None:
                candidates.append((i, v))
        if not candidates:
            continue
        if kind == "low":
            idx = min(candidates, key=lambda t: t[1])[0]
        else:
            idx = max(candidates, key=lambda t: t[1])[0]
        best[key] = idx
    return best


# ── reliability-chart data ────────────────────────────────────────────────────

def calibration_chart_df(selected: Optional[dict],
                         by_kind: dict[str, dict]) -> pd.DataFrame:
    """Long-form frame for the reliability diagram: one point per odds band per
    series, with ``series`` / ``band`` / ``predicted`` / ``actual`` columns.

    Each model contributes its own ``model_mean`` (read from that run's band CSV);
    the market contributes the selected run's ``market_mean``. Aggregate rows and
    empty bands are dropped. Returns an empty frame when no band data exists."""
    rows: list[dict] = []

    def _emit(run: dict, label: str, prob_col: str) -> None:
        band = MH.load_band_table(run["dir"])
        if band is None:
            return
        for _i, r in band.iterrows():
            if str(r.get("band")) in _AGG_BANDS:
                continue
            pred, act = _num(r.get(prob_col)), _num(r.get("actual_rate"))
            if pred is None or act is None:
                continue
            rows.append({"series": label, "band": str(r.get("band")),
                         "predicted": pred, "actual": act})

    for k in _MODEL_ORDER:
        run = by_kind.get(k)
        if run:
            _emit(run, run["label"], "model_mean")
    if selected is not None:
        _emit(selected, _MARKET_LABEL, "market_mean")

    return pd.DataFrame(rows, columns=["series", "band", "predicted", "actual"])


# ── HTML / chart builders ─────────────────────────────────────────────────────

_COMPARE_CSS = """
<style>
.mc-card { background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r-md); box-shadow:var(--shadow-card); overflow:hidden;
  margin-bottom:var(--s3); }
.mc-tbl { width:100%; border-collapse:collapse; }
.mc-tbl th, .mc-tbl td { padding:var(--s3) var(--s4); text-align:left;
  font-size:var(--t-sm); border-bottom:1px solid var(--border); }
.mc-tbl thead th { font-size:var(--t-xs); text-transform:uppercase;
  letter-spacing:0.06em; color:var(--muted); font-weight:600; }
.mc-tbl td.num, .mc-tbl th.num { text-align:right; font-family:var(--f-mono);
  font-variant-numeric:tabular-nums; }
.mc-tbl tbody tr:last-child td { border-bottom:none; }
.mc-tbl tr.mc-market td { color:var(--muted); }
.mc-tbl tr.mc-market td.mc-name { color:var(--ink-2); font-weight:600; }
.mc-tbl td.mc-name { color:var(--ink); font-weight:600; }
.mc-tbl td.mc-best { color:var(--value); font-weight:700; }
.mc-swatch { display:inline-block; width:10px; height:10px; border-radius:2px;
  margin-right:7px; vertical-align:middle; }
.mc-tag { font-size:var(--t-xs); font-weight:700; letter-spacing:0.04em;
  padding:1px 7px; border-radius:var(--r-sm); margin-left:8px; }
.mc-tag.go   { background:var(--value-wash);  color:var(--value); }
.mc-tag.nogo { background:var(--danger-wash); color:var(--danger); }
.mc-tag.caution { background:var(--amber-wash); color:var(--amber); }
</style>
"""


def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "ll":
        return f"{v:.4f}"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "pct_signed":
        return f"{v:+.1f}%"
    return str(v)


def compare_table_html(rows: list[dict]) -> str:
    """The scorecard: a model per row plus the market, best cell per column marked."""
    if not rows:
        return ('<p class="rp-sub" style="margin:0">No model runs to compare yet.</p>')
    best = best_cells(rows)
    body = []
    for i, r in enumerate(rows):
        is_market = r.get("is_market")
        tr_cls = ' class="mc-market"' if is_market else ""
        colour = _SERIES_COLOUR.get(r["label"], _THEME["series"]["market"])
        swatch = f'<span class="mc-swatch" style="background:{colour}"></span>'
        # GO / NO-GO + paper-only caution, mirroring the honesty panel's gate.
        tag = ""
        if not is_market and r.get("beats_market") is not None:
            if r.get("beats_market"):
                clv = r.get("clv_pct")
                if clv is not None and clv < 0:
                    tag = '<span class="mc-tag caution">paper-only</span>'
                else:
                    tag = '<span class="mc-tag go">beats market</span>'
            else:
                tag = '<span class="mc-tag nogo">no edge</span>'

        cells = []
        for key, _hdr, _kind in _COLS:
            v = r.get(key)
            disp = _fmt(v, "ll" if key in ("log_loss", "brier", "ece")
                        else "pct" if key == "ev_roi" else "pct_signed")
            cls = "num mc-best" if best.get(key) == i else "num"
            cells.append(f'<td class="{cls}">{disp}</td>')
        body.append(
            f'<tr{tr_cls}>'
            f'<td class="mc-name">{swatch}{escape(r["label"])}{tag}</td>'
            f'{"".join(cells)}</tr>'
        )
    heads = "".join(f'<th class="num">{escape(h)}</th>' for _k, h, _t in _COLS)
    return (
        '<div class="mc-card"><table class="mc-tbl"><thead><tr>'
        f'<th>Model</th>{heads}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def reliability_chart(df: pd.DataFrame) -> Optional["alt.LayerChart"]:
    """Predicted vs actual win rate per odds band, one line per series, with a
    y=x reference diagonal (perfect calibration). ``None`` when there's no data."""
    if df is None or df.empty:
        return None
    hi = float(max(df["predicted"].max(), df["actual"].max()))
    hi = min(1.0, hi * 1.15) if hi > 0 else 1.0
    domain = [s for s in _SERIES_COLOUR if s in set(df["series"])]
    rng = [_SERIES_COLOUR[s] for s in domain]

    diag = (
        alt.Chart(pd.DataFrame({"x": [0.0, hi], "y": [0.0, hi]}))
        .mark_line(color=_THEME["reference"], strokeDash=[4, 4], strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )
    base = alt.Chart(df).encode(
        x=alt.X("predicted:Q", title="Predicted win probability",
                scale=alt.Scale(domain=[0, hi]),
                axis=alt.Axis(format="%", labelColor=_THEME["axis_label"], labelFontSize=10,
                              ticks=False)),
        y=alt.Y("actual:Q", title="Actual win rate",
                scale=alt.Scale(domain=[0, hi]),
                axis=alt.Axis(format="%", labelColor=_THEME["axis_label"], labelFontSize=10,
                              ticks=False)),
        color=alt.Color("series:N", title=None,
                        scale=alt.Scale(domain=domain, range=rng),
                        legend=alt.Legend(orient="top", labelColor=_THEME["legend_label"])),
        tooltip=[
            alt.Tooltip("series:N", title="Source"),
            alt.Tooltip("band:N", title="Odds band"),
            alt.Tooltip("predicted:Q", title="Predicted", format=".1%"),
            alt.Tooltip("actual:Q", title="Actual", format=".1%"),
        ],
    )
    line = base.mark_line(strokeWidth=2, point=True).encode(order="predicted:Q")
    return (
        alt.layer(diag, line)
        .properties(height=360, background="transparent")
        .configure_view(strokeWidth=0, fill=None)
        .configure_axis(grid=False, domainWidth=0,
                        domainColor=_THEME["grid"],
                        titleColor=_THEME["legend_label"], titleFontWeight="normal")
        .configure_legend(labelColor=_THEME["legend_label"],
                          titleColor=_THEME["legend_label"], symbolStrokeWidth=0)
    )


# ── Streamlit render (the only Streamlit-bound entry point) ───────────────────

def render(*, backtests_dir: Path = _BACKTEST_DIR) -> None:
    """Render the head-to-head comparison. Reads the discovered holdout runs and
    degrades to a calm empty state when none carry head-to-head data yet."""
    st.markdown(_COMPARE_CSS + MH._PANEL_CSS, unsafe_allow_html=True)
    st.markdown(
        C.app_bar(
            'Model comparison <span class="v">— head to head</span>',
            "CatBoost vs LightGBM vs the de-vigged market on one holdout window",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rp-paper-note"><span class="dot"></span>'
        '<b>Paper / Practice — no real money.</b> Every figure is a leak-free '
        'holdout score against the fair market line. Beating log-loss while CLV is '
        'negative is paper-only, never a green light to bet.</div>',
        unsafe_allow_html=True,
    )

    runs = discover_holdouts(backtests_dir)
    if not runs:
        st.markdown(
            C.empty_state(
                "No comparable holdout runs yet",
                "Run a leak-free holdout for each model to compare them against the "
                "de-vigged market on one window:<br><br>"
                "<code>python -m backtest.holdout</code>",
            ),
            unsafe_allow_html=True,
        )
        return

    # ── control: which holdout run to view in detail ─────────────────────────
    names = [r["name"] for r in runs]
    labels = {r["name"]: f'{r["label"]} · {r["name"]}' for r in runs}
    selected_name = st.selectbox(
        "Holdout run", names, index=0,
        format_func=lambda n: labels.get(n, n),
        help="Picks the run shown in the per-band detail and the market line. The "
             "scorecard and chart compare the latest run of each model family.",
    )
    selected, by_kind = comparison_set(runs, selected_name)
    rows = compare_rows(selected, by_kind)

    # ── scorecard ────────────────────────────────────────────────────────────
    win = (selected or {}).get("window") or {}
    win_note = (f'{win.get("start")} → {win.get("end")}'
                if win.get("start") and win.get("end") else "latest window")
    st.markdown(C.section("Scorecard", f"lower is better for log-loss / Brier / ECE · {win_note}"),
                unsafe_allow_html=True)
    st.markdown(compare_table_html(rows), unsafe_allow_html=True)

    missing = [_MODEL_LABELS[k] for k in _MODEL_ORDER if k not in by_kind]
    if missing:
        st.caption(f"No holdout run on disk for {', '.join(missing)} — showing the "
                   "available model(s) against the market. Run "
                   "`python -m backtest.holdout` for the missing model to fill the row.")
    st.caption("Best cell per column is highlighted. EV-ROI / CLV apply to the "
               "models only — the market is the benchmark they must clear. "
               "“paper-only” marks a model that edges log-loss but has negative CLV.")

    # ── reliability chart ────────────────────────────────────────────────────
    st.markdown(C.section("Calibration by odds band",
                          "predicted vs actual win rate — the diagonal is perfect"),
                unsafe_allow_html=True)
    cdf = calibration_chart_df(selected, by_kind)
    chart = reliability_chart(cdf)
    if chart is None:
        st.markdown(
            '<p class="rp-sub" style="margin:0">No per-band calibration data found '
            'for these runs — re-run <code>python -m backtest.holdout</code>.</p>',
            unsafe_allow_html=True,
        )
    else:
        # theme=None: Streamlit's own chart theme follows `[theme] base`, which
        # is still "light", so it would paint a white plate under this chart on
        # a dark page. The chart carries its own themed chrome instead.
        st.altair_chart(chart, width="stretch", theme=None)
        st.caption("Each point is one odds band. A series sitting on the diagonal is "
                   "perfectly calibrated; below it is over-confident, above it "
                   "under-confident. The market hugging the line tightest is the "
                   "honest tell that a model's edge is thin.")

    # ── per-band detail for the selected run ─────────────────────────────────
    if selected is not None:
        st.markdown(
            C.section(f"Per-band detail — {selected['label']}", win_note),
            unsafe_allow_html=True,
        )
        st.markdown(MH.band_table_html(MH.load_band_table(selected["dir"])),
                    unsafe_allow_html=True)

    # ── footer ───────────────────────────────────────────────────────────────
    kinds_shown = ", ".join(_MODEL_LABELS.get(k, k) for k in _MODEL_ORDER if k in by_kind)
    st.markdown(
        C.footer(
            f'{C.status_dot("ok")}Comparing <span class="mono">{escape(kinds_shown or "—")}</span>',
            f'{len(runs)} holdout run(s) · viewing '
            f'<span class="mono">{escape((selected or {}).get("name", "—"))}</span>',
        ),
        unsafe_allow_html=True,
    )
