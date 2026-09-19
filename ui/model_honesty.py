"""ui/model_honesty.py — the panel that tells the truth.

The single most valuable read in the whole app: for each model line, *does it
actually beat the de-vigged market?* This is what racing_ingestion's dashboard
banner does and v4 has lacked. It surfaces, side by side and in plain English:

* a prominent **GO / NO-GO banner** per model line — green when the model's
  win-probabilities score a lower (better) log-loss than the fair, margin-free
  market line, amber/red when they do not (paper-only either way);
* **CLV** — the mean closing-line value and the beat-close rate, the deciding
  tie-breaker that exposed v4's illusory backtest edge ([[v4-vs-racing-ingestion-3wk]]);
* the **integrity suite** as an OK / WARN / FAIL status list; and
* the **per-odds-band model-vs-market calibration table**, so a model that is
  well set on favourites but mis-priced on longshots can't hide behind a single
  aggregate number.

Honesty rules baked in:

* Beating the market on **log-loss** is the headline gate, but a positive
  ``go`` is *not* a green light to bet — a model can edge the log-loss while its
  closing-line value is negative (the v3 holdout does exactly this:
  [[v3-lgbm-holdout-verdict]]). When that happens the banner says so explicitly.
* Read-only: nothing here places a bet or implies real money. Every surface is
  labelled paper / practice.
* A missing holdout degrades to a calm "run ``python -m backtest.holdout``"
  message, never a stack trace.

The data/formatter layer below is deliberately Streamlit-free so it stays
unit-testable headless (``tests/ui/test_model_honesty.py``); :func:`render` is
the only Streamlit-bound entry point, called by ``ui/pages/14_Model_Honesty.py``.
"""
from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ui import _components as C

_ROOT = Path(__file__).resolve().parent.parent
_BACKTEST_DIR = _ROOT / "data" / "backtests"
_PREDICTIONS = _ROOT / "data" / "predictions.json"
_BAND_CSV = "odds_band_calibration.csv"
_SUMMARY_JSON = "summary.json"

# Verdict keys (in predictions.json) → human label for the banner.
_MODEL_LABELS: dict[str, str] = {
    "lgbm": "LightGBM — v3 softmax",
    "catboost": "CatBoost",
}

# Integrity status → (badge css class, status dot tone). Anything unrecognised
# is treated as a WARN so a new check never renders as a confident green.
_STATUS_CLASS = {"OK": "ok", "WARN": "warn", "FAIL": "fail", "ERROR": "fail"}


# ── discovery / loading ───────────────────────────────────────────────────────

def _holdout_sort_key(p: Path) -> tuple[str, float]:
    """Sort holdout dirs newest-last. Prefer the embedded ``YYYYMMDD_HHMMSS``
    stamp (``holdout_lgbm_20260618_212927``); fall back to mtime when the name
    carries no stamp so out-of-convention dirs still order sensibly."""
    parts = p.name.split("_")
    stamp = ""
    for i in range(len(parts) - 1):
        a, b = parts[i], parts[i + 1]
        if len(a) == 8 and a.isdigit() and len(b) == 6 and b.isdigit():
            stamp = f"{a}_{b}"
            break
    try:
        mtime = (p / _SUMMARY_JSON).stat().st_mtime
    except OSError:
        mtime = 0.0
    return (stamp, mtime)


def find_latest_holdout(backtests_dir: Path = _BACKTEST_DIR) -> Optional[Path]:
    """The most recent ``holdout_*`` run dir that carries a ``summary.json``.

    Returns ``None`` when the backtests directory is absent or holds no holdout
    run yet (the cold-start case the UI must handle gracefully)."""
    backtests_dir = Path(backtests_dir)
    if not backtests_dir.exists():
        return None
    runs = [p for p in backtests_dir.glob("holdout_*")
            if p.is_dir() and (p / _SUMMARY_JSON).exists()]
    if not runs:
        return None
    return sorted(runs, key=_holdout_sort_key)[-1]


def load_summary(holdout_dir: Optional[Path]) -> Optional[dict]:
    """Parse ``summary.json`` from a holdout dir; ``None`` on absent/corrupt."""
    if holdout_dir is None:
        return None
    path = Path(holdout_dir) / _SUMMARY_JSON
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_band_table(holdout_dir: Optional[Path]) -> Optional[pd.DataFrame]:
    """The per-odds-band calibration CSV as a DataFrame; ``None`` if absent or
    unreadable. An empty (header-only) file is treated as no table."""
    if holdout_dir is None:
        return None
    path = Path(holdout_dir) / _BAND_CSV
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return df if not df.empty else None


def load_verdict(predictions_path: Path = _PREDICTIONS) -> dict:
    """The ``verdict`` block from predictions.json; ``{}`` when missing/corrupt."""
    try:
        data = json.loads(Path(predictions_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    verdict = data.get("verdict")
    return verdict if isinstance(verdict, dict) else {}


# ── verdict → banner lines ────────────────────────────────────────────────────

def verdict_lines(verdict: dict) -> list[dict]:
    """Normalise the per-model verdict block into banner-ready rows.

    Each row carries a ``status`` of ``"go"`` / ``"nogo"`` / ``"unavailable"``
    plus the numbers the banner and CLV block render. The GO/NO-GO split is on
    ``model_beats_market_logloss`` (the honest gate) — NOT the looser ``go``
    flag, which can be true while closing-line value is negative; that nuance is
    surfaced as a caution rather than a green light ([[v3-lgbm-holdout-verdict]]).
    """
    lines: list[dict] = []
    for key, label in _MODEL_LABELS.items():
        v = verdict.get(key)
        if not isinstance(v, dict):
            continue
        has_verdict = (v.get("model_beats_market_logloss") is not None
                       or v.get("go") is not None)
        if v.get("available") is False or not has_verdict:
            lines.append({
                "key": key, "label": label, "status": "unavailable",
                "note": v.get("note") or "No holdout verdict on disk for this model.",
            })
            continue
        beats = v.get("model_beats_market_logloss")
        if beats is None:
            beats = bool(v.get("go"))
        lines.append({
            "key": key,
            "label": label,
            "status": "go" if beats else "nogo",
            "go": bool(v.get("go")),
            "beats_logloss": bool(beats),
            "model_log_loss": v.get("model_log_loss"),
            "market_log_loss": v.get("market_log_loss"),
            "logloss_gap": v.get("logloss_gap_market_minus_model"),
            "mean_clv_log": v.get("mean_clv_log"),
            "clv_beat_rate": v.get("clv_beat_rate"),
            "ev_roi": v.get("ev_roi"),
            "ev_n_bets": v.get("ev_n_bets"),
            "n_races_kept": v.get("n_races_kept"),
            "model_version": v.get("model_version"),
            "window": v.get("holdout_window") or {},
        })
    return lines


# ── plain-English helpers ─────────────────────────────────────────────────────

def _fmt_ll(x) -> str:
    return "—" if x is None or _isnan(x) else f"{float(x):.4f}"


def _fmt_pct(x, dp: int = 1) -> str:
    return "—" if x is None or _isnan(x) else f"{float(x) * 100:.{dp}f}%"


def _isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


def clv_summary(mean_clv_log, beat_rate) -> Optional[dict]:
    """Render the CLV numbers into a plain-English read.

    ``mean_clv_log`` is ``mean(log(bet_price / closing_price))``. Negative means
    bets were struck at prices *shorter* than the closing line — you systematically
    got worse value than the market's final word, the classic sign the "edge" is
    noise. Returns ``None`` when CLV was never computed (no settled bets)."""
    if mean_clv_log is None or _isnan(mean_clv_log):
        return None
    pct = (math.exp(float(mean_clv_log)) - 1.0) * 100.0
    positive = pct >= 0.0
    direction = "longer than" if positive else "shorter than"
    if positive:
        plain = (f"On average your bets were struck about {abs(pct):.1f}% "
                 f"{direction} the closing line — you beat the market's final "
                 f"price, the one signal that survives an honest test.")
    else:
        plain = (f"On average your bets were struck about {abs(pct):.1f}% "
                 f"{direction} the closing line — you got *worse* value than the "
                 f"market's final price, so any backtest edge is almost certainly "
                 f"noise, not skill.")
    return {
        "mean_pct": pct,
        "positive": positive,
        "beat_rate": None if beat_rate is None or _isnan(beat_rate) else float(beat_rate),
        "plain": plain,
    }


# ── HTML builders (pure, themed via _design.py tokens) ────────────────────────

_PANEL_CSS = """
<style>
/* GO / NO-GO verdict banner — body text on a near-black surface (AA-clear),
   colour carried by a 4px rail + a bright heading + a word, never colour alone. */
.mh-banner { display:flex; gap:var(--s3); align-items:flex-start;
  background:var(--surface); border:1px solid var(--border); border-left-width:4px;
  border-radius:var(--r-md); padding:var(--s4) var(--s5); margin-bottom:var(--s3);
  box-shadow:var(--shadow-card); }
.mh-banner.go   { border-left-color:var(--value); }
.mh-banner.nogo { border-left-color:var(--danger); }
.mh-banner.unavailable { border-left-color:var(--border-strong); }
.mh-banner .mh-mark { font-size:1.05rem; line-height:1.4; flex:none; }
.mh-banner.go   .mh-mark { color:var(--value); }
.mh-banner.nogo .mh-mark { color:var(--danger); }
.mh-banner.unavailable .mh-mark { color:var(--muted); }
.mh-banner .mh-body { flex:1; min-width:0; }
.mh-banner .mh-model { font-size:var(--t-xs); color:var(--muted);
  text-transform:uppercase; letter-spacing:0.08em; font-weight:600; }
.mh-banner .mh-title { font-size:var(--t-md); font-weight:600; margin-top:2px;
  color:var(--ink); }
.mh-banner.go   .mh-title { color:var(--value); }
.mh-banner.nogo .mh-title { color:var(--danger); }
.mh-banner .mh-nums { font-family:var(--f-mono); font-variant-numeric:tabular-nums;
  color:var(--ink-2); }
.mh-banner .mh-sub { font-size:var(--t-sm); color:var(--ink-2); margin-top:6px;
  line-height:1.55; }
.mh-banner .mh-sub em { color:var(--amber); font-style:normal; font-weight:600; }
.mh-banner .mh-meta { font-size:var(--t-xs); color:var(--muted); margin-top:8px;
  font-family:var(--f-mono); font-variant-numeric:tabular-nums; }

/* CLV read */
.mh-clv { background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r-md); padding:var(--s4) var(--s5); margin-bottom:var(--s3);
  box-shadow:var(--shadow-card); }
.mh-clv .mh-clv-figs { display:flex; gap:var(--s8); flex-wrap:wrap;
  margin-bottom:var(--s3); }
.mh-clv .mh-fig .mh-fig-lab { font-size:var(--t-xs); color:var(--muted);
  text-transform:uppercase; letter-spacing:0.08em; }
.mh-clv .mh-fig .mh-fig-val { font-size:var(--t-lg); font-weight:600;
  font-family:var(--f-mono); font-variant-numeric:tabular-nums; color:var(--ink);
  line-height:1.2; margin-top:2px; }
.mh-clv .mh-fig .mh-fig-val.pos { color:var(--value); }
.mh-clv .mh-fig .mh-fig-val.neg { color:var(--danger); }
.mh-clv .mh-clv-plain { font-size:var(--t-sm); color:var(--ink-2); line-height:1.55; }
.mh-clv .mh-clv-plain em { color:var(--danger); font-style:normal; font-weight:600; }

/* integrity checks */
.mh-checks { display:grid; gap:var(--s2); margin-bottom:var(--s3); }
.mh-check { display:flex; gap:var(--s3); align-items:flex-start;
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--r-sm); padding:var(--s3) var(--s4); }
.mh-check .mh-badge { font-size:var(--t-xs); font-weight:700; letter-spacing:0.05em;
  padding:2px 9px; border-radius:var(--r-sm); flex:none; line-height:1.7;
  min-width:46px; text-align:center; }
.mh-badge.ok   { background:var(--value-wash);  color:var(--value);
  border:1px solid oklch(0.76 0.15 152 / .4); }
.mh-badge.warn { background:var(--amber-wash);  color:var(--amber);
  border:1px solid oklch(0.79 0.13 75 / .35); }
.mh-badge.fail { background:var(--danger-wash); color:var(--danger);
  border:1px solid oklch(0.65 0.19 28 / .4); }
.mh-check .mh-cbody { min-width:0; }
.mh-check .mh-cname { font-size:var(--t-sm); color:var(--ink); font-weight:600; }
.mh-check .mh-cmsg { font-size:var(--t-sm); color:var(--ink-2); line-height:1.5;
  margin-top:1px; }

/* band table — reuses rp-tbl, with model/market columns paired */
.mh-band td.mh-better { color:var(--value); font-weight:600; }
.mh-band tr.mh-agg td { color:var(--muted); border-top:1px solid var(--border-strong); }
.mh-band tr.mh-agg td.mh-band-name { color:var(--ink-2); font-weight:600; }
.mh-band .mh-win { color:var(--value); }
</style>
"""


def banner_html(line: dict) -> str:
    """One GO / NO-GO verdict banner for a model line."""
    label = escape(line.get("label") or line.get("key") or "Model")
    status = line.get("status")
    if status == "unavailable":
        note = escape(line.get("note") or "No holdout verdict on disk.")
        return (
            '<div class="mh-banner unavailable">'
            '<span class="mh-mark">○</span><div class="mh-body">'
            f'<div class="mh-model">{label}</div>'
            '<div class="mh-title">No verdict yet</div>'
            f'<div class="mh-sub">{note}</div>'
            '</div></div>'
        )

    model_ll = _fmt_ll(line.get("model_log_loss"))
    market_ll = _fmt_ll(line.get("market_log_loss"))
    beats = line.get("status") == "go"
    mark = "✓" if beats else "✗"
    if beats:
        title = "Beats the market on log-loss"
        sub = (f'Model log-loss <span class="mh-nums">{model_ll}</span> vs '
               f'market <span class="mh-nums">{market_ll}</span> over the '
               f'de-vigged closing-odds-free benchmark.')
    else:
        title = "Does NOT beat the market"
        sub = (f'Model log-loss <span class="mh-nums">{model_ll}</span> vs '
               f'market <span class="mh-nums">{market_ll}</span> — the fair '
               f'market line is at least as sharp, so there is no edge to bet.')

    # Honest caution: a positive log-loss gate paired with negative closing-line
    # value is the trap the memory warns about — flag it inline, never hide it.
    clv = line.get("mean_clv_log")
    if beats and clv is not None and not _isnan(clv) and float(clv) < 0:
        sub += ('<br><em>Caution:</em> it edges the log-loss but its closing-line '
                'value is negative (below) — treat the verdict as paper-only, not '
                'a green light to bet.')

    meta_bits = []
    if line.get("model_version"):
        meta_bits.append(escape(str(line["model_version"])))
    win = line.get("window") or {}
    if win.get("start") and win.get("end"):
        meta_bits.append(f'{escape(str(win["start"]))} → {escape(str(win["end"]))}')
    if line.get("n_races_kept") is not None:
        meta_bits.append(f'{int(line["n_races_kept"]):,} races')
    meta = (f'<div class="mh-meta">{" · ".join(meta_bits)}</div>'
            if meta_bits else "")

    return (
        f'<div class="mh-banner {"go" if beats else "nogo"}">'
        f'<span class="mh-mark">{mark}</span><div class="mh-body">'
        f'<div class="mh-model">{label}</div>'
        f'<div class="mh-title">{escape(title)}</div>'
        f'<div class="mh-sub">{sub}</div>{meta}'
        '</div></div>'
    )


def clv_block_html(line: dict) -> str:
    """The CLV read for a model line: mean CLV, beat-close rate, plain English."""
    summ = clv_summary(line.get("mean_clv_log"), line.get("clv_beat_rate"))
    if summ is None:
        return (
            '<div class="mh-clv"><div class="mh-clv-plain">'
            'No closing-line value recorded — the EV simulation placed no settled '
            'bets in this window.</div></div>'
        )
    mean_cls = "pos" if summ["positive"] else "neg"
    mean_val = f'{summ["mean_pct"]:+.1f}%'
    beat = summ["beat_rate"]
    beat_val = "—" if beat is None else f"{beat * 100:.0f}%"
    beat_cls = "pos" if (beat is not None and beat >= 0.5) else "neg"
    plain = summ["plain"]
    # Mark the damning clause so the eye lands on it (word + colour, not colour alone).
    plain_html = escape(plain).replace("*worse*", "<em>worse</em>")
    return (
        '<div class="mh-clv"><div class="mh-clv-figs">'
        '<div class="mh-fig"><div class="mh-fig-lab">Mean CLV</div>'
        f'<div class="mh-fig-val {mean_cls}">{mean_val}</div></div>'
        '<div class="mh-fig"><div class="mh-fig-lab">Beat-close rate</div>'
        f'<div class="mh-fig-val {beat_cls}">{beat_val}</div></div>'
        '</div>'
        f'<div class="mh-clv-plain">{plain_html}</div></div>'
    )


def integrity_rows_html(integrity: Optional[list]) -> str:
    """The integrity suite as an OK / WARN / FAIL status list."""
    if not integrity:
        return ('<p class="rp-sub" style="margin:0">No integrity checks recorded '
                'in this holdout summary.</p>')
    rows = []
    for chk in integrity:
        status = str(chk.get("status") or "WARN").upper()
        cls = _STATUS_CLASS.get(status, "warn")
        name = escape(str(chk.get("name") or "check").replace("_", " ").title())
        msg = escape(str(chk.get("message") or ""))
        rows.append(
            '<div class="mh-check">'
            f'<span class="mh-badge {cls}">{escape(status)}</span>'
            f'<div class="mh-cbody"><div class="mh-cname">{name}</div>'
            f'<div class="mh-cmsg">{msg}</div></div></div>'
        )
    return f'<div class="mh-checks">{"".join(rows)}</div>'


_AGG_BANDS = {"ALL", "10/1+ (>=11.0)"}


def band_table_html(df: Optional[pd.DataFrame]) -> str:
    """Per-odds-band actual-vs-model-vs-market calibration as a themed table.

    For each band: runners, actual win rate, the model's and the market's mean
    probability, their A/E ratios (actual ÷ expected; 1.00 is perfect), and which
    side fits the band's true win rate more closely. Aggregate rows (ALL, 10/1+)
    are de-emphasised below the per-band rows."""
    if df is None or df.empty:
        return ('<p class="rp-sub" style="margin:0">No per-band calibration table '
                'found for this holdout — re-run <code>python -m backtest.holdout</code>.</p>')

    def cell(v, kind: str) -> str:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "—"
        if kind == "pct":
            return f"{float(v) * 100:.1f}%"
        if kind == "ae":
            return f"{float(v):.2f}"
        if kind == "int":
            return f"{int(v):,}"
        return escape(str(v))

    rows = []
    for _, r in df.iterrows():
        band = str(r.get("band") or "—")
        is_agg = band in _AGG_BANDS
        m_gap = r.get("model_gap")
        k_gap = r.get("market_gap")
        better = ""  # which side is closer to the band's actual win rate
        try:
            if not (math.isnan(float(m_gap)) or math.isnan(float(k_gap))):
                if abs(float(m_gap)) < abs(float(k_gap)):
                    better = "model"
                elif abs(float(k_gap)) < abs(float(m_gap)):
                    better = "market"
        except (TypeError, ValueError):
            better = ""
        model_cls = ' class="num mh-better"' if better == "model" else ' class="num"'
        market_cls = ' class="num mh-better"' if better == "market" else ' class="num"'
        better_lbl = {"model": "Model", "market": "Market"}.get(better, "—")
        tr_cls = ' class="mh-agg"' if is_agg else ""
        rows.append(
            f'<tr{tr_cls}>'
            f'<td class="mh-band-name">{escape(band)}</td>'
            f'<td class="num">{cell(r.get("n_runners"), "int")}</td>'
            f'<td class="num mh-win">{cell(r.get("actual_rate"), "pct")}</td>'
            f'<td class="num">{cell(r.get("model_mean"), "pct")}</td>'
            f'<td class="num">{cell(r.get("market_mean"), "pct")}</td>'
            f'<td{model_cls}>{cell(r.get("ae_model"), "ae")}</td>'
            f'<td{market_cls}>{cell(r.get("ae_market"), "ae")}</td>'
            f'<td>{better_lbl}</td>'
            '</tr>'
        )

    return (
        '<div class="rp-card mh-band"><div class="rp-body">'
        '<table class="rp-tbl"><thead><tr>'
        '<th>Odds band</th><th class="num">Runners</th><th class="num">Actual win</th>'
        '<th class="num">Model</th><th class="num">Market</th>'
        '<th class="num">Model A/E</th><th class="num">Market A/E</th>'
        '<th>Closer fit</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'
    )


# ── compact verdict strip (for embedding atop other pages) ────────────────────

def latest_verdict_line(verdict: dict) -> Optional[dict]:
    """The single most relevant graded line for a one-line strip: the first
    GO / NO-GO line in model order, or ``None`` when no model has a verdict yet."""
    graded = [ln for ln in verdict_lines(verdict)
              if ln.get("status") in ("go", "nogo")]
    return graded[0] if graded else None


_STRIP_CSS = """
<style>
/* Compact GO / NO-GO strip — tokens fall back so it reads on both the dark
   design-system pages and the light performance pages. */
.mh-strip { display:flex; gap:12px; align-items:center; flex-wrap:wrap;
  background:var(--surface,#fff); border:1px solid var(--border,#d8dee8);
  border-left:4px solid var(--border,#d8dee8); border-radius:10px;
  padding:11px 16px; margin:0 0 18px; }
.mh-strip.go   { border-left-color:var(--value); }
.mh-strip.nogo { border-left-color:var(--danger,#dc2626); }
.mh-strip.unavailable { border-left-color:var(--muted,#6b7689); }
.mh-strip .mh-s-tag { font-size:11px; font-weight:700; letter-spacing:0.05em;
  text-transform:uppercase; padding:2px 9px; border-radius:6px; }
.mh-strip.go   .mh-s-tag { background:rgba(22,163,74,.14);
  color:var(--value); }
.mh-strip.nogo .mh-s-tag { background:rgba(220,38,38,.14); color:var(--danger,#dc2626); }
.mh-strip.unavailable .mh-s-tag { background:rgba(107,118,137,.14); color:var(--muted,#6b7689); }
.mh-strip .mh-s-model { font-weight:600; font-size:13px;
  color:var(--ink); }
.mh-strip .mh-s-nums { font-size:12px; color:var(--muted,#6b7689);
  font-variant-numeric:tabular-nums; }
.mh-strip .mh-s-nums b { color:var(--ink-2); font-weight:600; }
.mh-strip .mh-s-caution { font-size:12px; font-weight:600;
  color:var(--amber); margin-left:auto; }
</style>
"""


def verdict_strip_html(verdict: dict) -> str:
    """A one-line GO / NO-GO summary of the latest model verdict, for embedding at
    the top of the performance pages so the bankroll view is read in context. The
    honest trap — a log-loss win paired with negative CLV — is flagged inline."""
    line = latest_verdict_line(verdict)
    if line is None:
        return (
            '<div class="mh-strip unavailable">'
            '<span class="mh-s-tag">No verdict</span>'
            '<span class="mh-s-nums">No GO / NO-GO holdout verdict on disk — run '
            '<code>python -m backtest.holdout</code>.</span></div>'
        )
    go = line.get("status") == "go"
    cls = "go" if go else "nogo"
    tag = "GO" if go else "NO-GO"
    model = escape(line.get("label") or "Model")
    nums = (f'log-loss <b>{_fmt_ll(line.get("model_log_loss"))}</b> '
            f'vs market {_fmt_ll(line.get("market_log_loss"))}')
    clv = line.get("mean_clv_log")
    if clv is not None and not _isnan(clv):
        nums += f' · CLV {(math.exp(float(clv)) - 1.0) * 100.0:+.1f}%'
    caution = ""
    if go and clv is not None and not _isnan(clv) and float(clv) < 0:
        caution = ('<span class="mh-s-caution">⚠ paper-only — negative CLV, '
                   'not a green light</span>')
    return (
        f'<div class="mh-strip {cls}">'
        f'<span class="mh-s-tag">{tag}</span>'
        f'<span class="mh-s-model">{model}</span>'
        f'<span class="mh-s-nums">{nums}</span>'
        f'{caution}</div>'
    )


def render_verdict_strip(*, predictions_path: Path = _PREDICTIONS) -> None:
    """Streamlit helper: inject the strip CSS + the latest verdict strip. Called
    from the performance pages to anchor the bankroll view in the honest verdict."""
    st.markdown(_STRIP_CSS, unsafe_allow_html=True)
    st.markdown(verdict_strip_html(load_verdict(predictions_path)),
                unsafe_allow_html=True)


# ── Streamlit render (the only Streamlit-bound entry point) ───────────────────

def render(*, backtests_dir: Path = _BACKTEST_DIR,
           predictions_path: Path = _PREDICTIONS) -> None:
    """Render the honesty panel. Reads the latest holdout summary + the
    predictions verdict; shows a calm empty state when no holdout exists yet.

    Designed to be dropped into a page (``ui/pages/14_Model_Honesty.py``) or an
    ``app.py`` section — it renders content only, leaving page config / design
    injection to the caller."""
    st.markdown(_PANEL_CSS, unsafe_allow_html=True)

    st.markdown(
        C.app_bar(
            'Model honesty <span class="v">— does it beat the market?</span>',
            "The de-vigged market is the benchmark every edge must clear",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="rp-paper-note"><span class="dot"></span>'
        '<b>Paper / Practice — no real money.</b> These verdicts judge the model '
        'against the fair market line on a leak-free holdout. They never place a '
        'bet.</div>',
        unsafe_allow_html=True,
    )

    holdout = find_latest_holdout(backtests_dir)
    summary = load_summary(holdout)
    verdict = load_verdict(predictions_path)
    lines = verdict_lines(verdict)

    if not lines and summary is None:
        st.markdown(
            C.empty_state(
                "No holdout verdict yet",
                "Run a leak-free holdout to generate a GO / NO-GO verdict, the "
                "CLV read, the integrity suite and the per-band calibration table:"
                "<br><br><code>python -m backtest.holdout</code>",
            ),
            unsafe_allow_html=True,
        )
        return

    # ── 1. GO / NO-GO banner + CLV, per model line ───────────────────────────
    st.markdown(C.section("Verdict", "model vs the de-vigged market"),
                unsafe_allow_html=True)
    if lines:
        for line in lines:
            st.markdown(banner_html(line), unsafe_allow_html=True)
            if line.get("status") != "unavailable":
                st.markdown(clv_block_html(line), unsafe_allow_html=True)
    else:
        st.markdown(
            '<p class="rp-sub">No per-model verdict in predictions.json yet — '
            'showing the latest holdout below.</p>',
            unsafe_allow_html=True,
        )

    if summary is None:
        st.markdown(
            C.empty_state(
                "No holdout artefacts on disk",
                "The verdict above came from the predictions cache, but the "
                "holdout run dir is missing its <code>summary.json</code>. "
                "Re-run <code>python -m backtest.holdout</code> for the integrity "
                "suite and per-band calibration table.",
            ),
            unsafe_allow_html=True,
        )
        return

    # ── 2. Integrity suite ───────────────────────────────────────────────────
    integrity = summary.get("integrity") or []
    n_warn = sum(1 for c in integrity if str(c.get("status")).upper() == "WARN")
    n_fail = sum(1 for c in integrity
                 if str(c.get("status")).upper() in ("FAIL", "ERROR"))
    note = (f"{n_fail} fail · {n_warn} warn" if (n_fail or n_warn)
            else "all clear")
    st.markdown(C.section("Integrity checks", note), unsafe_allow_html=True)
    st.markdown(integrity_rows_html(integrity), unsafe_allow_html=True)

    # ── 3. Per-odds-band calibration ─────────────────────────────────────────
    band_df = load_band_table(holdout)
    st.markdown(C.section("Calibration by odds band",
                          "actual win rate vs model vs de-vigged market"),
                unsafe_allow_html=True)
    st.markdown(band_table_html(band_df), unsafe_allow_html=True)
    st.caption("A/E = actual wins ÷ expected (mean probability). 1.00 is perfectly "
               "calibrated; below 1 over-confident, above 1 under-confident. "
               "“Closer fit” marks whichever side tracks the band’s true win rate "
               "more tightly — the market winning most bands is the honest tell.")

    # ── footer ───────────────────────────────────────────────────────────────
    betting = summary.get("betting") or {}
    model_meta = summary.get("model") or {}
    leak = model_meta.get("leakage_verified")
    leak_tone = "ok" if leak else "warn"
    leak_txt = "leakage verified" if leak else "leakage NOT verified"
    src = holdout.name if holdout is not None else "—"
    st.markdown(
        C.footer(
            f'{C.status_dot(leak_tone)}Holdout <span class="mono">{escape(src)}</span> '
            f'· {leak_txt}',
            f'{int(betting.get("n_bets") or 0):,} EV bets · '
            f'devig {escape(str(summary.get("devig_method") or "—"))}',
        ),
        unsafe_allow_html=True,
    )
