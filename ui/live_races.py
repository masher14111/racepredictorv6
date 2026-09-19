"""Live race cards — MASHR’s Predictor v6.

Displays all runners from available odds sources with race status
(Upcoming / Starting Soon / In-Play / Finished) and highlights any
horse that appears in the active predictions cache.

Launch:
    streamlit run ui/live_races.py

Recommended manual refresh interval: every 5 minutes for live odds
accuracy.  Auto-refresh is intentionally absent — a full scrape +
normalize round-trip is ≈ 2–4 MB over the wire and counts against
any proxy data allowance.
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

from utils.timezone import now, to_local
from utils.currency import GBP_EUR_RATE, currency_for_venue, from_eur
from utils.odds_format import format_book_prices, to_fraction
from utils.stake_advice import recommend_stake
from ui._components import headline_win_prob

# ── page config (first Streamlit call) ───────────────────────────────────────
st.set_page_config(
    page_title="Live Races — MASHR’s Predictor v6",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="auto",
)

_UNIFIED_PATH    = _ROOT / "data" / "unified_races.parquet"
_LIVE_ODDS_PATH  = _ROOT / "data" / "live_odds.parquet"
_LSB_CACHE_PATH  = _ROOT / "data" / "cache" / "livescorebet.json"
_PP_CACHE_PATH   = _ROOT / "data" / "cache" / "paddy_power.json"
_PREDICTIONS_PATH = _ROOT / "data" / "predictions.json"



# ── design system ─────────────────────────────────────────────
# Live race board. Was 184 lines of its own light stylesheet.
# The rp-* classes this page emits are mapped onto the tokens in ui/_design.py
# (see its "legacy page class map" section).
from ui import _components as C  # noqa: E402
from ui._design import inject_design  # noqa: E402

inject_design()


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_iso(val) -> Optional[str]:
    """Normalise race_time from any type to an ISO string, or None."""
    if val is None:
        return None
    if isinstance(val, str):
        return val if val.strip() else None
    if hasattr(val, "isoformat"):
        return val.isoformat()
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return str(val)


def _fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "TBC"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo:
            dt = to_local(dt)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return iso


def _fmt_odds(v: Optional[float]) -> str:
    return f"{v:.2f}" if v and not pd.isna(v) else "—"


def _first_text(*vals) -> str:
    """First value that is real text. Guards against NaN, which is truthy and
    would otherwise render as a literal "nan" in the card."""
    for v in vals:
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "<na>"):
            return s
    return "—"


def _data_age(fetched_at: Optional[str]) -> str:
    if not fetched_at:
        return "age unknown"
    try:
        dt = datetime.fromisoformat(fetched_at)
        if dt.tzinfo:
            dt = to_local(dt)
        secs = (now() - dt).total_seconds()
        if secs < 60:
            return f"{int(secs)}s ago"
        if secs < 3600:
            return f"{int(secs / 60)}m ago"
        return f"{secs / 3600:.1f}h ago"
    except (ValueError, TypeError):
        return "age unknown"


# ── race status ───────────────────────────────────────────────────────────────

_STATUS = {
    "Upcoming":      ("pill-upcoming", "🕐"),
    "Starting Soon": ("pill-starting", "🟡"),
    "In-Play":       ("pill-inplay",   "🟢"),
    "Finished":      ("pill-finished", "✓"),
}


def _race_status(race_time_iso: Optional[str]) -> tuple[str, str, str]:
    """Return (label, pill_css_class, icon)."""
    if not race_time_iso:
        return ("Upcoming", "pill-upcoming", "🕐")
    try:
        rt = datetime.fromisoformat(race_time_iso)
        local_rt = to_local(rt) if rt.tzinfo else rt
        diff_min = (local_rt - now()).total_seconds() / 60
        if diff_min > 5:
            label = "Upcoming"
        elif diff_min > 0:
            label = "Starting Soon"
        elif diff_min > -25:
            label = "In-Play"
        else:
            label = "Finished"
    except (ValueError, TypeError):
        label = "Upcoming"
    css, icon = _STATUS[label]
    return label, css, icon


# ── data loading ──────────────────────────────────────────────────────────────

def _live_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only today-and-future rows — the live-card universe.

    ``unified_races.parquet`` holds the full multi-year history (~584k rows /
    ~31k races). Building cards for all of it costs minutes and then gets thrown
    away by the status filter, so the page appears to hang. A live page only ever
    shows races whose ``race_date`` is today or later (earlier races today still
    render as 'Finished' off their race_time), so we drop the historical tail
    up front — turning a ~580k-row, ~4-minute build into a ~2k-row, ~1s one."""
    if "race_date" not in df.columns:
        return df
    rd = pd.to_datetime(df["race_date"], errors="coerce", utc=True)
    today = pd.Timestamp(now().date(), tz="UTC")
    return df[rd >= today]


@st.cache_data(ttl=0)
def _load_races() -> tuple[pd.DataFrame, list[str]]:
    """
    Load runner rows from all available sources. Returns (df, source_list).

    Priority order:
      1. unified_races.parquet   — normalized, multi-source, best quality
      2. live_odds.parquet       — boylesports live scrape
      3. data/cache/*.json       — livescorebet + paddy_power raw caches
    """
    frames: list[pd.DataFrame] = []
    sources_found: list[str] = []

    # 1 — unified (best). Restrict to the live window before anything else;
    # the full history would make the per-race build take minutes.
    if _UNIFIED_PATH.exists():
        try:
            df = _live_only(pd.read_parquet(_UNIFIED_PATH))
            if not df.empty:
                frames.append(df)
                sources_found.append("unified")
        except Exception:
            pass

    # 2 — boylesports live_odds
    if not frames and _LIVE_ODDS_PATH.exists():
        try:
            df = pd.read_parquet(_LIVE_ODDS_PATH)
            if not df.empty:
                frames.append(df)
                sources_found.append("boylesports")
        except Exception:
            pass

    # 3 — livescorebet.json cache
    if _LSB_CACHE_PATH.exists():
        try:
            with open(_LSB_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            rows = raw.get("rows") or []
            if rows:
                df_lsb = pd.DataFrame(rows)
                df_lsb["fetched_at"] = raw.get("fetched_at")
                df_lsb["source"] = df_lsb.get("source", pd.Series(["livescorebet"] * len(df_lsb)))
                frames.append(df_lsb)
                sources_found.append("livescorebet")
        except Exception:
            pass

    # 4 — paddy_power.json cache (flatten races → markets → selections)
    if _PP_CACHE_PATH.exists():
        try:
            with open(_PP_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            fetched_at = raw.get("fetched_at")
            pp_rows: list[dict] = []
            for race in raw.get("races") or []:
                venue = race.get("venue", "")
                race_time = race.get("race_time", "")
                race_id = race.get("race_id", "")
                for mkt in race.get("markets") or []:
                    for sel in mkt.get("selections") or []:
                        pp_rows.append({
                            "race_id":       race_id,
                            "race_time":     race_time,
                            "venue":         venue,
                            "market_type":   mkt.get("market_type", "WIN"),
                            "horse_name":    sel.get("horse_name", ""),
                            "odds_decimal":  sel.get("odds_decimal"),
                            "sp":            sel.get("sp"),
                            "source":        "paddy_power",
                            "fetched_at":    fetched_at,
                            "currency":      "EUR",
                        })
            if pp_rows:
                frames.append(pd.DataFrame(pp_rows))
                sources_found.append("paddy_power")
        except Exception:
            pass

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)
    return combined, sources_found


@st.cache_data(ttl=0)
def _load_pred_lookup() -> dict[str, dict]:
    """
    Build {horse_id: runner_dict, horse_name_lower: runner_dict} from
    predictions.json.  Name-based lookup is the fallback when sources
    (e.g. boylesports) don't expose a horse_id.
    """
    if not _PREDICTIONS_PATH.exists():
        return {}
    try:
        with open(_PREDICTIONS_PATH, encoding="utf-8") as f:
            preds = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    lookup: dict[str, dict] = {}
    for race in preds.get("races") or []:
        for sel in race.get("selections") or []:
            hid = sel.get("horse_id")
            hname = sel.get("horse_name")
            if hid:
                lookup[str(hid)] = sel
            if hname:
                lookup[hname.strip().lower()] = sel
    return lookup


# ── race grouping ─────────────────────────────────────────────────────────────

def _build_races(df: pd.DataFrame) -> list[dict]:
    """
    Group flat runner rows into race dicts, deduplicating runners across
    bookmakers (best odds wins when a horse appears in multiple sources).
    """
    if "market_type" in df.columns:
        mt = df["market_type"].str.upper()
        win_mask = mt == "WIN"
        df = df[win_mask] if win_mask.any() else df

    # Normalise race_time to ISO str so groupby works consistently
    if "race_time" in df.columns:
        df = df.copy()
        df["race_time"] = df["race_time"].apply(_to_iso)

    group_cols = [c for c in ("venue", "race_time") if c in df.columns]
    if not group_cols:
        return []

    races: list[dict] = []
    for keys, grp in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        kv = dict(zip(group_cols, keys))

        venue     = str(kv.get("venue") or "Unknown")
        race_time = kv.get("race_time")
        race_time = str(race_time) if race_time and not pd.isna(race_time) else None

        # Dedupe by horse_name (best odds first, then first occurrence)
        id_col = next((c for c in ("horse_id", "horse_name") if c in grp.columns), None)
        if id_col:
            runners_df = (
                grp.sort_values("odds_decimal", ascending=False, na_position="last")
                .groupby(id_col, as_index=False)
                .first()
            )
        else:
            runners_df = grp.copy()

        # Keep every book's price before the dedupe throws them away — the card
        # shows "Paddy 1/2 · Boyle 6/4", not just the best of the two.
        prices: dict[str, dict[str, float]] = {}
        if id_col and "source" in grp.columns and "odds_decimal" in grp.columns:
            for key, src, dec in zip(grp[id_col], grp["source"], grp["odds_decimal"]):
                if not src or dec is None or pd.isna(dec):
                    continue
                try:
                    dec_f = float(dec)
                except (TypeError, ValueError):
                    continue
                if dec_f <= 1.0:
                    continue
                # Same book twice for one horse (a re-poll) — keep the best price.
                book = prices.setdefault(str(key), {})
                name = str(src)
                if dec_f > book.get(name, 0.0):
                    book[name] = dec_f

        runners = runners_df.to_dict("records")
        if prices:
            for r in runners:
                r["odds_by_book"] = prices.get(str(r.get(id_col)), {})
        runners.sort(key=lambda r: r.get("odds_decimal") or float("inf"))

        sources   = sorted({str(r.get("source") or "") for r in runners if r.get("source")})
        fetched_at_vals = grp["fetched_at"].dropna() if "fetched_at" in grp.columns else pd.Series([], dtype=object)
        fetched_at = _to_iso(fetched_at_vals.max()) if not fetched_at_vals.empty else None

        races.append({
            "venue":      venue,
            "race_time":  race_time,
            "field_size": len(runners),
            "runners":    runners,
            "sources":    sources,
            "fetched_at": fetched_at,
        })

    races.sort(key=lambda r: r.get("race_time") or "")
    return races


# ── card renderer ─────────────────────────────────────────────────────────────

_RK_CSS = {1: "rk1", 2: "rk2", 3: "rk3"}


def _render_card(race: dict, pred_lookup: dict[str, dict]) -> None:
    venue      = race["venue"]
    race_time  = race.get("race_time")
    field_size = race["field_size"]
    runners    = race.get("runners") or []
    sources    = race.get("sources") or []

    ccy        = currency_for_venue(venue)
    sym        = "£" if ccy == "GBP" else "€"
    ccy_cls    = "pill-gbp" if ccy == "GBP" else "pill-eur"
    status_label, status_css, status_icon = _race_status(race_time)

    src_pills = "".join(
        f'<span class="pill pill-src">{escape(s)}</span>' for s in sources
    )

    header = (
        f'<div class="rp-card-hdr">'
        f'<span class="venue">{escape(venue)}</span>'
        f'<span class="rtime">{_fmt_time(race_time)}</span>'
        f'<span class="pill {status_css}">{status_icon} {escape(status_label)}</span>'
        f'<span class="pill pill-field">{field_size} runners</span>'
        f'<span class="pill {ccy_cls}">{sym}</span>'
        f'{src_pills}'
        f'</div>'
    )

    rows_html = ""
    for runner in runners:
        hid   = str(runner.get("horse_id") or "")
        hname = str(runner.get("horse_name") or "—")
        # NaN is truthy, so a plain `or` chain renders a literal "nan" here.
        jockey = _first_text(runner.get("jockey_name"), runner.get("jockey"))

        pred = pred_lookup.get(hid) or pred_lookup.get(hname.strip().lower())
        row_cls = ' class="pred-row"' if pred else ""

        rank_html = ""
        pred_badge = ""
        model_prob_html = ""
        wp = None
        if pred:
            pred_badge = '<span class="pred-badge">★ Predicted</span>'
            rank = pred.get("rank")
            rk_cls = _RK_CSS.get(rank, "")
            if rk_cls:
                rank_html = f'<span class="rk {rk_cls}">{rank}</span>'
            wp = headline_win_prob(pred)
            if wp is not None:
                model_prob_html = (
                    f'<span style="color:var(--info);font-weight:600">'
                    f'{wp * 100:.0f}%</span>'
                )

        # Per-book prices. Prefer what the card carries; fall back to the
        # prediction's own odds_by_book so predicted runners still show a split.
        by_book = runner.get("odds_by_book") or (pred or {}).get("odds_by_book") or {}
        book_rows = format_book_prices(by_book)

        odds_val = runner.get("odds_decimal")
        if pd.isna(odds_val) if odds_val is not None else True:
            odds_val = None
        if odds_val is None and book_rows:
            odds_val = book_rows[0]["decimal"]

        # Odds are a RATIO, not money — no currency symbol, no FX conversion.
        # The venue's currency only matters for the stake and the return below.
        if odds_val:
            odds_cell = (
                f'<span style="font-weight:600">{to_fraction(odds_val)}</span>'
                f'<span style="color:var(--muted);font-size:11px">'
                f'&nbsp;{float(odds_val):.2f}</span>'
            )
        else:
            odds_cell = "—"

        if book_rows:
            books_cell = " ".join(
                f'<span class="bk{" bk-best" if r["is_best"] else ""}" '
                f'title="{escape(r["label"])} {escape(r["fraction"])} '
                f'({r["decimal"]:.2f})">'
                f'{escape(r["label"])}&nbsp;{escape(r["fraction"])}</span>'
                for r in book_rows
            )
        else:
            books_cell = '<span style="color:var(--muted)">—</span>'

        implied = (
            f'{100 / float(odds_val):.1f}%'
            if odds_val and float(odds_val) > 1.0
            else "—"
        )

        # Recommended PAPER stake. Sized off a calibrated prob only, and vetoed
        # by the value layer's own EV — see utils.stake_advice. Zero for
        # anything without a real edge, which is most runners.
        advice = recommend_stake(
            (pred or {}).get("value_win_prob") or wp,
            odds_val,
            market_ev=(pred or {}).get("expected_value"),
        )
        if advice.is_bet:
            # EUR stake x a dimensionless price = EUR return; no FX enters here.
            # A GBP venue's book still deals in pounds, so surface that too.
            ret_eur = advice.stake * float(odds_val)
            tip = f"{advice.reason}. Returns about €{ret_eur:,.2f} if it wins."
            if ccy == "GBP":
                tip += (
                    f" At this UK venue the book takes about "
                    f"£{from_eur(advice.stake, ccy):,.2f} and pays "
                    f"£{from_eur(ret_eur, ccy):,.2f} (rate {GBP_EUR_RATE:.2f})."
                )
            stake_cell = (
                f'<span class="stake" title="{escape(tip)} Paper only.">'
                f'€{advice.stake:,.2f}'
                f'<span class="stake-band">{escape(advice.band)}</span>'
                f"</span>"
            )
        else:
            stake_cell = (
                f'<span style="color:var(--muted)" '
                f'title="{escape(advice.reason)}">—</span>'
            )

        rows_html += (
            f"<tr{row_cls}>"
            f"<td>{rank_html}{escape(hname)}&nbsp;{pred_badge}</td>"
            f"<td style='color:var(--muted);font-size:12px'>{escape(jockey)}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{odds_cell}</td>"
            f"<td>{books_cell}</td>"
            f"<td style='color:var(--muted)'>{implied}</td>"
            f"<td>{model_prob_html}</td>"
            f"<td style='font-variant-numeric:tabular-nums'>{stake_cell}</td>"
            f"</tr>"
        )

    if not rows_html:
        rows_html = (
            "<tr><td colspan='7' style='color:var(--muted);text-align:center;"
            "padding:16px'>No runners available</td></tr>"
        )

    age_str = _data_age(race.get("fetched_at"))
    body = (
        '<div class="rp-card-body">'
        '<table class="rp-tbl"><thead><tr>'
        "<th>Horse</th><th>Jockey</th><th>Best price</th><th>By bookmaker</th>"
        "<th>Implied</th><th>Model Win %</th>"
        '<th title="Fractional-Kelly paper stake on the calibrated win '
        'probability. Paper only - never a real-money recommendation.">'
        "Paper stake</th>"
        f'</tr></thead><tbody>{rows_html}</tbody></table>'
        f'<div class="rp-card-meta">Data: {escape(age_str)}</div>'
        "</div>"
    )

    st.markdown(
        f'<div class="rp-card">{header}{body}</div>',
        unsafe_allow_html=True,
    )


# ── refresh helper ────────────────────────────────────────────────────────────

def _do_refresh() -> str:
    """
    Trigger Paddy Power scrape then normalize → unified_races.parquet.
    Returns a human-readable status string.  Boylesports (DOM scraper)
    is intentionally skipped here — run it separately to avoid heavy
    per-event HTTP requests eating proxy allowance.
    """
    msgs: list[str] = []
    try:
        from scraper.paddy_power import scrape as pp_scrape
        result = pp_scrape(force=True)
        n = len(result.get("races") or [])
        msgs.append(f"Paddy Power: {n} races fetched")
    except Exception as exc:
        msgs.append(f"Paddy Power: skipped ({type(exc).__name__})")

    try:
        from utils.normalizer import normalize
        df_norm = normalize()
        msgs.append(f"Normalized: {len(df_norm)} rows → unified_races.parquet")
    except Exception as exc:
        msgs.append(f"Normalize: skipped ({type(exc).__name__})")

    return " · ".join(msgs) or "Nothing refreshed"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            C.brand_lockup(),
            unsafe_allow_html=True,
        )

        st.markdown("### Filters")
        status_filter: list[str] = st.multiselect(
            "Race status",
            options=["Upcoming", "Starting Soon", "In-Play", "Finished"],
            default=["Upcoming", "Starting Soon", "In-Play"],
            label_visibility="collapsed",
        )
        pred_only: bool = st.checkbox(
            "Show only races with model predictions",
            value=False,
        )

        st.markdown("---")
        st.markdown("### Refresh")
        st.markdown(
            '<div class="rp-tip">'
            "Recommended interval: <strong>every 5 min</strong> for live odds.<br>"
            "A full scrape + normalize is ≈ 2–4 MB — auto-refresh is off "
            "to protect proxy bandwidth."
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh odds now"):
            with st.spinner("Scraping & normalizing…"):
                msg = _do_refresh()
                _load_races.clear()
                _load_pred_lookup.clear()
            st.success(msg)
            st.rerun()

    # ── load data ──────────────────────────────────────────────────────────────
    df, sources_found = _load_races()
    pred_lookup = _load_pred_lookup()

    # ── page header ────────────────────────────────────────────────────────────
    local_now = now()
    st.markdown(
        f'<div class="rp-header">'
        f'<h1>Live Race Cards '
        f'<span style="color:var(--muted);font-weight:400">v3</span></h1>'
        f'<div class="sub">'
        f'{local_now.strftime("%A, %d %B %Y")}'
        f'&nbsp;·&nbsp;{local_now.strftime("%H:%M")} IST'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── empty state ────────────────────────────────────────────────────────────
    if df.empty:
        st.markdown(
            '<div class="rp-empty">'
            '<div class="ei">🏇</div>'
            '<h3>No race data available</h3>'
            '<p>Click <strong>Refresh odds now</strong> in the sidebar to scrape '
            'live cards, or run '
            '<code>python -m scraper.paddy_power</code> from the terminal.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── group into races ───────────────────────────────────────────────────────
    races = _build_races(df)

    # Apply status filter
    if status_filter:
        races = [
            r for r in races
            if _race_status(r.get("race_time"))[0] in status_filter
        ]

    # Apply predictions-only filter
    if pred_only and pred_lookup:
        def _has_pred(r: dict) -> bool:
            for runner in r.get("runners") or []:
                hid   = str(runner.get("horse_id") or "")
                hname = str(runner.get("horse_name") or "").strip().lower()
                if pred_lookup.get(hid) or pred_lookup.get(hname):
                    return True
            return False
        races = [r for r in races if _has_pred(r)]

    # ── KPI strip ──────────────────────────────────────────────────────────────
    total_races   = len(races)
    total_runners = sum(r["field_size"] for r in races)
    inplay_count  = sum(
        1 for r in races if _race_status(r.get("race_time"))[0] == "In-Play"
    )
    pred_races    = sum(
        1 for r in races
        if any(
            pred_lookup.get(str(runner.get("horse_id") or "")) or
            pred_lookup.get(str(runner.get("horse_name") or "").strip().lower())
            for runner in (r.get("runners") or [])
        )
    )

    st.markdown(
        f'<div class="rp-kpi-row">'

        f'<div class="rp-kpi">'
        f'<div class="kpi-label">Races</div>'
        f'<div class="kpi-val">{total_races}</div>'
        f'<div class="kpi-sub">showing</div></div>'

        f'<div class="rp-kpi">'
        f'<div class="kpi-label">Runners</div>'
        f'<div class="kpi-val">{total_runners}</div>'
        f'<div class="kpi-sub">across all races</div></div>'

        f'<div class="rp-kpi">'
        f'<div class="kpi-label">In-Play</div>'
        f'<div class="kpi-val">{inplay_count}</div>'
        f'<div class="kpi-sub">currently running</div></div>'

        f'<div class="rp-kpi">'
        f'<div class="kpi-label">With Predictions</div>'
        f'<div class="kpi-val">{pred_races}</div>'
        f'<div class="kpi-sub">races flagged</div></div>'

        f'</div>',
        unsafe_allow_html=True,
    )

    # ── race cards ──────────────────────────────────────────────────────────────
    if not races:
        st.markdown(
            '<div class="rp-empty">'
            '<div class="ei">🔍</div>'
            '<h3>No races match the current filters</h3>'
            '<p>Adjust the status or prediction filter in the sidebar.</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for race in races:
            _render_card(race, pred_lookup)

    # ── page footer ────────────────────────────────────────────────────────────
    src_label = ", ".join(sources_found) if sources_found else "none"
    sdot_cls  = "sdot-ok" if sources_found else "sdot-warn"
    fetched_vals = df["fetched_at"].dropna() if "fetched_at" in df.columns else pd.Series([], dtype=object)
    latest_fetch = _data_age(_to_iso(fetched_vals.max()) if not fetched_vals.empty else None)
    st.markdown(
        f'<div class="rp-footer">'
        f'<span>'
        f'<span class="sdot {sdot_cls}"></span>'
        f'Sources: {escape(src_label)}'
        f'</span>'
        f'<span>'
        f'{total_races} races · {total_runners} runners · '
        f'data {escape(latest_fetch)}'
        f'</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


main()
