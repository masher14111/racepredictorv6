"""Scrape & Refresh — pull fresh live odds per bookmaker and rebuild predictions.

Appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

Scraping a book updates the live odds on disk (its cache + data/live_odds.parquet);
predictions only change after a rebuild (normalize -> build -> predict). So the
per-book buttons are for a quick targeted pull, and the rebuild buttons fold the
scraped odds into data/predictions.json that the dashboard reads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui.refresh_ops import SCRAPERS, scrape_one, rebuild_predictions
from utils import source_health
from utils.timezone import now

st.set_page_config(
    page_title="Scrape & Refresh — MASHR’s Predictor v6",
    page_icon="🛰️",
    layout="wide",
)

st.title("🛰️ Scrape & Refresh")
st.caption(
    "Pull fresh live odds from each bookmaker, then rebuild predictions. "
    "Scraping updates the odds on disk; click **Rebuild predictions** (or "
    "**Scrape all + rebuild**) to fold them into the dashboard."
)

# ── per-book scrape buttons ───────────────────────────────────────────────────
st.subheader("Scrape a bookmaker")
cols = st.columns(len(SCRAPERS))
for col, (key, label) in zip(cols, SCRAPERS.items()):
    with col:
        if st.button(f"Scrape {label}", width="stretch", key=f"scrape_{key}"):
            with st.spinner(f"Scraping {label}… (browser fallbacks can take ~30s)"):
                st.session_state["scrape_result"] = (label, scrape_one(key))

res = st.session_state.get("scrape_result")
if res:
    label, r = res
    if r["ok"]:
        st.success(f"✅ {label}: {r['summary']}.")
        if r.get("stale"):
            age = r.get("age_seconds")
            age_txt = f" (~{age / 60:.0f} min old)" if age is not None else ""
            st.warning(
                f"⚠️ {label} served **stale cached odds**{age_txt} — every live fetch "
                "tier failed. These are not fresh prices; rebuilt predictions will "
                "exclude them once they exceed the staleness TTL."
            )
        st.caption("Odds written to disk. Rebuild predictions below to use them.")
    else:
        st.error(f"❌ {label} failed: {r['error']}")

st.divider()

# ── per-source health ──────────────────────────────────────────────────────---
st.subheader("Source health")
health = source_health.get_health()
health_rows = []
for key, label in SCRAPERS.items():
    rec = health.get(key, {})
    age = rec.get("age_seconds")
    success_rate = rec.get("success_rate")
    health_rows.append({
        "Book": label,
        "Status": rec.get("status") or "unknown",
        "Success rate": f"{success_rate:.0%}" if success_rate is not None else "—",
        "Last success age": f"{age / 60:.0f} min ago" if age is not None else "—",
        "Rows / races": (
            f"{rec.get('last_row_count')} / {rec.get('last_race_count')}"
            if rec.get("last_row_count") is not None else "—"
        ),
        "Last rejection reason": rec.get("last_rejection_reason") or "—",
        "Proxy reachable": (
            "✅" if rec.get("proxy_reachable") is True
            else "❌" if rec.get("proxy_reachable") is False else "—"
        ),
    })
st.dataframe(health_rows, width="stretch", hide_index=True)

st.divider()

# ── rebuild / full refresh ────────────────────────────────────────────────────
st.subheader("Rebuild predictions")
st.caption(
    "Normalize the scraped sources, derive today's live runners, and re-score — "
    "this writes data/predictions.json. May take a minute or two."
)
c1, c2 = st.columns(2)
with c1:
    if st.button("🔁 Rebuild predictions (no scrape)", width="stretch"):
        prog = st.empty()
        out = rebuild_predictions(progress=lambda m: prog.info(m))
        prog.empty()
        st.session_state["rebuild_out"] = out
with c2:
    if st.button("🛰️ Scrape all + rebuild", type="primary", width="stretch"):
        prog = st.empty()
        summaries = []
        for key, label in SCRAPERS.items():
            prog.info(f"Scraping {label}…")
            r = scrape_one(key)
            summaries.append(f"{label}: {r['summary'] if r['ok'] else 'failed'}")
        st.session_state["scrape_all"] = summaries
        out = rebuild_predictions(progress=lambda m: prog.info(m))
        prog.empty()
        st.session_state["rebuild_out"] = out

scrape_all = st.session_state.pop("scrape_all", None)
if scrape_all:
    st.caption(" · ".join(scrape_all))

out = st.session_state.get("rebuild_out")
if out:
    if out["ok"]:
        st.cache_data.clear()  # force the dashboard's prediction cache to reload
        st.success(
            f"✅ Rebuilt: {out['races']} races scored "
            f"({out['live_runners']} live runners from {out['unified_rows']:,} unified rows). "
            "Open the main dashboard / Live Races to see them."
        )
        if out["live_runners"] == 0:
            st.warning(
                "No upcoming races found — the books returned nothing for today "
                "(or all scrapes failed)."
            )
    else:
        st.error(f"❌ Rebuild failed: {out['error']}")

st.divider()
st.caption(
    f"Now: {now().strftime('%A %d %b %Y · %H:%M IST')}. "
    "Equivalent CLI: `python -m scripts.refresh` (full) / `--no-scrape` (rebuild only)."
)
