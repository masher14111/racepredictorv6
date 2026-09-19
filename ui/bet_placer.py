"""ui/bet_placer.py — Bet placement interface.

Select a predicted horse, enter stake, choose win or each-way, and confirm.
Bets are recorded via ``utils.bet_tracker.BetTracker`` into its own local
SQLite ledger; the sidebar shows live P&L.

**This is a separate, manual paper ledger from the canonical one.** It is not
``execution.tickets.TicketStore``/``paper_tickets`` (the ledger
``scripts.daily_paper_loop`` and the forward-release gate read), carries no
model-gate decision, no stake-exposure ceiling and no settlement-engine
(Rule 4/dead-heat/non-runner) treatment. Reports and reconciliation checks
must not fold this ledger's figures into the canonical one's totals unless a
future stage explicitly reconciles the two (Stage 21 / B10).

Launch:
    streamlit run ui/bet_placer.py
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
from utils import config_loader

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.timezone import now, to_local
from utils.currency import currency_for_venue
from utils.bet_tracker import BetTracker, Strategy, StopLossError
from ui._components import headline_win_prob
from ui._winrate import realized_rate_value

st.set_page_config(
    page_title="Place Bet — MASHR’s Predictor v6",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="auto",
)

_CACHE_PATH = _ROOT / "data" / "predictions.json"



# ── design system ─────────────────────────────────────────────
# Single-column bet form, so it takes the narrow shell.
# The rp-* classes this page emits are mapped onto the tokens in ui/_design.py
# (see its "legacy page class map" section).
from ui import _components as C  # noqa: E402
from ui._design import inject_design  # noqa: E402

inject_design(width="narrow")


# ── data / resource loaders ───────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_predictions() -> Optional[dict]:
    if not _CACHE_PATH.exists():
        return None
    try:
        with open(_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


@st.cache_resource
def _get_tracker() -> BetTracker:
    """One BetTracker singleton per server process; bankroll always re-queries SQLite."""
    bt = config_loader.get("bet_tracker", {}) or {}
    return BetTracker(
        initial_bankroll=bt.get("initial_bankroll", 1000.0),
        flat_stake=bt.get("flat_stake", 10.0),
        kelly_fraction=bt.get("kelly_fraction", 0.25),
        stop_loss_pct=bt.get("stop_loss_pct", 0.20),
        ew_fraction=0.20,
        ew_places=3,
    )


# ── formatting helpers ────────────────────────────────────────────────────────

def _fmt_time(iso: Optional[str]) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo:
            dt = to_local(dt)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return iso


def _risk_tier(composite: float) -> str:
    if composite >= 0.35:
        return "strong"
    if composite >= 0.25:
        return "good"
    if composite >= 0.15:
        return "moderate"
    return "speculative"


def _pct(val: Optional[float]) -> str:
    if val is None:
        return "—"
    return f"{float(val) * 100:.1f}%"


def _sym(venue: str) -> str:
    return "£" if currency_for_venue(venue) == "GBP" else "€"


# ── sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar(tracker: BetTracker) -> tuple[Strategy, float]:
    """Render P&L summary and stake settings. Returns (strategy, flat_stake)."""
    with st.sidebar:
        st.markdown(
            C.brand_lockup(),
            unsafe_allow_html=True,
        )

        summ = tracker.summary()
        br = summ["bankroll"]
        init_br = summ["initial_bankroll"]
        delta_pct = (br - init_br) / init_br * 100 if init_br else 0.0
        br_colour = "var(--value)" if br >= init_br else "var(--danger)"

        st.markdown("### Bankroll")
        st.markdown(
            f'<div style="font-size:27px;font-weight:700;color:{br_colour};'
            f'font-variant-numeric:tabular-nums">€{br:,.2f}</div>'
            f'<div style="font-size:12px;color:var(--muted);margin-top:2px">'
            f'{delta_pct:+.1f}% vs initial €{init_br:,.2f}</div>',
            unsafe_allow_html=True,
        )

        if summ["stop_loss_active"]:
            st.markdown(
                '<div style="background:var(--danger-wash);border:1px solid var(--danger);'
                'border-radius:6px;padding:8px 10px;margin-top:8px;font-size:12px;'
                'color:var(--danger);font-weight:500">⛔ Stop-loss active</div>',
                unsafe_allow_html=True,
            )

        st.markdown("### P&L")
        profit = summ["total_profit"]
        p_col = "var(--value)" if profit >= 0 else "var(--danger)"
        st.markdown(
            f'<div style="display:flex;gap:16px;flex-wrap:wrap">'
            f'<div><div style="font-size:10px;color:var(--muted);text-transform:uppercase;'
            f'letter-spacing:.05em">Total P&L</div>'
            f'<div style="font-size:18px;font-weight:700;color:{p_col};'
            f'font-variant-numeric:tabular-nums">{"+" if profit >= 0 else ""}€{profit:,.2f}</div></div>'
            f'<div><div style="font-size:10px;color:var(--muted);text-transform:uppercase;'
            f'letter-spacing:.05em">ROI</div>'
            f'<div style="font-size:18px;font-weight:700;color:{p_col}">'
            f'{summ["roi_pct"]:+.1f}%</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Stats")
        st.markdown(
            f'<div style="font-size:13px;color:var(--ink-2);line-height:2.1">'
            f'Bets placed: <b style="color:#fff">{summ["total_bets"]}</b><br>'
            f'Pending: <b style="color:#fff">{summ["pending_bets"]}</b><br>'
            f'Win rate (settled): <b style="color:#fff">'
            f'{realized_rate_value(summ.get("wins", 0), summ["total_bets"])}</b><br>'
            f'Max drawdown: <b style="color:#fff">{summ["max_drawdown_pct"]:.1f}%</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### Stake Strategy")
        strategy_label = st.selectbox(
            "Strategy",
            ["Flat", "Fractional Kelly (¼)", "Full Kelly"],
            index=0,
            label_visibility="collapsed",
            help=(
                "How each stake is sized:\n\n"
                "• **Flat** — bet the same fixed amount every time (the 'Flat stake' below), "
                "regardless of how strong the edge is.\n\n"
                "• **Fractional Kelly (¼)** — the Kelly formula works out the mathematically "
                "growth-optimal stake from your edge and the odds; we bet only a quarter of it. "
                "Staking a *fraction* of full Kelly trades a little growth for much lower "
                "swings in the bankroll, which is the standard safe choice.\n\n"
                "• **Full Kelly** — the full growth-optimal stake. Highest long-run growth but "
                "the most volatile; a losing run can cut the bankroll hard."
            ),
        )
        strategy_map = {
            "Flat": Strategy.FLAT,
            "Fractional Kelly (¼)": Strategy.FRACTIONAL_KELLY,
            "Full Kelly": Strategy.KELLY,
        }
        strategy = strategy_map[strategy_label]

        flat_stake = st.number_input(
            "Flat stake (€)",
            min_value=0.50,
            max_value=500.0,
            value=10.0,
            step=0.50,
            format="%.2f",
            help=(
                "The fixed cash amount (€) staked on each bet under the **Flat** strategy. "
                "It also acts as a safety cap on Kelly-suggested stakes, so a single bet "
                "never risks more than this. Default comes from `bet_tracker.flat_stake` "
                "in config.yaml."
            ),
        )

    return strategy, flat_stake


# ── horse detail card ─────────────────────────────────────────────────────────

def _horse_card_html(sel: dict, venue: str, race_time: str) -> str:
    tier = _risk_tier(sel.get("composite_score") or 0.0)
    dec_odds = sel.get("decimal_odds")
    # a price is a ratio — board fraction, no currency symbol, no FX
    odds_html = C.price_html(dec_odds) if dec_odds else '<span class="rp-sub">—</span>'
    ew_pill = '<span class="pill pill-ew" style="margin-left:8px">E/W ✓</span>' if sel.get("each_way_value") else ""
    model_edge = (sel.get("won_prob") or 0.0) - (sel.get("implied_prob") or 0.0)
    val_pill = (
        f'<span class="pill pill-val" style="margin-left:4px">+{model_edge*100:.1f}% edge</span>'
        if model_edge >= 0.05 else ""
    )
    return (
        f'<div class="horse-card tier-{tier}">'
        f'<div class="hc-name">{escape(sel.get("horse_name") or "—")}{ew_pill}{val_pill}</div>'
        f'<div class="hc-sub">'
        f'{escape(sel.get("jockey") or "—")} · {escape(sel.get("trainer") or "—")}'
        f' · {escape(venue)} {_fmt_time(race_time)}'
        f'</div>'
        f'<div class="hc-stats">'
        f'<div class="hc-stat"><div class="hc-stat-lbl">Best price</div>'
        f'<div class="hc-stat-val">{odds_html}</div></div>'
        f'<div class="hc-stat"><div class="hc-stat-lbl">Win %</div>'
        f'<div class="hc-stat-val">{_pct(headline_win_prob(sel))}</div></div>'
        f'<div class="hc-stat"><div class="hc-stat-lbl">Place %</div>'
        f'<div class="hc-stat-val">{_pct(sel.get("placed_2_prob"))}</div></div>'
        f'<div class="hc-stat"><div class="hc-stat-lbl">Score</div>'
        f'<div class="hc-stat-val">{sel.get("composite_score") or 0.0:.3f}</div></div>'
        f'<div class="hc-stat"><div class="hc-stat-lbl">Market</div>'
        f'<div class="hc-stat-val">{_pct(sel.get("implied_prob"))}</div></div>'
        f'</div>'
        f'</div>'
    )


# ── recent bets table ─────────────────────────────────────────────────────────

# Column headers are derived from the snake_case field names, which turned
# `odds_decimal` into "Odds Decimal" — naming the storage format rather than the
# thing. Overridden here; everything else title-cases cleanly.
_COL_LABELS = {"odds_decimal": "Best Price"}


def _render_recent_bets(tracker: BetTracker) -> None:
    all_bets = tracker.all_bets()
    if not all_bets:
        return
    st.markdown("---")
    st.markdown('<div class="section-label">Recent Bets</div>', unsafe_allow_html=True)
    df = pd.DataFrame(all_bets)
    show_cols = ["placed_at", "horse_name", "venue", "bet_type",
                 "stake", "odds_decimal", "outcome", "profit"]
    present = [c for c in show_cols if c in df.columns]
    display = df[present].tail(20).sort_values("placed_at", ascending=False).reset_index(drop=True)

    if "placed_at" in display.columns:
        def _fmt(v):
            try:
                dt = datetime.fromisoformat(v)
                return to_local(dt).strftime("%d %b %H:%M") if dt.tzinfo else v
            except Exception:
                return v
        display["placed_at"] = display["placed_at"].apply(_fmt)

    if "odds_decimal" in display.columns:
        # the ledger recorded a decimal; it still shows as a board price
        display["odds_decimal"] = display["odds_decimal"].apply(C.price_cell)

    display.columns = [_COL_LABELS.get(c, c.replace("_", " ").title())
                       for c in display.columns]
    st.dataframe(display, width="stretch", hide_index=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    tracker = _get_tracker()
    strategy, flat_stake = _render_sidebar(tracker)

    # propagate sidebar flat_stake into tracker for recommend_stake
    tracker._flat_stake = flat_stake

    preds = _load_predictions()
    races = (preds or {}).get("races") or []

    local_now = now()
    st.markdown(
        f'<div class="pg-hdr">'
        f'<h1>Place a Bet</h1>'
        f'<div class="sub">'
        f'{local_now.strftime("%A, %d %B %Y")} · {local_now.strftime("%H:%M")} IST'
        f' · {len(races)} race{"s" if len(races) != 1 else ""} in cache'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    if not races:
        st.info(
            "No predictions loaded. Run **Refresh predictions** from the Predictions page, "
            "or run `python -m models.predictor` first.",
            icon="🏇",
        )
        _render_recent_bets(tracker)
        return

    if tracker.is_stopped():
        st.markdown(
            '<div class="stop-banner">⛔ Stop-loss active — bankroll is below the safety floor. '
            'No new bets can be placed until the bankroll recovers.</div>',
            unsafe_allow_html=True,
        )

    # ── success banner (shown until user picks a new race/horse) ─────────────
    placed = st.session_state.get("last_placed")
    if placed:
        sym = _sym(placed["venue"])
        bet_lbl = "each-way" if placed["bet_type"] == "each_way" else "win"
        st.markdown(
            f'<div class="success-card">'
            f'<div class="success-icon">✅</div>'
            f'<div class="success-text">'
            f'<h3>Bet #{placed["bet_id"]} recorded</h3>'
            f'<p>{escape(placed["horse_name"])} · {sym}{placed["stake"]:.2f} {bet_lbl}'
            f' @ {C.fmt_price(placed["odds_decimal"])} · {escape(placed["venue"])}</p>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── race selector ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Select Race</div>', unsafe_allow_html=True)
    race_labels = [
        f"{r.get('venue','?')} @ {_fmt_time(r.get('race_time'))}  "
        f"({len(r.get('selections') or [])} selections, {r.get('field_size',0)} runners)"
        for r in races
    ]
    race_idx = st.selectbox(
        "Race", range(len(races)),
        format_func=lambda i: race_labels[i],
        label_visibility="collapsed",
        on_change=lambda: st.session_state.pop("last_placed", None),
    )
    race = races[race_idx]
    selections = race.get("selections") or []
    venue = race.get("venue") or "Unknown"
    race_time = race.get("race_time") or ""

    if not selections:
        st.warning("No model selections for this race.")
        _render_recent_bets(tracker)
        return

    # ── horse selector ────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Select Horse</div>', unsafe_allow_html=True)
    horse_labels = [
        f"#{s.get('rank','-')}  {s.get('horse_name','—')}"
        f"  ({C.fmt_price(s.get('decimal_odds'))}"
        f", score {s.get('composite_score',0):.3f})"
        for s in selections
    ]
    horse_idx = st.selectbox(
        "Horse", range(len(selections)),
        format_func=lambda i: horse_labels[i],
        label_visibility="collapsed",
        on_change=lambda: st.session_state.pop("pending_bet", None),
    )
    sel = selections[horse_idx]

    # ── horse detail card ─────────────────────────────────────────────────────
    st.markdown(_horse_card_html(sel, venue, race_time), unsafe_allow_html=True)

    # ── recommended stake ─────────────────────────────────────────────────────
    dec_odds = sel.get("decimal_odds") or 0.0
    win_prob = sel.get("won_prob") or 0.0
    recommended = (
        tracker.recommend_stake(dec_odds, win_prob, strategy)
        if dec_odds > 1.0 and win_prob > 0.0
        else (flat_stake if strategy == Strategy.FLAT else 0.0)
    )
    default_stake = max(recommended, 0.50) if recommended > 0 else flat_stake
    max_stake = max(tracker.bankroll, 0.50)
    ew_available = race.get("each_way_available", False) or bool(sel.get("each_way_value"))

    # ── stake form ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Stake &amp; Bet Type</div>', unsafe_allow_html=True)
    with st.form("bet_form", clear_on_submit=False):
        col_stake, col_type = st.columns([2, 1])
        with col_stake:
            stake_input = st.number_input(
                "Stake (€)",
                min_value=0.50,
                max_value=float(max_stake),
                value=float(min(default_stake, max_stake)),
                step=0.50,
                format="%.2f",
                help=f"Recommended ({strategy.value.replace('_',' ')}): €{recommended:.2f}",
            )
        with col_type:
            type_opts = ["Win", "Each Way"] if ew_available else ["Win"]
            bet_type_label = st.radio(
                "Bet type", type_opts,
                index=0,
                horizontal=True,
            )

        notes_input = st.text_input(
            "Notes (optional)",
            value="",
            placeholder="e.g. going suits, trainer hot streak…",
            max_chars=200,
        )

        submit = st.form_submit_button(
            "Review Bet →",
            type="primary",
            width="stretch",
            disabled=tracker.is_stopped(),
        )

    if submit:
        st.session_state["pending_bet"] = {
            "horse_name": sel.get("horse_name") or "—",
            "horse_id": sel.get("horse_id"),
            "venue": venue,
            "race_time": race_time,
            "odds_decimal": dec_odds,
            "stake": stake_input,
            "bet_type": "each_way" if bet_type_label == "Each Way" else "win",
            "composite_score": sel.get("composite_score"),
            "won_prob": win_prob,
            "strategy": strategy.value,
            "notes": notes_input.strip() or None,
        }
        st.session_state.pop("last_placed", None)

    # ── confirmation card ─────────────────────────────────────────────────────
    pending = st.session_state.get("pending_bet")
    if pending:
        p_sym = _sym(pending["venue"])
        bet_lbl = "Each Way" if pending["bet_type"] == "each_way" else "Win"
        ew_note = " (win + place legs, stake split ½ each)" if pending["bet_type"] == "each_way" else ""

        confirm_html = (
            '<div class="confirm-card">'
            '<div class="confirm-title">Confirm Bet</div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Horse</span>'
            f'<span class="confirm-val">{escape(pending["horse_name"])}</span></div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Race</span>'
            f'<span class="confirm-val">{escape(pending["venue"])} @ {_fmt_time(pending["race_time"])}</span></div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Best price</span>'
            f'<span class="confirm-val">{C.price_html(pending["odds_decimal"])}</span></div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Type</span>'
            f'<span class="confirm-val">{bet_lbl}{ew_note}</span></div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Total stake</span>'
            f'<span class="confirm-val" style="font-size:16px">{p_sym}{pending["stake"]:.2f}</span></div>'
            f'<div class="confirm-row"><span class="confirm-lbl">Strategy</span>'
            f'<span class="confirm-val">{pending["strategy"].replace("_"," ")}</span></div>'
            '</div>'
        )
        st.markdown(confirm_html, unsafe_allow_html=True)

        col_ok, col_cancel = st.columns(2)
        with col_ok:
            if st.button("✓ Confirm & Place", type="primary", width="stretch"):
                try:
                    bet_id = tracker.record_bet(
                        horse_name=pending["horse_name"],
                        odds_decimal=pending["odds_decimal"],
                        stake=pending["stake"],
                        bet_type=pending["bet_type"],
                        horse_id=pending.get("horse_id"),
                        venue=pending["venue"],
                        race_time=pending["race_time"],
                        composite_score=pending.get("composite_score"),
                        won_prob=pending.get("won_prob"),
                        strategy=Strategy(pending["strategy"]),
                        notes=pending.get("notes"),
                    )
                    st.session_state["last_placed"] = {
                        "bet_id": bet_id,
                        "horse_name": pending["horse_name"],
                        "venue": pending["venue"],
                        "stake": pending["stake"],
                        "bet_type": pending["bet_type"],
                        "odds_decimal": pending["odds_decimal"],
                    }
                    st.session_state.pop("pending_bet", None)
                    st.rerun()
                except StopLossError as exc:
                    st.error(f"Stop-loss active: {exc}")
                except Exception as exc:
                    st.error(f"Failed to record bet: {exc}")
        with col_cancel:
            if st.button("✕ Cancel", width="stretch"):
                st.session_state.pop("pending_bet", None)
                st.rerun()

    # ── recent bets ───────────────────────────────────────────────────────────
    _render_recent_bets(tracker)

    # ── footer ────────────────────────────────────────────────────────────────
    gen_at = (preds or {}).get("generated_at") or ""
    try:
        gen_str = to_local(datetime.fromisoformat(gen_at)).strftime("%d %b %H:%M IST")
    except Exception:
        gen_str = gen_at or "—"
    st.markdown(
        f'<div class="rp-footer">Prediction cache: {gen_str} &nbsp;·&nbsp; '
        f'Bankroll: €{tracker.bankroll:,.2f}</div>',
        unsafe_allow_html=True,
    )


main()
