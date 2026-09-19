"""Generate a static HTML preview of the value-betting UI treatment.

Reproduces the exact CSS + card/KPI markup that ui/app.py emits, populated with
synthetic value data, so the design can be screenshot-verified without a live
prediction cache or a running Streamlit server. Throwaway dev tool.
"""
from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CSS = re.search(r'_CSS = """(.*?)"""', (_ROOT / "ui" / "app.py").read_text(encoding="utf-8"), re.S).group(1)


def prob_bar(pct: float, color: str) -> str:
    return (
        f'<div class="pb-wrap"><div class="pb-bg">'
        f'<div class="pb-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'
        f'<span class="pb-val">{pct:.0f}%</span></div>'
    )


def ev_cell(ev: float | None, value_bet: bool, edge: float | None) -> str:
    if ev is None:
        return '<td class="ev ev-flat">—</td>'
    cls = "ev-pos" if value_bet else "ev-flat"
    title = f' title="Model {edge*100:+.1f}pp vs market"' if edge is not None else ""
    return f'<td class="ev {cls}"{title}>{ev*100:+.0f}%</td>'


def runner_row(rank, horse, jockey, odds, win, score, ev, value_bet, edge, ew=False):
    rk = {1: "rk1", 2: "rk2", 3: "rk3"}.get(rank, "")
    rk_cell = f'<span class="rk {rk}">{rank}</span>' if rk else f'<span style="color:var(--muted)">{rank or "—"}</span>'
    ew_badge = '<span class="ew-badge">E/W ✓</span>' if ew else ""
    val_chip = '<span class="val-chip"><span class="vdot"></span>VALUE</span>' if value_bet else ""
    return (
        f"<tr><td>{rk_cell}</td>"
        f"<td><strong>{horse}</strong>&nbsp;<span style='color:var(--muted);font-size:12px'>{jockey}</span>"
        f"&nbsp;{ew_badge}&nbsp;{val_chip}</td>"
        f"<td style='font-variant-numeric:tabular-nums'>€{odds:.2f}</td>"
        f"<td>{prob_bar(win, '#2563eb')}</td>"
        f"<td>{prob_bar(score, '#16a34a')}</td>"
        f"{ev_cell(ev, value_bet, edge)}</tr>"
    )


def race_card(venue, rtime, field, n_value, rows):
    value_pill = f'<span class="pill pill-value">{n_value} value</span>' if n_value else ""
    header = (
        f'<div class="rp-card-hdr"><span class="venue">{venue}</span>'
        f'<span class="rtime">{rtime}</span>'
        f'<span class="pill pill-field">{field} runners</span>'
        f'<span class="pill pill-eur">€</span>'
        f'<span class="pill pill-ew">E/W</span>{value_pill}</div>'
    )
    table = (
        '<div class="rp-card-body"><table class="rp-tbl"><thead><tr>'
        "<th>#</th><th>Horse · Jockey</th><th>Odds</th><th>Win %</th><th>Score</th>"
        '<th title="Expected value per unit stake, from the price-free model vs. market odds">EV</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    return f'<div class="rp-card">{header}{table}</div>'


def kpi(label, val, sub, color=None):
    style = f' style="color:{color}"' if color else ""
    return (
        f'<div class="rp-kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-val"{style}>{val}</div><div class="kpi-sub">{sub}</div></div>'
    )


kpi_row = (
    '<div class="rp-kpi-row">'
    + kpi("Races", 5, "upcoming")
    + kpi("Value bets", 3, "positive EV vs market", "#166534")
    + kpi("Selections", 15, "top-3 per race")
    + kpi("Each-Way", 4, "races flagged")
    + kpi("Models", "won, placed_2, showed", "active")
    + "</div>"
)

# Race 1 — a value bet that is the favourite (short price, model rates it higher)
race1 = race_card("Leopardstown", "14:25", 11, 2, [
    runner_row(1, "Galway Wind", "C O'Dwyer", 3.50, 41, 38, 0.18, True, 7.2, ew=True),
    runner_row(2, "Saffron Beach", "R Moore", 5.00, 24, 29, -0.04, False, -1.1),
    runner_row(3, "Mount Leinster", "W Lordan", 8.50, 14, 21, 0.22, True, 4.8, ew=True),
])

# Race 2 — no value bets; EV column present but muted
race2 = race_card("Newmarket", "15:00", 9, 0, [
    runner_row(1, "Highfield Lad", "T Marquand", 4.20, 32, 34, -0.08, False, -2.4),
    runner_row(2, "Crimson Tide", "O Murphy", 6.50, 19, 25, -0.02, False, -0.7, ew=True),
    runner_row(3, "Dunmore Dancer", "J Doyle", 11.0, 11, 18, -0.15, False, -3.1, ew=True),
])

html = f"""<!doctype html><html><head><meta charset="utf-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');</style>
{_CSS}
<style>body{{background:var(--bg);margin:0;font-family:Inter,system-ui,sans-serif;}}
.wrap{{max-width:1180px;margin:0 auto;padding:28px 36px;}}</style>
</head><body><div class="wrap">
<div class="rp-header"><h1>Race Predictor <span style="color:var(--muted);font-weight:400">v3</span></h1>
<div class="sub">Sunday, 15 June 2026&nbsp;·&nbsp;03:20 IST</div></div>
{kpi_row}
{race1}
{race2}
</div></body></html>"""

out = _ROOT / "data" / "_value_ui_preview.html"
out.write_text(html, encoding="utf-8")
print(out)
