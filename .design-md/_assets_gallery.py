"""Render an in-context gallery of the in-house CSS/SVG assets to a single HTML
file, using the REAL design system stylesheet (ui/_design._CSS) so the marks are
shown exactly as they appear in the app. Screenshot target for ui-26.

    python .design-md/_assets_gallery.py   # writes .design-md/_assets_gallery.html
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from ui._design import _CSS  # the actual stylesheet
from ui import _icons as I
from ui import _components as C

OUT = _ROOT / ".design-md" / "_assets_gallery.html"

ICONS = ["race", "runner", "value", "bankroll", "calibration", "trophy",
         "selection", "clock"]


def icon_card(name: str, tone: str) -> str:
    return (
        f'<div class="cell">'
        f'<span class="rp-ico-plate {tone}">{I.icon(name, 18)}</span>'
        f'<div class="cap"><b>{name}</b><span>{tone}</span></div>'
        f'</div>'
    )


def kpi_with_icon(plate_tone: str, icon_name: str, label: str, value: str,
                  vtone: str = "") -> str:
    vcls = "k-val" + (" value" if vtone == "value" else "")
    return (
        f'<div class="rp-kpi"><div class="k-head">'
        f'<span class="rp-ico-plate {plate_tone}">{I.icon(icon_name, 16)}</span>'
        f'<div class="k-lab">{label}</div></div>'
        f'<div class="{vcls}">{value}</div></div>'
    )


def build() -> str:
    icon_grid = "".join(
        icon_card(n, t) for n, t in zip(
            ICONS, ["oxblood", "ink", "value", "amber", "info", "oxblood",
                    "value", "ink"])
    )

    kpi_rail = (
        '<div class="rp-kpis">'
        + kpi_with_icon("info", "race", "Races", "11")
        + kpi_with_icon("value", "value", "Value bets", "4", "value")
        + kpi_with_icon("oxblood", "selection", "Selections", "33")
        + kpi_with_icon("amber", "calibration", "Each-way", "6")
        + '</div>'
    )

    meters = (
        '<div class="meter-row">'
        + "".join(
            f'<div class="cell"><div>{I.strength_meter(v, tone=t, label=f"{int(v*100)}%")}</div>'
            f'<div class="cap"><span>{t} · {int(v*100)}%</span></div></div>'
            for v, t in [(0.8, "info"), (0.6, "value"), (0.4, "amber"), (0.2, "oxblood")]
        )
        + '</div>'
    )

    gauges = (
        '<div class="meter-row">'
        + "".join(
            f'<div class="cell">{I.prob_gauge(p, 84, tone=t)}'
            f'<div class="cap"><span>gauge · {t}</span></div></div>'
            for p, t in [(0.72, "info"), (0.45, "value"), (0.28, "amber")]
        )
        + '</div>'
    )

    logos = (
        '<div class="meter-row" style="align-items:center">'
        + "".join(f'<div class="cell">{I.logo_mark(sz)}'
                  f'<div class="cap"><span>logo {sz}px</span></div></div>'
                  for sz in (24, 32, 48, 64))
        + f'<div class="cell">{C.brand_lockup()}<div class="cap"><span>lockup</span></div></div>'
        + '</div>'
    )

    empty_motif = (
        '<div class="rp-empty hero-motif">'
        + I.horse_illustration(150)
        + '<h3>No racecards loaded yet</h3>'
        '<p>Scrape today&rsquo;s cards, then run the predictor to see ranked '
        'runners, value flags and confidence here.</p>'
        '</div>'
    )
    empty_finish = (
        '<div class="rp-empty hero-motif">'
        + I.finishline_scene(160)
        + '<h3>Racing&rsquo;s done for today</h3>'
        '<p>All of today&rsquo;s races have gone off. Check back tomorrow.</p>'
        '</div>'
    )

    hero = (
        '<div class="rp-hero">'
        + I.horse_illustration(160)
        + '<h3 style="font-family:var(--f-serif);font-size:var(--t-xl);'
        'color:var(--ink);margin:0 0 8px">Race Predictor</h3>'
        '<p style="color:var(--ink-2);max-width:46ch;margin:0 auto">A form &amp; '
        'value desk for UK &amp; Irish racing — model-ranked, calibrated, honest.</p>'
        '</div>'
    )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Race Predictor — in-house assets</title>
{_CSS}
<style>
  body {{ background: var(--bg); margin: 0; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 40px 32px 80px; }}
  .gtitle {{ font-family: var(--f-serif); color: var(--ink); font-size: 30px;
    margin: 0 0 4px; }}
  .gsub {{ color: var(--muted); font-size: 14px; margin: 0 0 32px;
    font-family: var(--f-sans); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px,1fr));
    gap: 14px; margin-bottom: 36px; }}
  .meter-row {{ display: flex; flex-wrap: wrap; gap: 28px; margin-bottom: 36px; }}
  .cell {{ display: flex; flex-direction: column; align-items: center; gap: 10px;
    padding: 16px; background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r-md); }}
  .cap {{ display: flex; flex-direction: column; align-items: center; gap: 1px;
    font-family: var(--f-sans); }}
  .cap b {{ color: var(--ink); font-size: 13px; }}
  .cap span {{ color: var(--muted); font-size: 11px; text-transform: uppercase;
    letter-spacing: .07em; }}
</style></head>
<body><div class="wrap">
  <h1 class="gtitle">In-house visual assets</h1>
  <p class="gsub">CSS / SVG only · theme-driven (OKLCH) · no external images · ui/_icons.py</p>

  {C.section("Logo & brand mark")}
  {logos}

  {C.section("Icon set", "8 glyphs · currentColor · 24-grid line-art")}
  <div class="grid">{icon_grid}</div>

  {C.section("KPI rail with icon plates")}
  {kpi_rail}

  {C.section("Strength meters", "segmented · fill-count carries the read")}
  {meters}

  {C.section("Probability gauges", "semicircular arc dial")}
  {gauges}

  {C.section("Empty state — galloping horse illustration")}
  {empty_motif}

  {C.section("Empty state — finish-line scene")}
  {empty_finish}

  {C.section("Hero band — in-house track-rail motif (no photo)")}
  {hero}
</div></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
