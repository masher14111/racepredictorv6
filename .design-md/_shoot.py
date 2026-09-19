"""Screenshot the assets gallery with Playwright (full page)."""
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / ".design-md" / "_assets_gallery.html").as_uri()
out = ROOT / ".design-md" / "assets-gallery.png"

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1180, "height": 1200}, device_scale_factor=2)
    pg.goto(html)
    pg.wait_for_timeout(900)  # let fonts + arc transitions settle
    pg.screenshot(path=str(out), full_page=True)
    b.close()
print(f"wrote {out}")
