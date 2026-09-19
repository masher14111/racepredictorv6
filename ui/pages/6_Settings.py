"""Settings — appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

Wraps ui/settings.py (editable config: proxy pool, scraping/betting thresholds,
model training params, drift-retrain thresholds; atomic save to config.yaml).
Run_path re-executes it each rerun so its form stays live in the sidebar nav.
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "ui" / "settings.py"), run_name="__main__")
