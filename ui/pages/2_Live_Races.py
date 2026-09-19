"""Live Races — appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

The full live-odds / race-status UI lives in ui/live_races.py (also runnable
standalone). This thin wrapper re-executes it on every rerun (run_path, not
import — import would cache the module and skip the widget logic on reruns) so it
shows up in the sidebar navigation.
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "ui" / "live_races.py"), run_name="__main__")
