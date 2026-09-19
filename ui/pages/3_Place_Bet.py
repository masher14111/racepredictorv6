"""Place Bet — appears as a page in the multipage app (launch: `streamlit run ui/app.py`),
alongside the main dashboard and Test Results.

The full bet-placement UI (race/horse picker, stake form, win/each-way, confirm,
BetTracker P&L sidebar) lives in ui/bet_placer.py, which is also runnable
standalone via `streamlit run ui/bet_placer.py`. This thin wrapper re-executes
that module on every rerun (run_path, not import — import would cache the module
and skip the widget logic on reruns) so the bet form shows up in the sidebar nav.
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "ui" / "bet_placer.py"), run_name="__main__")
