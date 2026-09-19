"""Performance — appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

Wraps ui/performance_dashboard.py (bankroll equity curve, rolling ROI, monthly
P&L, win-rate-by-odds, pending-bet settlement, bet history, PDF export). Run_path
re-executes it each rerun so its widgets stay live in the sidebar navigation.
"""
import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

runpy.run_path(str(_ROOT / "ui" / "performance_dashboard.py"), run_name="__main__")
