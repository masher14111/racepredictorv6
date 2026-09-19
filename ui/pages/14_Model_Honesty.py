"""Model Honesty — appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

The panel that tells the truth: for each model line, does it actually beat the
de-vigged market (GO/NO-GO), what's its closing-line value, do the integrity
checks pass, and how does its calibration look band-by-band. All logic lives in
``ui.model_honesty`` (Streamlit-free, unit tested); this file is the Streamlit
assembly only. Themed to the Prompt-19 design system via ``ui._design``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui import model_honesty as MH  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Model Honesty · MASHR’s Predictor v6",
                   page_icon="⚖️", layout="wide", initial_sidebar_state="auto")

inject_design()

from ui import _components as C  # noqa: E402

with st.sidebar:
    st.markdown(C.brand_lockup(), unsafe_allow_html=True)
    st.markdown("### Navigate")
    st.page_link("app.py", label="← All racecards")
    st.page_link("pages/10_Dashboard.py", label="Performance dashboard")


MH.render()
