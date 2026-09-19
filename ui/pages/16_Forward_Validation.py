"""Forward Validation — appears as a page in the multipage app (launch: `streamlit run ui/app.py`).

Three lanes of evidence kept visibly apart: historical backtest, paper/shadow
bets, and real execution (structurally empty). All logic lives in
``ui.forward_validation`` (Streamlit-free, unit tested); this file is the
Streamlit assembly only. Themed to the Prompt-19 design system via ``ui._design``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ui import forward_validation as FV  # noqa: E402
from ui._design import inject_design  # noqa: E402

st.set_page_config(page_title="Forward Validation · MASHR’s Predictor v6",
                   page_icon="🧪", layout="wide", initial_sidebar_state="auto")

inject_design()

from ui import _components as C  # noqa: E402

with st.sidebar:
    st.markdown(C.brand_lockup(), unsafe_allow_html=True)
    st.markdown("### Navigate")
    st.page_link("app.py", label="← All racecards")
    st.page_link("pages/14_Model_Honesty.py", label="Model honesty")
    st.page_link("pages/15_Model_Compare.py", label="Model comparison")
    st.page_link("pages/9_Paper_Betting.py", label="Paper betting")


FV.render()
