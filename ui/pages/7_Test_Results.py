"""Test Results — Streamlit dashboard for the pytest suite.

Renders the JUnit XML produced by:
    python -m pytest -q --junitxml=reports/testrun/junit.xml

Appears as a page in the multipage app (launch: `streamlit run ui/app.py`),
alongside the main dashboard.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_JUNIT_PATH = _ROOT / "reports" / "testrun" / "junit.xml"

st.set_page_config(page_title="Test Results · MASHR’s Predictor v6", page_icon="🧪", layout="wide")


def _status(case: ET.Element) -> str:
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def _detail(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        node = case.find(tag)
        if node is not None:
            msg = node.get("message", "") or ""
            body = (node.text or "").strip()
            return (msg + ("\n\n" + body if body else "")).strip()
    return ""


@st.cache_data(show_spinner=False)
def _load(path_str: str, mtime: float) -> tuple[dict, pd.DataFrame]:
    root = ET.parse(path_str).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    summary = {
        "tests": int(suite.get("tests", 0)),
        "failures": int(suite.get("failures", 0)),
        "errors": int(suite.get("errors", 0)),
        "skipped": int(suite.get("skipped", 0)),
        "time": float(suite.get("time", 0.0)),
        "timestamp": suite.get("timestamp", ""),
    }
    summary["passed"] = (
        summary["tests"] - summary["failures"] - summary["errors"] - summary["skipped"]
    )
    rows = []
    for case in suite.findall("testcase"):
        rows.append(
            {
                "status": _status(case),
                "module": case.get("classname", ""),
                "test": case.get("name", ""),
                "seconds": round(float(case.get("time", 0.0)), 3),
                "detail": _detail(case),
            }
        )
    return summary, pd.DataFrame(rows)


_ICON = {"passed": "✅", "skipped": "⏭️", "failed": "❌", "error": "💥"}

st.title("🧪 Test Results")

if not _JUNIT_PATH.exists():
    st.warning(
        "No test report found yet. Generate one with:\n\n"
        "```\npython -m pytest -q --junitxml=reports/testrun/junit.xml\n```"
    )
    st.stop()

mtime = _JUNIT_PATH.stat().st_mtime
summary, df = _load(str(_JUNIT_PATH), mtime)

ran_at = "—"
if summary["timestamp"]:
    try:
        ran_at = datetime.fromisoformat(summary["timestamp"]).strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except ValueError:
        ran_at = summary["timestamp"]

ok = summary["failures"] == 0 and summary["errors"] == 0
if ok:
    st.success(
        f"**All {summary['passed']} tests passed**"
        + (f" · {summary['skipped']} skipped" if summary["skipped"] else "")
        + f" · {summary['time']:.1f}s"
    )
else:
    st.error(
        f"**{summary['failures']} failed, {summary['errors']} errored** of {summary['tests']} tests"
    )

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total", summary["tests"])
c2.metric("Passed", summary["passed"])
c3.metric("Skipped", summary["skipped"])
c4.metric("Failed", summary["failures"] + summary["errors"])
pass_rate = (summary["passed"] / summary["tests"] * 100) if summary["tests"] else 0.0
c5.metric("Pass rate", f"{pass_rate:.1f}%")

st.caption(f"Last run: {ran_at} · duration {summary['time']:.1f}s · source: `{_JUNIT_PATH.relative_to(_ROOT)}`")

# ── failures first (only when present) ────────────────────────────────────────
problems = df[df["status"].isin(["failed", "error"])]
if not problems.empty:
    st.subheader("Failures & errors")
    for _, r in problems.iterrows():
        with st.expander(f"{_ICON[r['status']]} {r['module']}::{r['test']}", expanded=True):
            st.code(r["detail"] or "(no detail captured)", language="text")

# ── full table with filters ──────────────────────────────────────────────────
st.subheader("All tests")
fc1, fc2 = st.columns([2, 3])
statuses = sorted(df["status"].unique())
chosen = fc1.multiselect("Status", statuses, default=statuses)
query = fc2.text_input("Filter by module / test name", "")

view = df[df["status"].isin(chosen)]
if query:
    q = query.lower()
    view = view[
        view["module"].str.lower().str.contains(q) | view["test"].str.lower().str.contains(q)
    ]

view = view.assign(status=view["status"].map(lambda s: f"{_ICON.get(s, '')} {s}"))
st.dataframe(
    view[["status", "module", "test", "seconds"]].sort_values("seconds", ascending=False),
    width="stretch",
    hide_index=True,
    height=480,
)

# ── slowest tests ─────────────────────────────────────────────────────────────
with st.expander("⏱️ 15 slowest tests"):
    slow = df.sort_values("seconds", ascending=False).head(15)[["module", "test", "seconds"]]
    st.dataframe(slow, width="stretch", hide_index=True)
