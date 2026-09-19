"""
app.py - SSCC multimodal anomaly detection presentation.
Single scrollable page; sticky sidebar navigation; sections render in order.
Run from repo root: streamlit run src/app.py
"""
import sys
from pathlib import Path

# Ensure src/ is on the path when running from repo root
SRC = Path(__file__).parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
import theme as _theme
from sections import (
    s01_context, s02_gap, s03_data, s04_solution,
    s05_methodology, s06_results, s07_demo,
    s08_deployment, s09_scalability, s10_limitations,
    s11_further_work, s12_conclusion,
)
import data as data_mod

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SSCC Multimodal Anomaly Detection",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

_theme.apply_theme()

# ── Sidebar navigation ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### SSCC Fault Detection")
    st.caption("Multimodal anomaly detection for conveyor systems")
    st.markdown("---")


    nav_items = [
        ("context",     "1. Problem and context"),
        ("gap",         "2. Current practice"),
        ("data",        "3. Data"),
        ("solution",    "4. Proposed approach"),
        ("methodology", "5. Methodology"),
        ("results",     "6. Results"),
        ("demo",        "7. Live demonstration"),
        ("deployment",  "8. Deployment"),
        ("scalability", "9. Scalability"),
        ("limitations", "10. Limitations"),
        ("further-work","11. Further work"),
        ("conclusion",  "12. Conclusion"),
    ]
    st.markdown("### Sections")
    for slug, label in nav_items:
        st.markdown(f"[{label}](#{slug})", unsafe_allow_html=True)

# ── Page header ────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="margin-bottom:0.1rem;">SSCC Multimodal Anomaly Detection</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-size:17px;color:#4A5560;margin-top:0.2rem;max-width:70ch;">'
    "A reproducible multimodal fault detection pipeline for conveyor systems - "
    "vibration + audio feature fusion with an explicit temporal decision layer."
    "</p>",
    unsafe_allow_html=True,
)
st.markdown('<hr style="margin:1rem 0 2rem 0;">', unsafe_allow_html=True)

# ── Sections ───────────────────────────────────────────────────────────────
s01_context.render()
s02_gap.render()
s03_data.render()
s04_solution.render()
s05_methodology.render()
s06_results.render()
s07_demo.render()
s08_deployment.render()
s09_scalability.render()
s10_limitations.render()
s11_further_work.render()
s12_conclusion.render()

