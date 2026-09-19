import streamlit as st
from theme import section_header

def render():
    st.markdown('<div class="canvas-section">', unsafe_allow_html=True)
    section_header("11. Further Work", "further-work")
    st.markdown("""
    - **Cross-Condition Protocol:** Train on one speed set, test on another. This is the highest-value next experiment because it directly probes the saturated-metric problem.
    - **GCN Over the Sensor Array:** Node features (per-channel window features) and adjacency (mounting geometry or cross-correlation) can target fault type and location, but this is premature on 288 clips from one machine.
    - **Wavelet-Encoder Deep Architecture:** Expected to gain on impulsive faults (e.g., loose, screwdrop). However, on this sample size, it would need heavy augmentation and likely still not beat XGBoost.
    - **Sequence Models Over Windows:** Using TCN or small GRU to replace hand-tuned smoothing rules with a learned decision function optimizing latency and false alarms jointly.
    - **Calibration and Drift:** Probability calibration (Isotonic/Platt) and drift monitoring on feature distributions to handle long-term plant changes.
    - **Remaining Useful Life (RUL):** This requires a new data collection programme focused on run-to-failure acquisition.
    - **Leverage Video:** Implement item detection and belt health using the existing video frames on the same edge device.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
