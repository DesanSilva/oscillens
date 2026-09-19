import streamlit as st
from theme import section_header

def render():
    section_header("4. Proposed Approach", "solution")
    st.markdown("""
    To effectively detect faults while controlling false alarms, this pipeline introduces three core components:
    
    1. **Multimodal Evidence:** We fuse vibration data (providing structural depth) with audio data (capturing high-frequency surface events).
    2. **Feature-Level Fusion:** We employ early, feature-level concatenation with an XGBoost tree ensemble. Given the small dataset size (288 clips), this approach is more robust than deep learned fusion.
    3. **Temporal Decision Layer:** Instead of raw per-window alarms, we introduce an explicit temporal decision layer (e.g. Moving Average, N-of-M, Persistence) between the classifier and the alarm.
    
    **Goal:** Maximise fault sensitivity while explicitly controlling and trading off false alarms and detection latency.
    """)
