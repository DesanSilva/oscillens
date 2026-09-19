import streamlit as st
from theme import section_header

def render():
    section_header("10. Innovation, Impact, and Limitations", "limitations")
    st.markdown("""
    **Innovation:** The contribution is a reproducible, hash-sealed multimodal subset combined with an explicit temporal decision layer evaluated on false alarms vs. missed detections, rather than a new model architecture.
    
    **Limitations:**
    - **Classification, not prognostics:** No Remaining Useful Life (RUL) estimation is possible because there is no run-to-failure data.
    - **Limited Distribution:** Evaluated on one load level, one noise condition, and three speeds.
    - **Small Sample Size:** 288 clips total. Validation/test splits of 44 clips and 22 fault clips mean every per-class metric is coarse (one error heavily impacts macro-F1).
    - **Unused Modalities:** Video frames are carried in the dataset but unused by the model.
    - **Rate Compromise:** 25 kHz vibration did not meet the original 99% spectral-energy criterion.
    - **No Benchmark:** No latency or throughput benchmark has been executed.
    """)
