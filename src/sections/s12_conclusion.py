import streamlit as st
from theme import section_header

def render():
    section_header("12. Conclusion", "conclusion")
    st.markdown("""
    This application demonstrates a functional multimodal pipeline for mechanical anomaly detection built on a reproducible, offline ETL process. The evidence supports that feature-level fusion of vibration and audio combined with an explicit temporal decision layer effectively controls false alarms. However, this implementation is currently constrained by a small sample size from stable environments, and true generalisation requires cross-condition evaluation and run-to-failure testing.
    """)
