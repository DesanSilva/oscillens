import streamlit as st
from theme import section_header
import config as config

def render():
    section_header("8. Deployment in a Plant", "deployment")
    st.markdown("""
    **Inference Cost Estimate:** The computational bottleneck is not the boosted-tree ensemble (which evaluates ~1.5k float features in microseconds on a CPU), but rather the feature extraction stage (Welch, Hilbert, MFCC per 1-second window). *Note: The notebooks did not benchmark this; it is an estimate requiring empirical measurement.*
    
    **Runtime Sketch:**
    """)
    st.graphviz_chart("""
    digraph Runtime {
        rankdir=LR;
        node [shape=box, style=filled, fontname="Inter", fillcolor="#FFFFFF", color="#1F5C8B", penwidth=1.5];
        Sensor -> Windowing -> "Feature\nExtraction" -> Scaler -> Ensemble -> "Temporal\nRule" -> "MQTT\nTopic" -> "SCADA/CMMS";
    }
    """)
    
    st.markdown("""
    **Deployable Artifacts:**
    To ensure exactly the same transforms are applied in production as during training, the following artifacts must be deployed:
    - `vib_scaler.pkl`, `aud_scaler.pkl`, `fused_scaler.pkl`, `fused_window_scaler.pkl`
    - The fold models
    """)
