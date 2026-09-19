"""s09_scalability.py — Scalability and alternative architectures (design only)."""
import streamlit as st
from theme import section_header, figcaption


_CARD_CSS = (
    "background:#FFFFFF;padding:16px 18px;border:1px solid #DDE2E7;"
    "border-radius:4px;margin-bottom:12px;"
)

def _card(title: str, buys: str, costs: str, data_needed: str):
    st.markdown(
        f"""
        <div style="{_CARD_CSS}">
          <p style="font-size:14px;font-weight:600;margin:0 0 8px 0;color:#14181D;">{title}</p>
          <p style="font-size:13px;margin:0 0 4px 0;"><strong>What it buys:</strong> {buys}</p>
          <p style="font-size:13px;margin:0 0 4px 0;"><strong>What it costs:</strong> {costs}</p>
          <p style="font-size:13px;margin:0;"><strong>Data required:</strong> {data_needed}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render():
    st.markdown('<div class="canvas-section">', unsafe_allow_html=True)
    section_header("9. Scalability and Alternative Architectures", "scalability")



    st.markdown("### Candidate architecture directions")

    col_a, col_b = st.columns(2)
    with col_a:
        _card(
            "1. Late-fusion ensemble",
            "Graceful degradation when a sensor drops; per-modality confidence scores.",
            "Loss of cross-modal interaction terms captured by the current early-fusion ratio feature.",
            "No new data required.",
        )
        _card(
            "3. Graph neural network over sensor array",
            "Explicit fault-propagation modelling across mounting points; fault localisation, "
            "not just classification; transfer to machines with different channel counts.",
            "Needs physical sensor geometry (not recorded in the compact subset); "
            "requires data from multiple machines to learn a transferable edge function.",
            "Sensor coordinates + multi-machine dataset.",
        )
    with col_b:
        _card(
            "2. Wavelet / scattering-transform encoder",
            "Multi-resolution decomposition adapted to impulsive, non-stationary faults "
            "(loose, screwdrop); invariance to small time shifts.",
            "Higher compute per window; more hyperparameters; "
            "on 288 clips would likely underperform engineered features.",
            "More labelled clips before competing with XGBoost on engineered features.",
        )
        _card(
            "4. Self-supervised pretraining on unlabelled operation",
            "Uses the plant's own abundant unlabelled data via masked-autoencoding "
            "or contrastive objectives; reduces labelling requirement.",
            "Substantial unlabelled collection and compute; "
            "supervised head still requires labelled fault examples.",
            "Long unlabelled operational runs from the target machine.",
        )

    st.divider()

    st.markdown("### Horizontal scaling considerations")
    st.markdown("""
- **Per-machine model instances vs. conditioned model.** Adding machine ID and belt speed as features
  allows one model to serve a fleet; requires pooling labelled data across machines.
- **Shard-and-stream ingestion.** The ETL already demonstrates TAR-sharded ingestion with resumable
  state — this structure scales to streaming ingestion over MQTT or Kafka with minimal changes.
- **25 kHz vibration limit.** The current downsampling is a size-driven compromise. If bearing
  defect frequencies (often 5–20 kHz for small bearings) are diagnostically important, the
  storage rate would need revisiting.
""")

    st.markdown("</div>", unsafe_allow_html=True)
