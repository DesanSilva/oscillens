"""s05_methodology.py — Methodology and architecture."""
import streamlit as st
import pandas as pd
from theme import section_header, figcaption
import diagrams as diagrams
import data as data_mod


def render():
    st.markdown('<div class="canvas-section">', unsafe_allow_html=True)
    section_header("5. Methodology and Architecture", "methodology")

    # ── Pipeline diagram ──────────────────────────────────────────────────
    st.markdown("### Pipeline architecture")
    st.graphviz_chart(diagrams.PIPELINE_DOT)
    figcaption(
        "Left-to-right pipeline. Dashed edges: video branch — carried in the dataset, "
        "not consumed by the current model. Decision cluster uses alarm colour only for state, "
        "not as UI chrome."
    )


    st.divider()

    # ── Fusion topology diagram ───────────────────────────────────────────
    st.markdown("### Fusion topology")
    st.graphviz_chart(diagrams.FUSION_DOT)
    figcaption(
        "Feature-level (early) concatenation: scaled vibration clip features + scaled audio clip "
        "features + one cross-modal ratio term → 1 486 fused features → 5-fold XGBoost ensemble. "
        "Late-fusion (probability combination) is discussed as a candidate in Section 9 but is not implemented."
    )

    st.divider()

    # ── Feature inventory ─────────────────────────────────────────────────
    st.markdown("### Feature inventory")
    schema = data_mod.load_feature_schema()
    vib_n = schema["vibration"]["count"]
    aud_n = schema["audio"]["count"]

    feat_table = pd.DataFrame([
        {
            "Modality": "Vibration (window)",
            "Per-window features": 156,
            "Clip-aggregated": vib_n,
            "Rationale": "Time + spectral + envelope + cross-channel; aggregated as mean / std / min / max / median / p90",
        },
        {
            "Modality": "Audio (window)",
            "Per-window features": 106,
            "Clip-aggregated": aud_n,
            "Rationale": "RMS / ZCR / spectral centroid / bandwidth / rolloff / entropy + 5 bands + 13 MFCC × {mean,std,min,max} + 13 ΔMFCC × {mean,std}",
        },
        {
            "Modality": "Cross-modal (fused)",
            "Per-window features": "—",
            "Clip-aggregated": 1,
            "Rationale": "cross_aud_vib_rms_ratio — captures relative energy between modalities",
        },
    ])
    st.table(feat_table)
    figcaption(
        f"Total fused clip features: {vib_n} + {aud_n} + 1 = {vib_n + aud_n + 1}. "
        "Feature QC dropped 3 zero-variance vibration columns before training."
    )

    st.divider()

    # ── Training protocol ─────────────────────────────────────────────────
    st.markdown("### Training protocol")
    metrics = data_mod.load_metrics()
    sz = metrics.get("split_sizes", {})

    proto_table = pd.DataFrame([
        {"Item": "Split",              "Detail": f"Train {sz.get('train','200')} / val {sz.get('val','44')} / test {sz.get('test','44')} clips — clip-level stratified on 5-class label, seed 42"},
        {"Item": "Leakage control",    "Detail": "All windows of a clip stay in one split; no window appears in two splits. Scaler fit on train only."},
        {"Item": "Scaler",             "Detail": "StandardScaler fit separately for vibration, audio, and fused matrices"},
        {"Item": "Binary XGBoost",     "Detail": "n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=n_neg/n_pos, early_stopping_rounds=30, seed=42"},
        {"Item": "Multiclass XGBoost", "Detail": "Same hyperparameters; objective=multi:softprob, eval_metric=mlogloss"},
        {"Item": "Fused ensemble",     "Detail": "5-fold StratifiedKFold on fused training features; mean probability across folds. Not 5-fold CV with early stopping — a fold ensemble."},
        {"Item": "Window model",       "Detail": "Separate XGBClassifier(n_estimators=100, max_depth=4) on window-level fused features for temporal evaluation (notebook 05)"},
        {"Item": "Best threshold",     "Detail": "0.10 for fused detector (from validation sweep); 0.32 for audio binary"},
    ])
    st.table(proto_table)

    st.markdown("</div>", unsafe_allow_html=True)
