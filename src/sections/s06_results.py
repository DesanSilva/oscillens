"""s06_results.py - Results and error analysis."""
import streamlit as st
import pandas as pd
import numpy as np
from theme import section_header, figcaption
import data as data_mod
import figures as figures


def render():
    section_header("6. Results and Error Analysis", "results")

    metrics = data_mod.load_metrics()

    # ── Metric tiles ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    def _n(key):
        v = metrics.get(key, {})
        return v.get("n", "?") if isinstance(v, dict) else "?"

    def _split(key):
        v = metrics.get(key, {})
        return v.get("split", "") if isinstance(v, dict) else ""

    def _val(key):
        v = metrics.get(key, {})
        return v.get("value", 0.0) if isinstance(v, dict) else float(v)

    with col1:
        st.metric("Fused ensemble F1", f"{_val('fusion_f1'):.3f}")
        st.caption(f"n={_n('fusion_f1')} {_split('fusion_f1')}")
    with col2:
        st.metric("Fused ensemble AUC", f"{_val('fusion_auc'):.3f}")
        st.caption(f"n={_n('fusion_auc')} {_split('fusion_auc')}")
    with col3:
        st.metric("Vib multiclass accuracy", f"{_val('vibration_mc_acc'):.3f}")
        st.caption(f"n={_n('vibration_mc_acc')} {_split('vibration_mc_acc')}")
    with col4:
        st.metric("Vib multiclass macro-F1", f"{_val('vibration_mc_f1'):.3f}")
        st.caption(f"n={_n('vibration_mc_f1')} {_split('vibration_mc_f1')}")

    vib_n = _val("vib_features") if isinstance(metrics.get("vib_features"), (int, float)) else metrics.get("vib_features", {}).get("value", "?")
    aud_n = _val("aud_features") if isinstance(metrics.get("aud_features"), (int, float)) else metrics.get("aud_features", {}).get("value", "?")
    fuse_n = metrics.get("fused_features", {}).get("value", "?") if isinstance(metrics.get("fused_features"), dict) else "?"
    sz    = metrics.get("split_sizes", {})
    st.caption(
        f"Features: {vib_n} vibration clip + {aud_n} audio clip + 1 cross-modal = {fuse_n} fused. "
        f"Split: train {sz.get('train','?')} / val {sz.get('val','?')} / test {sz.get('test','?')} clips."
    )



    st.divider()

    # ── Confusion matrices ────────────────────────────────────────────────
    cm_bin = data_mod.load_confusion_binary()
    cm_mc  = data_mod.load_confusion_multiclass()

    col_a, col_b = st.columns(2)
    with col_a:
        fig = figures.create_confusion_matrix(
            cm_bin,
            f"Binary fused ensemble - {cm_bin.get('split', 'validation clips')}",
        )
        st.pyplot(fig, use_container_width=True)
        figcaption("Threshold 0.10. Perfect separation on validation clips.")

    with col_b:
        fig = figures.create_confusion_matrix(
            cm_mc,
            f"Vibration-only multiclass (4 faults) - {cm_mc.get('split', 'validation clips, faults only')}",
        )
        st.pyplot(fig, use_container_width=True)
        figcaption(
            cm_mc.get("note", "")
            + " No audio or fused multiclass model exists - this matrix is vibration-only."
        )

    st.divider()

    # ── Per-class table ───────────────────────────────────────────────────
    st.markdown("### Per-class precision / recall / F1 - vibration multiclass (validation, faults only)")
    per_class = pd.DataFrame([
        {"class": "dry",       "n": 5, "precision": 1.00, "recall": 0.80, "F1": 0.89},
        {"class": "lean",      "n": 6, "precision": 0.86, "recall": 1.00, "F1": 0.92},
        {"class": "loose",     "n": 5, "precision": 1.00, "recall": 1.00, "F1": 1.00},
        {"class": "screwdrop", "n": 6, "precision": 1.00, "recall": 1.00, "F1": 1.00},
    ])
    st.table(per_class.style.format({"precision": "{:.2f}", "recall": "{:.2f}", "F1": "{:.2f}"}))
    figcaption(
        "Single error: one dry clip predicted lean. "
        "Dry and lean are both non-impulsive continuous faults - the expected failure mode. "
        "n values are small; one misclassification shifts macro-F1 by several points."
    )

    st.divider()

    # ── Temporal strategy table ───────────────────────────────────────────
    st.markdown("### Temporal decision-layer evaluation - validation windows")
    st.markdown(
        "These strategies stabilise the *current-state* decision and do **not** predict future faults. "
        "Source: `05_evaluation.ipynb`, validation windows."
    )

    tc_df = data_mod.load_temporal_comparison()
    st.table(
        tc_df.style.format({
            "recall": "{:.3f}", "f1": "{:.3f}", "specificity": "{:.3f}",
            "false_alarms": "{:d}", "missed": "{:d}",
        })
    )
    figcaption(
        "Persistence (K=3) removes all false alarms on validation windows "
        "but converts 1 miss into 5 - it trades detection latency for silence. "
        "Moving avg and EMA achieve zero missed detections at the cost of more false alarms."
    )

    # Grouped bar chart
    fig_bar = figures.create_temporal_bar_chart(tc_df)
    st.plotly_chart(fig_bar, use_container_width=True)
    figcaption("False alarms vs missed detections across strategies - validation windows.")

    st.divider()

