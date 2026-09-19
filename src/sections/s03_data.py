"""s03_data.py - Data section: provenance, inventory, coverage, signal previews."""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from theme import section_header, figcaption
import data as data_mod
import config as config
import figures as figures


def render():
    canvas_open = '<div class="canvas-section">'
    st.markdown(canvas_open, unsafe_allow_html=True)
    section_header("3. Data: What It Is, What Was Kept, and Why", "data")

    manifest = data_mod.load_manifest()
    coverage = data_mod.load_coverage()

    # ── 3a. Provenance ─────────────────────────────────────────────────────
    st.markdown("### 3a. Provenance and policy")
    policy_hash = manifest.get("etl_policy_hash", "bf3c00d0e004d")
    st.markdown(f"""
The source is the **SSCC dataset** (yucongzh.github.io/SSCC-Dataset, CC BY 4.0),
comprising 6 724 synchronised source clips across fault types, velocities, loads, and noise conditions.

A deterministic sampling policy selected **288 clips** (15 operating conditions).
The policy is sealed with the SHA-256 hash `{policy_hash}`:
any rerun that would produce different sample IDs or media settings aborts rather than silently diverging.
""")

    st.markdown("**ETL stages (in order):**")
    etl_table = pd.DataFrame([
        {"Stage": "1. QC & jitter check",        "Verification": "Timestamp jitter ≤ 2 %; inferred rate within 2 % of 100 kHz"},
        {"Stage": "2. Resample vibration",        "Verification": "polyphase `resample_poly` 100 kHz → 25 kHz; size-driven compromise (stated)"},
        {"Stage": "3. Audio → FLAC",              "Verification": "Bit-exact against source WAV via `np.allclose` atol 1e-7"},
        {"Stage": "4. Video → WebP frames",       "Verification": "Frame saturation & mean-change QC per frame"},
        {"Stage": "5. Pack to TAR + parquet",     "Verification": "Shard + parquet committed as verified pair; state updated only after byte-size check"},
        {"Stage": "6. Policy hash assertion",     "Verification": "SHA-256 over sealed sampling policy asserted on every rerun"},
    ])
    st.table(etl_table)

    st.divider()

    # ── 3b. Component inventory ────────────────────────────────────────────
    st.markdown("### 3b. Component inventory")
    inventory = pd.DataFrame([
        {"Component": "Recorder audio", "Format": "FLAC, stereo",    "Rate / shape": "44.1 kHz, 5.0 s",             "Why": "Lossless; verified bit-exact against source WAV; ~2× smaller"},
        {"Component": "Vibration",      "Format": "float32 NPZ",     "Rate / shape": "4 ch × 25 kHz × 5 s",         "Why": "Polyphase anti-aliased from 100 kHz; size-driven compromise (stated)"},
        {"Component": "Video frames",   "Format": "WebP",            "Rate / shape": "5 frames/clip/device, 256 px", "Why": "Evidence that a camera can share the edge device; not used by the model"},
        {"Component": "Original MP4",   "Format": "H.264",           "Rate / shape": "10 files (5 clips × 2 devices)","Why": "Demonstration and loader tests"},
        {"Component": "Metadata",       "Format": "JSON + parquet",  "Rate / shape": "288 rows × 35 columns",        "Why": "Labels, roles, QC dicts, source references"},
    ])
    st.table(inventory)

    st.divider()

    # ── 3c. Coverage and balance ───────────────────────────────────────────
    st.markdown("### 3c. Coverage and balance")
    st.markdown(
        "One load level (`med`), one noise condition (`clean`), three speeds (60, 80, 100 rpm). "
        "The only distribution shift this subset can test is **speed**. "
        "Results do not generalise across loads or noise conditions not represented here."
    )

    col_cls, col_heat, col_role = st.columns(3)

    with col_cls:
        st.markdown("**Clips by class**")
        fig_cls = figures.create_class_dist_chart(coverage["counts_by_class"])
        st.plotly_chart(fig_cls, use_container_width=True)
        figcaption("144 normal / 144 fault; 36 each of dry, lean, loose, screwdrop.")

    with col_heat:
        st.markdown("**Fault × velocity**")
        vel_rows = [
            {"fault": k.rsplit("_", 1)[0], "velocity": k.rsplit("_", 1)[1], "count": v}
            for k, v in coverage["counts_by_velocity"].items()
            if k != "normal_60" and k != "normal_80" and k != "normal_100"
        ]
        vel_df = pd.DataFrame(vel_rows)
        if not vel_df.empty:
            fig_heat = px.density_heatmap(
                vel_df, x="velocity", y="fault", z="count",
                text_auto=True, color_continuous_scale="Blues",
            )
            fig_heat = figures.apply_plotly_layout(fig_heat, "", 260)
            fig_heat.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_heat, use_container_width=True)
            figcaption("Fault clips per class per velocity. Normal: 48 each at 60/80/100.")

    with col_role:
        st.markdown("**fc_role balance**")
        rb = coverage.get("role_balance", {})
        if rb:
            fig_role = figures.create_bar_chart(
                list(rb.keys()), [int(v) for v in rb.values()], ""
            )
            fig_role.update_layout(height=260)
            st.plotly_chart(fig_role, use_container_width=True)
            figcaption("fc_role governs fault-classification train/test assignment.")

    st.divider()

    # ── 3d. What the signals look like ────────────────────────────────────
    st.markdown("### 3d. What the signals look like")
    st.markdown(
        "Select a class to load the representative demo clip and inspect its signals. "
        "See Section 7 (Live Demonstration) for the full interactive panel."
    )

    videos_meta = data_mod.load_presentation_videos()
    class_to_sid = {v["class"]: sid for sid, v in videos_meta.items()}
    available_classes = list(class_to_sid.keys())

    selected_cls = st.selectbox("Class", available_classes, key="s03_class_selector")
    sid = class_to_sid[selected_cls]
    demo_data = data_mod.load_demo_data(sid)

    vib = demo_data.get("vib")
    if vib is not None:
        t_vib = np.linspace(0.0, 5.0, vib.shape[1])
        hints = {
            "normal":    "Stable low-amplitude oscillation; no impulsive events.",
            "dry":       "Elevated baseline amplitude; continuous high-frequency friction content.",
            "lean":      "Slight asymmetry in amplitude envelope; steady harmonic modulation.",
            "loose":     "Periodic impact spikes at regular intervals; consistent with chatter at belt velocity.",
            "screwdrop": "Isolated high-amplitude transient bursts; consistent with a foreign object striking the belt.",
        }
        fig_vib = figures.create_waveform_plot(vib, t_vib, title=f"Vibration - {selected_cls} ({sid})")
        st.plotly_chart(fig_vib, use_container_width=True)
        figcaption(hints.get(selected_cls, "") + f" Sample: {sid}.")
    else:
        st.info("Vibration data not available for this clip.")

    psd = demo_data.get("psd")
    if psd:
        fig_psd = figures.create_psd_plot(
            np.array(psd["f"]), np.array(psd["Pxx"]), bands=psd.get("bands")
        )
        st.plotly_chart(fig_psd, use_container_width=True)
        figcaption(
            f"Welch PSD - channel 1, {selected_cls} clip. "
            "Bands are physical (corrected for 25 kHz storage rate)."
        )


    st.markdown("</div>", unsafe_allow_html=True)
