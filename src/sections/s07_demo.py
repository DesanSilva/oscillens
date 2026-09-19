"""
s07_demo.py - Live demonstration section.
Full-width dashboard: clip selector, audio player, synced signal panel,
prediction lane with interactive decision controls, feature inspector.
"""
import streamlit as st
import numpy as np
import pandas as pd

import data as data_mod
import config as config
import figures as figures
from theme import section_header, canvas_band_open, canvas_band_close, status_badge, figcaption

# ── Smoothing / decision functions (exact semantics of notebook 05) ────────

def _smooth_mavg(probs: np.ndarray, k: int) -> np.ndarray:
    out = np.empty_like(probs)
    for i in range(len(probs)):
        out[i] = probs[max(0, i - k + 1): i + 1].mean()
    return out


def _smooth_ema(probs: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(probs)
    if len(probs) == 0:
        return out
    out[0] = probs[0]  # initialise at first observation
    for i in range(1, len(probs)):
        out[i] = alpha * probs[i] + (1.0 - alpha) * out[i - 1]
    return out


def _decide_nom(probs: np.ndarray, threshold: float, N: int, M: int) -> np.ndarray:
    raw = (probs >= threshold).astype(int)
    out = np.zeros_like(raw)
    for i in range(len(raw)):
        window = raw[max(0, i - M + 1): i + 1]
        out[i] = int(window.sum() >= N)
    return out


def _decide_persist(probs: np.ndarray, threshold: float, K: int) -> np.ndarray:
    raw = (probs >= threshold).astype(int)
    out = np.zeros_like(raw)
    consec = 0
    for i, d in enumerate(raw):
        consec = consec + 1 if d else 0
        out[i] = int(consec >= K)
    return out


# ── Main render ────────────────────────────────────────────────────────────

def render():
    section_header("7. Live Demonstration", "demo")

    videos_meta = data_mod.load_presentation_videos()
    if not videos_meta:
        st.error("presentation_videos.json not found. Run prepare_assets.py first.")
        return

    sample_ids = list(videos_meta.keys())

    # Friendly label helper
    def _label(sid: str) -> str:
        cls  = videos_meta[sid]["class"]
        vel  = sid.split("vel")[1].split("_")[0] if "vel" in sid else "?"
        role = "fault" if cls != "normal" else "normal"
        return f"{cls.title()} - vel {vel} rpm  [{role}]"

    # ── Clip selector ──────────────────────────────────────────────────────
    st.markdown("**Select a demo clip to explore its signals and model output.**")
    selected_id = st.selectbox(
        "Clip", sample_ids, format_func=_label,
        help="Six representative clips: one per class (dry, lean, loose, screwdrop) plus two normal clips.",
    )
    cls_name      = videos_meta[selected_id]["class"]
    is_fault_truth = cls_name != "normal"

    demo_data = data_mod.load_demo_data(selected_id)
    windows   = demo_data.get("windows", pd.DataFrame())

    # ── Controls (above signal panel so they update before render) ─────────
    with st.expander("Decision strategy controls", expanded=True):
        col_s, col_th = st.columns([2, 1])
        strategy = col_s.radio(
            "Smoothing / decision strategy",
            ["Raw", "Moving avg", "EMA", "N-of-M", "Persistence"],
            horizontal=True,
        )
        th = col_th.slider("Threshold (θ)", 0.0, 1.0, 0.10, 0.01)

        k, alpha, N, M, K = 3, 0.4, 3, 5, 3
        param_cols = st.columns(4)
        if strategy == "Moving avg":
            k = param_cols[0].slider("Window size k", 1, 10, 3)
        elif strategy == "EMA":
            alpha = param_cols[0].slider("Alpha (α)", 0.05, 1.0, 0.4, 0.05)
        elif strategy == "N-of-M":
            N = param_cols[0].slider("N", 1, 10, 3)
            M = param_cols[1].slider("M", N, 20, 5)
        elif strategy == "Persistence":
            K = param_cols[0].slider("Consecutive windows K", 1, 10, 3)

    with st.expander("Smoothing equations & definitions", expanded=False):
        st.markdown(r"""
        **1. Moving Average**
        Computes the unweighted mean over the last $k$ windows.

        $$ P_{smooth}[t] = \frac{1}{k} \sum_{i=0}^{k-1} P_{raw}[t-i] $$
        
        **2. Exponential Moving Average (EMA)**
        Applies exponentially decreasing weights over time. $\alpha$ is the smoothing factor ($0 < \alpha \le 1$).

        $$ P_{smooth}[t] = \alpha P_{raw}[t] + (1 - \alpha) P_{smooth}[t-1] $$
        
        **3. N-of-M**
        Triggers an alarm if at least $N$ out of the last $M$ windows exceed the threshold $\theta$.

        $$ \text{Alarm}[t] = \left( \sum_{i=0}^{M-1} \mathbb{I}(P_{smooth}[t-i] \ge \theta) \right) \ge N $$
        
        **4. Persistence**
        A strict case of N-of-M where $N = M = K$. Triggers only if $K$ consecutive windows exceed $\theta$.

        $$ \text{Alarm}[t] = \prod_{i=0}^{K-1} \mathbb{I}(P_{smooth}[t-i] \ge \theta) $$
        """)

    # ── Render Logic ───────────────────────────────────────────────────────
    def render_ui(windows_df, current_t=None):
        # Compute smoothed probabilities & decisions on the sliced windows
        p_fault_raw = windows_df["p_fault"].values if (not windows_df.empty and "p_fault" in windows_df.columns) else np.zeros(0)

        if strategy == "Moving avg":
            p_smooth = _smooth_mavg(p_fault_raw, k)
            strategy_label = f"Moving avg k={k}"
        elif strategy == "EMA":
            p_smooth = _smooth_ema(p_fault_raw, alpha)
            strategy_label = f"EMA α={alpha:.2f}"
        else:
            p_smooth = p_fault_raw.copy()
            strategy_label = strategy

        if strategy == "N-of-M":
            decisions = _decide_nom(p_smooth, th, N, M)
            strategy_label = f"N-of-M {N}/{M}"
        elif strategy == "Persistence":
            decisions = _decide_persist(p_smooth, th, K)
            strategy_label = f"Persistence K={K}"
        else:
            decisions = (p_smooth >= th).astype(int)

        alarms_raised   = int(decisions.max()) if len(decisions) > 0 else 0
        total_alarm_wins = int(decisions.sum())
        first_idx        = int(np.argmax(decisions)) if alarms_raised else -1
        latency_s: float | None = None
        if first_idx >= 0 and not windows_df.empty and "win_start_s" in windows_df.columns:
            latency_s = float(windows_df["win_start_s"].iloc[first_idx])

        is_fault_truth_current = bool(windows_df["is_fault"].iloc[-1]) if not windows_df.empty and "is_fault" in windows_df.columns else is_fault_truth
        current_fault_type = str(windows_df["fault"].iloc[-1]).title() if not windows_df.empty and "fault" in windows_df.columns else cls_name.title()

        if is_fault_truth_current:
            missed_clip    = 1 if alarms_raised == 0 else 0
            fa_clip        = 0
        else:
            missed_clip    = 0
            fa_clip        = 1 if alarms_raised > 0 else 0

        current_decision = int(decisions[-1]) if len(decisions) > 0 else 0
        alarm_badge  = status_badge("alarm" if current_decision else "ok")
        if current_decision:
            alarm_text = f"Alarm ({current_fault_type})" if is_fault_truth_current else "Alarm (False Pos)"
        else:
            alarm_text = "Ok"
            
        truth_badge  = status_badge("alarm" if is_fault_truth_current else "ok")
        truth_text   = current_fault_type

        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:20px;padding:12px 0;flex-wrap:wrap;">
              <div>
                <span style="font-size:14px;font-weight:500;color:#4A5560;display:block;margin-bottom:3px;">Model decision</span>
                {alarm_badge}&nbsp;<span style="font-size:16px;font-weight:600;">{alarm_text}</span>
              </div>
              <div style="color:#DDE2E7;font-size:24px;">|</div>
              <div>
                <span style="font-size:14px;font-weight:500;color:#4A5560;display:block;margin-bottom:3px;">Ground truth</span>
                {truth_badge}&nbsp;<span style="font-size:16px;font-weight:600;">{truth_text}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Windows in alarm", total_alarm_wins, help="Windows where the decision rule fired")
        mc2.metric("Detection latency", f"{latency_s:.1f} s" if latency_s is not None else "N/A", help="Time of first alarm window from clip start")
        mc3.metric("False alarm (this clip)", fa_clip, delta=None, help="1 if normal clip triggered an alarm")
        mc4.metric("Missed fault (this clip)", missed_clip, delta=None, help="1 if fault clip raised no alarm")

        st.markdown("")

        # Combined signal panel
        vib = demo_data.get("vib")
        if vib is not None:
            total_duration = float(windows["win_end_s"].max()) if not windows.empty else 5.0
            if current_t is not None:
                frac = current_t / total_duration
                max_idx = min(int(vib.shape[1] * frac) + 1, vib.shape[1])
                vib_show = vib[:, :max_idx]
                t_vib = np.linspace(0.0, current_t, vib_show.shape[1])
            else:
                vib_show = vib
                t_vib = np.linspace(0.0, total_duration, vib.shape[1])
                
            fig_combined = figures.create_combined_signal_panel(
                vib_data=vib_show,
                t_vib=t_vib,
                windows_df=windows_df,
                p_smooth=p_smooth,
                threshold=th,
                strategy_label=strategy_label,
                decisions=decisions,
            )
            if current_t is not None:
                fig_combined.update_xaxes(range=[0, total_duration])

            st.plotly_chart(fig_combined, use_container_width=True)
            pfault_note = demo_data.get("p_fault_source", "unknown")
            figcaption(
                f"Top: 4-channel vibration. Bottom: window-level fault probability - strategy: {strategy_label}, θ={th:.2f}. "
                f"Shaded regions are alarm spans. p_fault source: {pfault_note}."
            )
        else:
            st.info("Vibration signal not available in this bundle.")

        spec = demo_data.get("spectrogram")
        if spec is not None:
            if current_t is not None:
                total_duration = float(windows["win_end_s"].max()) if not windows.empty else 5.0
                mask = spec["t"] <= current_t
                t_spec = spec["t"][mask]
                Sxx = spec["Sxx_db"][:, mask]
            else:
                t_spec = spec["t"]
                Sxx = spec["Sxx_db"]
                
            fig_spec = figures.create_spectrogram_plot(spec["f"], t_spec, Sxx)
            if current_t is not None:
                fig_spec.update_xaxes(range=[0, total_duration])
            st.plotly_chart(fig_spec, use_container_width=True)
            figcaption(
                f"Audio log spectrogram. Parameters: nperseg=1024, noverlap=512; lower quarter of frequency axis shown."
            )

    # ── Render Mode Dispatch ───────────────────────────────────────────────
    if selected_id == "continuous_stream":
        audio_path = demo_data.get("audio_path")
        if audio_path:
            st.audio(audio_path, format="audio/flac")
            
        if st.button("Start Live Stream"):
            import time
            placeholder = st.empty()
            total_duration = float(windows["win_end_s"].max()) if not windows.empty else 30.0
            for t_step in np.arange(1.0, total_duration + 1.0, 1.0):
                with placeholder.container():
                    w_mask = windows["win_start_s"] <= t_step
                    render_ui(windows[w_mask], current_t=t_step)
                time.sleep(1.0)
        else:
            render_ui(windows)
            
    else:
        audio_path = demo_data.get("audio_path")
        if audio_path:
            st.markdown("**Audio recording**")
            st.audio(audio_path, format="audio/flac")
            figcaption(f"Audio clip from {selected_id}. Press play to listen; the signal panel below shows the same clip.")
        else:
            st.info("Audio file not available in this bundle.")

        st.markdown("---")
        render_ui(windows)

    # ── PSD + Envelope ────────────────────────────────────────────────────
    psd  = demo_data.get("psd")
    env  = demo_data.get("envelope")

    col_psd, col_env = st.columns(2)
    if psd:
        with col_psd:
            st.markdown("**Welch PSD - vibration channel 1**")
            fig_psd = figures.create_psd_plot(
                np.array(psd["f"]), np.array(psd["Pxx"]),
                bands=psd.get("bands"),
            )
            st.plotly_chart(fig_psd, use_container_width=True)
            figcaption("Physical bands (corrected for 25 kHz): low 0–1250 Hz, mid 1250–5000 Hz, high 5000–12500 Hz.")

    if env:
        with col_env:
            st.markdown("**Envelope - channel 1 (bandpass 1250–10 000 Hz)**")
            t_env = np.linspace(0.0, 5.0, len(env["env_trace"]))
            fig_env = figures.create_envelope_plot(
                np.array(env["env_trace"]), t_env,
                np.array(env["f"]), np.array(env["Pxx"]),
            )
            st.plotly_chart(fig_env, use_container_width=True)
            figcaption("Envelope extracted via Butterworth bandpass → Hilbert transform → |analytic signal|.")

    # ── Video panel ───────────────────────────────────────────────────────
    has_ios     = videos_meta[selected_id].get("ios", False)
    has_android = videos_meta[selected_id].get("android", False)

    if has_ios or has_android:
        st.markdown("---")
        st.markdown("**Video recordings** - unsynchronised original recordings; model consumes no video.")
        vc1, vc2 = st.columns(2)
        vid_base = config.ASSETS_DIR / "demo" / selected_id
        if has_ios and (vid_base / "video_ios.mp4").exists():
            vc1.caption("iOS camera")
            vc1.video(str(vid_base / "video_ios.mp4"))
        elif has_ios:
            vc1.info("iOS video not in bundle.")
        if has_android and (vid_base / "video_android.mp4").exists():
            vc2.caption("Android camera")
            vc2.video(str(vid_base / "video_android.mp4"))
        elif has_android:
            vc2.info("Android video not in bundle.")

    # ── Feature inspector ─────────────────────────────────────────────────
    with st.expander("Window feature inspector"):
        if not windows.empty:
            disp_cols = [c for c in
                         ["window_idx", "win_start_s", "win_end_s", "is_fault",
                          "vib_ch1_rms", "vib_ch1_kurtosis", "vib_ch1_crest",
                          "aud_rms", "aud_spec_centroid", "p_fault"]
                         if c in windows.columns]
            st.dataframe(
                windows[disp_cols].style.format(
                    {c: "{:.4f}" for c in disp_cols if c not in
                     ["window_idx", "is_fault", "win_start_s", "win_end_s"]}
                ),
                use_container_width=True,
            )
            figcaption(
                "Per-window features shown alongside p_fault for the selected clip. "
                "vib_ch1_rms: channel-1 vibration RMS energy; "
                "vib_ch1_kurtosis: impulsiveness indicator; "
                "vib_ch1_crest: peak-to-RMS ratio; "
                "aud_rms: audio energy; aud_spec_centroid: spectral centre of mass (Hz)."
            )
        else:
            st.info("Window data not available in this bundle.")

    # ── WebP frame gallery ────────────────────────────────────────────────
    frames = demo_data.get("frames", [])
    if frames:
        st.markdown("---")
        st.markdown("**Static frames from the clip** (iOS and Android, 1 fps, 5 frames per device).")
        cols = st.columns(min(len(frames), 5))
        for i, fp in enumerate(frames[:10]):
            cols[i % len(cols)].image(fp, use_container_width=True)
        figcaption("WebP frames - carried in the dataset as visual evidence; not consumed by the current model.")
