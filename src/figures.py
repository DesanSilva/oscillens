"""
figures.py — Plotly and Matplotlib figure builders.
All figures use the project palette; no purple/violet outside categorical plot series.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import config as config

# ── Plotly layout defaults ────────────────────────────────────────────────

def _base_layout(title: str = "", height: int = 300) -> dict:
    return dict(
        title=dict(text=title, font=dict(family="Inter, sans-serif", size=14, color=config.PALETTE["ink"])),
        height=height,
        paper_bgcolor=config.PALETTE["surface"],
        plot_bgcolor=config.PALETTE["canvas"],
        font=dict(family="Inter, sans-serif", color=config.PALETTE["ink"], size=12),
        margin=dict(l=48, r=16, t=40 if title else 10, b=44),
        xaxis=dict(
            showgrid=True, gridcolor=config.PALETTE["hairline"],
            zeroline=False, linecolor=config.PALETTE["hairline"],
        ),
        yaxis=dict(
            showgrid=True, gridcolor=config.PALETTE["hairline"],
            zeroline=False, linecolor=config.PALETTE["hairline"],
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.35,
            xanchor="center", x=0.5,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        hovermode="x unified",
    )


def apply_plotly_layout(fig: go.Figure, title: str = "", height: int = 300) -> go.Figure:
    fig.update_layout(**_base_layout(title, height))
    return fig


# ── Vibration waveform ────────────────────────────────────────────────────

CH_COLORS = ["#1F5C8B", "#4A5560", "#2F7D4F", "#C2410C"]

def create_waveform_plot(vib_data: np.ndarray, t_axis: np.ndarray,
                         title: str = "Vibration — 4 channels") -> go.Figure:
    fig = go.Figure()
    for i in range(vib_data.shape[0]):
        fig.add_trace(go.Scatter(
            x=t_axis, y=vib_data[i],
            mode="lines", name=f"Ch {i+1}",
            line=dict(color=CH_COLORS[i % len(CH_COLORS)], width=1.0),
        ))
    apply_plotly_layout(fig, title, 240)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude")
    return fig


# ── Audio waveform ────────────────────────────────────────────────────────

def create_audio_waveform(y: np.ndarray, t_axis: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_axis, y=y, mode="lines", name="Audio",
        line=dict(color=config.PALETTE["signal"], width=0.8),
        showlegend=False,
    ))
    apply_plotly_layout(fig, "", 140)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Amplitude", tickformat=".3f")
    return fig


# ── Spectrogram ───────────────────────────────────────────────────────────

def create_spectrogram_plot(f: np.ndarray, t: np.ndarray,
                             Sxx_db: np.ndarray) -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db, x=t, y=f,
        colorscale="Blues",
        reversescale=False,
        zmin=float(np.percentile(Sxx_db, 5)),
        zmax=float(np.percentile(Sxx_db, 95)),
        showscale=True,
        colorbar=dict(thickness=12, title=dict(text="dB", side="right"),
                      tickfont=dict(size=10)),
    ))
    apply_plotly_layout(fig, "", 220)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="Frequency (Hz)")
    return fig


# ── PSD plot ──────────────────────────────────────────────────────────────

def create_psd_plot(f: np.ndarray, Pxx: np.ndarray,
                    bands: dict | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f, y=10 * np.log10(np.maximum(Pxx, 1e-12)),
        mode="lines", name="PSD (dB/Hz)",
        line=dict(color=config.PALETTE["signal"], width=1.5),
    ))
    if bands:
        band_colors = ["#1F5C8B", "#2F7D4F", "#C2410C"]
        for i, (bname, (b0, b1)) in enumerate(bands.items()):
            mid = (b0 + b1) / 2
            fig.add_vrect(
                x0=b0, x1=b1,
                fillcolor=band_colors[i % len(band_colors)],
                opacity=0.07, line_width=0,
                annotation_text=bname, annotation_position="top",
                annotation_font_size=10,
            )
    apply_plotly_layout(fig, "Welch PSD — channel 1 (physical 25 kHz)", 280)
    fig.update_xaxes(title_text="Frequency (Hz)")
    fig.update_yaxes(title_text="Power (dB/Hz)")
    return fig


# ── Envelope plot ─────────────────────────────────────────────────────────

def create_envelope_plot(env_trace: np.ndarray, t_axis: np.ndarray,
                          f_env: np.ndarray, Pxx_env: np.ndarray) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Envelope trace", "Envelope spectrum"])
    fig.add_trace(go.Scatter(
        x=t_axis, y=env_trace, mode="lines", name="Envelope",
        line=dict(color=config.PALETTE["alarm"], width=1.2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f_env, y=10 * np.log10(np.maximum(Pxx_env, 1e-12)),
        mode="lines", name="Env PSD",
        line=dict(color=config.PALETTE["signal"], width=1.2),
    ), row=1, col=2)
    apply_plotly_layout(fig, "", 220)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="dB", row=1, col=2)
    fig.update_layout(showlegend=False)
    return fig


# ── Combined signal panel (demo — shared x-axis) ──────────────────────────

def create_combined_signal_panel(
    vib_data: np.ndarray,
    t_vib: np.ndarray,
    windows_df: pd.DataFrame,
    p_smooth: np.ndarray,
    threshold: float,
    strategy_label: str,
    decisions: np.ndarray,
) -> go.Figure:
    """
    Subplot stack sharing x-axis:
      row 1  — 4-ch vibration
      row 2  — prediction lane (p_fault step trace + threshold + alarm shading)
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.04,
        subplot_titles=["Vibration — 4 channels (decimated, 5 s)", f"Fault probability — {strategy_label}"],
    )

    # Row 1: vibration
    for i in range(vib_data.shape[0]):
        fig.add_trace(go.Scatter(
            x=t_vib, y=vib_data[i],
            mode="lines", name=f"Ch {i+1}",
            line=dict(color=CH_COLORS[i % len(CH_COLORS)], width=0.9),
        ), row=1, col=1)

    # Row 2: prediction lane
    if not windows_df.empty and "win_start_s" in windows_df.columns:
        t_win = windows_df["win_start_s"].values

        # Alarm shading spans
        in_alarm = False
        alarm_start = None
        for j, dec in enumerate(decisions):
            t_j = float(t_win[j]) if j < len(t_win) else None
            if dec and not in_alarm:
                in_alarm = True
                alarm_start = t_j
            elif not dec and in_alarm:
                fig.add_vrect(
                    x0=alarm_start, x1=t_j,
                    fillcolor=config.PALETTE["alarm"], opacity=0.12,
                    line_width=0, row=2, col=1,
                )
                in_alarm = False
        if in_alarm and alarm_start is not None and len(t_win) > 0:
            fig.add_vrect(
                x0=alarm_start, x1=float(t_win[-1]) + 0.5,
                fillcolor=config.PALETTE["alarm"], opacity=0.12,
                line_width=0, row=2, col=1,
            )

        # Raw p_fault (lighter)
        if "p_fault" in windows_df.columns:
            fig.add_trace(go.Scatter(
                x=t_win, y=windows_df["p_fault"].values,
                mode="lines", name="p_fault (raw)",
                line=dict(color="#9BBDD6", width=1.0, dash="dot", shape="hv"),
                opacity=0.7,
            ), row=2, col=1)

        # Smoothed
        fig.add_trace(go.Scatter(
            x=t_win, y=p_smooth,
            mode="lines+markers", name="p_fault (smoothed)",
            line=dict(color=config.PALETTE["signal"], width=2.0, shape="hv"),
            marker=dict(size=5, color=config.PALETTE["signal"]),
        ), row=2, col=1)

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash", line_color=config.PALETTE["alarm"], line_width=1.5,
            annotation_text=f"θ = {threshold:.2f}",
            annotation_position="top right",
            annotation_font_size=10,
            row=2, col=1,
        )

    fig.update_layout(
        height=480,
        paper_bgcolor=config.PALETTE["surface"],
        plot_bgcolor=config.PALETTE["canvas"],
        font=dict(family="Inter, sans-serif", color=config.PALETTE["ink"], size=11),
        margin=dict(l=48, r=16, t=50, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=10)),
        hovermode="x unified",
    )
    for ax in ["xaxis", "xaxis2"]:
        fig.update_layout(**{ax: dict(
            showgrid=True, gridcolor=config.PALETTE["hairline"],
            zeroline=False, linecolor=config.PALETTE["hairline"],
        )})
    for ax in ["yaxis", "yaxis2"]:
        fig.update_layout(**{ax: dict(
            showgrid=True, gridcolor=config.PALETTE["hairline"],
            zeroline=False, linecolor=config.PALETTE["hairline"],
        )})

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="P(fault)", range=[-0.05, 1.08], row=2, col=1)

    return fig


# ── Confusion matrix (matplotlib) ─────────────────────────────────────────

def create_confusion_matrix(cm_data: dict, title: str) -> plt.Figure:
    mat = np.array(cm_data["matrix"])
    labels = cm_data["labels"]

    fig, ax = plt.subplots(figsize=(max(3.5, len(labels) * 1.1), max(3.0, len(labels) * 0.9)))
    sns.heatmap(
        mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, linecolor=config.PALETTE["hairline"],
        cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=11, color=config.PALETTE["ink"])
    ax.set_ylabel("True", fontsize=11, color=config.PALETTE["ink"])
    ax.set_title(title, fontsize=12, fontweight="600", color=config.PALETTE["ink"], pad=10)
    ax.tick_params(axis="both", labelsize=10)
    fig.patch.set_facecolor(config.PALETTE["surface"])
    ax.set_facecolor(config.PALETTE["canvas"])
    plt.tight_layout()
    return fig


# ── Bar chart ─────────────────────────────────────────────────────────────

def create_bar_chart(x: list, y: list, title: str = "",
                     color: str | None = None) -> go.Figure:
    c = color or config.PALETTE["signal"]
    fig = go.Figure([go.Bar(
        x=x, y=y,
        marker_color=c,
        marker_line_color=config.PALETTE["hairline"],
        marker_line_width=0.8,
    )])
    apply_plotly_layout(fig, title, 300)
    return fig


# ── Temporal strategy grouped bar chart ───────────────────────────────────

def create_temporal_bar_chart(temporal_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="False alarms",
        x=temporal_df["strategy"],
        y=temporal_df["false_alarms"],
        marker_color=config.PALETTE["alarm"],
        marker_line_color=config.PALETTE["hairline"],
        marker_line_width=0.8,
    ))
    fig.add_trace(go.Bar(
        name="Missed detections",
        x=temporal_df["strategy"],
        y=temporal_df["missed"],
        marker_color="#6B3A24",
        marker_line_color=config.PALETTE["hairline"],
        marker_line_width=0.8,
    ))
    apply_plotly_layout(fig, "False alarms vs missed detections — validation windows", 300)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Strategy")
    fig.update_yaxes(title_text="Count")
    return fig


# ── Class distribution bar chart ──────────────────────────────────────────

def create_class_dist_chart(counts: dict) -> go.Figure:
    labels = list(counts.keys())
    values = [int(v) for v in counts.values()]
    colors = [config.CLASS_COLORS.get(l, config.PALETTE["signal"]) for l in labels]
    fig = go.Figure([go.Bar(
        x=labels, y=values,
        marker_color=colors,
        marker_line_color=config.PALETTE["hairline"],
        marker_line_width=0.8,
        text=values,
        textposition="outside",
    )])
    apply_plotly_layout(fig, "Clip count by class (binary: 144 normal / 144 fault)", 280)
    fig.update_xaxes(title_text="Class")
    fig.update_yaxes(title_text="Clips")
    return fig
