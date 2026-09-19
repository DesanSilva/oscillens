"""
data.py - cached loaders over src/assets/ only.
All heavy I/O is cached; callers get plain Python structures or DataFrames.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

import config as config


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@st.cache_data
def load_manifest() -> dict:
    p = config.ASSETS_DIR / "manifest.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metadata & coverage
# ---------------------------------------------------------------------------

@st.cache_data
def load_meta() -> pd.DataFrame:
    return pd.read_parquet(config.ASSETS_DIR / "meta.parquet")


@st.cache_data
def load_coverage() -> dict:
    with open(config.ASSETS_DIR / "coverage.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------

@st.cache_data
def load_feature_schema() -> dict:
    with open(config.ASSETS_DIR / "feature_schema.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics & confusion
# ---------------------------------------------------------------------------

@st.cache_data
def load_metrics() -> dict:
    with open(config.ASSETS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_confusion_binary() -> dict:
    with open(config.ASSETS_DIR / "confusion" / "binary.json") as f:
        return json.load(f)


@st.cache_data
def load_confusion_multiclass() -> dict:
    with open(config.ASSETS_DIR / "confusion" / "multiclass.json") as f:
        return json.load(f)


@st.cache_data
def load_temporal_comparison() -> pd.DataFrame:
    p = config.ASSETS_DIR / "temporal_comparison.json"
    if p.exists():
        return pd.read_json(p)
    # Verbatim fallback from notebook 05
    return pd.DataFrame([
        {"strategy": "Raw classifier",    "recall": 0.995, "f1": 0.992, "specificity": 0.990, "false_alarms": 2,  "missed": 1},
        {"strategy": "Moving avg (k=3)",  "recall": 1.000, "f1": 0.980, "specificity": 0.960, "false_alarms": 8,  "missed": 0},
        {"strategy": "EMA (α=0.4)",       "recall": 1.000, "f1": 0.975, "specificity": 0.949, "false_alarms": 10, "missed": 0},
        {"strategy": "N-of-M (3/5)",      "recall": 0.990, "f1": 0.990, "specificity": 0.990, "false_alarms": 2,  "missed": 2},
        {"strategy": "Persistence (K=3)", "recall": 0.975, "f1": 0.987, "specificity": 1.000, "false_alarms": 0,  "missed": 5},
    ])


# ---------------------------------------------------------------------------
# Presentation videos metadata
# ---------------------------------------------------------------------------

@st.cache_data
def load_presentation_videos() -> dict:
    p = config.ASSETS_DIR / "presentation_videos.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Per-clip demo data
# ---------------------------------------------------------------------------

@st.cache_data
def load_demo_data(sample_id: str) -> dict:
    d_dir = config.ASSETS_DIR / "demo" / sample_id
    data: dict = {}

    # Vibration (decimated, shape 4×N)
    vib_path = d_dir / "vib.npy"
    if vib_path.exists():
        data["vib"] = np.load(vib_path)

    # Spectrogram
    spec_path = d_dir / "spectrogram.npz"
    if spec_path.exists():
        npz = np.load(spec_path)
        data["spectrogram"] = {k: npz[k] for k in npz.files}

    # Audio waveform (decimated)
    aw_path = d_dir / "audio_wave.npy"
    if aw_path.exists():
        data["audio_wave"] = np.load(aw_path)

    # Audio meta (sr, duration_s)
    am_path = d_dir / "audio_meta.json"
    if am_path.exists():
        with open(am_path) as f:
            data["audio_meta"] = json.load(f)

    # Audio file path (for st.audio)
    audio_path = d_dir / "audio.flac"
    if audio_path.exists():
        data["audio_path"] = str(audio_path)

    # Windows parquet (window_idx, win_start_s, win_end_s, is_fault, features, p_fault)
    win_path = d_dir / "windows.parquet"
    if win_path.exists():
        data["windows"] = pd.read_parquet(win_path)
    else:
        data["windows"] = pd.DataFrame()

    # PSD
    psd_path = d_dir / "psd.json"
    if psd_path.exists():
        with open(psd_path) as f:
            data["psd"] = json.load(f)

    # Envelope
    env_path = d_dir / "envelope.json"
    if env_path.exists():
        with open(env_path) as f:
            data["envelope"] = json.load(f)

    # WebP frames
    frames_dir = d_dir / "frames"
    if frames_dir.exists():
        data["frames"] = sorted([str(p) for p in frames_dir.glob("*.webp")])
    else:
        data["frames"] = []

    # p_fault source info
    if not data.get("windows", pd.DataFrame()).empty:
        w = data["windows"]
        data["p_fault_source"] = str(w["p_fault_source"].iloc[0]) if "p_fault_source" in w.columns else "unknown"
    else:
        data["p_fault_source"] = "unavailable"

    return data
