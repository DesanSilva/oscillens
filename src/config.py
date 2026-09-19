import os
from pathlib import Path

# Paths
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = SRC_DIR / "assets"
FONTS_DIR = ASSETS_DIR / "fonts"

# Palette from the specs
PALETTE = {
    "ink": "#14181D",
    "slate": "#4A5560",
    "hairline": "#DDE2E7",
    "surface": "#FFFFFF",
    "canvas": "#F5F7F9",
    "signal": "#1F5C8B",
    "alarm": "#C2410C",
    "ok": "#2F7D4F",
}

# Class colors
CLASS_COLORS = {
    "normal": "#2ecc71",
    "dry": "#e74c3c",
    "lean": "#e67e22",
    "loose": "#9b59b6",
    "screwdrop": "#3498db"
}

# Signal Parameters
VIB_RATE_HZ = 25000  # physical rate
AUD_RATE_HZ = 44100
CLIP_DURATION_S = 5.0

VIB_BANDS = {
    "low": (0, 1250),
    "mid": (1250, 5000),
    "high": (5000, 12500),
}
AUD_BANDS = {
    "sub": (0, 200),
    "low": (200, 1000),
    "mid": (1000, 4000),
    "high": (4000, 8000),
    "air": (8000, 16000),
}
ENV_BANDPASS_HZ = (1250, 10000)
