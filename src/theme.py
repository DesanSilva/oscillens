import streamlit as st
import config as config


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Base typography ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
    color: #1E293B;
    background-color: #F8FAFC !important;
}

h1 {
    font-size: 44px !important;
    font-weight: 700 !important;
    line-height: 1.20 !important;
    color: #0F172A !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem !important;
}
h2 {
    font-size: 32px !important;
    font-weight: 600 !important;
    line-height: 1.25 !important;
    color: #0F172A !important;
    margin-top: 1rem !important;
    margin-bottom: 0.75rem !important;
}
h3 {
    font-size: 24px !important;
    font-weight: 600 !important;
    line-height: 1.30 !important;
    color: #1E293B !important;
    margin-bottom: 0.5rem !important;
}
h4 {
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1E293B !important;
}
p, li {
    font-size: 18px !important;
    line-height: 1.60 !important;
    max-width: 72ch;
    color: #334155;
}
small, .caption, figcaption, .stCaption p {
    font-size: 15px !important;
    color: #475569 !important;
    max-width: 80ch;
}

/* tabular figures on all numeric display */
table, .stMetricValue, .stMetricDelta, code {
    font-variant-numeric: tabular-nums;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* ── Rules ───────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #DDE2E7 !important;
    margin: 2.5rem 0 !important;
}

/* ── Layout ──────────────────────────────────────────────────── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    max-width: 1100px;
}

/* ── Animations ──────────────────────────────────────────────── */
@keyframes fadeUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* ── Section bands ───────────────────────────────────────────── */
.surface-section {
    background: #FFFFFF;
    padding: 2.5rem 2rem;
    border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
    animation: fadeUp 0.6s ease-out forwards;
    transition: box-shadow 0.3s ease;
}
.surface-section:hover {
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}
.canvas-section {
    background: #F1F5F9;
    padding: 2.5rem 2rem;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    margin: 1.5rem 0;
    animation: fadeUp 0.6s ease-out forwards;
}

/* ── Metric tiles ────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}
[data-testid="stMetricLabel"] p {
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #475569 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
    font-variant-numeric: tabular-nums;
}

/* ── Status badges ───────────────────────────────────────────── */
.badge-alarm {
    display: inline-block;
    background: #C2410C;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.02em;
}
.badge-ok {
    display: inline-block;
    background: #2F7D4F;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 4px;
    letter-spacing: 0.02em;
}
.badge-info {
    display: inline-block;
    background: #1F5C8B;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 4px;
}

/* ── Tables ──────────────────────────────────────────────────── */
table {
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 14px !important;
}
th {
    background: #F5F7F9 !important;
    color: #14181D !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-bottom: 1px solid #DDE2E7 !important;
    padding: 8px 12px !important;
    text-align: left !important;
}
td {
    border-bottom: 1px solid #DDE2E7 !important;
    padding: 7px 12px !important;
    color: #14181D !important;
    vertical-align: top;
}
tr:nth-child(even) td { background: #F5F7F9 !important; }

/* ── Sidebar ─────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #F1F5F9 !important;
    border-right: 1px solid #E2E8F0;
}
[data-testid="stSidebar"] a {
    color: #0F172A !important;
    font-size: 15px !important;
    text-decoration: none;
    display: block;
    padding: 4px 0;
}
[data-testid="stSidebar"] a:hover {
    color: #2563EB !important;
}
[data-testid="stSidebar"] h3 {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    text-transform: none;
    letter-spacing: 0;
    margin-top: 1.5rem !important;
    margin-bottom: 0.4rem !important;
}

/* ── Buttons / controls ──────────────────────────────────────── */
.stButton button {
    background: #1F5C8B;
    color: #FFFFFF;
    border: none;
    border-radius: 4px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    padding: 6px 16px;
    cursor: pointer;
}
.stButton button:hover {
    background: #174D78;
}

/* ── Plotly / chart containers ───────────────────────────────── */
.js-plotly-plot .plotly {
    border-radius: 4px;
    border: 1px solid #DDE2E7;
}

/* ── Audio widget ────────────────────────────────────────────── */
audio {
    width: 100%;
    border-radius: 4px;
    margin-bottom: 0.5rem;
}

/* ── Warning / info boxes ────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left-width: 4px !important;
    font-size: 14px !important;
}

/* ── Expander ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #DDE2E7 !important;
    border-radius: 4px !important;
    background: #FFFFFF;
}
[data-testid="stExpanderToggleIcon"] { color: #1F5C8B !important; }

/* ── No purple anywhere in UI chrome (class colours only in plots) */
</style>
"""


def apply_theme():
    st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def section_header(title: str, id_slug: str):
    """Render a section anchor + H2 heading + hairline rule."""
    st.markdown(f'<div id="{id_slug}"></div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    st.markdown('<hr style="margin:0.5rem 0 1.5rem 0;">', unsafe_allow_html=True)


def canvas_band_open():
    st.markdown('<div class="canvas-section">', unsafe_allow_html=True)


def canvas_band_close():
    st.markdown("</div>", unsafe_allow_html=True)


def status_badge(state: str) -> str:
    """Return HTML badge string for alarm/ok/info state."""
    cls = {"alarm": "badge-alarm", "ok": "badge-ok"}.get(state.lower(), "badge-info")
    label = state.title()
    return f'<span class="{cls}">{label}</span>'


def figcaption(text: str):
    st.markdown(f'<p class="caption">{text}</p>', unsafe_allow_html=True)
