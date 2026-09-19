"""s01_context.py — Problem and industrial context."""
import streamlit as st
from theme import section_header


def render():
    section_header("1. Problem and Industrial Context", "context")
    st.markdown("""
Rotational and chain-driven equipment forms the backbone of manufacturing and logistics.
Unplanned stoppages of conveyor systems disrupt production lines and incur significant operational costs.

Conveyor systems are already equipped with cameras for quality inspection in most facilities.
Adding an acoustic and vibration health channel lets that same edge device perform predictive maintenance
without deploying additional hardware — the sensors ride on installed infrastructure.

**Fault taxonomy** (as labelled in the SSCC dataset):

| Fault | Physical cause |
|---|---|
| `dry` | Lubrication loss in moving parts |
| `lean` | Physical imbalance of the system |
| `loose` | Improper mounting or loose connections |
| `screwdrop` | Foreign object (e.g. a loose screw) introduced to the belt |
| `normal` | Healthy operating state |
""")
