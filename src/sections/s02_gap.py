"""s02_gap.py - Why current practice falls short."""
import streamlit as st
from theme import section_header


def render():
    section_header("2. Why Current Practice Falls Short", "gap")
    st.markdown("""
Four specific gaps motivate this pipeline:

1. **Single-modality monitoring misses evidence.** Relying only on vibration misses high-frequency
   surface events; relying only on audio misses structural-depth signatures. A single sensor is
   insufficient when different fault types produce complementary evidence across modalities.

2. **Fixed thresholds fire on transients.** A hard RMS or temperature limit will trigger on normal
   operational peaks - start-up, load changes - rather than learning the signal structure of a fault
   from examples. The result is a high false-alarm rate from the very first day of deployment.

3. **Alarm fatigue destroys operator trust.** When operators ignore alerts because the alert rate is
   high, a missed fault is no better than no monitoring at all. Controlling false alarms is as
   important as detecting faults, and the two objectives must be made explicit.

4. **Per-clip evaluation ignores continuity.** A real installation produces a continuous stream.
   Evaluating a classifier at the clip level hides latency and alarm-persistence behaviour that
   operators actually experience. A temporal decision layer makes this trade-off explicit and tunable.
""")
