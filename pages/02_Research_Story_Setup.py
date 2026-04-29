import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "samples"

st.set_page_config(layout="wide")

st.title("Research Story")
st.caption("How the project evolved under real constraints")

st.markdown("---")

# -------------------------
# Motivation
# -------------------------
st.subheader("Motivation")

st.markdown("""
Initially, the goal was to build a robust retinal disease classifier across multiple datasets.
However, we quickly encountered major limitations.
""")

# -------------------------
# Challenges
# -------------------------
st.subheader("Challenges Faced")

st.markdown("""
- Many datasets had **different label structures**
- Some datasets had **limited samples per class**
- Several useful datasets were **not accessible**
- Cross-domain training was unstable
""")

# -------------------------
# Image Variation
# -------------------------
col1, col2 = st.columns(2)

col1.image(str(IMG_DIR / "odir/cataract/cataract_1.jpg"), caption="ODIR Cataract")
col2.image(str(IMG_DIR / "rfmid_v2/cataract/cataract_5.jpg"), caption="RFMiD v2 Cataract")

st.markdown("""
Even visually, the same disease appears differently across datasets.
This increases the difficulty of generalization.
""")

# -------------------------
# Decision
# -------------------------
st.subheader("Key Design Decision")

st.markdown("""
Due to data limitations, we reduced the problem to **4 harmonized classes**:

- Normal
- Diabetic Retinopathy
- Glaucoma
- Cataract

This ensured consistency across datasets.
""")

# -------------------------
# Experiment Flow
# -------------------------
st.subheader("Experiment Strategy")

st.markdown("""
1. Train baseline model
2. Evaluate using LODO (unseen dataset)
3. Apply DG methods
4. Propose hybrid model
""")

st.info("Each step was designed to progressively address domain shift.")