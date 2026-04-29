import streamlit as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "data" / "samples"


st.set_page_config(layout="wide")

st.title("Project Overview")
st.caption("Retinal Disease Classification under Domain Shift")

st.markdown("---")

# -------------------------
# Problem
# -------------------------
st.subheader("The Problem")

st.markdown("""
Deep learning models often perform well when training and testing data come from the same distribution.
However, in real-world medical applications, data comes from different sources, devices, and populations.

This creates a **domain shift problem**, where a model trained on one dataset fails on another.
""")

# -------------------------
# Example Images
# -------------------------
col1, col2, col3 = st.columns(3)

col1.image(str(IMG_DIR / "odir/normal/normal_1.jpg"), caption="ODIR - Normal")
col2.image(str(IMG_DIR / "rfmid_v1/diabetic_retinopathy/dr_3.png"), caption="RFMiD v1 - DR")
col3.image(str(IMG_DIR / "rfmid_v2/glaucoma/glaucoma_5.jpg"), caption="RFMiD v2 - Glaucoma")

# -------------------------
# Objective
# -------------------------
st.subheader("Objective")

st.markdown("""
The goal of this project is to evaluate how well models generalize across datasets using:

- Baseline training
- Cross-domain testing (LODO)
- Domain Generalization techniques
- A proposed Hybrid model
""")

# -------------------------
# Pipeline Overview
# -------------------------
st.subheader("Pipeline Overview")

st.image("assets/diagrams/pipeline_overview.png")

# -------------------------
# Key Insight
# -------------------------
st.subheader("Key Insight")

st.success("""
A model performing well on one dataset does not guarantee performance on unseen datasets.
This project focuses on understanding and improving that gap.
""")