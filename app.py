import streamlit as st

st.set_page_config(
    page_title="Retinal DG System",
    page_icon="🧠",
    layout="wide"
)

st.title("Multi-Dataset Retinal Disease Classification System")
st.caption("Domain Generalization | LODO Evaluation | Hybrid Model")

st.markdown("---")

st.markdown("""
This application presents a complete research pipeline for **retinal disease classification**
using multiple datasets under domain shift conditions.

### What this app demonstrates:
- Multi-dataset harmonization
- Baseline vs Cross-domain (LODO)
- Domain Generalization (DG)
- Proposed Hybrid Model
- Interactive comparison & prediction

---

👉 Use the sidebar to navigate through the research story.
""")

st.success("Start from the **Home** page →")