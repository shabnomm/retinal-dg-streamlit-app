from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
PAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PAGE_DIR.parent
SAMPLES_DIR: Final[Path] = ROOT_DIR / "data" / "samples"


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Dataset Explorer | Retinal DG App",
    page_icon="🗂️",
    layout="wide",
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
DATASET_ORDER: Final[list[str]] = ["ODIR", "RFMiD v1", "RFMiD v2"]

DATASET_TO_FOLDER: Final[dict[str, str]] = {
    "ODIR": "odir",
    "RFMiD v1": "rfmid_v1",
    "RFMiD v2": "rfmid_v2",
}

DISPLAY_CLASS_TO_FOLDER: Final[dict[str, str]] = {
    "Normal": "normal",
    "Diabetic Retinopathy": "diabetic_retinopathy",
    "Glaucoma": "glaucoma",
    "Cataract": "cataract",
}

SUPPORTED_IMAGE_SUFFIXES: Final[set[str]] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


# ---------------------------------------------------------------------
# Hardcoded research statistics from final harmonization output
# ---------------------------------------------------------------------
DATASET_SPLIT_COUNTS: Final[dict[str, dict[str, int]]] = {
    "ODIR": {"Train": 4318, "Val": 622, "Test": 1252},
    "RFMiD v1": {"Train": 1005, "Val": 319, "Test": 337},
    "RFMiD v2": {"Train": 220, "Val": 75, "Test": 74},
}

TOTAL_COUNTS: Final[dict[str, int]] = {
    "Total Train": 5543,
    "Total Val": 1016,
    "Total Test": 1663,
    "Grand Total": 8222,
}

LABEL_DISTRIBUTION: Final[dict[str, dict[str, dict[str, int]]]] = {
    "ODIR": {
        "Train": {"Normal": 2280, "Diabetic Retinopathy": 1502, "Glaucoma": 290, "Cataract": 358},
        "Val": {"Normal": 324, "Diabetic Retinopathy": 218, "Glaucoma": 44, "Cataract": 54},
        "Test": {"Normal": 648, "Diabetic Retinopathy": 446, "Glaucoma": 86, "Cataract": 110},
    },
    "RFMiD v1": {
        "Train": {"Normal": 401, "Diabetic Retinopathy": 277, "Glaucoma": 166, "Cataract": 228},
        "Val": {"Normal": 134, "Diabetic Retinopathy": 102, "Glaucoma": 33, "Cataract": 74},
        "Test": {"Normal": 134, "Diabetic Retinopathy": 103, "Glaucoma": 48, "Cataract": 85},
    },
    "RFMiD v2": {
        "Train": {"Normal": 156, "Diabetic Retinopathy": 42, "Glaucoma": 8, "Cataract": 15},
        "Val": {"Normal": 53, "Diabetic Retinopathy": 14, "Glaucoma": 6, "Cataract": 3},
        "Test": {"Normal": 52, "Diabetic Retinopathy": 14, "Glaucoma": 5, "Cataract": 3},
    },
}

FILTER_SUMMARY_TEXT: Final[dict[str, dict[str, str]]] = {
    "ODIR": {
        "Train": "kept 4318/7000, dropped 2682",
        "Val": "kept 622/1000, dropped 378",
        "Test": "kept 1252/2000, dropped 748",
    },
    "RFMiD v1": {
        "Train": "kept 1005/1920, dropped 915",
        "Val": "kept 319/640, dropped 321",
        "Test": "kept 337/640, dropped 303",
    },
    "RFMiD v2": {
        "Train": "found 507, skipped 2 → kept 220/507, dropped 287",
        "Val": "kept 75/177, dropped 102",
        "Test": "found 170, skipped 4 → kept 74/170, dropped 96",
    },
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def dir_exists(path: Path) -> bool:
    """Safely check whether a directory exists."""
    try:
        return path.exists() and path.is_dir()
    except OSError:
        return False


def get_dataset_folder(dataset_name: str) -> Path:
    """Map dataset display name to sample folder path."""
    if dataset_name not in DATASET_TO_FOLDER:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return SAMPLES_DIR / DATASET_TO_FOLDER[dataset_name]


def list_sample_images_for_class(
    dataset_name: str,
    class_name: str,
    max_images: int = 4,
) -> list[Path]:
    """List sample images for a dataset/class combination."""
    dataset_folder = get_dataset_folder(dataset_name)
    class_folder_name = DISPLAY_CLASS_TO_FOLDER.get(class_name)

    if class_folder_name is None:
        return []

    class_folder = dataset_folder / class_folder_name
    if not dir_exists(class_folder):
        return []

    images = [
        path
        for path in sorted(class_folder.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]

    return images[:max_images]


def build_split_summary_df(dataset_name: str) -> pd.DataFrame:
    """Create split count table for a dataset."""
    split_info = DATASET_SPLIT_COUNTS[dataset_name]
    return pd.DataFrame(
        {
            "Split": list(split_info.keys()),
            "Images": list(split_info.values()),
        }
    )


def build_distribution_df(dataset_name: str, split_name: str) -> pd.DataFrame:
    """Create class distribution table with percentages."""
    split_counts = LABEL_DISTRIBUTION[dataset_name][split_name]
    total = sum(split_counts.values())

    rows = []
    for class_name, count in split_counts.items():
        percentage = (count / total * 100.0) if total > 0 else 0.0
        rows.append(
            {
                "Class": class_name,
                "Count": count,
                "Percentage": percentage,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------
def render_title_block() -> None:
    st.title("Dataset Explorer")
    st.caption("Harmonization rules, dataset statistics, and retinal image samples")

    st.markdown(
        """
        This page summarizes how the three datasets — **ODIR, RFMiD v1, and RFMiD v2** —
        were harmonized into a common **4-class retinal disease classification problem**.
        It also highlights the dataset imbalance and the practical data limitations that
        directly affected the design of the research.
        """
    )


def render_harmonization_rules() -> None:
    st.subheader("Harmonization Rules")

    st.markdown(
        """
        The datasets were harmonized under a **strict 4-class setup**:

        - **Normal**
        - **Diabetic Retinopathy**
        - **Glaucoma**
        - **Cataract**
        """
    )

    st.warning(
        """
        Harmonization constraints:
        - Samples with any disease **outside NDGC** were discarded
        - Multi-label samples were allowed **only within D / G / C**
        - **Normal** was treated as an exclusive class
        """
    )


def render_overall_statistics() -> None:
    st.subheader("Overall Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Train", f"{TOTAL_COUNTS['Total Train']}")
    with col2:
        st.metric("Total Val", f"{TOTAL_COUNTS['Total Val']}")
    with col3:
        st.metric("Total Test", f"{TOTAL_COUNTS['Total Test']}")
    with col4:
        st.metric("Grand Total", f"{TOTAL_COUNTS['Grand Total']}")

    overall_df = pd.DataFrame(
        {
            "Dataset Split": [
                "ODIR_train", "ODIR_val", "ODIR_test",
                "RFMiD_v1_train", "RFMiD_v1_val", "RFMiD_v1_test",
                "RFMiD_v2_train", "RFMiD_v2_val", "RFMiD_v2_test",
            ],
            "Images": [
                4318, 622, 1252,
                1005, 319, 337,
                220, 75, 74,
            ],
        }
    )

    st.dataframe(overall_df, use_container_width=True, hide_index=True)


def render_dataset_selector() -> str:
    st.subheader("Dataset Inspection")

    selected_dataset = st.selectbox(
        "Select a dataset",
        options=DATASET_ORDER,
        index=0,
        key="dataset_explorer_selector",
    )
    return selected_dataset


def render_dataset_summary(selected_dataset: str) -> None:
    st.subheader(f"{selected_dataset} Summary")

    split_df = build_split_summary_df(selected_dataset)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.dataframe(split_df, use_container_width=True, hide_index=True)

    with col2:
        st.bar_chart(split_df, x="Split", y="Images", use_container_width=True)

    st.caption(f"Filtering summary — {selected_dataset}")
    for split_name in ["Train", "Val", "Test"]:
        st.write(f"- **{split_name}:** {FILTER_SUMMARY_TEXT[selected_dataset][split_name]}")


def render_split_distribution(selected_dataset: str) -> None:
    st.subheader("Split-wise Label Distribution")

    selected_split = st.radio(
        "Choose split",
        options=["Train", "Val", "Test"],
        horizontal=True,
        key=f"{selected_dataset}_split_selector",
    )

    dist_df = build_distribution_df(selected_dataset, selected_split)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        styled_df = dist_df.style.format({"Percentage": "{:.1f}%"})
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with col2:
        st.bar_chart(dist_df, x="Class", y="Count", use_container_width=True)

    st.markdown(
        f"""
        In **{selected_dataset} ({selected_split})**, the class distribution is not uniform.
        This imbalance is one of the reasons why cross-domain generalization becomes difficult,
        especially under the LODO setting where training data are further reduced.
        """
    )


def render_visual_note(selected_dataset: str) -> None:
    st.subheader("Why Visual Inspection Matters")

    st.markdown(
        f"""
        Even after label harmonization, **{selected_dataset}** still differs from the other datasets in
        terms of image appearance, acquisition style, and class balance. These differences help explain
        why a model can perform well under a matched-distribution setting but struggle on an unseen dataset.
        """
    )


def render_sample_gallery(selected_dataset: str) -> None:
    st.subheader("Sample Retinal Images")

    any_found = False

    for display_class in DISPLAY_CLASS_TO_FOLDER:
        st.markdown(f"**{display_class}**")

        image_paths = list_sample_images_for_class(
            dataset_name=selected_dataset,
            class_name=display_class,
            max_images=4,
        )

        if not image_paths:
            st.caption(f"No sample images found yet for **{display_class}** in **{selected_dataset}**.")
            continue

        any_found = True
        cols = st.columns(min(4, len(image_paths)))

        for col, img_path in zip(cols, image_paths):
            with col:
                st.image(str(img_path), caption=img_path.name, use_container_width=True)

        st.markdown("")

    if not any_found:
        st.info(
            "No sample images were found in the current `data/samples/` folder structure yet."
        )


def render_dataset_role(selected_dataset: str) -> None:
    st.subheader("Role in the Research Pipeline")

    st.markdown(
        f"""
        **{selected_dataset}** contributes to the project in two different ways:

        1. As part of the **combined in-domain baseline** setting  
        2. As a possible **held-out target dataset** in the LODO evaluation

        This dual role is central to the project, because it allows us to compare
        performance under a relatively favorable distribution-matched setting versus a
        much harder unseen-domain scenario.
        """
    )


def render_takeaway_box() -> None:
    st.subheader("Key Takeaway")

    st.success(
        "The Dataset Explorer shows that the project is built under a genuinely constrained "
        "multi-dataset setup: harmonization was necessary, class balance is uneven, and the "
        "available images are limited enough to make cross-domain generalization challenging."
    )


def render_footer_note() -> None:
    st.markdown("---")
    st.caption(
        "This page uses the final harmonized dataset statistics and automatically shows sample images "
        "from the local `data/samples/` structure."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    render_title_block()

    st.markdown("---")
    render_harmonization_rules()

    st.markdown("---")
    render_overall_statistics()

    st.markdown("---")
    selected_dataset = render_dataset_selector()

    st.markdown("---")
    render_dataset_summary(selected_dataset)

    st.markdown("---")
    render_split_distribution(selected_dataset)

    st.markdown("---")
    render_visual_note(selected_dataset)

    st.markdown("---")
    render_sample_gallery(selected_dataset)

    st.markdown("---")
    render_dataset_role(selected_dataset)

    st.markdown("---")
    render_takeaway_box()
    render_footer_note()


if __name__ == "__main__":
    main()