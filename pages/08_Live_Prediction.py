from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st
import torch

from utils.data_loader import load_class_labels, load_model_registry, resolve_model_path
from utils.inference import (
    build_prediction_table,
    format_confidence,
    get_probability_distribution,
    predict,
)
from utils.model_loader import get_device, get_model_file_info, load_registered_model
from utils.preprocessing import (
    get_preprocessing_summary,
    prepare_uploaded_image,
)


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
    page_title="Live Prediction | Retinal DG App",
    page_icon="🧠",
    layout="wide",
)


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
MODEL_ORDER: Final[list[str]] = ["Baseline", "DG", "Hybrid"]
DATASET_ORDER: Final[list[str]] = ["ODIR", "RFMiD v1", "RFMiD v2"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def file_exists(path: Path) -> bool:
    """Safely check whether a file exists."""
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def get_available_model_types() -> list[str]:
    """Read available model groups from registry and keep preferred order."""
    registry = load_model_registry()
    available = [model_name for model_name in MODEL_ORDER if model_name in registry]
    return available


def get_available_dataset_contexts(model_type: str) -> list[str]:
    """Return available dataset contexts for a selected model type."""
    registry = load_model_registry()
    dataset_map = registry.get(model_type, {})
    return [item for item in DATASET_ORDER if item in dataset_map]


def dataset_to_sample_dir(dataset_name: str) -> Path:
    """Map dataset display name to sample directory."""
    mapping = {
        "ODIR": SAMPLES_DIR / "odir",
        "RFMiD v1": SAMPLES_DIR / "rfmid_v1",
        "RFMiD v2": SAMPLES_DIR / "rfmid_v2",
    }
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return mapping[dataset_name]


@st.cache_resource(show_spinner=False)
def load_cached_model(model_type: str, dataset_context: str):
    """
    Cached model loader for Streamlit.

    This avoids reloading the same checkpoint repeatedly when the same
    model + dataset context are selected.
    """
    device = get_device(prefer_gpu=True)
    model = load_registered_model(
        model_type=model_type,
        dataset_context=dataset_context,
        device=device,
        strict=True,
    )
    return model, device


def try_list_sample_images(dataset_name: str, max_images: int = 6) -> list[Path]:
    """
    Optionally discover sample images from data/samples.
    This is a convenience feature and does not block the page.
    """
    sample_root = dataset_to_sample_dir(dataset_name)
    if not sample_root.exists() or not sample_root.is_dir():
        return []

    image_paths: list[Path] = []
    supported = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

    for subdir in sorted(sample_root.iterdir()):
        if not subdir.is_dir():
            continue
        for img_path in sorted(subdir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in supported:
                image_paths.append(img_path)
                if len(image_paths) >= max_images:
                    return image_paths

    return image_paths


def render_title_block() -> None:
    st.title("Live Prediction")
    st.caption("Interactive inference using selected model type and LODO dataset context")

    st.markdown(
        """
        This page allows interactive prediction on a retinal image using the selected
        **model type** and **dataset context**. Because the trained models are LODO-based,
        the dataset context is required to load the correct model checkpoint.
        """
    )


def render_context_note() -> None:
    st.info(
        "Important: The model selection here is **dataset-context aware**. "
        "For example, selecting **Test on ODIR** loads the model corresponding "
        "to the ODIR LODO evaluation setting."
    )


def render_controls() -> tuple[str, str]:
    """Render model and dataset selection controls."""
    st.subheader("Model Selection")

    available_models = get_available_model_types()
    if not available_models:
        st.error("No model groups found in `data/metadata/model_registry.json`.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        selected_model_type = st.selectbox(
            "Select model type",
            options=available_models,
            index=available_models.index("Hybrid") if "Hybrid" in available_models else 0,
            key="live_prediction_model_type",
        )

    with col2:
        available_contexts = get_available_dataset_contexts(selected_model_type)
        if not available_contexts:
            st.error(f"No dataset contexts found for model type: {selected_model_type}")
            st.stop()

        selected_dataset_context = st.selectbox(
            "Select dataset context",
            options=available_contexts,
            index=0,
            key="live_prediction_dataset_context",
        )

    return selected_model_type, selected_dataset_context


def render_model_info(model_type: str, dataset_context: str) -> None:
    """Show model path/config path for transparency."""
    st.subheader("Selected Model Info")

    try:
        info = get_model_file_info(model_type, dataset_context)
    except Exception as exc:
        st.warning(f"Could not resolve model info: {exc}")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.code(info["model_path"], language="text")
    with col2:
        st.code(info["config_path"], language="text")


def render_preprocessing_info() -> None:
    """Display preprocessing summary."""
    st.subheader("Preprocessing Setup")

    try:
        summary = get_preprocessing_summary()
    except Exception as exc:
        st.warning(f"Could not load preprocessing configuration: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Image Size", str(summary["image_size"]))
    with col2:
        st.metric("Color Mode", str(summary["color_mode"]))
    with col3:
        st.metric("Normalization", "Enabled" if summary["normalization"] else "Disabled")

    with st.expander("Show normalization statistics"):
        st.write("Mean:", summary["mean"])
        st.write("Std:", summary["std"])


def render_uploader() -> object:
    """Render file uploader."""
    st.subheader("Upload Retinal Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="live_prediction_uploader",
    )

    st.caption(
        "Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF, WEBP. "
        "Uploaded images are converted to RGB and preprocessed automatically."
    )

    return uploaded_file


def render_sample_gallery(dataset_context: str) -> None:
    """Optional sample image gallery for quick context."""
    st.subheader("Optional Dataset Sample Preview")

    sample_images = try_list_sample_images(dataset_context, max_images=4)
    if not sample_images:
        st.caption(
            "No sample dataset images were found in `data/samples/`. "
            "This does not affect live prediction."
        )
        return

    columns = st.columns(min(4, len(sample_images)))
    for col, img_path in zip(columns, sample_images):
        with col:
            st.image(str(img_path), caption=img_path.name, use_container_width=True)


def render_prediction_results(
    model_type: str,
    dataset_context: str,
    uploaded_file,
) -> None:
    """Run model inference and render outputs."""
    st.subheader("Prediction Output")

    if uploaded_file is None:
        st.info("Upload an image to run prediction.")
        return

    try:
        pil_image, input_tensor = prepare_uploaded_image(uploaded_file)
    except Exception as exc:
        st.error(f"Unable to prepare uploaded image: {exc}")
        return

    left_col, right_col = st.columns([1, 1.2])

    with left_col:
        st.markdown("**Uploaded Image**")
        st.image(pil_image, use_container_width=True)

    with right_col:
        with st.spinner("Loading model and running prediction..."):
            try:
                model, device = load_cached_model(model_type, dataset_context)
            except Exception as exc:
                st.error(
                    "Model loading failed. This usually means the checkpoint and model "
                    f"architecture are not aligned.\n\nDetails: {exc}"
                )
                return

            try:
                prediction_output = predict(model, input_tensor.to(device), top_k=3)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
                return

        predicted_class = prediction_output["predicted_class"]
        confidence = prediction_output["confidence"]

        st.success("Prediction completed successfully.")
        st.metric("Predicted Class", predicted_class)
        st.metric("Confidence", format_confidence(confidence))

        st.caption(
            f"Model used: **{model_type}** | Dataset context: **{dataset_context}**"
        )

        table_rows = build_prediction_table(prediction_output)
        table_df = pd.DataFrame(table_rows)
        st.markdown("**Top Predictions**")
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        distribution = get_probability_distribution(prediction_output)
        distribution_df = pd.DataFrame(distribution)
        distribution_df = distribution_df.rename(
            columns={"class": "Class", "probability": "Probability"}
        )

        st.markdown("**Class Probability Distribution**")
        st.bar_chart(distribution_df, x="Class", y="Probability", use_container_width=True)


def render_interpretation_note(model_type: str, dataset_context: str) -> None:
    """Add a polished interpretation note below prediction."""
    st.subheader("Interpretation Note")

    st.markdown(
        f"""
        The current prediction is generated using the **{model_type}** model under the
        **{dataset_context}** LODO context. This means the selected checkpoint represents
        a specific unseen-domain evaluation setting from the research pipeline.

        The output should therefore be interpreted as a **research demonstration result**
        rather than a clinical diagnosis. The purpose of this page is to show how trained
        models behave interactively under the project’s experimental design.
        """
    )


def render_footer_note() -> None:
    st.markdown("---")
    st.caption(
        "Next page: Explainability — where visual interpretation support can be added for model attention analysis."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    render_title_block()
    render_context_note()

    st.markdown("---")
    selected_model_type, selected_dataset_context = render_controls()

    st.markdown("---")
    render_model_info(selected_model_type, selected_dataset_context)

    st.markdown("---")
    render_preprocessing_info()

    st.markdown("---")
    uploaded_file = render_uploader()

    st.markdown("---")
    render_sample_gallery(selected_dataset_context)

    st.markdown("---")
    render_prediction_results(selected_model_type, selected_dataset_context, uploaded_file)

    st.markdown("---")
    render_interpretation_note(selected_model_type, selected_dataset_context)
    render_footer_note()


if __name__ == "__main__":
    main()