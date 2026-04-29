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

METRICS_DIR: Final[Path] = ROOT_DIR / "data" / "metrics"

COMPARISON_CSV: Final[Path] = METRICS_DIR / "comparison_results.csv"
LODO_CSV: Final[Path] = METRICS_DIR / "lodo_results.csv"
HYBRID_CSV: Final[Path] = METRICS_DIR / "hybrid_results.csv"


# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Comparison Mode | Retinal DG App",
    page_icon="📈",
    layout="wide",
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def file_exists(path: Path) -> bool:
    """Safely check whether a file exists."""
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    """Load CSV safely."""
    if not file_exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def format_metric(value: float) -> str:
    """Format metric values consistently."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def normalize_model_name(name: str) -> str:
    """Normalize model labels for comparison."""
    normalized = str(name).strip().lower()
    mapping = {
        "baseline": "Baseline",
        "dg": "DG",
        "hybrid": "Hybrid",
    }
    return mapping.get(normalized, str(name).strip())


def is_wide_comparison_format(df: pd.DataFrame) -> bool:
    """Detect if the CSV uses wide comparison format."""
    expected_subset = {"dataset", "baseline_mAUC", "dg_mAUC", "hybrid_mAUC"}
    return expected_subset.issubset(df.columns)


def is_long_comparison_format(df: pd.DataFrame) -> bool:
    """Detect if the CSV uses long format."""
    expected_subset = {"dataset", "model", "auc"}
    return expected_subset.issubset(df.columns)


def build_long_df_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a wide comparison DataFrame into long format using mAUC only.
    """
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        dataset = row["dataset"]

        wide_map = {
            "Baseline": row.get("baseline_mAUC"),
            "DG": row.get("dg_mAUC"),
            "Hybrid": row.get("hybrid_mAUC"),
        }

        for model_name, auc_value in wide_map.items():
            records.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "auc": auc_value,
                }
            )

    return pd.DataFrame(records)


def build_long_df_from_separate_files(
    lodo_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a long-format DataFrame using baseline LODO + hybrid CSVs.
    DG values are taken from comparison_results.csv if possible.
    Comparison is based on mAUC only.
    """
    required_new = {"domain", "mAUC", "mAP", "macro_f1", "micro_f1"}
    required_old = {"held_out_dataset", "model", "accuracy", "f1", "auc", "precision", "recall"}

    def normalize_result_df(df: pd.DataFrame, model_label: str) -> pd.DataFrame:
        if required_new.issubset(df.columns):
            out = df.copy()
            out["dataset"] = out["domain"].astype(str).str.strip()
            out["model"] = model_label
            out["auc"] = pd.to_numeric(out["mAUC"], errors="coerce")
            return out[["dataset", "model", "auc"]]

        if required_old.issubset(df.columns):
            out = df.copy()
            out["dataset"] = out["held_out_dataset"].astype(str).str.strip()
            out["model"] = model_label
            out["auc"] = pd.to_numeric(out["auc"], errors="coerce")
            return out[["dataset", "model", "auc"]]

        raise ValueError("Result CSV does not contain supported columns for mAUC comparison.")

    baseline_long = normalize_result_df(lodo_df, "Baseline")
    hybrid_long = normalize_result_df(hybrid_df, "Hybrid")

    merged_parts = [baseline_long, hybrid_long]

    dg_rows: list[dict[str, object]] = []
    if comparison_df is not None and is_wide_comparison_format(comparison_df):
        for _, row in comparison_df.iterrows():
            dg_rows.append(
                {
                    "dataset": row["dataset"],
                    "model": "DG",
                    "auc": row.get("dg_mAUC"),
                }
            )
    elif comparison_df is not None and is_long_comparison_format(comparison_df):
        dg_subset = comparison_df.copy()
        dg_subset["model"] = dg_subset["model"].map(normalize_model_name)
        dg_subset = dg_subset[dg_subset["model"] == "DG"]
        if not dg_subset.empty:
            merged_parts.append(dg_subset[["dataset", "model", "auc"]])
    else:
        dg_fallback = hybrid_long.copy()
        dg_fallback["model"] = "DG"
        merged_parts.append(dg_fallback)

    if dg_rows:
        dg_df = pd.DataFrame(dg_rows)
        merged_parts.append(dg_df)

    combined = pd.concat(merged_parts, ignore_index=True, sort=False)
    combined["model"] = combined["model"].map(normalize_model_name)

    return combined


def load_comparison_long_df() -> pd.DataFrame:
    """
    Central loader for comparison mode.
    Supports:
    - comparison_results.csv in wide format with mAUC columns
    - comparison_results.csv in long format with auc
    - fallback assembly from lodo_results.csv + hybrid_results.csv
    """
    comparison_df = None
    if file_exists(COMPARISON_CSV):
        comparison_df = load_csv(COMPARISON_CSV)

        if is_long_comparison_format(comparison_df):
            long_df = comparison_df.copy()
            long_df["model"] = long_df["model"].map(normalize_model_name)
            return long_df[["dataset", "model", "auc"]]

        if is_wide_comparison_format(comparison_df):
            return build_long_df_from_wide(comparison_df)

    if not file_exists(LODO_CSV) or not file_exists(HYBRID_CSV):
        raise FileNotFoundError(
            "Comparison Mode requires either a valid `comparison_results.csv` or both "
            "`lodo_results.csv` and `hybrid_results.csv`."
        )

    lodo_df = load_csv(LODO_CSV)
    hybrid_df = load_csv(HYBRID_CSV)
    return build_long_df_from_separate_files(lodo_df, hybrid_df, comparison_df)


def prepare_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a pivoted view for dataset-wise mAUC comparison."""
    subset = df[["dataset", "model", "auc"]].copy()
    subset = subset.dropna(subset=["auc"])

    pivot_df = subset.pivot_table(
        index="dataset",
        columns="model",
        values="auc",
        aggfunc="first",
    ).reset_index()

    preferred_columns = ["dataset", "Baseline", "DG", "Hybrid"]
    ordered_columns = [col for col in preferred_columns if col in pivot_df.columns]
    return pivot_df[ordered_columns]


def render_bar_chart(df: pd.DataFrame) -> None:
    """Render a simple grouped bar chart for mAUC."""
    chart_df = df[["dataset", "model", "auc"]].dropna(subset=["auc"]).copy()
    chart_df = chart_df.rename(columns={"auc": "value"})

    st.bar_chart(
        data=chart_df,
        x="dataset",
        y="value",
        color="model",
        horizontal=False,
        use_container_width=True,
    )


def render_title_block() -> None:
    st.title("Comparison Mode")
    st.caption("Dataset-wise comparison across Baseline, DG, and Hybrid models")

    st.markdown(
        """
        This page acts as the central comparison hub of the application. It is designed to help
        compare how different modelling strategies behave across the three unseen-domain LODO settings.
        The goal is to make it easy to answer the key defence question:

        **Which approach performs better, and on which held-out dataset?**
        """
    )


def render_context_note() -> None:
    st.info(
        "This page focuses on **cross-domain comparison** using **mAUC**, interpreted in the context of "
        "LODO evaluation, where one dataset is held out as the unseen test domain."
    )


def render_controls(df: pd.DataFrame) -> list[str]:
    st.subheader("Comparison Controls")

    available_models = sorted(df["model"].dropna().unique().tolist())
    selected_models = st.multiselect(
        "Select models to include",
        options=available_models,
        default=available_models,
        key="comparison_model_selector",
    )

    if not selected_models:
        st.warning("Please select at least one model.")
        st.stop()

    return selected_models


def render_comparison_table(df: pd.DataFrame, selected_models: list[str]) -> pd.DataFrame:
    st.subheader("Comparison Table")

    filtered = df[df["model"].isin(selected_models)].copy()
    pivot_df = prepare_display_df(filtered)

    if pivot_df.empty:
        st.warning("No comparison data available for the selected model set.")
        st.stop()

    styled_df = pivot_df.style.format(
        {col: "{:.4f}" for col in pivot_df.columns if col != "dataset"}
    )
    st.dataframe(styled_df, use_container_width=True)

    return pivot_df


def render_best_model_summary(df: pd.DataFrame, selected_models: list[str]) -> None:
    st.subheader("Best Model by Dataset")

    filtered = df[df["model"].isin(selected_models)].copy()
    filtered = filtered.dropna(subset=["auc"])

    if filtered.empty:
        st.info("No valid rows available for best-model summary.")
        return

    summary_rows: list[dict[str, object]] = []
    for dataset_name, group in filtered.groupby("dataset"):
        best_row = group.sort_values("auc", ascending=False).iloc[0]
        summary_rows.append(
            {
                "Dataset": dataset_name,
                "Best Model": best_row["model"],
                "mAUC": float(best_row["auc"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(
        summary_df.style.format({"mAUC": "{:.4f}"}),
        use_container_width=True,
    )


def render_chart_section(df: pd.DataFrame, selected_models: list[str]) -> None:
    st.subheader("Visual Comparison")

    filtered = df[df["model"].isin(selected_models)].copy()
    if filtered.empty:
        st.info("No chart data available for the selected model set.")
        return

    render_bar_chart(filtered)


def render_interpretation(df: pd.DataFrame, selected_models: list[str]) -> None:
    st.subheader("Interpretation")

    filtered = df[df["model"].isin(selected_models)].copy()
    filtered = filtered.dropna(subset=["auc"])

    if filtered.empty:
        st.info("No valid rows available for interpretation.")
        return

    per_dataset_notes: list[str] = []
    for dataset_name, group in filtered.groupby("dataset"):
        best_row = group.sort_values("auc", ascending=False).iloc[0]
        per_dataset_notes.append(
            f"- On **{dataset_name}**, the best selected model is **{best_row['model']}** "
            f"with mAUC = **{format_metric(best_row['auc'])}**."
        )

    st.markdown(
        f"""
        This comparison view summarizes how the selected models behave across unseen-domain settings
        under the metric **mAUC**.

        {"".join(note + "\n" for note in per_dataset_notes)}

        The purpose of this page is not only to identify the numerically strongest model, but to support
        a **dataset-wise interpretation** of performance stability across domains.
        """
    )


def render_takeaway() -> None:
    st.subheader("Key Takeaway")

    st.success(
        "Comparison Mode provides a single place to inspect how Baseline, DG, and Hybrid models differ "
        "across unseen datasets using mAUC. This page is especially useful during defence when answering model "
        "selection and performance justification questions."
    )


def render_footer_note() -> None:
    st.markdown("---")
    st.caption(
        "Next page: Live Prediction — where a selected model and dataset context can be used for inference."
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    render_title_block()
    render_context_note()

    try:
        comparison_long_df = load_comparison_long_df()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to prepare comparison data: {exc}")
        st.stop()

    st.markdown("---")
    selected_models = render_controls(comparison_long_df)

    st.markdown("---")
    render_comparison_table(comparison_long_df, selected_models)

    st.markdown("---")
    render_best_model_summary(comparison_long_df, selected_models)

    st.markdown("---")
    render_chart_section(comparison_long_df, selected_models)

    st.markdown("---")
    render_interpretation(comparison_long_df, selected_models)

    st.markdown("---")
    render_takeaway()
    render_footer_note()


if __name__ == "__main__":
    main()