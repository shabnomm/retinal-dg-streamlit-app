from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st


PAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PAGE_DIR.parent

METRICS_DIR: Final[Path] = ROOT_DIR / "data" / "metrics"
LODO_CSV: Final[Path] = METRICS_DIR / "lodo_results.csv"
IN_DOMAIN_CSV: Final[Path] = METRICS_DIR / "in_domain_results.csv"


st.set_page_config(
    page_title="Cross-Domain LODO Results | Retinal DG App",
    page_icon="🌍",
    layout="wide",
)


def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not file_exists(csv_path):
        raise FileNotFoundError(f"Missing file: {csv_path}")
    return pd.read_csv(csv_path)


def validate_lodo_df(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {
        "domain",
        "mAUC",
        "mAP",
        "macro_f1",
        "micro_f1",
    }
    missing_cols = expected_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(
            "The LODO results file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    normalized_df = df.copy()
    normalized_df["held_out_dataset"] = normalized_df["domain"].astype(str).str.strip()
    normalized_df["model"] = "Baseline"
    normalized_df["accuracy"] = pd.to_numeric(normalized_df["micro_f1"], errors="coerce")
    normalized_df["f1"] = pd.to_numeric(normalized_df["macro_f1"], errors="coerce")
    normalized_df["auc"] = pd.to_numeric(normalized_df["mAUC"], errors="coerce")
    normalized_df["precision"] = pd.to_numeric(normalized_df["mAP"], errors="coerce")
    normalized_df["recall"] = pd.to_numeric(normalized_df["micro_f1"], errors="coerce")

    return normalized_df


def validate_in_domain_df(df: pd.DataFrame) -> pd.DataFrame:
    expected_columns = {
        "domain",
        "mAUC",
        "mAP",
        "macro_f1",
        "micro_f1",
    }
    missing_cols = expected_columns.difference(df.columns)
    if missing_cols:
        raise ValueError(
            "The in-domain results file is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    normalized_df = df.copy()
    normalized_df["dataset"] = normalized_df["domain"].astype(str).str.strip()
    normalized_df["model"] = "Baseline"
    normalized_df["accuracy"] = pd.to_numeric(normalized_df["micro_f1"], errors="coerce")
    normalized_df["f1"] = pd.to_numeric(normalized_df["macro_f1"], errors="coerce")
    normalized_df["auc"] = pd.to_numeric(normalized_df["mAUC"], errors="coerce")
    normalized_df["precision"] = pd.to_numeric(normalized_df["mAP"], errors="coerce")
    normalized_df["recall"] = pd.to_numeric(normalized_df["micro_f1"], errors="coerce")

    return normalized_df


def format_metric(value: float) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def to_delta(baseline_value: float, comparison_value: float) -> str:
    try:
        delta = float(comparison_value) - float(baseline_value)
        return f"{delta:+.4f}"
    except (TypeError, ValueError):
        return "N/A"


def get_combined_in_domain_baseline(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("The in-domain results file is empty.")
    return df.iloc[0]


def get_dataset_row(df: pd.DataFrame, held_out_dataset: str) -> pd.Series:
    subset = df[df["held_out_dataset"].str.lower() == held_out_dataset.lower()]
    if subset.empty:
        raise ValueError(f"No LODO row found for dataset: {held_out_dataset}")
    return subset.iloc[0]


def render_title_block() -> None:
    st.title("Cross-Domain LODO Results")
    st.caption("Baseline performance under unseen-domain evaluation")

    st.markdown(
        """
        This page presents the **LODO baseline results**, where one dataset is kept completely unseen during training
        and used only at test time. This is the central evaluation design of the project because it directly measures
        how well the model generalizes across datasets.
        """
    )


def render_context_box() -> None:
    st.warning(
        "In LODO testing, the model is evaluated on a dataset that was not used for training. "
        "This makes the setting much harder than the in-domain baseline."
    )


def render_lodo_selector(df: pd.DataFrame) -> str:
    available_datasets = df["held_out_dataset"].dropna().unique().tolist()
    preferred_order = ["ODIR", "RFMiD v1", "RFMiD v2", "RFMiD_v1", "RFMiD_v2"]
    ordered = [d for d in preferred_order if d in available_datasets] + [
        d for d in available_datasets if d not in preferred_order
    ]

    return st.selectbox(
        "Select held-out test dataset",
        options=ordered,
        index=0,
        key="lodo_dataset_selector",
    )


def render_summary_metrics(lodo_row: pd.Series, in_domain_row: pd.Series) -> None:
    st.subheader("LODO Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "mAUC",
            format_metric(lodo_row["auc"]),
            delta=to_delta(in_domain_row["auc"], lodo_row["auc"]),
        )

    with col2:
        st.metric(
            "Macro F1",
            format_metric(lodo_row["f1"]),
            delta=to_delta(in_domain_row["f1"], lodo_row["f1"]),
        )

    with col3:
        st.metric(
            "Micro F1",
            format_metric(lodo_row["accuracy"]),
            delta=to_delta(in_domain_row["accuracy"], lodo_row["accuracy"]),
        )

    with col4:
        st.metric("Held-Out Domain", str(lodo_row["held_out_dataset"]))


def render_full_table(df: pd.DataFrame) -> None:
    st.subheader("All Baseline LODO Results")

    display_df = df.copy()
    numeric_cols = ["auc", "f1", "accuracy"]
    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 4))

    st.dataframe(
        display_df[
            ["held_out_dataset", "auc", "f1", "accuracy"]
        ].rename(
            columns={
                "held_out_dataset": "Domain",
                "auc": "mAUC",
                "f1": "Macro F1",
                "accuracy": "Micro F1",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_why_this_page_matters() -> None:
    st.subheader("Why We Performed LODO Testing")

    st.markdown(
        """
        The main research question of this project is not only whether a model can classify retinal disease,
        but whether it can continue to do so when the test data come from a **different dataset**.
        This is why LODO evaluation is essential.
        """
    )

    st.markdown(
        """
        If we only reported in-domain performance, the project would not meaningfully test domain generalization.
        LODO creates a realistic cross-domain scenario by forcing the model to face an unseen target distribution.
        This lets us observe how much performance falls once the training and testing environments no longer match.
        """
    )


def render_interpretation(lodo_row: pd.Series, in_domain_row: pd.Series) -> None:
    st.subheader("Interpretation")

    st.markdown(
        f"""
        When the model is tested on **{lodo_row['held_out_dataset']}** as a completely unseen dataset,
        the baseline achieves **mAUC = {format_metric(lodo_row["auc"])}**, **Macro F1 = {format_metric(lodo_row["f1"])}**,
        and **Micro F1 = {format_metric(lodo_row["accuracy"])}**.

        Compared with the in-domain reference, this result reflects the effect of **domain shift**.
        The performance drop is not a minor detail; it is the core reason this project moves beyond baseline learning
        into domain generalization methods and the final hybrid proposal.
        """
    )

    st.markdown(
        """
        In other words, the LODO result shows that a model trained on retinal data from some datasets
        does not automatically transfer well to every unseen dataset. That observation directly motivates
        the DG and hybrid stages that follow.
        """
    )


def render_takeaway(selected_dataset: str) -> None:
    st.subheader("Key Takeaway")

    st.success(
        f"The baseline result on **{selected_dataset}** shows how performance changes when the test domain is unseen. "
        "This is the central evidence that domain shift matters in the project."
    )


def main() -> None:
    render_title_block()
    render_context_box()

    try:
        lodo_df = validate_lodo_df(load_csv(LODO_CSV))
        in_domain_df = validate_in_domain_df(load_csv(IN_DOMAIN_CSV))
        in_domain_row = get_combined_in_domain_baseline(in_domain_df)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load required results: {exc}")
        st.stop()

    st.markdown("---")
    selected_dataset = render_lodo_selector(lodo_df)

    try:
        lodo_row = get_dataset_row(lodo_df, selected_dataset)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("---")
    render_summary_metrics(lodo_row, in_domain_row)

    st.markdown("---")
    render_full_table(lodo_df)

    st.markdown("---")
    render_why_this_page_matters()

    st.markdown("---")
    render_interpretation(lodo_row, in_domain_row)

    st.markdown("---")
    render_takeaway(selected_dataset)


if __name__ == "__main__":
    main()