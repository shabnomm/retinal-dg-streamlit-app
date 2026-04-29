from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st


PAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PAGE_DIR.parent

METRICS_DIR: Final[Path] = ROOT_DIR / "data" / "metrics"

LODO_CSV: Final[Path] = METRICS_DIR / "lodo_results.csv"
HYBRID_CSV: Final[Path] = METRICS_DIR / "hybrid_results.csv"
COMPARISON_CSV: Final[Path] = METRICS_DIR / "comparison_results.csv"
HYBRID_EXPERIMENTAL_CSV: Final[Path] = METRICS_DIR / "hybrid_experimental_notes.csv"


st.set_page_config(
    page_title="DG vs Hybrid Analysis | Retinal DG App",
    page_icon="🧬",
    layout="wide",
)


def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not file_exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def validate_result_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"domain", "mAUC", "mAP", "macro_f1", "micro_f1"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))

    out = df.copy()
    out["held_out_dataset"] = out["domain"].astype(str).str.strip()
    out["model"] = "Baseline"
    out["accuracy"] = pd.to_numeric(out["micro_f1"], errors="coerce")
    out["f1"] = pd.to_numeric(out["macro_f1"], errors="coerce")
    out["auc"] = pd.to_numeric(out["mAUC"], errors="coerce")
    out["precision"] = pd.to_numeric(out["mAP"], errors="coerce")
    out["recall"] = pd.to_numeric(out["micro_f1"], errors="coerce")
    return out


def validate_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "baseline_mAUC", "dg_mAUC", "hybrid_mAUC"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "comparison_results.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    for col in ["baseline_mAUC", "dg_mAUC", "hybrid_mAUC"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def validate_experimental_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "accuracy", "f1", "auc", "precision", "recall", "status", "note"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "hybrid_experimental_notes.csv is missing required columns: "
            + ", ".join(sorted(missing))
        )

    out = df.copy()
    out["held_out_dataset"] = out["dataset"].astype(str).str.strip()
    out["model"] = "Hybrid"
    return out


def format_metric(value: float) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def get_available_datasets(
    baseline_df: pd.DataFrame,
    hybrid_df: pd.DataFrame,
) -> list[str]:
    baseline_ds = set(baseline_df["held_out_dataset"].dropna().unique().tolist())
    hybrid_ds = set(hybrid_df["held_out_dataset"].dropna().unique().tolist())
    all_ds = baseline_ds.union(hybrid_ds)

    preferred_order = ["RFMiD v1", "RFMiD_v1", "RFMiD v2", "RFMiD_v2"]
    ordered = [d for d in preferred_order if d in all_ds] + [
        d for d in sorted(all_ds) if d not in preferred_order and d != "ODIR"
    ]
    return ordered


def get_row_by_dataset(df: pd.DataFrame, dataset_name: str) -> pd.Series:
    subset = df[df["held_out_dataset"].str.lower() == dataset_name.lower()]
    if subset.empty:
        raise ValueError(f"No row found for dataset: {dataset_name}")
    return subset.iloc[0]


def get_comparison_row(comparison_df: pd.DataFrame | None, dataset_name: str) -> pd.Series | None:
    if comparison_df is None:
        return None

    subset = comparison_df[
        comparison_df["dataset"].astype(str).str.strip().str.lower() == dataset_name.lower()
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def render_title_block() -> None:
    st.title("DG vs Hybrid Analysis")
    st.caption("Method-level comparison beyond the baseline")

    st.markdown(
        """
        This page explains how the project progressed from baseline cross-domain testing
        into **domain generalization methods** and finally into the **proposed hybrid model**.
        The goal is to show whether the more advanced methods provide a stronger response
        to domain shift than the baseline alone.
        """
    )


def render_context_box() -> None:
    st.info(
        "This page focuses on the research transition: baseline LODO drop → DG experimentation → hybrid proposal."
    )


def render_dataset_selector(available_datasets: list[str]) -> str:
    st.subheader("Analysis Controls")
    return st.selectbox(
        "Select held-out dataset",
        options=available_datasets,
        index=0,
        key="dg_hybrid_dataset_selector",
    )


def render_summary_table(
    selected_dataset: str,
    baseline_row: pd.Series,
    hybrid_row: pd.Series,
    comparison_row: pd.Series | None,
) -> None:
    st.subheader("Method Summary Table")

    dg_mauc = None if comparison_row is None else comparison_row.get("dg_mAUC")

    metric_df = pd.DataFrame(
        {
            "Model": ["Baseline", "Hybrid"],
            "mAUC": [float(baseline_row["auc"]), float(hybrid_row["auc"])],
            "Macro F1": [float(baseline_row["f1"]), float(hybrid_row["f1"])],
            "Micro F1": [float(baseline_row["accuracy"]), float(hybrid_row["accuracy"])],
        }
    )

    if dg_mauc is not None and pd.notna(dg_mauc):
        dg_row = pd.DataFrame(
            {
                "Model": ["DG"],
                "mAUC": [float(dg_mauc)],
                "Macro F1": [None],
                "Micro F1": [None],
            }
        )
        metric_df = pd.concat([metric_df.iloc[:1], dg_row, metric_df.iloc[1:]], ignore_index=True)

    st.dataframe(
        metric_df.style.format(
            {
                "mAUC": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Micro F1": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_why_this_page_matters() -> None:
    st.subheader("Why We Added DG and Hybrid Stages")

    st.markdown(
        """
        Once the baseline LODO results showed a clear performance drop on unseen datasets,
        the next logical step was to ask whether the model could be made more robust.
        That is why the project moved into domain generalization methods.
        """
    )

    st.markdown(
        """
        DG methods attempt to reduce the dependency on dataset-specific patterns. However,
        because different DG methods behave differently under different domain conditions,
        the project did not stop there. A hybrid strategy was proposed in order to combine
        multiple DG ideas rather than relying on only one mechanism.
        """
    )


def render_interpretation(
    selected_dataset: str,
    baseline_row: pd.Series,
    hybrid_row: pd.Series,
    comparison_row: pd.Series | None,
) -> None:
    st.subheader("Interpretation")

    baseline_mauc = float(baseline_row["auc"])
    hybrid_mauc = float(hybrid_row["auc"])
    dg_mauc = None if comparison_row is None else comparison_row.get("dg_mAUC")

    if hybrid_mauc > baseline_mauc:
        trend_text = (
            f"For **{selected_dataset}**, the hybrid model improves over the baseline in mAUC "
            f"({format_metric(baseline_mauc)} → {format_metric(hybrid_mauc)})."
        )
    elif hybrid_mauc < baseline_mauc:
        trend_text = (
            f"For **{selected_dataset}**, the hybrid model does not outperform the baseline in mAUC "
            f"({format_metric(baseline_mauc)} → {format_metric(hybrid_mauc)})."
        )
    else:
        trend_text = (
            f"For **{selected_dataset}**, the hybrid model matches the baseline in mAUC "
            f"({format_metric(hybrid_mauc)})."
        )

    dg_text = ""
    if dg_mauc is not None and pd.notna(dg_mauc):
        dg_text = (
            f" The DG reference mAUC for this setting is **{format_metric(dg_mauc)}**, "
            "which helps position the hybrid result relative to an intermediate generalization strategy."
        )

    st.markdown(
        f"""
        {trend_text}{dg_text}

        The purpose of this page is not only to highlight which number is larger,
        but to show how the research evolved methodologically. The baseline result revealed the domain shift problem,
        DG methods addressed it partially, and the hybrid model was proposed as a more adaptive response.
        """
    )


def render_experimental_odir_note() -> None:
    st.subheader("Experimental Observation: Hybrid on ODIR")

    try:
        note_df = validate_experimental_df(load_csv(HYBRID_EXPERIMENTAL_CSV))
    except Exception as exc:
        st.warning(f"Could not load hybrid experimental note: {exc}")
        return

    st.warning(
        """
        The hybrid result on **ODIR** is currently treated as an experimental case
        rather than as part of the main hybrid summary.
        """
    )

    row = note_df.iloc[0]

    summary_df = pd.DataFrame(
        {
            "Domain": [row["held_out_dataset"]],
            "mAUC": [round(float(row["auc"]), 4)],
            "Macro F1": [round(float(row["f1"]), 4)],
            "Micro F1": [round(float(row["accuracy"]), 4)],
            "Status": [row["status"]],
        }
    )

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        **Limitation Explanation:**

        {row["note"]}
        """
    )


def render_takeaway() -> None:
    st.subheader("Key Takeaway")

    st.success(
        "This page shows that the project did not stop at baseline evaluation. "
        "It moved into domain generalization and hybrid modelling in order to respond to the cross-domain failure observed under LODO."
    )


def main() -> None:
    render_title_block()
    render_context_box()

    try:
        baseline_df = validate_result_df(load_csv(LODO_CSV))
        hybrid_df = validate_result_df(load_csv(HYBRID_CSV))
        comparison_df = validate_comparison_df(load_csv(COMPARISON_CSV)) if file_exists(COMPARISON_CSV) else None
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load DG/Hybrid analysis resources: {exc}")
        st.stop()

    available_datasets = get_available_datasets(baseline_df, hybrid_df)
    if not available_datasets:
        st.error("No datasets available for DG vs Hybrid analysis.")
        st.stop()

    st.markdown("---")
    selected_dataset = render_dataset_selector(available_datasets)

    try:
        baseline_row = get_row_by_dataset(baseline_df, selected_dataset)
        hybrid_row = get_row_by_dataset(hybrid_df, selected_dataset)
        comparison_row = get_comparison_row(comparison_df, selected_dataset)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.markdown("---")
    render_summary_table(selected_dataset, baseline_row, hybrid_row, comparison_row)

    st.markdown("---")
    render_why_this_page_matters()

    st.markdown("---")
    render_interpretation(selected_dataset, baseline_row, hybrid_row, comparison_row)

    st.markdown("---")
    render_experimental_odir_note()

    st.markdown("---")
    render_takeaway()


if __name__ == "__main__":
    main()