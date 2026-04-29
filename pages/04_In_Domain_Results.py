from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import streamlit as st


PAGE_DIR: Final[Path] = Path(__file__).resolve().parent
ROOT_DIR: Final[Path] = PAGE_DIR.parent
METRICS_DIR: Final[Path] = ROOT_DIR / "data" / "metrics"
IN_DOMAIN_CSV: Final[Path] = METRICS_DIR / "in_domain_results.csv"


st.set_page_config(
    page_title="In-Domain Results | Retinal DG App",
    page_icon="📊",
    layout="wide",
)


def file_exists(path: Path) -> bool:
    try:
        return path.exists() and path.is_file()
    except OSError:
        return False


@st.cache_data(show_spinner=False)
def load_in_domain_results(csv_path: Path) -> pd.DataFrame:
    if not file_exists(csv_path):
        raise FileNotFoundError(f"Missing results file: {csv_path}")

    df = pd.read_csv(csv_path)

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

    # Sort by mAUC descending so best domain stays first
    df = df.sort_values(by="mAUC", ascending=False).reset_index(drop=True)

    return df


def format_metric(value: float) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def get_primary_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        raise ValueError("The in-domain results file is empty.")
    return df.iloc[0]


def render_title_block() -> None:
    st.title("In-Domain Results")
    st.caption("Matched-distribution baseline performance")

    st.markdown(
        """
        This page presents the **in-domain baseline results**, where each domain is evaluated
        under the matched-distribution setting. In other words, the model is tested on data
        coming from the same domain setting, which gives the most favorable baseline condition
        before moving into cross-domain evaluation.
        """
    )


def render_context_box() -> None:
    st.info(
        "This page is important because it shows how well the model performs when "
        "training and testing distributions are aligned, giving a baseline reference "
        "before domain shift is introduced."
    )


def render_summary_metrics(row: pd.Series) -> None:
    st.subheader("Best In-Domain Performance")

    col1, col2 = st.columns(2)
    col3, col4, col5 = st.columns(3)

    with col1:
        st.metric("Best Domain", str(row["domain"]))
    with col2:
        st.metric("mAUC", format_metric(row["mAUC"]))
    with col3:
        st.metric("mAP", format_metric(row["mAP"]))
    with col4:
        st.metric("Macro F1", format_metric(row["macro_f1"]))
    with col5:
        st.metric("Micro F1", format_metric(row["micro_f1"]))


def render_results_table(df: pd.DataFrame) -> None:
    st.subheader("Result Table")

    display_df = df.copy()
    numeric_cols = ["mAUC", "mAP", "macro_f1", "micro_f1"]

    for col in numeric_cols:
        display_df[col] = display_df[col].map(lambda x: round(float(x), 4))

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_why_this_page_matters() -> None:
    st.subheader("Why This Stage Was Necessary")

    st.markdown(
        """
        We began with the in-domain baseline because a model should first be evaluated
        under a standard setting where training and testing come from the same domain
        distribution. This helps verify whether the model can learn meaningful retinal
        disease patterns before being challenged with unseen-domain data.
        """
    )

    st.markdown(
        """
        Without this reference, it would be difficult to interpret the later LODO and DG results.
        A weaker cross-domain performance could mean either the model failed to learn the task properly,
        or that domain shift genuinely reduced generalization. The in-domain result helps separate
        these two situations clearly.
        """
    )


def render_interpretation(row: pd.Series) -> None:
    st.subheader("Interpretation")

    st.markdown(
        f"""
        In the in-domain setting, **{row["domain"]}** achieved the strongest result with an
        **mAUC of {format_metric(row["mAUC"])}**, along with **mAP = {format_metric(row["mAP"])}**,
        **Macro F1 = {format_metric(row["macro_f1"])}**, and **Micro F1 = {format_metric(row["micro_f1"])}**.
        This indicates that the model can learn useful retinal disease classification patterns
        when the source and target distributions remain aligned.

        At the same time, in-domain performance should be treated as a reference rather than
        the final goal. The central challenge of this project is not only to perform well on
        familiar data, but also to maintain performance when evaluated on a different dataset.
        That is why the later stages focus on LODO testing and domain generalization methods.
        """
    )


def render_takeaway() -> None:
    st.subheader("Key Takeaway")

    st.success(
        "The in-domain baseline confirms that the model can perform strongly under a favorable "
        "same-domain setting. This acts as the reference point for judging how much performance "
        "changes when domain shift is introduced in later stages."
    )


def main() -> None:
    render_title_block()
    render_context_box()

    try:
        df = load_in_domain_results(IN_DOMAIN_CSV)
        row = get_primary_row(df)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Unable to load in-domain results: {exc}")
        st.stop()

    st.markdown("---")
    render_summary_metrics(row)

    st.markdown("---")
    render_results_table(df)

    st.markdown("---")
    render_why_this_page_matters()

    st.markdown("---")
    render_interpretation(row)

    st.markdown("---")
    render_takeaway()


if __name__ == "__main__":
    main()