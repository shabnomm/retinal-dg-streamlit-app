from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Limitations & Future Work | Retinal DG App",
    page_icon="⚠️",
    layout="wide",
)


def render_title_block() -> None:
    st.title("Limitations & Future Work")
    st.caption("Research constraints, practical challenges, and next directions")

    st.markdown(
        """
        This page explains the main constraints of the study and how those constraints affected
        the interpretation of the results. It also outlines the realistic next steps for extending
        the work beyond the current capstone scope.
        """
    )


def render_limitations_overview() -> None:
    st.subheader("Why limitations matter in this project")

    st.markdown(
        """
        The goal of this project was not only to train a classifier, but to study
        **cross-domain generalization** under a realistic low-resource setting.
        Because of that, the limitations are not side notes; they directly influence
        how the results should be understood.
        """
    )

    st.info(
        "The results should be interpreted as part of a constrained multi-dataset research setting, "
        "not as a final large-scale clinical benchmark."
    )


def render_data_limitations() -> None:
    st.subheader("1. Limited Data Availability")

    st.markdown(
        """
        One of the biggest limitations of the study was the **limited amount of usable data**.
        Even though multiple public datasets were included, the number of images that could be
        consistently harmonized across shared disease categories was still limited for a robust
        multi-domain generalization study.
        """
    )

    st.markdown(
        """
        This limitation becomes even more serious under the **LODO setting**, where one dataset
        is completely held out during training. In that scenario, the effective training data become smaller,
        which increases instability and makes overfitting more likely.
        """
    )


def render_class_scope_limitations() -> None:
    st.subheader("2. Reduced Class Scope")

    st.markdown(
        """
        The original label spaces across retinal datasets are broader and not perfectly aligned.
        A larger disease taxonomy could have been explored in principle, but the study did not have
        enough balanced and consistently harmonizable images to support that reliably.
        """
    )

    st.markdown(
        """
        For that reason, the work was restricted to **four harmonized classes**:
        **Normal, Diabetic Retinopathy, Glaucoma, and Cataract**. This was a deliberate choice
        to keep the study experimentally meaningful under limited data rather than forcing a larger
        but unstable class setting.
        """
    )


def render_access_constraints() -> None:
    st.subheader("3. Dataset Access Constraints")

    st.markdown(
        """
        Another important limitation was the **lack of access to additional retinal datasets**.
        Some potentially useful datasets were private, restricted, or required permissions that were
        not available within the project timeline. This prevented broader dataset diversity.
        """
    )

    st.markdown(
        """
        As a result, the study could not fully test the models under a wider range of domain conditions.
        More datasets would likely improve both the robustness of training and the fairness of evaluation.
        """
    )


def render_method_limitations() -> None:
    st.subheader("4. Method-Level Limitations")

    st.markdown(
        """
        The study explored baseline learning, DG methods, and a proposed hybrid approach.
        However, improvement was not uniformly strong across every unseen-domain setting.
        This is expected in a low-data environment, where different datasets can respond very differently
        to the same generalization strategy.
        """
    )

    st.markdown(
        """
        Therefore, the hybrid model should be interpreted as a research attempt to improve robustness,
        not as a universally final solution across all domains.
        """
    )


def render_odir_hybrid_limitation() -> None:
    st.subheader("5. Hybrid Result Limitation on ODIR")

    st.markdown(
        """
        One specific limitation observed in this study is that the proposed hybrid model did not produce
        a satisfactory result on the **ODIR unseen-domain setting**. Although the hybrid approach showed
        stronger behaviour in other cases, the ODIR result remains comparatively weak and is therefore
        treated as an experimental observation rather than a finalized strong outcome.
        """
    )

    st.markdown(
        """
        This behaviour is likely influenced by several factors:
        - limited harmonized data under the 4-class setting
        - strong domain differences between ODIR and the RFMiD datasets
        - class imbalance after filtering and harmonization
        - the overall difficulty of low-resource unseen-domain learning
        """
    )

    st.markdown(
        """
        For this reason, the ODIR hybrid outcome is not emphasized as a final success case.
        Instead, it is presented transparently as an area that still requires improvement.
        In future work, this setting should be revisited using **larger and more diverse retinal datasets**,
        better domain balancing, and stronger generalization strategies.
        """
    )


def render_deployment_limitations() -> None:
    st.subheader("6. Not a Clinical Deployment System")

    st.markdown(
        """
        This project is a **research prototype**, not a clinically validated deployment system.
        The app is designed to support capstone defence, result comparison, and interactive demonstration.
        It should not be interpreted as a tool for real-world medical decision making.
        """
    )

    st.markdown(
        """
        Additional steps would be necessary before any real deployment could be considered, including:
        larger validation cohorts, stronger external testing, expert ophthalmology review,
        calibration analysis, and regulatory considerations.
        """
    )


def render_practical_obstacles() -> None:
    st.subheader("7. Practical Obstacles Faced During the Research")

    st.markdown(
        """
        Beyond methodological limitations, the project also faced practical obstacles such as:
        dataset alignment difficulty, limited usable images, access restrictions, and the challenge of maintaining
        stable training under multiple LODO and DG settings.
        """
    )

    st.markdown(
        """
        These constraints made the work harder, but they also make the project more realistic as an academic capstone,
        where large-scale clinical resources and perfect dataset access are rarely available.
        """
    )


def render_future_work() -> None:
    st.subheader("Future Work")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            **Data expansion**
            - include more retinal datasets
            - obtain access to larger private or institutional datasets
            - extend beyond the current 4-class setting
            - improve class balance and domain diversity
            """
        )

        st.markdown(
            """
            **Evaluation improvements**
            - stronger external validation
            - richer per-class generalization analysis
            - calibration and confidence analysis
            - deeper failure-mode investigation
            """
        )

    with col2:
        st.markdown(
            """
            **Modeling improvements**
            - stronger DG approaches
            - better hybrid fusion mechanisms
            - improved domain balancing
            - more stable unseen-domain optimization
            """
        )

        st.markdown(
            """
            **System improvements**
            - refine the research demonstration app
            - strengthen model inspection tools
            - improve interactive prediction robustness
            - support future richer interpretability extensions
            """
        )


def render_takeaway_box() -> None:
    st.subheader("Final Takeaway")

    st.success(
        "The main value of this project lies in studying domain generalization under constrained multi-dataset conditions. "
        "The limitations do not invalidate the work; they define the setting in which the results should be interpreted "
        "and point directly toward the next stage of research."
    )


def main() -> None:
    render_title_block()
    st.markdown("---")
    render_limitations_overview()
    st.markdown("---")
    render_data_limitations()
    st.markdown("---")
    render_class_scope_limitations()
    st.markdown("---")
    render_access_constraints()
    st.markdown("---")
    render_method_limitations()
    st.markdown("---")
    render_odir_hybrid_limitation()
    st.markdown("---")
    render_deployment_limitations()
    st.markdown("---")
    render_practical_obstacles()
    st.markdown("---")
    render_future_work()
    st.markdown("---")
    render_takeaway_box()


if __name__ == "__main__":
    main()