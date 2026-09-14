"""Admitra Streamlit application entry point."""

from __future__ import annotations

import logging

import streamlit as st

from config.constants import (
    APP_ICON,
    APP_TITLE,
    REGISTRY_PATH,
)
from services.registry import (
    load_registry,
    validate_registry,
)
from ui.components import (
    render_brand,
    render_disclaimer,
)
from ui.styles import inject_styles
from views import (
    operations,
    patient_workup,
    risk_assessment,
)


logging.basicConfig(
    level=logging.INFO
)

LOGGER = logging.getLogger(
    __name__
)

DEFAULT_PAGE = "Operations Console"

NAVIGATION_OPTIONS = [
    "Operations Console",
    "Patient Workup",
    "Risk Assessment",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="collapsed",
)


inject_styles()


@st.cache_data(
    show_spinner="Loading scored patient registry…"
)
def get_registry():
    return load_registry(
        REGISTRY_PATH
    )


def main() -> None:

    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

    if (
        "active_page"
        not in st.session_state
    ):
        st.session_state[
            "active_page"
        ] = DEFAULT_PAGE


    # --------------------------------------------------------
    # LOAD REGISTRY
    # --------------------------------------------------------

    try:

        registry = get_registry()

    except (
        FileNotFoundError,
        RuntimeError,
    ) as error:

        LOGGER.exception(
            "Unable to load patient registry"
        )

        st.error(
            "Admitra could not load the scored patient registry. "
            "Verify that the processed data file is available."
        )

        if st.session_state.get(
            "debug",
            False,
        ):
            st.exception(
                error
            )

        st.stop()


    # --------------------------------------------------------
    # VALIDATE REGISTRY
    # --------------------------------------------------------

    validation = validate_registry(
        registry
    )


    if not validation.is_valid:

        for error in validation.errors:

            st.error(
                f"Registry validation failed: {error}"
            )

        st.stop()


    for warning in validation.warnings:

        st.warning(
            f"Registry validation warning: {warning}"
        )


    # --------------------------------------------------------
    # TOP BAR
    # --------------------------------------------------------

    brand_col, nav_col = st.columns(
        [
            0.34,
            0.66,
        ],
        vertical_alignment="center",
    )


    with brand_col:

        render_brand()


    with nav_col:

        page = st.radio(
            "Navigation",
            NAVIGATION_OPTIONS,
            horizontal=True,
            label_visibility="collapsed",
            key="active_page",
        )


    # --------------------------------------------------------
    # PAGE DISPATCH
    # --------------------------------------------------------

    if page == "Operations Console":

        operations.render(
            registry
        )


    elif page == "Patient Workup":

        patient_workup.render(
            registry
        )


    elif page == "Risk Assessment":

        risk_assessment.render(
            registry
        )


    else:

        st.error(
            "Unknown Admitra workspace."
        )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    render_disclaimer()


if __name__ == "__main__":
    main()