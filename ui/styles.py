import streamlit as st


def inject_styles() -> None:
    """Apply Admitra's product styling without interfering with Streamlit controls."""

    st.markdown(
        """
        <style>

        :root {
            --admitra-cyan: #64c8ff;
            --admitra-blue: #4c97ff;
            --admitra-violet: #9b8cff;

            --admitra-green: #69d59a;
            --admitra-yellow: #f0c96b;
            --admitra-red: #ff7f8e;

            --admitra-text: var(--text-color);

            --admitra-muted:
                color-mix(
                    in srgb,
                    var(--text-color) 58%,
                    transparent
                );

            --admitra-faint:
                color-mix(
                    in srgb,
                    var(--text-color) 12%,
                    transparent
                );

            --admitra-panel:
                color-mix(
                    in srgb,
                    var(--secondary-background-color) 84%,
                    var(--background-color)
                );
        }


        /* ================================================
           APP
        ================================================ */

        .stApp {
            color: var(--admitra-text);
        }


        /*
        IMPORTANT:
        Do not modify, collapse, overlay, reposition, or make
        Streamlit's native header transparent.

        The previous CSS caused the native header to intercept
        clicks on Admitra's navigation.
        */


        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }


        /* ================================================
           BRAND
        ================================================ */

        .admitra-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            min-height: 2.4rem;
        }


        .admitra-brand-mark {
            display: flex;
            align-items: center;
            justify-content: center;

            width: 29px;
            height: 29px;

            border-radius: 50%;

            border:
                1px solid
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 70%,
                    transparent
                );

            color: var(--admitra-cyan);

            font-size: 0.75rem;
            font-weight: 650;
        }


        .admitra-brand-name {
            color: var(--admitra-text);
            font-size: 1.02rem;
            font-weight: 720;
            letter-spacing: -0.02em;
        }


        .admitra-brand-subtitle {
            color: var(--admitra-muted);
            font-size: 0.69rem;
            margin-left: 0.25rem;
        }


        /* ================================================
           NATIVE TOP NAVIGATION
        ================================================ */

        div[data-testid="stRadio"] {
            position: relative !important;
            z-index: auto !important;
            pointer-events: auto !important;
        }


        div[data-testid="stRadio"] * {
            pointer-events: auto !important;
        }


        div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            gap: 0.35rem;
            justify-content: flex-end;
        }


        div[data-testid="stRadio"] div[role="radiogroup"] label {
            position: relative;

            padding: 0.48rem 0.78rem;

            border-radius: 999px;

            border:
                1px solid transparent;

            transition:
                background 0.15s ease,
                border-color 0.15s ease;

            cursor: pointer !important;

            pointer-events: auto !important;
        }


        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background:
                color-mix(
                    in srgb,
                    var(--admitra-blue) 8%,
                    transparent
                );
        }


        div[data-testid="stRadio"]
        div[role="radiogroup"]
        label:has(input:checked) {
            border-color:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 45%,
                    transparent
                );

            background:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 8%,
                    transparent
                );
        }


        /* ================================================
           PAGE HEADER
        ================================================ */

        .page-hero {
            padding: 1.15rem 0 1.5rem;
            margin-bottom: 1.25rem;

            border-bottom:
                1px solid var(--admitra-faint);
        }


        .eyebrow,
        .panel-label {
            color: var(--admitra-cyan);

            font-size: 0.66rem;
            font-weight: 760;

            letter-spacing: 0.13em;

            text-transform: uppercase;
        }


        .page-title {
            margin: 0.38rem 0 0.42rem;

            color: var(--admitra-text);

            font-size:
                clamp(
                    2.1rem,
                    4vw,
                    3.5rem
                );

            font-weight: 590;

            letter-spacing: -0.055em;

            line-height: 1;
        }


        .page-description {
            max-width: 680px;

            color: var(--admitra-muted);

            font-size: 0.96rem;

            line-height: 1.5;
        }


        /* ================================================
           INLINE STATS
        ================================================ */

        .inline-stats {
            display: grid;

            grid-template-columns:
                repeat(
                    4,
                    minmax(
                        0,
                        1fr
                    )
                );

            margin: 0 0 1.8rem;

            border-top:
                1px solid var(--admitra-faint);

            border-bottom:
                1px solid var(--admitra-faint);
        }


        .inline-stat {
            padding: 1rem 1rem;

            border-right:
                1px solid var(--admitra-faint);
        }


        .inline-stat:first-child {
            padding-left: 0;
        }


        .inline-stat:last-child {
            border-right: 0;
        }


        .inline-stat-value {
            color: var(--admitra-text);

            font-size: 1.18rem;

            font-weight: 720;

            letter-spacing: -0.025em;
        }


        .inline-stat-label {
            margin-top: 0.18rem;

            color: var(--admitra-muted);

            font-size: 0.72rem;
        }


        /* ================================================
           SECTIONS
        ================================================ */

        .section-header {
            margin: 0 0 0.9rem;
        }


        .section-title {
            color: var(--admitra-text);

            font-size: 1.15rem;

            font-weight: 670;

            letter-spacing: -0.025em;
        }


        .section-description {
            color: var(--admitra-muted);

            font-size: 0.84rem;

            margin-top: 0.25rem;

            line-height: 1.45;
        }


        .section-tag {
            margin-top: 0.5rem;

            color: var(--admitra-cyan);

            font-size: 0.63rem;

            letter-spacing: 0.11em;

            text-transform: uppercase;
        }


        /* ================================================
           PANELS
        ================================================ */

        [data-testid="stVerticalBlockBorderWrapper"] {
            border:
                1px solid var(--admitra-faint);

            border-radius: 12px;

            background: var(--admitra-panel);

            box-shadow: none;
        }


        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1.15rem;
        }


        /* ================================================
           RISK
        ================================================ */

        .risk-panel {
            padding: 1rem 0;
        }


        .risk-number {
            margin: 0.38rem 0;

            color: var(--admitra-text);

            font-size:
                clamp(
                    3rem,
                    6vw,
                    5.2rem
                );

            font-weight: 560;

            letter-spacing: -0.075em;

            line-height: 0.95;
        }


        .risk-detail {
            color: var(--admitra-muted);

            font-size: 0.83rem;
        }


        /* ================================================
           PATIENT
        ================================================ */

        .patient-banner {
            padding: 0.9rem 0;

            margin-bottom: 1.3rem;

            border-top:
                1px solid var(--admitra-faint);

            border-bottom:
                1px solid var(--admitra-faint);

            color: var(--admitra-muted);
        }


        /* ================================================
           DRIVERS
        ================================================ */

        .driver-card {
            padding: 0.75rem 0;

            border-bottom:
                1px solid var(--admitra-faint);
        }


        .driver-card:last-child {
            border-bottom: 0;
        }


        .driver-feature {
            color: var(--admitra-text);

            font-size: 0.92rem;

            font-weight: 650;
        }


        .driver-value {
            margin-top: 0.18rem;

            color: var(--admitra-muted);

            font-size: 0.78rem;
        }


        .driver-up {
            color: var(--admitra-red);
        }


        .driver-down {
            color: var(--admitra-green);
        }


        /* ================================================
           BUTTONS
        ================================================ */

        div[data-testid="stButton"] > button {
            min-height: 2.4rem;

            border-radius: 999px;

            border:
                1px solid var(--admitra-faint);

            background: transparent;

            color: var(--admitra-text);

            box-shadow: none;

            font-size: 0.78rem;

            font-weight: 620;
        }


        div[data-testid="stButton"] > button:hover {
            border-color:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 55%,
                    transparent
                );
        }


        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--admitra-text);
            color: var(--background-color);
        }


        /* ================================================
           INPUTS
        ================================================ */

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input {
            border-radius: 8px;

            border-color:
                var(--admitra-faint);

            background:
                var(--admitra-panel);
        }


        /* ================================================
           DATAFRAME
        ================================================ */

        [data-testid="stDataFrame"] {
            border:
                1px solid var(--admitra-faint);

            border-radius: 10px;

            overflow: hidden;
        }


        .stCaption {
            color:
                var(--admitra-muted) !important;
        }


        /* ================================================
           MOBILE
        ================================================ */

        @media (max-width: 800px) {

            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }


            .inline-stats {
                grid-template-columns:
                    repeat(
                        2,
                        minmax(
                            0,
                            1fr
                        )
                    );
            }


            .inline-stat:nth-child(2) {
                border-right: 0;
            }


            .inline-stat:nth-child(1),
            .inline-stat:nth-child(2) {
                border-bottom:
                    1px solid var(--admitra-faint);
            }
        }


        @media (max-width: 560px) {

            .inline-stats {
                grid-template-columns: 1fr;
            }


            .inline-stat {
                border-right: 0;

                border-bottom:
                    1px solid var(--admitra-faint);

                padding-left: 0;
            }


            .inline-stat:last-child {
                border-bottom: 0;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )