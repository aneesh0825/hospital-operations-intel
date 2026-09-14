import streamlit as st


def inject_styles() -> None:
    """Apply Admitra's product styling without interfering with Streamlit controls."""

    st.markdown(
        """
        <style>

        :root {
            --admitra-cyan: #64c8ff;
            --admitra-blue: #4c97ff;
            --admitra-green: #69d59a;
            --admitra-yellow: #f0c96b;
            --admitra-red: #ff7f8e;

            --admitra-page: var(--background-color);

            --admitra-elevated:
                color-mix(
                    in srgb,
                    var(--secondary-background-color) 90%,
                    var(--background-color)
                );

            --admitra-card:
                color-mix(
                    in srgb,
                    var(--secondary-background-color) 84%,
                    var(--background-color)
                );

            --admitra-control:
                color-mix(
                    in srgb,
                    var(--secondary-background-color) 76%,
                    var(--background-color)
                );

            --admitra-field:
                color-mix(
                    in srgb,
                    var(--secondary-background-color) 96%,
                    var(--background-color)
                );

            --admitra-hover:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 8%,
                    var(--secondary-background-color)
                );

            --admitra-text-primary: var(--text-color);

            --admitra-text-secondary:
                color-mix(
                    in srgb,
                    var(--text-color) 72%,
                    transparent
                );

            --admitra-muted:
                color-mix(
                    in srgb,
                    var(--text-color) 58%,
                    transparent
                );

            --admitra-border:
                color-mix(
                    in srgb,
                    var(--text-color) 14%,
                    transparent
                );

            --admitra-border-strong:
                color-mix(
                    in srgb,
                    var(--text-color) 25%,
                    transparent
                );

            --admitra-accent-border:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 54%,
                    transparent
                );

            --admitra-text: var(--admitra-text-primary);
            --admitra-faint: var(--admitra-border);
            --admitra-panel: var(--admitra-card);
        }


        /* ================================================
           APP
        ================================================ */

        .stApp {
            background: var(--admitra-page);
            color: var(--admitra-text-primary);
            font-variant-numeric: tabular-nums;
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
            padding-top: 3.5rem;
            padding-bottom: 4rem;
        }


        /* ================================================
           BRAND
        ================================================ */

        .admitra-brand {
            display: flex;
            align-items: center;
            gap: 0.84rem;
            min-height: 2.4rem;
        }


        .admitra-brand-mark {
            display: flex;
            align-items: center;
            justify-content: center;

            width: 30px;
            height: 30px;

            border-radius: 50%;

            border: 1.5px solid color-mix(
                in srgb,
                var(--admitra-cyan) 82%,
                transparent
            );

            background:
                color-mix(
                    in srgb,
                    var(--admitra-cyan) 13%,
                    var(--admitra-elevated)
                );

            box-shadow:
                0 0 0 3px color-mix(in srgb, var(--admitra-cyan) 10%, transparent),
                0 4px 12px color-mix(in srgb, var(--admitra-cyan) 11%, transparent),
                inset 0 1px 0 color-mix(in srgb, var(--admitra-cyan) 20%, transparent);

            color: var(--admitra-cyan);

            font-size: 0.75rem;
            font-weight: 650;
        }


        .admitra-brand-name {
            color: var(--admitra-text);
            font-size: 1.02rem;
            font-weight: 790;
            letter-spacing: -0.045em;
        }


        .admitra-brand-subtitle {
            color: var(--admitra-text-secondary);
            font-size: 0.65rem;
            letter-spacing: 0.04em;
            margin-left: 0.4rem;
        }

        /* The app's existing top-row columns: brand / workspace switcher / toolbar room. */
        div[data-testid="stHorizontalBlock"]:has(.admitra-brand)
        > div[data-testid="stColumn"]:nth-child(1) {
            flex: 0 0 25% !important;
            width: 25% !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.admitra-brand)
        > div[data-testid="stColumn"]:nth-child(2) {
            flex: 0 0 50% !important;
            width: 50% !important;
        }


        /* ================================================
           NATIVE TOP NAVIGATION
        ================================================ */

        .st-key-active_page div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            gap: 0.72rem;
            justify-content: center;
            align-items: center;
            --primary-color: var(--admitra-cyan);
        }

        .st-key-active_page div[data-testid="stRadio"] {
            --primary-color: var(--admitra-cyan);
        }


        .st-key-active_page div[data-testid="stRadio"] div[role="radiogroup"] label {
            padding: 0.54rem 1rem;
            border-radius: 12px;
            border: 1px solid transparent;
            color: var(--admitra-text-secondary);
            background: transparent;
            font-size: 0.81rem;
            font-weight: 610;
            transition: border-color 170ms ease, background-color 170ms ease,
                box-shadow 170ms ease, color 170ms ease;

            cursor: pointer !important;
        }

        .st-key-active_page div[data-testid="stRadio"] input[type="radio"] {
            accent-color: var(--admitra-cyan) !important;
        }

        .st-key-active_page div[data-testid="stRadio"] input[type="radio"]:checked + div {
            background-color: var(--admitra-cyan) !important;
            border-color: var(--admitra-cyan) !important;
            box-shadow: 0 0 0 2px color-mix(in srgb, var(--admitra-cyan) 12%, transparent);
        }

        .st-key-active_page div[data-testid="stRadio"] input[type="radio"]:checked {
            filter: hue-rotate(170deg) saturate(0.85);
        }

        .st-key-active_page div[data-testid="stRadio"]
        div[role="radiogroup"] label > span:first-child {
            width: 0;
            margin: 0;
            opacity: 0;
        }

        .st-key-active_page div[data-testid="stRadio"] [role="radio"][aria-checked="true"],
        .st-key-active_page div[data-testid="stRadio"] [data-checked="true"] {
            background-color: var(--admitra-cyan) !important;
            border-color: var(--admitra-cyan) !important;
            color: var(--admitra-cyan) !important;
        }

        .st-key-active_page div[data-testid="stRadio"] label:has(input:checked) svg {
            color: var(--admitra-cyan) !important;
            fill: var(--admitra-cyan) !important;
            stroke: var(--admitra-cyan) !important;
        }


        .st-key-active_page div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: var(--admitra-hover);
            border-color: var(--admitra-border-strong);
            color: var(--admitra-text-primary);
        }


        .st-key-active_page div[data-testid="stRadio"]
        div[role="radiogroup"]
        label:has(input:checked) {
            border-color: color-mix(in srgb, var(--admitra-cyan) 68%, var(--admitra-border));
            background: color-mix(in srgb, var(--admitra-cyan) 13%, var(--admitra-elevated));
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-cyan) 20%, transparent),
                0 7px 18px color-mix(in srgb, var(--admitra-cyan) 12%, transparent);
            color: var(--admitra-text-primary);
        }

        .st-key-active_page div[data-testid="stRadio"]
        div[role="radiogroup"]
        label:has(input:checked) > div:last-child {
            color: var(--admitra-text-primary);
        }


        /* ================================================
           PAGE HEADER
        ================================================ */

        .page-hero {
            padding: 0.85rem 0 0.45rem;
            margin-bottom: 0.25rem;

            border-bottom:
                1px solid var(--admitra-faint);
        }


        .eyebrow,
        .panel-label {
            color: var(--admitra-cyan);

            font-size: 0.61rem;
            font-weight: 760;

            letter-spacing: 0.16em;

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

        .page-hero + div[data-testid="stElementContainer"] {
            margin-top: 0;
        }

        div[data-testid="stHtml"]:has(.page-hero) {
            margin-bottom: -0.9rem;
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

            gap: 0.6rem;

            margin: 0 0 1.8rem;
        }


        .inline-stat {
            min-width: 0;
            padding: 1.05rem 1.15rem 1rem;
            border: 1px solid var(--admitra-border);
            border-top: 2px solid var(--admitra-cyan);
            border-radius: 13px;
            background: var(--admitra-elevated);
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 4%, transparent),
                0 5px 16px rgba(0, 0, 0, 0.1);
            transition: border-color 170ms ease, background-color 170ms ease,
                box-shadow 170ms ease, transform 170ms ease;
        }

        .inline-stat:nth-child(2) {
            border-top-color: var(--admitra-blue);
        }

        .inline-stat:nth-child(3) {
            border-top-color: #9b8cff;
        }

        .inline-stat:nth-child(4) {
            border-top-color: var(--admitra-green);
        }

        .inline-stat:hover {
            border-color: var(--admitra-border-strong);
            background: color-mix(in srgb, var(--admitra-elevated) 94%, var(--admitra-cyan));
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 5%, transparent),
                0 8px 20px rgba(0, 0, 0, 0.14);
            transform: translateY(-1px);
        }


        .inline-stat-value {
            color: var(--admitra-text);

            font-size: clamp(1.6rem, 2vw, 1.85rem);

            font-weight: 710;

            letter-spacing: -0.045em;
            line-height: 1;
            font-variant-numeric: tabular-nums;
        }


        .inline-stat-label {
            margin-top: 0.42rem;

            color: var(--admitra-muted);

            font-size: 0.65rem;
            font-weight: 630;
            letter-spacing: 0.085em;
            line-height: 1.25;
            text-transform: uppercase;
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
            border: 1px solid color-mix(in srgb, var(--admitra-border) 82%, transparent);
            border-radius: 15px;
            background: var(--admitra-card);
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 5%, transparent),
                0 8px 30px rgba(0, 0, 0, 0.12);
        }


        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1.4rem;
        }

        .st-key-assessment_summary_card [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: color-mix(in srgb, var(--admitra-cyan) 30%, var(--admitra-border));
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-cyan) 22%, transparent),
                0 12px 32px color-mix(in srgb, var(--admitra-cyan) 8%, transparent);
        }

        .st-key-assessment_input_card [data-testid="stVerticalBlockBorderWrapper"] {
            background: color-mix(in srgb, var(--admitra-card) 94%, var(--admitra-elevated));
        }


        /* ================================================
           RISK
        ================================================ */

        .risk-panel {
            padding: 0.3rem 0 0.15rem;
        }


        .risk-number {
            margin: 0.38rem 0;

            color: var(--admitra-text-primary);

            font-size:
                clamp(
                    3rem,
                    6vw,
                    5.2rem
                );

            font-weight: 610;

            letter-spacing: -0.075em;

            line-height: 0.95;
        }


        .risk-detail {
            display: inline-flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            color: var(--admitra-text-secondary);
            font-size: 0.83rem;
            line-height: 1.5;
        }

        .risk-threshold {
            margin: 1.25rem 0 0.8rem;
        }

        .risk-threshold-track {
            position: relative;
            height: 6px;
            overflow: hidden;
            border-radius: 999px;
            background: var(--admitra-control);
            border: 1px solid var(--admitra-border);
        }

        .risk-threshold-fill {
            display: block;
            width: var(--risk-position);
            height: 100%;
            border-radius: inherit;
            background: var(--risk-color, var(--admitra-cyan));
        }

        .risk-threshold-marker {
            position: absolute;
            top: -4px;
            bottom: -4px;
            left: var(--threshold-position);
            width: 1px;
            background: var(--admitra-cyan);
            transform: translateX(-1px);
        }

        .risk-threshold-range {
            display: flex;
            justify-content: space-between;
            margin-top: 0.35rem;
            color: var(--admitra-muted);
            font-size: 0.7rem;
        }

        .risk-threshold-note {
            width: max-content;
            max-width: 100%;
            margin-top: 0.3rem;
            margin-left: var(--threshold-position);
            color: var(--admitra-cyan);
            font-size: 0.68rem;
            font-weight: 650;
            transform: translateX(-50%);
        }

        .risk-empty-state {
            padding: 0.3rem 0 0.15rem;
        }

        .risk-empty-number {
            margin: 0.42rem 0 0.18rem;
            color: var(--admitra-text-secondary);
            font-size: clamp(2.7rem, 5vw, 4.3rem);
            font-weight: 560;
            letter-spacing: -0.07em;
            line-height: 0.95;
        }

        .risk-empty-caption {
            color: var(--admitra-muted);
            font-size: 0.82rem;
        }

        .risk-low { --risk-color: var(--admitra-green); }
        .risk-moderate { --risk-color: var(--admitra-yellow); }
        .risk-high { --risk-color: var(--admitra-red); }

        .risk-low .risk-number { color: var(--admitra-green); }
        .risk-moderate .risk-number { color: var(--admitra-yellow); }
        .risk-high .risk-number { color: var(--admitra-red); }


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
            border-radius: 10px;
            border: 1px solid var(--admitra-border);
            background: var(--admitra-elevated);
            color: var(--admitra-text-primary);
            box-shadow: none;

            font-size: 0.78rem;
            font-weight: 620;
            transition: border-color 160ms ease, background-color 160ms ease,
                transform 160ms ease, box-shadow 160ms ease;
        }


        div[data-testid="stButton"] > button:hover {
            border-color: var(--admitra-accent-border);
            background: var(--admitra-hover);
            transform: translateY(-1px);
            box-shadow: 0 5px 16px rgba(0, 0, 0, 0.16);
        }

        div[data-testid="stButton"] > button:active {
            transform: translateY(0);
            box-shadow: none;
        }


        div[data-testid="stButton"] > button[kind="primary"] {
            background: var(--admitra-blue);
            border-color: var(--admitra-blue);
            color: #f7fbff;
            box-shadow: 0 6px 18px color-mix(in srgb, var(--admitra-blue) 26%, transparent);
        }

        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: var(--admitra-cyan);
            border-color: var(--admitra-cyan);
        }


        /* ================================================
           INPUTS
        ================================================ */

        div[data-testid="stSelectbox"] > div > div,
        div[data-testid="stTextInput"] input,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stNumberInput"] div[data-baseweb="input"] {
            min-height: 2.55rem;
            border-radius: 10px;
            border-color: var(--admitra-border);
            background: var(--admitra-field);
            color: var(--admitra-text-primary);
            transition: border-color 160ms ease, background-color 160ms ease,
                box-shadow 160ms ease;
        }

        div[data-testid="stSelectbox"] > div > div:hover,
        div[data-testid="stTextInput"] input:hover,
        div[data-testid="stNumberInput"] div[data-baseweb="input"]:hover {
            border-color: var(--admitra-border-strong);
        }

        div[data-testid="stSelectbox"]:focus-within > div > div,
        div[data-testid="stTextInput"]:focus-within input,
        div[data-testid="stNumberInput"]:focus-within div[data-baseweb="input"] {
            border-color: var(--admitra-accent-border);
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--admitra-cyan) 12%, transparent);
        }

        div[data-testid="stNumberInput"] button {
            border-color: var(--admitra-border) !important;
            background: var(--admitra-elevated) !important;
            color: var(--admitra-text-secondary) !important;
            transition: background-color 160ms ease, color 160ms ease;
        }

        div[data-testid="stNumberInput"] button:hover {
            background: var(--admitra-hover) !important;
            color: var(--admitra-text-primary) !important;
        }

        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: var(--admitra-text-secondary) !important;
            font-size: 0.76rem !important;
            font-weight: 620 !important;
        }

        /* Risk Assessment's existing preset radio, scoped by its container key. */
        .st-key-assessment_preset_control div[data-testid="stRadio"] div[role="radiogroup"] {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }

        .st-key-assessment_preset_control {
            margin-top: -1.25rem;
            margin-bottom: 0.15rem;
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] div[role="radiogroup"] label {
            min-width: 8.35rem;
            min-height: 2.75rem;
            padding: 0.64rem 0.95rem;
            justify-content: center;
            border: 1px solid var(--admitra-border-strong) !important;
            border-radius: 11px;
            background: color-mix(in srgb, var(--admitra-elevated) 92%, var(--text-color)) !important;
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 4%, transparent);
            color: var(--admitra-text-secondary);
            font-size: 0.8rem;
            font-weight: 630;
            transition: border-color 170ms ease, background-color 170ms ease,
                box-shadow 170ms ease, color 170ms ease, transform 170ms ease;
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            border-color: var(--admitra-border-strong);
            background: color-mix(in srgb, var(--admitra-elevated) 94%, var(--admitra-cyan));
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--text-color) 6%, transparent),
                0 5px 14px rgba(0, 0, 0, 0.12);
            color: var(--admitra-text-primary);
            transform: translateY(-1px);
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(1) input[type="radio"] {
            accent-color: var(--admitra-cyan) !important;
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(1):has(input:checked) {
            border-color: color-mix(in srgb, var(--admitra-cyan) 72%, var(--admitra-border)) !important;
            background: color-mix(in srgb, var(--admitra-cyan) 15%, var(--admitra-elevated)) !important;
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-cyan) 22%, transparent),
                0 7px 18px color-mix(in srgb, var(--admitra-cyan) 12%, transparent);
            color: var(--admitra-text-primary);
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(2):has(input:checked) {
            border-color: color-mix(in srgb, var(--admitra-green) 70%, var(--admitra-border)) !important;
            background: color-mix(in srgb, var(--admitra-green) 15%, var(--admitra-elevated)) !important;
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-green) 20%, transparent),
                0 7px 18px color-mix(in srgb, var(--admitra-green) 10%, transparent);
            color: var(--admitra-text-primary);
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(2) input[type="radio"] {
            accent-color: var(--admitra-green) !important;
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(3):has(input:checked) {
            border-color: color-mix(in srgb, var(--admitra-yellow) 72%, var(--admitra-border)) !important;
            background: color-mix(in srgb, var(--admitra-yellow) 15%, var(--admitra-elevated)) !important;
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-yellow) 20%, transparent),
                0 7px 18px color-mix(in srgb, var(--admitra-yellow) 10%, transparent);
            color: var(--admitra-text-primary);
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(3) input[type="radio"] {
            accent-color: var(--admitra-yellow) !important;
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(4):has(input:checked) {
            border-color: color-mix(in srgb, var(--admitra-red) 72%, var(--admitra-border)) !important;
            background: color-mix(in srgb, var(--admitra-red) 15%, var(--admitra-elevated)) !important;
            box-shadow: inset 0 1px 0 color-mix(in srgb, var(--admitra-red) 20%, transparent),
                0 7px 18px color-mix(in srgb, var(--admitra-red) 10%, transparent);
            color: var(--admitra-text-primary);
        }

        .st-key-assessment_preset_control div[data-testid="stRadio"] label:nth-of-type(4) input[type="radio"] {
            accent-color: var(--admitra-red) !important;
        }


        /* ================================================
           DATAFRAME
        ================================================ */

        [data-testid="stDataFrame"] {
            border:
                1px solid var(--admitra-border);

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


            .st-key-active_page div[data-testid="stRadio"] div[role="radiogroup"] {
                justify-content: flex-start;
                flex-wrap: wrap;
            }
        }


        @media (max-width: 560px) {

            .inline-stats {
                grid-template-columns: 1fr;
            }


            .inline-stat {
                padding: 1rem;
            }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
