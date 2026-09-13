import altair as alt
import pandas as pd
import streamlit as st

from src.predict_readmission import (
    explain_readmission,
    predict_readmission,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Admitra | Hospital Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# Uses Streamlit's native System / Light / Dark setting.
# ============================================================

st.html(
    """
<style>

:root {
    --admitra-cyan: #38d9d0;
    --admitra-blue: #5ea8ff;
    --admitra-violet: #9b8cff;
    --admitra-pink: #ec7fb5;

    --admitra-green: #58d68d;
    --admitra-yellow: #f3c95f;
    --admitra-red: #ff7485;

    --admitra-bg: var(--background-color);
    --admitra-panel: var(--secondary-background-color);
    --admitra-text: var(--text-color);
    --admitra-primary: var(--primary-color);

    --admitra-border:
        color-mix(
            in srgb,
            var(--text-color) 14%,
            transparent
        );

    --admitra-border-soft:
        color-mix(
            in srgb,
            var(--text-color) 8%,
            transparent
        );

    --admitra-muted:
        color-mix(
            in srgb,
            var(--text-color) 58%,
            transparent
        );

    --admitra-muted-2:
        color-mix(
            in srgb,
            var(--text-color) 40%,
            transparent
        );

    --admitra-panel-strong:
        color-mix(
            in srgb,
            var(--secondary-background-color) 90%,
            var(--background-color)
        );
}


/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 78% -5%,
            color-mix(
                in srgb,
                var(--admitra-blue) 14%,
                transparent
            ),
            transparent 30%
        ),
        radial-gradient(
            circle at 15% 18%,
            color-mix(
                in srgb,
                var(--admitra-cyan) 9%,
                transparent
            ),
            transparent 24%
        ),
        radial-gradient(
            circle at 90% 58%,
            color-mix(
                in srgb,
                var(--admitra-violet) 8%,
                transparent
            ),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 88%,
            color-mix(
                in srgb,
                var(--admitra-pink) 5%,
                transparent
            ),
            transparent 23%
        ),
        var(--admitra-bg);

    color: var(--admitra-text);
}


.block-container {
    max-width: 1500px;
    padding-top: 1.15rem;
    padding-bottom: 3rem;
}


/* Keep native toolbar/menu so System / Light / Dark remains available */

header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

[data-testid="stToolbar"] {
    background: transparent !important;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    width: 285px !important;

    background:
        linear-gradient(
            180deg,
            color-mix(
                in srgb,
                var(--secondary-background-color) 93%,
                var(--admitra-blue)
            ),
            var(--secondary-background-color)
        );

    border-right: 1px solid var(--admitra-border);
}


section[data-testid="stSidebar"] > div {
    padding-top: 1.35rem;
    padding-left: 1rem;
    padding-right: 1rem;
}


.sidebar-brand {
    padding: 0.15rem 0 1.05rem 0;
    border-bottom: 1px solid var(--admitra-border);
    margin-bottom: 0.95rem;
}


.sidebar-logo {
    font-size: 1.52rem;
    font-weight: 900;
    letter-spacing: 0.18em;
    color: var(--admitra-text) !important;
    -webkit-text-fill-color: var(--admitra-text) !important;
}

background:
    linear-gradient(
        90deg,
        var(--admitra-text),
        var(--admitra-blue),
        var(--admitra-cyan)
    );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}


.sidebar-subtitle {
    color: var(--admitra-muted);
    font-size: 0.68rem;
    margin-top: 0.24rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}


.sidebar-label {
    color: var(--admitra-muted-2);
    font-size: 0.61rem;
    font-weight: 780;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin: 0.75rem 0 0.45rem 0.05rem;
}


section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.22rem;
}


section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.68rem 0.74rem;
    border-radius: 9px;
    transition: 0.15s ease;
    color: var(--admitra-text) !important;
}


section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background:
        color-mix(
            in srgb,
            var(--admitra-cyan) 8%,
            transparent
        );
}


section[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked) {
    background:
        linear-gradient(
            90deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 21%,
                transparent
            ),
            color-mix(
                in srgb,
                var(--admitra-violet) 13%,
                transparent
            )
        );

    border:
        1px solid
        color-mix(
            in srgb,
            var(--admitra-blue) 30%,
            transparent
        );

    box-shadow:
        inset 3px 0 0
        var(--admitra-cyan);
}


/* ==========================================================
   MODEL CARD
========================================================== */

.model-card {
    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 9%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-violet) 5%,
                var(--admitra-panel)
            )
        );

    border: 1px solid var(--admitra-border);
    border-radius: 12px;
    padding: 0.92rem;
}


.model-title {
    color: var(--admitra-text);
    font-size: 0.8rem;
    font-weight: 760;
}


.model-description {
    color: var(--admitra-muted);
    font-size: 0.68rem;
    margin-top: 0.3rem;
    line-height: 1.45;
}


.active-pill {
    display: inline-block;
    margin-top: 0.65rem;

    color: var(--admitra-green);

    background:
        color-mix(
            in srgb,
            var(--admitra-green) 10%,
            transparent
        );

    border:
        1px solid
        color-mix(
            in srgb,
            var(--admitra-green) 23%,
            transparent
        );

    padding: 0.24rem 0.5rem;
    border-radius: 999px;
    font-size: 0.61rem;
    font-weight: 800;
}


.sidebar-bottom {
    margin-top: 1.25rem;
    color: var(--admitra-muted-2);
    font-size: 0.64rem;
    line-height: 1.55;
}


/* ==========================================================
   PAGE HERO
========================================================== */

.page-hero {
    position: relative;
    overflow: hidden;

    border-radius: 18px;
    padding: 1.45rem 1.5rem;
    margin-bottom: 1.25rem;

    border: 1px solid var(--admitra-border);

    background:
        linear-gradient(
            125deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 18%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-violet) 12%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-cyan) 6%,
                var(--admitra-panel)
            )
        );

    box-shadow:
        0 18px 42px
        color-mix(
            in srgb,
            #000000 13%,
            transparent
        );
}


.page-hero::after {
    content: "";

    position: absolute;
    width: 360px;
    height: 360px;

    right: -150px;
    top: -200px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            color-mix(
                in srgb,
                var(--admitra-cyan) 24%,
                transparent
            ),
            color-mix(
                in srgb,
                var(--admitra-blue) 10%,
                transparent
            ),
            transparent 70%
        );

    pointer-events: none;
}


.page-hero::before {
    content: "";

    position: absolute;

    width: 230px;
    height: 230px;

    left: 42%;
    bottom: -190px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            color-mix(
                in srgb,
                var(--admitra-pink) 12%,
                transparent
            ),
            transparent 70%
        );

    pointer-events: none;
}


.page-kicker {
    color: var(--admitra-cyan);
    font-size: 0.66rem;
    font-weight: 820;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}


.page-title {
    color: var(--admitra-text);
    font-size: 1.9rem;
    font-weight: 850;
    margin-top: 0.25rem;
    letter-spacing: -0.02em;
}


.page-description {
    color: var(--admitra-muted);
    font-size: 0.86rem;
    margin-top: 0.38rem;
}


/* ==========================================================
   METRIC GRID
========================================================== */

.metric-grid {
    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(
                0,
                1fr
            )
        );

    gap: 0.85rem;
    margin-bottom: 1.25rem;
}


.metric-card {
    position: relative;
    overflow: hidden;

    border-radius: 14px;
    min-height: 128px;
    padding: 1rem 1.05rem;

    border: 1px solid var(--admitra-border);

    box-shadow:
        0 12px 28px
        color-mix(
            in srgb,
            #000000 9%,
            transparent
        );
}


.metric-card::before {
    content: "";
    position: absolute;

    left: 0;
    top: 0;

    width: 4px;
    height: 100%;

    background:
        var(
            --metric-accent,
            var(--admitra-cyan)
        );
}


.metric-blue {
    --metric-accent: var(--admitra-blue);

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 19%,
                var(--admitra-panel)
            ),
            var(--admitra-panel)
        );
}


.metric-violet {
    --metric-accent: var(--admitra-violet);

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-violet) 18%,
                var(--admitra-panel)
            ),
            var(--admitra-panel)
        );
}


.metric-cyan {
    --metric-accent: var(--admitra-cyan);

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-cyan) 16%,
                var(--admitra-panel)
            ),
            var(--admitra-panel)
        );
}


.metric-pink {
    --metric-accent: var(--admitra-pink);

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-pink) 15%,
                var(--admitra-panel)
            ),
            var(--admitra-panel)
        );
}


.metric-label {
    color: var(--admitra-muted);
    font-size: 0.65rem;
    font-weight: 740;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}


.metric-value {
    color: var(--admitra-text);
    font-size: 1.72rem;
    font-weight: 850;
    margin-top: 0.4rem;
}


.metric-note {
    color: var(--admitra-muted-2);
    font-size: 0.71rem;
    margin-top: 0.28rem;
}


/* ==========================================================
   SECTION HEADER
========================================================== */

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 1rem;

    margin-top: 0.65rem;
    margin-bottom: 0.65rem;
}


.section-title {
    color: var(--admitra-text);
    font-size: 1.03rem;
    font-weight: 790;
}


.section-description {
    color: var(--admitra-muted);
    font-size: 0.75rem;
    margin-top: 0.16rem;
}


.section-tag {
    color: var(--admitra-muted-2);
    font-size: 0.62rem;
    font-family: Consolas, monospace;
    letter-spacing: 0.04em;
}


/* ==========================================================
   PANELS
========================================================== */

.panel-card {
    border: 1px solid var(--admitra-border);
    border-radius: 14px;
    padding: 1rem;

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 6%,
                var(--admitra-panel-strong)
            ),
            var(--admitra-panel)
        );

    box-shadow:
        0 12px 26px
        color-mix(
            in srgb,
            #000000 7%,
            transparent
        );
}


.review-row {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 0.72rem 0;

    border-bottom:
        1px solid var(--admitra-border-soft);
}


.review-row:last-child {
    border-bottom: none;
}


.review-label {
    color: var(--admitra-text);
    font-size: 0.79rem;
}


.review-sub {
    color: var(--admitra-muted);
    font-size: 0.68rem;
    margin-top: 0.14rem;
}


.review-value {
    color: var(--admitra-text);
    font-size: 1rem;
    font-weight: 780;
}


/* ==========================================================
   ADMITRA RISK SIGNAL
========================================================== */

.risk-panel {
    border: 1px solid var(--admitra-border);
    border-radius: 14px;
    padding: 1rem 1.05rem;

    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-cyan) 9%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-blue) 5%,
                var(--admitra-panel)
            )
        );
}


.risk-headline {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 1rem;
}


.risk-label {
    color: var(--admitra-muted);
    font-size: 0.64rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}


.risk-number {
    color: var(--admitra-text);
    font-size: 2.7rem;
    font-weight: 880;
    line-height: 1;
    margin-top: 0.28rem;
    letter-spacing: -0.04em;
}


.risk-status {
    font-size: 0.64rem;
    font-weight: 820;
    text-transform: uppercase;
    letter-spacing: 0.08em;

    padding: 0.33rem 0.55rem;
    border-radius: 999px;

    border: 1px solid var(--admitra-border);
    background: var(--admitra-panel-strong);
}


.risk-status.low {
    color: var(--admitra-green);
}


.risk-status.moderate {
    color: var(--admitra-yellow);
}


.risk-status.high {
    color: var(--admitra-red);
}


.risk-track {
    position: relative;

    height: 10px;
    margin-top: 0.92rem;
    border-radius: 999px;

    background:
        linear-gradient(
            90deg,
            color-mix(
                in srgb,
                var(--admitra-green) 78%,
                transparent
            )
            0 35%,

            color-mix(
                in srgb,
                var(--admitra-yellow) 80%,
                transparent
            )
            35% 65%,

            color-mix(
                in srgb,
                var(--admitra-red) 80%,
                transparent
            )
            65% 100%
        );
}


.risk-marker {
    position: absolute;

    top: -5px;

    width: 3px;
    height: 20px;

    border-radius: 999px;

    background: var(--admitra-text);

    box-shadow:
        0 0 0 3px
        color-mix(
            in srgb,
            var(--admitra-text) 12%,
            transparent
        );
}


.threshold-marker {
    position: absolute;

    top: -3px;

    width: 1px;
    height: 16px;

    background: var(--admitra-text);

    opacity: 0.42;
}


.risk-scale {
    display: flex;
    justify-content: space-between;

    margin-top: 0.3rem;

    color: var(--admitra-muted-2);

    font-size: 0.58rem;
    font-family: Consolas, monospace;
}


/* ==========================================================
   DRIVER CARDS
========================================================== */

.driver-card {
    background:
        linear-gradient(
            145deg,
            color-mix(
                in srgb,
                var(--admitra-violet) 5%,
                var(--admitra-panel-strong)
            ),
            var(--admitra-panel)
        );

    border: 1px solid var(--admitra-border);
    border-radius: 12px;

    padding: 0.88rem 0.95rem;
    margin-bottom: 0.55rem;
}


.driver-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}


.driver-feature {
    color: var(--admitra-text);
    font-size: 0.82rem;
    font-weight: 740;
}


.driver-up {
    color: var(--admitra-red);
    font-size: 0.64rem;
    font-weight: 800;
}


.driver-down {
    color: var(--admitra-green);
    font-size: 0.64rem;
    font-weight: 800;
}


.driver-value {
    color: var(--admitra-text);
    font-size: 0.76rem;
    font-weight: 650;
    margin-top: 0.36rem;
}


.driver-reasoning {
    color: var(--admitra-muted);
    font-size: 0.72rem;
    line-height: 1.48;
    margin-top: 0.3rem;
}


/* ==========================================================
   PATIENT BANNER
========================================================== */

.patient-banner {
    border: 1px solid var(--admitra-border);
    border-left: 4px solid var(--admitra-cyan);

    border-radius: 12px;

    background:
        linear-gradient(
            100deg,
            color-mix(
                in srgb,
                var(--admitra-cyan) 10%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-blue) 5%,
                var(--admitra-panel)
            )
        );

    padding: 0.95rem 1rem;
    margin-bottom: 0.8rem;
}


.patient-id {
    color: var(--admitra-text);
    font-size: 1rem;
    font-family: Consolas, monospace;
    font-weight: 760;
}


.patient-meta {
    color: var(--admitra-muted);
    font-size: 0.69rem;
    margin-top: 0.2rem;
}


/* ==========================================================
   STREAMLIT ELEMENTS
========================================================== */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color:
        var(--admitra-panel) !important;

    border-color:
        var(--admitra-border) !important;

    border-radius:
        9px !important;
}


label {
    color:
        var(--admitra-text) !important;
}


.stButton > button {
    min-height: 2.85rem;

    border-radius:
        9px !important;

    border:
        1px solid
        color-mix(
            in srgb,
            var(--admitra-blue) 28%,
            transparent
        );

    background:
        linear-gradient(
            90deg,
            color-mix(
                in srgb,
                var(--admitra-blue) 22%,
                var(--admitra-panel)
            ),
            color-mix(
                in srgb,
                var(--admitra-violet) 15%,
                var(--admitra-panel)
            )
        );

    color:
        var(--admitra-text);

    font-weight:
        760;
}


.stButton > button:hover {
    border-color:
        var(--admitra-blue);

    box-shadow:
        0 0 0 1px
        color-mix(
            in srgb,
            var(--admitra-blue) 20%,
            transparent
        );
}


div[data-testid="stDataFrame"] {
    border:
        1px solid var(--admitra-border);

    border-radius:
        11px;

    overflow:
        hidden;
}


.disclaimer {
    margin-top: 2rem;
    padding-top: 1rem;

    border-top:
        1px solid var(--admitra-border);

    color:
        var(--admitra-muted-2);

    font-size:
        0.66rem;
}


/* ==========================================================
   MOBILE
========================================================== */

@media (max-width: 900px) {

    .block-container {
        padding-left: 0.8rem;
        padding-right: 0.8rem;
        padding-top: 0.8rem;
    }


    .metric-grid {
        grid-template-columns:
            repeat(
                2,
                minmax(
                    0,
                    1fr
                )
            );
    }


    .risk-headline {
        display: block;
    }


    .risk-status {
        display: inline-block;
        margin-top: 0.7rem;
    }


    section[data-testid="stSidebar"] {
        width:
            min(
                88vw,
                300px
            ) !important;

        min-width:
            min(
                88vw,
                300px
            ) !important;
    }
}

</style>
"""
)


# ============================================================
# LOAD REGISTRY
# ============================================================

@st.cache_data
def load_registry():

    return pd.read_csv(
        "data/processed/admitra_patient_registry.csv"
    )


registry = load_registry()


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_number(
    value,
    fallback,
):

    if pd.isna(value):
        return fallback

    return value


def short_patient_id(
    patient_id,
):

    return (
        "P-"
        + str(patient_id)[-6:].upper()
    )


def short_encounter_id(
    encounter_id,
):

    return (
        "E-"
        + str(encounter_id)[-6:].upper()
    )


def condition_summary(
    row,
):

    conditions = []


    if row.get(
        "diabetes",
        0,
    ) == 1:

        conditions.append(
            "Diabetes"
        )


    if row.get(
        "hypertension",
        0,
    ) == 1:

        conditions.append(
            "Hypertension"
        )


    if row.get(
        "kidney_disease",
        0,
    ) == 1:

        conditions.append(
            "Kidney Disease"
        )


    if not conditions:

        return "None flagged"


    return ", ".join(
        conditions
    )


# ============================================================
# UI HELPERS
# ============================================================

def page_header(
    kicker,
    title,
    description,
):

    st.html(
        (
            '<div class="page-hero">'

            f'<div class="page-kicker">'
            f'{kicker}'
            f'</div>'

            f'<div class="page-title">'
            f'{title}'
            f'</div>'

            f'<div class="page-description">'
            f'{description}'
            f'</div>'

            '</div>'
        )
    )


def section_header(
    title,
    description,
    tag="",
):

    st.html(
        (
            '<div class="section-header">'

            '<div>'

            f'<div class="section-title">'
            f'{title}'
            f'</div>'

            f'<div class="section-description">'
            f'{description}'
            f'</div>'

            '</div>'

            f'<div class="section-tag">'
            f'{tag}'
            f'</div>'

            '</div>'
        )
    )


def metric_grid(
    items,
):

    html = ""


    for (
        label,
        value,
        note,
        style_class,
    ) in items:

        html += (
            f'<div class="metric-card {style_class}">'

            f'<div class="metric-label">'
            f'{label}'
            f'</div>'

            f'<div class="metric-value">'
            f'{value}'
            f'</div>'

            f'<div class="metric-note">'
            f'{note}'
            f'</div>'

            '</div>'
        )


    st.html(
        (
            '<div class="metric-grid">'
            f'{html}'
            '</div>'
        )
    )


# ============================================================
# MODEL INPUT
# ============================================================

def row_to_model_input(
    row,
):

    return {

        "age_at_admission":
            row[
                "age_at_admission"
            ],

        "length_of_stay":
            row[
                "length_of_stay"
            ],

        "previous_inpatient_admissions":
            row[
                "previous_inpatient_admissions"
            ],

        "admissions_last_30_days":
            row[
                "admissions_last_30_days"
            ],

        "admissions_last_90_days":
            row[
                "admissions_last_90_days"
            ],

        "admissions_last_365_days":
            row[
                "admissions_last_365_days"
            ],

        "days_since_last_inpatient_admission":
            row[
                "days_since_last_inpatient_admission"
            ],

        "has_prior_inpatient_admission":
            row[
                "has_prior_inpatient_admission"
            ],

        "prior_readmissions_last_365_days":
            row[
                "prior_readmissions_last_365_days"
            ],

        "condition_count":
            row[
                "condition_count"
            ],

        "medication_count":
            row[
                "medication_count"
            ],

        "procedure_count":
            row[
                "procedure_count"
            ],

        "diabetes":
            row[
                "diabetes"
            ],

        "hypertension":
            row[
                "hypertension"
            ],

        "kidney_disease":
            row[
                "kidney_disease"
            ],

        "bmi":
            row[
                "bmi"
            ],
    }


# ============================================================
# DRIVER CONFIGURATION
# ============================================================

DRIVER_EXPLANATIONS = {

    "Admissions in last 30 days":
        (
            "Very recent hospitalization history can indicate "
            "an active pattern of repeated inpatient care."
        ),

    "Admissions in last 90 days":
        (
            "Recent inpatient utilization is one of the "
            "strongest signals associated with 30-day "
            "readmission in this model."
        ),

    "Admissions in last 365 days":
        (
            "Repeated hospital admissions over the previous "
            "year indicate sustained inpatient utilization."
        ),

    "Prior readmissions in last 365 days":
        (
            "Previous confirmed readmissions provide direct "
            "evidence of a recent pattern of returning to "
            "inpatient care."
        ),

    "Days since last inpatient admission":
        (
            "A more recent prior hospitalization can increase "
            "modeled readmission risk because the patient "
            "returned to inpatient care within a shorter interval."
        ),

    "Prior inpatient history":
        (
            "The model considers whether the patient has any "
            "documented inpatient history before the current encounter."
        ),

    "Previous inpatient admissions":
        (
            "The patient's cumulative history of inpatient "
            "admissions contributes additional context about "
            "long-term healthcare utilization."
        ),

    "Condition burden":
        (
            "The number of documented health conditions "
            "contributes to the model's estimate of overall "
            "clinical complexity."
        ),

    "Procedure count":
        (
            "The number of procedures contributes information "
            "about the complexity of the patient's recent course of care."
        ),

    "Medication count":
        (
            "The number of medications contributes information "
            "about treatment complexity."
        ),

    "Age":
        (
            "Age contributes to the prediction based on patterns "
            "learned across patients of different ages."
        ),

    "Length of stay":
        (
            "The duration of the current hospitalization contributes "
            "to the model's assessment of the encounter."
        ),

    "Diabetes":
        (
            "Diabetes status contributes to the patient's "
            "overall clinical profile."
        ),

    "Hypertension":
        (
            "Hypertension status contributes to the patient's "
            "overall clinical profile."
        ),

    "Kidney disease":
        (
            "Kidney disease status contributes to the patient's "
            "overall clinical profile."
        ),

    "BMI":
        (
            "BMI contributes to the prediction based on patterns "
            "observed across the synthetic training population."
        ),
}


DRIVER_UNITS = {

    "Age":
        "years",

    "Length of stay":
        "days",

    "Previous inpatient admissions":
        "prior admissions",

    "Admissions in last 30 days":
        "admissions",

    "Admissions in last 90 days":
        "admissions",

    "Admissions in last 365 days":
        "admissions",

    "Days since last inpatient admission":
        "days",

    "Prior readmissions in last 365 days":
        "prior readmissions",

    "Condition burden":
        "documented conditions",

    "Medication count":
        "medications",

    "Procedure count":
        "procedures",
}


BINARY_FEATURES = {
    "Prior inpatient history",
    "Diabetes",
    "Hypertension",
    "Kidney disease",
}


def render_drivers(
    drivers,
):

    for driver in drivers:

        feature = driver[
            "feature"
        ]

        direction = driver[
            "direction"
        ]


        if direction == "increases risk":

            direction_class = (
                "driver-up"
            )

            direction_label = (
                "↑ PUSHED HIGHER"
            )


        elif direction == "decreases risk":

            direction_class = (
                "driver-down"
            )

            direction_label = (
                "↓ PUSHED LOWER"
            )


        else:

            direction_class = ""

            direction_label = (
                "NEUTRAL"
            )


        raw_value = driver[
            "value"
        ]


        if feature in BINARY_FEATURES:

            try:

                display_value = (
                    "Present"
                    if float(
                        raw_value
                    )
                    == 1
                    else "Not present"
                )

            except Exception:

                display_value = str(
                    raw_value
                )


        else:

            try:

                number = round(
                    float(
                        raw_value
                    ),
                    1,
                )

            except Exception:

                number = (
                    raw_value
                )


            unit = DRIVER_UNITS.get(
                feature,
                "",
            )


            display_value = (
                f"{number} {unit}"
                .strip()
            )


        explanation = DRIVER_EXPLANATIONS.get(
            feature,
            (
                "This factor influenced the prediction "
                "based on patterns learned by the model."
            ),
        )


        st.html(
            (
                '<div class="driver-card">'

                '<div class="driver-top">'

                f'<div class="driver-feature">'
                f'{feature}'
                f'</div>'

                f'<div class="{direction_class}">'
                f'{direction_label}'
                f'</div>'

                '</div>'

                f'<div class="driver-value">'
                f'{display_value}'
                f'</div>'

                f'<div class="driver-reasoning">'
                f'{explanation}'
                f'</div>'

                '</div>'
            )
        )


# ============================================================
# RISK SIGNAL
# ============================================================

def risk_signal(
    probability,
    risk_level,
    threshold_percent,
    review_recommended,
):

    probability = float(
        probability
    )

    threshold_percent = float(
        threshold_percent
    )


    marker_left = max(
        0.0,
        min(
            100.0,
            probability,
        ),
    )


    threshold_left = max(
        0.0,
        min(
            100.0,
            threshold_percent,
        ),
    )


    status_class = (
        str(
            risk_level
        )
        .lower()
    )


    review_text = (
        "REVIEW REQUIRED"
        if review_recommended
        else "NO REVIEW FLAG"
    )


    st.html(
        f"""
<div class="risk-panel">

    <div class="risk-headline">

        <div>

            <div class="risk-label">
                Calibrated 30-Day Readmission Risk
            </div>

            <div class="risk-number">
                {probability:.1f}%
            </div>

        </div>

        <div class="risk-status {status_class}">
            {risk_level} · {review_text}
        </div>

    </div>

    <div class="risk-track">

        <div
            class="risk-marker"
            style="
                left:
                calc(
                    {marker_left:.2f}%
                    - 1px
                )
            "
        ></div>

        <div
            class="threshold-marker"
            style="
                left:
                {threshold_left:.2f}%
            "
        ></div>

    </div>

    <div class="risk-scale">

        <span>
            LOW
        </span>

        <span>
            REVIEW THRESHOLD
            {threshold_percent:.0f}%
        </span>

        <span>
            HIGH
        </span>

    </div>

</div>
"""
    )


# ============================================================
# PRESET EXAMPLES
# ============================================================

def get_registry_example(
    risk_level,
    target_probability,
):

    candidates = registry[
        registry[
            "risk_level"
        ]
        == risk_level
    ].copy()


    candidates[
        "distance_from_target"
    ] = (
        candidates[
            "readmission_probability_percent"
        ]
        - target_probability
    ).abs()


    return (
        candidates
        .sort_values(
            "distance_from_target"
        )
        .iloc[
            0
        ]
    )


def row_to_defaults(
    row,
):

    return {

        "age":
            int(
                safe_number(
                    row[
                        "age_at_admission"
                    ],
                    65,
                )
            ),

        "los":
            float(
                safe_number(
                    row[
                        "length_of_stay"
                    ],
                    5.0,
                )
            ),

        "previous_admissions":
            int(
                safe_number(
                    row[
                        "previous_inpatient_admissions"
                    ],
                    0,
                )
            ),

        "admissions_30":
            int(
                safe_number(
                    row[
                        "admissions_last_30_days"
                    ],
                    0,
                )
            ),

        "admissions_90":
            int(
                safe_number(
                    row[
                        "admissions_last_90_days"
                    ],
                    0,
                )
            ),

        "admissions_365":
            int(
                safe_number(
                    row[
                        "admissions_last_365_days"
                    ],
                    0,
                )
            ),

        "days_since":
            float(
                safe_number(
                    row[
                        "days_since_last_inpatient_admission"
                    ],
                    365,
                )
            ),

        "prior_readmissions":
            int(
                safe_number(
                    row[
                        "prior_readmissions_last_365_days"
                    ],
                    0,
                )
            ),

        "conditions":
            int(
                safe_number(
                    row[
                        "condition_count"
                    ],
                    0,
                )
            ),

        "medications":
            int(
                safe_number(
                    row[
                        "medication_count"
                    ],
                    0,
                )
            ),

        "procedures":
            int(
                safe_number(
                    row[
                        "procedure_count"
                    ],
                    0,
                )
            ),

        "diabetes":
            bool(
                row[
                    "diabetes"
                ]
            ),

        "hypertension":
            bool(
                row[
                    "hypertension"
                ]
            ),

        "kidney_disease":
            bool(
                row[
                    "kidney_disease"
                ]
            ),

        "bmi":
            float(
                safe_number(
                    row[
                        "bmi"
                    ],
                    28.0,
                )
            ),

        "source_probability":
            float(
                row[
                    "readmission_probability_percent"
                ]
            ),

        "source_risk_level":
            row[
                "risk_level"
            ],
    }


low_example = row_to_defaults(
    get_registry_example(
        "LOW",
        1.5,
    )
)


moderate_example = row_to_defaults(
    get_registry_example(
        "MODERATE",
        12.0,
    )
)


high_example = row_to_defaults(
    get_registry_example(
        "HIGH",
        55.0,
    )
)


# ============================================================
# SIDEBAR
# ============================================================

review_count_sidebar = int(
    registry[
        "intervention_recommended"
    ].sum()
)


st.sidebar.html(
    """
<div class="sidebar-brand">

    <div class="sidebar-logo">
        ADMITRA
    </div>

    <div class="sidebar-subtitle">
        Hospital Intelligence Platform
    </div>

</div>
"""
)


st.sidebar.html(
    '<div class="sidebar-label">Workspace</div>'
)


page = st.sidebar.radio(
    "Workspace",
    [
        "Operations Console",
        "Patient Workup",
        "Risk Assessment",
    ],
    label_visibility="collapsed",
)


st.sidebar.html(
    '<div class="sidebar-label">System</div>'
)


st.sidebar.html(
    (
        '<div class="model-card">'

        '<div class="model-title">'
        'Readmission Intelligence'
        '</div>'

        '<div class="model-description">'
        'History-aware calibrated XGBoost model'
        '</div>'

        '<div class="active-pill">'
        '● MODEL ACTIVE'
        '</div>'

        '</div>'
    )
)


st.sidebar.html(
    (
        '<div class="sidebar-bottom">'

        f'{review_count_sidebar:,} encounters in review queue'

        '<br><br>'

        'Synthetic Synthea environment'

        '<br>'

        'Portfolio demonstration'

        '</div>'
    )
)


# ============================================================
# OPERATIONS CONSOLE
# ============================================================

if page == "Operations Console":

    total_hospitalizations = len(
        registry
    )


    actual_readmissions = int(
        registry[
            "actual_readmitted_30_days"
        ].sum()
    )


    readmission_rate = (
        actual_readmissions
        / total_hospitalizations
        * 100
    )


    intervention_count = int(
        registry[
            "intervention_recommended"
        ].sum()
    )


    intervention_rate = (
        intervention_count
        / total_hospitalizations
        * 100
    )


    high_risk_count = int(
        (
            registry[
                "risk_level"
            ]
            == "HIGH"
        ).sum()
    )


    high_risk_rate = (
        high_risk_count
        / total_hospitalizations
        * 100
    )


    average_risk = float(
        registry[
            "readmission_probability_percent"
        ].mean()
    )


    not_flagged_count = (
        total_hospitalizations
        - intervention_count
    )


    page_header(
        "OPERATIONS CONSOLE",
        "Readmission Operations",
        (
            "Monitor population risk, review workload, "
            "and priority patients across the synthetic "
            "hospital population."
        ),
    )


    metric_grid(
        [
            (
                "Hospitalizations",
                f"{total_hospitalizations:,}",
                "Scored inpatient encounters",
                "metric-blue",
            ),

            (
                "Observed Readmission",
                f"{readmission_rate:.1f}%",
                f"{actual_readmissions:,} observed events",
                "metric-violet",
            ),

            (
                "Review Queue",
                f"{intervention_count:,}",
                f"{intervention_rate:.1f}% of encounters",
                "metric-cyan",
            ),

            (
                "High Risk",
                f"{high_risk_count:,}",
                f"{high_risk_rate:.1f}% of encounters",
                "metric-pink",
            ),
        ]
    )


    section_header(
        "Population Overview",
        (
            "Risk distribution and review demand "
            "across scored encounters."
        ),
        "POPULATION",
    )


    chart_col, review_col = st.columns(
        [
            1.45,
            0.55,
        ],
        gap="large",
    )


    with chart_col:

        risk_counts = (
            registry[
                "risk_level"
            ]
            .value_counts()
            .reindex(
                [
                    "LOW",
                    "MODERATE",
                    "HIGH",
                ]
            )
            .fillna(
                0
            )
            .astype(
                int
            )
        )


        risk_df = pd.DataFrame(
            {
                "Risk Tier":
                    risk_counts.index,

                "Encounters":
                    risk_counts.values,
            }
        )


        risk_chart = (
            alt.Chart(
                risk_df
            )
            .mark_bar(
                cornerRadiusEnd=7,
                size=34,
            )
            .encode(

                y=alt.Y(
                    "Risk Tier:N",
                    sort=[
                        "LOW",
                        "MODERATE",
                        "HIGH",
                    ],
                    title=None,
                ),

                x=alt.X(
                    "Encounters:Q",
                    title=None,
                ),

                color=alt.Color(
                    "Risk Tier:N",

                    scale=alt.Scale(
                        domain=[
                            "LOW",
                            "MODERATE",
                            "HIGH",
                        ],

                        range=[
                            "#58d68d",
                            "#f3c95f",
                            "#ff7485",
                        ],
                    ),

                    legend=None,
                ),

                tooltip=[
                    "Risk Tier",

                    alt.Tooltip(
                        "Encounters:Q",
                        format=",",
                    ),
                ],
            )
            .properties(
                height=260,
            )
            .configure_view(
                strokeOpacity=0,
            )
        )


        st.altair_chart(
            risk_chart,
            width="stretch",
        )


    with review_col:

        st.html(
            (
                '<div class="panel-card">'

                '<div class="review-row">'

                '<div>'

                '<div class="review-label">'
                'Review recommended'
                '</div>'

                f'<div class="review-sub">'
                f'{intervention_rate:.1f}% of encounters'
                f'</div>'

                '</div>'

                f'<div class="review-value">'
                f'{intervention_count:,}'
                f'</div>'

                '</div>'


                '<div class="review-row">'

                '<div>'

                '<div class="review-label">'
                'Not flagged'
                '</div>'

                f'<div class="review-sub">'
                f'{100 - intervention_rate:.1f}% of encounters'
                f'</div>'

                '</div>'

                f'<div class="review-value">'
                f'{not_flagged_count:,}'
                f'</div>'

                '</div>'


                '<div class="review-row">'

                '<div>'

                '<div class="review-label">'
                'Average modeled risk'
                '</div>'

                '<div class="review-sub">'
                'Across scored encounters'
                '</div>'

                '</div>'

                f'<div class="review-value">'
                f'{average_risk:.1f}%'
                f'</div>'

                '</div>'


                '</div>'
            )
        )


    st.write("")


    section_header(
        "Priority Patient Queue",
        (
            "Filter and review the highest-risk "
            "encounter for each patient."
        ),
        "TOP 20",
    )


    filter_col1, filter_col2, filter_col3 = st.columns(
        [
            0.30,
            0.30,
            0.40,
        ],
        gap="medium",
    )


    with filter_col1:

        risk_filter = st.selectbox(
            "Risk Tier",
            [
                "All Risk Tiers",
                "HIGH",
                "MODERATE",
                "LOW",
            ],
        )


    with filter_col2:

        review_filter = st.selectbox(
            "Review Status",
            [
                "All Review Statuses",
                "Recommended",
                "Not Flagged",
            ],
        )


    with filter_col3:

        patient_search = st.text_input(
            "Patient Search",
            placeholder="Search patient ID...",
        )


    unique_priority = (
        registry
        .sort_values(
            "readmission_probability",
            ascending=False,
        )
        .drop_duplicates(
            subset="patient_id",
            keep="first",
        )
        .copy()
    )


    if risk_filter != "All Risk Tiers":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "risk_level"
                ]
                == risk_filter
            ]
        )


    if review_filter == "Recommended":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "intervention_recommended"
                ]
                == True
            ]
        )


    elif review_filter == "Not Flagged":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "intervention_recommended"
                ]
                == False
            ]
        )


    unique_priority[
        "Patient"
    ] = (
        unique_priority[
            "patient_id"
        ]
        .apply(
            short_patient_id
        )
    )


    if patient_search:

        search_text = (
            patient_search
            .strip()
            .upper()
        )


        unique_priority = (
            unique_priority[
                unique_priority[
                    "Patient"
                ]
                .str.upper()
                .str.contains(
                    search_text,
                    na=False,
                )
            ]
        )


    unique_priority[
        "Risk (%)"
    ] = (
        unique_priority[
            "readmission_probability_percent"
        ]
        .round(
            1
        )
    )


    unique_priority[
        "Review Status"
    ] = (
        unique_priority[
            "intervention_recommended"
        ]
        .map(
            {
                True:
                    "RECOMMENDED",

                False:
                    "NOT FLAGGED",
            }
        )
    )


    unique_priority[
        "Conditions"
    ] = (
        unique_priority.apply(
            condition_summary,
            axis=1,
        )
    )


    queue = unique_priority[
        [
            "Patient",
            "Risk (%)",
            "risk_level",
            "Review Status",
            "admissions_last_90_days",
            "prior_readmissions_last_365_days",
            "Conditions",
        ]
    ].head(
        20
    ).copy()


    queue = queue.rename(
        columns={
            "risk_level":
                "Risk Tier",

            "admissions_last_90_days":
                "Admissions / 90 Days",

            "prior_readmissions_last_365_days":
                "Prior Readmissions / Year",
        }
    )


    st.caption(
        (
            f"{len(unique_priority):,} patients "
            f"match the current filters"
        )
    )


    st.dataframe(
        queue,
        width="stretch",
        hide_index=True,
        height=520,

        column_config={
            "Risk (%)":
                st.column_config.ProgressColumn(
                    "Readmission Risk",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
        },
    )


    with st.expander(
        "Model validation"
    ):

        v1, v2, v3 = st.columns(
            3
        )


        with v1:

            st.metric(
                "ROC-AUC",
                "0.932",
            )


        with v2:

            st.metric(
                "PR-AUC",
                "0.754",
            )


        with v3:

            st.metric(
                "Brier Score",
                "0.0559",
            )


        st.caption(
            (
                "Held-out patient-level validation on "
                "synthetic Synthea data. "
                "No patients overlap between training and testing. "
                "Not intended as clinical validation."
            )
        )


# ============================================================
# PATIENT WORKUP
# ============================================================

elif page == "Patient Workup":

    page_header(
        "PATIENT WORKUP",
        "Individual Risk Profile",
        (
            "Review longitudinal utilization, current modeled risk, "
            "clinical context, and patient-specific model drivers."
        ),
    )


    patient_ids = (
        registry[
            "patient_id"
        ]
        .dropna()
        .drop_duplicates()
        .tolist()
    )


    patient_lookup = {

        short_patient_id(
            patient_id
        ):
            patient_id

        for patient_id
        in patient_ids
    }


    selected_label = st.selectbox(
        "Search Patient",
        list(
            patient_lookup.keys()
        ),
    )


    selected_patient = (
        patient_lookup[
            selected_label
        ]
    )


    patient_history = (
        registry[
            registry[
                "patient_id"
            ]
            == selected_patient
        ]
        .copy()
    )


    # The production registry does not contain calendar admission dates.
    # We therefore use cumulative prior admissions as encounter order.

    patient_history = (
        patient_history
        .sort_values(
            [
                "previous_inpatient_admissions",
                "readmission_probability",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


    patient_history[
        "Encounter Sequence"
    ] = range(
        1,
        len(
            patient_history
        ) + 1,
    )


    patient = (
        patient_history
        .sort_values(
            "readmission_probability",
            ascending=False,
        )
        .iloc[
            0
        ]
    )


    st.html(
        f"""
<div class="patient-banner">

    <div class="patient-id">
        {selected_label}
    </div>

    <div class="patient-meta">
        {len(patient_history)} recorded hospitalization(s) ·
        displaying the patient's highest-risk scored encounter
    </div>

</div>
"""
    )


    risk_signal(
        patient[
            "readmission_probability_percent"
        ],

        patient[
            "risk_level"
        ],

        22.0,

        bool(
            patient[
                "intervention_recommended"
            ]
        ),
    )


    st.write("")


    section_header(
        "Why This Patient Scored This Way",
        (
            "The strongest patient-specific contributors "
            "to the current prediction."
        ),
        "SHAP",
    )


    try:

        patient_data = row_to_model_input(
            patient
        )


        patient_drivers = explain_readmission(
            patient_data,
            top_n=5,
        )


        render_drivers(
            patient_drivers
        )


        st.caption(
            (
                "These contributors explain the model's prediction. "
                "They do not establish that a factor causes "
                "or prevents readmission."
            )
        )


    except Exception as error:

        st.warning(
            "Risk-driver explanation could not be generated."
        )


        st.caption(
            str(
                error
            )
        )


    st.write("")


    section_header(
        "Risk Progression",
        (
            "Recorded encounters ordered by cumulative "
            "prior inpatient utilization."
        ),
        "LONGITUDINAL",
    )


    progression = patient_history[
        [
            "Encounter Sequence",
            "readmission_probability_percent",
            "risk_level",
            "actual_readmitted_30_days",
        ]
    ].copy()


    progression_chart = (
        alt.Chart(
            progression
        )
        .mark_line(
            point=True,
            strokeWidth=2.5,
            color="#38d9d0",
        )
        .encode(

            x=alt.X(
                "Encounter Sequence:O",
                title="Encounter sequence",
            ),

            y=alt.Y(
                "readmission_probability_percent:Q",
                title="Calibrated risk (%)",

                scale=alt.Scale(
                    domain=[
                        0,
                        100,
                    ]
                ),
            ),

            tooltip=[

                alt.Tooltip(
                    "Encounter Sequence:O",
                    title="Encounter",
                ),

                alt.Tooltip(
                    "readmission_probability_percent:Q",
                    title="Risk",
                    format=".1f",
                ),

                alt.Tooltip(
                    "risk_level:N",
                    title="Risk Tier",
                ),

                alt.Tooltip(
                    "actual_readmitted_30_days:Q",
                    title="Readmitted",
                ),
            ],
        )
        .properties(
            height=230,
        )
        .configure_view(
            strokeOpacity=0,
        )
    )


    st.altair_chart(
        progression_chart,
        width="stretch",
    )


    history_col, clinical_col = st.columns(
        2,
        gap="large",
    )


    with history_col:

        section_header(
            "Recent Patient History",
            (
                "Recent inpatient utilization "
                "and known readmission history."
            ),
            "UTILIZATION",
        )


        recent_history = pd.DataFrame(
            {
                "Metric": [

                    "Admissions in last 30 days",

                    "Admissions in last 90 days",

                    "Admissions in last 365 days",

                    "Prior readmissions in last 365 days",

                    "Days since last inpatient admission",

                    "Lifetime prior admissions",
                ],

                "Value": [

                    patient[
                        "admissions_last_30_days"
                    ],

                    patient[
                        "admissions_last_90_days"
                    ],

                    patient[
                        "admissions_last_365_days"
                    ],

                    patient[
                        "prior_readmissions_last_365_days"
                    ],

                    round(
                        float(
                            patient[
                                "days_since_last_inpatient_admission"
                            ]
                        ),
                        1,
                    ),

                    patient[
                        "previous_inpatient_admissions"
                    ],
                ],
            }
        )


        st.dataframe(
            recent_history,
            width="stretch",
            hide_index=True,
        )


    with clinical_col:

        section_header(
            "Clinical Profile",
            (
                "Clinical complexity and documented "
                "conditions available in the registry."
            ),
            "PROFILE",
        )


        clinical = pd.DataFrame(
            {
                "Metric": [

                    "Age",

                    "Length of Stay",

                    "Condition Count",

                    "Medication Count",

                    "Procedure Count",

                    "BMI",

                    "Diabetes",

                    "Hypertension",

                    "Kidney Disease",
                ],

                "Value": [

                    patient[
                        "age_at_admission"
                    ],

                    round(
                        float(
                            patient[
                                "length_of_stay"
                            ]
                        ),
                        1,
                    ),

                    patient[
                        "condition_count"
                    ],

                    patient[
                        "medication_count"
                    ],

                    patient[
                        "procedure_count"
                    ],

                    round(
                        float(
                            safe_number(
                                patient[
                                    "bmi"
                                ],
                                0,
                            )
                        ),
                        1,
                    ),

                    (
                        "Yes"
                        if bool(
                            patient[
                                "diabetes"
                            ]
                        )
                        else "No"
                    ),

                    (
                        "Yes"
                        if bool(
                            patient[
                                "hypertension"
                            ]
                        )
                        else "No"
                    ),

                    (
                        "Yes"
                        if bool(
                            patient[
                                "kidney_disease"
                            ]
                        )
                        else "No"
                    ),
                ],
            }
        )


        st.dataframe(
            clinical,
            width="stretch",
            hide_index=True,
        )


    st.write("")


    section_header(
        "Encounter Ledger",
        (
            "All scored encounters available "
            "for the selected patient."
        ),
        f"{len(patient_history)} RECORD(S)",
    )


    ledger = patient_history[
        [
            "Encounter Sequence",

            "encounter_id",

            "readmission_probability_percent",

            "risk_level",

            "intervention_recommended",

            "actual_readmitted_30_days",

            "admissions_last_90_days",

            "prior_readmissions_last_365_days",
        ]
    ].copy()


    ledger[
        "Encounter"
    ] = (
        ledger[
            "encounter_id"
        ]
        .apply(
            short_encounter_id
        )
    )


    ledger[
        "Review Status"
    ] = (
        ledger[
            "intervention_recommended"
        ]
        .map(
            {
                True:
                    "RECOMMENDED",

                False:
                    "NOT FLAGGED",
            }
        )
    )


    ledger = ledger[
        [
            "Encounter Sequence",

            "Encounter",

            "readmission_probability_percent",

            "risk_level",

            "Review Status",

            "actual_readmitted_30_days",

            "admissions_last_90_days",

            "prior_readmissions_last_365_days",
        ]
    ]


    ledger = ledger.rename(
        columns={

            "Encounter Sequence":
                "Seq",

            "readmission_probability_percent":
                "Risk (%)",

            "risk_level":
                "Risk Tier",

            "actual_readmitted_30_days":
                "Readmitted",

            "admissions_last_90_days":
                "Admissions / 90 Days",

            "prior_readmissions_last_365_days":
                "Prior Readmissions / Year",
        }
    )


    st.dataframe(
        ledger,
        width="stretch",
        hide_index=True,

        column_config={
            "Risk (%)":
                st.column_config.ProgressColumn(
                    "Readmission Risk",
                    min_value=0,
                    max_value=100,
                    format="%.1f%%",
                )
        },
    )


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    page_header(
        "RISK ASSESSMENT",
        "Readmission Risk Assessment",
        (
            "Estimate 30-day readmission risk using the "
            "production 16-feature history-aware model."
        ),
    )


    if (
        "assessment_preset"
        not in st.session_state
    ):

        st.session_state[
            "assessment_preset"
        ] = "Custom"


    if (
        "preset_counter"
        not in st.session_state
    ):

        st.session_state[
            "preset_counter"
        ] = 0


    presets = {

        "Low Risk":
            low_example,

        "Moderate Risk":
            moderate_example,

        "High Risk":
            high_example,
    }


    section_header(
        "Quick Start",
        (
            "Load a representative synthetic patient "
            "or enter a custom history."
        ),
        "PRESETS",
    )


    b1, b2, b3, b4 = st.columns(
        4
    )


    with b1:

        if st.button(
            "Low-Risk Example",
            width="stretch",
        ):

            st.session_state[
                "assessment_preset"
            ] = "Low Risk"


            st.session_state[
                "preset_counter"
            ] += 1


            st.rerun()


    with b2:

        if st.button(
            "Moderate-Risk Example",
            width="stretch",
        ):

            st.session_state[
                "assessment_preset"
            ] = "Moderate Risk"


            st.session_state[
                "preset_counter"
            ] += 1


            st.rerun()


    with b3:

        if st.button(
            "High-Risk Example",
            width="stretch",
        ):

            st.session_state[
                "assessment_preset"
            ] = "High Risk"


            st.session_state[
                "preset_counter"
            ] += 1


            st.rerun()


    with b4:

        if st.button(
            "Clear Form",
            width="stretch",
        ):

            st.session_state[
                "assessment_preset"
            ] = "Custom"


            st.session_state[
                "preset_counter"
            ] += 1


            st.rerun()


    preset_name = (
        st.session_state[
            "assessment_preset"
        ]
    )


    if preset_name == "Custom":

        defaults = {

            "age": 65,

            "los": 5.0,

            "previous_admissions": 0,

            "admissions_30": 0,

            "admissions_90": 0,

            "admissions_365": 0,

            "days_since": 365.0,

            "prior_readmissions": 0,

            "conditions": 2,

            "medications": 5,

            "procedures": 3,

            "diabetes": False,

            "hypertension": False,

            "kidney_disease": False,

            "bmi": 27.0,
        }


    else:

        defaults = (
            presets[
                preset_name
            ]
        )


        st.info(
            (
                f'{preset_name} example loaded from the scored '
                f'Synthea registry. Stored modeled risk: '
                f'{defaults["source_probability"]:.1f}% '
                f'({defaults["source_risk_level"]}).'
            )
        )


    key_suffix = (
        st.session_state[
            "preset_counter"
        ]
    )


    left, right = st.columns(
        [
            1.05,
            0.95,
        ],
        gap="large",
    )


    with left:

        section_header(
            "Patient Inputs",
            (
                "Current encounter, longitudinal history, "
                "and clinical complexity."
            ),
            "16 FEATURES",
        )


        p1, p2, p3 = st.columns(
            3
        )


        with p1:

            age = st.number_input(
                "Age",
                min_value=0,
                max_value=120,

                value=int(
                    defaults[
                        "age"
                    ]
                ),

                key=(
                    f"age_"
                    f"{key_suffix}"
                ),
            )


        with p2:

            length_of_stay = st.number_input(
                "Length of Stay",
                min_value=0.0,

                value=float(
                    defaults[
                        "los"
                    ]
                ),

                step=0.5,

                key=(
                    f"los_"
                    f"{key_suffix}"
                ),
            )


        with p3:

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,

                value=float(
                    defaults[
                        "bmi"
                    ]
                ),

                step=0.1,

                key=(
                    f"bmi_"
                    f"{key_suffix}"
                ),
            )


        st.markdown(
            "##### Recent Admission History"
        )


        h1, h2, h3 = st.columns(
            3
        )


        with h1:

            admissions_30 = st.number_input(
                "Admissions / 30 Days",
                min_value=0,
                max_value=20,

                value=int(
                    defaults[
                        "admissions_30"
                    ]
                ),

                key=(
                    f"admissions30_"
                    f"{key_suffix}"
                ),
            )


        with h2:

            admissions_90 = st.number_input(
                "Admissions / 90 Days",
                min_value=0,
                max_value=30,

                value=int(
                    defaults[
                        "admissions_90"
                    ]
                ),

                key=(
                    f"admissions90_"
                    f"{key_suffix}"
                ),
            )


        with h3:

            admissions_365 = st.number_input(
                "Admissions / 365 Days",
                min_value=0,
                max_value=100,

                value=int(
                    defaults[
                        "admissions_365"
                    ]
                ),

                key=(
                    f"admissions365_"
                    f"{key_suffix}"
                ),
            )


        h4, h5, h6 = st.columns(
            3
        )


        with h4:

            previous_admissions = st.number_input(
                "Total Prior Admissions",
                min_value=0,
                max_value=500,

                value=int(
                    defaults[
                        "previous_admissions"
                    ]
                ),

                key=(
                    f"previous_admissions_"
                    f"{key_suffix}"
                ),
            )


        with h5:

            prior_readmissions = st.number_input(
                "Prior Readmissions / Year",
                min_value=0,
                max_value=50,

                value=int(
                    defaults[
                        "prior_readmissions"
                    ]
                ),

                key=(
                    f"readmissions365_"
                    f"{key_suffix}"
                ),
            )


        with h6:

            days_since = st.number_input(
                "Days Since Last Admission",
                min_value=0.0,
                max_value=365.0,

                value=float(
                    defaults[
                        "days_since"
                    ]
                ),

                step=1.0,

                key=(
                    f"days_since_"
                    f"{key_suffix}"
                ),

                help=(
                    "Use 365 when the patient has no known prior "
                    "inpatient admission or the last admission "
                    "was more than one year ago."
                ),
            )


        st.markdown(
            "##### Clinical Complexity"
        )


        c1, c2, c3 = st.columns(
            3
        )


        with c1:

            condition_count = st.number_input(
                "Active Conditions",
                min_value=0,
                max_value=100,

                value=int(
                    defaults[
                        "conditions"
                    ]
                ),

                key=(
                    f"conditions_"
                    f"{key_suffix}"
                ),
            )


        with c2:

            medication_count = st.number_input(
                "Medication Count",
                min_value=0,
                max_value=100,

                value=int(
                    defaults[
                        "medications"
                    ]
                ),

                key=(
                    f"medications_"
                    f"{key_suffix}"
                ),
            )


        with c3:

            procedure_count = st.number_input(
                "Procedure Count",
                min_value=0,
                max_value=200,

                value=int(
                    defaults[
                        "procedures"
                    ]
                ),

                key=(
                    f"procedures_"
                    f"{key_suffix}"
                ),
            )


        st.markdown(
            "##### Documented Conditions"
        )


        d1, d2, d3 = st.columns(
            3
        )


        with d1:

            diabetes = st.checkbox(
                "Diabetes",

                value=defaults[
                    "diabetes"
                ],

                key=(
                    f"diabetes_"
                    f"{key_suffix}"
                ),
            )


        with d2:

            hypertension = st.checkbox(
                "Hypertension",

                value=defaults[
                    "hypertension"
                ],

                key=(
                    f"hypertension_"
                    f"{key_suffix}"
                ),
            )


        with d3:

            kidney_disease = st.checkbox(
                "Kidney Disease",

                value=defaults[
                    "kidney_disease"
                ],

                key=(
                    f"kidney_"
                    f"{key_suffix}"
                ),
            )


    with right:

        section_header(
            "Assessment",
            (
                "Submit the encounter to the calibrated "
                "production model."
            ),
            "22% REVIEW THRESHOLD",
        )


        st.html(
            """
<div class="panel-card">

    <div class="review-label">
        Operational interpretation
    </div>

    <div
        class="review-sub"
        style="margin-top:0.4rem"
    >
        Admitra separates estimated readmission risk
        from the operational review threshold.

        A review flag does not mean the patient
        will be readmitted.
    </div>

</div>
"""
        )


        st.write("")


        analyze = st.button(
            "Analyze Readmission Risk",
            type="primary",
            width="stretch",
        )


        if admissions_30 > admissions_90:

            st.warning(
                (
                    "Admissions in the last 30 days cannot exceed "
                    "admissions in the last 90 days. "
                    "Admitra will normalize the history before scoring."
                )
            )


        if admissions_90 > admissions_365:

            st.warning(
                (
                    "Admissions in the last 90 days cannot exceed "
                    "admissions in the last 365 days. "
                    "Admitra will normalize the history before scoring."
                )
            )


        if admissions_365 > previous_admissions:

            st.warning(
                (
                    "Admissions in the last year cannot exceed total "
                    "prior inpatient admissions. Admitra will normalize "
                    "the history before scoring."
                )
            )


    if analyze:

        has_prior_inpatient = int(
            (
                previous_admissions > 0
                or admissions_365 > 0
            )
        )


        patient_data = {

            "age_at_admission":
                age,

            "length_of_stay":
                length_of_stay,

            "previous_inpatient_admissions":
                previous_admissions,

            "admissions_last_30_days":
                admissions_30,

            "admissions_last_90_days":
                admissions_90,

            "admissions_last_365_days":
                admissions_365,

            "days_since_last_inpatient_admission":
                days_since,

            "has_prior_inpatient_admission":
                has_prior_inpatient,

            "prior_readmissions_last_365_days":
                prior_readmissions,

            "condition_count":
                condition_count,

            "medication_count":
                medication_count,

            "procedure_count":
                procedure_count,

            "diabetes":
                int(
                    diabetes
                ),

            "hypertension":
                int(
                    hypertension
                ),

            "kidney_disease":
                int(
                    kidney_disease
                ),

            "bmi":
                bmi,
        }


        try:

            result = predict_readmission(
                patient_data
            )


            drivers = explain_readmission(
                patient_data,
                top_n=5,
            )


            st.write("")


            section_header(
                "Risk Intelligence",
                (
                    "Calibrated 30-day readmission prediction "
                    "and operational review status."
                ),
                "RESULT",
            )


            risk_signal(
                result[
                    "probability_percent"
                ],

                result[
                    "risk_level"
                ],

                result[
                    "production_threshold_percent"
                ],

                result[
                    "intervention_recommended"
                ],
            )


            st.write("")


            section_header(
                "Why This Patient Scored This Way",
                (
                    "The strongest patient-specific contributors "
                    "to this prediction."
                ),
                "SHAP",
            )


            render_drivers(
                drivers
            )


            st.caption(
                (
                    "These contributors explain the model's prediction. "
                    "They do not establish that a factor causes "
                    "or prevents readmission."
                )
            )


            with st.expander(
                "Validation and threshold details"
            ):

                st.markdown(
                    f"""
The **{result["production_threshold_percent"]}% review threshold** was selected using patient-grouped out-of-fold validation.

Held-out production-cohort performance:

- ROC-AUC: **0.932**
- PR-AUC: **0.754**
- Brier score: **0.0559**
- Recall at threshold: **80.1%**
- Precision at threshold: **52.3%**
- F1 at threshold: **0.633**

The threshold is an operational prioritization rule for this synthetic demonstration and is not a clinical treatment guideline.
"""
                )


        except Exception as error:

            st.error(
                "Unable to generate the assessment."
            )


            st.exception(
                error
            )


# ============================================================
# DISCLAIMER
# ============================================================

st.html(
    """
<div class="disclaimer">

Admitra is a portfolio demonstration using synthetic Synthea healthcare data.
Predictions are generated by a history-aware calibrated XGBoost model
and are not intended for clinical use.

</div>
"""
)