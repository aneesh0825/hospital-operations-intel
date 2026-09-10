import altair as alt
import pandas as pd
import streamlit as st

from src.predict_readmission import (
    predict_readmission,
    explain_readmission,
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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

:root {
    --bg: #08111f;
    --sidebar: #0b1525;
    --panel: #101b2e;
    --border: rgba(148, 163, 184, 0.15);

    --text: #f8fafc;
    --muted: #94a3b8;
    --muted2: #64748b;

    --blue: #60a5fa;
    --cyan: #22d3ee;
    --purple: #a78bfa;
    --pink: #f472b6;
    --green: #4ade80;
    --yellow: #facc15;
    --red: #fb7185;
}


/* ==========================================================
   GLOBAL
========================================================== */

.stApp {
    background:
        radial-gradient(
            circle at 65% -10%,
            rgba(79, 70, 229, 0.15),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 20%,
            rgba(14, 165, 233, 0.08),
            transparent 22%
        ),
        #08111f;

    color: var(--text);
}

.block-container {
    max-width: 1480px;
    padding-top: 1.6rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}


/* ==========================================================
   SIDEBAR
========================================================== */

section[data-testid="stSidebar"] {
    width: 285px !important;

    background:
        linear-gradient(
            180deg,
            #0c1728 0%,
            #0a1322 100%
        );

    border-right: 1px solid rgba(96, 165, 250, 0.12);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.4rem;
    padding-left: 1rem;
    padding-right: 1rem;
}

.sidebar-logo {
    font-size: 1.55rem;
    font-weight: 850;
    letter-spacing: 0.15em;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #93c5fd,
            #c4b5fd
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    margin-bottom: 0.15rem;
}

.sidebar-subtitle {
    color: #64748b;
    font-size: 0.74rem;
    margin-bottom: 1.5rem;
}

.sidebar-label {
    color: #64748b;
    font-size: 0.64rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    margin-bottom: 0.6rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0.35rem;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.68rem 0.75rem;
    border-radius: 10px;
    transition: 0.15s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(96, 165, 250, 0.07);
}

section[data-testid="stSidebar"]
div[role="radiogroup"]
label:has(input:checked) {
    background:
        linear-gradient(
            90deg,
            rgba(59, 130, 246, 0.20),
            rgba(139, 92, 246, 0.10)
        );

    border: 1px solid rgba(96, 165, 250, 0.22);
}

.model-card {
    background:
        linear-gradient(
            145deg,
            rgba(30, 41, 59, 0.85),
            rgba(15, 23, 42, 0.9)
        );

    border: 1px solid rgba(96, 165, 250, 0.13);
    border-radius: 12px;

    padding: 0.9rem;
    margin-top: 0.25rem;
}

.model-title {
    color: #e2e8f0;
    font-size: 0.82rem;
    font-weight: 700;
}

.model-description {
    color: #64748b;
    font-size: 0.69rem;
    margin-top: 0.35rem;
}

.active-pill {
    display: inline-block;
    margin-top: 0.65rem;

    color: #86efac;
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.20);

    padding: 0.25rem 0.5rem;
    border-radius: 999px;

    font-size: 0.63rem;
    font-weight: 750;
}

.sidebar-bottom {
    color: #475569;
    font-size: 0.66rem;
    line-height: 1.5;
    margin-top: 1.4rem;
}


/* ==========================================================
   HERO
========================================================== */

.page-hero {
    border-radius: 16px;
    padding: 1.25rem 1.35rem;
    margin-bottom: 1.4rem;

    border: 1px solid rgba(96, 165, 250, 0.13);

    background:
        linear-gradient(
            120deg,
            rgba(37, 99, 235, 0.12),
            rgba(124, 58, 237, 0.08),
            rgba(15, 23, 42, 0.2)
        );
}

.page-kicker {
    color: #7dd3fc;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.page-title {
    color: #f8fafc;
    font-size: 1.85rem;
    font-weight: 820;
    margin-top: 0.25rem;
}

.page-description {
    color: #94a3b8;
    font-size: 0.87rem;
    margin-top: 0.35rem;
}


/* ==========================================================
   METRIC CARDS
========================================================== */

.metric-card {
    border-radius: 14px;
    padding: 1rem 1.05rem;
    min-height: 122px;

    border: 1px solid rgba(148, 163, 184, 0.12);

    box-shadow:
        0 12px 26px rgba(0, 0, 0, 0.16);
}

.metric-blue {
    background:
        linear-gradient(
            145deg,
            rgba(37, 99, 235, 0.18),
            rgba(15, 23, 42, 0.82)
        );
}

.metric-purple {
    background:
        linear-gradient(
            145deg,
            rgba(124, 58, 237, 0.18),
            rgba(15, 23, 42, 0.82)
        );
}

.metric-cyan {
    background:
        linear-gradient(
            145deg,
            rgba(8, 145, 178, 0.18),
            rgba(15, 23, 42, 0.82)
        );
}

.metric-pink {
    background:
        linear-gradient(
            145deg,
            rgba(219, 39, 119, 0.14),
            rgba(15, 23, 42, 0.82)
        );
}

.metric-label {
    color: #94a3b8;
    font-size: 0.68rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-value {
    color: #f8fafc;
    font-size: 1.75rem;
    font-weight: 820;
    margin-top: 0.35rem;
}

.metric-note {
    color: #64748b;
    font-size: 0.74rem;
    margin-top: 0.25rem;
}


/* ==========================================================
   SECTIONS
========================================================== */

.section-title {
    color: #f8fafc;
    font-size: 1.02rem;
    font-weight: 720;
    margin-top: 0.5rem;
}

.section-description {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.15rem;
    margin-bottom: 0.65rem;
}


/* ==========================================================
   PANEL
========================================================== */

.panel-card {
    background:
        linear-gradient(
            145deg,
            rgba(15, 23, 42, 0.95),
            rgba(17, 27, 46, 0.90)
        );

    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 14px;

    padding: 1rem 1.05rem;
}

.review-row {
    display: flex;
    justify-content: space-between;
    align-items: center;

    padding: 0.7rem 0;

    border-bottom:
        1px solid rgba(148, 163, 184, 0.08);
}

.review-row:last-child {
    border-bottom: none;
}

.review-label {
    color: #cbd5e1;
    font-size: 0.8rem;
}

.review-value {
    color: #f8fafc;
    font-size: 1rem;
    font-weight: 750;
}

.review-sub {
    color: #64748b;
    font-size: 0.7rem;
}


/* ==========================================================
   DRIVER CARDS
========================================================== */

.driver-card {
    background:
        linear-gradient(
            135deg,
            rgba(15, 23, 42, 0.94),
            rgba(20, 34, 56, 0.9)
        );

    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 12px;

    padding: 0.9rem 1rem;
    margin-bottom: 0.55rem;
}

.driver-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.driver-feature {
    color: #f8fafc;
    font-size: 0.85rem;
    font-weight: 700;
}

.driver-up {
    color: #fb7185;
    font-size: 0.69rem;
    font-weight: 700;
}

.driver-down {
    color: #4ade80;
    font-size: 0.69rem;
    font-weight: 700;
}

.driver-value {
    color: #dbeafe;
    font-size: 0.78rem;
    font-weight: 650;
    margin-top: 0.38rem;
}

.driver-reasoning {
    color: #94a3b8;
    font-size: 0.75rem;
    line-height: 1.5;
    margin-top: 0.32rem;
    max-width: 960px;
}


/* ==========================================================
   STREAMLIT ELEMENTS
========================================================== */

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #121d30 !important;
    border-color: rgba(96, 165, 250, 0.14) !important;
    border-radius: 9px !important;
}

label {
    color: #cbd5e1 !important;
}

.stButton > button {
    height: 3rem;
    border-radius: 10px;

    border: 1px solid rgba(147, 197, 253, 0.22);

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #7c3aed
        );

    color: white;
    font-weight: 750;
}

.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #1d4ed8,
            #6d28d9
        );

    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 11px;
    overflow: hidden;
}

.disclaimer {
    margin-top: 2rem;
    padding-top: 1rem;

    border-top:
        1px solid rgba(148, 163, 184, 0.11);

    color: #475569;
    font-size: 0.69rem;
}

</style>
""",
    unsafe_allow_html=True,
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
# HELPERS
# ============================================================

def short_patient_id(patient_id):

    return (
        "P-"
        + str(patient_id)[-6:].upper()
    )


def short_encounter_id(encounter_id):

    return (
        "E-"
        + str(encounter_id)[-6:].upper()
    )


def safe_number(
    value,
    fallback,
):

    if pd.isna(value):
        return fallback

    return value


def section_header(
    title,
    description,
):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True,
    )


def page_header(
    kicker,
    title,
    description,
):

    st.markdown(
        (
            '<div class="page-hero">'
            f'<div class="page-kicker">{kicker}</div>'
            f'<div class="page-title">{title}</div>'
            f'<div class="page-description">{description}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def metric_card(
    label,
    value,
    note,
    style_class,
):

    st.markdown(
        (
            f'<div class="metric-card {style_class}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-note">{note}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def condition_summary(row):

    conditions = []

    if row.get("diabetes", 0) == 1:
        conditions.append("Diabetes")

    if row.get("hypertension", 0) == 1:
        conditions.append("Hypertension")

    if row.get("kidney_disease", 0) == 1:
        conditions.append("Kidney Disease")

    if not conditions:
        return "None flagged"

    return ", ".join(
        conditions
    )


# ============================================================
# DRIVER EXPLANATIONS
# ============================================================

def render_drivers(
    drivers,
):

    explanations = {
        "Admissions in last 30 days":
            "Very recent hospitalization history can indicate an active pattern of repeated inpatient care.",

        "Admissions in last 90 days":
            "Recent inpatient utilization is one of the strongest signals associated with 30-day readmission in this model.",

        "Admissions in last 365 days":
            "Repeated hospital admissions over the previous year indicate a sustained pattern of inpatient utilization.",

        "Prior readmissions in last 365 days":
            "Previous confirmed readmissions provide direct evidence of a recent pattern of returning to inpatient care.",

        "Days since last inpatient admission":
            "A more recent prior hospitalization can increase modeled readmission risk because the patient returned to inpatient care within a shorter interval.",

        "Prior inpatient history":
            "The model considers whether the patient has any documented inpatient history before the current encounter.",

        "Previous inpatient admissions":
            "The patient's cumulative history of inpatient admissions contributes additional context about long-term healthcare utilization.",

        "Condition burden":
            "The number of documented health conditions contributes to the model's estimate of overall clinical complexity.",

        "Procedure count":
            "The number of procedures contributes information about the complexity of the patient's recent course of care.",

        "Medication count":
            "The number of medications contributes information about treatment complexity.",

        "Age":
            "Age contributes to the prediction based on patterns learned across patients of different ages.",

        "Length of stay":
            "The duration of the current hospitalization contributes to the model's assessment of the encounter.",

        "Diabetes":
            "Diabetes status contributes to the patient's overall clinical profile.",

        "Hypertension":
            "Hypertension status contributes to the patient's overall clinical profile.",

        "Kidney disease":
            "Kidney disease status contributes to the patient's overall clinical profile.",

        "BMI":
            "BMI contributes to the prediction based on patterns observed across the synthetic training population.",
    }


    units = {
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


    binary_features = {
        "Prior inpatient history",
        "Diabetes",
        "Hypertension",
        "Kidney disease",
    }


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
                "↑ PUSHED PREDICTION HIGHER"
            )

        elif direction == "decreases risk":

            direction_class = (
                "driver-down"
            )

            direction_label = (
                "↓ PUSHED PREDICTION LOWER"
            )

        else:

            direction_class = ""

            direction_label = (
                "NEUTRAL"
            )


        raw_value = driver[
            "value"
        ]


        if feature in binary_features:

            try:

                display_value = (
                    "Present"
                    if float(
                        raw_value
                    ) == 1
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


            unit = units.get(
                feature,
                "",
            )


            if unit:

                display_value = (
                    f"{number} {unit}"
                )

            else:

                display_value = str(
                    number
                )


        explanation = explanations.get(
            feature,
            "This factor influenced the prediction based on patterns learned by the model.",
        )


        st.markdown(
            (
                '<div class="driver-card">'

                '<div class="driver-top">'
                f'<div class="driver-feature">{feature}</div>'
                f'<div class="{direction_class}">{direction_label}</div>'
                '</div>'

                '<div class="driver-value">'
                f'{display_value}'
                '</div>'

                '<div class="driver-reasoning">'
                f'{explanation}'
                '</div>'

                '</div>'
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# FACTOR DEFINITIONS
# ============================================================

def factor_definitions():

    with st.expander(
        "What do these factors mean?"
    ):

        st.markdown(
            """
**Admissions in last 30 days**  
The number of inpatient hospital admissions recorded during the 30 days before the current hospitalization.

**Admissions in last 90 days**  
The number of inpatient hospital admissions recorded during the 90 days before the current hospitalization.

**Admissions in last 365 days**  
The number of inpatient hospital admissions recorded during the year before the current hospitalization.

**Prior readmissions in last 365 days**  
The number of earlier hospitalizations during the previous year that were followed by another inpatient admission within 30 days.

**Days since last inpatient admission**  
The number of days between the patient's most recent prior inpatient admission and the current hospitalization.

**Prior inpatient history**  
Indicates whether the patient had any inpatient hospitalization before the current encounter.

**Previous inpatient admissions**  
The patient's cumulative number of inpatient admissions before the current hospitalization.

**Condition burden**  
The number of documented health conditions associated with the patient.

**Procedure count**  
The number of procedures associated with the patient's hospitalization and available treatment record.

**Medication count**  
The number of medications represented in the patient's available treatment record.

**Length of stay**  
The number of days the patient remained hospitalized during the current encounter.

**Age**  
The patient's age at the time of admission.

**BMI**  
Body mass index, a measure calculated from height and weight.

**Diabetes**  
Whether diabetes is documented in the patient's available clinical history.

**Hypertension**  
Whether hypertension is documented in the patient's available clinical history.

**Kidney disease**  
Whether kidney disease is documented in the patient's available clinical history.
"""
        )


# ============================================================
# PATIENT DATA HELPER
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
        .iloc[0]
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

st.sidebar.markdown(
    '<div class="sidebar-logo">ADMITRA</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-subtitle">'
    'Hospital Intelligence Platform'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-label">Workspace</div>',
    unsafe_allow_html=True,
)


page = st.sidebar.radio(
    "Workspace",
    [
        "Operations Overview",
        "Patient Review",
        "Risk Assessment",
    ],
    label_visibility="collapsed",
)


st.sidebar.markdown(
    '<div class="sidebar-label">System</div>',
    unsafe_allow_html=True,
)


st.sidebar.markdown(
    (
        '<div class="model-card">'
        '<div class="model-title">Readmission Intelligence</div>'
        '<div class="model-description">'
        'History-aware calibrated XGBoost model'
        '</div>'
        '<div class="active-pill">● MODEL ACTIVE</div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)


review_count_sidebar = int(
    registry[
        "intervention_recommended"
    ].sum()
)


st.sidebar.markdown(
    (
        '<div class="sidebar-bottom">'
        f'{review_count_sidebar:,} encounters in review queue'
        '<br><br>'
        'Synthetic Synthea environment'
        '<br>'
        'Portfolio demonstration'
        '</div>'
    ),
    unsafe_allow_html=True,
)


# ============================================================
# OPERATIONS OVERVIEW
# ============================================================

if page == "Operations Overview":

    page_header(
        "OPERATIONS OVERVIEW",
        "Readmission Operations",
        "Monitor population risk, review workload, and priority patients.",
    )


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


    not_flagged_count = (
        total_hospitalizations
        - intervention_count
    )


    average_risk = float(
        registry[
            "readmission_probability_percent"
        ].mean()
    )


    m1, m2, m3, m4 = st.columns(
        4,
        gap="medium",
    )


    with m1:

        metric_card(
            "Hospitalizations",
            f"{total_hospitalizations:,}",
            "Scored inpatient encounters",
            "metric-blue",
        )


    with m2:

        metric_card(
            "Observed Readmission",
            f"{readmission_rate:.1f}%",
            f"{actual_readmissions:,} observed events",
            "metric-purple",
        )


    with m3:

        metric_card(
            "Review Queue",
            f"{intervention_count:,}",
            f"{intervention_rate:.1f}% of encounters",
            "metric-cyan",
        )


    with m4:

        metric_card(
            "High Risk",
            f"{high_risk_count:,}",
            f"{high_risk_rate:.1f}% of encounters",
            "metric-pink",
        )


    st.write("")


    section_header(
        "Population Overview",
        "Risk distribution and review demand across scored encounters.",
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
            .fillna(0)
            .astype(int)
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
                    axis=alt.Axis(
                        labelColor="#cbd5e1",
                        labelFontSize=13,
                        ticks=False,
                        domain=False,
                    ),
                ),

                x=alt.X(
                    "Encounters:Q",
                    title=None,
                    axis=alt.Axis(
                        labelColor="#64748b",
                        gridColor="#1e293b",
                        gridOpacity=0.8,
                        domain=False,
                    ),
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
                            "#4ade80",
                            "#facc15",
                            "#fb7185",
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
            use_container_width=True,
        )


    with review_col:

        review_rate = (
            intervention_count
            / total_hospitalizations
            * 100
        )


        not_flagged_rate = (
            not_flagged_count
            / total_hospitalizations
            * 100
        )


        st.markdown(
            (
                '<div class="panel-card">'

                '<div class="review-row">'
                '<div>'
                '<div class="review-label">Review recommended</div>'
                f'<div class="review-sub">{review_rate:.1f}% of encounters</div>'
                '</div>'
                f'<div class="review-value">{intervention_count:,}</div>'
                '</div>'

                '<div class="review-row">'
                '<div>'
                '<div class="review-label">Not flagged</div>'
                f'<div class="review-sub">{not_flagged_rate:.1f}% of encounters</div>'
                '</div>'
                f'<div class="review-value">{not_flagged_count:,}</div>'
                '</div>'

                '<div class="review-row">'
                '<div>'
                '<div class="review-label">Average modeled risk</div>'
                '<div class="review-sub">Across scored encounters</div>'
                '</div>'
                f'<div class="review-value">{average_risk:.1f}%</div>'
                '</div>'

                '</div>'
            ),
            unsafe_allow_html=True,
        )


    st.write("")


    section_header(
        "Priority Worklist",
        "Filter, review, and search patients by modeled readmission risk.",
    )


    filter_col1, filter_col2, filter_col3 = st.columns(
        [
            0.34,
            0.33,
            0.33,
        ],
        gap="medium",
    )


    with filter_col1:

        risk_filter = st.selectbox(
            "Risk Tier",
            [
                "All Risk Tiers",
                "High",
                "Moderate",
                "Low",
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


    if risk_filter == "High":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "risk_level"
                ]
                == "HIGH"
            ]
        )


    elif risk_filter == "Moderate":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "risk_level"
                ]
                == "MODERATE"
            ]
        )


    elif risk_filter == "Low":

        unique_priority = (
            unique_priority[
                unique_priority[
                    "risk_level"
                ]
                == "LOW"
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
        ].apply(
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


    st.caption(
        f"{len(unique_priority):,} patients match the current filters"
    )


    unique_priority[
        "Risk (%)"
    ] = (
        unique_priority[
            "readmission_probability_percent"
        ].round(1)
    )


    unique_priority[
        "Review Status"
    ] = (
        unique_priority[
            "intervention_recommended"
        ].map(
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


    st.dataframe(
        queue,
        use_container_width=True,
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
            "Held-out patient-level validation on synthetic Synthea data. "
            "No patients overlap between training and testing. "
            "Not intended as clinical validation."
        )


# ============================================================
# PATIENT REVIEW
# ============================================================

elif page == "Patient Review":

    page_header(
        "PATIENT REVIEW",
        "Individual Risk Profile",
        "Review a patient's readmission risk, recent history, clinical context, and model drivers.",
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
        .sort_values(
            "readmission_probability",
            ascending=False,
        )
        .copy()
    )


    patient = (
        patient_history.iloc[0]
    )


    c1, c2, c3, c4 = st.columns(
        4,
        gap="medium",
    )


    with c1:

        metric_card(
            "30-Day Risk",
            f'{patient["readmission_probability_percent"]:.1f}%',
            "Calibrated probability",
            "metric-blue",
        )


    with c2:

        metric_card(
            "Risk Tier",
            patient[
                "risk_level"
            ],
            "Population classification",
            "metric-purple",
        )


    with c3:

        review_status = (
            "RECOMMENDED"
            if bool(
                patient[
                    "intervention_recommended"
                ]
            )
            else "NOT FLAGGED"
        )


        metric_card(
            "Review Status",
            review_status,
            "22% operating threshold",
            "metric-cyan",
        )


    with c4:

        metric_card(
            "Admissions / 90 Days",
            int(
                patient[
                    "admissions_last_90_days"
                ]
            ),
            "Recent inpatient history",
            "metric-pink",
        )


    st.write("")


    section_header(
        "Why This Patient Scored This Way",
        "The strongest patient-specific contributors to the model prediction.",
    )


    patient_data = row_to_model_input(
        patient
    )


    try:

        patient_drivers = explain_readmission(
            patient_data,
            top_n=5,
        )


        render_drivers(
            patient_drivers
        )


        st.caption(
            "These contributors explain the model's prediction. "
            "They do not establish that a factor causes or prevents readmission."
        )


        factor_definitions()


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


    history_col, clinical_col = st.columns(
        2,
        gap="large",
    )


    with history_col:

        section_header(
            "Recent Patient History",
            "Recent inpatient utilization and known readmission history.",
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
                        patient[
                            "days_since_last_inpatient_admission"
                        ],
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
            use_container_width=True,
            hide_index=True,
        )


    with clinical_col:

        section_header(
            "Clinical Profile",
            "Clinical complexity and documented conditions available in the registry.",
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
                        patient[
                            "length_of_stay"
                        ],
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
                        safe_number(
                            patient[
                                "bmi"
                            ],
                            0,
                        ),
                        1,
                    ),

                    "Yes"
                    if bool(
                        patient[
                            "diabetes"
                        ]
                    )
                    else "No",

                    "Yes"
                    if bool(
                        patient[
                            "hypertension"
                        ]
                    )
                    else "No",

                    "Yes"
                    if bool(
                        patient[
                            "kidney_disease"
                        ]
                    )
                    else "No",

                ],
            }
        )


        st.dataframe(
            clinical,
            use_container_width=True,
            hide_index=True,
        )


    st.write("")


    section_header(
        "Prior Encounters",
        "Modeled risk and observed outcomes across this patient's recorded hospitalizations.",
    )


    encounter_table = patient_history[
        [
            "encounter_id",
            "readmission_probability_percent",
            "risk_level",
            "intervention_recommended",
            "actual_readmitted_30_days",
            "admissions_last_90_days",
            "prior_readmissions_last_365_days",
        ]
    ].copy()


    encounter_table[
        "Encounter"
    ] = (
        encounter_table[
            "encounter_id"
        ].apply(
            short_encounter_id
        )
    )


    encounter_table[
        "Review Status"
    ] = (
        encounter_table[
            "intervention_recommended"
        ].map(
            {
                True:
                    "RECOMMENDED",

                False:
                    "NOT FLAGGED",
            }
        )
    )


    encounter_table = encounter_table[
        [
            "Encounter",
            "readmission_probability_percent",
            "risk_level",
            "Review Status",
            "actual_readmitted_30_days",
            "admissions_last_90_days",
            "prior_readmissions_last_365_days",
        ]
    ]


    encounter_table = encounter_table.rename(
        columns={
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
        encounter_table,
        use_container_width=True,
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


    with st.expander(
        "What does Review Status mean?"
    ):

        st.markdown(
            """
**Review Status** indicates whether the encounter crossed Admitra's operating threshold for additional review.

The current history-aware model uses a **22% calibrated readmission probability threshold**.

- Below 22%: the encounter is not automatically flagged.
- At or above 22%: additional review is recommended.

Risk Tier and Review Status are separate. Risk Tier describes where the patient's probability falls within the modeled population, while Review Status determines whether the patient enters the operational review queue.

This threshold was selected using grouped out-of-fold validation and is not a clinically validated treatment guideline.
"""
        )


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    page_header(
        "RISK ASSESSMENT",
        "Readmission Risk Assessment",
        "Estimate 30-day readmission risk using recent patient history and clinical context.",
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
        "Load a representative synthetic patient or enter a custom history.",
    )


    b1, b2, b3, b4 = st.columns(
        4
    )


    with b1:

        if st.button(
            "Low-Risk Example",
            use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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
            use_container_width=True,
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

            "heart_failure": False,
            "chronic_lung_disease": False,

            "bmi": 27.0,
        }

    else:

        defaults = (
            presets[
                preset_name
            ]
        )


        st.info(
            f'{preset_name} example loaded from the scored '
            f'Synthea registry. Stored modeled risk: '
            f'{defaults["source_probability"]:.1f}% '
            f'({defaults["source_risk_level"]}).'
        )


    key_suffix = (
        st.session_state[
            "preset_counter"
        ]
    )


    # --------------------------------------------------------
    # PATIENT BASICS
    # --------------------------------------------------------

    st.write("")


    section_header(
        "Patient Basics",
        "Core information about the current hospitalization.",
    )


    p1, p2, p3 = st.columns(
        3,
        gap="medium",
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
            "Length of Stay (days)",
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


    # --------------------------------------------------------
    # RECENT ADMISSION HISTORY
    # --------------------------------------------------------

    st.write("")


    section_header(
        "Recent Admission History",
        "Recent inpatient utilization is a major component of the history-aware model.",
    )


    h1, h2, h3 = st.columns(
        3,
        gap="medium",
    )


    with h1:

        admissions_30 = st.number_input(
            "Admissions in Last 30 Days",
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
            "Admissions in Last 90 Days",
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
            "Admissions in Last 365 Days",
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
        3,
        gap="medium",
    )


    with h4:

        previous_admissions = st.number_input(
            "Total Prior Inpatient Admissions",
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
            "Prior Readmissions in Last 365 Days",
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
            "Days Since Last Inpatient Admission",
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


    if admissions_30 > admissions_90:

        st.warning(
            "Admissions in the last 30 days cannot exceed "
            "admissions in the last 90 days. Admitra will "
            "normalize the history before scoring."
        )


    if admissions_90 > admissions_365:

        st.warning(
            "Admissions in the last 90 days cannot exceed "
            "admissions in the last 365 days. Admitra will "
            "normalize the history before scoring."
        )


    if admissions_365 > previous_admissions:

        st.warning(
            "Admissions in the last year cannot exceed total "
            "prior inpatient admissions. Admitra will normalize "
            "the history before scoring."
        )


    # --------------------------------------------------------
    # CLINICAL COMPLEXITY
    # --------------------------------------------------------

    st.write("")


    section_header(
        "Clinical Complexity",
        "Provide the current patient's condition and treatment profile.",
    )


    c1, c2, c3 = st.columns(
        3,
        gap="medium",
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


    # --------------------------------------------------------
    # DOCUMENTED CONDITIONS
    # --------------------------------------------------------

    st.write("")


    section_header(
        "Documented Conditions",
        "Select chronic conditions documented for this patient.",
    )


    d1, d2, d3 = st.columns(
        3,
        gap="medium",
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


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    st.write("")


    analyze = st.button(
        "Analyze Readmission Risk",
        type="primary",
        use_container_width=True,
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
                "Calibrated 30-day readmission prediction and review status.",
            )


            r1, r2, r3 = st.columns(
                3,
                gap="medium",
            )


            with r1:

                metric_card(
                    "30-Day Risk",
                    f'{result["probability_percent"]}%',
                    "Calibrated estimate of 30-day readmission probability",
                    "metric-blue",
                )


            with r2:

                metric_card(
                    "Risk Tier",
                    result[
                        "risk_level"
                    ],
                    "Population risk band",
                    "metric-purple",
                )


            with r3:

                review_status = (
                    "RECOMMENDED"
                    if result[
                        "intervention_recommended"
                    ]
                    else "NOT FLAGGED"
                )


                metric_card(
                    "Review Status",
                    review_status,
                    "22% operating threshold",
                    "metric-cyan",
                )


            st.write("")


            if result[
                "intervention_recommended"
            ]:

                st.warning(
                    f'**Additional review recommended.** '
                    f'This patient exceeds Admitra’s '
                    f'{result["production_threshold_percent"]}% '
                    f'operating threshold.'
                )

            else:

                st.success(
                    f'**No additional review flag.** '
                    f'This patient remains below Admitra’s '
                    f'{result["production_threshold_percent"]}% '
                    f'operating threshold.'
                )


            with st.expander(
                "What does the 22% threshold mean?"
            ):

                st.markdown(
                    f"""
The **{result["production_threshold_percent"]}% review threshold** is the calibrated readmission probability at which Admitra places an encounter into the review queue.

It was selected using **patient-grouped out-of-fold validation**. The operating rule chose the threshold with the highest F1 score while maintaining at least 70% recall.

For the final held-out test population, the model achieved approximately:

- **80.1% recall**
- **52.3% precision**
- **0.633 F1**
- **0.932 ROC-AUC**
- **0.754 PR-AUC**

The threshold is used for prioritization. It is not a clinical treatment guideline.
"""
                )


            st.write("")


            section_header(
                "Why This Patient Scored This Way",
                "The strongest patient-specific contributors to this prediction.",
            )


            render_drivers(
                drivers
            )


            st.caption(
                "These contributors explain the model's prediction. "
                "They do not establish that a factor causes or prevents readmission."
            )


            factor_definitions()


            with st.expander(
                "Technical Prediction Details"
            ):

                st.write(
                    f'**Calibrated probability:** '
                    f'{result["probability_percent"]}%'
                )

                st.write(
                    f'**Raw XGBoost probability:** '
                    f'{result["raw_probability_percent"]}%'
                )

                st.write(
                    f'**Risk tier:** '
                    f'{result["risk_level"]}'
                )

                st.write(
                    f'**Review status:** '
                    f'{review_status}'
                )

                st.write(
                    f'**Review threshold:** '
                    f'{result["production_threshold_percent"]}%'
                )

                st.write(
                    f'**Low / Moderate cutoff:** '
                    f'{result["low_cutoff_percent"]}%'
                )

                st.write(
                    f'**Moderate / High cutoff:** '
                    f'{result["high_cutoff_percent"]}%'
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

st.markdown(
    (
        '<div class="disclaimer">'
        'Admitra is a portfolio demonstration using synthetic '
        'Synthea healthcare data. Predictions are generated by '
        'a history-aware calibrated XGBoost model and are not '
        'intended for clinical use.'
        '</div>'
    ),
    unsafe_allow_html=True,
)