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
    page_icon="🏥",
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
   PAGE HERO
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
   PANEL CARD
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

    padding: 0.78rem 0.9rem;
    margin-bottom: 0.5rem;
}

.driver-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.driver-feature {
    color: #f8fafc;
    font-size: 0.83rem;
    font-weight: 700;
}

.driver-up {
    color: #fb7185;
    font-size: 0.7rem;
    font-weight: 700;
}

.driver-down {
    color: #4ade80;
    font-size: 0.7rem;
    font-weight: 700;
}

.driver-meta {
    color: #64748b;
    font-size: 0.69rem;
    margin-top: 0.25rem;
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
    return "P-" + str(patient_id)[-6:].upper()


def short_encounter_id(encounter_id):
    return "E-" + str(encounter_id)[-6:].upper()


def condition_summary(row):

    conditions = []

    if row.get("heart_failure", 0) == 1:
        conditions.append("Heart Failure")

    if row.get("kidney_disease", 0) == 1:
        conditions.append("Kidney Disease")

    if row.get("diabetes", 0) == 1:
        conditions.append("Diabetes")

    if row.get("hypertension", 0) == 1:
        conditions.append("Hypertension")

    if row.get("chronic_lung_disease", 0) == 1:
        conditions.append("Chronic Lung Disease")

    if not conditions:
        return "None flagged"

    return ", ".join(conditions)


def section_header(title, description):

    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-description">{description}</div>',
        unsafe_allow_html=True,
    )


def page_header(kicker, title, description):

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


def render_drivers(drivers):

    for driver in drivers:

        if driver["direction"] == "increases risk":
            direction_class = "driver-up"
            direction_label = "↑ INCREASES RISK"

        elif driver["direction"] == "decreases risk":
            direction_class = "driver-down"
            direction_label = "↓ DECREASES RISK"

        else:
            direction_class = ""
            direction_label = "NEUTRAL"

        try:
            display_value = round(
                float(driver["value"]),
                1,
            )
        except Exception:
            display_value = driver["value"]

        st.markdown(
            (
                '<div class="driver-card">'
                '<div class="driver-top">'
                f'<div class="driver-feature">{driver["feature"]}</div>'
                f'<div class="{direction_class}">{direction_label}</div>'
                '</div>'
                '<div class="driver-meta">'
                f'Patient value: {display_value} · '
                f'SHAP impact: {driver["impact"]:.3f}'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# REGISTRY EXAMPLES
# ============================================================

def get_registry_example(
    risk_level,
    target_probability,
):

    candidates = registry[
        registry["risk_level"] == risk_level
    ].copy()

    candidates["distance_from_target"] = (
        candidates[
            "readmission_probability_percent"
        ]
        - target_probability
    ).abs()

    return (
        candidates
        .sort_values("distance_from_target")
        .iloc[0]
    )


def safe_number(
    value,
    fallback,
):

    if pd.isna(value):
        return fallback

    return value


def row_to_defaults(row):

    gender_value = str(
        row["gender"]
    ).upper()

    gender_label = (
        "Male"
        if gender_value == "M"
        else "Female"
    )

    return {
        "age":
            int(
                safe_number(
                    row["age_at_admission"],
                    65,
                )
            ),

        "gender":
            gender_label,

        "los":
            float(
                safe_number(
                    row["length_of_stay"],
                    5.0,
                )
            ),

        "admissions":
            int(
                safe_number(
                    row[
                        "previous_inpatient_admissions"
                    ],
                    0,
                )
            ),

        "ed_visits":
            int(
                safe_number(
                    row[
                        "previous_emergency_visits"
                    ],
                    0,
                )
            ),

        "encounters":
            int(
                safe_number(
                    row["previous_encounters"],
                    0,
                )
            ),

        "conditions":
            int(
                safe_number(
                    row["condition_count"],
                    0,
                )
            ),

        "medications":
            int(
                safe_number(
                    row["medication_count"],
                    0,
                )
            ),

        "procedures":
            int(
                safe_number(
                    row["procedure_count"],
                    0,
                )
            ),

        "diabetes":
            bool(row["diabetes"]),

        "hypertension":
            bool(row["hypertension"]),

        "heart_failure":
            bool(row["heart_failure"]),

        "kidney_disease":
            bool(row["kidney_disease"]),

        "lung_disease":
            bool(
                row[
                    "chronic_lung_disease"
                ]
            ),

        "bmi":
            float(
                safe_number(
                    row["bmi"],
                    28.1,
                )
            ),

        "systolic":
            int(
                round(
                    safe_number(
                        row["systolic_bp"],
                        117,
                    )
                )
            ),

        "diastolic":
            int(
                round(
                    safe_number(
                        row["diastolic_bp"],
                        78,
                    )
                )
            ),

        "heart_rate":
            int(
                round(
                    safe_number(
                        row["heart_rate"],
                        81,
                    )
                )
            ),

        "respiratory":
            int(
                round(
                    safe_number(
                        row["respiratory_rate"],
                        14,
                    )
                )
            ),

        "glucose":
            float(
                safe_number(
                    row["glucose"],
                    84.2,
                )
            ),

        "source_probability":
            float(
                row[
                    "readmission_probability_percent"
                ]
            ),

        "source_risk_level":
            row["risk_level"],
    }


low_example = row_to_defaults(
    get_registry_example(
        "LOW",
        3.0,
    )
)

moderate_example = row_to_defaults(
    get_registry_example(
        "MODERATE",
        20.0,
    )
)

high_example = row_to_defaults(
    get_registry_example(
        "HIGH",
        50.0,
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
        'Calibrated XGBoost prediction pipeline'
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


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # POPULATION OVERVIEW
    # --------------------------------------------------------

    section_header(
        "Population Overview",
        "Risk distribution and review demand across scored encounters.",
    )

    chart_col, review_col = st.columns(
        [1.45, 0.55],
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
            alt.Chart(risk_df)
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


    # --------------------------------------------------------
    # PRIORITY WORKLIST
    # --------------------------------------------------------

    section_header(
        "Priority Worklist",
        "Review and search the highest-risk patient encounters.",
    )

    filter_col1, filter_col2 = st.columns(
        [0.55, 0.45],
        gap="medium",
    )

    with filter_col1:

        worklist_filter = st.segmented_control(
            "Filter",
            options=[
                "All",
                "Review Recommended",
                "High Risk",
            ],
            default="All",
            label_visibility="collapsed",
        )

    with filter_col2:

        patient_search = st.text_input(
            "Search patient",
            placeholder="Search patient ID...",
            label_visibility="collapsed",
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


    if (
        worklist_filter
        == "Review Recommended"
    ):

        unique_priority = (
            unique_priority[
                unique_priority[
                    "intervention_recommended"
                ]
                == True
            ]
        )


    elif (
        worklist_filter
        == "High Risk"
    ):

        unique_priority = (
            unique_priority[
                unique_priority[
                    "risk_level"
                ]
                == "HIGH"
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
            "previous_inpatient_admissions",
            "previous_emergency_visits",
            "Conditions",
        ]
    ].head(15).copy()


    queue = queue.rename(
        columns={
            "risk_level":
                "Risk Tier",

            "previous_inpatient_admissions":
                "Prior Admissions",

            "previous_emergency_visits":
                "ED Visits",
        }
    )


    st.dataframe(
        queue,
        use_container_width=True,
        hide_index=True,
        height=430,

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

        v1, v2 = st.columns(2)

        with v1:

            st.metric(
                "ROC-AUC",
                "0.913",
            )

        with v2:

            st.metric(
                "PR-AUC",
                "0.633",
            )

        st.caption(
            "Synthetic Synthea validation results. "
            "Not intended as clinical validation."
        )


# ============================================================
# PATIENT REVIEW
# ============================================================

elif page == "Patient Review":

    page_header(
        "PATIENT REVIEW",
        "Individual Risk Profile",
        "Explore patient risk, utilization, clinical context, and model drivers.",
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
        ): patient_id

        for patient_id in patient_ids
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


    # --------------------------------------------------------
    # PATIENT SUMMARY
    # --------------------------------------------------------

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
            "Based on intervention threshold",
            "metric-cyan",
        )

    with c4:

        metric_card(
            "Prior Admissions",
            int(
                patient[
                    "previous_inpatient_admissions"
                ]
            ),
            "Before this encounter",
            "metric-pink",
        )

    st.write("")


    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    section_header(
        "Why This Patient Scored This Way",
        "Patient-specific SHAP contributions from the XGBoost model.",
    )

    patient_data = {
        "age_at_admission":
            patient["age_at_admission"],

        "gender":
            patient["gender"],

        "length_of_stay":
            patient["length_of_stay"],

        "previous_inpatient_admissions":
            patient[
                "previous_inpatient_admissions"
            ],

        "previous_emergency_visits":
            patient[
                "previous_emergency_visits"
            ],

        "previous_encounters":
            patient[
                "previous_encounters"
            ],

        "condition_count":
            patient[
                "condition_count"
            ],

        "diabetes":
            patient["diabetes"],

        "hypertension":
            patient["hypertension"],

        "heart_failure":
            patient["heart_failure"],

        "kidney_disease":
            patient["kidney_disease"],

        "chronic_lung_disease":
            patient[
                "chronic_lung_disease"
            ],

        "medication_count":
            patient[
                "medication_count"
            ],

        "procedure_count":
            patient[
                "procedure_count"
            ],

        "bmi":
            patient["bmi"],

        "systolic_bp":
            patient["systolic_bp"],

        "diastolic_bp":
            patient["diastolic_bp"],

        "heart_rate":
            patient["heart_rate"],

        "respiratory_rate":
            patient[
                "respiratory_rate"
            ],

        "glucose":
            patient["glucose"],
    }

    try:

        patient_drivers = (
            explain_readmission(
                patient_data,
                top_n=5,
            )
        )

        render_drivers(
            patient_drivers
        )

        st.caption(
            "SHAP values describe how individual features "
            "influenced this model prediction. They do not "
            "establish clinical causation."
        )

    except Exception:

        st.warning(
            "Risk-driver explanation could not be generated."
        )


    st.write("")


    with st.expander(
        "What does Review Status mean?"
    ):

        st.markdown(
            """
**Review Status** indicates whether this encounter crossed Admitra's
operating threshold for additional review.

It is separate from the patient's **Risk Tier**. Risk Tier describes the
relative level of modeled readmission risk, while Review Status determines
whether that risk is high enough to enter the review queue.

The threshold is an operating rule for this synthetic demonstration and is
not a clinically validated treatment guideline.
"""
        )


    # --------------------------------------------------------
    # UTILIZATION + CLINICAL PROFILE
    # --------------------------------------------------------

    st.write("")

    utilization_col, clinical_col = st.columns(
        2,
        gap="large",
    )

    with utilization_col:

        section_header(
            "Utilization Profile",
            "Healthcare utilization and treatment complexity.",
        )

        utilization = pd.DataFrame(
            {
                "Metric": [
                    "Length of Stay",
                    "Prior Admissions",
                    "ED Visits",
                    "Previous Encounters",
                    "Medications",
                    "Procedures",
                ],

                "Value": [
                    patient[
                        "length_of_stay"
                    ],

                    patient[
                        "previous_inpatient_admissions"
                    ],

                    patient[
                        "previous_emergency_visits"
                    ],

                    patient[
                        "previous_encounters"
                    ],

                    patient[
                        "medication_count"
                    ],

                    patient[
                        "procedure_count"
                    ],
                ],
            }
        )

        st.dataframe(
            utilization,
            use_container_width=True,
            hide_index=True,
        )


    with clinical_col:

        section_header(
            "Clinical Profile",
            "Major comorbid conditions available to the model.",
        )

        clinical = pd.DataFrame(
            {
                "Condition": [
                    "Diabetes",
                    "Hypertension",
                    "Heart Failure",
                    "Kidney Disease",
                    "Chronic Lung Disease",
                ],

                "Present": [
                    bool(
                        patient[
                            "diabetes"
                        ]
                    ),

                    bool(
                        patient[
                            "hypertension"
                        ]
                    ),

                    bool(
                        patient[
                            "heart_failure"
                        ]
                    ),

                    bool(
                        patient[
                            "kidney_disease"
                        ]
                    ),

                    bool(
                        patient[
                            "chronic_lung_disease"
                        ]
                    ),
                ],
            }
        )

        st.dataframe(
            clinical,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # ENCOUNTER HISTORY
    # --------------------------------------------------------

    st.write("")

    section_header(
        "Encounter History",
        "Historical modeled risk and observed outcomes.",
    )

    encounter_table = patient_history[
        [
            "encounter_id",
            "readmission_probability_percent",
            "risk_level",
            "intervention_recommended",
            "actual_readmitted_30_days",
            "length_of_stay",
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
            "length_of_stay",
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

            "length_of_stay":
                "LOS",
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


# ============================================================
# RISK ASSESSMENT
# ============================================================

elif page == "Risk Assessment":

    page_header(
        "RISK ASSESSMENT",
        "Readmission Risk Assessment",
        "Assess a patient manually or load a representative synthetic encounter.",
    )


    # --------------------------------------------------------
    # SESSION STATE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # QUICK START
    # --------------------------------------------------------

    section_header(
        "Quick Start",
        "Load a real synthetic encounter from the scored Synthea population.",
    )

    b1, b2, b3, b4 = st.columns(4)

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


    # --------------------------------------------------------
    # DEFAULTS
    # --------------------------------------------------------

    if preset_name == "Custom":

        defaults = {
            "age": 65,
            "gender": "Female",
            "los": 5.0,
            "admissions": 0,
            "ed_visits": 0,
            "encounters": 5,
            "conditions": 2,
            "medications": 5,
            "procedures": 5,
            "diabetes": False,
            "hypertension": False,
            "heart_failure": False,
            "kidney_disease": False,
            "lung_disease": False,
            "bmi": 28.1,
            "systolic": 117,
            "diastolic": 78,
            "heart_rate": 81,
            "respiratory": 14,
            "glucose": 84.2,
        }

    else:

        defaults = (
            presets[
                preset_name
            ]
        )

        st.info(
            f'{preset_name} example loaded from the '
            f'scored Synthea registry. Its stored modeled '
            f'risk is {defaults["source_probability"]:.1f}% '
            f'({defaults["source_risk_level"]}).'
        )


    key_suffix = (
        st.session_state[
            "preset_counter"
        ]
    )


    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    st.write("")

    section_header(
        "Patient",
        "Core patient and hospitalization information.",
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
                defaults["age"]
            ),
            key=f"age_{key_suffix}",
        )


    with p2:

        gender_options = [
            "Female",
            "Male",
        ]

        gender_index = (
            0
            if defaults[
                "gender"
            ]
            == "Female"
            else 1
        )

        gender = st.selectbox(
            "Gender",
            gender_options,
            index=gender_index,
            key=f"gender_{key_suffix}",
        )


    with p3:

        length_of_stay = (
            st.number_input(
                "Length of Stay (days)",
                min_value=0.0,
                value=float(
                    defaults["los"]
                ),
                step=0.5,
                key=f"los_{key_suffix}",
            )
        )


    # --------------------------------------------------------
    # UTILIZATION
    # --------------------------------------------------------

    st.write("")

    section_header(
        "Recent Utilization",
        "Prior healthcare utilization available to the model.",
    )

    u1, u2, u3 = st.columns(
        3,
        gap="medium",
    )


    with u1:

        previous_inpatient_admissions = (
            st.number_input(
                "Prior Admissions",
                min_value=0,
                value=int(
                    defaults[
                        "admissions"
                    ]
                ),
                key=(
                    f"admissions_"
                    f"{key_suffix}"
                ),
            )
        )


    with u2:

        previous_emergency_visits = (
            st.number_input(
                "ED Visits",
                min_value=0,
                value=int(
                    defaults[
                        "ed_visits"
                    ]
                ),
                key=f"ed_{key_suffix}",
            )
        )


    with u3:

        previous_encounters = (
            st.number_input(
                "Prior Encounters",
                min_value=0,
                value=int(
                    defaults[
                        "encounters"
                    ]
                ),
                key=(
                    f"encounters_"
                    f"{key_suffix}"
                ),
            )
        )


    # --------------------------------------------------------
    # CONDITIONS
    # --------------------------------------------------------

    st.write("")

    section_header(
        "Clinical Conditions",
        "Select documented conditions for this patient.",
    )

    cc1, cc2, cc3 = st.columns(
        3,
        gap="medium",
    )


    with cc1:

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


    with cc2:

        heart_failure = st.checkbox(
            "Heart Failure",
            value=defaults[
                "heart_failure"
            ],
            key=(
                f"heart_failure_"
                f"{key_suffix}"
            ),
        )

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


    with cc3:

        chronic_lung_disease = (
            st.checkbox(
                "Chronic Lung Disease",
                value=defaults[
                    "lung_disease"
                ],
                key=(
                    f"lung_"
                    f"{key_suffix}"
                ),
            )
        )


    # --------------------------------------------------------
    # TREATMENT DETAILS
    # --------------------------------------------------------

    with st.expander(
        "Treatment Details",
        expanded=False,
    ):

        t1, t2, t3 = st.columns(3)

        with t1:

            condition_count = (
                st.number_input(
                    "Active Conditions",
                    min_value=0,
                    value=int(
                        defaults[
                            "conditions"
                        ]
                    ),
                    key=(
                        f"condition_count_"
                        f"{key_suffix}"
                    ),
                )
            )

        with t2:

            medication_count = (
                st.number_input(
                    "Medications",
                    min_value=0,
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
            )

        with t3:

            procedure_count = (
                st.number_input(
                    "Procedures",
                    min_value=0,
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
            )


    # --------------------------------------------------------
    # VITALS & LABS
    # --------------------------------------------------------

    with st.expander(
        "Vitals & Labs",
        expanded=False,
    ):

        v1, v2, v3 = st.columns(
            3,
            gap="medium",
        )


        with v1:

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
                key=f"bmi_{key_suffix}",
            )

            heart_rate = (
                st.number_input(
                    "Heart Rate",
                    min_value=20,
                    max_value=250,
                    value=int(
                        defaults[
                            "heart_rate"
                        ]
                    ),
                    key=(
                        f"heart_rate_"
                        f"{key_suffix}"
                    ),
                )
            )


        with v2:

            systolic_bp = (
                st.number_input(
                    "Systolic BP",
                    min_value=50,
                    max_value=250,
                    value=int(
                        defaults[
                            "systolic"
                        ]
                    ),
                    key=(
                        f"systolic_"
                        f"{key_suffix}"
                    ),
                )
            )

            respiratory_rate = (
                st.number_input(
                    "Respiratory Rate",
                    min_value=5,
                    max_value=60,
                    value=int(
                        defaults[
                            "respiratory"
                        ]
                    ),
                    key=(
                        f"respiratory_"
                        f"{key_suffix}"
                    ),
                )
            )


        with v3:

            diastolic_bp = (
                st.number_input(
                    "Diastolic BP",
                    min_value=20,
                    max_value=180,
                    value=int(
                        defaults[
                            "diastolic"
                        ]
                    ),
                    key=(
                        f"diastolic_"
                        f"{key_suffix}"
                    ),
                )
            )

            glucose = st.number_input(
                "Glucose",
                min_value=20.0,
                max_value=600.0,
                value=float(
                    defaults[
                        "glucose"
                    ]
                ),
                step=0.1,
                key=(
                    f"glucose_"
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

        gender_code = (
            "F"
            if gender == "Female"
            else "M"
        )

        patient_data = {
            "age_at_admission":
                age,

            "gender":
                gender_code,

            "length_of_stay":
                length_of_stay,

            "previous_inpatient_admissions":
                previous_inpatient_admissions,

            "previous_emergency_visits":
                previous_emergency_visits,

            "previous_encounters":
                previous_encounters,

            "condition_count":
                condition_count,

            "diabetes":
                int(diabetes),

            "hypertension":
                int(hypertension),

            "heart_failure":
                int(heart_failure),

            "kidney_disease":
                int(kidney_disease),

            "chronic_lung_disease":
                int(
                    chronic_lung_disease
                ),

            "medication_count":
                medication_count,

            "procedure_count":
                procedure_count,

            "bmi":
                bmi,

            "systolic_bp":
                systolic_bp,

            "diastolic_bp":
                diastolic_bp,

            "heart_rate":
                heart_rate,

            "respiratory_rate":
                respiratory_rate,

            "glucose":
                glucose,
        }


        try:

            result = (
                predict_readmission(
                    patient_data
                )
            )

            drivers = (
                explain_readmission(
                    patient_data,
                    top_n=5,
                )
            )

            st.write("")

            section_header(
                "Risk Intelligence",
                "Calibrated readmission prediction and review status.",
            )

            r1, r2, r3 = st.columns(
                3,
                gap="medium",
            )


            with r1:

                metric_card(
                    "30-Day Risk",
                    (
                        f'{result["probability_percent"]}%'
                    ),
                    "Calibrated probability",
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
                    "Based on intervention threshold",
                    "metric-cyan",
                )


            st.write("")


            if result[
                "intervention_recommended"
            ]:

                st.warning(
                    f'**Additional review recommended.** '
                    f'This encounter exceeds Admitra’s '
                    f'{result["production_threshold_percent"]}% '
                    f'intervention threshold.'
                )

            else:

                st.success(
                    f'**No additional review flag.** '
                    f'This encounter remains below Admitra’s '
                    f'{result["production_threshold_percent"]}% '
                    f'intervention threshold.'
                )


            with st.expander(
                "What is the intervention threshold?"
            ):

                st.markdown(
                    f"""
The **{result["production_threshold_percent"]}% intervention threshold**
is the predicted readmission probability at which Admitra places an
encounter into the review queue.

During model development, the threshold was selected using
**out-of-fold validation**. Among the evaluated thresholds, the operating
cutoff was chosen to maximize **F1 score while maintaining at least 70%
recall**.

In practical terms:

- **Below {result["production_threshold_percent"]}%:** the encounter is not automatically flagged.
- **At or above {result["production_threshold_percent"]}%:** additional review is recommended.

The **Risk Tier** and **Review Status** serve different purposes. Risk Tier
describes where the patient's modeled probability falls within the
population, while Review Status determines whether that probability crosses
the operating cutoff used to prioritize encounters.

This threshold was developed for this synthetic demonstration and is not a
clinically validated treatment guideline.
"""
                )


            st.write("")


            section_header(
                "Why This Patient Scored This Way",
                "The strongest patient-specific contributors to the prediction.",
            )

            render_drivers(
                drivers
            )

            st.caption(
                "SHAP values describe the model's learned "
                "prediction behavior. They do not establish "
                "clinical causation."
            )


            with st.expander(
                "Technical Prediction Details"
            ):

                st.write(
                    f'**Calibrated probability:** '
                    f'{result["probability_percent"]}%'
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
                    f'**Intervention threshold:** '
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
        'Synthea healthcare data. Model outputs are not intended '
        'for clinical use.'
        '</div>'
    ),
    unsafe_allow_html=True,
)