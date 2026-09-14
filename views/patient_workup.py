from __future__ import annotations

import logging

import altair as alt
import pandas as pd
import streamlit as st

from config.constants import PRODUCTION_THRESHOLD_PERCENT
from services.registry import encounter_comparison, ordered_patient_history, row_to_model_input, short_patient_id
from src.predict_readmission import explain_readmission
from ui.components import page_header, render_drivers, risk_signal, section_header

LOGGER = logging.getLogger(__name__)


def _sync_selected_patient(labels: dict[str, object]) -> None:
    st.session_state["selected_patient_id"] = labels[st.session_state["workup_patient"]]


def _render_progression(history: pd.DataFrame) -> None:
    base = alt.Chart(history).encode(
        x=alt.X("Encounter Sequence:O", title="Encounter sequence"),
        y=alt.Y("readmission_probability_percent:Q", title="Calibrated risk (%)", scale=alt.Scale(domain=[0, 100])),
        tooltip=[
            alt.Tooltip("Encounter Sequence:O", title="Encounter"),
            alt.Tooltip("readmission_probability_percent:Q", title="Risk", format=".1f"),
            alt.Tooltip("risk_level:N", title="Risk tier"),
            alt.Tooltip("actual_readmitted_30_days:Q", title="Observed 30-day readmission"),
        ],
    )
    line = base.mark_line(point=True, strokeWidth=2.5, color="#38d9d0")
    threshold = alt.Chart(pd.DataFrame({"threshold": [PRODUCTION_THRESHOLD_PERCENT]})).mark_rule(color="#f3c95f", strokeDash=[6, 4], strokeWidth=2).encode(y="threshold:Q")
    observed = base.transform_filter(alt.datum.actual_readmitted_30_days == 1).mark_point(shape="diamond", size=110, filled=True, color="#ff7485")
    st.altair_chart((line + threshold + observed).properties(height=245), width="stretch")
    st.caption("Dashed line: 22% operational review threshold. Diamonds: observed 30-day readmissions.")


def _render_comparison(history: pd.DataFrame, patient: pd.Series) -> None:
    section_header("What Changed?", "Comparison uses the preceding encounter in cumulative-utilization order; encounter dates are not available in this registry.", "ENCOUNTER COMPARISON")
    comparison = encounter_comparison(history, patient["encounter_id"])
    if comparison is None:
        st.info("No preceding encounter exists for this patient under the available encounter-order proxy.")
        return
    display = comparison.copy()
    for column in ("Previous", "Current", "Change"):
        display[column] = display[column].map(lambda value: round(value, 1))
    display["Change"] = display["Change"].map(lambda value: f"{value:+.1f}")
    st.dataframe(display[["Metric", "Previous", "Current", "Change", "Unit"]], width="stretch", hide_index=True)


def render(registry: pd.DataFrame) -> None:
    page_header("PATIENT WORKUP", "Patient Risk Profile", "Review calibrated risk, longitudinal utilization, and the encounter signals driving operational review.")
    patient_ids = registry["patient_id"].dropna().drop_duplicates().tolist()
    labels = {short_patient_id(patient_id): patient_id for patient_id in patient_ids}
    pending_id = st.session_state.get("selected_patient_id")
    if pending_id is not None:
        st.session_state["workup_patient"] = short_patient_id(pending_id)
    selected_label = st.selectbox("Search Patient", sorted(labels), key="workup_patient", on_change=_sync_selected_patient, args=(labels,))
    history = ordered_patient_history(registry, labels[selected_label])
    patient = history.sort_values("readmission_probability", ascending=False).iloc[0]
    st.html(f'<div class="patient-banner"><strong>{selected_label}</strong><br><small>{len(history)} recorded hospitalization(s) · highest-risk scored encounter selected</small></div>')
    signal_col, driver_col = st.columns((0.44, 0.56), gap="large")
    with signal_col:
        with st.container(border=True):
            risk_signal(float(patient["readmission_probability_percent"]), str(patient["risk_level"]), PRODUCTION_THRESHOLD_PERCENT, bool(patient["intervention_recommended"]))
    with driver_col:
        with st.container(border=True):
            section_header("Why this encounter surfaced", "The highest-impact patient-specific model contributors.", "SHAP")
            try:
                render_drivers(explain_readmission(row_to_model_input(patient), top_n=5))
            except (KeyError, TypeError, ValueError, AttributeError) as error:
                LOGGER.exception("Unable to generate workup explanation for encounter %s", patient["encounter_id"])
                st.warning("Risk-driver explanation could not be generated for this encounter.")
                if st.session_state.get("debug", False):
                    st.exception(error)

    with st.container(border=True):
        section_header("Patient risk trajectory", "The selected patient's calibrated risk across recorded encounters, ordered by cumulative prior inpatient utilization.", "SIGNATURE VIEW")
        _render_progression(history)

    comparison_col, history_col = st.columns((0.55, 0.45), gap="large")
    with comparison_col:
        with st.container(border=True):
            _render_comparison(history, patient)
    with history_col:
        with st.container(border=True):
            section_header("Encounter history", "Recorded utilization context for this patient.", "HISTORY")
            history_table = history[["Encounter Sequence", "readmission_probability_percent", "risk_level", "actual_readmitted_30_days"]].copy()
            history_table = history_table.rename(columns={"readmission_probability_percent": "Risk (%)", "risk_level": "Risk tier", "actual_readmitted_30_days": "Observed readmission"})
            history_table["Risk (%)"] = history_table["Risk (%)"].round(1)
            st.dataframe(history_table, width="stretch", hide_index=True, height=285)

    utilization_col, clinical_col = st.columns(2, gap="large")
    with utilization_col:
        with st.container(border=True):
            section_header("Recent utilization", "Recent inpatient utilization and known readmission history.", "UTILIZATION")
            metrics = [
                ("Admissions in last 30 days", "admissions_last_30_days"), ("Admissions in last 90 days", "admissions_last_90_days"),
                ("Admissions in last 365 days", "admissions_last_365_days"), ("Prior readmissions in last 365 days", "prior_readmissions_last_365_days"),
                ("Days since last inpatient admission", "days_since_last_inpatient_admission"), ("Lifetime prior admissions", "previous_inpatient_admissions"),
            ]
            st.dataframe(pd.DataFrame({"Metric": [label for label, _ in metrics], "Value": [patient[key] for _, key in metrics]}), width="stretch", hide_index=True)
    with clinical_col:
        with st.container(border=True):
            section_header("Clinical profile", "Clinical complexity and documented conditions in the registry.", "PROFILE")
            metrics = [
                ("Age", "age_at_admission"), ("Length of stay", "length_of_stay"), ("Condition count", "condition_count"),
                ("Medication count", "medication_count"), ("Procedure count", "procedure_count"), ("BMI", "bmi"),
                ("Diabetes", "diabetes"), ("Hypertension", "hypertension"), ("Kidney disease", "kidney_disease"),
            ]
            st.dataframe(pd.DataFrame({"Metric": [label for label, _ in metrics], "Value": [patient[key] for _, key in metrics]}), width="stretch", hide_index=True)
