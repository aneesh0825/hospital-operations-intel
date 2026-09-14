from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from config.constants import PRODUCTION_THRESHOLD_PERCENT
from services.registry import row_to_model_input
from src.predict_readmission import explain_readmission, predict_readmission
from ui.components import page_header, render_drivers, risk_signal, section_header

LOGGER = logging.getLogger(__name__)


def _registry_defaults(registry: pd.DataFrame, risk_level: str) -> dict[str, object]:
    candidates = registry.loc[registry["risk_level"] == risk_level].copy()
    target = {"LOW": 10, "MODERATE": 45, "HIGH": 80}[risk_level]
    selected = candidates.iloc[(candidates["readmission_probability_percent"] - target).abs().argsort().iloc[0]]
    return row_to_model_input(selected)


def _custom_defaults() -> dict[str, object]:
    return {
        "age_at_admission": 65, "length_of_stay": 5.0, "previous_inpatient_admissions": 0,
        "admissions_last_30_days": 0, "admissions_last_90_days": 0, "admissions_last_365_days": 0,
        "days_since_last_inpatient_admission": 365.0, "has_prior_inpatient_admission": 0,
        "prior_readmissions_last_365_days": 0, "condition_count": 2, "medication_count": 5,
        "procedure_count": 3, "diabetes": 0, "hypertension": 0, "kidney_disease": 0, "bmi": 27.0,
    }


def render(registry: pd.DataFrame) -> None:
    page_header("RISK ASSESSMENT", "Readmission Risk Assessment", "Score a current encounter with the production history-aware model, then review the calibrated result and its strongest contributors.")
    preset = st.radio("Starting point", ["Custom", "Low Risk", "Moderate Risk", "High Risk"], horizontal=True, key="assessment_preset")
    defaults = _custom_defaults() if preset == "Custom" else _registry_defaults(registry, preset.split()[0].upper())
    if preset != "Custom":
        st.info(f"{preset} example loaded from the scored synthetic registry.")

    input_column, summary_column = st.columns((0.58, 0.42), gap="large")
    with input_column:
        with st.container(border=True):
            section_header("Patient inputs", "Current encounter, longitudinal history, and clinical complexity.", "16 FEATURES")
            with st.form("risk_assessment"):
                left, right = st.columns(2)
                with left:
                    age = st.number_input("Age", 0, 120, int(defaults["age_at_admission"]))
                    length_of_stay = st.number_input("Length of Stay", 0.0, value=float(defaults["length_of_stay"]), step=0.5)
                    bmi = st.number_input("BMI", 10.0, 80.0, float(defaults["bmi"]), step=0.1)
                    condition_count = st.number_input("Active Conditions", 0, 100, int(defaults["condition_count"]))
                    medication_count = st.number_input("Medication Count", 0, 100, int(defaults["medication_count"]))
                    procedure_count = st.number_input("Procedure Count", 0, 200, int(defaults["procedure_count"]))
                with right:
                    admissions_30 = st.number_input("Admissions / 30 Days", 0, 20, int(defaults["admissions_last_30_days"]))
                    admissions_90 = st.number_input("Admissions / 90 Days", 0, 30, int(defaults["admissions_last_90_days"]))
                    admissions_365 = st.number_input("Admissions / 365 Days", 0, 100, int(defaults["admissions_last_365_days"]))
                    previous_admissions = st.number_input("Total Prior Admissions", 0, 500, int(defaults["previous_inpatient_admissions"]))
                    prior_readmissions = st.number_input("Prior Readmissions / Year", 0, 50, int(defaults["prior_readmissions_last_365_days"]))
                    days_since = st.number_input("Days Since Last Admission", 0.0, 365.0, float(defaults["days_since_last_inpatient_admission"]), step=1.0)
                c1, c2, c3 = st.columns(3)
                diabetes = c1.checkbox("Diabetes", value=bool(defaults["diabetes"]))
                hypertension = c2.checkbox("Hypertension", value=bool(defaults["hypertension"]))
                kidney_disease = c3.checkbox("Kidney Disease", value=bool(defaults["kidney_disease"]))
                analyze = st.form_submit_button("Analyze readmission risk", type="primary", width="stretch")

    with summary_column:
        with st.container(border=True):
            section_header("Assessment summary", "Calibrated probability, review status, and model explanation.", "22% THRESHOLD")
            if not analyze:
                st.markdown("Enter encounter details and run the assessment to see the calibrated readmission estimate.")
                st.caption(f"The {PRODUCTION_THRESHOLD_PERCENT:.0f}% operational threshold prioritizes review. It is not a treatment recommendation.")
                return
            patient_data = {
                "age_at_admission": age, "length_of_stay": length_of_stay, "previous_inpatient_admissions": previous_admissions,
                "admissions_last_30_days": admissions_30, "admissions_last_90_days": admissions_90,
                "admissions_last_365_days": admissions_365, "days_since_last_inpatient_admission": days_since,
                "has_prior_inpatient_admission": int(previous_admissions > 0 or admissions_365 > 0),
                "prior_readmissions_last_365_days": prior_readmissions, "condition_count": condition_count,
                "medication_count": medication_count, "procedure_count": procedure_count, "diabetes": int(diabetes),
                "hypertension": int(hypertension), "kidney_disease": int(kidney_disease), "bmi": bmi,
            }
            if admissions_30 > admissions_90 or admissions_90 > admissions_365 or admissions_365 > previous_admissions:
                st.warning("Admission windows are inconsistent. The production scorer will normalize them before scoring.")
            try:
                result = predict_readmission(patient_data)
                risk_signal(result["probability_percent"], result["risk_level"], result["production_threshold_percent"], result["intervention_recommended"])
                section_header("Why the estimate moved", "The strongest patient-specific contributors.", "SHAP")
                render_drivers(explain_readmission(patient_data, top_n=5))
            except (KeyError, TypeError, ValueError, OSError, AttributeError) as error:
                LOGGER.exception("Unable to generate risk assessment")
                st.error("Unable to generate the assessment. Check the entered values and try again.")
                if st.session_state.get("debug", False):
                    st.exception(error)
