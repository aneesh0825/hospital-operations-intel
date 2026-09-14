import pytest


@pytest.fixture
def registry_row():
    return {
        "encounter_id": "encounter", "patient_id": "patient", "readmission_probability": 0.2,
        "readmission_probability_percent": 20.0, "risk_level": "LOW", "intervention_recommended": False,
        "actual_readmitted_30_days": 0, "age_at_admission": 60, "length_of_stay": 4.0,
        "previous_inpatient_admissions": 3, "admissions_last_30_days": 1, "admissions_last_90_days": 2,
        "admissions_last_365_days": 3, "days_since_last_inpatient_admission": 10.0,
        "has_prior_inpatient_admission": 1, "prior_readmissions_last_365_days": 1,
        "condition_count": 4, "medication_count": 5, "procedure_count": 6, "diabetes": 0,
        "hypertension": 1, "kidney_disease": 0, "bmi": 27.0,
    }
