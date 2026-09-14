from pathlib import Path

APP_TITLE = "Admitra | Hospital Intelligence"
APP_ICON = "A"
REGISTRY_PATH = Path("data/processed/admitra_patient_registry.csv")
PRODUCTION_THRESHOLD_PERCENT = 22.0
MODEL_LABEL = "History-aware calibrated XGBoost model"

VALIDATION_METRICS = {
    "ROC-AUC": "0.932",
    "PR-AUC": "0.754",
    "Brier Score": "0.0559",
}

REQUIRED_REGISTRY_COLUMNS = frozenset({
    "encounter_id", "patient_id", "readmission_probability",
    "readmission_probability_percent", "risk_level", "intervention_recommended",
    "actual_readmitted_30_days", "age_at_admission", "length_of_stay",
    "previous_inpatient_admissions", "admissions_last_30_days",
    "admissions_last_90_days", "admissions_last_365_days",
    "days_since_last_inpatient_admission", "has_prior_inpatient_admission",
    "prior_readmissions_last_365_days", "condition_count", "medication_count",
    "procedure_count", "diabetes", "hypertension", "kidney_disease", "bmi",
})

MODEL_FEATURES = (
    "age_at_admission", "length_of_stay", "previous_inpatient_admissions",
    "admissions_last_30_days", "admissions_last_90_days",
    "admissions_last_365_days", "days_since_last_inpatient_admission",
    "has_prior_inpatient_admission", "prior_readmissions_last_365_days",
    "condition_count", "medication_count", "procedure_count", "diabetes",
    "hypertension", "kidney_disease", "bmi",
)
