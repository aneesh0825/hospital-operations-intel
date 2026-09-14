import pandas as pd

from services.registry import condition_summary, encounter_comparison, row_to_model_input, short_patient_id, validate_registry


def test_short_patient_id_uses_final_six_characters():
    assert short_patient_id("a1b2c3d4-e5f6") == "P-4-E5F6"


def test_condition_summary_lists_known_flagged_conditions():
    assert condition_summary({"diabetes": 1, "hypertension": 0, "kidney_disease": 1}) == "Diabetes, Kidney Disease"
    assert condition_summary({}) == "None flagged"


def test_row_to_model_input_returns_only_production_features():
    row = {
        "age_at_admission": 70, "length_of_stay": 5, "previous_inpatient_admissions": 4,
        "admissions_last_30_days": 1, "admissions_last_90_days": 2, "admissions_last_365_days": 3,
        "days_since_last_inpatient_admission": 12, "has_prior_inpatient_admission": 1,
        "prior_readmissions_last_365_days": 1, "condition_count": 4, "medication_count": 5,
        "procedure_count": 6, "diabetes": 0, "hypertension": 1, "kidney_disease": 0, "bmi": 28,
        "unrelated": "excluded",
    }
    model_input = row_to_model_input(row)
    assert "unrelated" not in model_input
    assert len(model_input) == 16


def test_registry_validation_reports_missing_columns():
    result = validate_registry(pd.DataFrame({"patient_id": ["p1"]}))
    assert not result.is_valid
    assert "Missing required columns" in result.errors[0]


def test_registry_validation_reports_invalid_probability_and_windows(registry_row):
    registry = pd.DataFrame([{**registry_row, "readmission_probability": 1.2, "admissions_last_30_days": 4, "admissions_last_90_days": 2}])
    warnings = validate_registry(registry).warnings
    assert any("readmission_probability" in warning for warning in warnings)
    assert any("nested admission windows" in warning for warning in warnings)


def test_encounter_comparison_uses_immediate_preceding_encounter(registry_row):
    history = pd.DataFrame([
        {**registry_row, "encounter_id": "first", "readmission_probability_percent": 10, "admissions_last_30_days": 0},
        {**registry_row, "encounter_id": "second", "readmission_probability_percent": 30, "admissions_last_30_days": 2},
    ])
    comparison = encounter_comparison(history, "second")
    assert comparison is not None
    assert comparison.loc[comparison["Metric"] == "Modeled readmission risk", "Change"].iloc[0] == 20
    assert encounter_comparison(history, "first") is None
