import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path("models")

model = joblib.load(
    MODEL_DIR / "readmission_xgboost_v3.joblib"
)

calibrator = joblib.load(
    MODEL_DIR / "readmission_calibrator_v3.joblib"
)

with open(
    MODEL_DIR / "readmission_feature_columns_v3.json"
) as f:
    FEATURES = json.load(f)


def predict_v3(patient):

    X = pd.DataFrame(
        [[patient[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    raw_probability = model.predict_proba(X)[:, 1][0]

    calibrated_probability = calibrator.predict(
        [raw_probability]
    )[0]

    return calibrated_probability * 100


def test_patient(name, **changes):

    patient = {
        "age_at_admission": 60,
        "length_of_stay": 5.0,
        "previous_inpatient_admissions": 0,
        "previous_encounters": 5,
        "condition_count": 2,
        "medication_count": 5,
        "procedure_count": 3,
        "diabetes": 0,
        "hypertension": 0,
        "kidney_disease": 0,
        "bmi": 27.0,
    }

    patient.update(changes)

    probability = predict_v3(patient)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"Probability: {probability:.1f}%")


test_patient(
    "1. BASELINE"
)

test_patient(
    "2. HIGH PRIOR UTILIZATION",
    previous_inpatient_admissions=10,
    previous_encounters=20,
)

test_patient(
    "3. HIGH CLINICAL BURDEN",
    condition_count=15,
    medication_count=11,
    procedure_count=18,
    diabetes=1,
    hypertension=1,
    kidney_disease=1,
)

test_patient(
    "4. UTILIZATION + CLINICAL BURDEN",
    previous_inpatient_admissions=10,
    previous_encounters=20,
    condition_count=15,
    medication_count=11,
    procedure_count=18,
    diabetes=1,
    hypertension=1,
    kidney_disease=1,
)

test_patient(
    "5. VERY HIGH PRIOR UTILIZATION",
    previous_inpatient_admissions=28,
    previous_encounters=60,
)

test_patient(
    "6. VERY HIGH OVERALL BURDEN",
    age_at_admission=76,
    length_of_stay=8.5,
    previous_inpatient_admissions=28,
    previous_encounters=60,
    condition_count=27,
    medication_count=12,
    procedure_count=36,
    diabetes=1,
    hypertension=1,
    kidney_disease=1,
    bmi=31.0,
)