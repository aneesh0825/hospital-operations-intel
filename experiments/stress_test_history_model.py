import json
from pathlib import Path

import joblib
import pandas as pd


MODEL_DIR = Path("models")

model = joblib.load(
    MODEL_DIR / "readmission_history_xgboost_calibrated.joblib"
)

calibrator = joblib.load(
    MODEL_DIR / "readmission_history_calibrator.joblib"
)

with open(
    MODEL_DIR / "readmission_history_features.json"
) as f:
    FEATURES = json.load(f)


def predict(patient):

    X = pd.DataFrame(
        [[patient[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    raw = model.predict_proba(X)[0, 1]

    calibrated = calibrator.predict(
        [raw]
    )[0]

    return raw * 100, calibrated * 100


def test_patient(name, **changes):

    patient = {
        "age_at_admission": 60,
        "length_of_stay": 5.0,

        "previous_inpatient_admissions": 0,

        "admissions_last_30_days": 0,
        "admissions_last_90_days": 0,
        "admissions_last_365_days": 0,

        "days_since_last_inpatient_admission": 365,
        "has_prior_inpatient_admission": 0,

        "prior_readmissions_last_365_days": 0,

        "condition_count": 2,
        "medication_count": 5,
        "procedure_count": 3,

        "diabetes": 0,
        "hypertension": 0,
        "kidney_disease": 0,

        "bmi": 27.0,
    }

    patient.update(changes)

    raw, calibrated = predict(
        patient
    )

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Raw probability: "
        f"{raw:.1f}%"
    )

    print(
        f"Calibrated probability: "
        f"{calibrated:.1f}%"
    )


test_patient(
    "1. NO RECENT HISTORY"
)


test_patient(
    "2. ONE ADMISSION IN LAST 90 DAYS",
    previous_inpatient_admissions=1,
    admissions_last_90_days=1,
    admissions_last_365_days=1,
    days_since_last_inpatient_admission=60,
    has_prior_inpatient_admission=1,
)


test_patient(
    "3. TWO ADMISSIONS IN LAST 90 DAYS",
    previous_inpatient_admissions=2,
    admissions_last_90_days=2,
    admissions_last_365_days=2,
    days_since_last_inpatient_admission=30,
    has_prior_inpatient_admission=1,
)


test_patient(
    "4. THREE ADMISSIONS IN LAST 90 DAYS",
    previous_inpatient_admissions=3,
    admissions_last_30_days=1,
    admissions_last_90_days=3,
    admissions_last_365_days=3,
    days_since_last_inpatient_admission=15,
    has_prior_inpatient_admission=1,
)


test_patient(
    "5. PRIOR READMISSION HISTORY",
    previous_inpatient_admissions=3,
    admissions_last_90_days=2,
    admissions_last_365_days=3,
    days_since_last_inpatient_admission=30,
    has_prior_inpatient_admission=1,
    prior_readmissions_last_365_days=2,
)


test_patient(
    "6. HIGH RECENT UTILIZATION + CLINICAL BURDEN",
    age_at_admission=76,
    length_of_stay=8.5,

    previous_inpatient_admissions=8,

    admissions_last_30_days=2,
    admissions_last_90_days=3,
    admissions_last_365_days=5,

    days_since_last_inpatient_admission=10,
    has_prior_inpatient_admission=1,

    prior_readmissions_last_365_days=3,

    condition_count=27,
    medication_count=12,
    procedure_count=36,

    diabetes=1,
    hypertension=1,
    kidney_disease=1,

    bmi=31.0,
)