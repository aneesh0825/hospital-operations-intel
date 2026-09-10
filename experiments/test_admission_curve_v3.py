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


base_patient = {
    "age_at_admission": 60,
    "length_of_stay": 5.0,
    "previous_inpatient_admissions": 0,
    "previous_encounters": 20,
    "condition_count": 15,
    "medication_count": 8,
    "procedure_count": 20,
    "diabetes": 1,
    "hypertension": 1,
    "kidney_disease": 0,
    "bmi": 29.0,
}


print("=" * 65)
print("V3 PRIOR ADMISSION RESPONSE")
print("=" * 65)

for admissions in [
    0,
    1,
    2,
    3,
    5,
    10,
    15,
    20,
    28,
    40,
    60,
]:

    patient = base_patient.copy()

    patient[
        "previous_inpatient_admissions"
    ] = admissions

    X = pd.DataFrame(
        [[patient[f] for f in FEATURES]],
        columns=FEATURES,
    )

    raw = model.predict_proba(
        X
    )[0, 1]

    calibrated = calibrator.predict(
        [raw]
    )[0]

    print(
        f"Admissions: {admissions:>2} | "
        f"Raw: {raw * 100:>6.1f}% | "
        f"Calibrated: {calibrated * 100:>6.1f}%"
    )