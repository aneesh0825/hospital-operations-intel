import json
import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/readmission_xgboost_v2.joblib"
CALIBRATOR_PATH = "models/readmission_calibrator_v2.joblib"
FEATURE_COLUMNS_PATH = "models/readmission_feature_columns_v2.json"
MEDIANS_PATH = "models/readmission_clinical_medians_v2.json"
THRESHOLD_PATH = "models/readmission_threshold_v2.json"
RISK_BANDS_PATH = "models/readmission_risk_bands_v2.json"


# ============================================================
# LOAD PRODUCTION ARTIFACTS
# ============================================================

model = joblib.load(MODEL_PATH)
calibrator = joblib.load(CALIBRATOR_PATH)

with open(FEATURE_COLUMNS_PATH, "r") as file:
    feature_columns = json.load(file)

with open(MEDIANS_PATH, "r") as file:
    clinical_medians = json.load(file)

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

with open(RISK_BANDS_PATH, "r") as file:
    risk_bands = json.load(file)


production_threshold = float(
    threshold_data["threshold"]
)

low_cutoff = float(
    risk_bands["low_cutoff"]
)

high_cutoff = float(
    risk_bands["high_cutoff"]
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# CLINICAL FEATURES
# ============================================================

clinical_features = [
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose"
]


# ============================================================
# FRIENDLY FEATURE NAMES
# ============================================================

friendly_feature_names = {
    "age_at_admission": "Age",
    "length_of_stay": "Length of stay",
    "previous_inpatient_admissions": "Previous inpatient admissions",
    "previous_emergency_visits": "Previous emergency visits",
    "previous_encounters": "Previous healthcare encounters",
    "condition_count": "Condition burden",
    "diabetes": "Diabetes",
    "hypertension": "Hypertension",
    "heart_failure": "Heart failure",
    "kidney_disease": "Kidney disease",
    "chronic_lung_disease": "Chronic lung disease",
    "medication_count": "Medication count",
    "procedure_count": "Procedure count",
    "bmi": "BMI",
    "diastolic_bp": "Diastolic blood pressure",
    "glucose": "Glucose",
    "heart_rate": "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "systolic_bp": "Systolic blood pressure",
    "gender_F": "Female",
    "gender_M": "Male"
}


# ============================================================
# PROBABILITY -> LOGIT
# ============================================================

def probability_to_logit(probability):

    probability = np.clip(
        probability,
        1e-6,
        1 - 1e-6
    )

    return np.log(
        probability / (1 - probability)
    )


# ============================================================
# PREPARE PATIENT DATA
# ============================================================

def prepare_patient_data(patient_data):

    patient_df = pd.DataFrame(
        [patient_data]
    )

    required_features = [
        "age_at_admission",
        "gender",
        "length_of_stay",
        "previous_inpatient_admissions",
        "previous_emergency_visits",
        "previous_encounters",
        "condition_count",
        "diabetes",
        "hypertension",
        "heart_failure",
        "kidney_disease",
        "chronic_lung_disease",
        "medication_count",
        "procedure_count"
    ]

    missing_required = [
        feature
        for feature in required_features
        if feature not in patient_df.columns
    ]

    if missing_required:
        raise ValueError(
            "Missing required patient features: "
            + ", ".join(missing_required)
        )

    # --------------------------------------------------------
    # CLINICAL IMPUTATION
    # --------------------------------------------------------

    for column in clinical_features:

        if column not in patient_df.columns:

            patient_df[column] = clinical_medians[column]

        else:

            patient_df[column] = pd.to_numeric(
                patient_df[column],
                errors="coerce"
            )

            patient_df[column] = patient_df[column].fillna(
                clinical_medians[column]
            )

    # --------------------------------------------------------
    # ENCODE GENDER
    # --------------------------------------------------------

    patient_df = pd.get_dummies(
        patient_df,
        columns=["gender"],
        dtype=int
    )

    # --------------------------------------------------------
    # MATCH TRAINING COLUMNS
    # --------------------------------------------------------

    patient_df = patient_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return patient_df


# ============================================================
# PREDICT READMISSION
# ============================================================

def predict_readmission(patient_data):

    prepared_data = prepare_patient_data(
        patient_data
    )

    # Raw XGBoost probability
    raw_probability = float(
        model.predict_proba(
            prepared_data
        )[0, 1]
    )

    # Convert probability to logit because the calibrator
    # was trained using the model's logit output
    raw_logit = probability_to_logit(
        raw_probability
    )

    # Calibrated probability
    calibrated_probability = float(
        calibrator.predict_proba(
            np.array(
                [[raw_logit]]
            )
        )[0, 1]
    )

    # Intervention decision
    intervention_flag = int(
        calibrated_probability
        >= production_threshold
    )

    # Population risk band
    if calibrated_probability < low_cutoff:
        risk_level = "LOW"

    elif calibrated_probability < high_cutoff:
        risk_level = "MODERATE"

    else:
        risk_level = "HIGH"

    return {
        "calibrated_probability":
            calibrated_probability,

        "probability_percent":
            round(
                calibrated_probability * 100,
                1
            ),

        "risk_level":
            risk_level,

        "intervention_flag":
            intervention_flag,

        "intervention_recommended":
            bool(intervention_flag),

        "production_threshold":
            production_threshold,

        "production_threshold_percent":
            round(
                production_threshold * 100,
                1
            ),

        "low_cutoff":
            low_cutoff,

        "low_cutoff_percent":
            round(
                low_cutoff * 100,
                1
            ),

        "high_cutoff":
            high_cutoff,

        "high_cutoff_percent":
            round(
                high_cutoff * 100,
                1
            )
    }


# ============================================================
# EXPLAIN READMISSION WITH SHAP
# ============================================================

def explain_readmission(
    patient_data,
    top_n=5
):

    prepared_data = prepare_patient_data(
        patient_data
    )

    # Calculate SHAP values for this patient
    shap_values = explainer.shap_values(
        prepared_data
    )

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        values = np.array(
            shap_values[-1]
        )[0]

    else:

        values = np.array(
            shap_values
        )

        if values.ndim == 2:
            values = values[0]

        elif values.ndim == 3:
            values = values[0, :, -1]

    # Build explanation table
    explanation = pd.DataFrame(
        {
            "feature":
                prepared_data.columns,

            "value":
                prepared_data.iloc[0].values,

            "shap_value":
                values
        }
    )

    # Absolute SHAP value tells us the strength
    # of each feature's contribution
    explanation["absolute_impact"] = (
        explanation["shap_value"].abs()
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_impact",
            ascending=False
        )
        .head(top_n)
        .copy()
    )

    drivers = []

    for _, row in explanation.iterrows():

        raw_feature = row["feature"]

        friendly_name = friendly_feature_names.get(
            raw_feature,
            raw_feature
        )

        if row["shap_value"] > 0:
            direction = "increases risk"

        elif row["shap_value"] < 0:
            direction = "decreases risk"

        else:
            direction = "neutral"

        drivers.append(
            {
                "feature":
                    friendly_name,

                "value":
                    row["value"],

                "impact":
                    float(
                        row["shap_value"]
                    ),

                "absolute_impact":
                    float(
                        row["absolute_impact"]
                    ),

                "direction":
                    direction
            }
        )

    return drivers


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    sample_patient = {
        "age_at_admission": 75,
        "gender": "F",
        "length_of_stay": 10.0,
        "previous_inpatient_admissions": 12,
        "previous_emergency_visits": 10,
        "previous_encounters": 150,
        "condition_count": 25,
        "diabetes": 1,
        "hypertension": 1,
        "heart_failure": 1,
        "kidney_disease": 1,
        "chronic_lung_disease": 1,
        "medication_count": 20,
        "procedure_count": 40,
        "bmi": 34.0,
        "systolic_bp": 160,
        "diastolic_bp": 95,
        "heart_rate": 105,
        "respiratory_rate": 22,
        "glucose": 180
    }

    prediction = predict_readmission(
        sample_patient
    )

    drivers = explain_readmission(
        sample_patient,
        top_n=5
    )

    print()
    print("=" * 60)
    print("PREDICTION")
    print("=" * 60)

    print(
        "Calibrated readmission probability:",
        f'{prediction["probability_percent"]}%'
    )

    print(
        "Population risk level:",
        prediction["risk_level"]
    )

    print(
        "Intervention recommended:",
        prediction["intervention_recommended"]
    )

    print(
        "Intervention threshold:",
        f'{prediction["production_threshold_percent"]}%'
    )

    print()

    print("=" * 60)
    print("TOP RISK DRIVERS")
    print("=" * 60)

    for number, driver in enumerate(
        drivers,
        start=1
    ):

        print()

        print(
            f'{number}. {driver["feature"]}'
        )

        print(
            "   Direction:",
            driver["direction"]
        )

        print(
            "   SHAP impact:",
            round(
                driver["impact"],
                3
            )
        )

        print(
            "   Patient value:",
            driver["value"]
        )