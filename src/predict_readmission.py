import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR
    / "readmission_history_xgboost_calibrated.joblib"
)

CALIBRATOR_PATH = (
    MODEL_DIR
    / "readmission_history_calibrator.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "readmission_history_features.json"
)

MEDIANS_PATH = (
    MODEL_DIR
    / "readmission_history_medians.json"
)

THRESHOLD_PATH = (
    MODEL_DIR
    / "readmission_history_threshold.json"
)

RISK_BANDS_PATH = (
    MODEL_DIR
    / "readmission_history_risk_bands.json"
)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_PATH
)

calibrator = joblib.load(
    CALIBRATOR_PATH
)


with open(
    FEATURES_PATH,
    "r",
) as file:

    FEATURE_COLUMNS = json.load(
        file
    )


with open(
    MEDIANS_PATH,
    "r",
) as file:

    MEDIANS = json.load(
        file
    )


with open(
    THRESHOLD_PATH,
    "r",
) as file:

    THRESHOLD_CONFIG = json.load(
        file
    )


with open(
    RISK_BANDS_PATH,
    "r",
) as file:

    RISK_BANDS = json.load(
        file
    )


PRODUCTION_THRESHOLD = float(
    THRESHOLD_CONFIG[
        "production_threshold"
    ]
)

LOW_MODERATE_CUTOFF = float(
    RISK_BANDS[
        "low_moderate_cutoff"
    ]
)

MODERATE_HIGH_CUTOFF = float(
    RISK_BANDS[
        "moderate_high_cutoff"
    ]
)


# ============================================================
# DISPLAY NAMES
# ============================================================

FEATURE_DISPLAY_NAMES = {
    "age_at_admission":
        "Age",

    "length_of_stay":
        "Length of stay",

    "previous_inpatient_admissions":
        "Previous inpatient admissions",

    "admissions_last_30_days":
        "Admissions in last 30 days",

    "admissions_last_90_days":
        "Admissions in last 90 days",

    "admissions_last_365_days":
        "Admissions in last 365 days",

    "days_since_last_inpatient_admission":
        "Days since last inpatient admission",

    "has_prior_inpatient_admission":
        "Prior inpatient history",

    "prior_readmissions_last_365_days":
        "Prior readmissions in last 365 days",

    "condition_count":
        "Condition burden",

    "medication_count":
        "Medication count",

    "procedure_count":
        "Procedure count",

    "diabetes":
        "Diabetes",

    "hypertension":
        "Hypertension",

    "kidney_disease":
        "Kidney disease",

    "bmi":
        "BMI",
}


# ============================================================
# INPUT PREPARATION
# ============================================================

def _safe_value(
    patient_data,
    feature,
):

    if feature in patient_data:

        value = patient_data[
            feature
        ]

        if value is not None:

            try:
                if not pd.isna(
                    value
                ):
                    return value

            except Exception:
                return value


    return MEDIANS.get(
        feature,
        0
    )


def _prepare_patient(
    patient_data,
):

    row = {}

    for feature in FEATURE_COLUMNS:

        value = _safe_value(
            patient_data,
            feature,
        )

        row[
            feature
        ] = value


    # --------------------------------------------------------
    # NORMALIZE BINARY VALUES
    # --------------------------------------------------------

    binary_features = [
        "has_prior_inpatient_admission",
        "diabetes",
        "hypertension",
        "kidney_disease",
    ]

    for feature in binary_features:

        if feature in row:

            value = row[
                feature
            ]

            if isinstance(
                value,
                str,
            ):

                row[
                    feature
                ] = int(
                    value.strip().lower()
                    in [
                        "1",
                        "true",
                        "yes",
                        "y",
                        "present",
                    ]
                )

            else:

                row[
                    feature
                ] = int(
                    bool(
                        value
                    )
                )


    # --------------------------------------------------------
    # HISTORY CONSISTENCY
    # --------------------------------------------------------

    prior_admissions = float(
        row.get(
            "previous_inpatient_admissions",
            0,
        )
    )

    if prior_admissions <= 0:

        row[
            "has_prior_inpatient_admission"
        ] = 0

        row[
            "days_since_last_inpatient_admission"
        ] = 365

    else:

        row[
            "has_prior_inpatient_admission"
        ] = 1


    # Keep the recency value within the range
    # used during training.

    row[
        "days_since_last_inpatient_admission"
    ] = float(
        np.clip(
            row[
                "days_since_last_inpatient_admission"
            ],
            0,
            365,
        )
    )


    # History windows should never be negative.

    count_features = [
        "previous_inpatient_admissions",
        "admissions_last_30_days",
        "admissions_last_90_days",
        "admissions_last_365_days",
        "prior_readmissions_last_365_days",
        "condition_count",
        "medication_count",
        "procedure_count",
    ]

    for feature in count_features:

        row[
            feature
        ] = max(
            0,
            float(
                row[
                    feature
                ]
            ),
        )


    # --------------------------------------------------------
    # LOGICAL HISTORY RELATIONSHIPS
    # --------------------------------------------------------
    #
    # These windows are nested:
    #
    # 30-day admissions <= 90-day admissions
    # 90-day admissions <= 365-day admissions
    #
    # We normalize inconsistent manual inputs rather than
    # allowing impossible histories into the model.
    # --------------------------------------------------------

    row[
        "admissions_last_90_days"
    ] = max(
        row[
            "admissions_last_90_days"
        ],
        row[
            "admissions_last_30_days"
        ],
    )

    row[
        "admissions_last_365_days"
    ] = max(
        row[
            "admissions_last_365_days"
        ],
        row[
            "admissions_last_90_days"
        ],
    )

    row[
        "previous_inpatient_admissions"
    ] = max(
        row[
            "previous_inpatient_admissions"
        ],
        row[
            "admissions_last_365_days"
        ],
    )


    # --------------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------------

    patient_df = pd.DataFrame(
        [
            [
                row[
                    feature
                ]
                for feature
                in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    return (
        patient_df,
        row,
    )


# ============================================================
# RISK LEVEL
# ============================================================

def _risk_level(
    probability,
):

    if probability < LOW_MODERATE_CUTOFF:

        return "LOW"

    if probability < MODERATE_HIGH_CUTOFF:

        return "MODERATE"

    return "HIGH"


# ============================================================
# PREDICTION
# ============================================================

def predict_readmission(
    patient_data,
):

    patient_df, normalized = (
        _prepare_patient(
            patient_data
        )
    )


    raw_probability = float(
        model.predict_proba(
            patient_df
        )[0, 1]
    )


    calibrated_probability = float(
        calibrator.predict(
            [
                raw_probability
            ]
        )[0]
    )


    risk_level = _risk_level(
        calibrated_probability
    )


    intervention_recommended = bool(
        calibrated_probability
        >= PRODUCTION_THRESHOLD
    )


    return {
        "probability":
            calibrated_probability,

        "probability_percent":
            round(
                calibrated_probability
                * 100,
                1,
            ),

        "raw_probability":
            raw_probability,

        "raw_probability_percent":
            round(
                raw_probability
                * 100,
                1,
            ),

        "risk_level":
            risk_level,

        "intervention_recommended":
            intervention_recommended,

        "production_threshold":
            PRODUCTION_THRESHOLD,

        "production_threshold_percent":
            round(
                PRODUCTION_THRESHOLD
                * 100,
                1,
            ),

        "low_cutoff":
            LOW_MODERATE_CUTOFF,

        "low_cutoff_percent":
            round(
                LOW_MODERATE_CUTOFF
                * 100,
                1,
            ),

        "high_cutoff":
            MODERATE_HIGH_CUTOFF,

        "high_cutoff_percent":
            round(
                MODERATE_HIGH_CUTOFF
                * 100,
                1,
            ),

        "normalized_input":
            normalized,
    }


# ============================================================
# SHAP EXPLAINABILITY
# ============================================================

_explainer = None


def _get_explainer():

    global _explainer

    if _explainer is None:

        _explainer = shap.TreeExplainer(
            model
        )

    return _explainer


def explain_readmission(
    patient_data,
    top_n=5,
):

    patient_df, normalized = (
        _prepare_patient(
            patient_data
        )
    )


    explainer = _get_explainer()

    shap_values = explainer.shap_values(
        patient_df
    )


    # --------------------------------------------------------
    # NORMALIZE SHAP OUTPUT SHAPE
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list,
    ):

        shap_array = np.asarray(
            shap_values[-1]
        )

    else:

        shap_array = np.asarray(
            shap_values
        )


    if shap_array.ndim == 2:

        values = shap_array[
            0
        ]

    elif shap_array.ndim == 1:

        values = shap_array

    else:

        values = shap_array.reshape(
            -1
        )


    drivers = []


    for index, feature in enumerate(
        FEATURE_COLUMNS
    ):

        impact = float(
            values[
                index
            ]
        )

        if impact > 0:

            direction = (
                "increases risk"
            )

        elif impact < 0:

            direction = (
                "decreases risk"
            )

        else:

            direction = (
                "neutral"
            )


        drivers.append(
            {
                "feature":
                    FEATURE_DISPLAY_NAMES.get(
                        feature,
                        feature,
                    ),

                "feature_key":
                    feature,

                "direction":
                    direction,

                "impact":
                    impact,

                "absolute_impact":
                    abs(
                        impact
                    ),

                "value":
                    normalized[
                        feature
                    ],
            }
        )


    drivers = sorted(
        drivers,
        key=lambda item:
            item[
                "absolute_impact"
            ],
        reverse=True,
    )


    return drivers[
        :top_n
    ]


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    test_patient = {
        "age_at_admission": 67,
        "length_of_stay": 5.0,

        "previous_inpatient_admissions": 4,

        "admissions_last_30_days": 1,
        "admissions_last_90_days": 2,
        "admissions_last_365_days": 4,

        "days_since_last_inpatient_admission": 25,
        "has_prior_inpatient_admission": 1,

        "prior_readmissions_last_365_days": 1,

        "condition_count": 15,
        "medication_count": 8,
        "procedure_count": 20,

        "diabetes": 1,
        "hypertension": 1,
        "kidney_disease": 0,

        "bmi": 29.0,
    }


    result = predict_readmission(
        test_patient
    )


    print("=" * 60)
    print("HISTORY-AWARE READMISSION PREDICTION")
    print("=" * 60)

    print(
        "\nCalibrated probability:",
        f'{result["probability_percent"]}%'
    )

    print(
        "Raw model probability:",
        f'{result["raw_probability_percent"]}%'
    )

    print(
        "Risk level:",
        result[
            "risk_level"
        ],
    )

    print(
        "Review recommended:",
        result[
            "intervention_recommended"
        ],
    )

    print(
        "Review threshold:",
        f'{result["production_threshold_percent"]}%'
    )

    print(
        "Low / Moderate cutoff:",
        f'{result["low_cutoff_percent"]}%'
    )

    print(
        "Moderate / High cutoff:",
        f'{result["high_cutoff_percent"]}%'
    )


    print("\n" + "=" * 60)
    print("TOP RISK DRIVERS")
    print("=" * 60)


    drivers = explain_readmission(
        test_patient,
        top_n=5,
    )


    for number, driver in enumerate(
        drivers,
        start=1,
    ):

        print(
            f"\n{number}. "
            f'{driver["feature"]}'
        )

        print(
            "   Direction:",
            driver[
                "direction"
            ],
        )

        print(
            "   SHAP impact:",
            round(
                driver[
                    "impact"
                ],
                3,
            ),
        )

        print(
            "   Patient value:",
            driver[
                "value"
            ],
        )