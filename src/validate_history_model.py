import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/synthea_readmission_ml_dataset_v3.csv"
)

MODEL_DIR = Path("models")


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_DIR
    / "readmission_history_xgboost_calibrated.joblib"
)

calibrator = joblib.load(
    MODEL_DIR
    / "readmission_history_calibrator.joblib"
)


with open(
    MODEL_DIR
    / "readmission_history_features.json"
) as f:

    FEATURES = json.load(f)


with open(
    MODEL_DIR
    / "readmission_history_medians.json"
) as f:

    MEDIANS = json.load(f)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

TARGET = "readmitted_30_days"


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].astype(int)

groups = df[
    "patient_id"
].astype(str)


for feature in FEATURES:

    X[
        feature
    ] = (
        X[
            feature
        ]
        .fillna(
            MEDIANS[
                feature
            ]
        )
    )


# ============================================================
# RECREATE EXACT HELD-OUT TEST SET
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups,
    )
)


test = (
    df.iloc[
        test_idx
    ]
    .copy()
    .reset_index(
        drop=True
    )
)

X_test = (
    X.iloc[
        test_idx
    ]
    .copy()
    .reset_index(
        drop=True
    )
)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

raw_probabilities = (
    model
    .predict_proba(
        X_test
    )[:, 1]
)

calibrated_probabilities = (
    calibrator
    .predict(
        raw_probabilities
    )
)


test[
    "raw_probability"
] = raw_probabilities

test[
    "calibrated_probability"
] = calibrated_probabilities


# ============================================================
# SUMMARY
# ============================================================

print("=" * 72)
print("REAL HELD-OUT PATIENT VALIDATION")
print("=" * 72)

print(
    "\nHeld-out encounters:",
    len(test),
)

print(
    "Unique held-out patients:",
    test[
        "patient_id"
    ].nunique(),
)

print(
    "Actual readmission rate:",
    f"{test[TARGET].mean() * 100:.2f}%"
)

print(
    "Mean calibrated probability:",
    f"{test['calibrated_probability'].mean() * 100:.2f}%"
)


# ============================================================
# GROUP PERFORMANCE BY 90-DAY ADMISSION HISTORY
# ============================================================

print("\n" + "=" * 72)
print("MODEL BEHAVIOR BY 90-DAY ADMISSION HISTORY")
print("=" * 72)


group_summary = (
    test
    .groupby(
        "admissions_last_90_days"
    )
    .agg(
        encounters=(
            TARGET,
            "size",
        ),
        actual_readmission_rate=(
            TARGET,
            "mean",
        ),
        mean_raw_probability=(
            "raw_probability",
            "mean",
        ),
        mean_calibrated_probability=(
            "calibrated_probability",
            "mean",
        ),
    )
    .reset_index()
)


group_summary[
    "actual_readmission_rate"
] *= 100

group_summary[
    "mean_raw_probability"
] *= 100

group_summary[
    "mean_calibrated_probability"
] *= 100


print(
    group_summary
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# GROUP PERFORMANCE BY PRIOR READMISSIONS
# ============================================================

print("\n" + "=" * 72)
print("MODEL BEHAVIOR BY PRIOR READMISSION HISTORY")
print("=" * 72)


test[
    "prior_readmission_group"
] = pd.cut(
    test[
        "prior_readmissions_last_365_days"
    ],
    bins=[
        -0.1,
        0,
        1,
        2,
        5,
        np.inf,
    ],
    labels=[
        "0",
        "1",
        "2",
        "3-5",
        "6+",
    ],
)


prior_summary = (
    test
    .groupby(
        "prior_readmission_group",
        observed=False,
    )
    .agg(
        encounters=(
            TARGET,
            "size",
        ),
        actual_readmission_rate=(
            TARGET,
            "mean",
        ),
        mean_calibrated_probability=(
            "calibrated_probability",
            "mean",
        ),
    )
    .reset_index()
)


prior_summary[
    "actual_readmission_rate"
] *= 100

prior_summary[
    "mean_calibrated_probability"
] *= 100


print(
    prior_summary
    .round(2)
    .to_string(
        index=False
    )
)


# ============================================================
# DISPLAY INDIVIDUAL REAL PATIENTS
# ============================================================

DISPLAY_FEATURES = [
    "age_at_admission",
    "length_of_stay",

    "previous_inpatient_admissions",

    "admissions_last_30_days",
    "admissions_last_90_days",
    "admissions_last_365_days",

    "days_since_last_inpatient_admission",

    "prior_readmissions_last_365_days",

    "condition_count",
    "medication_count",
    "procedure_count",

    "diabetes",
    "hypertension",
    "kidney_disease",

    "bmi",
]


def print_patient(
    row,
    title,
):

    print(
        "\n"
        + "=" * 72
    )

    print(
        title
    )

    print(
        "=" * 72
    )

    print(
        "\nEncounter ID:"
    )

    print(
        row[
            "encounter_id"
        ]
    )

    print(
        "\nPatient ID:"
    )

    print(
        row[
            "patient_id"
        ]
    )

    print(
        "\nActual readmitted within 30 days:"
    )

    print(
        int(
            row[
                TARGET
            ]
        )
    )

    print(
        "\nRaw probability:"
    )

    print(
        f"{row['raw_probability'] * 100:.1f}%"
    )

    print(
        "\nCalibrated probability:"
    )

    print(
        f"{row['calibrated_probability'] * 100:.1f}%"
    )

    print(
        "\nPatient features:"
    )

    for feature in DISPLAY_FEATURES:

        print(
            f"{feature}: "
            f"{row[feature]}"
        )


# ============================================================
# FIND REPRESENTATIVE REAL PATIENTS
# ============================================================

print("\n" + "=" * 72)
print("REPRESENTATIVE REAL PATIENTS")
print("=" * 72)


for admission_count in [
    0,
    1,
    2,
    3,
]:

    candidates = test[
        test[
            "admissions_last_90_days"
        ]
        == admission_count
    ].copy()

    if candidates.empty:

        print(
            f"\nNo held-out patient found with "
            f"{admission_count} admissions "
            f"in the last 90 days."
        )

        continue


    # Select the patient closest to the group's median
    # calibrated probability rather than cherry-picking
    # an unusually high- or low-risk example.

    group_median = (
        candidates[
            "calibrated_probability"
        ]
        .median()
    )

    candidates[
        "distance_from_group_median"
    ] = (
        candidates[
            "calibrated_probability"
        ]
        - group_median
    ).abs()


    representative = (
        candidates
        .sort_values(
            "distance_from_group_median"
        )
        .iloc[0]
    )


    print_patient(
        representative,
        (
            f"REAL PATIENT: "
            f"{admission_count} "
            f"ADMISSION(S) IN LAST 90 DAYS"
        ),
    )


# ============================================================
# HIGH-RISK TRUE POSITIVE
# ============================================================

true_positive_candidates = test[
    test[
        TARGET
    ]
    == 1
].copy()


if not true_positive_candidates.empty:

    high_true_positive = (
        true_positive_candidates
        .sort_values(
            "calibrated_probability",
            ascending=False,
        )
        .iloc[0]
    )

    print_patient(
        high_true_positive,
        "HIGH-RISK PATIENT WHO WAS ACTUALLY READMITTED",
    )


# ============================================================
# HIGH-RISK FALSE POSITIVE
# ============================================================

false_positive_candidates = test[
    test[
        TARGET
    ]
    == 0
].copy()


if not false_positive_candidates.empty:

    high_false_positive = (
        false_positive_candidates
        .sort_values(
            "calibrated_probability",
            ascending=False,
        )
        .iloc[0]
    )

    print_patient(
        high_false_positive,
        "HIGH-RISK PATIENT WHO WAS NOT READMITTED",
    )


# ============================================================
# LOW-RISK FALSE NEGATIVE
# ============================================================

false_negative_candidates = test[
    test[
        TARGET
    ]
    == 1
].copy()


if not false_negative_candidates.empty:

    low_false_negative = (
        false_negative_candidates
        .sort_values(
            "calibrated_probability",
            ascending=True,
        )
        .iloc[0]
    )

    print_patient(
        low_false_negative,
        "LOWEST-RISK PATIENT WHO WAS ACTUALLY READMITTED",
    )


print("\n" + "=" * 72)
print("VALIDATION COMPLETE")
print("=" * 72)