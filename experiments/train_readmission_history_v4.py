import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_v3.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    MODEL_DIR
    / "readmission_history_xgboost_v4.joblib"
)

FEATURES_PATH = (
    MODEL_DIR
    / "readmission_history_features_v4.json"
)

MEDIANS_PATH = (
    MODEL_DIR
    / "readmission_history_medians_v4.json"
)

IMPORTANCE_PATH = (
    MODEL_DIR
    / "readmission_history_feature_importance_v4.csv"
)

TARGET = "readmitted_30_days"
GROUP = "patient_id"


# ============================================================
# V4 FEATURES
# ============================================================

# These are the 16 features from the working history-aware
# V3 model plus 5 experimental clinical/missingness features.

FEATURES = [

    # --------------------------------------------------------
    # PATIENT HISTORY
    # --------------------------------------------------------

    "previous_inpatient_admissions",

    "admissions_last_30_days",
    "admissions_last_90_days",
    "admissions_last_365_days",

    "days_since_last_inpatient_admission",

    "has_prior_inpatient_admission",

    "prior_readmissions_last_365_days",

    # --------------------------------------------------------
    # DEMOGRAPHIC / CURRENT ENCOUNTER
    # --------------------------------------------------------

    "age_at_admission",
    "length_of_stay",

    # --------------------------------------------------------
    # CLINICAL BURDEN
    # --------------------------------------------------------

    "condition_count",
    "medication_count",
    "procedure_count",

    "bmi",

    # --------------------------------------------------------
    # EXISTING CONDITION FLAGS
    # --------------------------------------------------------

    "diabetes",
    "hypertension",
    "kidney_disease",

    # --------------------------------------------------------
    # V4 CLINICAL MEASUREMENTS
    # --------------------------------------------------------

    "systolic_bp",
    "heart_rate",
    "glucose",

    # --------------------------------------------------------
    # V4 MEASUREMENT AVAILABILITY
    # --------------------------------------------------------

    "vitals_available",
    "glucose_available",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 72)
print("ADMITRA V4 HISTORY-AWARE MODEL TRAINING")
print("=" * 72)

data = pd.read_csv(DATA_PATH)

print()
print("Rows:")
print(f"{len(data):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURES
    + [
        TARGET,
        GROUP,
    ]
)

# Availability indicators do not exist yet.
# Remove them from the raw-column requirement.

raw_required_columns = [
    column
    for column in required_columns
    if column not in [
        "vitals_available",
        "glucose_available",
    ]
]

missing_columns = [
    column
    for column in raw_required_columns
    if column not in data.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# CREATE V4 AVAILABILITY FEATURES
# ============================================================

# All four vital-sign fields came from the same observation
# coverage pattern in our data. We use systolic BP as the
# indicator for whether that vital-sign information exists.

data["vitals_available"] = (
    data["systolic_bp"]
    .notna()
    .astype(int)
)

data["glucose_available"] = (
    data["glucose"]
    .notna()
    .astype(int)
)


print()
print("=" * 72)
print("V4 AVAILABILITY FEATURES")
print("=" * 72)

print()
print("Vitals available:")
print(
    data[
        "vitals_available"
    ].value_counts()
)

print()
print("Glucose available:")
print(
    data[
        "glucose_available"
    ].value_counts()
)


# ============================================================
# TARGET
# ============================================================

y = (
    data[TARGET]
    .astype(int)
)

groups = data[GROUP]

print()
print("Readmission rate:")
print(
    round(
        y.mean() * 100,
        2,
    ),
    "%",
)

print()
print("Features:")
print(len(FEATURES))


# ============================================================
# PATIENT-LEVEL TRAIN / TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_index, test_index = next(
    splitter.split(
        data,
        y,
        groups=groups,
    )
)

train_data = (
    data
    .iloc[train_index]
    .copy()
)

test_data = (
    data
    .iloc[test_index]
    .copy()
)

y_train = (
    y
    .iloc[train_index]
    .copy()
)

y_test = (
    y
    .iloc[test_index]
    .copy()
)


# ============================================================
# VERIFY PATIENT ISOLATION
# ============================================================

train_patients = set(
    train_data[GROUP]
)

test_patients = set(
    test_data[GROUP]
)

patient_overlap = (
    train_patients
    .intersection(
        test_patients
    )
)

print()
print("=" * 72)
print("PATIENT-LEVEL SPLIT")
print("=" * 72)

print()
print("Training rows:")
print(len(train_data))

print()
print("Testing rows:")
print(len(test_data))

print()
print("Training patients:")
print(len(train_patients))

print()
print("Testing patients:")
print(len(test_patients))

print()
print("Patient overlap:")
print(len(patient_overlap))


if len(patient_overlap) != 0:

    raise RuntimeError(
        "Patient leakage detected."
    )


# ============================================================
# CALCULATE TRAINING MEDIANS
# ============================================================

# IMPORTANT:
# Medians are calculated ONLY from the training set.
# This prevents information from the held-out patients
# influencing preprocessing.

medians = {}

for feature in FEATURES:

    if feature in [
        "vitals_available",
        "glucose_available",
    ]:
        continue

    median = (
        train_data[feature]
        .median()
    )

    if pd.isna(median):

        median = 0.0

    medians[feature] = float(
        median
    )


# ============================================================
# BUILD FEATURE MATRICES
# ============================================================

X_train = (
    train_data[
        FEATURES
    ]
    .copy()
)

X_test = (
    test_data[
        FEATURES
    ]
    .copy()
)


# ------------------------------------------------------------
# Impute numerical measurements
# ------------------------------------------------------------

for feature in FEATURES:

    if feature in [
        "vitals_available",
        "glucose_available",
    ]:
        continue

    X_train[feature] = (
        X_train[feature]
        .fillna(
            medians[feature]
        )
    )

    X_test[feature] = (
        X_test[feature]
        .fillna(
            medians[feature]
        )
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative_count = int(
    (y_train == 0).sum()
)

positive_count = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative_count
    / positive_count
)

print()
print("Negative training examples:")
print(
    f"{negative_count:,}"
)

print()
print("Positive training examples:")
print(
    f"{positive_count:,}"
)

print()
print("Scale pos weight:")
print(
    round(
        scale_pos_weight,
        3,
    )
)


# ============================================================
# TRAIN V4 MODEL
# ============================================================

print()
print("=" * 72)
print("TRAINING V4 MODEL")
print("=" * 72)

model = XGBClassifier(

    n_estimators=500,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    reg_alpha=0.1,

    reg_lambda=1.0,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale_pos_weight,

    random_state=42,

    n_jobs=-1,
)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# HELD-OUT TEST PREDICTIONS
# ============================================================

test_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)

roc_auc = roc_auc_score(
    y_test,
    test_probability,
)

pr_auc = average_precision_score(
    y_test,
    test_probability,
)


print()
print("=" * 72)
print("V4 HELD-OUT TEST PERFORMANCE")
print("=" * 72)

print()
print("ROC-AUC:")
print(
    round(
        roc_auc,
        4,
    )
)

print()
print("PR-AUC:")
print(
    round(
        pr_auc,
        4,
    )
)

print()
print("Actual test readmission rate:")
print(
    round(
        y_test.mean() * 100,
        2,
    ),
    "%",
)

print()
print("Mean raw predicted probability:")
print(
    round(
        test_probability.mean() * 100,
        2,
    ),
    "%",
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance":
            model.feature_importances_,
    }
)

importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


print()
print("=" * 72)
print("V4 FEATURE IMPORTANCE")
print("=" * 72)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# V4-SPECIFIC FEATURE IMPORTANCE
# ============================================================

v4_features = [
    "systolic_bp",
    "heart_rate",
    "glucose",
    "vitals_available",
    "glucose_available",
]

print()
print("=" * 72)
print("NEW V4 FEATURE IMPORTANCE")
print("=" * 72)

print(
    importance[
        importance[
            "feature"
        ].isin(
            v4_features
        )
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE EXPERIMENTAL ARTIFACTS
# ============================================================

joblib.dump(
    model,
    MODEL_PATH,
)

with open(
    FEATURES_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        FEATURES,
        file,
        indent=4,
    )


with open(
    MEDIANS_PATH,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        medians,
        file,
        indent=4,
    )


importance.to_csv(
    IMPORTANCE_PATH,
    index=False,
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 72)
print("V4 EXPERIMENTAL ARTIFACTS SAVED")
print("=" * 72)

print()
print(MODEL_PATH)

print(FEATURES_PATH)

print(MEDIANS_PATH)

print(IMPORTANCE_PATH)

print()
print(
    "NOTE: V3 production artifacts were NOT modified."
)