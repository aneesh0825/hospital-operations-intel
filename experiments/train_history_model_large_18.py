from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_large_v3_heart_kidney.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

TARGET = "readmitted_30_days"

FEATURES = [
    "age_at_admission",
    "length_of_stay",

    "previous_inpatient_admissions",

    "admissions_last_30_days",
    "admissions_last_90_days",
    "admissions_last_365_days",

    "days_since_last_inpatient_admission",
    "has_prior_inpatient_admission",
    "prior_readmissions_last_365_days",

    "condition_count",
    "medication_count",
    "procedure_count",

    "diabetes",
    "hypertension",
    "kidney_disease",

    "bmi",

    # --------------------------------------------------------
    # NEW 18-FEATURE EXPERIMENT
    # --------------------------------------------------------

    "ischemic_heart_disease",
    "kidney_failure",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)


print("=" * 72)
print("ADMITRA LARGE 18-FEATURE MODEL TRAINING")
print("=" * 72)

print()
print("Rows:")
print(
    f"{len(df):,}"
)

print()
print("Features:")
print(
    len(FEATURES)
)

print()
print("Readmission rate:")
print(
    round(
        df[
            TARGET
        ].mean()
        * 100,
        2,
    ),
    "%",
)


# ============================================================
# PREPARE DATA
# ============================================================

X = (
    df[
        FEATURES
    ]
    .copy()
)

y = (
    df[
        TARGET
    ]
    .astype(int)
)

groups = (
    df[
        "patient_id"
    ]
    .astype(str)
)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

medians = {}

for feature in FEATURES:

    median = float(
        X[
            feature
        ]
        .median()
    )

    medians[
        feature
    ] = median

    X[
        feature
    ] = (
        X[
            feature
        ]
        .fillna(
            median
        )
    )


# ============================================================
# PATIENT-LEVEL TRAIN / TEST SPLIT
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


X_train = (
    X
    .iloc[
        train_idx
    ]
    .copy()
)

X_test = (
    X
    .iloc[
        test_idx
    ]
    .copy()
)

y_train = (
    y
    .iloc[
        train_idx
    ]
    .copy()
)

y_test = (
    y
    .iloc[
        test_idx
    ]
    .copy()
)


train_patients = set(
    groups.iloc[
        train_idx
    ]
)

test_patients = set(
    groups.iloc[
        test_idx
    ]
)


print()
print("=" * 72)
print("PATIENT-LEVEL SPLIT")
print("=" * 72)

print()
print("Training rows:")
print(
    len(
        X_train
    )
)

print()
print("Testing rows:")
print(
    len(
        X_test
    )
)

print()
print("Training patients:")
print(
    len(
        train_patients
    )
)

print()
print("Testing patients:")
print(
    len(
        test_patients
    )
)

print()
print("Patient overlap:")
print(
    len(
        train_patients
        & test_patients
    )
)


if len(
    train_patients
    & test_patients
) != 0:

    raise RuntimeError(
        "Patient leakage detected."
    )


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = int(
    (
        y_train
        == 0
    ).sum()
)

positive = int(
    (
        y_train
        == 1
    ).sum()
)

scale_pos_weight = (
    negative
    / positive
)


print()
print("Negative training examples:")
print(
    f"{negative:,}"
)

print()
print("Positive training examples:")
print(
    f"{positive:,}"
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
# MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    reg_alpha=0.15,
    reg_lambda=1.5,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)


print()
print("=" * 72)
print("TRAINING 18-FEATURE MODEL")
print("=" * 72)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# HELD-OUT PREDICTIONS
# ============================================================

probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


roc_auc = (
    roc_auc_score(
        y_test,
        probabilities,
    )
)

pr_auc = (
    average_precision_score(
        y_test,
        probabilities,
    )
)


# ============================================================
# PERFORMANCE
# ============================================================

print()
print("=" * 72)
print("HELD-OUT TEST PERFORMANCE")
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
        y_test.mean()
        * 100,
        2,
    ),
    "%",
)

print()
print("Mean raw predicted probability:")
print(
    round(
        probabilities.mean()
        * 100,
        2,
    ),
    "%",
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame(
    {
        "feature":
            FEATURES,

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
print("FEATURE IMPORTANCE")
print("=" * 72)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# NEW FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 72)
print("NEW FEATURE IMPORTANCE")
print("=" * 72)

print(
    importance[
        importance[
            "feature"
        ].isin(
            [
                "ischemic_heart_disease",
                "kidney_failure",
            ]
        )
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE EXPERIMENTAL MODEL
# ============================================================

MODEL_PATH = (
    MODEL_DIR
    / "readmission_history_xgboost_large_18.joblib"
)

IMPORTANCE_PATH = (
    MODEL_DIR
    / "readmission_history_feature_importance_large_18.csv"
)


joblib.dump(
    model,
    MODEL_PATH,
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
print("18-FEATURE EXPERIMENT SAVED")
print("=" * 72)

print()
print(
    MODEL_PATH
)

print(
    IMPORTANCE_PATH
)

print()

print(
    "Existing production V3 model was NOT modified."
)