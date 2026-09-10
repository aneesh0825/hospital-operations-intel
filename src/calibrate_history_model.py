import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
)

from xgboost import XGBClassifier

# IMPORTANT:
# SigmoidCalibrator now lives in its own module.
# This allows joblib to load the saved calibrator from other scripts.
from src.calibrators import SigmoidCalibrator


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/synthea_readmission_ml_dataset_v3.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "readmitted_30_days"


# ============================================================
# FEATURES
# ============================================================

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
]


# ============================================================
# MODEL FACTORY
# ============================================================

def build_model(y_train):

    negative = int(
        (y_train == 0).sum()
    )

    positive = int(
        (y_train == 1).sum()
    )

    scale_pos_weight = (
        negative / positive
    )

    return XGBClassifier(
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


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("ADMITRA HISTORY MODEL CALIBRATION")
print("=" * 70)

print("\nRows:")
print(
    f"{len(df):,}"
)

print("\nReadmission rate:")
print(
    round(
        df[TARGET].mean() * 100,
        2,
    ),
    "%",
)


X = df[
    FEATURES
].copy()

y = df[
    TARGET
].astype(int)

groups = df[
    "patient_id"
].astype(str)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

medians = {}

for feature in FEATURES:

    median = float(
        X[feature].median()
    )

    medians[
        feature
    ] = median

    X[
        feature
    ] = (
        X[feature]
        .fillna(median)
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
    X.iloc[
        train_idx
    ]
    .reset_index(
        drop=True
    )
)

X_test = (
    X.iloc[
        test_idx
    ]
    .reset_index(
        drop=True
    )
)

y_train = (
    y.iloc[
        train_idx
    ]
    .reset_index(
        drop=True
    )
)

y_test = (
    y.iloc[
        test_idx
    ]
    .reset_index(
        drop=True
    )
)

groups_train = (
    groups.iloc[
        train_idx
    ]
    .reset_index(
        drop=True
    )
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


print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT")
print("=" * 70)

print("\nTraining rows:")
print(
    len(X_train)
)

print("\nTesting rows:")
print(
    len(X_test)
)

print("\nPatient overlap:")
print(
    len(
        train_patients
        & test_patients
    )
)


# ============================================================
# GROUPED OUT-OF-FOLD PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GROUPED OOF PREDICTIONS")
print("=" * 70)


oof_raw = np.zeros(
    len(X_train)
)


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


for fold_number, (
    fold_train_idx,
    fold_valid_idx,
) in enumerate(
    cv.split(
        X_train,
        y_train,
        groups_train,
    ),
    start=1,
):

    fold_model = build_model(
        y_train.iloc[
            fold_train_idx
        ]
    )

    fold_model.fit(
        X_train.iloc[
            fold_train_idx
        ],
        y_train.iloc[
            fold_train_idx
        ],
    )

    fold_probabilities = (
        fold_model
        .predict_proba(
            X_train.iloc[
                fold_valid_idx
            ]
        )[:, 1]
    )

    oof_raw[
        fold_valid_idx
    ] = fold_probabilities


    train_groups = set(
        groups_train.iloc[
            fold_train_idx
        ]
    )

    valid_groups = set(
        groups_train.iloc[
            fold_valid_idx
        ]
    )

    overlap = len(
        train_groups
        & valid_groups
    )

    print(
        f"Finished fold {fold_number} "
        f"| patient overlap: {overlap}"
    )


# ============================================================
# FIT CALIBRATORS
# ============================================================

isotonic = IsotonicRegression(
    out_of_bounds="clip"
)

isotonic.fit(
    oof_raw,
    y_train,
)

oof_isotonic = isotonic.predict(
    oof_raw
)


sigmoid = SigmoidCalibrator()

sigmoid.fit(
    oof_raw,
    y_train,
)

oof_sigmoid = sigmoid.predict(
    oof_raw
)


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calibration_metrics(
    name,
    probabilities,
):

    print(
        f"\n{name}"
    )

    print(
        "Mean probability:",
        round(
            float(
                np.mean(
                    probabilities
                )
            ),
            4,
        ),
    )

    print(
        "Brier score:",
        round(
            brier_score_loss(
                y_train,
                probabilities,
            ),
            4,
        ),
    )

    print(
        "ROC-AUC:",
        round(
            roc_auc_score(
                y_train,
                probabilities,
            ),
            4,
        ),
    )

    print(
        "PR-AUC:",
        round(
            average_precision_score(
                y_train,
                probabilities,
            ),
            4,
        ),
    )


print("\n" + "=" * 70)
print("OOF CALIBRATION COMPARISON")
print("=" * 70)

print(
    "\nActual OOF rate:",
    round(
        float(
            y_train.mean()
        ),
        4,
    ),
)


calibration_metrics(
    "RAW",
    oof_raw,
)

calibration_metrics(
    "ISOTONIC",
    oof_isotonic,
)

calibration_metrics(
    "SIGMOID",
    oof_sigmoid,
)


# ============================================================
# SELECT CALIBRATOR
# ============================================================

isotonic_brier = brier_score_loss(
    y_train,
    oof_isotonic,
)

sigmoid_brier = brier_score_loss(
    y_train,
    oof_sigmoid,
)


# Prefer sigmoid when its Brier score is very close to
# isotonic because sigmoid gives us a smoother probability
# mapping for individual patient predictions.

if sigmoid_brier <= isotonic_brier + 0.003:

    selected_name = "SIGMOID"

    selected_calibrator = sigmoid

    oof_calibrated = oof_sigmoid

else:

    selected_name = "ISOTONIC"

    selected_calibrator = isotonic

    oof_calibrated = oof_isotonic


print("\n" + "=" * 70)
print("SELECTED CALIBRATOR")
print("=" * 70)

print(
    "\nSelected:",
    selected_name,
)

print(
    "\nIsotonic Brier:",
    round(
        isotonic_brier,
        4,
    ),
)

print(
    "Sigmoid Brier:",
    round(
        sigmoid_brier,
        4,
    ),
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

threshold_results = []


for threshold in np.arange(
    0.05,
    0.61,
    0.01,
):

    predictions = (
        oof_calibrated
        >= threshold
    ).astype(int)

    threshold_results.append(
        {
            "threshold": float(
                threshold
            ),

            "accuracy": float(
                accuracy_score(
                    y_train,
                    predictions,
                )
            ),

            "precision": float(
                precision_score(
                    y_train,
                    predictions,
                    zero_division=0,
                )
            ),

            "recall": float(
                recall_score(
                    y_train,
                    predictions,
                    zero_division=0,
                )
            ),

            "f1": float(
                f1_score(
                    y_train,
                    predictions,
                    zero_division=0,
                )
            ),
        }
    )


eligible = [
    result
    for result in threshold_results
    if result["recall"] >= 0.70
]


best = max(
    eligible,
    key=lambda result:
        result["f1"],
)


production_threshold = best[
    "threshold"
]


print("\n" + "=" * 70)
print("SELECTED OPERATING THRESHOLD")
print("=" * 70)

print(
    "\nSelection rule:"
)

print(
    "Highest grouped OOF F1 with recall >= 0.70"
)

print(
    "\nThreshold:",
    round(
        production_threshold,
        3,
    ),
)

print(
    "\nAccuracy:",
    round(
        best["accuracy"],
        3,
    ),
)

print(
    "Precision:",
    round(
        best["precision"],
        3,
    ),
)

print(
    "Recall:",
    round(
        best["recall"],
        3,
    ),
)

print(
    "F1:",
    round(
        best["f1"],
        3,
    ),
)


# ============================================================
# RISK BANDS
# ============================================================

low_cutoff = float(
    np.quantile(
        oof_calibrated,
        0.50,
    )
)

high_cutoff = float(
    np.quantile(
        oof_calibrated,
        0.75,
    )
)


print("\n" + "=" * 70)
print("RISK BAND CUTOFFS")
print("=" * 70)

print(
    "\nLow / Moderate:",
    round(
        low_cutoff,
        3,
    ),
)

print(
    "Moderate / High:",
    round(
        high_cutoff,
        3,
    ),
)


# ============================================================
# FINAL MODEL
# ============================================================

final_model = build_model(
    y_train
)

final_model.fit(
    X_train,
    y_train,
)


# ============================================================
# HELD-OUT TEST
# ============================================================

test_raw = (
    final_model
    .predict_proba(
        X_test
    )[:, 1]
)


test_calibrated = (
    selected_calibrator
    .predict(
        test_raw
    )
)


test_predictions = (
    test_calibrated
    >= production_threshold
).astype(int)


print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 70)

print(
    "\nActual test rate:",
    round(
        float(
            y_test.mean()
        ),
        4,
    ),
)

print(
    "Mean calibrated probability:",
    round(
        float(
            test_calibrated.mean()
        ),
        4,
    ),
)

print(
    "\nBrier score:",
    round(
        brier_score_loss(
            y_test,
            test_calibrated,
        ),
        4,
    ),
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            test_calibrated,
        ),
        3,
    ),
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_test,
            test_calibrated,
        ),
        3,
    ),
)

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            test_predictions,
        ),
        3,
    ),
)

print(
    "Precision:",
    round(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        3,
    ),
)

print(
    "Recall:",
    round(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        3,
    ),
)

print(
    "F1:",
    round(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
        3,
    ),
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("HELD-OUT THRESHOLD COMPARISON")
print("=" * 70)


comparison_thresholds = sorted(
    set(
        [
            round(
                production_threshold,
                2,
            ),
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.50,
        ]
    )
)


for threshold in comparison_thresholds:

    predictions = (
        test_calibrated
        >= threshold
    ).astype(int)

    print(
        f"\nThreshold: "
        f"{threshold:.2f}"
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    print(
        "F1:",
        round(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    print(
        "Patients flagged:",
        int(
            predictions.sum()
        ),
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_DIR
    / "readmission_history_xgboost_calibrated.joblib",
)


# ============================================================
# SAVE CALIBRATOR
# ============================================================
#
# Because SigmoidCalibrator is imported from src.calibrators,
# joblib now stores its real module path instead of __main__.
# ============================================================

joblib.dump(
    selected_calibrator,
    MODEL_DIR
    / "readmission_history_calibrator.joblib",
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

with open(
    MODEL_DIR
    / "readmission_history_features.json",
    "w",
) as file:

    json.dump(
        FEATURES,
        file,
        indent=2,
    )


# ============================================================
# SAVE MEDIANS
# ============================================================

with open(
    MODEL_DIR
    / "readmission_history_medians.json",
    "w",
) as file:

    json.dump(
        medians,
        file,
        indent=2,
    )


# ============================================================
# SAVE THRESHOLD
# ============================================================

with open(
    MODEL_DIR
    / "readmission_history_threshold.json",
    "w",
) as file:

    json.dump(
        {
            "production_threshold":
                production_threshold,

            "selection_rule":
                "Highest grouped OOF F1 with recall >= 0.70",

            "calibration_method":
                selected_name,
        },
        file,
        indent=2,
    )


# ============================================================
# SAVE RISK BANDS
# ============================================================

with open(
    MODEL_DIR
    / "readmission_history_risk_bands.json",
    "w",
) as file:

    json.dump(
        {
            "low_moderate_cutoff":
                low_cutoff,

            "moderate_high_cutoff":
                high_cutoff,
        },
        file,
        indent=2,
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("HISTORY MODEL ARTIFACTS SAVED")
print("=" * 70)

print(
    "\nCalibration method:",
    selected_name,
)

print(
    "models/readmission_history_xgboost_calibrated.joblib"
)

print(
    "models/readmission_history_calibrator.joblib"
)

print(
    "models/readmission_history_features.json"
)

print(
    "models/readmission_history_medians.json"
)

print(
    "models/readmission_history_threshold.json"
)

print(
    "models/readmission_history_risk_bands.json"
)