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
    confusion_matrix,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/synthea_readmission_ml_dataset_v2.csv"
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
    "previous_encounters",
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
        n_estimators=350,
        max_depth=3,
        learning_rate=0.035,
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
print("ADMITRA V3 CALIBRATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nReadmission rate:")
print(
    round(
        df[TARGET].mean() * 100,
        2,
    ),
    "%",
)


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[
    FEATURES
].copy()

y = df[
    TARGET
].astype(int)


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

    X[feature] = (
        X[feature]
        .fillna(median)
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

if "patient_id" in df.columns:

    groups = df[
        "patient_id"
    ].astype(str)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.18,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

else:

    rng = np.random.RandomState(
        42
    )

    indices = np.arange(
        len(df)
    )

    rng.shuffle(
        indices
    )

    test_size = int(
        len(indices) * 0.18
    )

    test_idx = indices[
        :test_size
    ]

    train_idx = indices[
        test_size:
    ]


X_train = X.iloc[
    train_idx
].reset_index(
    drop=True
)

X_test = X.iloc[
    test_idx
].reset_index(
    drop=True
)

y_train = y.iloc[
    train_idx
].reset_index(
    drop=True
)

y_test = y.iloc[
    test_idx
].reset_index(
    drop=True
)


print("\n" + "=" * 70)
print("TRAIN / TEST CHECK")
print("=" * 70)

print("\nTraining rows:")
print(
    len(
        X_train
    )
)

print("\nTesting rows:")
print(
    len(
        X_test
    )
)


if "patient_id" in df.columns:

    train_patients = set(
        df.iloc[
            train_idx
        ][
            "patient_id"
        ]
    )

    test_patients = set(
        df.iloc[
            test_idx
        ][
            "patient_id"
        ]
    )

    overlap = (
        train_patients
        & test_patients
    )

    print("\nPatient overlap:")
    print(
        len(
            overlap
        )
    )


# ============================================================
# OUT-OF-FOLD RAW PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("BUILDING OUT-OF-FOLD CALIBRATION PREDICTIONS")
print("=" * 70)


oof_raw = np.zeros(
    len(
        X_train
    )
)


cv = StratifiedKFold(
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
        fold_model.predict_proba(
            X_train.iloc[
                fold_valid_idx
            ]
        )[:, 1]
    )

    oof_raw[
        fold_valid_idx
    ] = fold_probabilities

    print(
        f"Finished fold: "
        f"{fold_number}"
    )


# ============================================================
# FIT CALIBRATOR
# ============================================================

calibrator = IsotonicRegression(
    out_of_bounds="clip"
)

calibrator.fit(
    oof_raw,
    y_train,
)

oof_calibrated = (
    calibrator.predict(
        oof_raw
    )
)


# ============================================================
# CALIBRATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATION CHECK")
print("=" * 70)


print(
    "\nRaw mean predicted probability:"
)

print(
    round(
        float(
            np.mean(
                oof_raw
            )
        ),
        3,
    )
)


print(
    "\nCalibrated mean probability:"
)

print(
    round(
        float(
            np.mean(
                oof_calibrated
            )
        ),
        3,
    )
)


print(
    "\nActual training readmission rate:"
)

print(
    round(
        float(
            y_train.mean()
        ),
        3,
    )
)


print(
    "\nRaw Brier score:"
)

print(
    round(
        brier_score_loss(
            y_train,
            oof_raw,
        ),
        4,
    )
)


print(
    "\nCalibrated Brier score:"
)

print(
    round(
        brier_score_loss(
            y_train,
            oof_calibrated,
        ),
        4,
    )
)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

print("\n" + "=" * 70)
print("OOF THRESHOLD SELECTION")
print("=" * 70)


threshold_results = []


for threshold in np.arange(
    0.05,
    0.51,
    0.01,
):

    predictions = (
        oof_calibrated
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_train,
        predictions,
    )

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0,
    )

    threshold_results.append(
        {
            "threshold":
                float(threshold),

            "accuracy":
                float(accuracy),

            "precision":
                float(precision),

            "recall":
                float(recall),

            "f1":
                float(f1),
        }
    )


# Print useful comparison thresholds

for threshold in [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
]:

    closest = min(
        threshold_results,
        key=lambda result:
            abs(
                result[
                    "threshold"
                ]
                - threshold
            ),
    )

    print(
        f"\nThreshold: "
        f"{closest['threshold']:.2f}"
    )

    print(
        "Accuracy:",
        round(
            closest[
                "accuracy"
            ],
            3,
        ),
    )

    print(
        "Precision:",
        round(
            closest[
                "precision"
            ],
            3,
        ),
    )

    print(
        "Recall:",
        round(
            closest[
                "recall"
            ],
            3,
        ),
    )

    print(
        "F1:",
        round(
            closest[
                "f1"
            ],
            3,
        ),
    )

    print(
        "-" * 40
    )


# ============================================================
# SELECT PRODUCTION THRESHOLD
# ============================================================

eligible = [
    result
    for result
    in threshold_results
    if result[
        "recall"
    ] >= 0.70
]


best_threshold_result = max(
    eligible,
    key=lambda result:
        result[
            "f1"
        ],
)


production_threshold = (
    best_threshold_result[
        "threshold"
    ]
)


print("\n" + "=" * 70)
print("SELECTED PRODUCTION THRESHOLD")
print("=" * 70)


print(
    "\nSelection rule:"
)

print(
    "Highest OOF F1 with recall >= 0.70"
)


print(
    "\nSelected threshold:"
)

print(
    round(
        production_threshold,
        3,
    )
)


print(
    "\nOOF accuracy:"
)

print(
    round(
        best_threshold_result[
            "accuracy"
        ],
        3,
    )
)


print(
    "\nOOF precision:"
)

print(
    round(
        best_threshold_result[
            "precision"
        ],
        3,
    )
)


print(
    "\nOOF recall:"
)

print(
    round(
        best_threshold_result[
            "recall"
        ],
        3,
    )
)


print(
    "\nOOF F1:"
)

print(
    round(
        best_threshold_result[
            "f1"
        ],
        3,
    )
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
    "\nLow / Moderate cutoff:"
)

print(
    round(
        low_cutoff,
        3,
    )
)


print(
    "\nModerate / High cutoff:"
)

print(
    round(
        high_cutoff,
        3,
    )
)


# ============================================================
# TRAIN FINAL MODEL ON TRAINING DATA
# ============================================================

final_model = build_model(
    y_train
)

final_model.fit(
    X_train,
    y_train,
)


# ============================================================
# TEST SET
# ============================================================

test_raw = (
    final_model.predict_proba(
        X_test
    )[:, 1]
)

test_calibrated = (
    calibrator.predict(
        test_raw
    )
)

test_predictions = (
    test_calibrated
    >= production_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_predictions,
)

test_precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

test_roc_auc = roc_auc_score(
    y_test,
    test_calibrated,
)

test_pr_auc = average_precision_score(
    y_test,
    test_calibrated,
)

test_brier = brier_score_loss(
    y_test,
    test_calibrated,
)


print("\n" + "=" * 70)
print("FINAL CALIBRATED TEST RESULTS")
print("=" * 70)


print("\nAccuracy:")
print(
    round(
        test_accuracy,
        3,
    )
)

print("\nPrecision:")
print(
    round(
        test_precision,
        3,
    )
)

print("\nRecall:")
print(
    round(
        test_recall,
        3,
    )
)

print("\nF1:")
print(
    round(
        test_f1,
        3,
    )
)

print("\nROC-AUC:")
print(
    round(
        test_roc_auc,
        3,
    )
)

print("\nPR-AUC:")
print(
    round(
        test_pr_auc,
        3,
    )
)

print("\nCalibrated Brier score:")
print(
    round(
        test_brier,
        4,
    )
)

print("\nMean calibrated probability:")
print(
    round(
        float(
            test_calibrated.mean()
        ),
        3,
    )
)

print("\nActual test readmission rate:")
print(
    round(
        float(
            y_test.mean()
        ),
        3,
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions,
    )
)


# ============================================================
# TEST THRESHOLD COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CALIBRATED THRESHOLD COMPARISON")
print("=" * 70)


comparison_thresholds = sorted(
    set(
        [
            round(
                production_threshold,
                2,
            ),
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
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
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                predictions,
            ),
            3,
        ),
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

    print(
        "-" * 40
    )


# ============================================================
# SAVE V3 PRODUCTION ARTIFACTS
# ============================================================

joblib.dump(
    final_model,
    MODEL_DIR
    / "readmission_xgboost_v3.joblib",
)

joblib.dump(
    calibrator,
    MODEL_DIR
    / "readmission_calibrator_v3.joblib",
)


with open(
    MODEL_DIR
    / "readmission_feature_columns_v3.json",
    "w",
) as file:

    json.dump(
        FEATURES,
        file,
        indent=2,
    )


with open(
    MODEL_DIR
    / "readmission_clinical_medians_v3.json",
    "w",
) as file:

    json.dump(
        medians,
        file,
        indent=2,
    )


with open(
    MODEL_DIR
    / "readmission_threshold_v3.json",
    "w",
) as file:

    json.dump(
        {
            "production_threshold":
                production_threshold,

            "selection_rule":
                "Highest OOF F1 with recall >= 0.70",
        },
        file,
        indent=2,
    )


with open(
    MODEL_DIR
    / "readmission_risk_bands_v3.json",
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


print("\n" + "=" * 70)
print("V3 PRODUCTION ARTIFACTS SAVED")
print("=" * 70)

print(
    "\nmodels/readmission_xgboost_v3.joblib"
)

print(
    "models/readmission_calibrator_v3.joblib"
)

print(
    "models/readmission_feature_columns_v3.json"
)

print(
    "models/readmission_clinical_medians_v3.json"
)

print(
    "models/readmission_threshold_v3.json"
)

print(
    "models/readmission_risk_bands_v3.json"
)