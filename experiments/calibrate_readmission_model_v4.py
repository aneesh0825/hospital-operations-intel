import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
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


# ============================================================
# PATHS / SETTINGS
# ============================================================

DATA_PATH = Path(
    "data/processed/synthea_readmission_ml_dataset_v2.csv"
)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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
# SIGMOID / PLATT CALIBRATOR
# ============================================================

class SigmoidCalibrator:

    def __init__(self):
        self.model = LogisticRegression(
            solver="lbfgs"
        )

    def fit(self, probabilities, y):

        probabilities = np.asarray(
            probabilities
        )

        probabilities = np.clip(
            probabilities,
            1e-6,
            1 - 1e-6,
        )

        logits = np.log(
            probabilities
            / (1 - probabilities)
        ).reshape(-1, 1)

        self.model.fit(
            logits,
            y,
        )

        return self

    def predict(self, probabilities):

        probabilities = np.asarray(
            probabilities
        )

        probabilities = np.clip(
            probabilities,
            1e-6,
            1 - 1e-6,
        )

        logits = np.log(
            probabilities
            / (1 - probabilities)
        ).reshape(-1, 1)

        return self.model.predict_proba(
            logits
        )[:, 1]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH
)

print("=" * 70)
print("ADMITRA V4 CALIBRATION")
print("=" * 70)

print("\nRows:")
print(
    f"{len(df):,}"
)

print("\nOverall readmission rate:")
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

    X[feature] = (
        X[feature]
        .fillna(median)
    )


# ============================================================
# GROUPED TRAIN / TEST SPLIT
# ============================================================

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

groups_train = groups.iloc[
    train_idx
].reset_index(
    drop=True
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
print("GROUPED TRAIN / TEST SPLIT")
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
# GROUPED OOF PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GROUPED OUT-OF-FOLD PREDICTIONS")
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

    probabilities = (
        fold_model.predict_proba(
            X_train.iloc[
                fold_valid_idx
            ]
        )[:, 1]
    )

    oof_raw[
        fold_valid_idx
    ] = probabilities

    train_group_set = set(
        groups_train.iloc[
            fold_train_idx
        ]
    )

    valid_group_set = set(
        groups_train.iloc[
            fold_valid_idx
        ]
    )

    overlap = len(
        train_group_set
        & valid_group_set
    )

    print(
        f"Finished fold {fold_number} "
        f"| patient overlap: {overlap}"
    )


# ============================================================
# FIT BOTH CALIBRATORS
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
# OOF CALIBRATION COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("OOF CALIBRATION COMPARISON")
print("=" * 70)


def print_calibration_metrics(
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


print(
    "\nActual OOF readmission rate:",
    round(
        float(
            y_train.mean()
        ),
        4,
    ),
)


print_calibration_metrics(
    "RAW",
    oof_raw,
)

print_calibration_metrics(
    "ISOTONIC",
    oof_isotonic,
)

print_calibration_metrics(
    "SIGMOID",
    oof_sigmoid,
)


# ============================================================
# CHOOSE CALIBRATOR
# ============================================================

isotonic_brier = (
    brier_score_loss(
        y_train,
        oof_isotonic,
    )
)

sigmoid_brier = (
    brier_score_loss(
        y_train,
        oof_sigmoid,
    )
)


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
    "\nReason:"
)

print(
    "Prefer sigmoid when calibration quality is "
    "competitive because it produces a smooth probability mapping."
)


# ============================================================
# THRESHOLD SELECTION
# ============================================================

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

    result = {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            accuracy_score(
                y_train,
                predictions,
            ),

        "precision":
            precision_score(
                y_train,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_train,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_train,
                predictions,
                zero_division=0,
            ),
    }

    threshold_results.append(
        result
    )


eligible = [
    result
    for result
    in threshold_results
    if result[
        "recall"
    ] >= 0.70
]


best = max(
    eligible,
    key=lambda result:
        result[
            "f1"
        ],
)


production_threshold = (
    best[
        "threshold"
    ]
)


print("\n" + "=" * 70)
print("SELECTED OPERATING THRESHOLD")
print("=" * 70)

print(
    "\nThreshold:",
    round(
        production_threshold,
        3,
    )
)

print(
    "\nOOF Accuracy:",
    round(
        best[
            "accuracy"
        ],
        3,
    )
)

print(
    "OOF Precision:",
    round(
        best[
            "precision"
        ],
        3,
    )
)

print(
    "OOF Recall:",
    round(
        best[
            "recall"
        ],
        3,
    )
)

print(
    "OOF F1:",
    round(
        best[
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
print("RISK BANDS")
print("=" * 70)

print(
    "\nLow / Moderate:",
    round(
        low_cutoff,
        3,
    )
)

print(
    "Moderate / High:",
    round(
        high_cutoff,
        3,
    )
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


test_raw = (
    final_model.predict_proba(
        X_test
    )[:, 1]
)


if selected_name == "SIGMOID":

    test_calibrated = (
        selected_calibrator.predict(
            test_raw
        )
    )

else:

    test_calibrated = (
        selected_calibrator.predict(
            test_raw
        )
    )


# ============================================================
# TEST RESULTS
# ============================================================

test_predictions = (
    test_calibrated
    >= production_threshold
).astype(int)


print("\n" + "=" * 70)
print("FINAL HELD-OUT TEST RESULTS")
print("=" * 70)


print(
    "\nMean calibrated probability:",
    round(
        float(
            test_calibrated.mean()
        ),
        3,
    )
)

print(
    "Actual readmission rate:",
    round(
        float(
            y_test.mean()
        ),
        3,
    )
)

print(
    "\nBrier score:",
    round(
        brier_score_loss(
            y_test,
            test_calibrated,
        ),
        4,
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test,
            test_calibrated,
        ),
        3,
    )
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_test,
            test_calibrated,
        ),
        3,
    )
)

print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            test_predictions,
        ),
        3,
    )
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
    )
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
    )
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
    )
)


# ============================================================
# ADMISSION RESPONSE CURVE
# ============================================================

print("\n" + "=" * 70)
print("PRIOR ADMISSION RESPONSE")
print("=" * 70)


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

    patient = (
        base_patient.copy()
    )

    patient[
        "previous_inpatient_admissions"
    ] = admissions

    patient_df = pd.DataFrame(
        [
            [
                patient[
                    feature
                ]
                for feature
                in FEATURES
            ]
        ],
        columns=FEATURES,
    )

    raw = (
        final_model.predict_proba(
            patient_df
        )[0, 1]
    )

    calibrated = (
        selected_calibrator.predict(
            [raw]
        )[0]
    )

    print(
        f"Admissions: {admissions:>2} | "
        f"Raw: {raw * 100:>6.1f}% | "
        f"Calibrated: {calibrated * 100:>6.1f}%"
    )


# ============================================================
# SAVE V4
# ============================================================

joblib.dump(
    final_model,
    MODEL_DIR
    / "readmission_xgboost_v4.joblib",
)

joblib.dump(
    selected_calibrator,
    MODEL_DIR
    / "readmission_calibrator_v4.joblib",
)


with open(
    MODEL_DIR
    / "readmission_feature_columns_v4.json",
    "w",
) as file:

    json.dump(
        FEATURES,
        file,
        indent=2,
    )


with open(
    MODEL_DIR
    / "readmission_clinical_medians_v4.json",
    "w",
) as file:

    json.dump(
        medians,
        file,
        indent=2,
    )


with open(
    MODEL_DIR
    / "readmission_threshold_v4.json",
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


with open(
    MODEL_DIR
    / "readmission_risk_bands_v4.json",
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
print("V4 ARTIFACTS SAVED")
print("=" * 70)

print(
    "\nCalibration method:",
    selected_name,
)

print(
    "models/readmission_xgboost_v4.joblib"
)

print(
    "models/readmission_calibrator_v4.joblib"
)

print(
    "models/readmission_feature_columns_v4.json"
)

print(
    "models/readmission_clinical_medians_v4.json"
)

print(
    "models/readmission_threshold_v4.json"
)

print(
    "models/readmission_risk_bands_v4.json"
)