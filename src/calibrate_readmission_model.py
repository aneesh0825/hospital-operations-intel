import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GroupShuffleSplit,
    GroupKFold
)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss
)

from xgboost import XGBClassifier


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v2.csv"
)

MODEL_PATH = (
    "models/readmission_xgboost_v2.joblib"
)

CALIBRATOR_PATH = (
    "models/readmission_calibrator_v2.joblib"
)

FEATURE_COLUMNS_PATH = (
    "models/readmission_feature_columns_v2.json"
)

MEDIANS_PATH = (
    "models/readmission_clinical_medians_v2.json"
)

THRESHOLD_PATH = (
    "models/readmission_threshold_v2.json"
)

RISK_BANDS_PATH = (
    "models/readmission_risk_bands_v2.json"
)


clinical_features = [
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose"
]


# --------------------------------------------------
# HELPER FUNCTION
# --------------------------------------------------

def probability_to_logit(probabilities):

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6
    )

    return np.log(
        probabilities
        /
        (1 - probabilities)
    )


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

data = pd.read_csv(
    DATA_PATH
)


data = data.drop(
    columns=[
        "height",
        "weight",
        "temperature"
    ],
    errors="ignore"
)


X = data.drop(
    columns=[
        "encounter_id",
        "patient_id",
        "readmitted_30_days"
    ]
)

y = data[
    "readmitted_30_days"
]

groups = data[
    "patient_id"
]


print("=" * 60)
print("CALIBRATION DATASET")
print("=" * 60)

print()

print("Shape:")
print(data.shape)

print()

print("Readmission rate:")
print(
    round(
        y.mean() * 100,
        2
    ),
    "%"
)


# --------------------------------------------------
# CREATE UNTOUCHED TEST SET
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_index, test_index = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[
    train_index
].copy()

X_test = X.iloc[
    test_index
].copy()


y_train = y.iloc[
    train_index
].copy()

y_test = y.iloc[
    test_index
].copy()


train_groups = groups.iloc[
    train_index
].copy()

test_groups = groups.iloc[
    test_index
].copy()


overlap = set(
    train_groups
).intersection(
    set(test_groups)
)


print()

print("=" * 60)
print("TRAIN / TEST CHECK")
print("=" * 60)

print()

print("Training rows:")
print(len(X_train))

print()

print("Testing rows:")
print(len(X_test))

print()

print("Patient overlap:")
print(len(overlap))


# --------------------------------------------------
# OUT-OF-FOLD PROBABILITIES FOR CALIBRATION
# --------------------------------------------------

print()

print("=" * 70)
print("BUILDING OUT-OF-FOLD CALIBRATION PREDICTIONS")
print("=" * 70)


group_kfold = GroupKFold(
    n_splits=5
)


oof_probabilities = np.zeros(
    len(X_train)
)


for fold, (
    fold_train_index,
    fold_validation_index
) in enumerate(
    group_kfold.split(
        X_train,
        y_train,
        groups=train_groups
    ),
    start=1
):


    X_fold_train = X_train.iloc[
        fold_train_index
    ].copy()


    X_fold_validation = X_train.iloc[
        fold_validation_index
    ].copy()


    y_fold_train = y_train.iloc[
        fold_train_index
    ].copy()


    # ----------------------------------------------
    # IMPUTE USING TRAINING FOLD ONLY
    # ----------------------------------------------

    for column in clinical_features:

        median_value = (
            X_fold_train[
                column
            ].median()
        )

        X_fold_train[
            column
        ] = (
            X_fold_train[
                column
            ].fillna(
                median_value
            )
        )

        X_fold_validation[
            column
        ] = (
            X_fold_validation[
                column
            ].fillna(
                median_value
            )
        )


    # ----------------------------------------------
    # ENCODE GENDER
    # ----------------------------------------------

    X_fold_train_encoded = (
        pd.get_dummies(
            X_fold_train,
            columns=["gender"],
            dtype=int
        )
    )


    X_fold_validation_encoded = (
        pd.get_dummies(
            X_fold_validation,
            columns=["gender"],
            dtype=int
        )
    )


    X_fold_validation_encoded = (
        X_fold_validation_encoded.reindex(
            columns=
            X_fold_train_encoded.columns,
            fill_value=0
        )
    )


    # ----------------------------------------------
    # CLASS BALANCE
    # ----------------------------------------------

    negative_count = (
        y_fold_train == 0
    ).sum()

    positive_count = (
        y_fold_train == 1
    ).sum()


    scale_pos_weight = (
        negative_count
        /
        positive_count
    )


    # ----------------------------------------------
    # TRAIN FOLD MODEL
    # ----------------------------------------------

    fold_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=
            scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )


    fold_model.fit(
        X_fold_train_encoded,
        y_fold_train
    )


    fold_probabilities = (
        fold_model.predict_proba(
            X_fold_validation_encoded
        )[:, 1]
    )


    oof_probabilities[
        fold_validation_index
    ] = fold_probabilities


    print(
        "Finished fold:",
        fold
    )


# --------------------------------------------------
# FIT PROBABILITY CALIBRATOR
# --------------------------------------------------

oof_logits = probability_to_logit(
    oof_probabilities
).reshape(-1, 1)


calibrator = LogisticRegression()

calibrator.fit(
    oof_logits,
    y_train
)


calibrated_oof_probabilities = (
    calibrator.predict_proba(
        oof_logits
    )[:, 1]
)


print()

print("=" * 60)
print("CALIBRATION CHECK")
print("=" * 60)

print()

print("Raw mean predicted probability:")
print(
    round(
        oof_probabilities.mean(),
        3
    )
)

print()

print("Calibrated mean probability:")
print(
    round(
        calibrated_oof_probabilities.mean(),
        3
    )
)

print()

print("Actual training readmission rate:")
print(
    round(
        y_train.mean(),
        3
    )
)

print()

print("Raw Brier score:")
print(
    round(
        brier_score_loss(
            y_train,
            oof_probabilities
        ),
        4
    )
)

print()

print("Calibrated Brier score:")
print(
    round(
        brier_score_loss(
            y_train,
            calibrated_oof_probabilities
        ),
        4
    )
)


# --------------------------------------------------
# CHOOSE THRESHOLD FROM OOF CALIBRATED PREDICTIONS
# --------------------------------------------------

print()

print("=" * 70)
print("OOF THRESHOLD SELECTION")
print("=" * 70)


# Candidate thresholds are evaluated ONLY on
# out-of-fold training predictions.
#
# The final test set is NOT used to choose the threshold.

thresholds = np.arange(
    0.05,
    0.51,
    0.01
)


threshold_results = []


for threshold in thresholds:

    predictions = (
        calibrated_oof_probabilities
        >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_train,
        predictions
    )

    precision = precision_score(
        y_train,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_train,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_train,
        predictions,
        zero_division=0
    )


    threshold_results.append(
        {
            "threshold": float(threshold),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


# --------------------------------------------------
# REQUIRE AT LEAST 70% RECALL
# --------------------------------------------------

eligible_thresholds = [
    result
    for result in threshold_results
    if result["recall"] >= 0.70
]


# --------------------------------------------------
# SELECT HIGHEST F1 AMONG ELIGIBLE THRESHOLDS
# --------------------------------------------------

if len(eligible_thresholds) > 0:

    best_result = max(
        eligible_thresholds,
        key=lambda result:
            result["f1"]
    )

else:

    # Safety fallback:
    # if no threshold reaches 70% recall,
    # simply choose the highest-F1 threshold.

    best_result = max(
        threshold_results,
        key=lambda result:
            result["f1"]
    )


best_threshold = round(
    best_result["threshold"],
    2
)


best_f1 = (
    best_result["f1"]
)


# --------------------------------------------------
# DISPLAY USEFUL OOF THRESHOLDS
# --------------------------------------------------

for result in threshold_results:

    if round(
        result["threshold"],
        2
    ) in [
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40
    ]:

        print()

        print(
            "Threshold:",
            round(
                result["threshold"],
                2
            )
        )

        print(
            "Accuracy:",
            round(
                result["accuracy"],
                3
            )
        )

        print(
            "Precision:",
            round(
                result["precision"],
                3
            )
        )

        print(
            "Recall:",
            round(
                result["recall"],
                3
            )
        )

        print(
            "F1:",
            round(
                result["f1"],
                3
            )
        )

        print(
            "-" * 40
        )


# --------------------------------------------------
# FINAL SELECTED THRESHOLD
# --------------------------------------------------

print()

print("=" * 60)
print("SELECTED PRODUCTION THRESHOLD")
print("=" * 60)

print()

print("Selection rule:")
print(
    "Highest OOF F1 with recall >= 0.70"
)

print()

print("Selected threshold:")
print(
    best_threshold
)

print()

print("OOF accuracy:")
print(
    round(
        best_result["accuracy"],
        3
    )
)

print()

print("OOF precision:")
print(
    round(
        best_result["precision"],
        3
    )
)

print()

print("OOF recall:")
print(
    round(
        best_result["recall"],
        3
    )
)

print()

print("OOF F1:")
print(
    round(
        best_result["f1"],
        3
    )
)

# --------------------------------------------------
# CREATE RELATIVE RISK BANDS
# --------------------------------------------------

low_cutoff = float(
    np.quantile(
        calibrated_oof_probabilities,
        0.50
    )
)

high_cutoff = float(
    np.quantile(
        calibrated_oof_probabilities,
        0.80
    )
)


print()

print("=" * 60)
print("RISK BAND CUTOFFS")
print("=" * 60)

print()

print("Low / Moderate cutoff:")
print(
    round(
        low_cutoff,
        3
    )
)

print()

print("Moderate / High cutoff:")
print(
    round(
        high_cutoff,
        3
    )
)


# --------------------------------------------------
# PREPARE FULL TRAINING DATA
# --------------------------------------------------

clinical_medians = {}


for column in clinical_features:

    median_value = (
        X_train[
            column
        ].median()
    )

    clinical_medians[
        column
    ] = float(
        median_value
    )

    X_train[
        column
    ] = (
        X_train[
            column
        ].fillna(
            median_value
        )
    )

    X_test[
        column
    ] = (
        X_test[
            column
        ].fillna(
            median_value
        )
    )


# --------------------------------------------------
# ENCODE FULL TRAIN / TEST
# --------------------------------------------------

X_train_encoded = pd.get_dummies(
    X_train,
    columns=["gender"],
    dtype=int
)


X_test_encoded = pd.get_dummies(
    X_test,
    columns=["gender"],
    dtype=int
)


X_test_encoded = (
    X_test_encoded.reindex(
        columns=
        X_train_encoded.columns,
        fill_value=0
    )
)


feature_columns = (
    X_train_encoded.columns.tolist()
)


# --------------------------------------------------
# TRAIN FINAL XGBOOST MODEL
# --------------------------------------------------

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()


scale_pos_weight = (
    negative_count
    /
    positive_count
)


final_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=
        scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)


final_model.fit(
    X_train_encoded,
    y_train
)


# --------------------------------------------------
# RAW TEST PROBABILITIES
# --------------------------------------------------

raw_test_probabilities = (
    final_model.predict_proba(
        X_test_encoded
    )[:, 1]
)


# --------------------------------------------------
# CALIBRATE TEST PROBABILITIES
# --------------------------------------------------

test_logits = probability_to_logit(
    raw_test_probabilities
).reshape(-1, 1)


calibrated_test_probabilities = (
    calibrator.predict_proba(
        test_logits
    )[:, 1]
)


final_predictions = (
    calibrated_test_probabilities
    >= best_threshold
).astype(int)


# --------------------------------------------------
# FINAL CALIBRATED TEST RESULTS
# --------------------------------------------------

print()

print("=" * 70)
print("FINAL CALIBRATED TEST RESULTS")
print("=" * 70)

print()

print("Accuracy:")
print(
    round(
        accuracy_score(
            y_test,
            final_predictions
        ),
        3
    )
)

print()

print("Precision:")
print(
    round(
        precision_score(
            y_test,
            final_predictions,
            zero_division=0
        ),
        3
    )
)

print()

print("Recall:")
print(
    round(
        recall_score(
            y_test,
            final_predictions,
            zero_division=0
        ),
        3
    )
)

print()

print("F1:")
print(
    round(
        f1_score(
            y_test,
            final_predictions,
            zero_division=0
        ),
        3
    )
)

print()

print("ROC-AUC:")
print(
    round(
        roc_auc_score(
            y_test,
            calibrated_test_probabilities
        ),
        3
    )
)

print()

print("PR-AUC:")
print(
    round(
        average_precision_score(
            y_test,
            calibrated_test_probabilities
        ),
        3
    )
)

print()

print("Calibrated Brier score:")
print(
    round(
        brier_score_loss(
            y_test,
            calibrated_test_probabilities
        ),
        4
    )
)

print()

print("Mean calibrated probability:")
print(
    round(
        calibrated_test_probabilities.mean(),
        3
    )
)

print()

print("Actual test readmission rate:")
print(
    round(
        y_test.mean(),
        3
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        final_predictions
    )
)
# --------------------------------------------------
# CALIBRATED THRESHOLD COMPARISON
# --------------------------------------------------

print()

print("=" * 70)
print("CALIBRATED THRESHOLD COMPARISON")
print("=" * 70)


comparison_thresholds = [
    0.16,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]


for threshold in comparison_thresholds:

    predictions = (
        calibrated_test_probabilities
        >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )


    print()

    print(
        "Threshold:",
        threshold
    )

    print(
        "Accuracy:",
        round(
            accuracy,
            3
        )
    )

    print(
        "Precision:",
        round(
            precision,
            3
        )
    )

    print(
        "Recall:",
        round(
            recall,
            3
        )
    )

    print(
        "F1:",
        round(
            f1,
            3
        )
    )

    print(
        "Patients flagged:",
        int(
            predictions.sum()
        )
    )

    print(
        "-" * 40
    )

# --------------------------------------------------
# SAVE PRODUCTION ARTIFACTS
# --------------------------------------------------

joblib.dump(
    final_model,
    MODEL_PATH
)

joblib.dump(
    calibrator,
    CALIBRATOR_PATH
)


with open(
    FEATURE_COLUMNS_PATH,
    "w"
) as file:

    json.dump(
        feature_columns,
        file,
        indent=4
    )


with open(
    MEDIANS_PATH,
    "w"
) as file:

    json.dump(
        clinical_medians,
        file,
        indent=4
    )


with open(
    THRESHOLD_PATH,
    "w"
) as file:

    json.dump(
        {
            "threshold":
                best_threshold
        },
        file,
        indent=4
    )


with open(
    RISK_BANDS_PATH,
    "w"
) as file:

    json.dump(
        {
            "low_cutoff":
                low_cutoff,

            "high_cutoff":
                high_cutoff
        },
        file,
        indent=4
    )


print()

print("=" * 70)
print("CALIBRATED PRODUCTION ARTIFACTS SAVED")
print("=" * 70)

print()

print(MODEL_PATH)
print(CALIBRATOR_PATH)
print(FEATURE_COLUMNS_PATH)
print(MEDIANS_PATH)
print(THRESHOLD_PATH)
print(RISK_BANDS_PATH)