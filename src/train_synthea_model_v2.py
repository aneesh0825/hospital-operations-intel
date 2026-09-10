import os
import json
import joblib

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from xgboost import XGBClassifier


# --------------------------------------------------
# LOAD SYNTHEA V2 READMISSION DATASET
# --------------------------------------------------

data = pd.read_csv(
    "data/processed/synthea_readmission_ml_dataset_v2.csv"
)


# --------------------------------------------------
# KEEP SELECTED V2 CLINICAL FEATURES
# --------------------------------------------------

v2_features = [
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose"
]


# Drop sparse/redundant observation features
data = data.drop(
    columns=[
        "height",
        "weight",
        "temperature"
    ],
    errors="ignore"
)


print("=" * 50)
print("SYNTHEA V2 READMISSION DATASET")
print("=" * 50)

print()

print("Dataset shape:")
print(data.shape)

print()

print("Target distribution:")
print(
    data["readmitted_30_days"].value_counts()
)

print()

print("Target percentages:")
print(
    data["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()

print("V2 clinical features:")
print(v2_features)


# --------------------------------------------------
# DEFINE FEATURES, TARGET, AND PATIENT GROUPS
# --------------------------------------------------

X = data.drop(
    columns=[
        "encounter_id",
        "patient_id",
        "readmitted_30_days"
    ]
)

y = data["readmitted_30_days"]

groups = data["patient_id"]


print()

print("=" * 50)
print("MODEL FEATURES")
print("=" * 50)

print(X.columns.tolist())

print()

print("Number of features:")
print(X.shape[1])


# --------------------------------------------------
# PATIENT-LEVEL TRAIN / TEST SPLIT
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


X_train = X.iloc[train_index].copy()
X_test = X.iloc[test_index].copy()

y_train = y.iloc[train_index].copy()
y_test = y.iloc[test_index].copy()

train_patients = groups.iloc[train_index].copy()
test_patients = groups.iloc[test_index].copy()


# --------------------------------------------------
# VERIFY PATIENT SEPARATION
# --------------------------------------------------

patient_overlap = set(
    train_patients
).intersection(
    set(test_patients)
)


print()

print("=" * 50)
print("PATIENT-LEVEL TRAIN / TEST SPLIT")
print("=" * 50)

print()

print("Training hospitalizations:")
print(len(X_train))

print()

print("Testing hospitalizations:")
print(len(X_test))

print()

print("Unique training patients:")
print(train_patients.nunique())

print()

print("Unique testing patients:")
print(test_patients.nunique())

print()

print("Patients appearing in BOTH sets:")
print(len(patient_overlap))

print()

print("Training target distribution:")
print(y_train.value_counts())

print()

print("Training target percentages:")
print(
    y_train
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print()

print("Testing target distribution:")
print(y_test.value_counts())

print()

print("Testing target percentages:")
print(
    y_test
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# IMPUTE MISSING V2 CLINICAL VALUES
# --------------------------------------------------

print()

print("=" * 50)
print("V2 IMPUTATION")
print("=" * 50)


for column in v2_features:

    median_value = X_train[
        column
    ].median()

    X_train[column] = (
        X_train[column]
        .fillna(median_value)
    )

    X_test[column] = (
        X_test[column]
        .fillna(median_value)
    )

    print(
        column,
        "training median:",
        round(median_value, 3)
    )


print()

print("Remaining missing values in training data:")
print(
    X_train.isna().sum().sum()
)

print()

print("Remaining missing values in testing data:")
print(
    X_test.isna().sum().sum()
)


# --------------------------------------------------
# ENCODE FULL TRAIN / TEST DATA
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


X_test_encoded = X_test_encoded.reindex(
    columns=X_train_encoded.columns,
    fill_value=0
)


print()

print("=" * 50)
print("ENCODED FEATURES")
print("=" * 50)

print()

print("Training shape:")
print(X_train_encoded.shape)

print()

print("Testing shape:")
print(X_test_encoded.shape)

print()

print("Encoded columns:")
print(
    X_train_encoded.columns.tolist()
)


# --------------------------------------------------
# MAJORITY-CLASS BASELINE
# --------------------------------------------------

dummy_model = DummyClassifier(
    strategy="most_frequent"
)

dummy_model.fit(
    X_train_encoded,
    y_train
)

dummy_pred = dummy_model.predict(
    X_test_encoded
)


print()

print("=" * 50)
print("MAJORITY-CLASS BASELINE")
print("=" * 50)

print()

print("Accuracy:")
print(
    round(
        accuracy_score(
            y_test,
            dummy_pred
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
            dummy_pred,
            zero_division=0
        ),
        3
    )
)

print()

print("F1 Score:")
print(
    round(
        f1_score(
            y_test,
            dummy_pred,
            zero_division=0
        ),
        3
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        dummy_pred
    )
)


# --------------------------------------------------
# LOGISTIC REGRESSION BASELINE
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_encoded
)

X_test_scaled = scaler.transform(
    X_test_encoded
)


logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    random_state=42
)


logistic_model.fit(
    X_train_scaled,
    y_train
)


logistic_pred = logistic_model.predict(
    X_test_scaled
)

logistic_prob = logistic_model.predict_proba(
    X_test_scaled
)[:, 1]


# --------------------------------------------------
# LOGISTIC REGRESSION RESULTS
# --------------------------------------------------

print()

print("=" * 50)
print("LOGISTIC REGRESSION")
print("=" * 50)

print()

print("Accuracy:")
print(
    round(
        accuracy_score(
            y_test,
            logistic_pred
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
            logistic_pred,
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
            logistic_pred,
            zero_division=0
        ),
        3
    )
)

print()

print("F1 Score:")
print(
    round(
        f1_score(
            y_test,
            logistic_pred,
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
            logistic_prob
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
            logistic_prob
        ),
        3
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        logistic_pred
    )
)


# --------------------------------------------------
# TRAIN / VALIDATION SPLIT INSIDE TRAINING PATIENTS
# --------------------------------------------------

validation_splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=24
)

inner_train_index, validation_index = next(
    validation_splitter.split(
        X_train,
        y_train,
        groups=train_patients
    )
)


X_inner_train = X_train.iloc[
    inner_train_index
].copy()

X_validation = X_train.iloc[
    validation_index
].copy()

y_inner_train = y_train.iloc[
    inner_train_index
].copy()

y_validation = y_train.iloc[
    validation_index
].copy()

inner_train_patients = train_patients.iloc[
    inner_train_index
].copy()

validation_patients = train_patients.iloc[
    validation_index
].copy()


# --------------------------------------------------
# VERIFY INNER TRAIN / VALIDATION SEPARATION
# --------------------------------------------------

validation_overlap = set(
    inner_train_patients
).intersection(
    set(validation_patients)
)


print()

print("=" * 50)
print("INNER TRAIN / VALIDATION SPLIT")
print("=" * 50)

print()

print("Inner training hospitalizations:")
print(len(X_inner_train))

print()

print("Validation hospitalizations:")
print(len(X_validation))

print()

print("Unique inner training patients:")
print(inner_train_patients.nunique())

print()

print("Unique validation patients:")
print(validation_patients.nunique())

print()

print("Patients appearing in BOTH:")
print(len(validation_overlap))

print()

print("Inner training target distribution:")
print(y_inner_train.value_counts())

print()

print("Validation target distribution:")
print(y_validation.value_counts())

print()

print("Validation target percentages:")
print(
    y_validation
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# --------------------------------------------------
# IMPUTE INNER TRAIN / VALIDATION
# --------------------------------------------------

for column in v2_features:

    median_value = X_inner_train[
        column
    ].median()

    X_inner_train[column] = (
        X_inner_train[column]
        .fillna(median_value)
    )

    X_validation[column] = (
        X_validation[column]
        .fillna(median_value)
    )


# --------------------------------------------------
# ENCODE INNER TRAIN / VALIDATION
# --------------------------------------------------

X_inner_train_encoded = pd.get_dummies(
    X_inner_train,
    columns=["gender"],
    dtype=int
)

X_validation_encoded = pd.get_dummies(
    X_validation,
    columns=["gender"],
    dtype=int
)


X_validation_encoded = (
    X_validation_encoded.reindex(
        columns=X_inner_train_encoded.columns,
        fill_value=0
    )
)


# --------------------------------------------------
# TRAIN VALIDATION XGBOOST MODEL
# --------------------------------------------------

inner_negative_count = (
    y_inner_train == 0
).sum()

inner_positive_count = (
    y_inner_train == 1
).sum()

inner_scale_pos_weight = (
    inner_negative_count
    / inner_positive_count
)


validation_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=inner_scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)


validation_model.fit(
    X_inner_train_encoded,
    y_inner_train
)


validation_prob = validation_model.predict_proba(
    X_validation_encoded
)[:, 1]


# --------------------------------------------------
# SELECT THRESHOLD USING VALIDATION DATA
# --------------------------------------------------

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]

best_threshold = None
best_f1 = -1


print()

print("=" * 70)
print("VALIDATION THRESHOLD ANALYSIS")
print("=" * 70)


for threshold in thresholds:

    validation_pred = (
        validation_prob >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    precision = precision_score(
        y_validation,
        validation_pred,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_pred,
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


    if f1 > best_f1:

        best_f1 = f1
        best_threshold = threshold


print()

print("=" * 50)
print("SELECTED THRESHOLD")
print("=" * 50)

print()

print("Best threshold:")
print(best_threshold)

print()

print("Validation F1:")
print(
    round(
        best_f1,
        3
    )
)


# --------------------------------------------------
# FINAL XGBOOST MODEL
# --------------------------------------------------

negative_count = (
    y_train == 0
).sum()

positive_count = (
    y_train == 1
).sum()

scale_pos_weight = (
    negative_count
    / positive_count
)


print()

print("=" * 50)
print("XGBOOST CLASS BALANCE")
print("=" * 50)

print()

print("Negative training cases:")
print(negative_count)

print()

print("Positive training cases:")
print(positive_count)

print()

print("Scale positive weight:")
print(
    round(
        scale_pos_weight,
        3
    )
)


xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)


xgb_model.fit(
    X_train_encoded,
    y_train
)


xgb_prob = xgb_model.predict_proba(
    X_test_encoded
)[:, 1]


# --------------------------------------------------
# DEFAULT XGBOOST RESULTS
# --------------------------------------------------

xgb_default_pred = (
    xgb_prob >= 0.50
).astype(int)


print()

print("=" * 50)
print("XGBOOST DEFAULT THRESHOLD")
print("=" * 50)

print()

print("Threshold:")
print(0.50)

print()

print("Accuracy:")
print(
    round(
        accuracy_score(
            y_test,
            xgb_default_pred
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
            xgb_default_pred,
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
            xgb_default_pred,
            zero_division=0
        ),
        3
    )
)

print()

print("F1 Score:")
print(
    round(
        f1_score(
            y_test,
            xgb_default_pred,
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
            xgb_prob
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
            xgb_prob
        ),
        3
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        xgb_default_pred
    )
)


# --------------------------------------------------
# FINAL TEST RESULTS USING VALIDATION THRESHOLD
# --------------------------------------------------

final_pred = (
    xgb_prob >= best_threshold
).astype(int)


print()

print("=" * 60)
print("FINAL XGBOOST V2 TEST RESULTS")
print("=" * 60)

print()

print("Selected threshold:")
print(best_threshold)

print()

print("Accuracy:")
print(
    round(
        accuracy_score(
            y_test,
            final_pred
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
            final_pred,
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
            final_pred,
            zero_division=0
        ),
        3
    )
)

print()

print("F1 Score:")
print(
    round(
        f1_score(
            y_test,
            final_pred,
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
            xgb_prob
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
            xgb_prob
        ),
        3
    )
)

print()

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        final_pred
    )
)


# --------------------------------------------------
# 5-FOLD PATIENT-LEVEL CROSS-VALIDATION
# --------------------------------------------------

print()

print("=" * 70)
print("5-FOLD PATIENT-LEVEL CROSS-VALIDATION")
print("=" * 70)


group_kfold = GroupKFold(
    n_splits=5
)

cv_results = []


for fold, (
    cv_train_index,
    cv_validation_index
) in enumerate(
    group_kfold.split(
        X,
        y,
        groups=groups
    ),
    start=1
):

    X_cv_train = X.iloc[
        cv_train_index
    ].copy()

    X_cv_validation = X.iloc[
        cv_validation_index
    ].copy()

    y_cv_train = y.iloc[
        cv_train_index
    ].copy()

    y_cv_validation = y.iloc[
        cv_validation_index
    ].copy()

    cv_train_groups = groups.iloc[
        cv_train_index
    ]

    cv_validation_groups = groups.iloc[
        cv_validation_index
    ]


    # ----------------------------------------------
    # Impute missing clinical values
    # using training-fold medians only
    # ----------------------------------------------

    for column in v2_features:

        median_value = X_cv_train[
            column
        ].median()

        X_cv_train[column] = (
            X_cv_train[column]
            .fillna(median_value)
        )

        X_cv_validation[column] = (
            X_cv_validation[column]
            .fillna(median_value)
        )


    # ----------------------------------------------
    # Encode categorical variables
    # ----------------------------------------------

    X_cv_train_encoded = pd.get_dummies(
        X_cv_train,
        columns=["gender"],
        dtype=int
    )

    X_cv_validation_encoded = pd.get_dummies(
        X_cv_validation,
        columns=["gender"],
        dtype=int
    )


    X_cv_validation_encoded = (
        X_cv_validation_encoded.reindex(
            columns=X_cv_train_encoded.columns,
            fill_value=0
        )
    )


    # ----------------------------------------------
    # Calculate class imbalance
    # ----------------------------------------------

    cv_negative_count = (
        y_cv_train == 0
    ).sum()

    cv_positive_count = (
        y_cv_train == 1
    ).sum()

    cv_scale_pos_weight = (
        cv_negative_count
        / cv_positive_count
    )


    # ----------------------------------------------
    # Train XGBoost
    # ----------------------------------------------

    cv_model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=cv_scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )


    cv_model.fit(
        X_cv_train_encoded,
        y_cv_train
    )


    # ----------------------------------------------
    # Predictions
    # ----------------------------------------------

    cv_prob = cv_model.predict_proba(
        X_cv_validation_encoded
    )[:, 1]

    cv_pred = (
        cv_prob >= best_threshold
    ).astype(int)


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    cv_accuracy = accuracy_score(
        y_cv_validation,
        cv_pred
    )

    cv_precision = precision_score(
        y_cv_validation,
        cv_pred,
        zero_division=0
    )

    cv_recall = recall_score(
        y_cv_validation,
        cv_pred,
        zero_division=0
    )

    cv_f1 = f1_score(
        y_cv_validation,
        cv_pred,
        zero_division=0
    )

    cv_roc_auc = roc_auc_score(
        y_cv_validation,
        cv_prob
    )

    cv_pr_auc = average_precision_score(
        y_cv_validation,
        cv_prob
    )


    cv_results.append(
        {
            "fold": fold,
            "accuracy": cv_accuracy,
            "precision": cv_precision,
            "recall": cv_recall,
            "f1": cv_f1,
            "roc_auc": cv_roc_auc,
            "pr_auc": cv_pr_auc
        }
    )


    cv_overlap = set(
        cv_train_groups
    ).intersection(
        set(cv_validation_groups)
    )


    print()

    print("Fold:", fold)

    print(
        "Patient overlap:",
        len(cv_overlap)
    )

    print(
        "Accuracy:",
        round(
            cv_accuracy,
            3
        )
    )

    print(
        "Precision:",
        round(
            cv_precision,
            3
        )
    )

    print(
        "Recall:",
        round(
            cv_recall,
            3
        )
    )

    print(
        "F1:",
        round(
            cv_f1,
            3
        )
    )

    print(
        "ROC-AUC:",
        round(
            cv_roc_auc,
            3
        )
    )

    print(
        "PR-AUC:",
        round(
            cv_pr_auc,
            3
        )
    )

    print("-" * 40)


# --------------------------------------------------
# CROSS-VALIDATION SUMMARY
# --------------------------------------------------

cv_results_df = pd.DataFrame(
    cv_results
)


print()

print("=" * 70)
print("CROSS-VALIDATION SUMMARY")
print("=" * 70)

print()


for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc"
]:

    mean_value = cv_results_df[
        metric
    ].mean()

    std_value = cv_results_df[
        metric
    ].std()


    print(
        metric.upper(),
        "Mean:",
        round(
            mean_value,
            3
        ),
        "| Std:",
        round(
            std_value,
            3
        )
    )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_importance = pd.DataFrame(
    {
        "feature": X_train_encoded.columns,
        "importance": xgb_model.feature_importances_
    }
)


feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print()

print("=" * 70)
print("XGBOOST V2 FEATURE IMPORTANCE")
print("=" * 70)

print()

print(
    feature_importance.to_string(
        index=False
    )
)
# --------------------------------------------------
# SAVE PRODUCTION MODEL AND PREPROCESSING ARTIFACTS
# --------------------------------------------------

print()

print("=" * 70)
print("SAVING PRODUCTION MODEL")
print("=" * 70)

# Create models directory if it does not exist
os.makedirs(
    "models",
    exist_ok=True
)

# --------------------------------------------------
# SAVE XGBOOST MODEL
# --------------------------------------------------

joblib.dump(
    xgb_model,
    "models/readmission_xgboost_v2.joblib"
)

print()
print("Model saved:")
print("models/readmission_xgboost_v2.joblib")


# --------------------------------------------------
# SAVE FEATURE COLUMN ORDER
# --------------------------------------------------

feature_columns = X_train_encoded.columns.tolist()

with open(
    "models/readmission_feature_columns_v2.json",
    "w"
) as file:

    json.dump(
        feature_columns,
        file,
        indent=4
    )

print()
print("Feature columns saved:")
print("models/readmission_feature_columns_v2.json")


# --------------------------------------------------
# SAVE CLINICAL MEDIANS
# --------------------------------------------------

clinical_medians = {}

for column in v2_features:

    clinical_medians[column] = float(
        X.iloc[train_index][column].median()
    )


with open(
    "models/readmission_clinical_medians_v2.json",
    "w"
) as file:

    json.dump(
        clinical_medians,
        file,
        indent=4
    )

print()
print("Clinical medians saved:")
print("models/readmission_clinical_medians_v2.json")


# --------------------------------------------------
# SAVE PRODUCTION THRESHOLD
# --------------------------------------------------

production_threshold = 0.50

with open(
    "models/readmission_threshold_v2.json",
    "w"
) as file:

    json.dump(
        {
            "threshold": production_threshold
        },
        file,
        indent=4
    )

print()
print("Production threshold saved:")
print("models/readmission_threshold_v2.json")


print()
print("=" * 70)
print("PRODUCTION ARTIFACTS SAVED SUCCESSFULLY")
print("=" * 70)