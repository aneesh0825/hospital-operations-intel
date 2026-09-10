import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
)

from xgboost import XGBClassifier


# ==================================================
# LOAD DATA
# ==================================================

# Load the finished machine learning dataset
data = pd.read_csv("data/processed/readmission_ml_dataset.csv")

print("DATASET SHAPE")
print(data.shape)

print()

print("COLUMNS")
print(data.columns.tolist())

print()

print("TARGET DISTRIBUTION")
print(data["readmitted_30_days"].value_counts())

print()

print("TARGET PERCENTAGES")
print(data["readmitted_30_days"].value_counts(normalize=True) * 100)


# ==================================================
# CREATE FEATURES AND TARGET
# ==================================================

# X contains the information the models will use
X = data.drop(
    columns=[
        "subject_id",
        "hadm_id",
        "readmitted_30_days"
    ]
)

# y contains the value we want to predict
y = data["readmitted_30_days"]

print()

print("=" * 50)
print("MODEL FEATURES")
print("=" * 50)

print(X.columns.tolist())

print()

print("X SHAPE")
print(X.shape)

print()

print("Y SHAPE")
print(y.shape)

print()

print("FIRST FIVE TARGET VALUES")
print(y.head())


# ==================================================
# TRAIN / TEST SPLIT
# ==================================================

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()

print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print("Training features shape:")
print(X_train.shape)

print()

print("Testing features shape:")
print(X_test.shape)

print()

print("Training target distribution:")
print(y_train.value_counts())

print()

print("Testing target distribution:")
print(y_test.value_counts())


# ==================================================
# ENCODE CATEGORICAL FEATURES
# ==================================================

# Convert categorical columns into numerical columns
X_train_encoded = pd.get_dummies(
    X_train,
    columns=[
        "gender",
        "insurance",
        "admission_type"
    ],
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=[
        "gender",
        "insurance",
        "admission_type"
    ],
    dtype=int
)

# Make sure training and testing sets have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)

print()

print("=" * 50)
print("ENCODED FEATURES")
print("=" * 50)

print("Training shape:")
print(X_train_encoded.shape)

print()

print("Testing shape:")
print(X_test_encoded.shape)

print()

print("Encoded columns:")
print(X_train_encoded.columns.tolist())

print()

print("First five training rows:")
print(X_train_encoded.head())


# ==================================================
# LOGISTIC REGRESSION BASELINE
# ==================================================

# Create baseline model
baseline_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

# Train the model
baseline_model.fit(
    X_train_encoded,
    y_train
)

# Predict classes on unseen test data
y_pred = baseline_model.predict(
    X_test_encoded
)

# Predict readmission probabilities
y_prob = baseline_model.predict_proba(
    X_test_encoded
)[:, 1]


# ==================================================
# EVALUATE LOGISTIC REGRESSION
# ==================================================

print()

print("=" * 50)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 50)

print("Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))

print()

print("Precision:")
print(round(precision_score(y_test, y_pred), 3))

print()

print("Recall:")
print(round(recall_score(y_test, y_pred), 3))

print()

print("F1 Score:")
print(round(f1_score(y_test, y_pred), 3))

print()

print("ROC-AUC:")
print(round(roc_auc_score(y_test, y_prob), 3))

print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# XGBOOST MODEL
# ==================================================

# Calculate the class imbalance ratio
negative_count = y_train.value_counts()[0]
positive_count = y_train.value_counts()[1]

scale_pos_weight = negative_count / positive_count

# Create XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

# Train XGBoost
xgb_model.fit(
    X_train_encoded,
    y_train
)

# Predict classes
xgb_pred = xgb_model.predict(
    X_test_encoded
)

# Predict readmission probabilities
xgb_prob = xgb_model.predict_proba(
    X_test_encoded
)[:, 1]


# ==================================================
# EVALUATE XGBOOST
# ==================================================

print()

print("=" * 50)
print("XGBOOST MODEL")
print("=" * 50)

print("Accuracy:")
print(round(accuracy_score(y_test, xgb_pred), 3))

print()

print("Precision:")
print(round(precision_score(y_test, xgb_pred), 3))

print()

print("Recall:")
print(round(recall_score(y_test, xgb_pred), 3))

print()

print("F1 Score:")
print(round(f1_score(y_test, xgb_pred), 3))

print()

print("ROC-AUC:")
print(round(roc_auc_score(y_test, xgb_prob), 3))

print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, xgb_pred))   

# ==================================================
# 5-FOLD CROSS-VALIDATION
# ==================================================

# Identify categorical and numerical features
categorical_features = [
    "gender",
    "insurance",
    "admission_type"
]

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]


# Create preprocessing step
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# Create logistic regression pipeline
logistic_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)


# Calculate class imbalance using the full target
negative_count_cv = y.value_counts()[0]
positive_count_cv = y.value_counts()[1]

scale_pos_weight_cv = negative_count_cv / positive_count_cv


# Create XGBoost pipeline
xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight_cv,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)


# Create stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Metrics to evaluate
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}


# Run cross-validation for logistic regression
logistic_cv_results = cross_validate(
    logistic_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring
)


# Run cross-validation for XGBoost
xgb_cv_results = cross_validate(
    xgb_pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring
)


# ==================================================
# PRINT CROSS-VALIDATION RESULTS
# ==================================================

print()
print("=" * 50)
print("5-FOLD CROSS-VALIDATION RESULTS")
print("=" * 50)

print()
print("LOGISTIC REGRESSION")

print("Average Accuracy:")
print(round(logistic_cv_results["test_accuracy"].mean(), 3))

print()

print("Average Precision:")
print(round(logistic_cv_results["test_precision"].mean(), 3))

print()

print("Average Recall:")
print(round(logistic_cv_results["test_recall"].mean(), 3))

print()

print("Average F1 Score:")
print(round(logistic_cv_results["test_f1"].mean(), 3))

print()

print("Average ROC-AUC:")
print(round(logistic_cv_results["test_roc_auc"].mean(), 3))


print()
print("-" * 50)
print("XGBOOST")
print("-" * 50)

print("Average Accuracy:")
print(round(xgb_cv_results["test_accuracy"].mean(), 3))

print()

print("Average Precision:")
print(round(xgb_cv_results["test_precision"].mean(), 3))

print()

print("Average Recall:")
print(round(xgb_cv_results["test_recall"].mean(), 3))

print()

print("Average F1 Score:")
print(round(xgb_cv_results["test_f1"].mean(), 3))

print()

print("Average ROC-AUC:")
print(round(xgb_cv_results["test_roc_auc"].mean(), 3))