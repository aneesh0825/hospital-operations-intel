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
    "data/processed/synthea_readmission_ml_dataset_v3.csv"
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
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ADMITRA HISTORY-AWARE MODEL TRAINING")
print("=" * 70)

print("\nRows:")
print(f"{len(df):,}")

print("\nFeatures:")
print(len(FEATURES))

print("\nReadmission rate:")
print(
    round(
        df[TARGET].mean() * 100,
        2,
    ),
    "%",
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

groups = df["patient_id"].astype(str)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

medians = {}

for feature in FEATURES:

    median = float(
        X[feature].median()
    )

    medians[feature] = median

    X[feature] = (
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


X_train = X.iloc[train_idx].copy()
X_test = X.iloc[test_idx].copy()

y_train = y.iloc[train_idx].copy()
y_test = y.iloc[test_idx].copy()


train_patients = set(
    groups.iloc[train_idx]
)

test_patients = set(
    groups.iloc[test_idx]
)


print("\n" + "=" * 70)
print("PATIENT-LEVEL SPLIT")
print("=" * 70)

print("\nTraining rows:")
print(len(X_train))

print("\nTesting rows:")
print(len(X_test))

print("\nPatient overlap:")
print(
    len(
        train_patients
        & test_patients
    )
)


# ============================================================
# CLASS IMBALANCE
# ============================================================

negative = int(
    (y_train == 0).sum()
)

positive = int(
    (y_train == 1).sum()
)

scale_pos_weight = (
    negative / positive
)


print("\nScale pos weight:")
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


model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATE
# ============================================================

probabilities = (
    model.predict_proba(
        X_test
    )[:, 1]
)


roc_auc = roc_auc_score(
    y_test,
    probabilities,
)

pr_auc = average_precision_score(
    y_test,
    probabilities,
)


print("\n" + "=" * 70)
print("HELD-OUT TEST PERFORMANCE")
print("=" * 70)

print("\nROC-AUC:")
print(
    round(
        roc_auc,
        3,
    )
)

print("\nPR-AUC:")
print(
    round(
        pr_auc,
        3,
    )
)

print("\nActual test readmission rate:")
print(
    round(
        y_test.mean() * 100,
        2,
    ),
    "%",
)

print("\nMean raw predicted probability:")
print(
    round(
        probabilities.mean() * 100,
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
).sort_values(
    "importance",
    ascending=False,
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_DIR
    / "readmission_history_xgboost.joblib",
)


importance.to_csv(
    MODEL_DIR
    / "readmission_history_feature_importance.csv",
    index=False,
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    "models/readmission_history_xgboost.joblib"
)

print(
    "models/readmission_history_feature_importance.csv"
)