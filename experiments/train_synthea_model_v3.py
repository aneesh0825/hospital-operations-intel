import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ADMITRA V3 MODEL TRAINING")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Features: {len(FEATURES)}")
print(
    f"Readmission rate: "
    f"{df[TARGET].mean() * 100:.2f}%"
)


X = df[FEATURES].copy()
y = df[TARGET].astype(int)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

medians = {}

for feature in FEATURES:

    if X[feature].isna().any():

        median = float(
            X[feature].median()
        )

        X[feature] = (
            X[feature].fillna(median)
        )

        medians[feature] = median


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
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

print(
    f"Scale pos weight: "
    f"{scale_pos_weight:.3f}"
)


# ============================================================
# MODEL
# ============================================================

model = XGBClassifier(
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


model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATION
# ============================================================

probabilities = (
    model.predict_proba(X_test)[:, 1]
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
print("VALIDATION")
print("=" * 70)

print(
    f"ROC-AUC: {roc_auc:.3f}"
)

print(
    f"PR-AUC: {pr_auc:.3f}"
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
    / "readmission_xgboost_v3.joblib",
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


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print(
    "models/readmission_xgboost_v3.joblib"
)

print(
    "models/readmission_feature_columns_v3.json"
)

print(
    "models/readmission_clinical_medians_v3.json"
)