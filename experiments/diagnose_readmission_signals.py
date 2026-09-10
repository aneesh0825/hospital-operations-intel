import pandas as pd


DATA_PATH = "data/processed/synthea_readmission_ml_dataset_v2.csv"

df = pd.read_csv(DATA_PATH)


# ============================================================
# BASIC DATASET INFO
# ============================================================

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

target = "readmitted_30_days"

print("\nReadmission rate:")
print(round(df[target].mean() * 100, 2), "%")


# ============================================================
# HELPER
# ============================================================

def numeric_signal(feature, bins=5):

    print("\n" + "=" * 70)
    print(feature.upper())
    print("=" * 70)

    temp = df[
        [feature, target]
    ].dropna().copy()

    try:

        temp["group"] = pd.qcut(
            temp[feature],
            q=bins,
            duplicates="drop"
        )

    except Exception:

        temp["group"] = pd.cut(
            temp[feature],
            bins=bins
        )

    result = (
        temp
        .groupby(
            "group",
            observed=True
        )[target]
        .agg(
            [
                "count",
                "mean"
            ]
        )
    )

    result["readmission_rate_percent"] = (
        result["mean"]
        * 100
    ).round(2)

    print(
        result[
            [
                "count",
                "readmission_rate_percent"
            ]
        ]
    )


def binary_signal(feature):

    print("\n" + "=" * 70)
    print(feature.upper())
    print("=" * 70)

    result = (
        df
        .groupby(feature)[target]
        .agg(
            [
                "count",
                "mean"
            ]
        )
    )

    result["readmission_rate_percent"] = (
        result["mean"]
        * 100
    ).round(2)

    print(
        result[
            [
                "count",
                "readmission_rate_percent"
            ]
        ]
    )


# ============================================================
# UTILIZATION FEATURES
# ============================================================

numeric_signal(
    "previous_inpatient_admissions"
)

numeric_signal(
    "previous_emergency_visits"
)

numeric_signal(
    "previous_encounters"
)

numeric_signal(
    "length_of_stay"
)


# ============================================================
# COMPLEXITY FEATURES
# ============================================================

numeric_signal(
    "condition_count"
)

numeric_signal(
    "medication_count"
)

numeric_signal(
    "procedure_count"
)


# ============================================================
# DEMOGRAPHIC
# ============================================================

numeric_signal(
    "age_at_admission"
)


# ============================================================
# CHRONIC CONDITIONS
# ============================================================

binary_signal(
    "diabetes"
)

binary_signal(
    "hypertension"
)

binary_signal(
    "heart_failure"
)

binary_signal(
    "kidney_disease"
)

binary_signal(
    "chronic_lung_disease"
)


# ============================================================
# VITALS / LABS
# ============================================================

numeric_signal(
    "bmi"
)

numeric_signal(
    "heart_rate"
)

numeric_signal(
    "respiratory_rate"
)

numeric_signal(
    "systolic_bp"
)

numeric_signal(
    "diastolic_bp"
)

numeric_signal(
    "glucose"
)


# ============================================================
# SIMPLE CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION WITH READMISSION")
print("=" * 70)

numeric_columns = [
    "age_at_admission",
    "length_of_stay",
    "previous_inpatient_admissions",
    "previous_emergency_visits",
    "previous_encounters",
    "condition_count",
    "medication_count",
    "procedure_count",
    "diabetes",
    "hypertension",
    "heart_failure",
    "kidney_disease",
    "chronic_lung_disease",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose",
]

correlations = (
    df[
        numeric_columns
        + [target]
    ]
    .corr(numeric_only=True)[target]
    .drop(target)
    .sort_values(
        ascending=False
    )
)

print(correlations)
