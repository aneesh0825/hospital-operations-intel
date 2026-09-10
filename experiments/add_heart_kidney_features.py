from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_large_v3.csv"
)

CONDITIONS_PATH = Path(
    "../synthea/output/csv/conditions.csv"
)

ENCOUNTERS_PATH = Path(
    "../synthea/output/csv/encounters.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_large_v3_heart_kidney.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING ADMITRA LARGE HEART + KIDNEY FEATURES")
print("=" * 70)

data = pd.read_csv(
    DATA_PATH
)

conditions = pd.read_csv(
    CONDITIONS_PATH,
    low_memory=False,
)

encounters = pd.read_csv(
    ENCOUNTERS_PATH,
    usecols=[
        "Id",
        "START",
    ],
)


print("\nBase dataset:")
print(
    data.shape
)

print("\nConditions:")
print(
    conditions.shape
)


# ============================================================
# PREPARE HOSPITALIZATION DATES
# ============================================================

encounters[
    "START"
] = pd.to_datetime(
    encounters[
        "START"
    ],
    utc=True,
    errors="coerce",
)


encounters = encounters.rename(
    columns={
        "Id":
            "encounter_id",

        "START":
            "admission_start",
    }
)


data = data.merge(
    encounters,
    on="encounter_id",
    how="left",
)


print(
    "\nHospitalizations missing admission date:"
)

print(
    data[
        "admission_start"
    ]
    .isna()
    .sum()
)


# ============================================================
# SAFETY CHECK
# ============================================================

missing_admission_dates = int(
    data[
        "admission_start"
    ]
    .isna()
    .sum()
)

if missing_admission_dates > 0:

    raise RuntimeError(
        f"{missing_admission_dates:,} hospitalizations "
        "could not be matched to the current Synthea "
        "encounters.csv. Stop here to avoid creating "
        "invalid condition features."
    )


# ============================================================
# PREPARE CONDITION DATES
# ============================================================

conditions[
    "START"
] = pd.to_datetime(
    conditions[
        "START"
    ],
    utc=True,
    errors="coerce",
)


conditions = (
    conditions
    .dropna(
        subset=[
            "PATIENT",
            "START",
            "DESCRIPTION",
        ]
    )
    .copy()
)


# ============================================================
# DEFINE NEW CONDITIONS
# ============================================================

HEART_DESCRIPTION = (
    "Ischemic heart disease (disorder)"
)

KIDNEY_FAILURE_DESCRIPTION = (
    "End-stage renal disease (disorder)"
)


# ============================================================
# FILTER SOURCE DIAGNOSES
# ============================================================

heart_conditions = (
    conditions[
        conditions[
            "DESCRIPTION"
        ]
        == HEART_DESCRIPTION
    ][
        [
            "PATIENT",
            "START",
        ]
    ]
    .copy()
)


kidney_failure_conditions = (
    conditions[
        conditions[
            "DESCRIPTION"
        ]
        == KIDNEY_FAILURE_DESCRIPTION
    ][
        [
            "PATIENT",
            "START",
        ]
    ]
    .copy()
)


print("\n" + "=" * 70)
print("SOURCE DIAGNOSES")
print("=" * 70)

print(
    "\nIschemic heart disease records:"
)

print(
    f"{len(heart_conditions):,}"
)

print(
    "\nEnd-stage renal disease records:"
)

print(
    f"{len(kidney_failure_conditions):,}"
)


# ============================================================
# FIRST KNOWN DIAGNOSIS DATE PER PATIENT
# ============================================================

heart_first = (
    heart_conditions
    .groupby(
        "PATIENT",
        as_index=False,
    )[
        "START"
    ]
    .min()
    .rename(
        columns={
            "PATIENT":
                "patient_id",

            "START":
                "first_ischemic_heart_disease_date",
        }
    )
)


kidney_first = (
    kidney_failure_conditions
    .groupby(
        "PATIENT",
        as_index=False,
    )[
        "START"
    ]
    .min()
    .rename(
        columns={
            "PATIENT":
                "patient_id",

            "START":
                "first_kidney_failure_date",
        }
    )
)


# ============================================================
# MERGE FIRST DIAGNOSIS DATES
# ============================================================

data = data.merge(
    heart_first,
    on="patient_id",
    how="left",
)


data = data.merge(
    kidney_first,
    on="patient_id",
    how="left",
)


# ============================================================
# CREATE LEAKAGE-SAFE FEATURES
# ============================================================
#
# A diagnosis is only available to the model if it was already
# documented on or before the index hospitalization.
#
# Future diagnoses are not allowed to influence the prediction.
# ============================================================

data[
    "ischemic_heart_disease"
] = (
    data[
        "first_ischemic_heart_disease_date"
    ]
    .notna()
    &
    (
        data[
            "first_ischemic_heart_disease_date"
        ]
        <= data[
            "admission_start"
        ]
    )
).astype(int)


data[
    "kidney_failure"
] = (
    data[
        "first_kidney_failure_date"
    ]
    .notna()
    &
    (
        data[
            "first_kidney_failure_date"
        ]
        <= data[
            "admission_start"
        ]
    )
).astype(int)


# ============================================================
# NEW FEATURE PREVALENCE
# ============================================================

print("\n" + "=" * 70)
print("NEW FEATURE PREVALENCE")
print("=" * 70)


for feature in [
    "ischemic_heart_disease",
    "kidney_failure",
]:

    present = int(
        data[
            feature
        ]
        .sum()
    )

    percent = (
        present
        / len(data)
        * 100
    )

    print(
        f"\n{feature}:"
    )

    print(
        f"Present: "
        f"{present:,} "
        f"({percent:.2f}%)"
    )


# ============================================================
# READMISSION SIGNAL
# ============================================================

print("\n" + "=" * 70)
print("READMISSION SIGNAL")
print("=" * 70)


for feature in [
    "ischemic_heart_disease",
    "kidney_failure",
]:

    print()

    print(
        feature.upper()
    )

    print(
        "-" * 70
    )


    summary = (
        data
        .groupby(
            feature
        )[
            "readmitted_30_days"
        ]
        .agg(
            [
                "count",
                "mean",
            ]
        )
    )


    summary[
        "readmission_rate_percent"
    ] = (
        summary[
            "mean"
        ]
        * 100
    ).round(2)


    print(
        summary[
            [
                "count",
                "readmission_rate_percent",
            ]
        ]
    )


    correlation = (
        data[
            [
                feature,
                "readmitted_30_days",
            ]
        ]
        .corr()
        .iloc[
            0,
            1
        ]
    )


    print()

    print(
        "Correlation:",
        round(
            correlation,
            4,
        )
    )


# ============================================================
# KIDNEY FEATURE OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("KIDNEY FEATURE OVERLAP")
print("=" * 70)

print()

print(
    pd.crosstab(
        data[
            "kidney_disease"
        ],

        data[
            "kidney_failure"
        ],

        rownames=[
            "existing kidney_disease"
        ],

        colnames=[
            "new kidney_failure"
        ],
    )
)


# ============================================================
# HEART FEATURE OVERLAP
# ============================================================

print("\n" + "=" * 70)
print("HEART FEATURE OVERLAP")
print("=" * 70)

print()

if "heart_failure" in data.columns:

    print(
        pd.crosstab(
            data[
                "heart_failure"
            ],

            data[
                "ischemic_heart_disease"
            ],

            rownames=[
                "existing heart_failure"
            ],

            colnames=[
                "new ischemic_heart_disease"
            ],
        )
    )

else:

    print(
        "heart_failure column not present."
    )


# ============================================================
# QUALITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("QUALITY CHECK")
print("=" * 70)

print()

print(
    "Duplicate encounter IDs:"
)

print(
    data[
        "encounter_id"
    ]
    .duplicated()
    .sum()
)


print()

print(
    "Missing ischemic_heart_disease:"
)

print(
    data[
        "ischemic_heart_disease"
    ]
    .isna()
    .sum()
)


print()

print(
    "Missing kidney_failure:"
)

print(
    data[
        "kidney_failure"
    ]
    .isna()
    .sum()
)


# ============================================================
# CLEAN TEMPORARY COLUMNS
# ============================================================

data = data.drop(
    columns=[
        "admission_start",
        "first_ischemic_heart_disease_date",
        "first_kidney_failure_date",
    ]
)


# ============================================================
# FINAL DATASET
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print()

print("Shape:")
print(
    data.shape
)

print()

print("New columns:")

print(
    [
        "ischemic_heart_disease",
        "kidney_failure",
    ]
)


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("SAVED")
print("=" * 70)

print()

print(
    OUTPUT_PATH
)