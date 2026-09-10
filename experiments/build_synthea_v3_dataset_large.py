from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DATA_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_large_v2.csv"
)

ENCOUNTERS_PATH = Path(
    "../synthea/output/csv/encounters.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "synthea_readmission_ml_dataset_large_v3.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING ADMITRA LARGE TEMPORAL HISTORY DATASET")
print("=" * 70)

base = pd.read_csv(
    BASE_DATA_PATH
)

encounters = pd.read_csv(
    ENCOUNTERS_PATH,
    usecols=[
        "Id",
        "PATIENT",
        "START",
        "STOP",
        "ENCOUNTERCLASS",
    ],
)

print("\nBase ML dataset:")
print(base.shape)

print("\nRaw Synthea encounters:")
print(encounters.shape)


# ============================================================
# PREPARE ENCOUNTERS
# ============================================================

encounters["START"] = pd.to_datetime(
    encounters["START"],
    utc=True,
    errors="coerce",
)

encounters["STOP"] = pd.to_datetime(
    encounters["STOP"],
    utc=True,
    errors="coerce",
)

encounters = encounters.rename(
    columns={
        "Id": "encounter_id",
        "PATIENT": "patient_id",
        "START": "encounter_start",
        "STOP": "encounter_stop",
        "ENCOUNTERCLASS": "encounter_class",
    }
)

encounters = encounters.dropna(
    subset=[
        "encounter_id",
        "patient_id",
        "encounter_start",
    ]
).copy()


# ============================================================
# IDENTIFY INDEX HOSPITALIZATIONS
# ============================================================

index_encounters = base[
    [
        "encounter_id",
        "patient_id",
        "readmitted_30_days",
    ]
].merge(
    encounters[
        [
            "encounter_id",
            "encounter_start",
            "encounter_stop",
            "encounter_class",
        ]
    ],
    on="encounter_id",
    how="left",
)

missing_dates = (
    index_encounters[
        "encounter_start"
    ].isna().sum()
)

print("\nIndex encounters missing dates:")
print(missing_dates)

index_encounters = (
    index_encounters
    .dropna(
        subset=[
            "encounter_start"
        ]
    )
    .copy()
)


# ============================================================
# SORT ALL ENCOUNTERS BY PATIENT + TIME
# ============================================================

encounters = encounters.sort_values(
    [
        "patient_id",
        "encounter_start",
    ]
).reset_index(
    drop=True
)

index_encounters = (
    index_encounters
    .sort_values(
        [
            "patient_id",
            "encounter_start",
        ]
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# OUTPUT STORAGE
# ============================================================

history_records = []


# ============================================================
# BUILD TEMPORAL FEATURES
# ============================================================

print(
    "\nBuilding temporal patient-history features..."
)


for counter, row in index_encounters.iterrows():

    patient_id = row[
        "patient_id"
    ]

    current_encounter_id = row[
        "encounter_id"
    ]

    current_start = row[
        "encounter_start"
    ]


    # --------------------------------------------------------
    # ONLY INFORMATION BEFORE CURRENT ENCOUNTER
    # --------------------------------------------------------

    patient_history = encounters[
        (
            encounters[
                "patient_id"
            ]
            == patient_id
        )
        &
        (
            encounters[
                "encounter_start"
            ]
            < current_start
        )
    ].copy()


    # --------------------------------------------------------
    # NO PRIOR HISTORY
    # --------------------------------------------------------

    if patient_history.empty:

        history_records.append(
            {
                "encounter_id":
                    current_encounter_id,

                "admissions_last_30_days":
                    0,

                "admissions_last_90_days":
                    0,

                "admissions_last_365_days":
                    0,

                "ed_visits_last_90_days":
                    0,

                "ed_visits_last_365_days":
                    0,

                "encounters_last_90_days":
                    0,

                "encounters_last_365_days":
                    0,

                "days_since_last_inpatient_admission":
                    np.nan,

                "prior_readmissions_last_365_days":
                    0,
            }
        )

        continue


    # --------------------------------------------------------
    # DAYS BEFORE CURRENT ADMISSION
    # --------------------------------------------------------

    patient_history[
        "days_before"
    ] = (
        current_start
        - patient_history[
            "encounter_start"
        ]
    ).dt.total_seconds() / 86400


    # --------------------------------------------------------
    # TIME WINDOWS
    # --------------------------------------------------------

    last_30 = patient_history[
        (
            patient_history[
                "days_before"
            ] >= 0
        )
        &
        (
            patient_history[
                "days_before"
            ] <= 30
        )
    ]

    last_90 = patient_history[
        (
            patient_history[
                "days_before"
            ] >= 0
        )
        &
        (
            patient_history[
                "days_before"
            ] <= 90
        )
    ]

    last_365 = patient_history[
        (
            patient_history[
                "days_before"
            ] >= 0
        )
        &
        (
            patient_history[
                "days_before"
            ] <= 365
        )
    ]


    # --------------------------------------------------------
    # INPATIENT HISTORY
    # --------------------------------------------------------

    inpatient_30 = last_30[
        last_30[
            "encounter_class"
        ].str.lower()
        == "inpatient"
    ]

    inpatient_90 = last_90[
        last_90[
            "encounter_class"
        ].str.lower()
        == "inpatient"
    ]

    inpatient_365 = last_365[
        last_365[
            "encounter_class"
        ].str.lower()
        == "inpatient"
    ]


    # --------------------------------------------------------
    # EMERGENCY HISTORY
    # --------------------------------------------------------

    emergency_90 = last_90[
        last_90[
            "encounter_class"
        ].str.lower()
        == "emergency"
    ]

    emergency_365 = last_365[
        last_365[
            "encounter_class"
        ].str.lower()
        == "emergency"
    ]


    # --------------------------------------------------------
    # DAYS SINCE LAST INPATIENT ADMISSION
    # --------------------------------------------------------

    previous_inpatient = patient_history[
        patient_history[
            "encounter_class"
        ].str.lower()
        == "inpatient"
    ]

    if previous_inpatient.empty:

        days_since_last_inpatient = np.nan

    else:

        last_inpatient_start = (
            previous_inpatient[
                "encounter_start"
            ].max()
        )

        days_since_last_inpatient = (
            current_start
            - last_inpatient_start
        ).total_seconds() / 86400


    # --------------------------------------------------------
    # PRIOR READMISSIONS WITHIN PREVIOUS 365 DAYS
    # --------------------------------------------------------
    #
    # LEAKAGE PROTECTION:
    #
    # A previous hospitalization contributes its
    # readmitted_30_days outcome only when its entire
    # 30-day outcome window had already finished before
    # the current hospitalization began.
    # --------------------------------------------------------

    prior_index_rows = index_encounters[
        (
            index_encounters[
                "patient_id"
            ]
            == patient_id
        )
        &
        (
            index_encounters[
                "encounter_start"
            ]
            < current_start
        )
        &
        (
            index_encounters[
                "encounter_start"
            ]
            >= (
                current_start
                - pd.Timedelta(
                    days=365
                )
            )
        )
        &
        (
            (
                index_encounters[
                    "encounter_start"
                ]
                + pd.Timedelta(
                    days=30
                )
            )
            <= current_start
        )
    ]


    prior_readmissions = int(
        prior_index_rows[
            "readmitted_30_days"
        ]
        .fillna(0)
        .sum()
    )


    # --------------------------------------------------------
    # SAVE HISTORY RECORD
    # --------------------------------------------------------

    history_records.append(
        {
            "encounter_id":
                current_encounter_id,

            "admissions_last_30_days":
                len(
                    inpatient_30
                ),

            "admissions_last_90_days":
                len(
                    inpatient_90
                ),

            "admissions_last_365_days":
                len(
                    inpatient_365
                ),

            "ed_visits_last_90_days":
                len(
                    emergency_90
                ),

            "ed_visits_last_365_days":
                len(
                    emergency_365
                ),

            "encounters_last_90_days":
                len(
                    last_90
                ),

            "encounters_last_365_days":
                len(
                    last_365
                ),

            "days_since_last_inpatient_admission":
                days_since_last_inpatient,

            "prior_readmissions_last_365_days":
                prior_readmissions,
        }
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        counter + 1
    ) % 1000 == 0:

        print(
            f"Processed "
            f"{counter + 1:,} "
            f"hospitalizations"
        )


# ============================================================
# CREATE HISTORY DATAFRAME
# ============================================================

history = pd.DataFrame(
    history_records
)


print("\n" + "=" * 70)
print("RAW TEMPORAL HISTORY FEATURES")
print("=" * 70)

print(
    history.describe().round(2)
)


# ============================================================
# MERGE HISTORY INTO ML DATASET
# ============================================================

v3 = base.merge(
    history,
    on="encounter_id",
    how="left",
)


# ============================================================
# CLEAN DAYS-SINCE HISTORY
# ============================================================

v3[
    "has_prior_inpatient_admission"
] = (
    v3[
        "days_since_last_inpatient_admission"
    ]
    .notna()
    .astype(int)
)


v3[
    "days_since_last_inpatient_admission"
] = (
    v3[
        "days_since_last_inpatient_admission"
    ]
    .clip(
        lower=0,
        upper=365,
    )
    .fillna(365)
)


# ============================================================
# TEMPORAL FEATURE LIST
# ============================================================

temporal_features = [
    "admissions_last_30_days",
    "admissions_last_90_days",
    "admissions_last_365_days",
    "ed_visits_last_90_days",
    "ed_visits_last_365_days",
    "encounters_last_90_days",
    "encounters_last_365_days",
    "days_since_last_inpatient_admission",
    "has_prior_inpatient_admission",
    "prior_readmissions_last_365_days",
]


# ============================================================
# FINAL SANITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL LARGE V3 DATASET")
print("=" * 70)

print("\nShape:")
print(
    v3.shape
)

print(
    "\nTemporal feature missing values:"
)

print(
    v3[
        temporal_features
    ]
    .isna()
    .sum()
)


# ============================================================
# TEMPORAL FEATURE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANED TEMPORAL HISTORY FEATURES")
print("=" * 70)

print(
    v3[
        temporal_features
    ]
    .describe()
    .round(2)
)


# ============================================================
# CORRELATION WITH READMISSION
# ============================================================

print("\n" + "=" * 70)
print(
    "TEMPORAL FEATURE CORRELATIONS WITH READMISSION"
)
print("=" * 70)

correlations = (
    v3[
        temporal_features
        + [
            "readmitted_30_days"
        ]
    ]
    .corr(
        numeric_only=True
    )[
        "readmitted_30_days"
    ]
    .drop(
        "readmitted_30_days"
    )
    .sort_values(
        ascending=False
    )
)

print(
    correlations
)


# ============================================================
# READMISSION RATE BY 90-DAY ADMISSION HISTORY
# ============================================================

print("\n" + "=" * 70)
print(
    "READMISSION RATE BY 90-DAY ADMISSION HISTORY"
)
print("=" * 70)

admission_signal = (
    v3
    .groupby(
        "admissions_last_90_days"
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

admission_signal[
    "readmission_rate_percent"
] = (
    admission_signal[
        "mean"
    ]
    * 100
).round(2)

print(
    admission_signal[
        [
            "count",
            "readmission_rate_percent",
        ]
    ]
)


# ============================================================
# READMISSION RATE BY PRIOR READMISSIONS
# ============================================================

print("\n" + "=" * 70)
print(
    "READMISSION RATE BY PRIOR 365-DAY READMISSIONS"
)
print("=" * 70)

readmission_signal = (
    v3
    .groupby(
        "prior_readmissions_last_365_days"
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

readmission_signal[
    "readmission_rate_percent"
] = (
    readmission_signal[
        "mean"
    ]
    * 100
).round(2)

print(
    readmission_signal[
        [
            "count",
            "readmission_rate_percent",
        ]
    ]
)


# ============================================================
# ADDITIONAL LEAKAGE SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("PRIOR READMISSION FEATURE SANITY CHECK")
print("=" * 70)

print(
    "\nPatients with zero prior readmissions:"
)

print(
    (
        v3[
            "prior_readmissions_last_365_days"
        ]
        == 0
    ).sum()
)

print(
    "\nPatients with at least one prior readmission:"
)

print(
    (
        v3[
            "prior_readmissions_last_365_days"
        ]
        > 0
    ).sum()
)

print(
    "\nMaximum prior readmissions:"
)

print(
    v3[
        "prior_readmissions_last_365_days"
    ].max()
)


# ============================================================
# SAVE LARGE V3
# ============================================================

v3.to_csv(
    OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("LARGE V3 DATASET SAVED")
print("=" * 70)

print(
    OUTPUT_PATH
)