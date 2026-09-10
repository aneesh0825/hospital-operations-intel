import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

V1_PATH = "data/processed/synthea_readmission_ml_dataset_v1.csv"
OBSERVATIONS_PATH = "data/processed/synthea_filtered_observations.csv"
ENCOUNTERS_PATH = "../synthea/output/csv/encounters.csv"

OUTPUT_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v2.csv"
)

LOOKBACK_DAYS = 365


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

print("Loading V1 dataset...")
v1 = pd.read_csv(V1_PATH)

print("Loading filtered observations...")
observations = pd.read_csv(
    OBSERVATIONS_PATH,
    low_memory=False
)

print("Loading encounter dates...")
encounters = pd.read_csv(
    ENCOUNTERS_PATH,
    usecols=["Id", "START"]
)


# --------------------------------------------------
# PREPARE HOSPITALIZATIONS
# --------------------------------------------------

encounters["START"] = pd.to_datetime(
    encounters["START"],
    utc=True,
    errors="coerce"
)

encounters = encounters.rename(
    columns={
        "Id": "encounter_id",
        "START": "admission_start"
    }
)

hospitalizations = v1[
    ["encounter_id", "patient_id"]
].merge(
    encounters,
    on="encounter_id",
    how="left"
)

hospitalizations = hospitalizations.dropna(
    subset=["admission_start"]
).copy()


print()
print("=" * 60)
print("HOSPITALIZATIONS")
print("=" * 60)
print("Shape:")
print(hospitalizations.shape)


# --------------------------------------------------
# PREPARE OBSERVATIONS
# --------------------------------------------------

observations["DATE"] = pd.to_datetime(
    observations["DATE"],
    utc=True,
    errors="coerce"
)

observations["VALUE"] = pd.to_numeric(
    observations["VALUE"],
    errors="coerce"
)

observations = observations.dropna(
    subset=[
        "DATE",
        "VALUE",
        "PATIENT",
        "feature_name"
    ]
).copy()


# --------------------------------------------------
# MATCH BY PATIENT
# --------------------------------------------------

matched = hospitalizations.merge(
    observations[
        [
            "PATIENT",
            "DATE",
            "VALUE",
            "feature_name"
        ]
    ],
    left_on="patient_id",
    right_on="PATIENT",
    how="left"
)


print()
print("=" * 60)
print("PATIENT-MATCHED OBSERVATIONS")
print("=" * 60)
print("Shape:")
print(matched.shape)


# --------------------------------------------------
# CALCULATE TIME BEFORE ADMISSION
# --------------------------------------------------

matched["days_before_admission"] = (
    matched["admission_start"]
    - matched["DATE"]
).dt.total_seconds() / 86400


# Keep measurements from the 365 days BEFORE admission.
#
# >= 0 prevents future information from entering the model.
# <= LOOKBACK_DAYS prevents extremely old measurements from
# being treated as the patient's current clinical state.

matched = matched[
    (matched["days_before_admission"] >= 0)
    &
    (matched["days_before_admission"] <= LOOKBACK_DAYS)
].copy()


print()
print("=" * 60)
print(f"OBSERVATIONS WITHIN {LOOKBACK_DAYS}-DAY LOOKBACK")
print("=" * 60)

print("Shape:")
print(matched.shape)

print()

print("Observation counts:")
print(
    matched["feature_name"]
    .value_counts()
)


# --------------------------------------------------
# SELECT MOST RECENT PRE-ADMISSION VALUE
# --------------------------------------------------

matched = matched.sort_values(
    by=[
        "encounter_id",
        "feature_name",
        "DATE"
    ]
)

latest = (
    matched
    .groupby(
        [
            "encounter_id",
            "feature_name"
        ],
        as_index=False
    )
    .tail(1)
)


# --------------------------------------------------
# CONVERT FEATURES TO WIDE FORMAT
# --------------------------------------------------

features = (
    latest
    .pivot(
        index="encounter_id",
        columns="feature_name",
        values="VALUE"
    )
    .reset_index()
)

features.columns.name = None


print()
print("=" * 60)
print("PRE-ADMISSION OBSERVATION FEATURES")
print("=" * 60)

print("Shape:")
print(features.shape)

print()

print("Columns:")
print(features.columns.tolist())


# --------------------------------------------------
# MERGE WITH V1
# --------------------------------------------------

v2 = v1.merge(
    features,
    on="encounter_id",
    how="left"
)


observation_columns = [
    column
    for column in features.columns
    if column != "encounter_id"
]


print()
print("=" * 60)
print("V2 DATASET")
print("=" * 60)

print("Shape:")
print(v2.shape)


# --------------------------------------------------
# FEATURE COVERAGE
# --------------------------------------------------

print()
print("=" * 60)
print("PRE-ADMISSION FEATURE COVERAGE")
print("=" * 60)

for column in observation_columns:

    available = v2[column].notna().sum()

    percentage = (
        available / len(v2)
    ) * 100

    print(
        f"{column}: "
        f"{available} "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

print()
print("=" * 60)
print("V2 MISSING VALUES")
print("=" * 60)

print(
    v2[
        observation_columns
    ].isna().sum()
)


# --------------------------------------------------
# SAVE V2
# --------------------------------------------------

v2.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print("=" * 60)
print("V2 DATASET SAVED")
print("=" * 60)

print(OUTPUT_PATH)