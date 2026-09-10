import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

observations_path = (
    "../synthea/output/csv/observations.csv"
)

ml_dataset_path = (
    "data/processed/synthea_readmission_ml_dataset_v1.csv"
)


# --------------------------------------------------
# LOAD V1 DATASET
# --------------------------------------------------

ml_data = pd.read_csv(
    ml_dataset_path
)

print("=" * 60)
print("LOADED V1 ML DATASET")
print("=" * 60)

print("Shape:")
print(ml_data.shape)

print()

print("Columns:")
print(ml_data.columns.tolist())


# --------------------------------------------------
# DEFINE OBSERVATIONS WE CARE ABOUT
# --------------------------------------------------

observation_map = {
    "8302-2": "height",
    "29463-7": "weight",
    "39156-5": "bmi",
    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",
    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",
    "8310-5": "temperature",
    "59408-5": "oxygen_saturation",
    "2339-0": "glucose"
}


target_codes = set(
    observation_map.keys()
)


print()

print("=" * 60)
print("TARGET OBSERVATION CODES")
print("=" * 60)

for code, name in observation_map.items():
    print(code, "->", name)


# --------------------------------------------------
# READ OBSERVATIONS IN CHUNKS
# --------------------------------------------------

chunk_size = 250000

selected_chunks = []

chunk_number = 0


print()

print("=" * 60)
print("READING OBSERVATIONS IN CHUNKS")
print("=" * 60)


for chunk in pd.read_csv(
    observations_path,
    usecols=[
        "DATE",
        "PATIENT",
        "ENCOUNTER",
        "CODE",
        "DESCRIPTION",
        "VALUE",
        "UNITS",
        "TYPE"
    ],
    chunksize=chunk_size,
    low_memory=False
):

    chunk_number += 1

    print(
        "Processing chunk:",
        chunk_number
    )


    # Keep only the observation codes we care about
    filtered = chunk[
        chunk["CODE"]
        .astype(str)
        .isin(target_codes)
    ].copy()


    if len(filtered) > 0:

        selected_chunks.append(
            filtered
        )


# --------------------------------------------------
# COMBINE FILTERED OBSERVATIONS
# --------------------------------------------------

if len(selected_chunks) == 0:

    raise ValueError(
        "No matching observation codes were found."
    )


selected_observations = pd.concat(
    selected_chunks,
    ignore_index=True
)


print()

print("=" * 60)
print("FILTERED OBSERVATIONS")
print("=" * 60)

print("Shape:")
print(selected_observations.shape)

print()

print("Observation counts by code:")
print(
    selected_observations[
        "CODE"
    ].astype(str).value_counts()
)


# --------------------------------------------------
# CLEAN VALUES
# --------------------------------------------------

selected_observations["VALUE"] = pd.to_numeric(
    selected_observations["VALUE"],
    errors="coerce"
)

selected_observations["DATE"] = pd.to_datetime(
    selected_observations["DATE"],
    utc=True,
    errors="coerce"
)


selected_observations = (
    selected_observations
    .dropna(
        subset=[
            "VALUE",
            "DATE",
            "PATIENT"
        ]
    )
    .copy()
)


selected_observations["feature_name"] = (
    selected_observations[
        "CODE"
    ]
    .astype(str)
    .map(
        observation_map
    )
)


print()

print("=" * 60)
print("CLEAN OBSERVATIONS")
print("=" * 60)

print("Shape:")
print(selected_observations.shape)

print()

print("Feature distribution:")
print(
    selected_observations[
        "feature_name"
    ].value_counts()
)


# --------------------------------------------------
# SAVE FILTERED OBSERVATIONS
# --------------------------------------------------

output_path = (
    "data/processed/"
    "synthea_filtered_observations.csv"
)


selected_observations.to_csv(
    output_path,
    index=False
)


print()

print("=" * 60)
print("FILTERED OBSERVATIONS SAVED")
print("=" * 60)

print(output_path)