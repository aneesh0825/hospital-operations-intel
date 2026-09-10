import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

OBSERVATIONS_PATH = Path(
    "../synthea/output/csv/observations.csv"
)

OUTPUT_PATH = Path(
    "data/processed/"
    "synthea_filtered_observations_large.csv"
)


# ============================================================
# OBSERVATION FEATURES WE WANT
# ============================================================

# Synthea LOINC codes
FEATURE_CODES = {
    "8302-2": "height",
    "29463-7": "weight",
    "39156-5": "bmi",

    "8480-6": "systolic_bp",
    "8462-4": "diastolic_bp",

    "8867-4": "heart_rate",
    "9279-1": "respiratory_rate",

    "8310-5": "temperature",

    # Common glucose observation codes
    "2339-0": "glucose",
    "2345-7": "glucose",
}


# ============================================================
# CHUNK SETTINGS
# ============================================================

CHUNK_SIZE = 500_000

USECOLS = [
    "DATE",
    "PATIENT",
    "ENCOUNTER",
    "CODE",
    "DESCRIPTION",
    "VALUE",
    "UNITS",
    "TYPE",
]


# ============================================================
# PROCESS FILE
# ============================================================

print("=" * 70)
print("EXTRACTING LARGE-COHORT OBSERVATION FEATURES")
print("=" * 70)

print()
print("Source:")
print(OBSERVATIONS_PATH)

print()
print("Chunk size:")
print(f"{CHUNK_SIZE:,}")


filtered_chunks = []

rows_read = 0
rows_kept = 0


reader = pd.read_csv(
    OBSERVATIONS_PATH,
    usecols=USECOLS,
    chunksize=CHUNK_SIZE,
    low_memory=False,
)


for chunk_number, chunk in enumerate(
    reader,
    start=1,
):

    rows_read += len(chunk)

    # --------------------------------------------------------
    # KEEP ONLY NUMERIC OBSERVATIONS
    # --------------------------------------------------------

    chunk = chunk[
        chunk["TYPE"] == "numeric"
    ].copy()


    # --------------------------------------------------------
    # KEEP ONLY FEATURES WE NEED
    # --------------------------------------------------------

    chunk = chunk[
        chunk["CODE"]
        .astype(str)
        .isin(
            FEATURE_CODES.keys()
        )
    ].copy()


    if chunk.empty:

        print(
            f"Chunk {chunk_number}: "
            f"read {rows_read:,} total rows | "
            f"kept {rows_kept:,}"
        )

        continue


    # --------------------------------------------------------
    # MAP CODES TO MODEL FEATURE NAMES
    # --------------------------------------------------------

    chunk[
        "feature_name"
    ] = (
        chunk["CODE"]
        .astype(str)
        .map(
            FEATURE_CODES
        )
    )


    # --------------------------------------------------------
    # CLEAN VALUES
    # --------------------------------------------------------

    chunk["VALUE"] = pd.to_numeric(
        chunk["VALUE"],
        errors="coerce",
    )

    chunk["DATE"] = pd.to_datetime(
        chunk["DATE"],
        utc=True,
        errors="coerce",
    )


    chunk = chunk.dropna(
        subset=[
            "DATE",
            "PATIENT",
            "VALUE",
            "feature_name",
        ]
    ).copy()


    # --------------------------------------------------------
    # KEEP ONLY COLUMNS NEEDED BY V2 BUILDER
    # --------------------------------------------------------

    chunk = chunk[
        [
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
            "VALUE",
            "UNITS",
            "feature_name",
        ]
    ]


    filtered_chunks.append(
        chunk
    )

    rows_kept += len(chunk)


    print(
        f"Chunk {chunk_number}: "
        f"read {rows_read:,} total rows | "
        f"kept {rows_kept:,}"
    )


# ============================================================
# COMBINE RESULTS
# ============================================================

print()
print("=" * 70)
print("COMBINING FILTERED OBSERVATIONS")
print("=" * 70)


if filtered_chunks:

    filtered = pd.concat(
        filtered_chunks,
        ignore_index=True,
    )

else:

    filtered = pd.DataFrame(
        columns=[
            "DATE",
            "PATIENT",
            "ENCOUNTER",
            "CODE",
            "DESCRIPTION",
            "VALUE",
            "UNITS",
            "feature_name",
        ]
    )


print()
print("Rows read:")
print(f"{rows_read:,}")

print()
print("Rows retained:")
print(f"{len(filtered):,}")

print()
print("Feature counts:")
print(
    filtered[
        "feature_name"
    ].value_counts()
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

filtered.to_csv(
    OUTPUT_PATH,
    index=False,
)


print()
print("=" * 70)
print("FILTERED OBSERVATIONS SAVED")
print("=" * 70)

print()
print(OUTPUT_PATH)