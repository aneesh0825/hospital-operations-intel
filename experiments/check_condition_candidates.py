import pandas as pd


DATA_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v3.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(DATA_PATH)

print("=" * 70)
print("ADMITRA CONDITION CANDIDATE CHECK")
print("=" * 70)

print()
print("Rows:")
print(f"{len(data):,}")

print()
print("Overall readmission rate:")
print(
    f"{data['readmitted_30_days'].mean() * 100:.2f}%"
)


# ============================================================
# CONDITIONS ALREADY IN DATASET
# ============================================================

existing_conditions = [
    "diabetes",
    "hypertension",
    "kidney_disease",
    "heart_failure",
    "chronic_lung_disease",
]

print()
print("=" * 70)
print("EXISTING CONDITION FEATURES")
print("=" * 70)

for condition in existing_conditions:

    if condition not in data.columns:
        print()
        print(f"{condition}: NOT FOUND")
        continue

    positive = data[data[condition] == 1]
    negative = data[data[condition] == 0]

    print()
    print(condition.upper())
    print("-" * 70)

    print(
        f"Present: "
        f"{len(positive):,} "
        f"({len(positive) / len(data) * 100:.2f}%)"
    )

    if len(positive) > 0:
        print(
            "Readmission rate when present: "
            f"{positive['readmitted_30_days'].mean() * 100:.2f}%"
        )

    if len(negative) > 0:
        print(
            "Readmission rate when absent: "
            f"{negative['readmitted_30_days'].mean() * 100:.2f}%"
        )

    print(
        "Correlation with readmission: "
        f"{data[condition].corr(data['readmitted_30_days']):.4f}"
    )


# ============================================================
# CORONARY ARTERY DISEASE CHECK
# ============================================================

print()
print("=" * 70)
print("CORONARY ARTERY DISEASE")
print("=" * 70)

cad_candidates = [
    "coronary_artery_disease",
    "coronary_heart_disease",
    "ischemic_heart_disease",
    "cad",
]

found = False

for column in cad_candidates:

    if column in data.columns:

        found = True

        positive = data[data[column] == 1]
        negative = data[data[column] == 0]

        print()
        print(f"Found column: {column}")

        print(
            f"Present: "
            f"{len(positive):,} "
            f"({len(positive) / len(data) * 100:.2f}%)"
        )

        if len(positive) > 0:
            print(
                "Readmission rate when present: "
                f"{positive['readmitted_30_days'].mean() * 100:.2f}%"
            )

        if len(negative) > 0:
            print(
                "Readmission rate when absent: "
                f"{negative['readmitted_30_days'].mean() * 100:.2f}%"
            )

        print(
            "Correlation with readmission: "
            f"{data[column].corr(data['readmitted_30_days']):.4f}"
        )


if not found:

    print()
    print(
        "No coronary artery disease feature currently exists "
        "in the V3 dataset."
    )

    print(
        "We will need to derive it from the Synthea "
        "condition/diagnosis data before training V4."
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)