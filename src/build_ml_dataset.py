import pandas as pd


# Load patient and admission data
patients = pd.read_csv("data/raw/patients.csv.gz")
admissions = pd.read_csv("data/raw/admissions.csv.gz")

print("Patients shape:", patients.shape)
print("Admissions shape:", admissions.shape)

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

admissions["length_of_stay"] = (
    admissions["dischtime"] - admissions["admittime"]
).dt.total_seconds() / 86400

print()
print("LENGTH OF STAY CHECK")
print(admissions[["admittime", "dischtime", "length_of_stay"]].head())

ml_data = admissions.merge(
    patients[["subject_id", "gender", "anchor_age", "anchor_year"]],
    on="subject_id",
    how="left"
)

print()
print("MERGED DATA SHAPE")
print(ml_data.shape)

print()
print("MERGED DATA CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "gender",
            "anchor_age",
            "admission_type",
            "length_of_stay"
        ]
    ].head()
)

# Sort admissions so each patient's hospital stays are in time order
ml_data = ml_data.sort_values(
    by=["subject_id", "admittime"]
)

# Count how many admissions happened before the current admission
ml_data["previous_admissions"] = (
    ml_data.groupby("subject_id")
    .cumcount()
)

print()
print("PREVIOUS ADMISSIONS CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "admittime",
            "previous_admissions"
        ]
    ].head(15)
)

# Find each patient's next hospital admission
ml_data["next_admittime"] = (
    ml_data.groupby("subject_id")["admittime"]
    .shift(-1)
)

# Calculate days between discharge and next admission
ml_data["days_to_readmission"] = (
    ml_data["next_admittime"] - ml_data["dischtime"]
).dt.total_seconds() / 86400

print()
print("READMISSION TIMING CHECK")
print(
    ml_data[
        [
            "subject_id",
            "dischtime",
            "next_admittime",
            "days_to_readmission"
        ]
    ].head(15)
)

print()
print("READMISSION INTERVAL VALIDATION")

print("Negative intervals:")
print((ml_data["days_to_readmission"] < 0).sum())

print()

print("Minimum interval:")
print(ml_data["days_to_readmission"].min())

print()

print("Maximum interval:")
print(ml_data["days_to_readmission"].max())

ml_data["readmitted_30_days"] = (
    (ml_data["days_to_readmission"] >= 0) &
    (ml_data["days_to_readmission"] <= 30)
).astype(int)

print()
print("30-DAY READMISSION COUNTS")
print(ml_data["readmitted_30_days"].value_counts())

print()
print("30-DAY READMISSION PERCENTAGES")
print(
    ml_data["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# Load diagnosis data
diagnoses = pd.read_csv("data/raw/diagnoses_icd.csv.gz")

# Count diagnoses for each hospital admission
diagnosis_counts = (
    diagnoses.groupby("hadm_id")
    .size()
    .reset_index(name="diagnosis_count")
)

# Add diagnosis count to the ML dataset
ml_data = ml_data.merge(
    diagnosis_counts,
    on="hadm_id",
    how="left"
)

# Admissions with no diagnosis records get a count of 0
ml_data["diagnosis_count"] = (
    ml_data["diagnosis_count"]
    .fillna(0)
    .astype(int)
)

print()
print("DIAGNOSIS COUNT CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "diagnosis_count",
            "previous_admissions",
            "readmitted_30_days"
        ]
    ].head(10)
)

#Calculate patient's approximate age at this admission
ml_data["age_at_admission"] = (
    ml_data["anchor_age"]
    + (ml_data["admittime"].dt.year - ml_data["anchor_year"])
)

print()
print("AGE AT ADMISSION CHECK")
print(
    ml_data[
        [
            "subject_id",
            "anchor_age",
            "anchor_year",
            "admittime",
            "age_at_admission"
        ]
    ].head(10)
)

# Create a diabetes indicator for each diagnosis
diagnoses["diabetes"] = (
    (
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].str.startswith("250"))
    )
    |
    (
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith(("E08", "E09", "E10", "E11", "E13")))
    )
).astype(int)

# Find whether each admission had a diabetes diagnosis
diabetes_by_admission = (
    diagnoses.groupby("hadm_id")["diabetes"]
    .max()
    .reset_index()
)

# Add diabetes to our ML dataset
ml_data = ml_data.merge(
    diabetes_by_admission,
    on="hadm_id",
    how="left"
)

ml_data["diabetes"] = (
    ml_data["diabetes"]
    .fillna(0)
    .astype(int)
)

print()
print("DIABETES FEATURE CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "diabetes",
            "diagnosis_count",
            "readmitted_30_days"
        ]
    ].head(15)
)

print()
print("ADMISSIONS WITH DIABETES")
print(ml_data["diabetes"].value_counts())

# Create a hypertension indicator for each diagnosis
diagnoses["hypertension"] = (
    (
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].str.startswith(("401", "402", "403", "404", "405")))
    )
    |
    (
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith(("I10", "I11", "I12", "I13", "I15")))
    )
).astype(int)

# Find whether each admission had a hypertension diagnosis
hypertension_by_admission = (
    diagnoses.groupby("hadm_id")["hypertension"]
    .max()
    .reset_index()
)

# Add hypertension to our ML dataset
ml_data = ml_data.merge(
    hypertension_by_admission,
    on="hadm_id",
    how="left"
)

ml_data["hypertension"] = (
    ml_data["hypertension"]
    .fillna(0)
    .astype(int)
)

print()
print("HYPERTENSION FEATURE CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "diabetes",
            "hypertension",
            "diagnosis_count",
            "readmitted_30_days"
        ]
    ].head(15)
)

print()
print("ADMISSIONS WITH HYPERTENSION")
print(ml_data["hypertension"].value_counts())

# Create heart failure indicator
diagnoses["heart_failure"] = (
    (
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].str.startswith("428"))
    )
    |
    (
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith("I50"))
    )
).astype(int)


# Create kidney disease indicator
diagnoses["kidney_disease"] = (
    (
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].str.startswith(("585", "586")))
    )
    |
    (
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith(("N18", "N19")))
    )
).astype(int)


# Create chronic lung disease indicator
diagnoses["chronic_lung_disease"] = (
    (
        (diagnoses["icd_version"] == 9) &
        (diagnoses["icd_code"].str.startswith(("490", "491", "492", "493", "494", "496")))
    )
    |
    (
        (diagnoses["icd_version"] == 10) &
        (diagnoses["icd_code"].str.startswith(("J40", "J41", "J42", "J43", "J44", "J45", "J47")))
    )
).astype(int)


# Combine the condition indicators by hospital admission
conditions_by_admission = (
    diagnoses.groupby("hadm_id")[
        [
            "heart_failure",
            "kidney_disease",
            "chronic_lung_disease"
        ]
    ]
    .max()
    .reset_index()
)


# Add the conditions to the ML dataset
ml_data = ml_data.merge(
    conditions_by_admission,
    on="hadm_id",
    how="left"
)

condition_columns = [
    "heart_failure",
    "kidney_disease",
    "chronic_lung_disease"
]

ml_data[condition_columns] = (
    ml_data[condition_columns]
    .fillna(0)
    .astype(int)
)


print()
print("NEW CLINICAL FEATURES CHECK")
print(
    ml_data[
        [
            "subject_id",
            "hadm_id",
            "diabetes",
            "hypertension",
            "heart_failure",
            "kidney_disease",
            "chronic_lung_disease",
            "readmitted_30_days"
        ]
    ].head(15)
)


print()
print("CLINICAL CONDITION COUNTS")

for condition in condition_columns:
    print()
    print(condition)
    print(ml_data[condition].value_counts())

# Select columns for the machine learning dataset
model_columns = [
    "subject_id",
    "hadm_id",
    "age_at_admission",
    "gender",
    "insurance",
    "admission_type",
    "length_of_stay",
    "previous_admissions",
    "diagnosis_count",
    "diabetes",
    "hypertension",
    "heart_failure",
    "kidney_disease",
    "chronic_lung_disease",
    "readmitted_30_days"
]

final_ml_data = ml_data[model_columns].copy()

print()
print("=" * 50)
print("FINAL ML DATASET")
print("=" * 50)

print()
print("SHAPE")
print(final_ml_data.shape)

print()
print("FIRST FIVE ROWS")
print(final_ml_data.head())

print()
print("MISSING VALUES")
print(final_ml_data.isnull().sum())

# Save the finished dataset
final_ml_data.to_csv(
    "data/processed/readmission_ml_dataset.csv",
    index=False
)

print()
print("Dataset saved to data/processed/readmission_ml_dataset.csv")