import pandas as pd

patients  = pd.read_csv("data/raw/patients.csv.gz")

print("FIRST FIVE ROWS")
print(patients.head())

print()

print("SHAPE")
print(patients.shape)

print()

print("COLUMNS")
print(patients.columns.tolist())

print()

print("DATASET INFO")
patients.info()

print()
print("=" * 50)
print("ADMISSIONS DATA")
print("=" * 50)

admissions = pd.read_csv("data/raw/admissions.csv.gz")

print()
print("FIRST FIVE ROWS")
print(admissions.head())

print()
print("SHAPE")
print(admissions.shape)

print()
print("COLUMNS")
print(admissions.columns.tolist())

print()
print("DATASET INFO")
admissions.info()

print()
print("ADMISSIONS PER PATIENT")

admissions_per_patient = (
    admissions.groupby("subject_id")
    .size()
    .sort_values(ascending=False)
)

print(admissions_per_patient.head(10))

print()
print("PATIENTS WITH MORE THAN ONE ADMISSION")

multiple_admissions = (admissions_per_patient > 1).sum()

print(multiple_admissions)

print()
print("=" * 50)
print("DIAGNOSIS DATA")
print("=" * 50)

diagnoses = pd.read_csv("data/raw/diagnoses_icd.csv.gz")

print()
print("FIRST FIVE ROWS")
print(diagnoses.head())

print()
print("SHAPE")
print(diagnoses.shape)

print()
print("COLUMNS")
print(diagnoses.columns.tolist())

print()
print("DATASET INFO")
diagnoses.info()

print()
print("=" * 50)
print("ICD DIAGNOSIS DICTIONARY")
print("=" * 50)

diagnosis_names = pd.read_csv("data/raw/d_icd_diagnoses.csv.gz")

print()
print("FIRST FIVE ROWS")
print(diagnosis_names.head())

print()
print("SHAPE")
print(diagnosis_names.shape)

print()
print("COLUMNS")
print(diagnosis_names.columns.tolist())

diagnoses_with_names = diagnoses.merge(
    diagnosis_names,
    on = ["icd_code", "icd_version"],
    how="left"
)

print()
print("DIAGNOSES WITH NAMES")
print(
    diagnoses_with_names[
        ["subject_id" , "hadm_id", "seq_num", "icd_code", "long_title"]
    ].head(10)
)