import pandas as pd


# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

SYNTHEA_PATH = "../synthea/output/csv/"

patients_path = SYNTHEA_PATH + "patients.csv"
encounters_path = SYNTHEA_PATH + "encounters.csv"
conditions_path = SYNTHEA_PATH + "conditions.csv"


# --------------------------------------------------
# LOAD CORE TABLES
# --------------------------------------------------

print("Loading Synthea data...")

patients = pd.read_csv(patients_path)

encounters = pd.read_csv(encounters_path)

conditions = pd.read_csv(conditions_path)


# --------------------------------------------------
# DISPLAY BASIC INFORMATION
# --------------------------------------------------

print()

print("=" * 50)
print("PATIENTS")
print("=" * 50)

print("Shape:")
print(patients.shape)

print()

print("Columns:")
print(patients.columns.tolist())


print()

print("=" * 50)
print("ENCOUNTERS")
print("=" * 50)

print("Shape:")
print(encounters.shape)

print()

print("Columns:")
print(encounters.columns.tolist())


print()

print("=" * 50)
print("CONDITIONS")
print("=" * 50)

print("Shape:")
print(conditions.shape)

print()

print("Columns:")
print(conditions.columns.tolist())

# --------------------------------------------------
# BUILD BASE INPATIENT DATASET
# --------------------------------------------------

# Convert encounter dates to datetime
encounters["START"] = pd.to_datetime(
    encounters["START"],
    utc=True
)

encounters["STOP"] = pd.to_datetime(
    encounters["STOP"],
    utc=True
)


# Keep only inpatient encounters
inpatient = encounters[
    encounters["ENCOUNTERCLASS"] == "inpatient"
].copy()


# Sort each patient's inpatient encounters chronologically
inpatient = inpatient.sort_values(
    by=["PATIENT", "START"]
)


print()

print("=" * 50)
print("BASE INPATIENT DATASET")
print("=" * 50)

print("Number of inpatient encounters:")
print(len(inpatient))


# --------------------------------------------------
# LENGTH OF STAY
# --------------------------------------------------

inpatient["length_of_stay"] = (
    inpatient["STOP"]
    - inpatient["START"]
).dt.total_seconds() / (60 * 60 * 24)


print()

print("LENGTH OF STAY CHECK")
print(
    inpatient[
        [
            "PATIENT",
            "START",
            "STOP",
            "length_of_stay"
        ]
    ].head()
)


# --------------------------------------------------
# PREVIOUS INPATIENT ADMISSIONS
# --------------------------------------------------

inpatient["previous_inpatient_admissions"] = (
    inpatient
    .groupby("PATIENT")
    .cumcount()
)


print()

print("PREVIOUS INPATIENT ADMISSIONS CHECK")
print(
    inpatient[
        [
            "PATIENT",
            "START",
            "previous_inpatient_admissions"
        ]
    ].head(15)
)


# --------------------------------------------------
# NEXT INPATIENT ADMISSION
# --------------------------------------------------

inpatient["next_admission"] = (
    inpatient
    .groupby("PATIENT")["START"]
    .shift(-1)
)


inpatient["days_to_next_admission"] = (
    inpatient["next_admission"]
    - inpatient["STOP"]
).dt.total_seconds() / (60 * 60 * 24)


# --------------------------------------------------
# 30-DAY READMISSION TARGET
# --------------------------------------------------

inpatient["readmitted_30_days"] = (
    (inpatient["days_to_next_admission"] >= 0)
    &
    (inpatient["days_to_next_admission"] <= 30)
).astype(int)


print()

print("=" * 50)
print("READMISSION TARGET CHECK")
print("=" * 50)

print("Target counts:")
print(
    inpatient["readmitted_30_days"].value_counts()
)

print()

print("Target percentages:")
print(
    inpatient["readmitted_30_days"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# ADD PATIENT DEMOGRAPHICS
# --------------------------------------------------

# Keep only the patient columns needed for modeling
patient_features = patients[
    [
        "Id",
        "BIRTHDATE",
        "GENDER"
    ]
].copy()


# Rename patient ID so it matches the inpatient table
patient_features = patient_features.rename(
    columns={
        "Id": "PATIENT",
        "GENDER": "gender"
    }
)


# Convert birthdate to datetime
patient_features["BIRTHDATE"] = pd.to_datetime(
    patient_features["BIRTHDATE"],
    utc=True
)


# Merge patient information onto each hospitalization
inpatient = inpatient.merge(
    patient_features,
    on="PATIENT",
    how="left"
)


# --------------------------------------------------
# CALCULATE AGE AT ADMISSION
# --------------------------------------------------

inpatient["age_at_admission"] = (
    (
        inpatient["START"]
        - inpatient["BIRTHDATE"]
    ).dt.total_seconds()
    / (365.25 * 24 * 60 * 60)
)


# Convert age to whole years
inpatient["age_at_admission"] = (
    inpatient["age_at_admission"]
    .astype(int)
)


# --------------------------------------------------
# DEMOGRAPHIC CHECK
# --------------------------------------------------

print()

print("=" * 50)
print("PATIENT DEMOGRAPHICS")
print("=" * 50)

print(
    inpatient[
        [
            "PATIENT",
            "START",
            "BIRTHDATE",
            "age_at_admission",
            "gender"
        ]
    ].head(10)
)

print()

print("AGE SUMMARY")
print(
    inpatient["age_at_admission"].describe()
)

print()

print("GENDER DISTRIBUTION")
print(
    inpatient["gender"].value_counts(
        dropna=False
    )
)
# --------------------------------------------------
# PRIOR HEALTHCARE UTILIZATION
# --------------------------------------------------

# Create a smaller encounter table containing only the
# information needed for utilization history
utilization = encounters[
    [
        "PATIENT",
        "START",
        "ENCOUNTERCLASS"
    ]
].copy()


# Sort encounters chronologically
utilization = utilization.sort_values(
    by=["PATIENT", "START"]
)


# --------------------------------------------------
# COUNT PRIOR EMERGENCY VISITS
# --------------------------------------------------

emergency_visits = utilization[
    utilization["ENCOUNTERCLASS"] == "emergency"
].copy()


# --------------------------------------------------
# COUNT PRIOR EMERGENCY VISITS FOR EACH HOSPITALIZATION
# --------------------------------------------------

# We will calculate this directly for each inpatient admission
# using the patient's encounters that occurred before admission.

prior_emergency_counts = []

prior_total_counts = []


for patient_id, patient_inpatient in inpatient.groupby("PATIENT"):

    # Get all healthcare encounters for this patient
    patient_encounters = utilization[
        utilization["PATIENT"] == patient_id
    ]

    # Get this patient's emergency encounters
    patient_emergency = patient_encounters[
        patient_encounters["ENCOUNTERCLASS"] == "emergency"
    ]

    for admission_time in patient_inpatient["START"]:

        # Count emergency visits before this admission
        prior_emergency = (
            patient_emergency["START"] < admission_time
        ).sum()

        # Count all encounters before this admission
        prior_total = (
            patient_encounters["START"] < admission_time
        ).sum()

        prior_emergency_counts.append(
            prior_emergency
        )

        prior_total_counts.append(
            prior_total
        )


# Add utilization features to inpatient dataset
inpatient["previous_emergency_visits"] = (
    prior_emergency_counts
)

inpatient["previous_encounters"] = (
    prior_total_counts
)


# --------------------------------------------------
# UTILIZATION CHECK
# --------------------------------------------------

print()

print("=" * 50)
print("PRIOR HEALTHCARE UTILIZATION")
print("=" * 50)

print(
    inpatient[
        [
            "PATIENT",
            "START",
            "previous_inpatient_admissions",
            "previous_emergency_visits",
            "previous_encounters"
        ]
    ].head(20)
)

print()

print("PREVIOUS EMERGENCY VISITS SUMMARY")
print(
    inpatient["previous_emergency_visits"].describe()
)

print()

print("PREVIOUS ENCOUNTERS SUMMARY")
print(
    inpatient["previous_encounters"].describe()
)

# --------------------------------------------------
# CONDITION / COMORBIDITY FEATURES
# --------------------------------------------------

# Convert condition start dates to datetime
conditions["START"] = pd.to_datetime(
    conditions["START"],
    utc=True,
    errors="coerce"
)


# Create lowercase descriptions for easier matching
conditions["description_lower"] = (
    conditions["DESCRIPTION"]
    .fillna("")
    .str.lower()
)


# --------------------------------------------------
# DEFINE CONDITION GROUPS
# --------------------------------------------------

condition_groups = {
    "diabetes": [
        "diabetes",
        "diabetic"
    ],

    "hypertension": [
        "hypertension",
        "hypertensive"
    ],

    "heart_failure": [
        "heart failure",
        "congestive heart failure"
    ],

    "kidney_disease": [
        "chronic kidney disease",
        "renal failure",
        "kidney failure"
    ],

    "chronic_lung_disease": [
        "chronic obstructive pulmonary",
        "copd",
        "emphysema"
    ]
}


# --------------------------------------------------
# PREPARE CONDITION HISTORY BY PATIENT
# --------------------------------------------------

conditions_by_patient = {
    patient_id: group
    for patient_id, group in conditions.groupby("PATIENT")
}


condition_counts = []

condition_flags = {
    condition_name: []
    for condition_name in condition_groups
}


# --------------------------------------------------
# BUILD FEATURES FOR EACH HOSPITALIZATION
# --------------------------------------------------

for _, admission in inpatient.iterrows():

    patient_id = admission["PATIENT"]

    admission_stop = admission["STOP"]

    patient_conditions = conditions_by_patient.get(
        patient_id
    )

    if patient_conditions is None:

        condition_counts.append(0)

        for condition_name in condition_groups:
            condition_flags[condition_name].append(0)

        continue


    # Only use conditions documented by discharge
    known_conditions = patient_conditions[
        patient_conditions["START"] <= admission_stop
    ]


    # Number of unique known condition codes
    condition_counts.append(
        known_conditions["CODE"].nunique()
    )


    # Create disease indicators
    for condition_name, keywords in condition_groups.items():

        disease_present = (
            known_conditions["description_lower"]
            .str.contains(
                "|".join(keywords),
                regex=True,
                na=False
            )
            .any()
        )

        condition_flags[
            condition_name
        ].append(
            int(disease_present)
        )


# --------------------------------------------------
# ADD CONDITION FEATURES
# --------------------------------------------------

inpatient["condition_count"] = condition_counts


for condition_name in condition_groups:

    inpatient[condition_name] = (
        condition_flags[condition_name]
    )


# --------------------------------------------------
# CONDITION FEATURE CHECK
# --------------------------------------------------

print()

print("=" * 50)
print("CONDITION FEATURES")
print("=" * 50)

print(
    inpatient[
        [
            "PATIENT",
            "START",
            "condition_count",
            "diabetes",
            "hypertension",
            "heart_failure",
            "kidney_disease",
            "chronic_lung_disease"
        ]
    ].head(20)
)

print()

print("CONDITION COUNT SUMMARY")
print(
    inpatient["condition_count"].describe()
)

print()

print("CONDITION PREVALENCE")

for condition_name in condition_groups:

    print()

    print(condition_name + ":")

    print(
        inpatient[
            condition_name
        ].value_counts()
    )
# --------------------------------------------------
# MEDICATION AND PROCEDURE FEATURES
# --------------------------------------------------

print()

print("=" * 50)
print("LOADING MEDICATIONS AND PROCEDURES")
print("=" * 50)


medications_path = (
    "../synthea/output/csv/medications.csv"
)

procedures_path = (
    "../synthea/output/csv/procedures.csv"
)


# Load only columns we need
medications = pd.read_csv(
    medications_path,
    usecols=[
        "START",
        "PATIENT",
        "CODE"
    ],
    low_memory=False
)

procedures = pd.read_csv(
    procedures_path,
    usecols=[
        "START",
        "PATIENT",
        "CODE"
    ],
    low_memory=False
)


# Convert dates
medications["START"] = pd.to_datetime(
    medications["START"],
    utc=True,
    errors="coerce"
)

procedures["START"] = pd.to_datetime(
    procedures["START"],
    utc=True,
    errors="coerce"
)


print("Medication records:")
print(len(medications))

print()

print("Procedure records:")
print(len(procedures))


# --------------------------------------------------
# GROUP RECORDS BY PATIENT
# --------------------------------------------------

medications_by_patient = {
    patient_id: group
    for patient_id, group in medications.groupby("PATIENT")
}

procedures_by_patient = {
    patient_id: group
    for patient_id, group in procedures.groupby("PATIENT")
}


# --------------------------------------------------
# CREATE FEATURES
# --------------------------------------------------

medication_counts = []
procedure_counts = []


for _, admission in inpatient.iterrows():

    patient_id = admission["PATIENT"]
    discharge_time = admission["STOP"]


    # -----------------------------
    # MEDICATION HISTORY
    # -----------------------------

    patient_medications = medications_by_patient.get(
        patient_id
    )

    if patient_medications is None:

        medication_counts.append(0)

    else:

        known_medications = patient_medications[
            patient_medications["START"]
            <= discharge_time
        ]

        medication_counts.append(
            known_medications["CODE"].nunique()
        )


    # -----------------------------
    # PROCEDURE HISTORY
    # -----------------------------

    patient_procedures = procedures_by_patient.get(
        patient_id
    )

    if patient_procedures is None:

        procedure_counts.append(0)

    else:

        known_procedures = patient_procedures[
            patient_procedures["START"]
            <= discharge_time
        ]

        procedure_counts.append(
            known_procedures["CODE"].nunique()
        )


# --------------------------------------------------
# ADD FEATURES TO DATASET
# --------------------------------------------------

inpatient["medication_count"] = medication_counts

inpatient["procedure_count"] = procedure_counts


# --------------------------------------------------
# FEATURE CHECK
# --------------------------------------------------

print()

print("=" * 50)
print("MEDICATION / PROCEDURE FEATURES")
print("=" * 50)

print(
    inpatient[
        [
            "PATIENT",
            "START",
            "medication_count",
            "procedure_count"
        ]
    ].head(20)
)

print()

print("MEDICATION COUNT SUMMARY")

print(
    inpatient[
        "medication_count"
    ].describe()
)

print()

print("PROCEDURE COUNT SUMMARY")

print(
    inpatient[
        "procedure_count"
    ].describe()
)
# --------------------------------------------------
# CREATE FINAL V1 MACHINE LEARNING DATASET
# --------------------------------------------------

final_columns = [
    "Id",
    "PATIENT",
    "age_at_admission",
    "gender",
    "length_of_stay",
    "previous_inpatient_admissions",
    "previous_emergency_visits",
    "previous_encounters",
    "condition_count",
    "diabetes",
    "hypertension",
    "heart_failure",
    "kidney_disease",
    "chronic_lung_disease",
    "medication_count",
    "procedure_count",
    "readmitted_30_days"
]


ml_dataset = inpatient[
    final_columns
].copy()


# Rename encounter ID for clarity
ml_dataset = ml_dataset.rename(
    columns={
        "Id": "encounter_id",
        "PATIENT": "patient_id"
    }
)


# --------------------------------------------------
# DATA QUALITY CHECK
# --------------------------------------------------

print()

print("=" * 50)
print("FINAL V1 ML DATASET")
print("=" * 50)

print("Shape:")
print(ml_dataset.shape)

print()

print("Columns:")
print(ml_dataset.columns.tolist())

print()

print("Missing values:")
print(ml_dataset.isna().sum())

print()

print("Duplicate encounter IDs:")
print(
    ml_dataset["encounter_id"].duplicated().sum()
)

print()

print("Target distribution:")
print(
    ml_dataset[
        "readmitted_30_days"
    ].value_counts()
)

print()

print("First five rows:")
print(ml_dataset.head())


# --------------------------------------------------
# SAVE DATASET
# --------------------------------------------------

output_path = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v1.csv"
)

ml_dataset.to_csv(
    output_path,
    index=False
)

print()

print("=" * 50)
print("DATASET SAVED")
print("=" * 50)

print(output_path)