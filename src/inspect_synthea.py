import pandas as pd

encounters = pd.read_csv(
    "../synthea/output/csv/encounters.csv"
)

print("ENCOUNTERS SHAPE")
print(encounters.shape)

print()

print("ENCOUNTER CLASSES")
print(encounters["ENCOUNTERCLASS"].value_counts())

inpatient = encounters[
    encounters["ENCOUNTERCLASS"] == "inpatient"
].copy()

print()

print("=" * 50)
print("INPATIENT ENCOUNTERS")
print("=" * 50)

print("Number of inpatient encounters:")
print(len(inpatient))

inpatient["START"] = pd.to_datetime(
    inpatient["START"],
    utc=True
)

inpatient["STOP"] = pd.to_datetime(
    inpatient["STOP"],
    utc=True
)

inpatient = inpatient.sort_values(
    by=["PATIENT", "START"]
)

inpatient["next_admission"] = (
    inpatient
    .groupby("PATIENT")["START"]
    .shift(-1)
)

inpatient["days_to_next_admission"] = (
    inpatient["next_admission"]
    - inpatient["STOP"]
).dt.total_seconds() / (60 * 60 * 24)

inpatient["readmitted_30_days"] = (
    (inpatient["days_to_next_admission"] >= 0)
    &
    (inpatient["days_to_next_admission"] <= 30)
).astype(int)

print()

print("=" * 50)
print("30-DAY READMISSION RESULTS")
print("=" * 50)

print("Target distribution:")
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

print()

print("Total readmissions within 30 days:")
print(inpatient["readmitted_30_days"].sum())

print()

print("Total inpatient encounters:")
print(len(inpatient))

readmissions = inpatient[
    inpatient["readmitted_30_days"] == 1
]

print()

print("=" * 50)
print("EXAMPLE 30-DAY READMISSIONS")
print("=" * 50)

print(
    readmissions[
        [
            "PATIENT",
            "START",
            "STOP",
            "next_admission",
            "days_to_next_admission",
            "DESCRIPTION"
        ]
    ].head(10)
)