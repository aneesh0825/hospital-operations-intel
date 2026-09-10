import pandas as pd

from src.predict_readmission import predict_readmission


# ============================================================
# LOAD V2 DATA
# ============================================================

DATA_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v2.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "admitra_patient_registry.csv"
)


data = pd.read_csv(
    DATA_PATH
)


# Drop features we decided not to use
data = data.drop(
    columns=[
        "height",
        "weight",
        "temperature"
    ],
    errors="ignore"
)


print("=" * 60)
print("BUILDING ADMITRA PATIENT REGISTRY")
print("=" * 60)

print()

print("Hospitalizations:")
print(len(data))


# ============================================================
# SCORE EACH HOSPITALIZATION
# ============================================================

results = []


for index, row in data.iterrows():

    patient_data = {

        "age_at_admission":
            row["age_at_admission"],

        "gender":
            row["gender"],

        "length_of_stay":
            row["length_of_stay"],

        "previous_inpatient_admissions":
            row["previous_inpatient_admissions"],

        "previous_emergency_visits":
            row["previous_emergency_visits"],

        "previous_encounters":
            row["previous_encounters"],

        "condition_count":
            row["condition_count"],

        "diabetes":
            row["diabetes"],

        "hypertension":
            row["hypertension"],

        "heart_failure":
            row["heart_failure"],

        "kidney_disease":
            row["kidney_disease"],

        "chronic_lung_disease":
            row["chronic_lung_disease"],

        "medication_count":
            row["medication_count"],

        "procedure_count":
            row["procedure_count"],

        "bmi":
            row["bmi"],

        "systolic_bp":
            row["systolic_bp"],

        "diastolic_bp":
            row["diastolic_bp"],

        "heart_rate":
            row["heart_rate"],

        "respiratory_rate":
            row["respiratory_rate"],

        "glucose":
            row["glucose"]
    }


    prediction = predict_readmission(
        patient_data
    )


    results.append(
        {
            "encounter_id":
                row["encounter_id"],

            "patient_id":
                row["patient_id"],

            "readmission_probability":
                prediction[
                    "calibrated_probability"
                ],

            "readmission_probability_percent":
                prediction[
                    "probability_percent"
                ],

            "risk_level":
                prediction[
                    "risk_level"
                ],

            "intervention_recommended":
                prediction[
                    "intervention_recommended"
                ],

            "actual_readmitted_30_days":
                row[
                    "readmitted_30_days"
                ]
        }
    )


    # Progress update every 500 rows
    if (index + 1) % 500 == 0:

        print(
            "Scored",
            index + 1,
            "hospitalizations"
        )


# ============================================================
# CREATE REGISTRY
# ============================================================

registry = pd.DataFrame(
    results
)


# ============================================================
# ADD USEFUL PATIENT CONTEXT
# ============================================================

context_columns = [
    "encounter_id",
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
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "respiratory_rate",
    "glucose"
]


registry = registry.merge(
    data[
        context_columns
    ],
    on="encounter_id",
    how="left"
)


# ============================================================
# SORT HIGHEST RISK FIRST
# ============================================================

registry = registry.sort_values(
    by="readmission_probability",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)
print("REGISTRY SUMMARY")
print("=" * 60)

print()

print("Registry shape:")
print(registry.shape)

print()

print("Risk levels:")
print(
    registry[
        "risk_level"
    ].value_counts()
)

print()

print("Intervention flags:")
print(
    registry[
        "intervention_recommended"
    ].value_counts()
)

print()

print("Average predicted risk:")
print(
    round(
        registry[
            "readmission_probability_percent"
        ].mean(),
        2
    ),
    "%"
)

print()

print("Top 10 highest-risk encounters:")
print(
    registry[
        [
            "patient_id",
            "readmission_probability_percent",
            "risk_level",
            "intervention_recommended"
        ]
    ].head(10)
)


# ============================================================
# SAVE
# ============================================================

registry.to_csv(
    OUTPUT_PATH,
    index=False
)


print()     

print("=" * 60)
print("PATIENT REGISTRY SAVED")
print("=" * 60)

print()

print(OUTPUT_PATH)