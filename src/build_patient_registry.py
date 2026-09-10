import pandas as pd

from src.predict_readmission import predict_readmission


# ============================================================
# PATHS
# ============================================================

DATA_PATH = (
    "data/processed/"
    "synthea_readmission_ml_dataset_v3.csv"
)

OUTPUT_PATH = (
    "data/processed/"
    "admitra_patient_registry.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    DATA_PATH
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

        "length_of_stay":
            row["length_of_stay"],

        "previous_inpatient_admissions":
            row["previous_inpatient_admissions"],

        "admissions_last_30_days":
            row["admissions_last_30_days"],

        "admissions_last_90_days":
            row["admissions_last_90_days"],

        "admissions_last_365_days":
            row["admissions_last_365_days"],

        "days_since_last_inpatient_admission":
            row["days_since_last_inpatient_admission"],

        "has_prior_inpatient_admission":
            row["has_prior_inpatient_admission"],

        "prior_readmissions_last_365_days":
            row["prior_readmissions_last_365_days"],

        "condition_count":
            row["condition_count"],

        "medication_count":
            row["medication_count"],

        "procedure_count":
            row["procedure_count"],

        "diabetes":
            row["diabetes"],

        "hypertension":
            row["hypertension"],

        "kidney_disease":
            row["kidney_disease"],

        "bmi":
            row["bmi"],
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
                    "probability"
                ],

            "readmission_probability_percent":
                prediction[
                    "probability_percent"
                ],

            "raw_readmission_probability":
                prediction[
                    "raw_probability"
                ],

            "raw_readmission_probability_percent":
                prediction[
                    "raw_probability_percent"
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
                ],
        }
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        index + 1
    ) % 500 == 0:

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
# ADD PATIENT CONTEXT
# ============================================================

context_columns = [
    "encounter_id",

    "age_at_admission",
    "gender",

    "length_of_stay",

    "previous_inpatient_admissions",
    "previous_emergency_visits",
    "previous_encounters",

    "admissions_last_30_days",
    "admissions_last_90_days",
    "admissions_last_365_days",

    "days_since_last_inpatient_admission",
    "has_prior_inpatient_admission",
    "prior_readmissions_last_365_days",

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

    "glucose",
]


available_context_columns = [
    column
    for column in context_columns
    if column in data.columns
]


registry = registry.merge(
    data[
        available_context_columns
    ],
    on="encounter_id",
    how="left"
)


# ============================================================
# SORT HIGHEST RISK FIRST
# ============================================================

registry = (
    registry
    .sort_values(
        by="readmission_probability",
        ascending=False
    )
    .reset_index(
        drop=True
    )
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
print(
    registry.shape
)

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

print("Average raw model risk:")
print(
    round(
        registry[
            "raw_readmission_probability_percent"
        ].mean(),
        2
    ),
    "%"
)

print()

print("Observed readmission rate:")
print(
    round(
        registry[
            "actual_readmitted_30_days"
        ].mean()
        * 100,
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
            "intervention_recommended",
            "admissions_last_90_days",
            "prior_readmissions_last_365_days",
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

print(
    OUTPUT_PATH
)