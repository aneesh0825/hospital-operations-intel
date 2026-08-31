import pandas as pd
import numpy as np
import plotly.express as px

# Load the raw patient dataset
df = pd.read_csv("data/raw/patient_data.csv")


# Look at the first five rows
print("FIRST FIVE ROWS")
print(df.head())

print()


# Check the size of the dataset
print("DATASET SHAPE")
print(df.shape)

print()


# Check column names
print("COLUMNS")
print(df.columns.tolist())

print()


# Check data types and missing values
print("DATASET INFORMATION")
df.info()

print()
print("MISSING VALUES")
print(df.isnull().sum())

print()

print("DUPLICATE ROWS")
print(df.duplicated().sum())

print()

print("NUMERICAL SUMMARY")
print(df.describe())

print()

print("DUPLICATE PATIENT IDs")
print(df["patient_id"].duplicated().sum())

print()

print("READMISSION COUNTS")
print(df["readmitted_30_days"].value_counts())

print()

print("READMISSION PERCENTAGES")
print(df["readmitted_30_days"].value_counts(normalize=True) * 100)

df["readmitted"] = df["readmitted_30_days"].map({
    "Yes": 1,
    "No": 0
})

diabetes_readmission = (
    df.groupby("diabetes")["readmitted"]
    .mean()
    .mul(100)
    .round(2)
)

print()
print("READMISSION RATE BY DIABETES")
print(diabetes_readmission)

previous_admission_readmission = (
    df.groupby("previous_admissions")["readmitted"]
    .mean()
    .mul(100)
    .round(2)
)

print()
print("READMISSION RATE BY PREVIOUS ADMISSIONS")
print(previous_admission_readmission)

print()
print("PATIENT COUNT BY PREVIOUS ADMISSIONS")
print(df["previous_admissions"].value_counts().sort_index())

readmission_by_diabetes = (
    df.groupby("diabetes")["readmitted"]
    .mean()
    .mul(100)
    .reset_index()
)

fig = px.bar(
    readmission_by_diabetes,
    x="diabetes",
    y="readmitted",
    title="30-Day Readmission Rate by Diabetes Status",
    labels={
        "diabetes": "Diabetes",
        "readmitted": "Readmission Rate (%)"
    },
    text_auto=".1f"
)

print("Opening Plotly chart...")
fig.show()
df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 34, 49, 64, 79, 90],
    labels = ["18-34", "35-49", "50-64", "65-79", "80-90"]
)

age_readmission =(
    df.groupby("age_group", observed=True)["readmitted"]
    .mean()
    .mul(100)
    .round(2)
)

print()
print("READMISSION RATE BY AGE GROUP")
print(age_readmission)


# HYPERTENSION ANALYSIS
hypertension_readmission = (
    df.groupby("hypertension")["readmitted"]
    .mean()
    .mul(100)
    .round(2)
)

print()
print("READMISSION RATE BY HYPERTENSION")
print(hypertension_readmission)


df["long_stay"] = np.where(
    df["length_of_stay"] >= 7,
    "7+ days",
    "Under 7 days"
)

stay_readmission = (
    df.groupby("long_stay")["readmitted"]
    .mean()
    .mul(100)
    .round(2)
)

print()
print("READMISSION RATE BY LENGTH OF STAY")
print(stay_readmission)