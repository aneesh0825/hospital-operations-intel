import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

np.random.seed(42)
Faker.seed(42)

num_patients = 5000

patient_data = []

for i in range(num_patients):
    age = np.random.randint(18, 91)

    bmi = round(
    np.clip(np.random.normal(27, 5), 15, 50),
    1
    )

    diabetes_probability = 0.08

    if age >= 50:
        diabetes_probability += 0.10

    if bmi >= 30:
        diabetes_probability += 0.15

    diabetes = np.random.choice(
        ["Yes", "No"],
        p=[diabetes_probability, 1 - diabetes_probability]
    )

    hypertension_probability = 0.10

    if age >= 50:
        hypertension_probability += 0.20

    if bmi >= 30:
        hypertension_probability += 0.10

    hypertension = np.random.choice(
        ["Yes", "No"],
        p=[hypertension_probability, 1 - hypertension_probability]
    )

    previous_admissions = np.random.poisson(1.5)

    length_of_stay = np.random.randint(1, 15)

    readmission_probability = 0.08 

    if age >= 65:
        readmission_probability += 0.08

    if diabetes == "Yes":
        readmission_probability += 0.10

    if hypertension == "Yes":
        readmission_probability += 0.06

    if previous_admissions >= 2:
        readmission_probability += 0.12

    if length_of_stay >= 7:
        readmission_probability += 0.08

    readmitted_30_days = np.random.choice(
        ["Yes", "No"],
        p=[readmission_probability, 1 - readmission_probability]
    )

    patient = {
        "patient_id": f"P{i + 1:05d}",
        "name": fake.name(),
        "age": np.random.randint(18, 91),
        "sex": np.random.choice(["Male", "Female"]),
        "insurance": np.random.choice(
            ["Private", "Medicare", "Medicaid", "Uninsured"]
        ),
        "department": np.random.choice(
            ["Emergency", "Cardiology", "Neurology", "Orthopedics", "General Medicine"]
        ),
        "bmi": bmi,
        "diabetes": diabetes,
        "hypertension": hypertension,
        "previous_admissions": previous_admissions,
        "length_of_stay": length_of_stay,
        "readmitted_30_days": readmitted_30_days
    }
    patient_data.append(patient)

df = pd.DataFrame(patient_data)

print(df.head())
print()
print(f"Total patients: {len(df)}")

df.to_csv("data/raw/patient_data.csv", index=False)
