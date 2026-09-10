from src.predict_readmission import predict_readmission


def test_patient(name, **changes):
    patient = {
        "age_at_admission": 60,
        "gender": "M",
        "length_of_stay": 5.0,
        "previous_inpatient_admissions": 0,
        "previous_emergency_visits": 0,
        "previous_encounters": 5,
        "condition_count": 2,
        "diabetes": 0,
        "hypertension": 0,
        "heart_failure": 0,
        "kidney_disease": 0,
        "chronic_lung_disease": 0,
        "medication_count": 5,
        "procedure_count": 3,
        "bmi": 27.0,
        "systolic_bp": 120,
        "diastolic_bp": 80,
        "heart_rate": 75,
        "respiratory_rate": 16,
        "glucose": 95.0,
    }

    patient.update(changes)

    result = predict_readmission(patient)

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f'Probability: {result["probability_percent"]}%')
    print(f'Risk level: {result["risk_level"]}')
    print(f'Review recommended: {result["intervention_recommended"]}')


test_patient("1. BASELINE PATIENT")

test_patient(
    "2. HIGH UTILIZATION",
    previous_inpatient_admissions=10,
    previous_emergency_visits=7,
    previous_encounters=20,
)

test_patient(
    "3. HIGH CLINICAL BURDEN",
    condition_count=15,
    medication_count=11,
    procedure_count=18,
    diabetes=1,
    hypertension=1,
    heart_failure=1,
    kidney_disease=1,
    chronic_lung_disease=1,
)

test_patient(
    "4. HIGH UTILIZATION + HIGH CLINICAL BURDEN",
    previous_inpatient_admissions=10,
    previous_emergency_visits=7,
    previous_encounters=20,
    condition_count=15,
    medication_count=11,
    procedure_count=18,
    diabetes=1,
    hypertension=1,
    heart_failure=1,
    kidney_disease=1,
    chronic_lung_disease=1,
)

test_patient(
    "5. EXTREME VITALS",
    systolic_bp=80,
    diastolic_bp=45,
    heart_rate=130,
    respiratory_rate=8,
    glucose=250,
)

test_patient(
    "6. EVERYTHING COMBINED",
    age_at_admission=73,
    length_of_stay=10,
    previous_inpatient_admissions=10,
    previous_emergency_visits=7,
    previous_encounters=20,
    condition_count=15,
    medication_count=11,
    procedure_count=18,
    diabetes=1,
    hypertension=1,
    heart_failure=1,
    kidney_disease=1,
    chronic_lung_disease=1,
    bmi=31,
    systolic_bp=90,
    diastolic_bp=55,
    heart_rate=120,
    respiratory_rate=8,
    glucose=220,
)