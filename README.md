# Admitra

**History-aware machine learning for 30-day hospital readmission risk assessment**

Admitra is an end-to-end machine learning project that estimates a patient's probability of hospital readmission within 30 days. It combines longitudinal healthcare utilization, clinical characteristics, calibrated XGBoost predictions, patient-specific explanations, and an interactive Streamlit application.

The project uses synthetic electronic health record data generated with Synthea. It is intended to demonstrate applied machine learning, healthcare analytics, model validation, probability calibration, and explainable AI.

> **Important:** Admitra is a portfolio and research project built with synthetic data. It is not a medical device and should not be used for clinical decision-making.

## What Admitra Does

Hospital readmission risk depends on more than the current hospitalization. A patient's recent admission history and prior utilization can contain important information about future risk.

Admitra therefore uses a history-aware approach. For each hospitalization, the model incorporates information available at or before that encounter, including recent admissions, prior readmissions, time since the previous inpatient admission, clinical burden, utilization, selected chronic conditions, age, BMI, and length of stay.

The Streamlit application converts the model output into:

- a calibrated 30-day readmission probability
- a Low, Moderate, or High risk level
- an operational review recommendation
- patient-specific risk drivers
- a registry for reviewing scored encounters

## Final Production Model

The production model is an XGBoost binary classifier using **16 features**.

### Longitudinal Patient History

- Previous inpatient admissions
- Admissions in the last 30 days
- Admissions in the last 90 days
- Admissions in the last 365 days
- Days since last inpatient admission
- Prior inpatient admission indicator
- Prior readmissions in the last 365 days

### Current Hospitalization and Clinical Utilization

- Age at admission
- Length of stay
- Condition count
- Medication count
- Procedure count
- BMI

### Documented Conditions

- Diabetes
- Hypertension
- Kidney disease

Recent utilization variables, especially admissions and prior readmissions during the previous year, emerged as the strongest predictive signals in the final model.

## Leakage-Safe Validation

A patient can have multiple hospitalizations. Randomly splitting individual encounters could place encounters from the same patient in both training and testing, creating information leakage.

Admitra uses a **patient-level grouped train/test split** so that no patient in the held-out test set appears in the training set.

| Item | Value |
| --- | ---: |
| Total hospitalizations | 11,225 |
| Overall readmission rate | 15.02% |
| Training encounters | 9,201 |
| Held-out test encounters | 2,024 |
| Unique held-out patients | 810 |
| Patient overlap | 0 |

The training procedure also accounts for class imbalance using XGBoost's `scale_pos_weight`.

## Held-Out Performance

The final calibrated history-aware model achieved:

| Metric | Result |
| --- | ---: |
| ROC-AUC | **0.932** |
| PR-AUC | **0.754** |
| Brier Score | **0.0559** |
| Precision at operating threshold | **52.3%** |
| Recall at operating threshold | **80.1%** |
| F1 Score at operating threshold | **0.633** |

The held-out population had an observed 30-day readmission rate of **13.19%**.

The mean calibrated predicted probability was **12.04%**, compared with the observed **13.19%** rate.

## Probability Calibration

Class weighting improved identification of readmissions, but the raw XGBoost probabilities were substantially higher than the observed event rate. Admitra therefore includes a separate probability-calibration stage.

The production pipeline uses **sigmoid calibration** to transform the model's raw score into a more interpretable estimated probability while preserving ranking performance.

On the held-out validation population:

```text
Observed readmission rate:    13.19%
Mean calibrated probability:  12.04%
```

This distinction is important: the raw XGBoost output is not presented directly to the user as the estimated clinical risk.

## Operating Threshold

A 50% cutoff is not automatically appropriate for an imbalanced classification problem.

Admitra selects an operational review threshold from grouped out-of-fold training predictions. The selection objective was to maximize F1 while maintaining recall of at least 70%.

The resulting production review threshold is:

```text
22%
```

At that threshold, held-out performance was:

```text
Precision: 52.3%
Recall:    80.1%
F1:        0.633
```

Crossing the threshold means that an encounter is recommended for review. It does **not** mean the model is declaring that the patient will be readmitted.

## Risk Bands

Risk categorization is separate from the operational review threshold.

```text
LOW       < 3.2%
MODERATE  3.2% to 25.7%
HIGH      > 25.7%
```

The independent review threshold is **22%**.

This allows the application to communicate estimated risk while separately representing the point at which an operational workflow would flag an encounter for additional review.

## Explainable Predictions

Admitra uses **SHAP values** to explain individual predictions.

For each scored patient, the application identifies the strongest contributors to the model output and shows whether each factor increased or decreased predicted risk.

Common influential features include:

- Admissions in the last 365 days
- Prior readmissions in the last 365 days
- Admissions in the last 90 days
- Days since the last inpatient admission
- Length of stay
- Procedure and medication utilization

SHAP values explain the behavior of the model. They should not be interpreted as evidence that a feature clinically causes readmission.

## Patient Registry

Admitra includes a patient registry containing scored hospital encounters.

The registry supports:

- ranking encounters by predicted readmission risk
- identifying encounters that exceed the review threshold
- reviewing risk categories
- comparing predictions with observed outcomes
- population-level dashboard analysis

The current production registry contains **11,225 scored hospitalizations**.

## Larger-Cohort Robustness Experiment

The final architecture was also tested on a separately generated, substantially larger Synthea population.

The second generation produced:

```text
28,764 synthetic patients
1,691,624 encounters
27,796 qualifying hospitalizations
```

Using the same 16-feature architecture, performance was:

| Cohort | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Production development cohort | **0.932** | **0.754** |
| Larger independent synthetic cohort | **0.883** | **0.549** |

The decrease is an important limitation. Performance on one generated population did not transfer perfectly to another synthetic population.

The larger-cohort model was therefore retained as a robustness experiment rather than used to replace the production model.

## Additional Feature Experiments

Several additional clinical variables were investigated during development, including vital signs, glucose, feature-availability indicators, ischemic heart disease, and kidney failure.

An 18-feature experiment added ischemic heart disease and kidney failure to the larger-cohort model:

| Model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| 16-feature large-cohort model | 0.8828 | 0.5494 |
| 18-feature large-cohort model | 0.8827 | 0.5501 |

The added diagnoses produced essentially no improvement, so they were excluded from the final production feature set.

## Streamlit Application

Admitra is presented through an interactive Streamlit interface with three primary views.

### Command Center

Provides a population-level view of the patient registry and summarizes risk distribution and review recommendations.

### Risk Assessment

Allows a user to enter patient characteristics and utilization history and receive:

- calibrated readmission probability
- risk level
- review recommendation
- patient-specific risk drivers

### Patient Registry

Provides encounter-level access to scored patients and their model outputs.

## Project Structure

```text
hospital-operations-intel/
|
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- data/
|   `-- processed/
|       |-- admitra_patient_registry.csv
|       |-- synthea_readmission_ml_dataset_v1.csv
|       |-- synthea_readmission_ml_dataset_v2.csv
|       `-- synthea_readmission_ml_dataset_v3.csv
|
|-- models/
|   |-- readmission_history_calibrator.joblib
|   |-- readmission_history_feature_importance.csv
|   |-- readmission_history_features.json
|   |-- readmission_history_medians.json
|   |-- readmission_history_risk_bands.json
|   |-- readmission_history_threshold.json
|   |-- readmission_history_xgboost.joblib
|   `-- readmission_history_xgboost_calibrated.joblib
|
|-- src/
|   |-- build_patient_registry.py
|   |-- build_synthea_ml_dataset.py
|   |-- build_synthea_v2_dataset.py
|   |-- build_synthea_v3_dataset.py
|   |-- calibrate_history_model.py
|   |-- calibrators.py
|   |-- extract_observation_features.py
|   |-- predict_readmission.py
|   |-- train_history_model.py
|   `-- validate_history_model.py
|
`-- experiments/
    `-- experimental and previous model-development scripts
```

Large intermediate datasets and experimental model artifacts are excluded from version control.

## Running Admitra

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Launch Admitra

```powershell
python -m streamlit run app.py
```

Streamlit will provide a local address for opening the application in a browser.

## Technology

- Python
- Pandas
- XGBoost
- scikit-learn
- SHAP
- Streamlit
- Synthea
- Joblib

## Data

Admitra uses synthetic electronic health record data generated with **Synthea**.

Synthetic data makes it possible to demonstrate an end-to-end healthcare machine learning workflow without using real patient records or protected health information.

The synthetic nature of the data is also a major limitation. Relationships learned from Synthea should not be assumed to represent relationships in real clinical populations.

## Limitations

Admitra is an experimental machine learning project, not a validated clinical prediction system.

Key limitations include:

- all patient records are synthetic
- the model has not been externally validated on real-world hospital data
- synthetic populations can contain relationships that differ from real clinical populations
- performance decreased on a separately generated Synthea cohort
- feature importance and SHAP values describe model behavior rather than clinical causation
- the operating threshold has not been evaluated against real clinical costs, staffing constraints, or intervention workflows
- predictions should not be interpreted as medical advice

Real-world deployment would require external validation, prospective testing, clinical review, fairness assessment, privacy and security controls, governance, ongoing monitoring, and any applicable regulatory review.

## Development Workflow

The project evolved through a sequence of increasingly rigorous modeling steps:

1. Built encounter-level 30-day readmission labels.
2. Integrated clinical and observation data.
3. Engineered longitudinal patient-history features.
4. Prevented patient leakage with grouped train/test splitting.
5. Addressed class imbalance during model training.
6. Calibrated raw model probabilities.
7. Selected an operating threshold using grouped out-of-fold predictions.
8. Validated model behavior across patient-history groups.
9. Added patient-specific SHAP explanations.
10. Built a scored patient registry and Streamlit interface.
11. Tested additional clinical variables.
12. Tested the architecture on a larger independently generated synthetic population.
13. Retained unsuccessful or superseded approaches as documented experiments rather than production components.

## Disclaimer

**For educational, research, and portfolio purposes only. Not for clinical use.**