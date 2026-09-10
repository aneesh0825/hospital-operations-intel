\# Admitra



\*\*A history-aware machine learning system for 30-day hospital readmission risk assessment\*\*



Admitra is an end-to-end machine learning project that estimates a patient's probability of hospital readmission within 30 days. The system combines longitudinal healthcare utilization, clinical characteristics, calibrated XGBoost predictions, and patient-level explanations in an interactive Streamlit application.



The project was developed using synthetic patient data generated with Synthea and is intended as a demonstration of applied machine learning, healthcare analytics, model validation, and explainable AI.



> \*\*Important:\*\* Admitra is a portfolio and research project built with synthetic data. It is not a medical device and should not be used for clinical decision-making.



\---



\## Overview



Hospital readmission risk is influenced not only by a patient's current hospitalization, but also by their recent healthcare utilization and prior readmission history.



Admitra therefore uses a \*\*history-aware modeling approach\*\*. Instead of treating each hospitalization as an isolated event, the model incorporates information available before the index hospitalization, including:



\- admissions within the previous 30, 90, and 365 days

\- prior readmissions within the previous year

\- time since the patient's previous inpatient admission

\- cumulative inpatient history

\- clinical burden

\- medication and procedure utilization

\- selected chronic conditions

\- demographic and hospitalization characteristics



The application converts the model output into a calibrated readmission probability, risk category, review recommendation, and patient-specific explanation.



\---



\## Final Model



Admitra's production model is an \*\*XGBoost binary classifier\*\* using 16 features.



\### Patient history



\- Previous inpatient admissions

\- Admissions in the last 30 days

\- Admissions in the last 90 days

\- Admissions in the last 365 days

\- Days since last inpatient admission

\- Prior inpatient admission indicator

\- Prior readmissions in the last 365 days



\### Current hospitalization and clinical utilization



\- Age at admission

\- Length of stay

\- Condition count

\- Medication count

\- Procedure count

\- BMI



\### Documented conditions



\- Diabetes

\- Hypertension

\- Kidney disease



Recent utilization emerged as the strongest source of predictive information, particularly admissions and prior readmissions within the previous year.



\---



\## Model Development



\### Leakage-safe patient splitting



A single patient may have multiple hospitalizations in the dataset. Randomly splitting individual encounters could therefore place encounters belonging to the same patient in both the training and test sets.



Admitra uses a \*\*patient-level grouped train/test split\*\* instead.



Final production development dataset:



\- \*\*11,225 hospitalizations\*\*

\- \*\*15.02% overall readmission rate\*\*

\- \*\*9,201 training encounters\*\*

\- \*\*2,024 held-out test encounters\*\*

\- \*\*0 overlapping patients between training and testing\*\*



This prevents the model from being evaluated on patients it encountered during training.



\### Class imbalance



Because readmissions represent a minority of encounters, the training procedure accounts for class imbalance using XGBoost's `scale\_pos\_weight`.



For the production training cohort:



```text

Scale pos weight: 5.484

```



\---



\## Held-Out Performance



The final calibrated history-aware model achieved:



| Metric | Result |

|---|---:|

| ROC-AUC | \*\*0.932\*\* |

| PR-AUC | \*\*0.754\*\* |

| Brier Score | \*\*0.0559\*\* |

| Accuracy | \*\*87.7%\*\* |

| Precision | \*\*52.3%\*\* |

| Recall | \*\*80.1%\*\* |

| F1 Score | \*\*0.633\*\* |



The held-out test population had an observed 30-day readmission rate of \*\*13.19%\*\*.



After calibration, the model's mean predicted probability was \*\*12.04%\*\*, compared with the observed \*\*13.19%\*\* rate.



\---



\## Probability Calibration



Class weighting improved the model's ability to identify readmissions but caused the raw XGBoost probabilities to systematically overestimate absolute risk.



On out-of-fold training predictions:



```text

Actual readmission rate:       15.42%

Mean raw probability:          30.32%

Mean calibrated probability:   15.42%

```



Admitra evaluated both \*\*isotonic\*\* and \*\*sigmoid\*\* calibration.



The production pipeline uses \*\*sigmoid calibration\*\*, providing substantially more interpretable probabilities while preserving the model's ranking performance.



On the final held-out test set:



```text

Observed readmission rate:     13.19%

Mean calibrated probability:   12.04%

```



\---



\## Operating Threshold



A probability of 50% is not automatically the appropriate decision threshold for an imbalanced healthcare classification problem.



Admitra selects its operating threshold using grouped out-of-fold predictions from the training data.



The selection rule was:



> Highest F1 score while maintaining recall of at least 70%.



This produced a production review threshold of:



```text

22%

```



On the held-out test set, this threshold produced:



```text

Precision: 52.3%

Recall:    80.1%

F1:        0.633

```



A patient exceeding the threshold is flagged for \*\*review\*\*, not automatically classified as someone who will be readmitted.



\---



\## Risk Bands



The application separates estimated probability from the operational review decision.



The calibrated risk bands are:



```text

LOW       < 3.2%

MODERATE  3.2% - 25.7%

HIGH      > 25.7%

```



The operational review threshold is independently set at \*\*22%\*\*.



This means risk categorization and review recommendations serve related but different purposes.



\---



\## Explainable Predictions



Admitra uses \*\*SHAP values\*\* to explain individual model predictions.



For each patient, the application identifies the strongest contributors to the model's prediction and indicates whether each factor increased or decreased estimated readmission risk.



Example factors may include:



```text

Admissions in last 365 days

Prior readmissions in last 365 days

Days since last inpatient admission

Admissions in last 90 days

Length of stay

Procedure count

```



This allows the application to show more than a probability by providing insight into \*\*why the model produced that estimate\*\*.



\---



\## Patient Registry



Admitra also includes a patient registry generated by scoring hospital encounters with the production model.



The registry can be used to:



\- rank encounters by predicted readmission risk

\- identify patients exceeding the review threshold

\- inspect risk categories

\- compare predictions with observed outcomes

\- support dashboard-level population analysis



The production registry contains \*\*11,225 scored hospitalizations\*\*.



\---



\## Robustness Testing



A second, independently generated Synthea population was used to investigate how the modeling approach behaved on a substantially larger synthetic cohort.



The additional generation produced:



```text

28,764 synthetic patients

1,691,624 encounters

27,796 qualifying hospitalizations

```



The same 16-feature architecture achieved:



| Cohort | ROC-AUC | PR-AUC |

|---|---:|---:|

| Production development cohort | \*\*0.932\*\* | \*\*0.754\*\* |

| Larger independent synthetic cohort | \*\*0.883\*\* | \*\*0.549\*\* |



The decline demonstrates that performance on one synthetic population does not guarantee identical performance on another generated population.



For this reason, the larger-cohort model was retained as an experiment rather than replacing the production model.



\---



\## Additional Feature Experiments



Several additional clinical features were evaluated during development.



These included:



\- systolic blood pressure

\- heart rate

\- glucose

\- vital-sign availability

\- glucose availability

\- ischemic heart disease

\- end-stage kidney failure



The larger-cohort baseline and an experimental model adding ischemic heart disease and kidney failure produced:



| Model | ROC-AUC | PR-AUC |

|---|---:|---:|

| 16-feature model | 0.8828 | 0.5494 |

| 18-feature model | 0.8827 | 0.5501 |



The additional diagnoses provided essentially no improvement, so they were not added to the production model.



This experimentation favored a smaller feature set when additional variables did not demonstrate meaningful held-out benefit.



\---



\## Application



Admitra is presented through a Streamlit interface with three primary views.



\### Command Center



Provides a population-level view of the patient registry, including risk distribution, review recommendations, and operational summaries.



\### Risk Assessment



Allows a user to enter patient characteristics and utilization history and receive:



\- calibrated 30-day readmission probability

\- risk level

\- review recommendation

\- patient-specific risk drivers



\### Patient Registry



Provides encounter-level access to scored patients and their predicted risks.



\---



\## Project Structure



```text

hospital-operations-intel/

│

├── app.py

│

├── README.md

│

├── requirements.txt

│

├── .gitignore

│

├── data/

│   └── processed/

│       ├── admitra\_patient\_registry.csv

│       ├── synthea\_readmission\_ml\_dataset\_v1.csv

│       ├── synthea\_readmission\_ml\_dataset\_v2.csv

│       └── synthea\_readmission\_ml\_dataset\_v3.csv

│

├── models/

│   ├── readmission\_history\_calibrator.joblib

│   ├── readmission\_history\_feature\_importance.csv

│   ├── readmission\_history\_features.json

│   ├── readmission\_history\_medians.json

│   ├── readmission\_history\_risk\_bands.json

│   ├── readmission\_history\_threshold.json

│   ├── readmission\_history\_xgboost.joblib

│   └── readmission\_history\_xgboost\_calibrated.joblib

│

├── src/

│   ├── build\_patient\_registry.py

│   ├── build\_synthea\_ml\_dataset.py

│   ├── build\_synthea\_v2\_dataset.py

│   ├── build\_synthea\_v3\_dataset.py

│   ├── calibrate\_history\_model.py

│   ├── calibrators.py

│   ├── extract\_observation\_features.py

│   ├── predict\_readmission.py

│   ├── train\_history\_model.py

│   └── validate\_history\_model.py

│

└── experiments/

&#x20;   └── experimental and previous model-development scripts

```



Large intermediate datasets and experimental model artifacts are excluded from version control.



\---



\## Running Admitra



\### 1. Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



\### 2. Install dependencies



```powershell

pip install -r requirements.txt

```



\### 3. Launch the application



```powershell

python -m streamlit run app.py

```



Streamlit will provide a local address where the Admitra interface can be opened in a browser.



\---



\## Technology



Admitra was developed with:



\- \*\*Python\*\*

\- \*\*Pandas\*\*

\- \*\*XGBoost\*\*

\- \*\*scikit-learn\*\*

\- \*\*SHAP\*\*

\- \*\*Streamlit\*\*

\- \*\*Synthea\*\*

\- \*\*Joblib\*\*



\---



\## Data



The project uses synthetic electronic health record data generated by \*\*Synthea\*\*.



Synthetic data makes it possible to develop and demonstrate the complete machine learning pipeline without using real patient records or protected health information.



The synthetic nature of the data is also an important limitation. Relationships learned from Synthea should not be assumed to represent relationships in real-world clinical populations.



\---



\## Limitations



Admitra is an experimental machine learning project rather than a validated clinical prediction system.



Important limitations include:



\- all patient records are synthetic

\- performance has not been validated on real-world hospital data

\- generated populations can exhibit artificial relationships between variables

\- model performance changed on an independently generated Synthea cohort

\- feature importance and SHAP values describe model behavior, not clinical causation

\- the selected operating threshold has not been evaluated for real-world clinical costs or workflows

\- predictions should not be interpreted as medical advice



Real clinical deployment would require external validation, fairness analysis, prospective evaluation, clinical review, governance, monitoring, and appropriate regulatory and privacy controls.



\---



\## Development Philosophy



The final Admitra model was selected based on held-out performance and validation rather than feature count.



Development included:



1\. building encounter-level readmission labels

2\. incorporating clinical and observation data

3\. engineering leakage-safe longitudinal patient-history features

4\. separating patients between training and testing

5\. addressing class imbalance

6\. calibrating predicted probabilities

7\. selecting an operating threshold using out-of-fold predictions

8\. validating behavior across patient-history groups

9\. adding patient-level SHAP explanations

10\. testing additional clinical variables and a larger independently generated population



Features and model variants that did not provide meaningful improvement were retained as experiments rather than incorporated into the production system.



\---



\## License and Use



This project is intended for educational, research, and portfolio purposes.



\*\*Not for clinical use.\*\*

