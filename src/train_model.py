import pandas as pd

# Load the finished machine learning dataset
data = pd.read_csv("data/processed/readmission_ml_dataset.csv")

print("DATASET SHAPE")
print(data.shape)

print()
print("COLUMNS")
print(data.columns.tolist())

print()
print("TARGET DISTRIBUTION")
print(data["readmitted_30_days"].value_counts())

print()
print("TARGET PERCENTAGES")
print(data["readmitted_30_days"].value_counts(normalize=True) * 100)

X = data.drop(
    columns =[
        "subject_id",
        "hadm_id",
        "readmitted_30_days"
    ]
)

y = data["readmitted_30_days"]

print()
print("=" * 50)
print("MODEL FEATURES")
print("=" * 50)

print(X.columns.tolist())

print()
print("X SHAPE")
print(X.shape)

print()
print("Y SHAPE")
print(y.shape)

print()
print("FIRST FIVE TARGET VALUES")
print(y.head())

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("=" * 50)
print("TRAIN / TEST SPLIT")
print("=" * 50)

print("Training features shape:")
print(X_train.shape)

print()

print("Testing features shape:")
print(X_test.shape)

print()

print("Training target distribution:")
print(y_train.value_counts())

print()

print("Testing target distribution:")
print(y_test.value_counts())

# Convert categorical columns into numerical columns
X_train_encoded = pd.get_dummies(
    X_train,
    columns=["gender", "insurance", "admission_type"],
    dtype=int
)

X_test_encoded = pd.get_dummies(
    X_test,
    columns=["gender", "insurance", "admission_type"],
    dtype=int
)

# Make sure training and testing sets have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)

print()
print("=" * 50)
print("ENCODED FEATURES")
print("=" * 50)

print("Training shape:")
print(X_train_encoded.shape)

print()

print("Testing shape:")
print(X_test_encoded.shape)

print()

print("Encoded columns:")
print(X_train_encoded.columns.tolist())

print()

print("First five training rows:")
print(X_train_encoded.head())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#Creates Baseline Model
baseline_model = LogisticRegression(
    class_weight= "balanced",
    max_iter=1000,
    random_state = 42
)

# Train the model
baseline_model.fit(X_train_encoded, y_train)

# Make predictions on unseen test data
y_pred = baseline_model.predict(X_test_encoded)

print()
print("=" * 50)
print("LOGISTIC REGRESSION BASELINE")
print("=" * 50)

print("Accuracy:")
print(round(accuracy_score(y_test, y_pred), 3))

print()

print("Precision:")
print(round(precision_score(y_test, y_pred), 3))

print()

print("Recall:")
print(round(recall_score(y_test, y_pred), 3))

print()

print("F1 Score:")
print(round(f1_score(y_test, y_pred), 3))

print()

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))