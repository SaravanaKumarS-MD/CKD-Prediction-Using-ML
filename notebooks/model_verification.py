
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score

# ----------------------------------
# STEP 1: Load Saved Data
# ----------------------------------

data = joblib.load("models/preprocessed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

X_train_raw = data["X_train_raw"]
X_test_raw = data["X_test_raw"]

preprocessor = data["preprocessor"]

print("Verification Data Loaded Successfully!")

# ----------------------------------
# STEP 2: Duplicate Record Check
# ----------------------------------

print("\n========== DUPLICATE CHECK ==========")

print(
    "Training Duplicate Records:",
    X_train_raw.duplicated().sum()
)

print(
    "Testing Duplicate Records:",
    X_test_raw.duplicated().sum()
)

# ----------------------------------
# STEP 3: Train-Test Overlap Check
# ----------------------------------

train_records = set(
    map(tuple, X_train_raw.astype(str).values)
)

test_records = set(
    map(tuple, X_test_raw.astype(str).values)
)

overlap = train_records.intersection(test_records)

print("\n========== DATA LEAKAGE CHECK ==========")

print("Overlapping Records:", len(overlap))

# ----------------------------------
# STEP 4: Create Random Forest Pipeline
# ----------------------------------

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

# ----------------------------------
# STEP 5: Five-Fold Cross Validation
# ----------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    pipeline,
    X_train_raw,
    y_train,
    cv=cv,
    scoring="accuracy"
)

print("\n========== 5-FOLD CROSS VALIDATION ==========")

print("Fold Accuracies:", scores)

print("Mean Accuracy:", round(scores.mean() * 100, 2), "%")

print("Standard Deviation:", round(scores.std() * 100, 2), "%")

# ----------------------------------
# STEP 6: Train and Test Accuracy
# ----------------------------------

pipeline.fit(X_train_raw, y_train)

train_predictions = pipeline.predict(X_train_raw)
test_predictions = pipeline.predict(X_test_raw)

train_accuracy = accuracy_score(
    y_train, train_predictions
)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

print("\n========== OVERFITTING CHECK ==========")

print("Training Accuracy:", round(train_accuracy * 100, 2), "%")

print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

print(
    "Accuracy Difference:",
    round(abs(train_accuracy - test_accuracy) * 100, 2),
    "%"
)

print("\nModel Verification Completed!")