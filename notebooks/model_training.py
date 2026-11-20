
import joblib
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# STEP 1: LOAD PREPROCESSED DATA
# ==========================================

data = joblib.load("models/preprocessed_data.pkl")

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Preprocessed Data Loaded Successfully")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# STEP 2: TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nRandom Forest Training Completed!")

# ==========================================
# STEP 3: PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# STEP 4: MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not CKD", "CKD"]
))

# ==========================================
# STEP 5: SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nRandom Forest Model Saved Successfully!")
