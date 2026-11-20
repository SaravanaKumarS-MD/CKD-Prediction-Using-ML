
import joblib
import os
from sklearn.pipeline import Pipeline

# Load saved preprocessing data
data = joblib.load("models/preprocessed_data.pkl")

# Load trained Random Forest model
model = joblib.load("models/random_forest_model.pkl")

# Combine preprocessing and model
deployment_pipeline = Pipeline([
    ("preprocessor", data["preprocessor"]),
    ("classifier", model)
])

# Create models folder if not available
os.makedirs("models", exist_ok=True)

# Save complete prediction pipeline
joblib.dump(
    deployment_pipeline,
    "models/ckd_prediction_pipeline.pkl"
)

print("Deployment Pipeline Saved Successfully!")
