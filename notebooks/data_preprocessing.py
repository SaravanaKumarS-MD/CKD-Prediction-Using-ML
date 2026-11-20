
import os
import csv
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# STEP 1: LOAD DATASET
# ==========================================

dataset_path = "dataset/chronic_kidney_disease.arff"

column_names = [
    "age", "bp", "sg", "al", "su",
    "rbc", "pc", "pcc", "ba", "bgr",
    "bu", "sc", "sod", "pot", "hemo",
    "pcv", "wbcc", "rbcc", "htn", "dm",
    "cad", "appet", "pe", "ane", "class"
]

rows = []
data_started = False

with open(dataset_path, "r", encoding="utf-8-sig") as file:
    for line in file:
        if line.strip().lower() == "@data":
            data_started = True
            continue

        if not data_started or not line.strip():
            continue

        row = next(csv.reader([line.strip()], skipinitialspace=True))
        row = [value.strip().strip("'").strip('"').strip() for value in row]

        if row and row[-1] == "":
            row.pop()

        rows.append(row)

# Detect whether dataset contains an ID column
if rows and len(rows[0]) == 26:
    column_names = ["id"] + column_names
    df = pd.DataFrame(rows, columns=column_names)
    df.drop(columns=["id"], inplace=True)
elif rows and len(rows[0]) == 25:
    df = pd.DataFrame(rows, columns=column_names)
else:
    raise ValueError(
        f"Unexpected dataset columns: {len(rows[0]) if rows else 0}"
    )

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

# ==========================================
# STEP 2: CLEAN DATA
# ==========================================

df = df.replace(r"^\s*$", np.nan, regex=True)
df = df.replace("?", np.nan)

for column in df.columns:
    if df[column].dtype == "object":
        df[column] = df[column].astype(str).str.strip()

# ==========================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["class"]).copy()

target = (
    df["class"]
    .astype(str)
    .str.strip()
    .str.lower()
    .apply(lambda value: re.sub(r"[^a-z]", "", value))
)

target_mapping = {
    "ckd": 1,
    "notckd": 0
}

y = target.map(target_mapping)

if y.isnull().any():
    print("Invalid target values:", target[y.isnull()].unique())
    raise ValueError("Please inspect invalid target values.")

y = y.astype(int)

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)
print("\nTarget Distribution:")
print(y.value_counts())

# ==========================================
# STEP 4: DEFINE FEATURE TYPES
# ==========================================

numerical_columns = [
    "age", "bp", "sg", "al", "su",
    "bgr", "bu", "sc", "sod", "pot",
    "hemo", "pcv", "wbcc", "rbcc"
]

categorical_columns = [
    "rbc", "pc", "pcc", "ba", "htn",
    "dm", "cad", "appet", "pe", "ane"
]

# Convert numerical features
for column in numerical_columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Clean categorical features
for column in categorical_columns:
    X[column] = (
        X[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"": pd.NA, "?": pd.NA})
    )

print("\nNumerical Features:", len(numerical_columns))
print("Categorical Features:", len(categorical_columns))

# ==========================================
# STEP 5: TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# STEP 6: NUMERICAL PIPELINE
# ==========================================

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ==========================================
# STEP 7: CATEGORICAL PIPELINE
# ==========================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# ==========================================
# STEP 8: COLUMN TRANSFORMER
# ==========================================

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

# Fit on training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed Successfully!")
print("Processed Training Shape:", X_train_processed.shape)
print("Processed Testing Shape:", X_test_processed.shape)

# ==========================================
# STEP 9: SAVE PREPROCESSED DATA
# ==========================================

os.makedirs("models", exist_ok=True)

preprocessed_data = {
    "preprocessor": preprocessor,
    "X_train": X_train_processed,
    "X_test": X_test_processed,
    "y_train": y_train,
    "y_test": y_test,
    "X_train_raw": X_train,
    "X_test_raw": X_test
}

joblib.dump(
    preprocessed_data,
    "models/preprocessed_data.pkl"
)

print("\n24-Feature Preprocessed Data Saved Successfully!")
