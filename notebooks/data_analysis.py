
import csv
import pandas as pd
from io import StringIO

# ----------------------------------
# STEP 1: File paths
# ----------------------------------

file_path = "dataset/chronic_kidney_disease.arff"

# ----------------------------------
# STEP 2: Read original ARFF file
# ----------------------------------

with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

# ----------------------------------
# STEP 3: Extract attributes and clean rows
# ----------------------------------

columns = []
cleaned_lines = []
data_started = False

for line in lines:

    stripped = line.strip()

    if stripped.lower().startswith("@attribute"):
        parts = stripped.split()
        columns.append(parts[1].strip("'\""))

    if stripped.lower() == "@data":
        data_started = True
        cleaned_lines.append(line)
        continue

    if data_started and stripped:

        values = next(csv.reader([stripped]))
        values = [value.strip() for value in values]

        # Remove extra trailing empty field
        if len(values) == len(columns) + 1 and values[-1] == "":
            values.pop()

        # Convert empty fields into missing values
        values = ["?" if value == "" else value for value in values]

        cleaned_lines.append(
            ",".join(values) + "\n"
        )

    else:
        cleaned_lines.append(line)

# ----------------------------------
# STEP 4: Check dataset structure
# ----------------------------------

print("Number of attributes:", len(columns))

# ----------------------------------
# STEP 5: Check invalid values
# ----------------------------------

print("\nChecking dataset rows...")

data_started = False
invalid_count = 0
row_count = 0

allowed = {
    "rbc": ["normal", "abnormal"],
    "pc": ["normal", "abnormal"],
    "pcc": ["present", "notpresent"],
    "ba": ["present", "notpresent"],
    "htn": ["yes", "no"],
    "dm": ["yes", "no"],
    "cad": ["yes", "no"],
    "appet": ["good", "poor"],
    "pe": ["yes", "no"],
    "ane": ["yes", "no"],
    "class": ["ckd", "notckd"]
}

for line_number, line in enumerate(cleaned_lines, start=1):

    if line.strip().lower() == "@data":
        data_started = True
        continue

    if data_started and line.strip():

        values = next(csv.reader([line.strip()]))
        values = [value.strip() for value in values]

        row_count += 1

        if len(values) != len(columns):
            print("\nColumn count mismatch!")
            print("Line:", line_number)
            print("Expected:", len(columns))
            print("Found:", len(values))
            print("Row:", values)
            invalid_count += 1
            continue

        for index, value in enumerate(values):

            if value == "?":
                continue

            attribute = columns[index]

            if attribute in allowed:
                if value.lower() not in allowed[attribute]:
                    print("\nInvalid value found!")
                    print("Line:", line_number)
                    print("Column:", attribute)
                    print("Value:", repr(value))
                    invalid_count += 1

# ----------------------------------
# STEP 6: Validation Summary
# ----------------------------------

print("\nDataset Rows:", row_count)
print("Invalid Entries:", invalid_count)

if invalid_count == 0:
    print("\nNo invalid categorical values found.")
else:
    print("\nPlease inspect invalid entries.")

# ----------------------------------
# STEP 7: Load Cleaned Dataset
# ----------------------------------

data_text = "".join(cleaned_lines).split("@data", 1)[1]

df = pd.read_csv(
    StringIO(data_text),
    header=None,
    names=columns,
    na_values=["?"],
    skipinitialspace=True
)

# ----------------------------------
# STEP 8: Dataset Information
# ----------------------------------

print("\n========== DATASET INFORMATION ==========")

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

# ----------------------------------
# STEP 9: Missing Value Analysis
# ----------------------------------

print("\n========== MISSING VALUE ANALYSIS ==========")

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# ----------------------------------
# STEP 10: CKD Class Distribution
# ----------------------------------

print("\n========== CLASS DISTRIBUTION ==========")

print("\nCKD and Non-CKD Count:")
print(df["class"].value_counts())

print("\nClass Percentage:")
print(df["class"].value_counts(normalize=True) * 100)

# ----------------------------------
# STEP 11: Statistical Summary
# ----------------------------------

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

print("\nDataset Analysis Completed Successfully!")