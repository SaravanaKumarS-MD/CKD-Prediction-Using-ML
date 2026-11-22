# Chronic Kidney Disease Prediction Using Machine Learning

## Project Overview

Chronic Kidney Disease (CKD) Prediction is a machine learning-based healthcare support application designed to predict the possibility of Chronic Kidney Disease using patient clinical parameters.

This project uses a **Random Forest Classifier** to analyze patient data and classify the prediction as CKD or Not CKD. A web application developed using Flask allows users to enter clinical parameters and view prediction results.

## Problem Statement

Chronic Kidney Disease may progress silently during its early stages. Early identification of potential CKD risk through clinical parameters can support timely medical evaluation.

The objective of this project is to develop a machine learning model that analyzes clinical data and predicts the possibility of Chronic Kidney Disease.

## Objectives

- Predict CKD using machine learning.
- Analyze multiple clinical parameters.
- Handle missing values and preprocess medical data.
- Train and evaluate a Random Forest model.
- Develop a user-friendly web application using Flask.
- Provide prediction results and model confidence scores.

## Technologies Used

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Algorithm:** Random Forest Classifier
- **Data Processing:** Pandas, NumPy
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Development Environment:** Visual Studio Code

## Dataset

The dataset used in this project is the Chronic Kidney Disease dataset obtained from the UCI Machine Learning Repository.

- Total records: 400
- Total columns: 25
- Input features: 24
- Target variable: CKD classification

### Dataset Source

https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

## Input Features

The model uses 24 clinical parameters:

| No. | Feature | Description |
|---|---|---|
| 1 | Age | Patient age |
| 2 | BP | Blood pressure |
| 3 | SG | Specific gravity |
| 4 | AL | Albumin |
| 5 | SU | Sugar |
| 6 | RBC | Red blood cells |
| 7 | PC | Pus cells |
| 8 | PCC | Pus cell clumps |
| 9 | BA | Bacteria |
| 10 | BGR | Blood glucose random |
| 11 | BU | Blood urea |
| 12 | SC | Serum creatinine |
| 13 | SOD | Sodium |
| 14 | POT | Potassium |
| 15 | HEMO | Hemoglobin |
| 16 | PCV | Packed cell volume |
| 17 | WBCC | White blood cell count |
| 18 | RBCC | Red blood cell count |
| 19 | HTN | Hypertension |
| 20 | DM | Diabetes mellitus |
| 21 | CAD | Coronary artery disease |
| 22 | APPET | Appetite |
| 23 | PE | Pedal edema |
| 24 | ANE | Anemia |

## Machine Learning Methodology

The project follows these steps:

1. Dataset collection
2. Data analysis
3. Data cleaning
4. Missing value handling
5. Feature preprocessing
6. Encoding categorical variables
7. Train-test splitting
8. Model training
9. Model evaluation
10. Flask application development

### Data Preprocessing

- Numerical missing values are handled using median imputation.
- Categorical missing values are handled using most-frequent imputation.
- Numerical features are standardized using StandardScaler.
- Categorical features are encoded using OneHotEncoder.

### Model

**Random Forest Classifier**

The Random Forest algorithm combines multiple decision trees to generate a classification prediction.

Model configuration:
- Number of estimators: 100
- Random state: 42
- Training data: 80%
- Testing data: 20%

## Model Evaluation

The model was evaluated using accuracy, confusion matrix, precision, recall and F1-score.

### Results

| Metric | Result |
|---|---:|
| Test Accuracy | 100% |
| True Positive | 50 |
| True Negative | 30 |
| False Positive | 0 |
| False Negative | 0 |
| Test Samples | 80 |

### Confusion Matrix

| | Predicted CKD | Predicted Not CKD |
|---|---:|---:|
| Actual CKD | 50 | 0 |
| Actual Not CKD | 0 | 30 |

The model achieved 100% accuracy on the selected test split. These results are based on a small dataset and do not guarantee similar performance on real-world patient data.

## Application Features

- User-friendly web interface
- Input form for 24 clinical parameters
- CKD / Not CKD prediction
- Model confidence score
- Prediction explanation
- Patient input summary
- Flask-based application

## Project Structure

```text
CKD-Prediction-Using-ML/
│
├── dataset/
│   └── chronic_kidney_disease.arff
│
├── models/
│   ├── ckd_prediction_pipeline.pkl
│   ├── random_forest_model.pkl
│   └── preprocessed_data.pkl
│
├── notebooks/
│   ├── data_analysis.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_verification.py
│   └── save_deployment_model.py
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/SaravanaKumarS-MD/CKD-Prediction-Using-ML.git
```

### 2. Navigate to the Project Directory

```bash
cd CKD-Prediction-Using-ML
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Application

```bash
python app.py
```

### 7. Open in Browser

```text
http://127.0.0.1:5000
```

Enter the required clinical parameters and click the prediction button to view the result.

## Future Enhancements

- Deploy the application to a cloud platform.
- Create a Jupyter Notebook for complete project documentation.
- Improve model validation using larger datasets.
- Add additional model evaluation and explainability techniques.
- Improve the user interface and reporting features.

## Disclaimer

This application is developed for educational and research purposes only. The predictions are not medical diagnoses. Clinical decisions must be made by qualified healthcare professionals using appropriate medical evaluation and laboratory investigations.

## Author

**Saravana Kumar S**  
B.Tech Artificial Intelligence and Data Science

GitHub Repository:  
https://github.com/SaravanaKumarS-MD/CKD-Prediction-Using-ML
