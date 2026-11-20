
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved ML pipeline
model = joblib.load("models/ckd_prediction_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        input_data = {
            "age": float(request.form["age"]),
            "bp": float(request.form["bp"]),
            "sg": request.form["sg"],
            "al": request.form["al"],
            "su": request.form["su"],
            "rbc": request.form["rbc"],
            "pc": request.form["pc"],
            "pcc": request.form["pcc"],
            "ba": request.form["ba"],
            "bgr": float(request.form["bgr"]),
            "bu": float(request.form["bu"]),
            "sc": float(request.form["sc"]),
            "sod": float(request.form["sod"]),
            "pot": float(request.form["pot"]),
            "hemo": float(request.form["hemo"]),
            "pcv": float(request.form["pcv"]),
            "wbcc": float(request.form["wbcc"]),
            "rbcc": float(request.form["rbcc"]),
            "htn": request.form["htn"],
            "dm": request.form["dm"],
            "cad": request.form["cad"],
            "appet": request.form["appet"],
            "pe": request.form["pe"],
            "ane": request.form["ane"]
        }

        # Convert input into DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict
        prediction = model.predict(input_df)[0]

        # Prediction result
        if prediction == 1:
            result = "CKD"
            explanation = (
                "The machine learning model identified patterns "
                "associated with Chronic Kidney Disease."
            )
        else:
            result = "Not CKD"
            explanation = (
                "The machine learning model did not identify "
                "patterns associated with Chronic Kidney Disease."
            )

        # Prediction probability
        probabilities = model.predict_proba(input_df)[0]
        confidence = round(max(probabilities) * 100, 2)

        # Input summary
        input_summary = [
            {"name": "Age", "value": input_data["age"]},
            {"name": "Blood Pressure", "value": input_data["bp"]},
            {"name": "Specific Gravity", "value": input_data["sg"]},
            {"name": "Albumin", "value": input_data["al"]},
            {"name": "Sugar", "value": input_data["su"]},
            {"name": "Red Blood Cells", "value": input_data["rbc"]},
            {"name": "Pus Cells", "value": input_data["pc"]},
            {"name": "Pus Cell Clumps", "value": input_data["pcc"]},
            {"name": "Bacteria", "value": input_data["ba"]},
            {"name": "Blood Glucose", "value": input_data["bgr"]},
            {"name": "Blood Urea", "value": input_data["bu"]},
            {"name": "Serum Creatinine", "value": input_data["sc"]},
            {"name": "Sodium", "value": input_data["sod"]},
            {"name": "Potassium", "value": input_data["pot"]},
            {"name": "Hemoglobin", "value": input_data["hemo"]},
            {"name": "Packed Cell Volume", "value": input_data["pcv"]},
            {"name": "White Blood Cell Count", "value": input_data["wbcc"]},
            {"name": "Red Blood Cell Count", "value": input_data["rbcc"]},
            {"name": "Hypertension", "value": input_data["htn"]},
            {"name": "Diabetes Mellitus", "value": input_data["dm"]},
            {"name": "Coronary Artery Disease", "value": input_data["cad"]},
            {"name": "Appetite", "value": input_data["appet"]},
            {"name": "Pedal Edema", "value": input_data["pe"]},
            {"name": "Anemia", "value": input_data["ane"]}
        ]

        return render_template(
            "index.html",
            prediction=result,
            explanation=explanation,
            confidence=confidence,
            input_summary=input_summary,
            model_name="Random Forest Classifier"
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            prediction="Invalid or missing input. Please check all fields."
        )


if __name__ == "__main__":
    app.run(debug=True)