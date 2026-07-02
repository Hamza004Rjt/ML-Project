"""
Smart Healthcare Prediction System — Flask Web Application
==========================================================
Run:  python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import joblib

from flask import Flask, render_template, request

from sklearn.ensemble     import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute        import SimpleImputer
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
app = Flask(__name__)

FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

CLASS_NAMES = ["Non-Diabetic", "Pre-Diabetic", "Diabetic"]

MODEL_PATH   = "model.pkl"
SCALER_PATH  = "scaler.pkl"
IMPUTER_PATH = "imputer.pkl"

RECOMMENDATIONS = {
    0: {
        "title"  : "Non-Diabetic",
        "emoji"  : "✅",
        "badge"  : "safe",
        "summary": "Great news — your indicators suggest no diabetes risk. Keep up the healthy habits!",
        "tips"   : [
            "Maintain a balanced diet rich in vegetables, whole grains, and lean proteins.",
            "Exercise at least 150 min/week (brisk walking, cycling, swimming).",
            "Schedule annual blood glucose screenings as a precaution.",
            "Avoid smoking and keep alcohol consumption moderate.",
            "Stay hydrated and prioritize 7–9 hours of quality sleep nightly.",
        ],
    },
    1: {
        "title"  : "Pre-Diabetic",
        "emoji"  : "⚠️",
        "badge"  : "warning",
        "summary": "Your results indicate pre-diabetic indicators. Lifestyle changes now can prevent progression.",
        "tips"   : [
            "Reduce refined sugars and simple carbohydrates; increase dietary fibre.",
            "Aim for 5–7% body-weight reduction if you are overweight.",
            "Monitor blood glucose levels every 3–6 months with your doctor.",
            "Consult a registered dietitian for a personalised meal plan.",
            "Consider Metformin therapy if lifestyle changes prove insufficient.",
            "Increase physical activity to at least 200 min/week of moderate exercise.",
        ],
    },
    2: {
        "title"  : "Diabetic",
        "emoji"  : "🚨",
        "badge"  : "danger",
        "summary": "Your profile shows diabetic indicators. Please consult a healthcare professional promptly.",
        "tips"   : [
            "Strictly follow your prescribed medication regimen (insulin / oral agents).",
            "Test blood glucose at home 2–4 times daily and log the readings.",
            "Follow a low-glycaemic-index diet and control portion sizes.",
            "Schedule quarterly HbA1c tests and annual eye, kidney, and foot exams.",
            "Engage in regular moderate exercise under direct medical supervision.",
            "Attend certified diabetes education classes to manage your condition effectively.",
        ],
    },
}


# ─────────────────────────────────────────────
# Model Training / Loading
# ─────────────────────────────────────────────
def train_model(data_path: str = "diabetic_data.csv"):
    """Train Random Forest on the BRFSS dataset and persist to disk."""
    print("[*] Training model — this may take a few minutes …")
    df = pd.read_csv(data_path)
    y  = df["Diabetes_012"].astype(int).values
    X  = df[FEATURE_COLS].copy()

    imputer  = SimpleImputer(strategy="median")
    X_imp    = imputer.fit_transform(X)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_scaled, y)

    joblib.dump(model,   MODEL_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump(imputer, IMPUTER_PATH)
    print("[✓] Model trained and saved.")
    return model, scaler, imputer


def load_model():
    """Load persisted model; train from scratch if not found."""
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, IMPUTER_PATH]):
        print("[✓] Loading saved model …")
        return (
            joblib.load(MODEL_PATH),
            joblib.load(SCALER_PATH),
            joblib.load(IMPUTER_PATH),
        )
    # No saved model — train now
    data_path = "diabetic_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"'{data_path}' not found. Place it in the same directory as app.py."
        )
    return train_model(data_path)


MODEL, SCALER, IMPUTER = load_model()

# Register enumerate as a Jinja2 filter so result.html can use it
app.jinja_env.globals['enumerate'] = enumerate


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form.get(col, 0)) for col in FEATURE_COLS]

        X        = np.array(features).reshape(1, -1)
        X_imp    = IMPUTER.transform(X)
        X_scaled = SCALER.transform(X_imp)

        pred  = int(MODEL.predict(X_scaled)[0])
        probs = MODEL.predict_proba(X_scaled)[0]

        result = {
            "prediction"   : pred,
            "class_name"   : CLASS_NAMES[pred],
            "probabilities": [
                {"label": CLASS_NAMES[i], "value": round(float(p) * 100, 1)}
                for i, p in enumerate(probs)
            ],
            "recommendation": RECOMMENDATIONS[pred],
            # Echo back readable inputs for the summary card
            "inputs": {
                "BMI"        : request.form.get("BMI"),
                "Age"        : request.form.get("Age"),
                "GenHlth"    : request.form.get("GenHlth"),
                "HighBP"     : request.form.get("HighBP"),
                "HighChol"   : request.form.get("HighChol"),
                "Smoker"     : request.form.get("Smoker"),
                "PhysActivity": request.form.get("PhysActivity"),
            },
        }
        return render_template("result.html", result=result)

    except Exception as exc:
        return render_template("index.html", error=str(exc))


if __name__ == "__main__":
    app.run(debug=True)
