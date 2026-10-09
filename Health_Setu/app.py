"""
app.py — Health Setu Flask Backend
------------------------------------
Endpoints
---------
GET  /                  → Home dashboard
GET  /symptom           → Symptom Guidance page
POST /api/predict       → JSON: {symptoms:[...]} → prediction + advice
GET  /hospital          → Hospital Finder page
GET  /api/hospitals     → JSON: search hospitals by query/city
GET  /schemes           → Health Schemes page
"""

import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify

from classifier import BernoulliNaiveBayes

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load ML model once at startup (if already trained)
# ---------------------------------------------------------------------------
MODEL_PATH    = os.path.join("model", "symptom_model.json")
FEATURES_PATH = os.path.join("model", "features.json")
HOSPITAL_CSV  = os.path.join("data", "hospitals.csv")

_model    = None
_features = None


def load_model():
    global _model, _features
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        _model = BernoulliNaiveBayes.load(MODEL_PATH)
        with open(FEATURES_PATH, "r", encoding="utf-8") as fh:
            _features = json.load(fh)
    else:
        _model = None
        _features = None


load_model()

# ---------------------------------------------------------------------------
# Condition → general advice mapping  (educational, NOT medical diagnosis)
# ---------------------------------------------------------------------------
CONDITION_ADVICE = {
    "Flu": {
        "icon": "🤧",
        "general_info": "Influenza (flu) is a contagious respiratory illness caused by influenza viruses. Symptoms typically include fever, cough, body aches, and fatigue.",
        "next_steps": [
            "Rest and stay hydrated.",
            "Over-the-counter medicines (paracetamol/ibuprofen) may help relieve fever and aches — follow the dosage on the pack.",
            "Avoid contact with others to prevent spreading the illness.",
            "See a doctor if symptoms worsen, breathing becomes difficult, or fever persists beyond 3–4 days.",
        ],
        "urgency": "moderate",
    },
    "Cold": {
        "icon": "🤒",
        "general_info": "The common cold is a viral infection of the upper respiratory tract. It is usually mild and resolves on its own within 7–10 days.",
        "next_steps": [
            "Rest and drink plenty of fluids.",
            "Saline nasal drops or steam inhalation may ease congestion.",
            "Consult a pharmacist or doctor before taking any medication.",
            "See a doctor if symptoms persist beyond 10 days or are severe.",
        ],
        "urgency": "low",
    },
    "Gastroenteritis": {
        "icon": "🤢",
        "general_info": "Gastroenteritis (stomach flu) involves inflammation of the stomach and intestines, often causing nausea, vomiting, and diarrhoea.",
        "next_steps": [
            "Stay well hydrated — sip ORS (oral rehydration solution) or clear fluids frequently.",
            "Avoid solid foods until nausea settles, then introduce bland foods gradually.",
            "Seek medical attention if vomiting/diarrhoea is severe, blood is present, or symptoms last more than 48 hours.",
            "Wash hands thoroughly to avoid spreading infection.",
        ],
        "urgency": "moderate",
    },
    "Dengue": {
        "icon": "🦟",
        "general_info": "Dengue fever is a mosquito-borne viral disease. It can range from mild to severe. A rash, high fever, and severe headache/body ache are common features.",
        "next_steps": [
            "See a doctor immediately — dengue requires professional monitoring.",
            "Do NOT take ibuprofen or aspirin; use only paracetamol for fever if advised by a doctor.",
            "Stay hydrated and rest.",
            "Platelet count monitoring may be necessary.",
        ],
        "urgency": "high",
    },
    "Heart Issue (Possible)": {
        "icon": "❤️",
        "general_info": "Symptoms such as chest pain, shortness of breath, and dizziness together can sometimes indicate a cardiac-related issue.",
        "next_steps": [
            "If chest pain is severe, spreads to the arm/jaw, or is accompanied by sweating — call emergency services (112) immediately.",
            "Do not ignore these symptoms. Seek medical evaluation as soon as possible.",
            "Avoid strenuous activity until you have been assessed by a doctor.",
        ],
        "urgency": "emergency",
    },
    "Asthma / Respiratory Issue": {
        "icon": "💨",
        "general_info": "Recurring shortness of breath and cough may be related to asthma or another respiratory condition.",
        "next_steps": [
            "See a doctor for a proper diagnosis and lung-function assessment.",
            "If you have a prescribed inhaler, use it as directed.",
            "Avoid known triggers such as dust, smoke, or strong odours.",
            "Call emergency services if breathing becomes severely difficult.",
        ],
        "urgency": "moderate",
    },
    "Migraine": {
        "icon": "🧠",
        "general_info": "Migraine is a neurological condition commonly characterised by intense, throbbing headache, often accompanied by nausea and sensitivity to light.",
        "next_steps": [
            "Rest in a quiet, dark room.",
            "Stay hydrated.",
            "Over-the-counter pain relief may help in mild cases — follow dosage instructions.",
            "See a doctor if headaches are frequent, very severe, or accompanied by visual disturbances or neurological symptoms.",
        ],
        "urgency": "low",
    },
    "Food Poisoning": {
        "icon": "🍽️",
        "general_info": "Food poisoning occurs after consuming contaminated food or water, causing nausea, vomiting, diarrhoea, and stomach cramps.",
        "next_steps": [
            "Drink plenty of fluids and ORS to prevent dehydration.",
            "Avoid solid food until vomiting has settled.",
            "Seek medical attention if symptoms are severe, if blood is present in vomit/stool, or if the affected person is very young, elderly, or immunocompromised.",
        ],
        "urgency": "moderate",
    },
    "Allergy": {
        "icon": "🌿",
        "general_info": "Allergies occur when the immune system reacts to a substance such as pollen, dust, food, or pet dander.",
        "next_steps": [
            "Try to identify and avoid the trigger.",
            "Over-the-counter antihistamines may provide relief — follow the instructions on the pack.",
            "Consult a doctor for persistent or worsening allergic reactions.",
            "Seek emergency care immediately if you experience throat swelling or difficulty breathing.",
        ],
        "urgency": "low",
    },
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/symptom")
def symptom():
    features = _features if _features else []
    return render_template("symptom.html", features=features, model_ready=(_model is not None))


@app.route("/api/predict", methods=["POST"])
def predict():
    if _model is None or _features is None:
        return jsonify({"error": "Model not trained yet. Please run: python train_model.py"}), 503

    data = request.get_json(force=True)
    selected = data.get("symptoms", [])

    if not selected:
        return jsonify({"error": "Please select at least one symptom."}), 400

    # Build feature vector
    import numpy as np
    vector = [[1 if f in selected else 0 for f in _features]]

    prediction = _model.predict(vector)[0]
    proba      = _model.predict_proba(vector)[0]
    confidence = round(float(max(proba)) * 100, 1)

    advice = CONDITION_ADVICE.get(prediction, {
        "icon": "ℹ️",
        "general_info": "No specific information available for this result.",
        "next_steps": ["Please consult a qualified healthcare professional for proper evaluation."],
        "urgency": "moderate",
    })

    return jsonify({
        "condition":  prediction,
        "confidence": confidence,
        "advice":     advice,
        "disclaimer": (
            "DISCLAIMER: This tool is for general educational purposes only. "
            "It is NOT a medical diagnosis. Always consult a qualified doctor or "
            "healthcare professional for any health concerns."
        ),
    })


@app.route("/hospital")
def hospital():
    cities = []
    if os.path.exists(HOSPITAL_CSV):
        df = pd.read_csv(HOSPITAL_CSV)
        cities = sorted(df["city"].dropna().unique().tolist())
    return render_template("hospital.html", cities=cities)


@app.route("/api/hospitals")
def get_hospitals():
    if not os.path.exists(HOSPITAL_CSV):
        return jsonify({"error": "Hospital data not found."}), 404

    df = pd.read_csv(HOSPITAL_CSV)
    city  = request.args.get("city",  "").strip().lower()
    query = request.args.get("query", "").strip().lower()

    if city:
        df = df[df["city"].str.lower() == city]
    if query:
        df = df[
            df["name"].str.lower().str.contains(query, na=False)
            | df["type"].str.lower().str.contains(query, na=False)
        ]

    df = df.fillna("")
    return jsonify(df.to_dict(orient="records"))


@app.route("/schemes")
def schemes():
    return render_template("schemes.html")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
