# Health Setu 🏥
### *Connecting People to Better Healthcare*

A complete AI/ML-based healthcare web application built with **Python Flask**, **scikit-learn**, **HTML/CSS/JavaScript**. Runs fully offline — no paid APIs or external services required.

---

## Features

| Feature | Description |
|---|---|
| **Symptom Guidance** | Select symptoms → locally trained Random Forest classifier → general health information + next steps |
| **Hospital Finder** | Searchable hospital directory (CSV-based) with city and name filters |
| **Health Schemes** | Information about government healthcare schemes (Ayushman Bharat, CGHS, ESI, JSY, NHM) |

---

## Project Structure

```
Health_Setu/
├── app.py                   # Flask backend + API routes
├── train_model.py           # ML model training script
├── requirements.txt         # Python dependencies
├── data/
│   ├── symptoms_dataset.csv # Educational training dataset (60 samples, 8 conditions)
│   └── hospitals.csv        # Demo hospital directory
├── model/                   # Auto-created after training
│   ├── symptom_model.joblib
│   └── features.joblib
├── templates/
│   ├── index.html           # Home dashboard
│   ├── symptom.html         # Symptom Guidance page
│   ├── hospital.html        # Hospital Finder page
│   └── schemes.html         # Health Schemes page
├── static/
│   ├── style.css            # Main stylesheet
│   └── script.js            # Frontend JavaScript
└── README.md
```

---

## Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- pip

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the ML Model

```bash
python train_model.py
```

This reads `data/symptoms_dataset.csv`, trains a Random Forest classifier, and saves the model to `model/symptom_model.joblib`. You will see a classification report in the terminal.

### 4. Run the Application

```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

---

## Using the Application

### Home Dashboard
- Navigate to `http://localhost:5000`
- Three feature cards link to all main services.

### Symptom Guidance (`/symptom`)
1. Tick the symptoms you are experiencing.
2. Click **Get Guidance**.
3. The ML model returns a suggested condition category, confidence level, general information, and recommended next steps.
4. A medical disclaimer is always displayed.

### Hospital Finder (`/hospital`)
1. Select a city from the dropdown and/or type a name/type in the search box.
2. Click **Search** (or press Enter).
3. Matching hospital cards are displayed.
4. All records are labelled **"Demo Data – Verify before use"**.

### Health Schemes (`/schemes`)
- Browse five government healthcare schemes with links to official portals.
- All content is labelled for verification.

---

## ML Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest (100 estimators) |
| Dataset | 60 hand-crafted educational samples |
| Features | 15 binary symptom indicators |
| Conditions | Flu, Cold, Gastroenteritis, Dengue, Heart Issue (Possible), Asthma / Respiratory Issue, Migraine, Food Poisoning, Allergy |
| Purpose | **Educational demonstration only** |

> ⚠️ **This model is NOT medically validated.** It is trained on a tiny, synthetic dataset for demonstration purposes. Do not use it as a substitute for professional medical advice.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `GET` | `/symptom` | Symptom Guidance page |
| `POST` | `/api/predict` | `{"symptoms":["fever","cough",...]}` → prediction JSON |
| `GET` | `/hospital` | Hospital Finder page |
| `GET` | `/api/hospitals?city=Delhi&query=apollo` | Search hospitals |
| `GET` | `/schemes` | Health Schemes page |

---

## Disclaimers

- **Not a medical device.** Health Setu provides general information only.
- **Not a diagnosis tool.** Always consult a qualified healthcare professional.
- **Demo data.** Hospital records are for demonstration and have not been verified.
- **Scheme information.** Eligibility rules and benefits change; verify through official government portals.
- **Emergency:** Call **112** immediately for medical emergencies.

---

## License

This project is intended for educational and demonstration purposes only.
