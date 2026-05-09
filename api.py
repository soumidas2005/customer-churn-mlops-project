from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("finalized_churn_forecaster.pkl")

# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Churn API Running"}

# -----------------------------
# PREDICT ENDPOINT
# -----------------------------
@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Align features
    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])

    return {
        "prediction": pred,
        "churn_probability": prob
    }

# -----------------------------
# EXPLAIN ENDPOINT (SIMPLE)
# -----------------------------
@app.post("/explain")
def explain(data: dict):

    df = pd.DataFrame([data])

    expected_cols = model.feature_names_in_

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_cols]

    importances = model.feature_importances_

    feature_importance = dict(zip(expected_cols, importances))

    # Top 5 important features
    top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "top_features": top_features
    }

# -----------------------------
# SEGMENT ENDPOINT (SIMPLE LOGIC)
# -----------------------------
@app.post("/segment")
def segment(data: dict):

    recency = data.get("Recency", 0)
    monetary = data.get("Monetary", 0)

    if monetary > 80:
        segment = "High Value"
    elif recency < 10:
        segment = "Active"
    else:
        segment = "At Risk"

    return {"segment": segment}