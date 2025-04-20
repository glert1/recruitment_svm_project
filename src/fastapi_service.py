from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="SVM Recruitment Predictor API")


BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")


try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please train the model first.")

try:
    scaler = joblib.load(SCALER_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Scaler not found at {SCALER_PATH}. Please save it during the training.")


class CandidateFeatures(BaseModel):
    tecrube_yili: float
    teknik_puan: float


@app.post("/predict")
def predict_candidate(features: CandidateFeatures):
    if not (0 <= features.tecrube_yili <= 10 and 0 <= features.teknik_puan <= 100):
        raise HTTPException(status_code=400, detail="Values out of range.")

    X_input = np.array([[features.tecrube_yili, features.teknik_puan]])
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]

    result = "ACCEPTED" if prediction == 0 else "REJECTED"
    return {
        "prediction": int(prediction),
        "result": result
    }
