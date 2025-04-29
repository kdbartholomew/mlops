# housingApp.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Housing Price Predictor")

class HousingRequest(BaseModel):
    features: list  # expects a list of 8 floats

@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("housing_model.joblib")

@app.post("/predict")
def predict(data: HousingRequest):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)[0]
    return {"Prediction": prediction}