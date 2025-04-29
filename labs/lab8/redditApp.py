# redditApp.py
from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI(
    title="Reddit Comment Classifier",
    description="Classify Reddit comments as either 1 = Remove or 0 = Do Not Remove.",
    version="0.1",
)

class request_body(BaseModel):
    reddit_comment: str

@app.on_event('startup')
def load_artifacts():
    global model_pipeline
    model_pipeline = joblib.load("reddit_model_pipeline.joblib")

@app.get('/')
def index():
    return {'message': 'Reddit app is running'}
@app.post('/predict')
def predict(data: request_body):
    X = [data.reddit_comment]
    predictions = model_pipeline.predict_proba(X)
    # predictions is a 2D array: [[prob_class_0, prob_class_1]]
    predicted_class = int(predictions[0].argmax())
    return {'Prediction': predicted_class}