from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Initialize the model and vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
model = LogisticRegression()

# Sample training data (you should replace this with your actual model)
X_train = ["This is a sample text", "Another sample text"]
y_train = [0, 1]
vectorizer.fit(X_train)
model.fit(vectorizer.transform(X_train), y_train)

class TextInput(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Reddit Model API is running"}

@app.post("/predict")
def predict(input_data: TextInput):
    # Transform the input text
    X = vectorizer.transform([input_data.text])
    # Make prediction
    prediction = model.predict(X)[0]
    return {"prediction": int(prediction)} 