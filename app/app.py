from fastapi import FastAPI, BackgroundTasks
from prometheus_client import Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import joblib
import pandas as pd
import threading
import os
from train_model import train_and_save_model  # Import retraining function

# Initialize FastAPI
app = FastAPI()

# Load the trained model
MODEL_PATH = "model.pkl"
model = joblib.load(MODEL_PATH)

# Prometheus Metrics
prediction_counter = Counter("predictions_total", "Total number of predictions made")
api_requests = Counter("api_requests_total", "Total API requests received")
model_accuracy = Gauge("model_accuracy", "Current accuracy of the model")

# Set a minimum accuracy threshold for retraining
ACCURACY_THRESHOLD = 0.80  # Adjust based on requirements


@app.get("/")
def home():
    """Welcome Message"""
    api_requests.inc()
    return {"message": "Titanic Survival Prediction API"}


@app.post("/predict")
def predict(Pclass: int, Sex: int, Age: float, Fare: float):
    """Make Predictions and Monitor API Usage"""
    global model

    # Convert input to DataFrame
    input_data = pd.DataFrame([[Pclass, Sex, Age, Fare]], 
                              columns=["Pclass", "Sex", "Age", "Fare"])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Log prediction count
    prediction_counter.inc()

    return {"survived": bool(prediction)}


@app.get("/metrics")
def metrics():
    """Expose Metrics for Prometheus"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def retrain_model():
    """Background Task: Retrain Model if Accuracy Drops"""
    global model
    print("🔄 Retraining model...")
    accuracy = train_and_save_model(MODEL_PATH)  # Retrain model
    print(f"✅ Model retrained. New Accuracy: {accuracy}")

    # Update model accuracy metric
    model_accuracy.set(accuracy)

    # Reload the updated model
    model = joblib.load(MODEL_PATH)


@app.get("/check-retrain")
def check_retrain(background_tasks: BackgroundTasks):
    """Check Model Performance and Trigger Retraining if Needed"""
    accuracy = model_accuracy._value.get()  # Get current metric value

    if accuracy < ACCURACY_THRESHOLD:
        print("⚠️ Accuracy below threshold! Triggering model retraining...")
        background_tasks.add_task(retrain_model)
        return {"status": "Retraining triggered"}
    else:
        return {"status": "Model accuracy is stable"}
