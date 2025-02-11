from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message": "Titanic Survival Prediction API"}

@app.post("/predict")
def predict(Pclass: int, Sex: int, Age: float, Fare: float):
    # Convert input into DataFrame format
    input_data = pd.DataFrame([[Pclass, Sex, Age, Fare]], 
                              columns=["Pclass", "Sex", "Age", "Fare"])
    
    # Make prediction
    prediction = model.predict(input_data)[0]

    # Return response
    return {"survived": bool(prediction)}
