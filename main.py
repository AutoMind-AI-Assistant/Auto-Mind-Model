import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Vehicle Maintenance API",
    description="API to predict vehicle maintenance requirements based on component rules.",
    version="1.1.0"
)

MODEL_FILE = 'vehicle_model.pkl'

# Load the trained model at startup
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Warning: {MODEL_FILE} not found. Please run training script first.")
    model = None

class ComponentData(BaseModel):
    component: str = Field(..., example="Spark Plug", description="Component name")
    current_km: float = Field(..., example=75000.0, description="Current mileage of component")
    current_months: float = Field(..., example=45.0, description="Months since component was installed/serviced")
    condition_metric_value: float = Field(..., example=4.0, description="Current value of the specific condition metric (e.g. misfire count)")

class PredictionResponse(BaseModel):
    health_status: str
    probabilities: dict
    sensor_summary: dict

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Vehicle Maintenance API!",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ComponentData):
    if model is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded/available.")
    
    try:
        # Convert Pydantic model to dictionary
        sensor_dict = {
            'component': data.component,
            'current_km': data.current_km,
            'current_months': data.current_months,
            'condition_metric_value': data.condition_metric_value
        }
        
        # Convert to DataFrame as expected by the model pipeline
        df = pd.DataFrame([sensor_dict])
        
        # Prediction
        prediction_idx = model.predict(df)[0]
        
        # Get class probabilities
        probabilities = model.predict_proba(df)[0]
        
        # Map prediction index to status
        status_map = {0: 'Healthy', 1: 'Warning', 2: 'Critical'}
        class_names = ['Healthy', 'Warning', 'Critical']
        
        health_status = status_map.get(prediction_idx, "Unknown")
        
        # Convert float32/float64 to Python float for JSON serialization
        prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
        
        return PredictionResponse(
            health_status=health_status,
            probabilities=prob_dict,
            sensor_summary=sensor_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
