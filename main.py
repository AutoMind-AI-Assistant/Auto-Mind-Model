import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Vehicle Diagnostics API",
    description="API to predict vehicle health status based on sensor inputs.",
    version="1.0.0"
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

class VehicleSensorData(BaseModel):
    engine_rpm: float = Field(..., example=1200.0, description="Engine RPM")
    oil_pressure: float = Field(..., example=45.0, description="Oil pressure (PSI)")
    fuel_level: float = Field(..., example=85.0, description="Fuel level percentage (0-100)")
    coolant_temp: float = Field(..., example=88.0, description="Coolant temperature (Celsius)")
    battery_voltage: float = Field(..., example=14.1, description="Battery voltage (V)")
    mileage: float = Field(..., example=45000.0, description="Mileage (miles/km)")
    days_since_service: float = Field(..., example=120.0, description="Days since last service")

class PredictionResponse(BaseModel):
    health_status: str
    probabilities: dict
    sensor_summary: dict

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Vehicle Diagnostics API!",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: VehicleSensorData):
    if model is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded/available.")
    
    try:
        # Convert Pydantic model to dictionary
        sensor_dict = {
            'engine_rpm': data.engine_rpm,
            'oil_pressure': data.oil_pressure,
            'fuel_level': data.fuel_level,
            'coolant_temp': data.coolant_temp,
            'battery_voltage': data.battery_voltage,
            'mileage': data.mileage,
            'days_since_service': data.days_since_service
        }
        
        # Convert to DataFrame as expected by the model
        df = pd.DataFrame([sensor_dict])
        
        # Prediction
        prediction_idx = model.predict(df)[0]
        
        # Get class probabilities
        probabilities = model.predict_proba(df)[0]
        
        # Map prediction index to status
        status_map = {0: 'Healthy', 1: 'Attention Required', 2: 'Critical'}
        class_names = ['Healthy', 'Attention Required', 'Critical']
        
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
