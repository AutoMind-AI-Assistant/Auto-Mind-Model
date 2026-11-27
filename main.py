import os
import joblib
import pandas as pd
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="AutoMind Vehicle Maintenance AI API",
    description="Machine Learning prediction engine for Toyota Axio and modern passenger vehicles.",
    version="2.1.0"
)

# Enable CORS for local web, emulator, and remote Expo devices
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, 'vehicle_model.pkl')

model = None
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        print(f"Successfully loaded model from {MODEL_FILE}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_FILE} not found.")

COMPONENT_METRIC_CONFIG = {
    'Front Brake Pad': {
        'metric_name': 'Pad Thickness',
        'unit': 'mm',
        'nominal_range': (2.0, 9.0),
        'typical_lifecycle_km': 45000,
        'description': 'Hydraulic disc friction pad wear'
    },
    'Rear Brake Pad': {
        'metric_name': 'Pad Thickness',
        'unit': 'mm',
        'nominal_range': (2.0, 9.0),
        'typical_lifecycle_km': 65000,
        'description': 'Rear braking system lining thickness'
    },
    'CVT/AT Oil': {
        'metric_name': 'Fluid Degradation Index',
        'unit': 'score',
        'nominal_range': (0.0, 100.0),
        'typical_lifecycle_km': 50000,
        'description': 'Transmission fluid thermal and shear oxidation'
    },
    'Spark Plug': {
        'metric_name': 'Electrode Gap Wear',
        'unit': 'mm',
        'nominal_range': (1.0, 4.5),
        'typical_lifecycle_km': 80000,
        'description': 'Iridium/Platinum spark plug gap erosion'
    },
    'Ignition Coil': {
        'metric_name': 'Resistance Drift / Misfires',
        'unit': 'misfires/cycle',
        'nominal_range': (0.0, 8.0),
        'typical_lifecycle_km': 100000,
        'description': 'Secondary ignition coil winding resistance and misfire log'
    },
    'Gearbox': {
        'metric_name': 'Vibration / Slip Score',
        'unit': 'pts',
        'nominal_range': (0.0, 60.0),
        'typical_lifecycle_km': 15000,
        'description': 'CVT belt & pulley torque transfer slip telemetry'
    },
    'Gearbox Filter': {
        'metric_name': 'Pressure Differential',
        'unit': 'kPa',
        'nominal_range': (5.0, 85.0),
        'typical_lifecycle_km': 60000,
        'description': 'Transmission oil strainer & filter restriction level'
    }
}

class ComponentData(BaseModel):
    component: str = Field(..., example="Front Brake Pad")
    current_km: float = Field(..., example=32000.0)
    current_months: float = Field(..., example=18.0)
    condition_metric_value: float = Field(..., example=4.5)

class PredictionResponse(BaseModel):
    health_status: str
    probabilities: dict
    sensor_summary: dict
    metric_info: Optional[dict] = None
    recommendation: Optional[str] = None

class VehiclePredictionRequest(BaseModel):
    odometer_km: Optional[float] = Field(67000.0, example=67000.0)
    vehicle_months: Optional[float] = Field(36.0, example=36.0)
    components: Optional[List[ComponentData]] = None

@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "AutoMind Predictive Maintenance Engine",
        "model_loaded": model is not None,
        "model_path": MODEL_FILE,
        "supported_components": list(COMPONENT_METRIC_CONFIG.keys())
    }

def get_recommendation(component: str, status: str, metric_val: float) -> str:
    if status == 'Critical':
        return f"Urgent: Immediate replacement or service recommended for {component}."
    elif status == 'Warning':
        return f"Attention: {component} showing moderate wear. Inspect within next 3,000 - 5,000 km."
    else:
        return f"{component} is in good operational condition. Continue standard check intervals."

@app.post("/predict", response_model=PredictionResponse)
def predict(data: ComponentData):
    if model is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")
    
    try:
        sensor_dict = {
            'component': data.component,
            'current_km': float(data.current_km),
            'current_months': float(data.current_months),
            'condition_metric_value': float(data.condition_metric_value)
        }
        
        df = pd.DataFrame([sensor_dict])
        prediction_idx = int(model.predict(df)[0])
        probabilities = model.predict_proba(df)[0]
        
        status_map = {0: 'Healthy', 1: 'Warning', 2: 'Critical'}
        class_names = ['Healthy', 'Warning', 'Critical']
        health_status = status_map.get(prediction_idx, "Unknown")
        
        prob_dict = {class_names[i]: round(float(probabilities[i]), 3) for i in range(len(class_names))}
        cfg = COMPONENT_METRIC_CONFIG.get(data.component, {})
        
        return PredictionResponse(
            health_status=health_status,
            probabilities=prob_dict,
            sensor_summary=sensor_dict,
            metric_info=cfg,
            recommendation=get_recommendation(data.component, health_status, data.condition_metric_value)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/vehicle")
def predict_vehicle(req: VehiclePredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Machine learning model is not loaded.")
    
    odo = req.odometer_km or 67000.0
    v_months = req.vehicle_months or 36.0
    
    # If custom components were provided, use them; otherwise use realistic vehicle telemetry from dataset
    items = []
    if req.components and len(req.components) > 0:
        items = req.components
    else:
        km = max(1000.0, float(odo))
        months = max(3.0, float(v_months))
        # Front Brake Pad: lifecycle ~55,000 km. Healthy (0-29k), Warning (30k-49k), Critical (50k+)
        front_km = km
        front_pad_mm = max(1.2, round(9.0 - (km / 55000.0) * 7.0, 2))
        # Rear Brake Pad: lifecycle ~75,000 km. Healthy (0-49k), Warning (50k-74k), Critical (75k+)
        rear_km = km
        rear_pad_mm = max(1.2, round(9.0 - (km / 75000.0) * 7.0, 2))
        # Spark Plug: lifecycle ~90,000 km. Healthy (0-69k), Warning (70k-89k), Critical (90k+)
        spark_km = km
        spark_gap = min(7.0, round(1.0 + (km / 90000.0) * 4.5, 2))
        # Ignition Coil: lifecycle ~120,000 km. Healthy (0-99k), Warning (100k-129k), Critical (130k+)
        coil_km = km
        coil_metric = min(11.0, round(0.5 + (km / 120000.0) * 8.0, 2))
        # CVT/AT Oil: quality index 95.0 down to 10.0. Healthy (0-39k), Warning (40k-54k), Critical (55k+)
        cvt_km = km
        cvt_metric = max(10.0, round(95.0 - (km / 55000.0) * 70.0, 2))
        # Gearbox: slip score 10.0 up to 75.0. Healthy (0-39k), Warning (40k-79k), Critical (80k+)
        gearbox_km = min(km, 120000.0)
        gearbox_metric = min(80.0, round(10.0 + (km / 100000.0) * 55.0, 2))

        items = [
            ComponentData(
                component='Front Brake Pad',
                current_km=front_km,
                current_months=max(3.0, round((front_km / 35000.0) * 24.0, 1)),
                condition_metric_value=front_pad_mm
            ),
            ComponentData(
                component='Rear Brake Pad',
                current_km=rear_km,
                current_months=max(3.0, round((rear_km / 45000.0) * 30.0, 1)),
                condition_metric_value=rear_pad_mm
            ),
            ComponentData(
                component='Spark Plug',
                current_km=spark_km,
                current_months=max(4.0, round((spark_km / 60000.0) * 36.0, 1)),
                condition_metric_value=spark_gap
            ),
            ComponentData(
                component='Ignition Coil',
                current_km=coil_km,
                current_months=max(6.0, round((coil_km / 90000.0) * 48.0, 1)),
                condition_metric_value=coil_metric
            ),
            ComponentData(
                component='CVT/AT Oil',
                current_km=cvt_km,
                current_months=max(3.0, round((cvt_km / 40000.0) * 24.0, 1)),
                condition_metric_value=cvt_metric
            ),
            ComponentData(
                component='Gearbox',
                current_km=gearbox_km,
                current_months=min(months, 72.0),
                condition_metric_value=gearbox_metric
            ),
        ]
    
    results = []
    healthy_count = 0
    warning_count = 0
    critical_count = 0
    
    for item in items:
        pred_res = predict(item)
        status = pred_res.health_status
        if status == 'Healthy':
            healthy_count += 1
        elif status == 'Warning':
            warning_count += 1
        elif status == 'Critical':
            critical_count += 1
            
        results.append({
            'component': item.component,
            'health_status': status,
            'probabilities': pred_res.probabilities,
            'current_km': item.current_km,
            'current_months': item.current_months,
            'condition_metric_value': item.condition_metric_value,
            'metric_info': pred_res.metric_info,
            'recommendation': pred_res.recommendation
        })
    
    total = len(results)
    health_score = int(round(((healthy_count * 1.0 + warning_count * 0.65 + critical_count * 0.1) / max(total, 1)) * 100))
    overall_status = 'Critical' if critical_count > 0 else ('Warning' if warning_count > 0 else 'Healthy')
    
    return {
        'odometer_km': odo,
        'model_source': 'C:\\Users\\Lenovo\\.gemini\\antigravity\\scratch\\vehicle-diagnostics\\vehicle_model.pkl',
        'overall_health_score': health_score,
        'overall_status': overall_status,
        'summary': {
            'healthy_components': healthy_count,
            'warning_components': warning_count,
            'critical_components': critical_count,
            'total_components': total
        },
        'predictions': results
    }
