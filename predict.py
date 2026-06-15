import joblib
import pandas as pd
import numpy as np
import os

# Configuration
MODEL_FILE = 'vehicle_model.pkl'

def predict_vehicle_health(sensor_data):
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} NOT found. Please run 'train_model.py' first.")
        return None

    # Load model
    model = joblib.load(MODEL_FILE)
    
    # Preprocessing (ensure dataframe structure)
    df = pd.DataFrame([sensor_data])
    
    # Prediction
    prediction = model.predict(df)
    
    status_map = {0: 'Healthy', 1: 'Attention Required', 2: 'Critical'}
    
    print("\nVehicle Sensors Analysis:")
    for key, value in sensor_data.items():
        print(f"- {key}: {value}")
        
    print(f"\nFinal Health Status: {status_map[prediction[0]].upper()}")
    return status_map[prediction[0]]

if __name__ == "__main__":
    # Example Sensor Input (Healthy)
    healthy_test = {
        'engine_rpm': 1200,
        'oil_pressure': 45.0,
        'fuel_level': 85.0,
        'coolant_temp': 88.0,
        'battery_voltage': 14.1,
        'mileage': 45000,
        'days_since_service': 120
    }
    
    # Example Sensor Input (Attention Required)
    warning_test = {
        'engine_rpm': 2500,
        'oil_pressure': 40.0,
        'fuel_level': 35.0,
        'coolant_temp': 98.0,
        'battery_voltage': 11.5,
        'mileage': 190000,
        'days_since_service': 450
    }
    
    # Example Sensor Input (Critical)
    critical_test = {
        'engine_rpm': 5000,
        'oil_pressure': 18.0,
        'fuel_level': 20.0,
        'coolant_temp': 112.0,
        'battery_voltage': 12.5,
        'mileage': 210000,
        'days_since_service': 500
    }
    
    print("Testing Healthy Car:")
    predict_vehicle_health(healthy_test)
    
    print("\n" + "="*30 + "\n")
    
    print("Testing Warning Car:")
    predict_vehicle_health(warning_test)
    
    print("\n" + "="*30 + "\n")
    
    print("Testing Critical Car:")
    predict_vehicle_health(critical_test)
