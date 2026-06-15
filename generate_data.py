import pandas as pd
import numpy as np
import os

# Configuration
NUM_SAMPLES = 5000
OUTPUT_FILE = 'vehicle_data.csv'

def generate_vehicle_data(num_samples=NUM_SAMPLES):
    np.random.seed(42)
    
    # Feature Generation
    engine_rpm = np.random.uniform(800, 5500, num_samples)
    oil_pressure = np.random.uniform(15, 75, num_samples)
    fuel_level = np.random.uniform(5, 100, num_samples)
    coolant_temp = np.random.uniform(70, 115, num_samples)
    battery_voltage = np.random.uniform(11.0, 14.8, num_samples)
    mileage = np.random.uniform(0, 250000, num_samples)
    days_since_service = np.random.uniform(0, 730, num_samples) # up to 2 years

    # Define targets based on conditions (health_status: 0: Healthy, 1: Attention, 2: Critical)
    health_status = []
    
    for i in range(num_samples):
        score = 0
        
        # Overheating
        if coolant_temp[i] > 105:
            score += 2
        elif coolant_temp[i] > 100:
            score += 1
            
        # Low Oil Pressure
        if oil_pressure[i] < 20:
            score += 2
        elif oil_pressure[i] < 30:
            score += 1
            
        # Weak Battery
        if battery_voltage[i] < 12.0:
            score += 1
            
        # High Wear (Mileage/Service)
        if mileage[i] > 180000 or days_since_service[i] > 400:
            score += 1
            
        # High RPM + High Temp = Critical
        if engine_rpm[i] > 4500 and coolant_temp[i] > 100:
            score += 2
            
        # Assign final status
        if score >= 3:
            status = 2 # Critical
        elif score >= 1:
            status = 1 # Attention Required
        else:
            status = 0 # Healthy
            
        health_status.append(status)

    data = {
        'engine_rpm': engine_rpm,
        'oil_pressure': oil_pressure,
        'fuel_level': fuel_level,
        'coolant_temp': coolant_temp,
        'battery_voltage': battery_voltage,
        'mileage': mileage,
        'days_since_service': days_since_service,
        'health_status': health_status
    }
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated with {num_samples} records and saved as '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    generate_vehicle_data()
