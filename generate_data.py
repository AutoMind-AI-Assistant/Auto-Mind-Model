import pandas as pd
import numpy as np
import os

# Configuration
NUM_SAMPLES = 5000
OUTPUT_FILE = 'vehicle_data.csv'

def generate_vehicle_data(num_samples=NUM_SAMPLES):
    np.random.seed(42)
    
    rules = [
        {"component": "Spark Plug", "interval_km": 80000, "interval_months": 48, "warning_km": 5000, "warning_days": 30, "metric_name": "misfire_count_30_days", "critical_metric": 5, "metric_dir": "up"},
        {"component": "Ignition Coil", "interval_km": 120000, "interval_months": 72, "warning_km": 10000, "warning_days": 45, "metric_name": "misfire_count_30_days", "critical_metric": 8, "metric_dir": "up"},
        {"component": "CVT/AT Oil", "interval_km": 40000, "interval_months": 36, "warning_km": 3000, "warning_days": 30, "metric_name": "gearbox_oil_quality_score", "critical_metric": 40, "metric_dir": "down"},
        {"component": "Gearbox", "interval_km": 10000, "interval_months": 6, "warning_km": 1000, "warning_days": 15, "metric_name": "oil_leak_severity_score", "critical_metric": 60, "metric_dir": "up"},
        {"component": "Gearbox Filter", "interval_km": 60000, "interval_months": 48, "warning_km": 5000, "warning_days": 30, "metric_name": "filter_blockage_score", "critical_metric": 70, "metric_dir": "up"},
        {"component": "Front Brake Pad", "interval_km": 45000, "interval_months": 24, "warning_km": 3000, "warning_days": 30, "metric_name": "pad_thickness_mm", "critical_metric": 3, "metric_dir": "down"},
        {"component": "Rear Brake Pad", "interval_km": 60000, "interval_months": 36, "warning_km": 5000, "warning_days": 30, "metric_name": "pad_thickness_mm", "critical_metric": 3, "metric_dir": "down"}
    ]
    
    data = []
    
    for _ in range(num_samples):
        # Pick a random rule
        rule = np.random.choice(rules)
        
        # Generate random current values around the intervals to get a mix of healthy, warning, critical
        current_km = np.random.uniform(0, rule["interval_km"] * 1.5)
        current_months = np.random.uniform(0, rule["interval_months"] * 1.5)
        
        if rule["metric_dir"] == "up":
            condition_metric_value = np.random.uniform(0, rule["critical_metric"] * 1.5)
        else:
            # for thickness or quality where lower is worse
            condition_metric_value = np.random.uniform(0, rule["critical_metric"] * 3)
            
        # Determine Status
        # 0 = Healthy, 1 = Warning, 2 = Critical
        status = 0
        
        # Check Warning
        warning_km_thresh = rule["interval_km"] - rule["warning_km"]
        warning_months_thresh = rule["interval_months"] - (rule["warning_days"] / 30.0)
        
        is_warning_km = current_km >= warning_km_thresh
        is_warning_months = current_months >= warning_months_thresh
        
        if is_warning_km or is_warning_months:
            status = 1
            
        # Check Critical
        is_critical_km = current_km >= rule["interval_km"]
        is_critical_months = current_months >= rule["interval_months"]
        
        if rule["metric_dir"] == "up":
            is_critical_metric = condition_metric_value >= rule["critical_metric"]
        else:
            is_critical_metric = condition_metric_value <= rule["critical_metric"]
            
        if is_critical_km or is_critical_months or is_critical_metric:
            status = 2
            
        data.append({
            "component": rule["component"],
            "current_km": round(current_km, 2),
            "current_months": round(current_months, 2),
            "condition_metric_value": round(condition_metric_value, 2),
            "health_status": status
        })
        
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset generated with {num_samples} records and saved as '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    generate_vehicle_data()
