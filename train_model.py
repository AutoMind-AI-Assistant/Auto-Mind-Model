import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
INPUT_FILE = 'vehicle_data.csv'
MODEL_FILE = 'vehicle_model.pkl'
CLASS_NAMES = ['Healthy', 'Warning', 'Critical']

def train_and_evaluate():
    # ----------------------------------------------------
    # Part 1: Load and Prepare Data
    # ----------------------------------------------------
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} NOT found. Please run 'generate_data.py' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Preprocessing
    X = df.drop('health_status', axis=1)
    y = df['health_status']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data loaded! Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # ----------------------------------------------------
    # Part 2: Train the Model with Pipeline
    # ----------------------------------------------------
    print("\nTraining Random Forest Classifier Pipeline...")
    
    # Define preprocessing for categorical columns
    categorical_features = ['component']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Keep numeric features as they are
    numeric_features = ['current_km', 'current_months', 'condition_metric_value']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', 'passthrough', numeric_features)
        ])
        
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    # Generate Predictions
    y_pred = model.predict(X_test)
    print("Model training complete!")

    # ----------------------------------------------------
    # Part 3: Overall Accuracy & Classification Report
    # ----------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ----------------------------------------------------
    # Part 4: Confusion Matrix Visualization
    # ----------------------------------------------------
    print("\nGenerating Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', 
                xticklabels=CLASS_NAMES, 
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Labels', fontsize=12)
    plt.ylabel('True Labels', fontsize=12)
    plt.tight_layout()
    
    # Save the plot as an image
    cm_plot_path = 'confusion_matrix.png'
    plt.savefig(cm_plot_path)
    print(f"Confusion Matrix plot saved to '{cm_plot_path}'.")

    # ----------------------------------------------------
    # Part 5: Save Model
    # ----------------------------------------------------
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved successfully as '{MODEL_FILE}'.")

if __name__ == "__main__":
    train_and_evaluate()
