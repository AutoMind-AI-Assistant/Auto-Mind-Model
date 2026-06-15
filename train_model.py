import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
CLASS_NAMES = ['Healthy', 'Attention Required', 'Critical']

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
    # Part 2: Train the Model
    # ----------------------------------------------------
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
    plt.show()

    # ----------------------------------------------------
    # Part 5: Sample Prediction
    # ----------------------------------------------------
    idx = np.random.randint(0, len(X_test))
    sample_features = X_test.iloc[idx]
    true_label = y_test.iloc[idx]
    
    # Get prediction and probabilities
    prediction = model.predict([sample_features])[0]
    probabilities = model.predict_proba([sample_features])[0]
    
    print("\n--- Sample Prediction ---")
    print(f"Vehicle Data:\n{sample_features.to_dict()}")
    print(f"\nTrue Label: {CLASS_NAMES[true_label]}")
    print(f"Predicted Label: {CLASS_NAMES[prediction]}")
    
    print("\nPrediction Probabilities:")
    for cls_name, prob in zip(CLASS_NAMES, probabilities):
        print(f"  {cls_name}: {prob * 100:.2f}%")

    # ----------------------------------------------------
    # Part 6: Feature Importances Visualization
    # ----------------------------------------------------
    print("\nGenerating Feature Importances...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (What the model cares about most)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
    plt.tight_layout()
    
    # Save the plot as an image
    fi_plot_path = 'feature_importances.png'
    plt.savefig(fi_plot_path)
    print(f"Feature Importances plot saved to '{fi_plot_path}'.")
    plt.show()

    # ----------------------------------------------------
    # Part 7: Save Model
    # ----------------------------------------------------
    joblib.dump(model, MODEL_FILE)
    print(f"\nModel saved successfully as '{MODEL_FILE}'.")

if __name__ == "__main__":
    train_and_evaluate()
