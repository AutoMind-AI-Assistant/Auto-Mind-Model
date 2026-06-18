import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Vehicle Maintenance - ML Model Training\n",
    "This notebook contains the complete pipeline to train and evaluate a Random Forest Classifier on component-level maintenance rules."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 1: Imports and Configuration**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import joblib\n",
    "import os\n",
    "import warnings\n",
    "\n",
    "# Suppress warnings for cleaner output\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Configuration\n",
    "INPUT_FILE = 'vehicle_data.csv'\n",
    "MODEL_FILE = 'vehicle_model.pkl'\n",
    "CLASS_NAMES = ['Healthy', 'Warning', 'Critical']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 2: Load and Prepare Data**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load dataset\n",
    "if not os.path.exists(INPUT_FILE):\n",
    "    print(f\"Error: {INPUT_FILE} NOT found. Please upload it to your workspace.\")\n",
    "else:\n",
    "    df = pd.read_csv(INPUT_FILE)\n",
    "    \n",
    "    # Preprocessing\n",
    "    X = df.drop('health_status', axis=1)\n",
    "    y = df['health_status']\n",
    "    \n",
    "    # Train-test split\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "    print(f\"Data loaded! Training samples: {len(X_train)} | Testing samples: {len(X_test)}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 3: Train the Model with Pipeline**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Training Random Forest Classifier Pipeline...\")\n",
    "\n",
    "# Define preprocessing for categorical columns\n",
    "categorical_features = ['component']\n",
    "categorical_transformer = OneHotEncoder(handle_unknown='ignore')\n",
    "\n",
    "# Keep numeric features as they are\n",
    "numeric_features = ['current_km', 'current_months', 'condition_metric_value']\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('cat', categorical_transformer, categorical_features),\n",
    "        ('num', 'passthrough', numeric_features)\n",
    "    ])\n",
    "    \n",
    "model = Pipeline(steps=[\n",
    "    ('preprocessor', preprocessor),\n",
    "    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))\n",
    "])\n",
    "\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Generate Predictions for the next cells\n",
    "y_pred = model.predict(X_test)\n",
    "print(\"Model training complete!\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 4: Overall Accuracy & Classification Report**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "accuracy = accuracy_score(y_test, y_pred)\n",
    "print(f\"Test accuracy: {accuracy * 100:.2f}%\\n\")\n",
    "\n",
    "print(\"Classification Report:\")\n",
    "print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 5: Confusion Matrix Visualization**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_test, y_pred)\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', \n",
    "            xticklabels=CLASS_NAMES, \n",
    "            yticklabels=CLASS_NAMES)\n",
    "plt.title('Confusion Matrix', fontsize=16)\n",
    "plt.xlabel('Predicted Labels', fontsize=12)\n",
    "plt.ylabel('True Labels', fontsize=12)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### **Cell 6: Save Model**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "joblib.dump(model, MODEL_FILE)\n",
    "print(f\"Model saved successfully as '{MODEL_FILE}'.\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("train_model.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Jupyter Notebook created successfully as 'train_model.ipynb'")
