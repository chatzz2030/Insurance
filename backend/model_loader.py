import joblib
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Models directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load trained model
model = joblib.load(os.path.join(MODEL_DIR, "insurance_fraud_xgboost.pkl"))

# Load scaler
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

# Load feature names
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))