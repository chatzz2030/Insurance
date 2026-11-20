
I built an end-to-end AI-powered Insurance Fraud Detection web application called FraudShield AI. The system takes a live insurance claim submitted through a web form, sends it to a FastAPI backend, runs it through a pre-trained XGBoost classifier, and instantly predicts whether the claim is fraudulent or genuine — along with a fraud probability score and an explainable AI breakdown using SHAP values that shows exactly which factors drove the model's decision. The entire pipeline — from data preprocessing in Jupyter Notebook, model training, saving trained artifacts as .pkl files, building a REST API, to deploying a premium responsive frontend — was built by me from scratch."

Tech Stack in one line: Python + XGBoost + SHAP + FastAPI + Pydantic + Joblib + Vanilla HTML/CSS/JS

insurance/
├── fraud_oracle.csv              ← Dataset (raw data, ~3.6 MB)
├── notebooks/
│   └── Insurance_Fraud.ipynb    ← Full EDA + model training notebook
├── models/
│   ├── insurance_fraud_xgboost.pkl   ← Trained XGBoost model (~764 KB)
│   ├── scaler.pkl                    ← StandardScaler artifact (~5 KB)
│   └── feature_names.pkl             ← Ordered list of features (~2 KB)
├── backend/
│   ├── app.py               ← FastAPI app + API endpoints
│   ├── predict.py           ← Core prediction + SHAP logic
│   ├── preprocessing.py     ← Transforms raw input into model-ready features
│   ├── mappings.py          ← All encoding maps (binary, ordinal, one-hot)
│   ├── model_loader.py      ← Loads .pkl files at startup (once)
│   ├── utils.py             ← Validation helpers (valid values per field)
│   └── requirements.txt     ← All Python dependencies
├── frontend/
│   ├── index.html           ← Full UI (form + results panel)
│   ├── style.css            ← Premium dark/glassmorphism design (~20 KB)
│   └── script.js            ← API calls, SHAP visualization, form logic
└── run_app.bat              ← One-click startup script (Windows)
