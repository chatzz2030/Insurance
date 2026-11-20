
I built an end-to-end AI-powered Insurance Fraud Detection web application called FraudShield AI. The system takes a live insurance claim submitted through a web form, sends it to a FastAPI backend, runs it through a pre-trained XGBoost classifier, and instantly predicts whether the claim is fraudulent or genuine — along with a fraud probability score and an explainable AI breakdown using SHAP values that shows exactly which factors drove the model's decision. The entire pipeline — from data preprocessing in Jupyter Notebook, model training, saving trained artifacts as .pkl files, building a REST API, to deploying a premium responsive frontend — was built by me from scratch."

Tech Stack in one line: Python + XGBoost + SHAP + FastAPI + Pydantic + Joblib + Vanilla HTML/CSS/JS
