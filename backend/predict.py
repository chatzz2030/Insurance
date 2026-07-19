# ============================================================
# predict.py
# Core prediction logic: run preprocessed input through the
# loaded XGBoost model and return a structured result.
# ============================================================

import shap
from model_loader import model
from preprocessing import preprocess_input

# Initialize SHAP explainer globally for efficiency
explainer = shap.TreeExplainer(model)

def run_prediction(user_input: dict) -> dict:
    """
    Run the full prediction pipeline for a single claim.

    Parameters
    ----------
    user_input : dict
        Raw field values from the API request (validated by Pydantic).

    Returns
    -------
    dict
        {
            "prediction": "Fraud" | "Not Fraud",
            "probability": float,
            "confidence": str,
            "shap_explanation": list of dicts,
            "explanation": list of strings
        }
    """
    # 1. Preprocess input into model-ready feature matrix
    processed = preprocess_input(user_input)

    # 2. Predict class (0 = Not Fraud, 1 = Fraud)
    predicted_class = int(model.predict(processed)[0])

    # 3. Get fraud probability from predict_proba (column index 1)
    fraud_probability = float(model.predict_proba(processed)[0][1])

    # 4. Generate SHAP values
    shap_values = explainer.shap_values(processed)[0]

    # 5. Aggregate SHAP values to original features
    shap_dict = {}
    for col, shap_val in zip(processed.columns, shap_values):
        original_feature = col
        # Sort keys by length descending to prevent substring match issues (e.g. Age vs AgeOfVehicle)
        for key in sorted(user_input.keys(), key=len, reverse=True):
            if col == key or col.startswith(f"{key}_"):
                original_feature = key
                break
        
        if original_feature not in shap_dict:
            shap_dict[original_feature] = 0.0
        shap_dict[original_feature] += float(shap_val)

    # 6. Build shap_explanation list
    shap_explanation = []
    for feature, shap_val in shap_dict.items():
        if feature in user_input:
            input_val = str(user_input[feature])
            impact = "positive" if shap_val > 0 else "negative"
            shap_explanation.append({
                "feature": feature,
                "value": input_val,
                "shap_value": round(shap_val, 4),
                "impact": impact
            })

    # 7. Sort explanation based on prediction outcome
    # For Fraud, prioritize features that increased risk (most positive). 
    # For Genuine, prioritize features that reduced risk (most negative).
    if predicted_class == 1:
        shap_explanation.sort(key=lambda x: x["shap_value"], reverse=True)
    else:
        shap_explanation.sort(key=lambda x: x["shap_value"])
    
    top_5 = shap_explanation[:5]
    
    # Generate human-readable explanation
    explanation = []
    if predicted_class == 1:
        explanation.append("AI detected higher fraud risk because:")
        for item in top_5:
            if item["shap_value"] > 0:
                explanation.append(f"- {item['feature']} ({item['value']}) increased risk")
            else:
                explanation.append(f"- {item['feature']} ({item['value']}) reduced risk")
    else:
        explanation.append("AI classified this claim as genuine because:")
        for item in top_5:
            if item["shap_value"] < 0:
                explanation.append(f"- {item['feature']} ({item['value']}) reduced fraud risk")
            else:
                explanation.append(f"- {item['feature']} ({item['value']}) slightly increased risk")

    # 8. Format output
    label = "Fraud" if predicted_class == 1 else "Not Fraud"
    confidence = f"{fraud_probability * 100:.0f}%"

    return {
        "prediction": label,
        "probability": round(fraud_probability, 4),
        "confidence": confidence,
        "shap_explanation": shap_explanation,
        "explanation": explanation,
    }

