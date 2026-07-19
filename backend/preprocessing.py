# ============================================================
# preprocessing.py
# Converts raw API input into the feature matrix expected
# by the XGBoost model, exactly matching the notebook pipeline.
#
# Pipeline (Insurance_Fraud.ipynb):
#   1. Drop PolicyNumber, RepNumber (already absent from API input)
#   2. Replace Age==0 with median, MonthClaimed/'0' with mode,
#      DayOfWeekClaimed/'0' with mode  (handled via validation)
#   3. LabelEncode binary columns
#   4. Ordinal-encode multi-category columns
#   5. pd.get_dummies(drop_first=True) on remaining object cols
#   6. Align columns with feature_names.pkl (add missing=0, drop extra)
#   7. Scale with scaler.pkl
# ============================================================

import pandas as pd
from model_loader import feature_names, scaler
from mappings import BINARY_ENCODING_MAP, ORDINAL_ENCODING_MAP, ONE_HOT_COLS


def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Transform a single raw input dict into a scaled DataFrame
    aligned with feature_names expected by the model.

    Parameters
    ----------
    user_input : dict
        Raw field values from the API request.

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame ready to pass to model.predict().
    """
    # --- Step 1: Start with a single-row DataFrame ---
    row = dict(user_input)  # shallow copy to avoid mutating original

    # --- Step 2: Apply binary (LabelEncoder-equivalent) encoding ---
    for col, mapping in BINARY_ENCODING_MAP.items():
        if col in row:
            row[col] = mapping[row[col]]

    # --- Step 3: Apply ordinal encoding ---
    for col, mapping in ORDINAL_ENCODING_MAP.items():
        if col in row:
            row[col] = mapping[row[col]]

    # --- Step 4: Build DataFrame so get_dummies can work ---
    input_df = pd.DataFrame([row])

    # --- Step 5: One-hot encode remaining string/object columns ---
    # Only encode columns that are still strings (categorical)
    ohe_cols_present = [
        c for c in ONE_HOT_COLS if c in input_df.columns
        and input_df[c].dtype == object
    ]
    if ohe_cols_present:
        input_df = pd.get_dummies(
            input_df,
            columns=ohe_cols_present,
            drop_first=True,
            dtype=int,
        )

    # --- Step 6: Align columns with training feature set ---
    # Add missing columns (set to 0) and remove extra columns,
    # then reorder exactly as feature_names.pkl specifies.
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]  # reorder + drop extras

    # --- Step 7: Scale using the saved StandardScaler ---
    scaled_array = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=feature_names)

    return scaled_df