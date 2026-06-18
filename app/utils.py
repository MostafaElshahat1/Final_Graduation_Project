import json
import pandas as pd

with open("models/features.json", "r") as f:
    FEATURES_ORDER = json.load(f)["features"]

def preprocess_input(inputs: dict) -> pd.DataFrame:

    for feature in FEATURES_ORDER:
        if feature not in inputs:
            raise ValueError(f"Missing required survey question: {feature}")
        
        if not (0 <= inputs[feature] <= 3):
            raise ValueError(f"Invalid value for {feature}. Must be between 0 and 3.")
            
    ordered_data = {feat: [inputs[feat]] for feat in FEATURES_ORDER}
    return pd.DataFrame(ordered_data)

def get_risk_level(score: float) -> str:

    if score < 0.25:
        return "Normal"
    elif score < 0.60:
        return "Moderate"
    else:
        return "Severe"