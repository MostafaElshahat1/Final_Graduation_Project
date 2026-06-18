import joblib
from app.utils import preprocess_input, get_risk_level

class ModelInference:
    def __init__(self, model_path: str = "models/mental_health_model.pkl"):
        self.model = joblib.load(model_path)

    def predict_risk(self, raw_inputs: dict) -> dict:
        input_df = preprocess_input(raw_inputs)
        
        prediction = self.model.predict(input_df)
        risk_score = float(prediction[0])
        
        risk_score = max(0.0, min(1.0, risk_score))
        
        risk_level = get_risk_level(risk_score)
        
        return {
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level
        }

# Singleton instance
inference_engine = ModelInference()