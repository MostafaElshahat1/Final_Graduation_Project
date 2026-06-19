from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.inference import inference_engine

app = FastAPI(
    title="Mental Health & Bullying Monitoring API",
    description="API for tracking psychological risks and bullying impacts in children.",
    version="2.0"
)

class SurveyPayload(BaseModel):
    PHQ_1: int = Field(..., ge=0, le=3)
    PHQ_2: int = Field(..., ge=0, le=3)
    PHQ_3: int = Field(..., ge=0, le=3)
    PHQ_4: int = Field(..., ge=0, le=3)
    PHQ_5: int = Field(..., ge=0, le=3)
    PHQ_6: int = Field(..., ge=0, le=3)
    PHQ_7: int = Field(..., ge=0, le=3)
    PHQ_8: int = Field(..., ge=0, le=3)
    
    GAD_1: int = Field(..., ge=0, le=3)
    GAD_2: int = Field(..., ge=0, le=3)
    GAD_3: int = Field(..., ge=0, le=3)
    GAD_4: int = Field(..., ge=0, le=3)
    GAD_5: int = Field(..., ge=0, le=3)
    GAD_6: int = Field(..., ge=0, le=3)
    GAD_7: int = Field(..., ge=0, le=3)
    
    BL_1: int = Field(..., ge=0, le=3)
    BL_2: int = Field(..., ge=0, le=3)
    BL_5: int = Field(..., ge=0, le=3)
    BL_9: int = Field(..., ge=0, le=3)

@app.get("/")
def read_root():
    return {"status": "Healthy", "message": "Model API is up and running."}

@app.post("/predict")
def predict(payload: SurveyPayload):
    try:
        input_data = payload.dict()
        result = inference_engine.predict_risk(input_data)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")