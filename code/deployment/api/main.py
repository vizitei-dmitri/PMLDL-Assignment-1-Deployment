import os, joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import Literal

app = FastAPI(title="Bank Marketing API")
class BankSample(BaseModel):
    age: conint(ge=18, le=95)
    job: Literal["admin.","blue-collar","entrepreneur","housemaid","management",
                 "retired","self-employed","services","student","technician",
                 "unemployed","unknown"]
    marital: Literal["married","single","divorced","unknown"]
    education: Literal["primary","secondary","tertiary","unknown"]
    balance: confloat(ge=-1e7, le=1e7)
    housing: bool
    loan: bool
    contact: Literal["cellular","telephone","unknown"]
    month: Literal["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    campaign: conint(ge=1, le=30)

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model_pipeline.joblib")
_model = None

@app.on_event("startup")
def load_model():
    global _model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(sample: BankSample):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = pd.DataFrame([sample.dict()])
    try:
        pred = int(_model.predict(df)[0])
        proba = float(_model.predict_proba(df)[0,1]) if hasattr(_model, "predict_proba") else None
        return {"prediction": pred, "proba_yes": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
