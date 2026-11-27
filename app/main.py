"""FastAPI service for MP093: POST /predict and GET /health.

Run locally (from the repo root, after pip install -e .):
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import time

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from mp093.config import ROOT, load_params
from mp093.logging_config import get_logger

log = get_logger("mp093.api")

PARAMS = load_params()
FEATURES = PARAMS["features"]            # column order the model was trained with
MODEL_FILE = ROOT / "models" / "model.joblib"
MODEL_VERSION = "v2"                     # the @champion (registry version 2)

# Loaded once, when the app starts, not on every request.
model = joblib.load(MODEL_FILE)

app = FastAPI(title="MP093 grain_kg predictor")


class PredictRequest(BaseModel):
    """Exactly the 8 fields of PROBLEM.md section 5. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    # Client rule 2: fewer than 100 cards is always a typing mistake.
    cards_attached: int = Field(..., ge=100, description="ration cards attached to the shop")
    # Client rule 1: more than 15 working hours is impossible.
    working_hours: float = Field(..., le=15, description="hours the shop was open")
    biometric_failures: int = Field(..., description="failed biometric checks that day")
    temp_c: float = Field(..., description="temperature in degrees Celsius")
    stock_kg_morning: float = Field(..., description="grain stock in kg at opening")
    days_to_month_end: int = Field(..., description="days left in the month")
    shop_area: str = Field(..., description="urban, rural or tribal")
    block: str = Field(..., description="development block, e.g. Block North")


class PredictResponse(BaseModel):
    prediction: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.perf_counter()
    fields = req.model_dump()
    row = pd.DataFrame([fields], columns=FEATURES)
    prediction = float(model.predict(row)[0])
    latency_ms = (time.perf_counter() - start) * 1000
    # One JSON line per prediction. The fields describe a shop, not a person.
    log.info("prediction", extra={
        "event": "prediction",
        **fields,
        "prediction": prediction,
        "model_version": MODEL_VERSION,
        "latency_ms": round(latency_ms, 3),
    })
    return {"prediction": prediction, "model_version": MODEL_VERSION}
