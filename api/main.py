"""FastAPI application for aerodynamic prediction and optimization."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AeroSurrogate")


class PredictionInput(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictionInput) -> dict:
    return {"prediction": None, "message": "Model not loaded"}


@app.post("/optimize")
def optimize(payload: dict) -> dict:
    return {"result": None, "message": "Optimizer not configured"}
