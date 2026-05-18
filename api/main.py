"""FastAPI spam classifier application."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from api.model_loader import load_models
from api.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    yield


app = FastAPI(
    title="Spam Classifier API",
    description="Classifies email text as ham or spam using TF-IDF + ML models.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    pipeline = request.app.state.models[body.model]
    proba = pipeline.predict_proba([body.text])[0]
    predicted_class = int(proba.argmax())
    label = "spam" if predicted_class == 1 else "ham"
    confidence = round(float(proba[predicted_class]), 4)
    return PredictResponse(label=label, confidence=confidence, model=body.model)
